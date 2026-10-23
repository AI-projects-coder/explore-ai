"""Thin wrapper around google-genai: one place for clients, models and structured output.

Every LLM call in the project goes through Llm.structured(). It:
  - asks Gemini for JSON in the exact shape of a Pydantic model,
  - validates it with that model (including our own validators, e.g. known metric names),
  - if invalid: shows the model its own answer + the validation error, once, and asks for a fix,
  - retries temporary API failures (429/5xx) with backoff,
  - turns every other failure into one clear LlmError subclass,
  - counts tokens and enforces the run's budget BEFORE every call (BudgetExceeded instead of spending).
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, ValidationError

from research_analyst.core.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Temporary failures worth retrying. Anything else (e.g. 400 = bad request) is our bug: fail fast.
_RETRYABLE_STATUS = [429, 500, 502, 503, 504]
_BLOCKED = {
    types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT, types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII, types.FinishReason.RECITATION,
}


class LlmError(Exception):
    """Base class for every LLM problem; nodes catch this one."""


class LlmUnavailable(LlmError):
    """The API kept failing (rate limits, outages, timeouts) after the SDK's retries."""


class LlmBlocked(LlmError):
    """Gemini refused to answer (safety filters)."""


class LlmTruncated(LlmError):
    """The answer hit max_output_tokens and was cut off (thinking tokens count toward the limit)."""


class LlmBadOutput(LlmError):
    """The answer still didn't match the schema after the repair attempt."""


class BudgetExceeded(LlmError):
    """The run's budget (tokens, calls, USD or time) is used up. Raised before the call, so nothing is spent."""


class Spend(BaseModel):
    """What a run has spent so far. Stored in the graph state, so it survives crashes and resumes."""

    model_config = ConfigDict(frozen=True)

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0  # estimate from Settings.model_prices; 0 when prices aren't configured

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def max_spend(current: Spend | None, new: Spend | None) -> Spend:
    """Reducer for state["spend"]: every node reports the run's running total, so keep the largest.

    Parallel researchers each report "total so far"; the max is the latest. Plain addition would
    double-count, because each report already includes everything before it.
    """
    a, b = current or Spend(), new or Spend()
    return Spend(
        llm_calls=max(a.llm_calls, b.llm_calls),
        input_tokens=max(a.input_tokens, b.input_tokens),
        output_tokens=max(a.output_tokens, b.output_tokens),
        usd=max(a.usd, b.usd),
    )


@dataclass
class Usage:
    """Tokens used, per model. Output includes thinking tokens: both are billed as output."""

    calls: int = 0
    input_tokens: dict[str, int] = field(default_factory=dict)
    output_tokens: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, meta: types.GenerateContentResponseUsageMetadata | None) -> None:
        self.calls += 1
        if meta is None:
            return
        out = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
        self.input_tokens[model] = self.input_tokens.get(model, 0) + (meta.prompt_token_count or 0)
        self.output_tokens[model] = self.output_tokens.get(model, 0) + out


class Llm:
    """One per working session of a run. Holds the Gemini client, the usage, and the budget."""

    def __init__(self, settings: Settings, *, client: genai.Client | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._max_output_tokens = settings.llm_max_output_tokens
        self._settings = settings
        self._clock = clock
        self._deadline = clock() + settings.budget_max_seconds
        self._baseline = Spend()  # what earlier sessions of this run spent (loaded from the checkpoint)
        self._client = client or genai.Client(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.gcp_location,
            http_options=types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000),  # milliseconds
                retry_options=types.HttpRetryOptions(
                    attempts=4, initial_delay=1.0, max_delay=30.0, http_status_codes=_RETRYABLE_STATUS,
                ),
            ),
        )
        self.usage = Usage()

    def set_baseline(self, spent: Spend | None) -> None:
        """Continue a run: count what it already spent before this session (a crash or a review pause)."""
        self._baseline = spent or Spend()

    def spend(self) -> Spend:
        """The run's total so far: earlier sessions + this one."""
        prices = self._settings.model_prices
        usd = sum(
            self.usage.input_tokens.get(m, 0) / 1e6 * price_in + self.usage.output_tokens.get(m, 0) / 1e6 * price_out
            for m, (price_in, price_out) in prices.items()
        )
        b = self._baseline
        return Spend(
            llm_calls=b.llm_calls + self.usage.calls,
            input_tokens=b.input_tokens + sum(self.usage.input_tokens.values()),
            output_tokens=b.output_tokens + sum(self.usage.output_tokens.values()),
            usd=round(b.usd + usd, 6),
        )

    def _check_budget(self) -> None:
        s, cfg = self.spend(), self._settings
        if s.llm_calls >= cfg.budget_max_llm_calls:
            raise BudgetExceeded(f"LLM call limit reached ({s.llm_calls} of {cfg.budget_max_llm_calls})")
        if s.tokens >= cfg.budget_max_tokens:
            raise BudgetExceeded(f"token limit reached ({s.tokens:,} of {cfg.budget_max_tokens:,})")
        if cfg.budget_max_usd is not None and s.usd >= cfg.budget_max_usd:
            raise BudgetExceeded(f"cost limit reached (about ${s.usd:.2f} of ${cfg.budget_max_usd:.2f})")
        if self._clock() > self._deadline:
            raise BudgetExceeded(f"time limit reached ({cfg.budget_max_seconds}s for this session)")

    async def structured(self, *, model: str, system: str, prompt: str, schema: type[T], repair: int = 1,
                         context: dict[str, Any] | None = None, max_output_tokens: int | None = None) -> T:
        """Ask `model` for an answer shaped like `schema`. Returns a validated instance, or raises LlmError.

        `context` is passed to the schema's validators (pydantic validation context), so they can check
        the answer against facts the schema alone doesn't know, e.g. which metric ids really exist.
        """
        contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        limit = max_output_tokens or self._max_output_tokens
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=limit,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
        )

        for attempt in range(repair + 1):
            text = await self._generate(model, contents, config, limit)
            try:
                return schema.model_validate_json(text, context=context)
            except ValidationError as e:
                problem = _short_errors(e)
                logger.warning("llm_bad_output", extra={"event": "llm_bad_output", "model": model,
                                                        "attempt": attempt, "errors": problem})
                if attempt == repair:
                    raise LlmBadOutput(f"{model} returned invalid {schema.__name__}: {problem}") from None
                # Show the model its own answer and exactly what was wrong with it.
                contents += [
                    types.Content(role="model", parts=[types.Part(text=text)]),
                    types.Content(role="user", parts=[types.Part(text=(
                        f"Your answer did not pass validation:\n{problem}\n"
                        "Return the corrected JSON only, in the same format."
                    ))]),
                ]
        raise AssertionError("unreachable")

    async def _generate(self, model: str, contents: list[types.Content], config: types.GenerateContentConfig,
                        limit: int) -> str:
        self._check_budget()  # before every call, repairs included
        try:
            response = await self._client.aio.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            if e.code in _RETRYABLE_STATUS:
                raise LlmUnavailable(f"{model} unavailable (HTTP {e.code}) after retries") from e
            raise  # 400/403/404: a bug or misconfiguration (wrong model name, region, permissions)
        except TimeoutError as e:
            raise LlmUnavailable(f"{model} timed out") from e

        self.usage.add(model, response.usage_metadata)

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise LlmBlocked(f"{model} blocked the prompt ({response.prompt_feedback.block_reason})")
        candidate = response.candidates[0] if response.candidates else None
        reason = candidate.finish_reason if candidate else None
        if reason in _BLOCKED:
            raise LlmBlocked(f"{model} stopped for safety reasons ({reason})")
        if reason == types.FinishReason.MAX_TOKENS:
            raise LlmTruncated(f"{model} hit max_output_tokens={limit}; the answer was cut off")
        if not response.text:
            raise LlmBadOutput(f"{model} returned an empty answer (finish_reason={reason})")
        return response.text


def _short_errors(e: ValidationError, limit: int = 5) -> str:
    """Validation errors as short lines the model can act on, e.g. 'tasks.0.kind: Input should be ...'."""
    lines = [f"- {'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}" for err in e.errors()[:limit]]
    return "\n".join(lines)
