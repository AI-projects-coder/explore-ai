"""Gemini wrapper: structured output, the repair loop, and mapping every failure to a clear error.

A fake client stands in for Gemini, so these tests are free, fast and offline.
"""

import pytest
from google.genai import errors, types
from pydantic import BaseModel, field_validator

from research_analyst.core.config import Settings
from research_analyst.core.llm import (
    Llm,
    LlmBadOutput,
    LlmBlocked,
    LlmError,
    LlmTruncated,
    LlmUnavailable,
)

SETTINGS = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io")


class Pick(BaseModel):
    number: int

    @field_validator("number")
    @classmethod
    def even(cls, v: int) -> int:
        if v % 2:
            raise ValueError("number must be even")
        return v


def response(text: str | None = '{"number": 2}', *, finish=types.FinishReason.STOP, block_reason=None,
             prompt_tokens=10, output_tokens=5, thinking_tokens=3) -> types.GenerateContentResponse:
    parts = [types.Part(text=text)] if text is not None else []
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)],
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block_reason) if block_reason else None,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=output_tokens,
            thoughts_token_count=thinking_tokens,
        ),
    )


class FakeGemini:
    """Mimics genai.Client: client.aio.models.generate_content(...). Plays a script of responses/exceptions."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests: list[dict] = []
        self.aio = self
        self.models = self

    async def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def make(*script) -> tuple[Llm, FakeGemini]:
    fake = FakeGemini(*script)
    return Llm(SETTINGS, client=fake), fake


async def ask(llm: Llm, **kw):
    return await llm.structured(model="gemini-test", system="sys", prompt="Pick 2.", schema=Pick, **kw)


# --- happy path --------------------------------------------------------------------------

async def test_returns_a_validated_instance():
    llm, fake = make(response('{"number": 2}'))
    assert await ask(llm) == Pick(number=2)
    assert len(fake.requests) == 1


async def test_sends_model_system_and_json_schema():
    llm, fake = make(response())
    await ask(llm)
    req = fake.requests[0]
    assert req["model"] == "gemini-test"
    assert req["config"].system_instruction == "sys"
    assert req["config"].response_mime_type == "application/json"
    assert req["config"].response_json_schema == Pick.model_json_schema()  # the shape Gemini must follow
    assert req["config"].max_output_tokens == SETTINGS.llm_max_output_tokens


async def test_counts_tokens_per_model_including_thinking():
    llm, _ = make(response(prompt_tokens=100, output_tokens=20, thinking_tokens=30), response(prompt_tokens=1))
    await ask(llm)
    await llm.structured(model="other-model", system="s", prompt="p", schema=Pick)
    assert llm.usage.calls == 2
    assert llm.usage.input_tokens == {"gemini-test": 100, "other-model": 1}
    assert llm.usage.output_tokens["gemini-test"] == 50  # thinking is billed as output


# --- repair loop -----------------------------------------------------------------------------

async def test_invalid_answer_is_repaired_once():
    llm, fake = make(response('{"number": 7}'), response('{"number": 8}'))
    assert await ask(llm) == Pick(number=8)

    # the second request shows the model its own bad answer and exactly what was wrong
    second = fake.requests[1]["contents"]
    assert [c.role for c in second] == ["user", "model", "user"]
    assert second[1].parts[0].text == '{"number": 7}'
    assert "number must be even" in second[2].parts[0].text


async def test_non_json_answer_is_repaired_too():
    llm, _ = make(response("Sure! The number is 2."), response('{"number": 2}'))
    assert await ask(llm) == Pick(number=2)


async def test_gives_up_after_the_repair_attempt():
    llm, fake = make(response('{"number": 7}'), response('{"number": 9}'))
    with pytest.raises(LlmBadOutput, match="number must be even"):
        await ask(llm)
    assert len(fake.requests) == 2


async def test_repair_can_be_disabled():
    llm, fake = make(response('{"number": 7}'))
    with pytest.raises(LlmBadOutput):
        await ask(llm, repair=0)
    assert len(fake.requests) == 1


# --- failures that must NOT be repaired --------------------------------------------------------

async def test_truncated_answer_raises_without_repair():
    llm, fake = make(response('{"numb', finish=types.FinishReason.MAX_TOKENS))
    with pytest.raises(LlmTruncated, match="max_output_tokens"):
        await ask(llm)
    assert len(fake.requests) == 1  # asking again would be cut off the same way


@pytest.mark.parametrize("finish", [types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT])
async def test_safety_stop_is_blocked(finish):
    llm, _ = make(response(None, finish=finish))
    with pytest.raises(LlmBlocked):
        await ask(llm)


async def test_blocked_prompt_is_blocked():
    llm, _ = make(response(None, block_reason=types.BlockedReason.SAFETY))
    with pytest.raises(LlmBlocked, match="prompt"):
        await ask(llm)


async def test_empty_answer_is_bad_output():
    llm, _ = make(response(None))
    with pytest.raises(LlmBadOutput, match="empty"):
        await ask(llm)


# --- API errors --------------------------------------------------------------------------------

def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "boom", "status": "X"}})


@pytest.mark.parametrize("code", [429, 503])
async def test_rate_limit_or_outage_is_unavailable(code):
    # the SDK already retried (HttpRetryOptions); reaching us means it kept failing
    llm, _ = make(api_error(code))
    with pytest.raises(LlmUnavailable, match=str(code)):
        await ask(llm)


async def test_timeout_is_unavailable():
    llm, _ = make(TimeoutError())
    with pytest.raises(LlmUnavailable, match="timed out"):
        await ask(llm)


@pytest.mark.parametrize("code", [400, 403, 404])
async def test_our_own_mistakes_are_not_hidden(code):
    # wrong model name / region / permission: a bug to fix, not a temporary problem to work around
    llm, _ = make(api_error(code))
    with pytest.raises(errors.APIError):
        await ask(llm)


def test_all_errors_share_one_base():
    for cls in (LlmUnavailable, LlmBlocked, LlmTruncated, LlmBadOutput):
        assert issubclass(cls, LlmError)


# --- client configuration -----------------------------------------------------------------------

def test_real_client_gets_timeout_and_retries(monkeypatch):
    captured = {}

    class SpyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("research_analyst.core.llm.genai.Client", SpyClient)
    Llm(SETTINGS)
    opts = captured["http_options"]
    assert captured["vertexai"] is True
    assert (captured["project"], captured["location"]) == ("p", "global")
    assert opts.timeout == SETTINGS.llm_timeout_seconds * 1000  # milliseconds
    assert 429 in opts.retry_options.http_status_codes
    assert 400 not in opts.retry_options.http_status_codes


# --- per-call options ----------------------------------------------------------------------------

class Bounded(BaseModel):
    number: int

    @field_validator("number")
    @classmethod
    def at_most(cls, v: int, info) -> int:
        limit = (info.context or {}).get("max")
        if limit is not None and v > limit:
            raise ValueError(f"number must be at most {limit}")
        return v


async def test_validation_context_reaches_the_validators():
    # e.g. the writer passes the real metric ids; the planner passes nothing
    llm, fake = make(response('{"number": 9}'), response('{"number": 3}'))
    r = await llm.structured(model="m", system="s", prompt="p", schema=Bounded, context={"max": 5})
    assert r.number == 3
    assert "at most 5" in fake.requests[1]["contents"][2].parts[0].text


async def test_max_output_tokens_can_be_raised_per_call():
    llm, fake = make(response(), response('{"x', finish=types.FinishReason.MAX_TOKENS))
    await llm.structured(model="m", system="s", prompt="p", schema=Pick, max_output_tokens=32768)
    assert fake.requests[0]["config"].max_output_tokens == 32768
    with pytest.raises(LlmTruncated, match="8192"):  # the default, named in the error
        await llm.structured(model="m", system="s", prompt="p", schema=Pick)
