"""Gemini on Vertex AI, returning a validated structured answer."""
from dataclasses import dataclass
from typing import Literal, Protocol

from google import genai
from google.genai import types
from pydantic import BaseModel

from hr_assistant.core.config import Settings
from hr_assistant.core.retry import google_api_retry


class LLMAnswer(BaseModel):
    """The JSON shape the model must return (enforced by response_schema)."""

    status: Literal["answered", "not_found", "off_topic"]
    answer: str
    citations: list[int]



@dataclass(frozen=True)
class LLMResult:
    parsed: LLMAnswer | None  # None if the model returned something that doesn't fit the schema
    input_tokens: int
    output_tokens: int  # includes "thinking" tokens: they are billed as output


class LLM(Protocol):
    model: str

    async def generate(self, system: str, prompt: str) -> LLMResult: ...


class GeminiLLM:
    def __init__(
        self,
        *,
        project: str,
        location: str,
        model: str,
        temperature: float = 0.0,
        thinking_level: Literal["low", "high"] | None = "low",
        max_output_tokens: int = 1024,
        timeout_s: int = 60,
    ) -> None:
        self.model = model
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(timeout=timeout_s * 1000),
        )
        self._temperature = temperature
        self._thinking = types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None
        self._max_output_tokens = max_output_tokens

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiLLM":
        return cls(
            project=settings.gcp_project,
            location=settings.gcp_location,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            thinking_level=settings.llm_thinking_level,
            max_output_tokens=settings.llm_max_output_tokens,
        )

    @google_api_retry
    async def generate(self, system: str, prompt: str) -> LLMResult:
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                thinking_config=self._thinking,
                response_mime_type="application/json",
                response_schema=LLMAnswer,
            ),
        )
        usage = response.usage_metadata
        parsed = response.parsed if isinstance(response.parsed, LLMAnswer) else None
        return LLMResult(
            parsed=parsed,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
        )
