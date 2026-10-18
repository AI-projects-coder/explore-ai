"""Question -> retrieval -> Gemini -> validated answer with citations, or an honest "no answer"."""
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from hr_assistant.core.auth import User
from hr_assistant.generation.llm import LLM, LLMAnswer
from hr_assistant.generation.prompts import SYSTEM_PROMPT, build_prompt
from hr_assistant.retrieval.search import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

# Matches "[1]" and also "[1, 3]" (models don't always follow the one-number-per-bracket rule).
_CITATION_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(text: str) -> set[int]:
    return {int(n) for group in _CITATION_MARKER.findall(text) for n in group.split(",")}


def normalize_markers(text: str) -> str:
    """'[1, 3]' -> '[1][3]', so the UI only ever has to render one format."""
    return _CITATION_MARKER.sub(lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), text)


@dataclass(frozen=True)
class Citation:
    number: int  # the [n] used in the answer text
    document_id: uuid.UUID
    title: str
    heading_path: str
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class Answer:
    text: str
    answerable: bool
    citations: list[Citation] = field(default_factory=list)
    no_answer_reason: str | None = None  # for logs and evals; never shown to the user
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


def validate(parsed: LLMAnswer | None, n_chunks: int) -> str | None:
    """Return why the model's output must not be shown, or None if it is safe to use."""
    if parsed is None:
        return "unparseable_output"
    if parsed.status == "off_topic":
        return "off_topic"
    if parsed.status == "not_found":
        return "not_found"
    cited = set(parsed.citations) | cited_numbers(parsed.answer)
    if not cited:
        return "no_citations"
    if any(n < 1 or n > n_chunks for n in cited):
        return "invalid_citation"  # cites an excerpt we never sent: treat the whole answer as made up
    return None

@dataclass(frozen=True)
class NoAnswerMessages:
    not_found: str  # an HR question the user's policies don't cover
    off_topic: str  # not an HR question at all
    no_context: str  # retrieval found nothing, so we can't tell which of the two it is

class AnswerService:
    def __init__(self, *, retriever: Retriever, llm: LLM, messages: NoAnswerMessages) -> None:
        self._retriever = retriever
        self._llm = llm
        self._messages = messages
    async def answer(self, question: str, user: User) -> Answer:
        started = time.perf_counter()
        chunks = await self._retriever.search(question, user)
        if not chunks:
            return self._no_answer("no_context", started)  # layer 1: nothing to ground on, skip the LLM

        result = await self._llm.generate(SYSTEM_PROMPT, build_prompt(question, chunks))
        reason = validate(result.parsed, len(chunks))  # layers 2 and 3
        if reason:
            return self._no_answer(reason, started, result.input_tokens, result.output_tokens)

        answer = Answer(
            text=normalize_markers(result.parsed.answer.strip()),
            answerable=True,
            citations=self._citations(result.parsed, chunks),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=self._ms(started),
        )
        self._log(user, answer, len(chunks))
        return answer

    def _citations(self, parsed: LLMAnswer, chunks: list[RetrievedChunk]) -> list[Citation]:
        numbers = sorted(set(parsed.citations) | cited_numbers(parsed.answer))
        return [
            Citation(
                number=n,
                document_id=chunks[n - 1].document_id,
                title=chunks[n - 1].title,
                heading_path=chunks[n - 1].heading_path,
                page_start=chunks[n - 1].page_start,
                page_end=chunks[n - 1].page_end,
            )
            for n in numbers
        ]

    def _no_answer(self, reason: str, started: float, input_tokens: int = 0, output_tokens: int = 0) -> Answer:
        if reason == "off_topic":
            text = self._messages.off_topic
        elif reason == "no_context":
            text = self._messages.no_context
        else:  # not_found, and every "the model's output can't be trusted" reason
            text = self._messages.not_found
        answer = Answer(
            text=text,
            answerable=False,
            no_answer_reason=reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=self._ms(started),
        )
        logger.info(
            "answer no_answer reason=%s tokens_in=%d tokens_out=%d ms=%d",
            reason, input_tokens, output_tokens, answer.latency_ms,
        )
        return answer

    @staticmethod
    def _log(user: User, answer: Answer, n_chunks: int) -> None:
        logger.info(
            "answer user=%s chunks=%d cited=%s tokens_in=%d tokens_out=%d ms=%d",
            user.id, n_chunks, [c.number for c in answer.citations],
            answer.input_tokens, answer.output_tokens, answer.latency_ms,
        )

    @staticmethod
    def _ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)
