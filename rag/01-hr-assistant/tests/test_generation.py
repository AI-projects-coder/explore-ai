"""Generation: prompt building, citation parsing, and every "no answer" path. Fake retriever + fake LLM: no network."""
import uuid

import pytest

from hr_assistant.core.auth import User
from hr_assistant.generation.answer import (
    AnswerService,
    NoAnswerMessages,
    cited_numbers,
    normalize_markers,
    validate,
)
from hr_assistant.generation.llm import LLMAnswer, LLMResult
from hr_assistant.generation.prompts import SYSTEM_PROMPT, build_prompt
from hr_assistant.retrieval.search import RetrievedChunk

MESSAGES = NoAnswerMessages(not_found="NOT FOUND. Contact HR.", off_topic="HR TOPICS ONLY.", no_context="NOTHING FOUND.")
INTERN = User("intern_ravi", frozenset({"employee"}))


def chunk(n: int, page_start: int | None = 1, page_end: int | None = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        title=f"Doc {n}",
        heading_path=f"Doc {n} > Section {n}",
        content=f"content {n}",
        page_start=page_start,
        page_end=page_end,
        score=1.0 - n / 10,
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    async def search(self, question: str, user: User) -> list[RetrievedChunk]:
        return self.chunks


class FakeLLM:
    """Returns whatever the test tells it to, and records what it was asked."""

    model = "fake-llm"

    def __init__(self, parsed: LLMAnswer | None) -> None:
        self.result = LLMResult(parsed=parsed, input_tokens=100, output_tokens=20)
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system: str, prompt: str) -> LLMResult:
        self.calls.append((system, prompt))
        return self.result


def service(chunks: list[RetrievedChunk], parsed: LLMAnswer | None) -> tuple[AnswerService, FakeLLM]:
    llm = FakeLLM(parsed)
    return AnswerService(retriever=FakeRetriever(chunks), llm=llm, messages=MESSAGES), llm


# ---------------------------------------------------------------- the happy path


async def test_valid_answer_keeps_text_and_maps_citations_to_chunks():
    chunks = [chunk(1), chunk(2), chunk(3)]
    svc, _ = service(chunks, LLMAnswer(status="answered", answer="Up to 4 days [1, 3].", citations=[1, 3]))

    answer = await svc.answer("casual leave?", INTERN)

    assert answer.answerable is True
    assert answer.text == "Up to 4 days [1][3]."  # markers normalized
    assert [(c.number, c.title) for c in answer.citations] == [(1, "Doc 1"), (3, "Doc 3")]
    assert answer.citations[1].document_id == chunks[2].document_id  # [3] -> chunks[2]
    assert (answer.input_tokens, answer.output_tokens) == (100, 20)


async def test_citation_only_in_the_list_still_counts():
    # Seen live: the model filled "citations" but put no [n] marker in the text.
    svc, _ = service([chunk(1)], LLMAnswer(status="answered", answer="Up to 4 days.", citations=[1]))

    answer = await svc.answer("casual leave?", INTERN)

    assert answer.answerable is True
    assert [c.number for c in answer.citations] == [1]


async def test_marker_only_in_the_text_still_counts():
    svc, _ = service([chunk(1), chunk(2)], LLMAnswer(status="answered", answer="Ten days [2].", citations=[]))

    answer = await svc.answer("sick leave?", INTERN)

    assert [c.number for c in answer.citations] == [2]


# ---------------------------------------------------------------- the three "no answer" layers


async def test_layer_1_no_chunks_means_no_llm_call():
    svc, llm = service([], LLMAnswer(status="answered", answer="should never be used [1]", citations=[1]))

    answer = await svc.answer("L4 salary?", INTERN)

    assert (answer.answerable, answer.text, answer.no_answer_reason) == (False, MESSAGES.no_context, "no_context")
    assert llm.calls == []  # no tokens spent
    assert answer.input_tokens == 0


@pytest.mark.parametrize(
    ("parsed", "reason"),
    [
        (None, "unparseable_output"),
        (LLMAnswer(status="answered", answer="   ", citations=[1]), "empty_answer"),
        (LLMAnswer(status="answered", answer="Up to 4 days.", citations=[]), "no_citations"),
        (LLMAnswer(status="answered", answer="Up to 4 days [1].", citations=[9]), "invalid_citation"),
        (LLMAnswer(status="answered", answer="Up to 4 days [1, 9].", citations=[1]), "invalid_citation"),
        (LLMAnswer(status="answered", answer="Up to 4 days [0].", citations=[]), "invalid_citation"),
    ],
    ids=["unparseable", "empty", "no-citations", "cites-9-in-list", "cites-9-in-text", "cites-0"],
)
async def test_layer_3_untrustworthy_output_becomes_not_found(parsed, reason):
    svc, llm = service([chunk(1), chunk(2), chunk(3)], parsed)

    answer = await svc.answer("question", INTERN)

    assert answer.answerable is False
    assert answer.text == MESSAGES.not_found  # the user never sees the unverified text
    assert answer.no_answer_reason == reason
    assert answer.citations == []
    assert len(llm.calls) == 1
    assert (answer.input_tokens, answer.output_tokens) == (100, 20)  # cost is still recorded


@pytest.mark.parametrize(
    ("status", "message"),
    [("not_found", MESSAGES.not_found), ("off_topic", MESSAGES.off_topic)],
)
async def test_layer_2_model_status_picks_the_matching_message(status, message):
    svc, _ = service([chunk(1)], LLMAnswer(status=status, answer="", citations=[]))

    answer = await svc.answer("question", INTERN)

    assert (answer.answerable, answer.text, answer.no_answer_reason) == (False, message, status)


async def test_off_topic_status_wins_even_if_the_model_also_wrote_an_answer():
    # e.g. the model says off_topic but still answers "New Delhi [1]": never show it.
    svc, _ = service([chunk(1)], LLMAnswer(status="off_topic", answer="New Delhi [1].", citations=[1]))

    answer = await svc.answer("capital of India?", INTERN)

    assert (answer.answerable, answer.text) == (False, MESSAGES.off_topic)


def test_no_answer_messages_never_name_a_document():
    # They are shown to everyone, so they must not reveal restricted document titles.
    from hr_assistant.core.config import get_settings

    s = get_settings()
    for text in (s.not_found_message, s.off_topic_message, s.no_context_message):
        assert "salary" not in text.lower() and "performance" not in text.lower()


def test_validate_accepts_a_correct_answer():
    assert validate(LLMAnswer(status="answered", answer="Yes [2].", citations=[2]), n_chunks=3) is None


# ---------------------------------------------------------------- citation helpers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no markers", set()),
        ("a [1] b [3]", {1, 3}),
        ("a [1, 3] b [2]", {1, 2, 3}),
        ("a [1,3]", {1, 3}),
        ("year [2026] counts too", {2026}),  # a bracketed number looks like a citation -> invalid -> rejected
        ("[a] [] [1.5]", set()),
    ],
)
def test_cited_numbers(text, expected):
    assert cited_numbers(text) == expected


def test_normalize_markers():
    assert normalize_markers("a [1, 3] b [2] c [4,5]") == "a [1][3] b [2] c [4][5]"


# ---------------------------------------------------------------- the prompt


def test_prompt_numbers_excerpts_in_retrieval_order_with_source_and_pages():
    prompt = build_prompt("  How many days?  ", [chunk(1), chunk(2, page_start=1, page_end=2), chunk(3, None, None)])

    assert prompt.index('<excerpt id="1" source="Doc 1 > Section 1" pages="p. 1">') < prompt.index(
        '<excerpt id="2" source="Doc 2 > Section 2" pages="pp. 1-2">'
    )
    assert '<excerpt id="3" source="Doc 3 > Section 3" pages="">' in prompt
    assert prompt.endswith("<question>\nHow many days?\n</question>")  # question stripped, after the context


async def test_llm_receives_system_rules_separately_from_the_data():
    svc, llm = service([chunk(1)], LLMAnswer(status="answered", answer="x [1]", citations=[1]))

    await svc.answer("question", INTERN)

    system, prompt = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "content 1" in prompt and "content 1" not in system
