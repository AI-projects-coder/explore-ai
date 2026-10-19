"""Structured logs: JSON format, request/user ids on every line, event names, and no personal data."""
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hr_assistant.api.deps import get_answer_service, get_session_factory
from hr_assistant.api.main import create_app
from hr_assistant.core.auth import DEV_USERS, User, issue_token
from hr_assistant.core.config import get_settings
from hr_assistant.core.logs import ContextFilter, JsonFormatter
from hr_assistant.generation.answer import AnswerService, NoAnswerMessages
from hr_assistant.generation.llm import GeminiLLM, LLMAnswer, LLMResult
from hr_assistant.retrieval.search import Retriever
from tests.support import FakeEmbedder

SECRET_QUESTION = "How do I report harassment by my manager Mr Sharma-7731?"  # must never reach the logs


class Captured(logging.Handler):
    """Collects records the way production does: through the ContextFilter, formatted as JSON."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.addFilter(ContextFilter())
        self.setFormatter(JsonFormatter())
        self.lines: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(self.format(record)))


@pytest.fixture
def captured():
    handler = Captured()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield handler
    root.removeHandler(handler)
    root.setLevel(old_level)


class FakeLLM:
    model = "fake"

    async def generate(self, system: str, prompt: str) -> LLMResult:
        return LLMResult(LLMAnswer(status="answered", answer="Up to 4 days [1].", citations=[1]), 100, 20)


@pytest.fixture
async def client(database):
    """The REAL answer service (real retrieval SQL, real logging), with a fake embedder and LLM."""
    engine = create_async_engine(database)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    service = AnswerService(
        retriever=Retriever(embedder=FakeEmbedder(), session_factory=factory),
        llm=FakeLLM(),
        messages=NoAnswerMessages("not found", "off topic", "no context"),
    )
    app = create_app()
    app.dependency_overrides[get_answer_service] = lambda: service
    app.dependency_overrides[get_session_factory] = lambda: factory
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await engine.dispose()


def auth(user_id: str = "intern_ravi", **headers: str) -> dict[str, str]:
    token = issue_token(User(user_id, DEV_USERS[user_id]), get_settings())
    return {"Authorization": f"Bearer {token}", **headers}


# ---------------------------------------------------------------- the JSON format


def test_json_line_has_severity_message_context_and_extras():
    record = logging.LogRecord("hr_assistant.x", logging.WARNING, __file__, 1, "budget %d%% used", (80,), None)
    record.event, record.tokens_today = "token_budget_alert", 4_000_000
    record.request_id, record.user_id = "abc12345", "intern_ravi"

    line = json.loads(JsonFormatter().format(record))

    assert line["severity"] == "WARNING"  # Cloud Logging's level field
    assert line["message"] == "budget 80% used"
    assert (line["request_id"], line["user_id"]) == ("abc12345", "intern_ravi")
    assert (line["event"], line["tokens_today"]) == ("token_budget_alert", 4_000_000)  # searchable fields


def test_exceptions_are_kept_inside_the_same_json_line():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", (), __import__("sys").exc_info())
    record.request_id = record.user_id = None

    line = json.loads(JsonFormatter().format(record))  # one line, still valid JSON

    assert "ValueError: boom" in line["exception"]


@pytest.mark.parametrize(
    ("environment", "log_format", "expected"),
    [("local", None, False), ("prod", None, True), ("local", "json", True), ("prod", "text", False)],
)
def test_json_logs_everywhere_except_a_laptop_unless_overridden(environment, log_format, expected):
    settings = get_settings().model_copy(update={"environment": environment, "log_format": log_format})

    assert settings.json_logs is expected


# ---------------------------------------------------------------- request ids


async def test_every_response_gets_a_request_id(client):
    response = await client.get("/healthz")

    assert len(response.headers["x-request-id"]) == 32


async def test_a_valid_incoming_request_id_is_reused(client):
    response = await client.get("/healthz", headers={"X-Request-ID": "bff-0123456789abcdef"})

    assert response.headers["x-request-id"] == "bff-0123456789abcdef"  # one id from browser to logs


@pytest.mark.parametrize(
    "bad", ["short", "x" * 65, 'abc12345"}, {"severity": "CRITICAL', "abc12345\nFAKE LOG LINE", "abc 12345678"]
)
async def test_suspicious_incoming_request_ids_are_replaced(client, bad):
    response = await client.get("/healthz", headers={"X-Request-ID": bad})

    assert response.headers["x-request-id"] != bad  # log injection is not possible through the header


async def test_all_lines_of_a_request_share_its_request_id_and_user(client, captured):
    response = await client.post("/ask", json={"question": "casual leave?"}, headers=auth())
    rid = response.headers["x-request-id"]

    lines = [line for line in captured.lines if line["request_id"] == rid]
    events = [line.get("event") for line in lines]

    assert {"retrieval", "answer", "http_request"} <= set(events)
    assert all(line["user_id"] == "intern_ravi" for line in lines)


async def test_access_log_line_has_status_path_and_duration(client, captured):
    await client.post("/ask", json={"question": "casual leave?"}, headers=auth())

    access = next(line for line in captured.lines if line.get("event") == "http_request")

    assert (access["method"], access["path"], access["status"]) == ("POST", "/ask", 200)
    assert isinstance(access["duration_ms"], int)


async def test_concurrent_requests_do_not_mix_up_their_ids(client, captured):
    import asyncio

    responses = await asyncio.gather(
        *(client.post("/ask", json={"question": "q"}, headers=auth(u)) for u in ["intern_ravi", "manager_priya"] * 5)
    )

    for response in responses:
        rid = response.headers["x-request-id"]
        users = {line["user_id"] for line in captured.lines if line["request_id"] == rid}
        assert len(users) == 1  # each request's lines carry only its own user


# ---------------------------------------------------------------- privacy


async def test_question_and_answer_text_never_appear_in_logs(client, captured):
    await client.post("/ask", json={"question": SECRET_QUESTION}, headers=auth())

    everything = json.dumps(captured.lines)
    assert "Sharma-7731" not in everything  # questions to HR are personal data
    assert "harassment" not in everything
    assert "Up to 4 days" not in everything  # answers can quote personal situations too


# ---------------------------------------------------------------- events worth alerting on


async def test_truncated_llm_output_logs_an_llm_truncated_warning(captured):
    response = SimpleNamespace(
        parsed=None,
        usage_metadata=SimpleNamespace(prompt_token_count=1000, candidates_token_count=24, thoughts_token_count=1000),
        candidates=[SimpleNamespace(finish_reason=types.FinishReason.MAX_TOKENS)],
    )
    with patch("hr_assistant.generation.llm.genai.Client") as client_cls:
        client_cls.return_value.aio.models.generate_content = AsyncMock(return_value=response)
        llm = GeminiLLM(project="p", location="l", model="m", max_output_tokens=1024)
        result = await llm.generate("system", "prompt")

    assert result.parsed is None  # becomes a safe no-answer...
    warning = next(line for line in captured.lines if line.get("event") == "llm_truncated")
    assert (warning["severity"], warning["max_output_tokens"]) == ("WARNING", 1024)  # ...but never silently


async def test_unknown_user_ids_in_logs_are_not_a_thing_for_unauthenticated_requests(client, captured):
    await client.post("/ask", json={"question": "q"})  # no token -> 401

    access = next(line for line in captured.lines if line.get("event") == "http_request")
    assert (access["status"], access["user_id"]) == (401, None)

