"""The HTTP API: authentication, input validation, what is (and isn't) returned, error handling.

The answer service is replaced by a fake, so these tests check the API layer only.
"""
from datetime import UTC, datetime, timedelta

import httpx
import jwt
import pytest
from google.genai import errors
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr, ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hr_assistant.api.deps import get_answer_service, get_session_factory
from hr_assistant.api.main import check_embedding_model, create_app
from hr_assistant.core.auth import DEV_USERS, InvalidTokenError, User, issue_token, verify_token
from hr_assistant.core.config import Settings, get_settings
from hr_assistant.generation.answer import Answer, Citation


class FakeAnswerService:
    def __init__(self, result: Answer | Exception | None = None) -> None:
        self.result = result or Answer(text="Up to 4 days [1].", answerable=True)
        self.calls: list[tuple[str, User]] = []

    async def answer(self, question: str, user: User) -> Answer:
        self.calls.append((question, user))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def fake_service() -> FakeAnswerService:
    return FakeAnswerService()


@pytest.fixture
async def client(fake_service, database):
    app = create_app()
    engine = create_async_engine(database)
    app.dependency_overrides[get_answer_service] = lambda: fake_service
    app.dependency_overrides[get_session_factory] = lambda: async_sessionmaker(engine, expire_on_commit=False)
    # ASGITransport calls the app in-process; the lifespan (Vertex, DB checks) is NOT run.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await engine.dispose()


def bearer(user_id: str, settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(User(user_id, DEV_USERS[user_id]), settings)}"}


def forged(settings: Settings, **overrides) -> str:
    """A token we build by hand to simulate attacks and broken tokens."""
    now = datetime.now(UTC)
    claims = {"sub": "attacker", "groups": ["hr"], "iss": settings.jwt_issuer, "exp": now + timedelta(minutes=5)}
    secret = overrides.pop("secret", settings.jwt_secret.get_secret_value())
    algorithm = overrides.pop("algorithm", "HS256")
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}  # None = leave the claim out
    return jwt.encode(claims, secret, algorithm=algorithm)


# ---------------------------------------------------------------- health


async def test_healthz_needs_nothing(client):
    response = await client.get("/healthz")

    assert (response.status_code, response.json()) == (200, {"status": "ok"})


# ---------------------------------------------------------------- authentication


async def test_ask_without_token_is_401(client, fake_service):
    response = await client.post("/ask", json={"question": "casual leave?"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert fake_service.calls == []  # never reached the AI


@pytest.mark.parametrize(
    "make_token",
    [
        lambda s: "not-a-jwt",
        lambda s: forged(s, secret="some-other-secret-that-is-at-least-32-chars"),
        lambda s: forged(s, exp=datetime.now(UTC) - timedelta(seconds=1)),
        lambda s: forged(s, iss="someone-else"),
        lambda s: forged(s, groups=None),
        lambda s: forged(s, groups="hr"),
        lambda s: forged(s, exp=None),
        lambda s: jwt.encode({"sub": "attacker", "groups": ["hr"], "iss": s.jwt_issuer}, None, algorithm="none"),
    ],
    ids=["garbage", "wrong-secret", "expired", "wrong-issuer", "no-groups", "groups-not-a-list", "no-expiry",
         "alg-none"],
)
async def test_bad_tokens_are_all_rejected_with_the_same_401(client, fake_service, settings, make_token):
    response = await client.post(
        "/ask", json={"question": "salary?"}, headers={"Authorization": f"Bearer {make_token(settings)}"}
    )

    assert (response.status_code, response.json()) == (401, {"detail": "Not authenticated"})
    assert fake_service.calls == []


async def test_user_and_groups_come_from_the_token(client, fake_service, settings):
    await client.post("/ask", json={"question": "calibration?"}, headers=bearer("manager_priya", settings))

    question, user = fake_service.calls[0]
    assert user == User("manager_priya", frozenset({"employee", "manager"}))


async def test_client_cannot_send_its_own_groups(client, fake_service, settings):
    response = await client.post(
        "/ask", json={"question": "L4 salary?", "groups": ["hr"]}, headers=bearer("intern_ravi", settings)
    )

    assert response.status_code == 422  # rejected outright, not silently ignored
    assert fake_service.calls == []


def test_token_round_trip(settings):
    user = User("hr_anita", frozenset({"employee", "hr"}))

    assert verify_token(issue_token(user, settings), settings) == user


def test_token_signed_with_another_secret_is_invalid(settings):
    other = settings.model_copy(update={"jwt_secret": SecretStr("x" * 40)})  # model_copy skips validation
    token = issue_token(User("intern_ravi", frozenset({"employee"})), other)

    with pytest.raises(InvalidTokenError):
        verify_token(token, settings)


# ---------------------------------------------------------------- dev login


async def test_dev_token_gives_the_fixed_groups_for_that_user(client, settings):
    response = await client.post("/auth/dev-token", json={"user_id": "hr_anita"})

    body = response.json()
    assert response.status_code == 200
    assert body["expires_in"] == settings.jwt_ttl_minutes * 60
    assert verify_token(body["access_token"], settings).groups == frozenset({"employee", "hr"})


async def test_dev_token_unknown_user_is_400(client):
    response = await client.post("/auth/dev-token", json={"user_id": "ceo"})

    assert response.status_code == 400


async def test_dev_token_cannot_choose_groups(client):
    response = await client.post("/auth/dev-token", json={"user_id": "intern_ravi", "groups": ["hr"]})
    token_groups = jwt.decode(response.json()["access_token"], options={"verify_signature": False})["groups"]

    assert token_groups == ["employee"]  # the extra field had no effect


@pytest.mark.parametrize("environment", ["dev", "prod"])
async def test_dev_token_endpoint_does_not_exist_outside_local(client, settings, environment):
    non_local = settings.model_copy(update={"environment": environment})
    client._transport.app.dependency_overrides[get_settings] = lambda: non_local

    response = await client.post("/auth/dev-token", json={"user_id": "hr_anita"})

    assert response.status_code == 404


def test_api_docs_are_hidden_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("JWT_SECRET", "a-real-production-secret-0123456789abcdef")
    get_settings.cache_clear()
    try:
        app = create_app()
    finally:
        get_settings.cache_clear()

    assert (app.docs_url, app.openapi_url) == (None, None)


# ---------------------------------------------------------------- settings safety


def test_short_jwt_secret_is_refused_without_printing_it(settings):
    with pytest.raises(ValidationError, match="at least 32") as error:
        Settings(**{**settings.model_dump(), "jwt_secret": "tiny-leaky-secret"})

    assert "tiny-leaky-secret" not in str(error.value)  # error messages end up in logs


def test_placeholder_secret_is_refused_outside_local(settings):
    data = {**settings.model_dump(), "jwt_secret": "change-me-to-a-long-random-secret-please"}

    Settings(**{**data, "environment": "local"})  # fine on a laptop
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(**{**data, "environment": "prod"})


# ---------------------------------------------------------------- input validation


@pytest.mark.parametrize(
    "body",
    [{}, {"question": ""}, {"question": "    "}, {"question": "a" * 1001}, {"question": 42}],
    ids=["missing", "empty", "blank", "too-long", "not-a-string"],
)
async def test_invalid_questions_are_422_and_never_reach_the_ai(client, fake_service, settings, body):
    response = await client.post("/ask", json=body, headers=bearer("intern_ravi", settings))

    assert response.status_code == 422
    assert fake_service.calls == []


async def test_question_is_stripped_before_use(client, fake_service, settings):
    await client.post("/ask", json={"question": "  casual leave?  "}, headers=bearer("intern_ravi", settings))

    assert fake_service.calls[0][0] == "casual leave?"


# ---------------------------------------------------------------- response contract


async def test_response_contains_only_public_fields(client, fake_service, settings):
    fake_service.result = Answer(
        text="Up to 4 days [1].",
        answerable=True,
        citations=[Citation(1, __import__("uuid").uuid4(), "Leave Policy 2026",
                            "Leave Policy 2026 > 3. Leave Entitlements > 3.1 Casual Leave", 1, 2)],
        no_answer_reason=None,
        input_tokens=1234,
        output_tokens=56,
        latency_ms=789,
    )

    response = await client.post("/ask", json={"question": "casual?"}, headers=bearer("intern_ravi", settings))

    body = response.json()
    assert set(body) == {"answer", "answerable", "citations", "interaction_id"}  # nothing internal
    assert body["citations"] == [{"number": 1, "title": "Leave Policy 2026",
                                  "section": "3. Leave Entitlements > 3.1 Casual Leave", "pages": "pp. 1-2"}]
    # no tokens, no latency, no document_id, no no_answer_reason


async def test_no_answer_hides_the_reason(client, fake_service, settings):
    fake_service.result = Answer(text="Not found.", answerable=False, no_answer_reason="no_context")

    response = await client.post("/ask", json={"question": "L4 salary?"}, headers=bearer("intern_ravi", settings))

    body = response.json()
    assert (body["answer"], body["answerable"], body["citations"]) == ("Not found.", False, [])
    assert "no_answer_reason" not in body


# ---------------------------------------------------------------- upstream failures


@pytest.mark.parametrize(
    "failure",
    [errors.ServerError(503, {"error": {"message": "overloaded"}}), httpx.ConnectTimeout("timed out")],
    ids=["gemini-503", "network-timeout"],
)
async def test_ai_outage_is_503_with_a_friendly_message(client, fake_service, settings, failure):
    fake_service.result = failure

    response = await client.post("/ask", json={"question": "casual?"}, headers=bearer("intern_ravi", settings))

    assert response.status_code == 503
    assert response.json() == {"detail": "The assistant is temporarily unavailable. Please try again."}
    assert "overloaded" not in response.text  # upstream details stay in the logs


# ---------------------------------------------------------------- startup contract


async def test_startup_refuses_a_different_embedding_model(database):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(database)
    factory = async_sessionmaker(engine)
    try:
        await check_embedding_model(factory, "test-embedding-model")  # matches the seeded chunks: fine
        with pytest.raises(RuntimeError, match="index was built with"):
            await check_embedding_model(factory, "some-other-model")
    finally:
        await engine.dispose()
