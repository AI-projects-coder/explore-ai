"""Interactions are logged on /ask; feedback can only be given on your OWN interactions."""
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hr_assistant.api.deps import get_answer_service, get_session_factory
from hr_assistant.api.main import create_app
from hr_assistant.core.auth import DEV_USERS, User, issue_token
from hr_assistant.core.config import get_settings
from hr_assistant.db.models import Feedback, Interaction
from hr_assistant.generation.answer import Answer


class FakeAnswerService:
    async def answer(self, question: str, user: User) -> Answer:
        return Answer(text="Up to 4 days [1].", answerable=True, input_tokens=100, output_tokens=20, latency_ms=50)


@pytest.fixture
async def env(database):
    engine = create_async_engine(database)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    app = create_app()
    app.dependency_overrides[get_answer_service] = lambda: FakeAnswerService()
    app.dependency_overrides[get_session_factory] = lambda: factory
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield app, client, factory
    await engine.dispose()


def auth(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(User(user_id, DEV_USERS[user_id]), get_settings())}"}


async def ask(client: AsyncClient, user_id: str, question: str = "casual leave?") -> str:
    response = await client.post("/ask", json={"question": question}, headers=auth(user_id))
    assert response.status_code == 200
    return response.json()["interaction_id"]


async def test_ask_logs_the_interaction_and_returns_its_id(env):
    _, client, factory = env

    interaction_id = await ask(client, "intern_ravi", "How many casual leaves can I carry forward?")

    async with factory() as s:
        row = await s.get(Interaction, uuid.UUID(interaction_id))
    assert (row.user_id, row.question, row.answer, row.answerable) == (
        "intern_ravi", "How many casual leaves can I carry forward?", "Up to 4 days [1].", True,
    )
    assert (row.input_tokens, row.output_tokens, row.latency_ms) == (100, 20, 50)


async def test_feedback_on_own_interaction_is_saved(env):
    _, client, factory = env
    interaction_id = await ask(client, "intern_ravi")

    response = await client.post(
        "/feedback", json={"interaction_id": interaction_id, "rating": "up"}, headers=auth("intern_ravi")
    )

    assert response.status_code == 204
    async with factory() as s:
        rating = await s.scalar(select(Feedback.rating).where(Feedback.interaction_id == uuid.UUID(interaction_id)))
    assert rating == "up"


async def test_second_click_changes_the_rating_instead_of_adding_a_row(env):
    _, client, factory = env
    interaction_id = await ask(client, "intern_ravi")
    headers = auth("intern_ravi")

    await client.post("/feedback", json={"interaction_id": interaction_id, "rating": "up"}, headers=headers)
    await client.post(
        "/feedback", json={"interaction_id": interaction_id, "rating": "down", "comment": "wrong number"},
        headers=headers,
    )

    async with factory() as s:
        rows = (await s.scalars(select(Feedback).where(Feedback.interaction_id == uuid.UUID(interaction_id)))).all()
    assert [(r.rating, r.comment) for r in rows] == [("down", "wrong number")]


async def test_cannot_give_feedback_on_someone_elses_interaction(env):
    _, client, factory = env
    managers_interaction = await ask(client, "manager_priya", "L4 salary?")

    response = await client.post(
        "/feedback", json={"interaction_id": managers_interaction, "rating": "down"}, headers=auth("intern_ravi")
    )

    assert response.status_code == 404
    async with factory() as s:
        assert await s.scalar(select(func.count()).select_from(Feedback).where(
            Feedback.interaction_id == uuid.UUID(managers_interaction))) == 0


async def test_unknown_and_foreign_interactions_get_the_same_404(env):
    _, client, _ = env
    managers_interaction = await ask(client, "manager_priya")

    unknown = await client.post(
        "/feedback", json={"interaction_id": str(uuid.uuid4()), "rating": "up"}, headers=auth("intern_ravi")
    )
    foreign = await client.post(
        "/feedback", json={"interaction_id": managers_interaction, "rating": "up"}, headers=auth("intern_ravi")
    )

    assert (unknown.status_code, unknown.json()) == (foreign.status_code, foreign.json())  # indistinguishable


@pytest.mark.parametrize(
    "body",
    [{"rating": "up"}, {"interaction_id": "not-a-uuid", "rating": "up"},
     {"interaction_id": str(uuid.uuid4()), "rating": "meh"},
     {"interaction_id": str(uuid.uuid4()), "rating": "up", "comment": "x" * 1001},
     {"interaction_id": str(uuid.uuid4()), "rating": "up", "user_id": "manager_priya"}],
    ids=["missing-id", "bad-uuid", "bad-rating", "comment-too-long", "extra-field"],
)
async def test_invalid_feedback_is_422(env, body):
    _, client, _ = env

    response = await client.post("/feedback", json=body, headers=auth("intern_ravi"))

    assert response.status_code == 422


async def test_feedback_requires_login(env):
    _, client, _ = env

    response = await client.post("/feedback", json={"interaction_id": str(uuid.uuid4()), "rating": "up"})

    assert response.status_code == 401


async def test_if_logging_fails_the_user_still_gets_the_answer(env):
    app, client, _ = env
    broken = async_sessionmaker(create_async_engine("postgresql+psycopg://hr:hr@localhost:1/nowhere"))
    app.dependency_overrides[get_session_factory] = lambda: broken

    response = await client.post("/ask", json={"question": "casual leave?"}, headers=auth("intern_ravi"))

    assert response.status_code == 200
    assert response.json()["answer"] == "Up to 4 days [1]."
    assert response.json()["interaction_id"] is None  # the UI simply hides the feedback buttons
