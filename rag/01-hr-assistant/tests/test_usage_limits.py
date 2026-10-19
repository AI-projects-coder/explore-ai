"""Usage limits: per-user questions per minute/day, and the company-wide daily token budget."""
import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hr_assistant.api import routes
from hr_assistant.api.deps import get_answer_service, get_session_factory
from hr_assistant.api.main import create_app
from hr_assistant.core.auth import User, issue_token
from hr_assistant.core.config import get_settings
from hr_assistant.db.usage import BudgetExhausted, LimitExceeded, check_and_count, record_tokens, window_starts
from hr_assistant.generation.answer import Answer


class CountingService:
    def __init__(self) -> None:
        self.calls = 0

    async def answer(self, question: str, user: User) -> Answer:
        self.calls += 1
        return Answer(text="ok [1]", answerable=True, input_tokens=100, output_tokens=20)


def new_user(groups: set[str] | None = None) -> User:
    # A fresh user id per test: counters are shared in the test database, so tests must not share users.
    return User(f"user-{uuid.uuid4().hex[:8]}", frozenset(groups or {"employee"}))


def a_future_day() -> datetime:
    # A day nobody else uses, so the company-wide counter starts at 0 for this test.
    return datetime(2030, 1, 1, tzinfo=UTC) + timedelta(days=uuid.uuid4().int % 10_000)


@pytest.fixture
async def factory(database):
    engine = create_async_engine(database)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
async def api(factory):
    """An API client whose limits each test can change through `api.settings`."""
    service = CountingService()
    app = create_app()
    app.dependency_overrides[get_answer_service] = lambda: service
    app.dependency_overrides[get_session_factory] = lambda: factory

    class Api:
        settings = get_settings()

        async def ask(self, user: User):
            headers = {"Authorization": f"Bearer {issue_token(user, get_settings())}"}
            return await client.post("/ask", json={"question": "casual leave?"}, headers=headers)

    api = Api()
    api.service = service
    app.dependency_overrides[get_settings] = lambda: api.settings
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield api


def limits(**overrides):
    return get_settings().model_copy(update=overrides)


# ---------------------------------------------------------------- windows


def test_the_day_resets_at_midnight_india_time_not_utc():
    now = datetime(2026, 10, 19, 18, 45, tzinfo=UTC)  # 00:15 on 20 Oct in India

    minute, day = window_starts(now, "Asia/Kolkata")

    assert minute == datetime(2026, 10, 19, 18, 45, tzinfo=UTC)
    assert day == datetime(2026, 10, 19, 18, 30, tzinfo=UTC)  # = 20 Oct 00:00 IST


@pytest.mark.parametrize(
    ("groups", "expected"),
    [({"employee"}, 50), ({"employee", "hr"}, 200), ({"employee", "manager"}, 50)],
)
def test_daily_limit_takes_the_most_generous_group(groups, expected):
    settings = limits(rate_limit_per_day=50, rate_limit_per_day_by_group={"hr": 200})

    assert settings.daily_limit_for(frozenset(groups)) == expected


# ---------------------------------------------------------------- per-user limits through the API


async def test_minute_limit_returns_429_with_retry_after_and_never_calls_the_ai(api):
    api.settings = limits(rate_limit_per_minute=3)
    user = new_user()

    statuses = [(await api.ask(user)).status_code for _ in range(3)]
    blocked = await api.ask(user)

    assert statuses == [200, 200, 200]
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["retry-after"]) <= 61
    assert api.service.calls == 3  # the rejected request cost nothing


async def test_daily_limit_returns_429_until_midnight(api):
    api.settings = limits(rate_limit_per_day=2)
    user = new_user()

    await api.ask(user)
    await api.ask(user)
    blocked = await api.ask(user)

    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["retry-after"]) <= 24 * 3600


async def test_hr_staff_get_their_higher_daily_limit(api):
    api.settings = limits(rate_limit_per_day=1, rate_limit_per_day_by_group={"hr": 3})
    employee, hr = new_user({"employee"}), new_user({"employee", "hr"})

    assert [(await api.ask(employee)).status_code for _ in range(2)] == [200, 429]
    assert [(await api.ask(hr)).status_code for _ in range(4)] == [200, 200, 200, 429]


async def test_one_users_limit_does_not_affect_another(api):
    api.settings = limits(rate_limit_per_minute=1)
    heavy, normal = new_user(), new_user()

    await api.ask(heavy)
    assert (await api.ask(heavy)).status_code == 429
    assert (await api.ask(normal)).status_code == 200


async def test_exhausted_budget_returns_a_generic_503(api, monkeypatch):
    async def budget_gone(*args, **kwargs):
        raise BudgetExhausted

    monkeypatch.setattr(routes, "check_and_count", budget_gone)

    response = await api.ask(new_user())

    assert response.status_code == 503
    assert "budget" not in response.text.lower()  # internal reason stays in the logs
    assert api.service.calls == 0


# ---------------------------------------------------------------- counters themselves


async def test_simultaneous_requests_cannot_get_past_the_limit(factory):
    settings = limits(rate_limit_per_minute=10)
    user = new_user()
    now = a_future_day()

    async def one_request():
        try:
            await check_and_count(factory, user, settings, now=now)
            return "ok"
        except LimitExceeded:
            return "blocked"

    outcomes = await asyncio.gather(*(one_request() for _ in range(25)))  # 25 requests at the same moment

    assert outcomes.count("ok") == 10  # the atomic upsert: never 11, however the requests interleave


async def test_company_budget_blocks_everyone_once_used_up(factory):
    settings = limits(daily_token_budget=1000)
    now = a_future_day()

    await record_tokens(factory, new_user(), 1000, settings, now=now)

    with pytest.raises(BudgetExhausted):
        await check_and_count(factory, new_user(), settings, now=now)  # any user, not just the heavy one
    await check_and_count(factory, new_user(), settings, now=now + timedelta(days=1))  # next day: fine again


async def test_budget_alert_is_logged_once_when_crossing_the_threshold(factory, caplog):
    settings = limits(daily_token_budget=1000, token_budget_alert_fraction=0.8)
    user, now = new_user(), a_future_day()

    with caplog.at_level(logging.WARNING, logger="hr_assistant.db.usage"):
        await record_tokens(factory, user, 500, settings, now=now)  # 50%: quiet
        await record_tokens(factory, user, 400, settings, now=now)  # 90%: crosses 80% -> one warning
        await record_tokens(factory, user, 50, settings, now=now)  # 95%: already warned

    warnings = [r for r in caplog.records if "token budget" in r.getMessage()]
    assert len(warnings) == 1
