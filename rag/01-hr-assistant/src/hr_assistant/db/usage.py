"""Usage limits: per-user request limits (minute, day) and a company-wide daily token budget.

Counters live in Postgres, so every API instance sees the same numbers. Each increment is ONE atomic
upsert, so two simultaneous requests can't both read "9" and both pass a limit of 10.
"""
import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hr_assistant.core.auth import User
from hr_assistant.core.config import Settings
from hr_assistant.db.models import UsageCounter, UsagePeriod

logger = logging.getLogger(__name__)

ORG_KEY = "org"

_COUNT_REQUEST = text("""
    INSERT INTO usage_counters (key, period, window_start, requests) VALUES (:key, :period, :start, 1)
    ON CONFLICT (key, period, window_start) DO UPDATE SET requests = usage_counters.requests + 1
    RETURNING requests
""")
_ADD_TOKENS = text("""
    INSERT INTO usage_counters (key, period, window_start, tokens) VALUES (:key, 'day', :start, :tokens)
    ON CONFLICT (key, period, window_start) DO UPDATE SET tokens = usage_counters.tokens + :tokens
    RETURNING tokens
""")


class LimitExceeded(Exception):
    def __init__(self, period: str, retry_after: int) -> None:
        super().__init__(f"{period} limit exceeded")
        self.period = period
        self.retry_after = retry_after  # seconds until the window resets (sent as the Retry-After header)


class BudgetExhausted(Exception):
    """The company-wide daily token budget is used up: protects the bill from bugs and abuse."""


def window_starts(now: datetime, timezone: str) -> tuple[datetime, datetime]:
    """(start of the current minute, start of the current day in `timezone`), both as aware datetimes."""
    minute = now.astimezone(UTC).replace(second=0, microsecond=0)
    day = now.astimezone(ZoneInfo(timezone)).replace(hour=0, minute=0, second=0, microsecond=0)
    return minute, day


async def check_and_count(
    session_factory: async_sessionmaker[AsyncSession], user: User, settings: Settings, now: datetime | None = None
) -> None:
    """Count this request; raise if it goes over a limit. Called BEFORE any paid AI call.

    FAIL-OPEN: if the counters can't be reached, the request is allowed and an ERROR is logged (alert on it).
    Users are authenticated employees, so availability wins over strict enforcement during an incident.
    """
    now = now or datetime.now(UTC)
    minute_start, day_start = window_starts(now, settings.usage_timezone)
    key = f"user:{user.id}"
    try:
        async with session_factory() as session, session.begin():
            org_tokens = await session.scalar(
                select(UsageCounter.tokens).where(
                    UsageCounter.key == ORG_KEY,
                    UsageCounter.period == UsagePeriod.DAY,
                    UsageCounter.window_start == day_start,
                )
            )
            if (org_tokens or 0) >= settings.daily_token_budget:
                raise BudgetExhausted
            minute_row = {"key": key, "period": "minute", "start": minute_start}
            day_row = {"key": key, "period": "day", "start": day_start}
            per_minute = (await session.execute(_COUNT_REQUEST, minute_row)).scalar_one()
            per_day = (await session.execute(_COUNT_REQUEST, day_row)).scalar_one()
    except SQLAlchemyError:
        logger.exception("usage limits unavailable: allowing the request for user=%s (fail-open)", user.id)
        return

    if per_minute > settings.rate_limit_per_minute:
        raise LimitExceeded("minute", _seconds_until(minute_start + timedelta(minutes=1), now))
    if per_day > settings.daily_limit_for(user.groups):
        raise LimitExceeded("day", _seconds_until(day_start + timedelta(days=1), now))


async def record_tokens(
    session_factory: async_sessionmaker[AsyncSession],
    user: User,
    tokens: int,
    settings: Settings,
    now: datetime | None = None,
) -> None:
    """Add the answer's tokens to the user's and the company's daily totals. Never fails the request."""
    if tokens <= 0:
        return
    now = now or datetime.now(UTC)
    _, day_start = window_starts(now, settings.usage_timezone)
    try:
        async with session_factory() as session, session.begin():
            await session.execute(_ADD_TOKENS, {"key": f"user:{user.id}", "start": day_start, "tokens": tokens})
            org_row = {"key": ORG_KEY, "start": day_start, "tokens": tokens}
            org_total = (await session.execute(_ADD_TOKENS, org_row)).scalar_one()
    except Exception:
        logger.exception("could not record token usage for user=%s", user.id)
        return
    threshold = settings.token_budget_alert_fraction * settings.daily_token_budget
    if org_total - tokens < threshold <= org_total:  # log once, at the moment the threshold is crossed
        logger.warning(
            "daily token budget %.0f%% used: %d of %d tokens",
            100 * settings.token_budget_alert_fraction, org_total, settings.daily_token_budget,
        )


def _seconds_until(moment: datetime, now: datetime) -> int:
    return max(1, int((moment - now).total_seconds()) + 1)
