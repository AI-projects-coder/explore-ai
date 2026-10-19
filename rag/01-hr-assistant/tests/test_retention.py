"""Data retention: expired interactions (personal data) and usage counters are deleted, nothing else."""
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from hr_assistant.core.config import Settings, get_settings
from hr_assistant.db.models import Feedback, Interaction, UsageCounter
from hr_assistant.db.retention import count_expired, purge_expired

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
SETTINGS = get_settings().model_copy(update={"interaction_retention_days": 90, "usage_counter_retention_days": 2})


@pytest.fixture
def factory(database):
    engine = create_engine(database)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def add_interaction(factory, age: timedelta, with_feedback: bool = False) -> uuid.UUID:
    with factory.begin() as s:
        row = Interaction(user_id=f"u-{uuid.uuid4().hex[:6]}", question="q", answer="a", answerable=True,
                          created_at=NOW - age)
        s.add(row)
        s.flush()
        if with_feedback:
            s.add(Feedback(interaction_id=row.id, user_id=row.user_id, rating="down"))
        return row.id


def add_counter(factory, age: timedelta) -> str:
    key = f"user:u-{uuid.uuid4().hex[:6]}"
    with factory.begin() as s:
        s.add(UsageCounter(key=key, period="day", window_start=NOW - age, requests=1))
    return key


def exists(factory, model, **where) -> bool:
    with factory() as s:
        return s.scalar(select(model).filter_by(**where)) is not None


def test_old_interactions_are_deleted_and_recent_ones_kept(factory):
    old = add_interaction(factory, timedelta(days=91))
    recent = add_interaction(factory, timedelta(days=89))

    purge_expired(factory, SETTINGS, now=NOW)

    assert not exists(factory, Interaction, id=old)
    assert exists(factory, Interaction, id=recent)


def test_feedback_goes_with_its_interaction(factory):
    old = add_interaction(factory, timedelta(days=100), with_feedback=True)
    recent = add_interaction(factory, timedelta(days=1), with_feedback=True)

    purge_expired(factory, SETTINGS, now=NOW)

    assert not exists(factory, Feedback, interaction_id=old)  # ON DELETE CASCADE
    assert exists(factory, Feedback, interaction_id=recent)


def test_old_usage_counters_are_deleted_and_current_ones_kept(factory):
    old = add_counter(factory, timedelta(days=3))
    today = add_counter(factory, timedelta(hours=5))

    purge_expired(factory, SETTINGS, now=NOW)

    assert not exists(factory, UsageCounter, key=old)
    assert exists(factory, UsageCounter, key=today)  # still needed for today's daily limit


def test_deletes_everything_expired_even_across_many_small_batches(factory):
    ids = [add_interaction(factory, timedelta(days=200)) for _ in range(7)]

    purge_expired(factory, SETTINGS, now=NOW, batch_size=2)  # 2 + 2 + 2 + 1

    assert not any(exists(factory, Interaction, id=i) for i in ids)


def test_dry_run_counts_but_deletes_nothing(factory):
    old = add_interaction(factory, timedelta(days=95))

    counts = count_expired(factory, SETTINGS, now=NOW)

    assert counts["interactions"] >= 1
    assert exists(factory, Interaction, id=old)


def test_cleanup_logs_a_retention_event_with_counts(factory, caplog):
    add_interaction(factory, timedelta(days=120))

    with caplog.at_level(logging.INFO, logger="hr_assistant.db.retention"):
        deleted = purge_expired(factory, SETTINGS, now=NOW)

    record = next(r for r in caplog.records if getattr(r, "event", None) == "retention_cleanup")
    assert record.deleted_interactions == deleted["interactions"] >= 1  # alert if this event stops appearing


@pytest.mark.parametrize("field", ["interaction_retention_days", "usage_counter_retention_days"])
def test_zero_retention_is_refused_so_a_typo_cannot_delete_everything(field):
    with pytest.raises(ValidationError):
        Settings(**{**get_settings().model_dump(), field: 0})
