"""Delete data we no longer need: old interactions (personal data) and old usage counters.

Deletes in small batches, each in its own short transaction: a single huge DELETE would hold one long
transaction (blocking VACUUM, as seen in the MVCC demo), lock rows for a long time, and roll back ALL
its work if it failed near the end.
"""
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from hr_assistant.core.config import Settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000

# table -> its timestamp column. Fixed names only: they go into SQL text, so never take them from input.
_TABLES = {"interactions": "created_at", "usage_counters": "window_start"}


def _cutoffs(settings: Settings, now: datetime) -> dict[str, datetime]:
    return {
        "interactions": now - timedelta(days=settings.interaction_retention_days),
        "usage_counters": now - timedelta(days=settings.usage_counter_retention_days),
    }


def count_expired(
    session_factory: sessionmaker[Session], settings: Settings, now: datetime | None = None
) -> dict[str, int]:
    """What purge_expired WOULD delete (for --dry-run)."""
    now = now or datetime.now(UTC)
    cutoffs = _cutoffs(settings, now)
    with session_factory() as session:
        return {
            table: session.execute(
                text(f"SELECT count(*) FROM {table} WHERE {column} < :cutoff"), {"cutoff": cutoffs[table]}
            ).scalar_one()
            for table, column in _TABLES.items()
        }


def purge_expired(
    session_factory: sessionmaker[Session],
    settings: Settings,
    now: datetime | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """Delete expired rows, batch by batch. Feedback is removed with its interaction (ON DELETE CASCADE)."""
    now = now or datetime.now(UTC)
    cutoffs = _cutoffs(settings, now)
    deleted: dict[str, int] = {}
    for table, column in _TABLES.items():
        # ctid = Postgres' physical row address: picks "N rows" for any table, whatever its primary key.
        stmt = text(
            f"DELETE FROM {table} WHERE ctid IN "
            f"(SELECT ctid FROM {table} WHERE {column} < :cutoff LIMIT :batch)"
        )
        total = 0
        while True:
            with session_factory.begin() as session:  # one short transaction per batch
                count = session.execute(stmt, {"cutoff": cutoffs[table], "batch": batch_size}).rowcount
            total += count
            if count < batch_size:  # a partial batch means nothing expired is left
                break
        deleted[table] = total

    logger.info(
        "retention cleanup done",
        extra={"event": "retention_cleanup", **{f"deleted_{table}": n for table, n in deleted.items()}},
    )
    return deleted
