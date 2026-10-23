"""Postgres: the LangGraph checkpointer and our own runs table, sharing one connection pool.

Two kinds of tables live in the database:
  - LangGraph's checkpoint tables (checkpoints, checkpoint_writes, ...): its private format,
    created by AsyncPostgresSaver.setup(). We never query them directly.
  - Our `runs` table (Alembic migration): one row per run with its status and final report.
    This is what `list` and `show` read, and what a future API or dashboard would read.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from research_analyst.core.config import Settings
from research_analyst.core.llm import Spend
from research_analyst.state import CHECKPOINT_TYPES, RunStatus

MIGRATIONS = Path(__file__).parent / "migrations"


class DatabaseUnavailable(Exception):
    """Postgres can't be reached. Locally: `docker compose up -d`."""


def checkpoint_serde() -> JsonPlusSerializer:
    """Only our own classes may be rebuilt from a stored checkpoint (see state.CHECKPOINT_TYPES)."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


@asynccontextmanager
async def open_persistence(settings: Settings) -> AsyncIterator[tuple[AsyncPostgresSaver, "RunStore"]]:
    """One pool for the whole session; closed on exit, even after an error."""
    pool = AsyncConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=settings.db_pool_max_size,
        open=False,
        # autocommit + dict rows: what AsyncPostgresSaver expects. prepare_threshold=None turns off
        # server-side prepared statements (they break behind connection poolers like PgBouncer).
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None},
    )
    try:
        await pool.open(wait=True, timeout=10)
    except Exception as e:  # psycopg_pool raises PoolTimeout; the cause is in the log
        raise DatabaseUnavailable(
            f"cannot connect to Postgres at {_safe_url(settings.database_url)} "
            "(locally: run `docker compose up -d` in agents/01-research-analyst)"
        ) from e
    try:
        yield AsyncPostgresSaver(pool, serde=checkpoint_serde()), RunStore(pool)
    finally:
        await pool.close()


def run_migrations(settings: Settings) -> None:
    """Create or upgrade our own tables (Alembic). LangGraph's tables come from checkpointer.setup()."""
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1))
    command.upgrade(cfg, "head")


class RunStore:
    """Our runs table. Every write is an upsert or a guarded update, so repeating it is harmless."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create(self, run_id: str, request: str) -> None:
        await self._execute(
            "INSERT INTO runs (run_id, request, status) VALUES (%s, %s, 'running') ON CONFLICT (run_id) DO NOTHING",
            (run_id, request),
        )

    async def set_status(self, run_id: str, status: RunStatus) -> None:
        # never move a finished run back to running/awaiting_review (e.g. a late duplicate update)
        await self._execute(
            "UPDATE runs SET status = %s, updated_at = now() WHERE run_id = %s "
            "AND status IN ('running', 'awaiting_review')",
            (status, run_id),
        )

    async def finish(self, run_id: str, *, status: RunStatus, ticker: str | None,
                     report_markdown: str, spend: Spend | None) -> None:
        await self._execute(
            "UPDATE runs SET status = %s, ticker = %s, report_markdown = %s, spend = %s, "
            "updated_at = now(), finished_at = COALESCE(finished_at, now()) WHERE run_id = %s",
            (status, ticker, report_markdown, json.dumps((spend or Spend()).model_dump()), run_id),
        )

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,))
            return await cur.fetchone()

    async def list(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT run_id, status, ticker, request, created_at, updated_at FROM runs "
                "ORDER BY created_at DESC LIMIT %s", (limit,))
            return await cur.fetchall()

    async def _execute(self, sql: str, params: tuple) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(sql, params)


def _safe_url(url: str) -> str:
    """The URL without its password, for error messages and logs."""
    if "@" not in url:
        return url
    scheme_user, host = url.rsplit("@", 1)
    scheme, _, user_pass = scheme_user.partition("://")
    return f"{scheme}://{user_pass.split(':', 1)[0]}:***@{host}"
