"""Postgres integration: the runs table and the real checkpointer. Skipped when the database isn't running.

    docker compose up -d && uv run research-analyst init-db
"""

import uuid
from datetime import date

import pytest
from langgraph.types import Command

from research_analyst.core.config import Settings
from research_analyst.core.llm import Spend
from research_analyst.graph import advance, build_graph
from research_analyst.storage import DatabaseUnavailable, open_persistence
from research_analyst.state import AgentContext

SETTINGS = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io")


@pytest.fixture
async def persistence():
    try:
        async with open_persistence(SETTINGS) as (checkpointer, runs):
            await checkpointer.setup()
            yield checkpointer, runs
    except DatabaseUnavailable:
        pytest.skip("Postgres is not running (docker compose up -d)")


def run_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


async def test_runs_lifecycle(persistence):
    _, runs = persistence
    rid = run_id()
    await runs.create(rid, "Analyze Apple")
    await runs.create(rid, "duplicate create is ignored")  # idempotent
    assert (await runs.get(rid))["request"] == "Analyze Apple"

    await runs.set_status(rid, "awaiting_review")
    await runs.finish(rid, status="approved", ticker="AAPL", report_markdown="# Apple", spend=Spend(llm_calls=4))
    await runs.set_status(rid, "running")  # a late update must not reopen a finished run
    row = await runs.get(rid)
    assert (row["status"], row["ticker"], row["report_markdown"]) == ("approved", "AAPL", "# Apple")
    assert row["spend"]["llm_calls"] == 4 and row["finished_at"] is not None
    assert any(r["run_id"] == rid for r in await runs.list(50))


async def test_status_is_constrained_in_the_database(persistence):
    _, runs = persistence
    rid = run_id()
    await runs.create(rid, "x")
    with pytest.raises(Exception, match="runs_status_check"):
        await runs.finish(rid, status="published", ticker=None, report_markdown="", spend=None)  # type: ignore[arg-type]


async def test_real_checkpointer_pauses_and_resumes_across_graph_instances(persistence, monkeypatch):
    """Two different graph objects (= two processes): the second continues what the first paused."""
    from tests.test_graph import FakeRuns, SchemaLlm, answers
    from tests.test_nodes import FakeSec

    class D(date):
        @classmethod
        def today(cls):
            return date(2026, 10, 20)
    monkeypatch.setattr("research_analyst.nodes.researcher.date", D)
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")

    checkpointer, _ = persistence
    rid = run_id()
    ctx = AgentContext(settings=SETTINGS, sec=FakeSec(), llm=SchemaLlm(answers()), run_id=rid, runs=FakeRuns())

    first = build_graph(checkpointer)
    snap = await advance(first, rid, {"request": "Analyze Apple"}, ctx)
    assert snap.next == ("human_review",)

    second = build_graph(checkpointer)  # a new "process" reading the same Postgres
    snap = await advance(second, rid, Command(resume={"action": "approve"}), ctx)
    assert snap.values["status"] == "approved"
