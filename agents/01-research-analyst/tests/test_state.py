"""State and schemas: what the LLM may produce, how parallel results merge, and that a run can be saved and restored."""

import asyncio
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import ValidationError

from research_analyst.core.config import Settings
from research_analyst.core.llm import Llm
from research_analyst.finance.metrics import MetricResult
from research_analyst.state import (
    CHECKPOINT_TYPES,
    AgentContext,
    AgentState,
    Claim,
    Finding,
    Plan,
    PlanDraft,
    PlannedTask,
    Report,
    ResearcherInput,
    ResearchTask,
    Source,
    merge_findings,
)
from research_analyst.tools.edgar import EdgarClient
from research_analyst.tools.xbrl import QuarterValue

QV = QuarterValue(
    metric="revenue", fiscal_year=2026, fiscal_quarter=1, period_start=date(2025, 10, 1),
    period_end=date(2025, 12, 31), value=143.76e9, unit="USD", source="reported",
    concept="X", form="10-Q", accession="0000320193-26-000001", filed=date(2026, 1, 30),
)


def task(**kw) -> dict:
    return {"topic": "Revenue trend", "question": "How did revenue change?", "kind": "financials",
            "metrics": ["revenue"], **kw}


def finding(task_id: str, summary: str = "s") -> Finding:
    return Finding(task_id=task_id, topic="t", status="ok", summary=summary)


# --- the planner's output is validated before anything runs -----------------------

def test_valid_plan_draft():
    d = PlanDraft(ticker="AAPL", quarters=4, tasks=[task(), task(kind="news", metrics=[])])
    assert len(d.tasks) == 2


@pytest.mark.parametrize("bad", [
    {"ticker": "", "quarters": 4, "tasks": [task()]},
    {"ticker": "AAPL", "quarters": 0, "tasks": [task()]},
    {"ticker": "AAPL", "quarters": 41, "tasks": [task()]},
    {"ticker": "AAPL", "quarters": 4, "tasks": []},                          # a plan with nothing to do
    {"ticker": "AAPL", "quarters": 4, "tasks": [task(kind="gossip")]},
    {"ticker": "AAPL", "quarters": 4, "tasks": [task(topic="")]},
    {"ticker": "AAPL", "quarters": 4, "tasks": [task(question="x" * 501)]},
])
def test_invalid_plan_draft_rejected(bad):
    with pytest.raises(ValidationError):
        PlanDraft(**bad)


def test_unknown_metric_rejected_with_the_valid_choices():
    # the error message goes back to the LLM so it can correct itself
    with pytest.raises(ValidationError, match="ebitda") as exc:
        PlannedTask(**task(metrics=["revenue", "ebitda"]))
    assert "net_income" in str(exc.value)


def test_financials_task_needs_metrics():
    # real-run lesson: without the metric list in the prompt, the planner returned financials tasks with no metrics
    with pytest.raises(ValidationError, match="at least one metric"):
        PlannedTask(**task(metrics=[]))


def test_news_task_has_no_metrics():
    with pytest.raises(ValidationError, match="must not list metrics"):
        PlannedTask(**task(kind="news", metrics=["revenue"]))


def test_duplicate_metrics_removed_in_order():
    assert PlannedTask(**task(metrics=["net_income", "revenue", "net_income"])).metrics == ["net_income", "revenue"]


def test_plan_draft_parses_llm_json():
    # this is what the planner node will do with Gemini's structured output
    raw = '{"ticker": "AAPL", "quarters": 4, "tasks": [{"topic": "Revenue", "question": "q", "kind": "financials", "metrics": ["revenue"]}]}'
    assert PlanDraft.model_validate_json(raw).tasks[0].metrics == ["revenue"]


def test_task_id_comes_from_our_code_not_the_llm():
    # PlannedTask (what the LLM writes) has no id field; ResearchTask (after our code accepts it) does
    assert "id" not in PlannedTask.model_fields
    t = ResearchTask(id="t1", **task())
    assert t.id == "t1"


def test_models_are_immutable():
    # a node must return changes, never edit state in place (LangGraph wouldn't see the edit)
    d = PlanDraft(ticker="AAPL", quarters=4, tasks=[task()])
    with pytest.raises(ValidationError):
        d.quarters = 8


# --- writer schema ---------------------------------------------------------------------

def test_report_and_claims():
    r = Report(title="Apple Q3", summary="Revenue grew.", body_markdown="# Apple",
               claims=[Claim(text="Revenue grew 16.4% YoY", kind="numeric", metric_id="m1", stated_value=0.164),
                       Claim(text="Services is the growth driver", kind="qualitative")])
    assert [c.kind for c in r.claims] == ["numeric", "qualitative"]


def test_claim_kind_is_restricted():
    with pytest.raises(ValidationError):
        Claim(text="x", kind="opinion")


# --- merge_findings: the reducer for parallel researchers -------------------------------

def test_merge_combines_parallel_results():
    merged = merge_findings([finding("t1")], [finding("t2")])
    assert [f.task_id for f in merged] == ["t1", "t2"]


def test_merge_replaces_a_rerun_instead_of_duplicating():
    # the same researcher ran twice (retry, or a duplicate delivery in v2): keep one, the newest
    merged = merge_findings([finding("t1", "old"), finding("t2")], [finding("t1", "new")])
    assert [(f.task_id, f.summary) for f in merged] == [("t1", "new"), ("t2", "s")]


def test_merge_with_empty_inputs():
    assert merge_findings([], []) == []
    assert merge_findings([], [finding("t1")]) == [finding("t1")]


# --- the whole design inside a real LangGraph graph --------------------------------------

def build_probe_graph():
    """A tiny graph shaped like the real one: plan -> parallel researchers -> finish."""

    def planner(state: AgentState):
        tasks = [ResearchTask(id="t1", **task()), ResearchTask(id="t2", **task(kind="news", metrics=[]))]
        return {"plan": Plan(ticker="AAPL", company_name="Apple Inc.", cik=320193, quarters=4, tasks=tasks)}

    def fan_out(state: AgentState):
        p = state["plan"]
        return [Send("researcher", ResearcherInput(task=t, ticker=p.ticker, quarters=p.quarters)) for t in p.tasks]

    async def researcher(inp: ResearcherInput, runtime: Runtime[AgentContext]):
        assert runtime.context.settings.gcp_project == "p"  # tools arrive via context, not state
        return {"findings": [Finding(
            task_id=inp["task"].id, topic=inp["task"].topic, status="ok", summary="found it", data=[QV],
            sources=[Source(kind="xbrl", accession=QV.accession, filed=QV.filed, description="10-Q")],
        )]}

    def finish(state: AgentState):
        metric = MetricResult(name="revenue_qoq_growth", period_label="Q1 FY2026", period_end=QV.period_end,
                              value=0.1, unit="ratio", formula="f", inputs=(QV,))
        return {"metrics": {"m1": metric}, "errors": ["t2: no 8-K filings found"]}

    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("finish", finish)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["researcher"])
    g.add_edge("researcher", "finish")
    g.add_edge("finish", END)
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    return g.compile(checkpointer=InMemorySaver(serde=serde))


async def test_state_works_in_a_real_graph_and_survives_a_checkpoint(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")  # fail on any type missing from CHECKPOINT_TYPES
    settings = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io")
    app = build_probe_graph()
    config = {"configurable": {"thread_id": "run-1"}}

    async with EdgarClient(settings) as sec:
        context = AgentContext(settings=settings, sec=sec, llm=Llm(settings, client=object()))  # no LLM calls here
        out = await app.ainvoke({"request": "Analyze AAPL"}, config, context=context)

    # both parallel researchers' findings arrived, merged by the reducer
    assert sorted(f.task_id for f in out["findings"]) == ["t1", "t2"]
    assert out["errors"] == ["t2: no 8-K filings found"]

    # reload the saved checkpoint: every object comes back as its real class, nested ones too
    saved = (await app.aget_state(config)).values
    assert isinstance(saved["plan"], Plan)
    assert isinstance(saved["findings"][0].data[0], QuarterValue)
    assert isinstance(saved["metrics"]["m1"].inputs[0], QuarterValue)
    assert saved["findings"][0].sources[0].filed == date(2026, 1, 30)
