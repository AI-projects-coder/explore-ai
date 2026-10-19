"""Planner node: the LLM proposes, our code decides.

A fake LLM returns a scripted PlanDraft, and a fake SEC client answers ticker lookups,
so every rule of the node is tested without real API calls.
"""

from datetime import date

import pytest

from research_analyst.core.config import Settings
from research_analyst.core.llm import LlmBadOutput
from research_analyst.nodes.planner import METRIC_DESCRIPTIONS, NO_COMPANY, planner
from research_analyst.state import AgentContext, Plan, PlanDraft, PlannedTask
from research_analyst.tools.edgar import Company, CompanyNotFound
from research_analyst.tools.xbrl import available_metrics

SETTINGS = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io",
                    max_plan_items=3, max_quarters=8)
APPLE = Company(cik=320193, ticker="AAPL", name="Apple Inc.")


def fin(topic="Revenue", metrics=("revenue",)) -> PlannedTask:
    return PlannedTask(topic=topic, question="How did it change?", kind="financials", metrics=list(metrics))


def news() -> PlannedTask:
    return PlannedTask(topic="News", question="What happened?", kind="news")


class FakeLlm:
    def __init__(self, answer: PlanDraft | Exception):
        self.answer = answer
        self.calls: list[dict] = []

    async def structured(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FakeSec:
    def __init__(self, companies=(APPLE,)):
        self.companies = {c.ticker: c for c in companies}
        self.lookups: list[str] = []

    async def resolve_ticker(self, ticker: str) -> Company:
        self.lookups.append(ticker)
        t = ticker.strip().upper()
        if t not in self.companies:
            raise CompanyNotFound(f"ticker {t!r} not found in SEC's company list")
        return self.companies[t]


class Runtime:
    """Minimal stand-in for langgraph.runtime.Runtime: nodes only use .context."""

    def __init__(self, context):
        self.context = context


async def run(request: str, answer, sec: FakeSec | None = None):
    llm, sec = FakeLlm(answer), sec or FakeSec()
    out = await planner({"request": request}, Runtime(AgentContext(settings=SETTINGS, sec=sec, llm=llm)))
    return out, llm, sec


def draft(**kw) -> PlanDraft:
    return PlanDraft(**{"ticker": "aapl", "quarters": 4, "tasks": [fin(), news()], **kw})


# --- happy path ------------------------------------------------------------------------

async def test_builds_a_plan_from_the_llm_draft():
    out, _, _ = await run("Analyze Apple's last 4 quarters", draft())
    plan = out["plan"]
    assert isinstance(plan, Plan)
    assert (plan.ticker, plan.company_name, plan.cik, plan.quarters) == ("AAPL", "Apple Inc.", 320193, 4)
    assert [(t.id, t.kind) for t in plan.tasks] == [("t1", "financials"), ("t2", "news")]
    assert "errors" not in out


async def test_company_details_come_from_sec_not_the_llm():
    out, _, sec = await run("Analyze Apple", draft(ticker="aapl"))
    assert sec.lookups == ["aapl"]
    assert out["plan"].ticker == "AAPL"            # SEC's spelling
    assert out["plan"].company_name == "Apple Inc."


async def test_ids_are_assigned_by_our_code_in_order():
    out, _, _ = await run("x", draft(tasks=[fin("A"), fin("B"), news()]))
    assert [t.id for t in out["plan"].tasks] == ["t1", "t2", "t3"]


# --- our code enforces the limits, whatever the LLM says ---------------------------------

async def test_caps_number_of_tasks():
    out, _, _ = await run("x", draft(tasks=[fin("A"), fin("B"), fin("C"), fin("D"), news()]))
    assert [t.topic for t in out["plan"].tasks] == ["A", "B", "C"]  # max_plan_items=3, most important first


async def test_caps_number_of_quarters():
    out, _, _ = await run("Analyze the last 30 quarters", draft(quarters=30))
    assert out["plan"].quarters == 8  # max_quarters


# --- requests we can't plan -----------------------------------------------------------------

async def test_not_about_a_company():
    out, _, sec = await run("What's the weather in Paris?", draft(ticker=NO_COMPANY))
    assert "plan" not in out
    assert "one US-listed public company" in out["errors"][0]
    assert sec.lookups == []  # no point asking SEC


async def test_unknown_ticker():
    out, _, _ = await run("Analyze Acme", draft(ticker="ACME"))
    assert "plan" not in out
    assert "ACME" in out["errors"][0]


async def test_llm_failure_becomes_an_error_not_a_crash():
    out, _, _ = await run("Analyze Apple", LlmBadOutput("gemini returned invalid PlanDraft"))
    assert "plan" not in out
    assert out["errors"][0].startswith("Planning failed")


@pytest.mark.parametrize("request_text", ["", "   \n "])
async def test_empty_request_never_calls_the_llm(request_text):
    out, llm, _ = await run(request_text, draft())
    assert llm.calls == []
    assert "empty" in out["errors"][0]


# --- what we send to the LLM --------------------------------------------------------------------

async def test_uses_planner_model_and_plan_schema():
    _, llm, _ = await run("Analyze Apple", draft())
    call = llm.calls[0]
    assert call["model"] == SETTINGS.planner_model
    assert call["schema"] is PlanDraft


async def test_system_prompt_lists_every_metric_and_the_limits():
    # the LLM can only use what we tell it about (real-run lesson: without this, metrics came back empty)
    _, llm, _ = await run("Analyze Apple", draft())
    system = llm.calls[0]["system"]
    for name in available_metrics():
        assert name in system
    assert "at most 3" in system
    assert "more than 8" in system
    assert NO_COMPANY in system


async def test_request_is_wrapped_as_data_with_todays_date():
    _, llm, _ = await run("Analyze Apple", draft())
    prompt = llm.calls[0]["prompt"]
    assert "<request>\nAnalyze Apple\n</request>" in prompt
    assert date.today().isoformat() in prompt


async def test_very_long_request_is_truncated():
    _, llm, _ = await run("A" * 50_000, draft())
    assert len(llm.calls[0]["prompt"]) < 2_200


def test_every_metric_has_a_description():
    assert set(METRIC_DESCRIPTIONS) == set(available_metrics())


# --- revenue is always fetched when there is a financials task ----------------------------------

async def test_adds_revenue_when_no_task_fetches_it():
    out, _, _ = await run("Focus on profitability", draft(tasks=[fin("Profit", ["net_income"]), news()]))
    assert out["plan"].tasks[0].metrics == ["revenue", "net_income"]


async def test_keeps_tasks_that_already_fetch_revenue():
    out, _, _ = await run("x", draft(tasks=[fin("Profit", ["net_income"]), fin("Sales", ["revenue"])]))
    assert [t.metrics for t in out["plan"].tasks] == [["net_income"], ["revenue"]]


async def test_news_only_plan_is_left_alone():
    out, _, _ = await run("Any news on Apple?", draft(tasks=[news()]))
    assert out["plan"].tasks[0].metrics == []
