"""v1 building blocks: the critic's number check, the budget, review decisions, final status, rendering."""

from datetime import date

import pytest
from pydantic import ValidationError

from research_analyst.core.config import Settings
from research_analyst.core.llm import BudgetExceeded, Llm, Spend, max_spend
from research_analyst.finance.metrics import MetricResult
from research_analyst.nodes.critic import Judgement, check_numbers
from research_analyst.nodes.finalize import final_status
from research_analyst.report import render
from research_analyst.state import Critique, Finding, Plan, Report, ReviewDecision
from research_analyst.storage import _safe_url
from tests.test_llm import FakeGemini, Pick, response

SETTINGS = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io")


def metric(name, value, unit) -> MetricResult:
    return MetricResult(name=name, period_label="Q3 FY2026", period_end=date(2026, 6, 27), value=value,
                        unit=unit, formula="f", inputs=())


METRICS = {
    "m1": metric("revenue_yoy_growth", 0.164, "ratio"),     # +16.4%
    "m2": metric("revenue", 109.42e9, "USD"),               # $109.42B
    "m3": metric("revenue_qoq_growth", -0.016, "ratio"),    # -1.6%
    "m4": metric("eps_diluted", 2.02, "USD/shares"),        # $2.02
    "m5": metric("capital_expenditure", 512.3e6, "USD"),    # $512.30M
}


def report(body: str, summary: str = "s", title: str = "t") -> Report:
    return Report(title=title, summary=summary, body_markdown=body)


# --- critic: every figure must be cited and correct ------------------------------------------

@pytest.mark.parametrize("body", [
    "Revenue grew +16.4% [m1] to $109.42B [m2].",
    "Revenue fell 1.6% [m3] quarter over quarter.",     # unsigned is fine: "fell" carries the sign
    "Revenue fell -1.6% [m3].",
    "EPS was $2.02 [m4]; capex $512.30M [m5].",
    "Growth of 16.36% [m1] rounds the same.",            # within rounding tolerance
    "Q3 FY2026 had 4 quarters of data and 8-K 0000320193-26-000018.",  # labels, counts, ids: not figures
])
def test_correct_reports_pass(body):
    assert check_numbers(report(body), METRICS) == []


@pytest.mark.parametrize("body,problem", [
    ("Revenue reached $193 billion.", "'$193 billion' has no [id]"),
    ("Revenue grew 18.2% [m1].", "does not match m1 = +16.4%"),
    ("Revenue grew +1.6% [m3].", "does not match m3 = -1.6%"),         # explicit wrong sign
    ("Revenue was $109.42M [m2].", "does not match m2 = $109.42B"),    # wrong scale
    ("Revenue grew 16.4% [m2].", "does not match m2 = $109.42B"),      # percent vs dollars
    ("Margin was 27.2% [m99].", "m99, which is not in the metrics table"),
])
def test_wrong_or_uncited_figures_are_problems(body, problem):
    problems = check_numbers(report(body), METRICS)
    assert len(problems) == 1 and problem in problems[0]


def test_title_and_summary_are_checked_too():
    problems = check_numbers(report("ok", summary="Net income was nearly $193 billion.", title="Up 16.4%"), METRICS)
    assert [p.split(":")[0] for p in problems] == ["title", "summary"]


def test_judge_verdict_index_is_validated():
    with pytest.raises(ValidationError, match="out of range"):
        Judgement.model_validate({"verdicts": [{"claim": 3, "supported": True, "reason": "r"}]}, context={"claims": 2})


# --- budget -----------------------------------------------------------------------------------------

def make_llm(*script, clock=None, **settings) -> Llm:
    s = SETTINGS.model_copy(update=settings)
    return Llm(s, client=FakeGemini(*script), **({"clock": clock} if clock else {}))


async def ask(llm):
    return await llm.structured(model="m", system="s", prompt="p", schema=Pick)


async def test_call_limit_blocks_before_the_call():
    llm = make_llm(response(), response(), budget_max_llm_calls=1)
    await ask(llm)
    with pytest.raises(BudgetExceeded, match="call limit"):
        await ask(llm)
    assert llm.usage.calls == 1  # the blocked call was never made


async def test_token_limit():
    llm = make_llm(response(prompt_tokens=90, output_tokens=5, thinking_tokens=5), budget_max_tokens=100)
    await ask(llm)
    with pytest.raises(BudgetExceeded, match="token limit"):
        await ask(llm)


async def test_usd_limit_uses_the_price_table():
    prices = {"m": (1.0, 10.0), "gemini-3.1-pro-preview": (1, 1), "gemini-3-flash-preview": (1, 1)}
    llm = make_llm(response(prompt_tokens=1_000_000, output_tokens=0, thinking_tokens=0), response(),
                   budget_max_usd=0.5, model_prices=prices, budget_max_tokens=10_000_000)
    await ask(llm)
    assert llm.spend().usd == pytest.approx(1.0)
    with pytest.raises(BudgetExceeded, match="cost limit"):
        await ask(llm)


async def test_time_limit_per_session():
    now = [0.0]
    llm = make_llm(response(), clock=lambda: now[0], budget_max_seconds=60)
    now[0] = 61
    with pytest.raises(BudgetExceeded, match="time limit"):
        await ask(llm)


async def test_baseline_from_an_earlier_session_counts():
    # after a crash or a review pause, the budget continues instead of starting again at zero
    llm = make_llm(response(), budget_max_llm_calls=5)
    llm.set_baseline(Spend(llm_calls=5))
    with pytest.raises(BudgetExceeded):
        await ask(llm)


async def test_repairs_count_against_the_budget():
    llm = make_llm(response('{"number": 7}'), response('{"number": 8}'), budget_max_llm_calls=1)
    with pytest.raises(BudgetExceeded):
        await ask(llm)  # first answer invalid; the repair call is blocked


def test_max_spend_reducer_keeps_the_running_total():
    a, b = Spend(llm_calls=3, input_tokens=100), Spend(llm_calls=2, input_tokens=300, usd=0.1)
    assert max_spend(a, b) == Spend(llm_calls=3, input_tokens=300, usd=0.1)
    assert max_spend(None, a) == a


def test_usd_budget_requires_prices_for_every_model():
    with pytest.raises(ValidationError, match="MODEL_PRICES has no price"):
        Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io", budget_max_usd=1.0)


# --- review decisions ---------------------------------------------------------------------------------

def test_review_decisions():
    assert ReviewDecision(action="approve").feedback == ""
    with pytest.raises(ValidationError, match="needs feedback"):
        ReviewDecision(action="revise", feedback=" ")
    with pytest.raises(ValidationError):
        ReviewDecision(action="publish")


# --- final status and rendering ---------------------------------------------------------------------------

def finding(status="ok"):
    return Finding(task_id="t1", topic="Revenue", status=status, summary="revenue: 6 quarters")


@pytest.mark.parametrize("state,expected", [
    ({"decision": "approved"}, "approved"),
    ({"decision": "rejected"}, "rejected"),
    ({"metrics": METRICS}, "partial"),
    ({"findings": [finding()]}, "partial"),
    ({"findings": [finding("failed")]}, "failed"),
    ({}, "failed"),
])
def test_final_status(state, expected):
    assert final_status(state) == expected


def test_data_only_report_when_writer_failed():
    plan = Plan(ticker="AAPL", company_name="Apple Inc.", cik=320193, quarters=4, tasks=[])
    md = render({"plan": plan, "metrics": {"m2": METRICS["m2"]}, "findings": [finding()],
                 "errors": ["Writing the report failed: LLM call limit reached (2 of 2)"]})
    assert md.startswith("# Data-only report: Apple Inc. (AAPL)")
    assert "LLM call limit reached" in md
    assert "| m2 | revenue | Q3 FY2026 | $109.42B |" in md
    assert "## Revenue\nrevenue: 6 quarters" in md


def test_review_notes_are_shown_when_checks_were_incomplete():
    state = {"report": report("Revenue was $109.42B [m2]."), "metrics": METRICS,
             "critique": Critique(passed=False, problems=["summary: '$5B' has no [id]."], fully_checked=False)}
    md = render(state)
    assert "## Review notes" in md and "not fully checked" in md and "'$5B' has no [id]" in md
    assert "| m2 | revenue |" in md  # cited numbers listed


def test_password_never_appears_in_database_errors():
    assert _safe_url("postgresql://agent:s3cret@db:5432/x") == "postgresql://agent:***@db:5432/x"
