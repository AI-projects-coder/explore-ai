"""Researcher, calculator and writer nodes, with fake SEC and LLM: every rule, no real API calls."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from research_analyst.core.config import Settings
from research_analyst.core.llm import LlmBadOutput, LlmTruncated
from research_analyst.finance.metrics import MetricResult
from research_analyst.nodes.calculator import calculator, display
from research_analyst.nodes.researcher import NewsDigest, NewsEvent, researcher
from research_analyst.nodes.writer import writer
from research_analyst.state import (
    AgentContext,
    Claim,
    Finding,
    Plan,
    Report,
    ResearcherInput,
    ResearchTask,
)
from research_analyst.tools.edgar import Company, SecAccessDenied, SecUnavailable
from research_analyst.tools.xbrl import QuarterValue

SETTINGS = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io")
APPLE = Company(cik=320193, ticker="AAPL", name="Apple Inc.")
REV = "RevenueFromContractWithCustomerExcludingAssessedTax"


class Runtime:
    def __init__(self, context):
        self.context = context


class FakeLlm:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []

    async def structured(self, **kwargs):
        self.calls.append(kwargs)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        # validate like the real wrapper does, with the same context
        return kwargs["schema"].model_validate(a.model_dump(), context=kwargs.get("context"))


def quarterly_facts(values_by_end: dict[str, float]) -> dict:
    rows = [{"start": (date.fromisoformat(e) - timedelta(days=90)).isoformat(), "end": e, "val": v,
             "fy": int(e[:4]), "fp": "Q1", "form": "10-Q", "filed": e, "accn": f"0000320193-{e[2:4]}-0000{i:02d}"}
            for i, (e, v) in enumerate(values_by_end.items())]
    return {"facts": {"us-gaap": {REV: {"units": {"USD": rows}}}}}


FACTS = quarterly_facts({"2025-03-31": 100, "2025-06-30": 110, "2025-09-30": 120, "2025-12-31": 130,
                         "2026-03-31": 120, "2026-06-30": 132})
SUBS = {"filings": {"recent": {
    "form": ["8-K", "8-K"], "accessionNumber": ["0000320193-26-000018", "0000320193-26-000011"],
    "filingDate": ["2026-07-30", "2026-04-30"], "items": ["5.02", "2.02,9.01"], "primaryDocument": ["a.htm", "b.htm"],
}}}


class FakeSec:
    def __init__(self, facts=FACTS, subs=SUBS, docs_fail: dict | None = None):
        self.facts, self.subs, self.docs_fail = facts, subs, docs_fail or {}

    async def resolve_ticker(self, ticker):
        return APPLE

    async def company_facts(self, cik):
        if isinstance(self.facts, Exception):
            raise self.facts
        return self.facts

    async def submissions(self, cik):
        return self.subs

    async def filing_index(self, cik, accession):
        if accession in self.docs_fail:
            raise self.docs_fail[accession]
        return ["a.htm", "b.htm"]

    async def filing_document(self, cik, accession, name):
        return f"<p>Item 5.02 text of {accession}</p>"


def ctx(sec=None, llm=None) -> Runtime:
    return Runtime(AgentContext(settings=SETTINGS, sec=sec or FakeSec(), llm=llm or FakeLlm()))


def fin_task(metrics=("revenue",), tid="t1") -> ResearchTask:
    return ResearchTask(id=tid, topic="Revenue", question="q", kind="financials", metrics=list(metrics))


def news_task(tid="t2") -> ResearchTask:
    return ResearchTask(id=tid, topic="News", question="What happened?", kind="news")


def inp(task, quarters=2) -> ResearcherInput:
    return ResearcherInput(task=task, ticker="AAPL", quarters=quarters)


# --- researcher: financials (no LLM) ----------------------------------------------------------

async def test_financials_fetches_quarters_plus_a_year_for_yoy():
    llm = FakeLlm()
    out = await researcher(inp(fin_task(), quarters=2), ctx(llm=llm))
    f = out["findings"][0]
    assert (f.task_id, f.status) == ("t1", "ok")
    assert len(f.data) == 6           # 2 requested + 4 extra, capped by what exists
    assert "revenue: 6 quarters" in f.summary
    assert f.sources and all(s.kind == "xbrl" for s in f.sources)
    assert llm.calls == []            # no LLM for numbers


async def test_financials_partial_when_some_metrics_are_missing():
    f = (await researcher(inp(fin_task(["revenue", "gross_profit"])), ctx()))["findings"][0]
    assert f.status == "partial"
    assert "gross_profit" in f.error


async def test_financials_failed_when_nothing_is_available():
    f = (await researcher(inp(fin_task(["gross_profit"])), ctx()))["findings"][0]
    assert (f.status, f.data) == ("failed", [])


async def test_sec_outage_becomes_a_failed_finding():
    f = (await researcher(inp(fin_task()), ctx(sec=FakeSec(facts=SecUnavailable("SEC down")))))["findings"][0]
    assert (f.status, f.error) == ("failed", "SEC down")


async def test_sec_access_denied_stops_the_run():
    with pytest.raises(SecAccessDenied):
        await researcher(inp(fin_task()), ctx(sec=FakeSec(facts=SecAccessDenied("403"))))


# --- researcher: news (code fetches, LLM reads) -----------------------------------------------------

def digest(*accessions) -> NewsDigest:
    return NewsDigest(overview="A new CEO.", events=[
        NewsEvent(accession=a, category="leadership", headline="New CEO", summary="The CEO changed.") for a in accessions
    ])


async def test_news_reads_filings_and_cites_them(monkeypatch):
    monkeypatch.setattr("research_analyst.nodes.researcher.date", _FixedDate)
    llm = FakeLlm(digest("0000320193-26-000018"))
    f = (await researcher(inp(news_task(), quarters=4), ctx(llm=llm)))["findings"][0]
    assert f.status == "ok"
    assert "New CEO" in f.summary and "0000320193-26-000018" in f.summary
    assert [s.accession for s in f.sources] == ["0000320193-26-000018"]
    call = llm.calls[0]
    assert call["model"] == SETTINGS.researcher_model
    assert '<filing accession="0000320193-26-000018"' in call["prompt"]
    assert call["context"] == {"accessions": {"0000320193-26-000018", "0000320193-26-000011"}}


async def test_news_event_citing_an_unknown_filing_is_rejected():
    # the validator (with context) catches an invented accession, so the real wrapper sends it back for repair
    with pytest.raises(ValidationError, match="not one of the filings"):
        NewsDigest.model_validate(digest("9999999999-99-999999").model_dump(), context={"accessions": {"a"}})


async def test_news_without_filings_is_ok_and_skips_the_llm():
    llm = FakeLlm()
    sec = FakeSec(subs={"filings": {"recent": {"form": [], "accessionNumber": [], "filingDate": [], "items": [], "primaryDocument": []}}})
    f = (await researcher(inp(news_task()), ctx(sec=sec, llm=llm)))["findings"][0]
    assert f.status == "ok" and "No 8-K filings" in f.summary
    assert llm.calls == []


async def test_news_partial_when_a_filing_cannot_be_read(monkeypatch):
    monkeypatch.setattr("research_analyst.nodes.researcher.date", _FixedDate)
    sec = FakeSec(docs_fail={"0000320193-26-000011": SecUnavailable("timeout")})
    f = (await researcher(inp(news_task(), quarters=4), ctx(sec=sec, llm=FakeLlm(digest("0000320193-26-000018")))))["findings"][0]
    assert f.status == "partial"
    assert "0000320193-26-000011" in f.error


async def test_news_llm_failure_becomes_a_failed_finding(monkeypatch):
    monkeypatch.setattr("research_analyst.nodes.researcher.date", _FixedDate)
    f = (await researcher(inp(news_task(), quarters=4), ctx(llm=FakeLlm(LlmBadOutput("bad")))))["findings"][0]
    assert f.status == "failed"


class _FixedDate(date):
    @classmethod
    def today(cls):
        return date(2026, 10, 20)


# --- calculator (no LLM) ----------------------------------------------------------------------------

def plan(quarters=2, kinds=("financials",)) -> Plan:
    tasks = [fin_task(tid=f"t{i}") if k == "financials" else news_task(f"t{i}") for i, k in enumerate(kinds, 1)]
    return Plan(ticker="AAPL", company_name="Apple Inc.", cik=320193, quarters=quarters, tasks=tasks)


async def findings_for(metrics=("revenue",)) -> list[Finding]:
    return (await researcher(inp(fin_task(metrics), quarters=2), ctx()))["findings"]


async def test_calculator_builds_a_table_for_the_requested_window():
    out = calculator({"plan": plan(quarters=2), "findings": await findings_for()})
    by_name = {}
    for r in out["metrics"].values():
        by_name.setdefault(r.name, []).append(r)
    assert [r.period_label for r in by_name["revenue"]] == ["Q1 FY2026", "Q1 FY2026"]  # last 2 quarters only
    assert [round(r.value, 3) for r in by_name["revenue_yoy_growth"]] == [0.2, 0.2]  # 120 vs 100, 132 vs 110
    assert "revenue_ttm" in by_name
    assert all(mid.startswith("m") for mid in out["metrics"])


async def test_calculator_ids_are_stable_across_reruns():
    state = {"plan": plan(), "findings": await findings_for()}
    assert calculator(state)["metrics"] == calculator(state)["metrics"]


def test_calculator_without_data_reports_why():
    out = calculator({"plan": plan(), "findings": []})
    assert out["metrics"] == {}
    assert "No financial data" in out["errors"][0]


def test_calculator_news_only_plan_has_no_error():
    assert calculator({"plan": plan(kinds=("news",)), "findings": []}) == {"metrics": {}}


def test_calculator_without_plan_does_nothing():
    assert calculator({}) == {}


@pytest.mark.parametrize("unit,name,value,expected", [
    ("ratio", "revenue_yoy_growth", 0.16359, ("+16.4%", 16.4)),
    ("ratio", "revenue_yoy_growth", -0.05, ("-5.0%", -5.0)),
    ("ratio", "net_margin", 0.272, ("27.2%", 27.2)),
    ("USD", "revenue", 109.417e9, ("$109.42B", 109.42)),
    ("USD", "free_cash_flow", -2.5e9, ("-$2.50B", -2.5)),
    ("USD", "capital_expenditure", 512.3e6, ("$512.30M", 512.3)),
    ("USD/shares", "eps_diluted", 2.0249, ("$2.02", 2.02)),
])
def test_display(unit, name, value, expected):
    r = MetricResult(name=name, period_label="Q", period_end=date(2026, 6, 30), value=value, unit=unit, formula="f", inputs=())
    assert display(r) == expected


# --- report validator: numeric claims must match the table -----------------------------------------

VALUES = {"m1": 16.4, "m2": 109.42}


def report(*claims) -> dict:
    return {"title": "t", "summary": "s", "body_markdown": "b", "claims": [c.model_dump() for c in claims]}


def test_claims_matching_the_table_pass():
    Report.model_validate(report(Claim(text="+16.4% [m1]", kind="numeric", metric_id="m1", stated_value=16.4),
                                 Claim(text="strong", kind="qualitative")), context={"metric_values": VALUES})


@pytest.mark.parametrize("claim,problem", [
    (Claim(text="x", kind="numeric", metric_id="m9", stated_value=1.0), "not in the metrics table"),
    (Claim(text="x", kind="numeric", metric_id="m1", stated_value=18.2), "is 16.4"),       # changed number
    (Claim(text="x", kind="numeric", metric_id="m1"), "stated_value None"),                 # no value
    (Claim(text="x", kind="numeric", stated_value=16.4), "not in the metrics table"),       # no id
])
def test_bad_numeric_claims_are_rejected(claim, problem):
    with pytest.raises(ValidationError, match=problem):
        Report.model_validate(report(claim), context={"metric_values": VALUES})


def test_rounding_tolerance():
    Report.model_validate(report(Claim(text="x", kind="numeric", metric_id="m2", stated_value=109.4)),
                          context={"metric_values": VALUES})


def test_without_context_only_structure_is_checked():
    Report.model_validate(report(Claim(text="x", kind="numeric", metric_id="m9", stated_value=1.0)))


# --- writer -------------------------------------------------------------------------------------------

async def test_writer_sends_the_table_and_validates_with_it():
    state = {"request": "Analyze Apple", "plan": plan(), "findings": await findings_for()}
    state["metrics"] = calculator(state)["metrics"]
    some_id, some = next(iter(state["metrics"].items()))
    good = Report(title="t", summary="s", body_markdown="b",
                  claims=[Claim(text="x", kind="numeric", metric_id=some_id, stated_value=display(some)[1])])
    llm = FakeLlm(good)
    out = await writer(state, ctx(llm=llm))
    assert out["report"].claims[0].metric_id == some_id
    call = llm.calls[0]
    assert call["model"] == SETTINGS.writer_model
    assert call["max_output_tokens"] == SETTINGS.writer_max_output_tokens  # real-run lesson: 8192 truncated the report
    assert f"{some_id} | {some.name}" in call["prompt"]
    assert call["context"]["metric_values"][some_id] == display(some)[1]


async def test_writer_mentions_problems_and_failed_research():
    failed = Finding(task_id="t2", topic="News", status="failed", summary="", error="SEC down")
    state = {"request": "r", "plan": plan(), "findings": [failed], "metrics": {}, "errors": ["something broke"]}
    llm = FakeLlm(Report(title="t", summary="s", body_markdown="b"))
    await writer(state, ctx(llm=llm))
    prompt = llm.calls[0]["prompt"]
    assert "status: failed" in prompt and "SEC down" in prompt and "something broke" in prompt
    assert "(no numbers available)" in prompt


async def test_writer_failure_becomes_an_error():
    state = {"request": "r", "plan": plan(), "findings": [], "metrics": {}}
    out = await writer(state, ctx(llm=FakeLlm(LlmTruncated("cut off"))))
    assert "report" not in out
    assert out["errors"][0].startswith("Writing the report failed")


async def test_writer_without_plan_does_nothing():
    assert await writer({}, ctx()) == {}
