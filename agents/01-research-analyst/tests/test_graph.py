"""The whole agent as a LangGraph graph, and the CLI on top of it. Fake SEC and fake Gemini: free and offline."""

from datetime import date

import pytest

from research_analyst import cli
from research_analyst.core.config import Settings
from research_analyst.core.llm import LlmBadOutput
from research_analyst.graph import build_graph, route_after_planner, run_analysis
from research_analyst.nodes.researcher import NewsDigest, NewsEvent
from research_analyst.state import AgentContext, Claim, Finding, PlanDraft, PlannedTask, Report, Source
from research_analyst.tools.edgar import SecAccessDenied
from tests.test_nodes import FakeSec

SETTINGS = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io")

PLAN = PlanDraft(ticker="AAPL", quarters=2, tasks=[
    PlannedTask(topic="Revenue", question="How did revenue change?", kind="financials", metrics=["revenue"]),
    PlannedTask(topic="News", question="What happened?", kind="news"),
])
DIGEST = NewsDigest(overview="New CEO.", events=[
    NewsEvent(accession="0000320193-26-000018", category="leadership", headline="New CEO", summary="The CEO changed."),
])


class SchemaLlm:
    """Answers by schema, because parallel researchers call it in no fixed order."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls: list[str] = []

    async def structured(self, **kwargs):
        schema = kwargs["schema"]
        self.calls.append(schema.__name__)
        answer = self.answers[schema.__name__]
        if callable(answer) and not isinstance(answer, type):
            answer = answer(kwargs)
        if isinstance(answer, Exception):
            raise answer
        return schema.model_validate(answer.model_dump(), context=kwargs.get("context"))


def report_citing_first_metric(kwargs) -> Report:
    """A report whose one numeric claim cites a real id with the right value (taken from the validation context)."""
    mid, value = next(iter(kwargs["context"]["metric_values"].items()))
    return Report(title="Apple", summary="Revenue grew.", body_markdown=f"Revenue grew [{mid}].",
                  claims=[Claim(text="Revenue grew", kind="numeric", metric_id=mid, stated_value=value)])


def context(llm, sec=None) -> AgentContext:
    return AgentContext(settings=SETTINGS, sec=sec or FakeSec(), llm=llm)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    class D(date):
        @classmethod
        def today(cls):
            return date(2026, 10, 20)
    monkeypatch.setattr("research_analyst.nodes.researcher.date", D)


# --- the graph -------------------------------------------------------------------------------

async def test_full_run_plan_research_calculate_write():
    llm = SchemaLlm({"PlanDraft": PLAN, "NewsDigest": DIGEST, "Report": report_citing_first_metric})
    progress: list[str] = []
    state = await run_analysis("Analyze Apple", context(llm), on_progress=progress.append)

    assert state["plan"].ticker == "AAPL"
    assert sorted(f.task_id for f in state["findings"]) == ["t1", "t2"]  # both parallel researchers merged
    assert state["metrics"]                                               # calculator ran after both
    assert state["report"].title == "Apple"
    assert sorted(llm.calls) == ["NewsDigest", "PlanDraft", "Report"]     # one LLM call per LLM step
    assert progress[0].startswith("Plan: Apple Inc. (AAPL)")
    assert progress[-1] == "Report written"


async def test_no_plan_stops_before_research():
    llm = SchemaLlm({"PlanDraft": PlanDraft(ticker="NONE", quarters=4, tasks=PLAN.tasks)})
    state = await run_analysis("What's the weather?", context(llm))
    assert "plan" not in state and "report" not in state
    assert not state.get("findings")  # keys with a reducer show up as [] even when nothing wrote to them
    assert "one US-listed public company" in state["errors"][0]
    assert llm.calls == ["PlanDraft"]  # nothing else was paid for


async def test_writer_failure_keeps_everything_else():
    llm = SchemaLlm({"PlanDraft": PLAN, "NewsDigest": DIGEST, "Report": LlmBadOutput("bad report")})
    state = await run_analysis("Analyze Apple", context(llm))
    assert "report" not in state
    assert state["metrics"] and len(state["findings"]) == 2
    assert state["errors"][-1].startswith("Writing the report failed")


async def test_one_failed_researcher_does_not_stop_the_run():
    llm = SchemaLlm({"PlanDraft": PLAN, "NewsDigest": LlmBadOutput("bad digest"), "Report": report_citing_first_metric})
    state = await run_analysis("Analyze Apple", context(llm))
    status = {f.task_id: f.status for f in state["findings"]}
    assert status == {"t1": "ok", "t2": "failed"}
    assert "report" in state


async def test_sec_access_denied_stops_the_whole_run():
    llm = SchemaLlm({"PlanDraft": PLAN, "NewsDigest": DIGEST, "Report": report_citing_first_metric})
    with pytest.raises(SecAccessDenied):
        await run_analysis("Analyze Apple", context(llm, sec=FakeSec(facts=SecAccessDenied("403"))))


def test_router_sends_one_researcher_per_task():
    from research_analyst.state import Plan, ResearchTask
    tasks = [ResearchTask(id=f"t{i}", **t.model_dump()) for i, t in enumerate(PLAN.tasks, 1)]
    sends = route_after_planner({"plan": Plan(ticker="AAPL", company_name="Apple Inc.", cik=1, quarters=2, tasks=tasks)})
    assert [(s.node, s.arg["task"].id, s.arg["quarters"]) for s in sends] == [("researcher", "t1", 2), ("researcher", "t2", 2)]
    assert route_after_planner({}) == "__end__"


def test_graph_shape():
    nodes = set(build_graph().get_graph().nodes)
    assert {"planner", "researcher", "calculator", "writer"} <= nodes


async def test_recursion_limit_comes_from_settings(monkeypatch):
    seen = {}
    real = build_graph()

    class Spy:
        def astream(self, *args, **kwargs):
            seen.update(kwargs["config"])
            return real.astream(*args, **kwargs)

    monkeypatch.setattr("research_analyst.graph.build_graph", lambda: Spy())
    llm = SchemaLlm({"PlanDraft": PlanDraft(ticker="NONE", quarters=4, tasks=PLAN.tasks)})
    await run_analysis("x", context(llm))
    assert seen["recursion_limit"] == SETTINGS.graph_recursion_limit


# --- the CLI -----------------------------------------------------------------------------------

def test_render_report_lists_cited_numbers_and_sources():
    from research_analyst.finance.metrics import MetricResult
    metric = MetricResult(name="revenue_yoy_growth", period_label="Q3 FY2026", period_end=date(2026, 6, 27),
                          value=0.164, unit="ratio", formula="(109.42B - 94.04B) / 94.04B", inputs=())
    unused = metric.model_copy(update={"name": "net_margin"})
    state = {
        "report": Report(title="Apple", summary="Strong quarter.", body_markdown="Revenue grew +16.4% [m1].",
                         claims=[Claim(text="x", kind="numeric", metric_id="m1", stated_value=16.4)]),
        "metrics": {"m1": metric, "m2": unused},
        "findings": [Finding(task_id="t1", topic="News", status="ok", summary="s", sources=[
            Source(kind="8-K", accession="0000320193-26-000018", filed=date(2026, 7, 30), url="https://sec.gov/x", description="New CEO"),
        ])],
    }
    md = cli.render_report(state)
    assert md.startswith("# Apple\n\nStrong quarter.")
    assert "| m1 | revenue_yoy_growth | Q3 FY2026 | +16.4% | (109.42B - 94.04B) / 94.04B |" in md
    assert "net_margin" not in md  # only numbers the report actually cites
    assert "- 2026-07-30 8-K: New CEO (0000320193-26-000018) https://sec.gov/x" in md
    assert "Not investment advice" in md


def test_cli_prints_report_and_saves_it(monkeypatch, tmp_path, capsys):
    state = {"report": Report(title="Apple", summary="s", body_markdown="b"), "errors": ["minor problem"]}

    async def fake_run(request, settings):
        from research_analyst.core.llm import Llm
        return state, Llm(settings, client=object())

    monkeypatch.setattr(cli, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(cli, "_run", fake_run)
    out_file = tmp_path / "apple.md"
    assert cli.main(["Analyze Apple", "-o", str(out_file)]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("# Apple")
    assert "! minor problem" in captured.err      # problems go to stderr, the report to stdout
    assert out_file.read_text().startswith("# Apple")


def test_cli_exit_code_1_without_report(monkeypatch, capsys):
    async def fake_run(request, settings):
        from research_analyst.core.llm import Llm
        return {"errors": ["I can analyze one US-listed public company per request."]}, Llm(settings, client=object())

    monkeypatch.setattr(cli, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(cli, "_run", fake_run)
    assert cli.main(["weather?"]) == 1
    assert "one US-listed" in capsys.readouterr().err


def test_cli_config_error_exit_code_2(monkeypatch, capsys):
    def broken():
        return Settings(_env_file=None, gcp_project="p", sec_user_agent="no-email")

    monkeypatch.setattr(cli, "get_settings", broken)
    assert cli.main(["Analyze Apple"]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_cli_sec_block_is_a_clear_message(monkeypatch, capsys):
    async def fake_run(request, settings):
        raise SecAccessDenied("SEC refused access (403)")

    monkeypatch.setattr(cli, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(cli, "_run", fake_run)
    assert cli.main(["Analyze Apple"]) == 1
    assert "SEC refused access" in capsys.readouterr().err
