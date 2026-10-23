"""The v1 graph end to end: review pause, revise loop, critic loop, crash + resume, budget, final status.

Fake SEC and fake Gemini, and LangGraph's in-memory checkpointer (same behaviour as Postgres for
these purposes; Postgres itself is tested in test_storage.py). Free and offline.
"""

from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research_analyst.core.config import Settings
from research_analyst.core.llm import Llm, LlmBadOutput, Spend
from research_analyst.graph import advance, build_graph, route_after_planner, route_after_writer
from research_analyst.nodes.critic import Judgement, Verdict
from research_analyst.nodes.researcher import NewsDigest, NewsEvent
from research_analyst.state import AgentContext, Claim, Critique, PlanDraft, PlannedTask, Report
from research_analyst.storage import checkpoint_serde
from tests.test_nodes import FakeSec

SETTINGS = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io")

PLAN = PlanDraft(ticker="AAPL", quarters=2, tasks=[
    PlannedTask(topic="Revenue", question="How did revenue change?", kind="financials", metrics=["revenue"]),
    PlannedTask(topic="News", question="What happened?", kind="news"),
])
DIGEST = NewsDigest(overview="New CEO.", events=[
    NewsEvent(accession="0000320193-26-000018", category="leadership", headline="New CEO", summary="The CEO changed."),
])
ALL_SUPPORTED = Judgement(verdicts=[Verdict(claim=0, supported=True, reason="The notes say so.")])


def good_report(kwargs) -> Report:
    """A report citing a real metric correctly (ids and values taken from the validation context)."""
    mid, value = next(iter(kwargs["context"]["metric_values"].items()))  # m1 = revenue, $120 in the fake data
    return Report(title="Apple", summary="Revenue grew.", body_markdown=f"Revenue was ${value:g} [{mid}].",
                  claims=[Claim(text="Revenue grew", kind="numeric", metric_id=mid, stated_value=value),
                          Claim(text="Growth was strong", kind="qualitative")])


def report_with_uncited_figure(kwargs) -> Report:
    r = good_report(kwargs)
    return r.model_copy(update={"summary": "Revenue reached $5.00B this year."})  # no [id]: the critic must catch it


class SchemaLlm:
    """Answers by schema name (parallel nodes call it in no fixed order). A list is used in order; the last repeats."""

    def __init__(self, answers: dict, settings: Settings = SETTINGS):
        self.answers = {k: (list(v) if isinstance(v, list) else [v]) for k, v in answers.items()}
        self.calls: list[str] = []
        self.prompts: dict[str, list[str]] = {}
        self._real = Llm(settings, client=object())  # for spend() / set_baseline

    def spend(self) -> Spend:
        b = self._real.spend()
        return b.model_copy(update={"llm_calls": b.llm_calls + len(self.calls)})

    def set_baseline(self, s):
        self._real.set_baseline(s)

    async def structured(self, **kwargs):
        name = kwargs["schema"].__name__
        self.calls.append(name)
        self.prompts.setdefault(name, []).append(kwargs["prompt"])
        script = self.answers[name]
        answer = script.pop(0) if len(script) > 1 else script[0]
        if callable(answer) and not isinstance(answer, type):
            answer = answer(kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return kwargs["schema"].model_validate(answer.model_dump(), context=kwargs.get("context"))


class FakeRuns:
    def __init__(self):
        self.finished: dict[str, dict] = {}

    async def finish(self, run_id, **kwargs):
        self.finished[run_id] = kwargs


def answers(**overrides) -> dict:
    return {"PlanDraft": PLAN, "NewsDigest": DIGEST, "Report": good_report, "Judgement": ALL_SUPPORTED, **overrides}


def setup(llm, *, sec=None, settings=SETTINGS, runs=None):
    graph = build_graph(InMemorySaver(serde=checkpoint_serde()))
    context = AgentContext(settings=settings, sec=sec or FakeSec(), llm=llm, run_id="run-test", runs=runs)
    return graph, context


async def start(graph, context, request="Analyze Apple"):
    return await advance(graph, context.run_id, {"request": request}, context)


async def decide(graph, context, **decision):
    return await advance(graph, context.run_id, Command(resume=decision), context)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    class D(date):
        @classmethod
        def today(cls):
            return date(2026, 10, 20)
    monkeypatch.setattr("research_analyst.nodes.researcher.date", D)


# --- human review -----------------------------------------------------------------------------

async def test_run_pauses_for_review_then_approve_finishes():
    runs = FakeRuns()
    graph, ctx = setup(SchemaLlm(answers()), runs=runs)
    snap = await start(graph, ctx)
    assert snap.next == ("human_review",) and snap.interrupts      # paused, state saved
    assert snap.values["critique"].passed

    snap = await decide(graph, ctx, action="approve")
    assert snap.next == () and snap.values["status"] == "approved"
    assert runs.finished["run-test"]["status"] == "approved"
    assert runs.finished["run-test"]["report_markdown"].startswith("# Apple")


async def test_reject_stores_no_report():
    runs = FakeRuns()
    graph, ctx = setup(SchemaLlm(answers()), runs=runs)
    await start(graph, ctx)
    snap = await decide(graph, ctx, action="reject")
    assert snap.values["status"] == "rejected"
    assert runs.finished["run-test"]["report_markdown"] == ""


async def test_revise_sends_feedback_to_the_writer_and_pauses_again():
    llm = SchemaLlm(answers())
    graph, ctx = setup(llm)
    await start(graph, ctx)
    snap = await decide(graph, ctx, action="revise", feedback="Shorter summary please")
    assert snap.next == ("human_review",)                  # reviewed again after the rewrite
    assert snap.values["review_rounds"] == 1
    assert llm.calls.count("Report") == 2
    assert "Shorter summary please" in llm.prompts["Report"][-1]
    assert "This is a REVISION" in llm.prompts["Report"][-1]


async def test_revision_limit_asks_again_instead_of_looping():
    graph, ctx = setup(SchemaLlm(answers()), settings=SETTINGS.model_copy(update={"review_max_revisions": 0}))
    await start(graph, ctx)
    snap = await decide(graph, ctx, action="revise", feedback="again")
    assert snap.next == ("human_review",)
    assert "Revision limit reached" in snap.values["errors"][-1]


async def test_invalid_review_answer_asks_again():
    graph, ctx = setup(SchemaLlm(answers()))
    await start(graph, ctx)
    snap = await decide(graph, ctx, action="revise", feedback="   ")  # revise without feedback
    assert snap.next == ("human_review",)
    assert "Invalid review decision" in snap.values["errors"][-1]


# --- critic loop ---------------------------------------------------------------------------------

async def test_critic_sends_an_uncited_figure_back_and_the_rewrite_passes():
    llm = SchemaLlm(answers(Report=[report_with_uncited_figure, good_report]))
    graph, ctx = setup(llm)
    snap = await start(graph, ctx)
    assert llm.calls.count("Report") == 2
    assert snap.values["critique"].passed and snap.values["revisions"] == 1
    assert "'$5.00B' has no [id]" in llm.prompts["Report"][1]   # the writer got the exact problem


async def test_critic_gives_up_after_max_revisions_and_shows_the_human():
    llm = SchemaLlm(answers(Report=report_with_uncited_figure))  # never fixed
    graph, ctx = setup(llm)
    snap = await start(graph, ctx)
    assert llm.calls.count("Report") == 1 + SETTINGS.critic_max_revisions  # bounded, not forever
    assert snap.next == ("human_review",)
    critique: Critique = snap.values["critique"]
    assert not critique.passed and not critique.send_back


async def test_unsupported_judgment_is_sent_back():
    unsupported = Judgement(verdicts=[Verdict(claim=0, supported=False, reason="Notes don't say this.")])
    llm = SchemaLlm(answers(Judgement=[unsupported, ALL_SUPPORTED]))
    graph, ctx = setup(llm)
    snap = await start(graph, ctx)
    assert llm.calls.count("Report") == 2
    assert snap.values["critique"].passed


# --- crashes and resume ------------------------------------------------------------------------------

async def test_crash_in_writer_resumes_without_redoing_earlier_steps():
    sec = FakeSec()
    calls = {"facts": 0}
    real_facts = sec.company_facts

    async def counting_facts(cik):
        calls["facts"] += 1
        return await real_facts(cik)

    sec.company_facts = counting_facts
    llm = SchemaLlm(answers(Report=[RuntimeError("worker killed"), good_report]))
    graph, ctx = setup(llm, sec=sec)

    with pytest.raises(RuntimeError, match="worker killed"):
        await start(graph, ctx)
    saved = await graph.aget_state({"configurable": {"thread_id": "run-test"}})
    assert saved.next == ("writer",)                    # the last finished step was saved

    snap = await advance(graph, "run-test", None, ctx)  # resume: no new input
    assert snap.next == ("human_review",)
    assert llm.calls.count("PlanDraft") == 1 and llm.calls.count("NewsDigest") == 1  # not paid for again
    assert calls["facts"] == 1


async def test_crash_of_one_parallel_researcher_reruns_only_that_one():
    sec = FakeSec()
    calls = {"facts": 0}
    real_facts = sec.company_facts

    async def counting_facts(cik):
        calls["facts"] += 1
        return await real_facts(cik)

    sec.company_facts = counting_facts
    llm = SchemaLlm(answers(NewsDigest=[RuntimeError("worker killed"), DIGEST]))
    graph, ctx = setup(llm, sec=sec)
    with pytest.raises(RuntimeError):
        await start(graph, ctx)

    snap = await advance(graph, "run-test", None, ctx)
    assert snap.next == ("human_review",)
    assert calls["facts"] == 1  # the financials researcher finished before the crash: its result was kept
    assert sorted(f.task_id for f in snap.values["findings"]) == ["t1", "t2"]


async def test_state_survives_strict_checkpoint_serialization(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")  # any class missing from CHECKPOINT_TYPES would fail
    graph, ctx = setup(SchemaLlm(answers()))
    await start(graph, ctx)
    values = (await graph.aget_state({"configurable": {"thread_id": "run-test"}})).values
    assert isinstance(values["critique"], Critique)
    assert isinstance(values["spend"], Spend)


# --- budget and final status ------------------------------------------------------------------------

async def test_writer_out_of_budget_gives_a_partial_data_only_result():
    from research_analyst.core.llm import BudgetExceeded
    runs = FakeRuns()
    graph, ctx = setup(SchemaLlm(answers(Report=BudgetExceeded("LLM call limit reached (2 of 2)"))), runs=runs)
    snap = await start(graph, ctx)
    assert snap.next == () and snap.values["status"] == "partial"
    assert "report" not in snap.values
    assert runs.finished["run-test"]["report_markdown"].startswith("# Data-only report: Apple Inc. (AAPL)")


async def test_not_a_company_ends_as_failed():
    runs = FakeRuns()
    llm = SchemaLlm(answers(PlanDraft=PlanDraft(ticker="NONE", quarters=4, tasks=PLAN.tasks)))
    graph, ctx = setup(llm, runs=runs)
    snap = await start(graph, ctx, "What's the weather?")
    assert snap.values["status"] == "failed"
    assert llm.calls == ["PlanDraft"]
    assert runs.finished["run-test"]["status"] == "failed"


async def test_spend_is_recorded_in_the_state():
    llm = SchemaLlm(answers())
    graph, ctx = setup(llm)
    snap = await start(graph, ctx)
    assert snap.values["spend"].llm_calls == len(llm.calls)  # PlanDraft, NewsDigest, Report, Judgement


async def test_one_failed_researcher_does_not_stop_the_run():
    graph, ctx = setup(SchemaLlm(answers(NewsDigest=LlmBadOutput("bad digest"))))
    snap = await start(graph, ctx)
    assert {f.task_id: f.status for f in snap.values["findings"]} == {"t1": "ok", "t2": "failed"}
    assert snap.next == ("human_review",)


# --- routing and shape ----------------------------------------------------------------------------------

def test_routes():
    from research_analyst.state import Plan, ResearchTask
    tasks = [ResearchTask(id=f"t{i}", **t.model_dump()) for i, t in enumerate(PLAN.tasks, 1)]
    sends = route_after_planner({"plan": Plan(ticker="AAPL", company_name="Apple Inc.", cik=1, quarters=2, tasks=tasks)})
    assert [(s.node, s.arg["task"].id) for s in sends] == [("researcher", "t1"), ("researcher", "t2")]
    assert route_after_planner({}) == "finalize"
    assert route_after_writer({}) == "finalize"
    assert route_after_writer({"report": Report(title="t", summary="s", body_markdown="b")}) == "critic"


def test_graph_shape():
    nodes = set(build_graph().get_graph().nodes)
    assert {"planner", "researcher", "calculator", "writer", "critic", "human_review", "finalize"} <= nodes
