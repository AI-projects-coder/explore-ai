"""Builds the LangGraph graph: nodes, edges and the researcher fan-out.

    START -> planner --(no plan)-------------------------------------------------> finalize -> END
                     --(one Send per task)--> researcher (parallel) -> calculator -> writer
    writer --(no report)--> finalize
           --(report)-----> critic --(problems, tries left)--> writer
                                   --(ok, or no tries left)--> human_review (run pauses: interrupt)
    human_review --approve/reject--> finalize      --revise + feedback--> writer

With a checkpointer, the state is saved after every step (durability="sync"), so a run can resume
after a crash or after days of waiting for review, using the same run id (LangGraph's thread_id).
"""

import functools
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send, StateSnapshot

from research_analyst.nodes.calculator import calculator
from research_analyst.nodes.critic import critic
from research_analyst.nodes.finalize import finalize
from research_analyst.nodes.planner import planner
from research_analyst.nodes.researcher import researcher
from research_analyst.nodes.review import human_review
from research_analyst.nodes.writer import writer
from research_analyst.state import AgentContext, AgentState, ResearcherInput


def route_after_planner(state: AgentState) -> list[Send] | str:
    """No plan (the planner reported an error) -> finalize records the failure. Otherwise one researcher per task."""
    plan = state.get("plan")
    if plan is None:
        return "finalize"
    return [Send("researcher", ResearcherInput(task=t, ticker=plan.ticker, quarters=plan.quarters)) for t in plan.tasks]


def route_after_writer(state: AgentState) -> str:
    """No report (writer failed, budget used up) -> finalize renders a data-only report."""
    return "critic" if state.get("report") is not None else "finalize"


def tracks_spend(node: Callable) -> Callable:
    """Wrap a node that calls the LLM so it also reports the run's spend into the state.

    The spend lives in the state (not only in memory), so after a crash or a review pause the
    budget continues from where it was instead of starting again at zero.
    """
    @functools.wraps(node)
    async def wrapper(state, runtime):
        result = await node(state, runtime)
        spend = runtime.context.llm.spend()
        if isinstance(result, Command):
            return Command(update={**(result.update or {}), "spend": spend}, goto=result.goto)
        return {**(result or {}), "spend": spend}
    return wrapper


def build_graph(checkpointer=None) -> CompiledStateGraph:
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("planner", tracks_spend(planner))
    g.add_node("researcher", tracks_spend(researcher))
    g.add_node("calculator", calculator)
    g.add_node("writer", tracks_spend(writer))
    g.add_node("critic", tracks_spend(critic))  # routes itself with Command: writer or human_review
    g.add_node("human_review", human_review)  # routes itself with Command: writer, finalize or itself
    g.add_node("finalize", finalize)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_after_planner, ["researcher", "finalize"])
    g.add_edge("researcher", "calculator")  # waits for ALL parallel researchers, then runs once
    g.add_edge("calculator", "writer")
    g.add_conditional_edges("writer", route_after_writer, ["critic", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def run_config(run_id: str, context: AgentContext) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": context.settings.graph_recursion_limit}


async def advance(graph: CompiledStateGraph, run_id: str, payload: Any, context: AgentContext,
                  on_progress: Callable[[str], None] | None = None) -> StateSnapshot:
    """Run until the graph finishes or pauses for review. Returns the saved state afterwards.

    payload: {"request": ...} to start, None to continue after a crash, Command(resume=...) after review.
    """
    config = run_config(run_id, context)
    async for chunk in graph.astream(payload, config=config, context=context,
                                     stream_mode="updates", durability="sync"):
        if on_progress:
            for node, update in chunk.items():
                message = _describe(node, update)
                if message:
                    on_progress(message)
    return await graph.aget_state(config)


def _describe(node: str, update: Any) -> str | None:
    """One human-readable line per finished step."""
    if not isinstance(update, dict):
        return None
    if node == "planner" and "plan" in update:
        p = update["plan"]
        topics = ", ".join(t.topic for t in p.tasks)
        return f"Plan: {p.company_name} ({p.ticker}), last {p.quarters} quarters. Researching: {topics}"
    if node == "researcher":
        return "\n".join(f"  {'✓' if f.status == 'ok' else '!'} {f.topic}: {f.status}" for f in update.get("findings", []))
    if node == "calculator":
        return f"Calculated {len(update.get('metrics', {}))} numbers"
    if node == "writer":
        return "Report written" if "report" in update else None
    if node == "critic" and update.get("critique"):
        c = update["critique"]
        if c.passed:
            return "Critic: all checks passed" + ("" if c.fully_checked else " (judgments not fully checked)")
        return f"Critic: {len(c.problems)} problem(s), " + ("sending back to the writer" if c.send_back else "no tries left")
    if node == "finalize":
        return f"Finished: {update.get('status')}"
    return None
