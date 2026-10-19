"""Builds the LangGraph graph: nodes, edges and the researcher fan-out.

    START -> planner --(no plan)----------> END
                     --(one Send per task)--> researcher (parallel) -> calculator -> writer -> END
"""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from research_analyst.nodes.calculator import calculator
from research_analyst.nodes.planner import planner
from research_analyst.nodes.researcher import researcher
from research_analyst.nodes.writer import writer
from research_analyst.state import AgentContext, AgentState, ResearcherInput


def route_after_planner(state: AgentState) -> list[Send] | str:
    """No plan (the planner reported an error) -> stop. Otherwise start one researcher per task."""
    plan = state.get("plan")
    if plan is None:
        return END
    return [Send("researcher", ResearcherInput(task=t, ticker=plan.ticker, quarters=plan.quarters)) for t in plan.tasks]


def build_graph(checkpointer=None) -> CompiledStateGraph:
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("calculator", calculator)
    g.add_node("writer", writer)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_after_planner, ["researcher", END])
    g.add_edge("researcher", "calculator")  # waits for ALL parallel researchers, then runs once
    g.add_edge("calculator", "writer")
    g.add_edge("writer", END)
    return g.compile(checkpointer=checkpointer)


async def run_analysis(request: str, context: AgentContext,
                       on_progress: Callable[[str], None] | None = None) -> AgentState:
    """Run the whole agent once. Returns the final state; reports each finished step to on_progress."""
    graph = build_graph()
    final: AgentState = {}
    async for mode, chunk in graph.astream(
        {"request": request},
        config={"recursion_limit": context.settings.graph_recursion_limit},
        context=context,
        stream_mode=["updates", "values"],
    ):
        if mode == "values":
            final = chunk  # the full state after each step; the last one is the result
        elif on_progress:
            for node, update in chunk.items():
                message = _describe(node, update or {})
                if message:
                    on_progress(message)
    return final


def _describe(node: str, update: dict) -> str | None:
    """One human-readable line per finished step."""
    if node == "planner" and "plan" in update:
        p = update["plan"]
        topics = ", ".join(t.topic for t in p.tasks)
        return f"Plan: {p.company_name} ({p.ticker}), last {p.quarters} quarters. Researching: {topics}"
    if node == "researcher":
        return "\n".join(f"  {'✓' if f.status == 'ok' else '!'} {f.topic}: {f.status}" for f in update.get("findings", []))
    if node == "calculator":
        return f"Calculated {len(update.get('metrics', {}))} numbers"
    if node == "writer" and "report" in update:
        return "Report written"
    return None
