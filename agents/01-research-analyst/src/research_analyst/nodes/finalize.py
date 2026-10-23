"""Finalize node: every run ends here, whatever happened. It sets the final status and stores the result.

    approved  - a human approved the report
    rejected  - a human rejected it
    partial   - no written report (writer failed, budget ran out), but verified data exists
    failed    - nothing useful (e.g. not a company, unknown ticker, SEC unreachable)

The database write is an UPDATE keyed by run_id, so if this node runs twice (crash, retry)
the result is the same.
"""

from langgraph.runtime import Runtime

from research_analyst.report import render
from research_analyst.state import AgentContext, AgentState, RunStatus


def final_status(state: AgentState) -> RunStatus:
    if state.get("decision") == "approved":
        return "approved"
    if state.get("decision") == "rejected":
        return "rejected"
    has_data = bool(state.get("metrics")) or any(f.status != "failed" for f in state.get("findings", []))
    return "partial" if has_data else "failed"


async def finalize(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    status = final_status(state)
    if ctx.runs is not None:
        plan = state.get("plan")
        await ctx.runs.finish(
            ctx.run_id,
            status=status,
            ticker=plan.ticker if plan else None,
            report_markdown=render(state) if status != "rejected" else "",
            spend=state.get("spend"),
        )
    return {"status": status}
