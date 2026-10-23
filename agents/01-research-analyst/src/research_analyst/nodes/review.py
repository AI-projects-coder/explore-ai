"""Human review node: the run pauses here until a person approves, asks for a revision, or rejects.

interrupt() saves the state (via the checkpointer) and stops the run. The process may exit; the run
waits in Postgres. Resuming with Command(resume={...}) makes interrupt() return that decision.

IMPORTANT: on resume LangGraph re-runs this node FROM ITS FIRST LINE, not from the interrupt() line.
So nothing before interrupt() may have side effects (no saving, no emails). Here, interrupt() is
the first thing the node does.
"""

from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from research_analyst.state import AgentContext, AgentState, ReviewDecision


def human_review(state: AgentState, runtime: Runtime[AgentContext]) -> Command[Literal["writer", "finalize", "human_review"]]:
    answer = interrupt({"review_round": state.get("review_rounds", 0)})  # the run pauses here

    try:
        decision = ReviewDecision.model_validate(answer)
    except ValidationError as e:
        # a malformed resume value must not crash a paused run: ask again
        return Command(update={"errors": [f"Invalid review decision: {e.errors()[0]['msg']}"]}, goto="human_review")

    if decision.action == "approve":
        return Command(update={"decision": "approved"}, goto="finalize")
    if decision.action == "reject":
        return Command(update={"decision": "rejected"}, goto="finalize")

    rounds, limit = state.get("review_rounds", 0), runtime.context.settings.review_max_revisions
    if rounds >= limit:
        return Command(update={"errors": [f"Revision limit reached ({limit}). Approve or reject this version."]},
                       goto="human_review")
    return Command(
        update={
            "feedback": [*state.get("feedback", []), decision.feedback.strip()],
            "review_rounds": rounds + 1,
            "revisions": 0,  # the critic gets its full number of tries again for the new draft
        },
        goto="writer",
    )
