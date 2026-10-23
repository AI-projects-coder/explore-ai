"""AgentState and the Pydantic schemas that flow between nodes (plan, findings, metrics, claims).

Two kinds of things live here:
  - Data (Pydantic models + AgentState): what the agent knows. Small, serializable, checkpointable.
  - Context (AgentContext): the tools the nodes use (settings, SEC client). Not data: never stored.
"""

import operator
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from research_analyst.core.config import Settings
from research_analyst.core.llm import Llm, Spend, max_spend
from research_analyst.finance.metrics import MetricResult
from research_analyst.tools.edgar import EdgarClient
from research_analyst.tools.xbrl import QuarterValue, available_metrics

if TYPE_CHECKING:
    from research_analyst.storage import RunStore


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


# --- planner ------------------------------------------------------------------

class PlannedTask(_Frozen):
    """One research topic, as the planner LLM writes it."""

    topic: str = Field(min_length=1, max_length=120, description="Short name, e.g. 'Revenue trend'")
    question: str = Field(min_length=1, max_length=500, description="What the researcher must find out")
    kind: Literal["financials", "news"] = Field(
        description="financials: numbers from SEC XBRL data; news: recent 8-K filings")
    metrics: list[str] = Field(default_factory=list, description="For financials: which metrics to fetch")

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(available_metrics()))
        if unknown:
            raise ValueError(f"unknown metrics {unknown}; choose from {available_metrics()}")
        return list(dict.fromkeys(v))  # drop duplicates, keep order

    @model_validator(mode="after")
    def _metrics_match_kind(self) -> "PlannedTask":
        if self.kind == "financials" and not self.metrics:
            raise ValueError("a financials task needs at least one metric")
        if self.kind == "news" and self.metrics:
            raise ValueError("a news task must not list metrics")
        return self


class PlanDraft(_Frozen):
    """The planner LLM's whole answer. Validated before anything else runs."""

    ticker: str = Field(min_length=1, max_length=12, description="Stock ticker, e.g. AAPL")
    quarters: int = Field(ge=1, le=40, description="How many recent quarters to analyze")
    tasks: list[PlannedTask] = Field(min_length=1, description="Research topics, most important first")


class ResearchTask(PlannedTask):
    """A PlannedTask after our code accepted it: it now has an id we assigned (never the LLM)."""

    id: str  # "t1", "t2", ...


class Plan(_Frozen):
    ticker: str
    company_name: str  # from SEC's company list, not from the LLM
    cik: int
    quarters: int
    tasks: list[ResearchTask]


# --- researchers ----------------------------------------------------------------

class Source(_Frozen):
    """A document a finding is based on, for citations."""

    kind: Literal["xbrl", "8-K"]
    accession: str | None
    filed: date | None
    url: str | None = None
    description: str


class Finding(_Frozen):
    """What one researcher brought back for one task."""

    task_id: str
    topic: str
    status: Literal["ok", "partial", "failed"]
    summary: str  # the researcher's short, qualitative notes (no calculated numbers)
    data: list[QuarterValue] = Field(default_factory=list)  # clean numbers from xbrl.py
    sources: list[Source] = Field(default_factory=list)
    error: str | None = None  # why it's partial/failed, in plain words


def merge_findings(existing: list[Finding], new: list[Finding]) -> list[Finding]:
    """Reducer: one finding per task_id, the newest wins.

    Researchers run in parallel and each returns [finding]; this merges them. Keyed by task_id
    (not just appended) so a node that runs twice -- a retry, or Cloud Tasks delivering twice in
    v2 -- replaces its finding instead of duplicating it. That's what makes the step idempotent.
    """
    by_task = {f.task_id: f for f in existing}
    for f in new:
        by_task[f.task_id] = f
    return list(by_task.values())


# --- writer ---------------------------------------------------------------------

class Claim(_Frozen):
    """One statement in the report. Numeric claims point at the calculated metric they rely on."""

    text: str = Field(min_length=1, max_length=500)
    kind: Literal["numeric", "qualitative"]
    metric_id: str | None = Field(default=None, description="For numeric claims: the id of the metric, e.g. 'm3'")
    stated_value: float | None = Field(default=None, description="For numeric claims: the number as written")


class Report(_Frozen):
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)
    body_markdown: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)

    @model_validator(mode="after")
    def _numeric_claims_match_metrics(self, info: ValidationInfo) -> "Report":
        """With context {"metric_values": {"m1": 16.4, ...}}, every numeric claim must cite a real
        metric id and state its value as shown in the table. Without context (tests, restored
        checkpoints) only the structure is checked."""
        values = (info.context or {}).get("metric_values")
        if values is None:
            return self
        problems = []
        for i, c in enumerate(self.claims):
            if c.kind != "numeric":
                continue
            if c.metric_id not in values:
                problems.append(f"claims.{i}: metric_id {c.metric_id!r} is not in the metrics table")
            elif c.stated_value is None or abs(c.stated_value - values[c.metric_id]) > 0.051:
                problems.append(f"claims.{i}: stated_value {c.stated_value} but {c.metric_id} is {values[c.metric_id]}")
        if problems:
            raise ValueError("; ".join(problems[:5]))
        return self


# --- critic and human review (v1) ------------------------------------------------

class Critique(_Frozen):
    """The critic's verdict on the current draft."""

    passed: bool
    problems: list[str] = Field(default_factory=list)  # exact, actionable: sent to the writer as-is
    fully_checked: bool = True  # False when the qualitative judge couldn't run (budget or LLM failure)
    send_back: bool = False  # True: the writer gets another try (within critic_max_revisions)


class ReviewDecision(_Frozen):
    """What the human decided at the review pause."""

    action: Literal["approve", "revise", "reject"]
    feedback: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _revise_needs_feedback(self) -> "ReviewDecision":
        if self.action == "revise" and not self.feedback.strip():
            raise ValueError("a revision needs feedback: what should change?")
        return self


RunStatus = Literal["running", "awaiting_review", "approved", "rejected", "partial", "failed"]


# --- graph state ------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """The shared notebook every node reads from and writes to.

    Each node returns only the keys it changes. Keys with a reducer (Annotated[..., fn]) are
    merged with fn; the others are simply replaced.
    """

    request: str
    plan: Plan
    findings: Annotated[list[Finding], merge_findings]
    metrics: dict[str, MetricResult]  # "m1" -> result; ids let claims and the critic refer to them
    report: Report
    errors: Annotated[list[str], operator.add]  # problems worth showing the user; never silently lost

    # v1
    spend: Annotated[Spend, max_spend]  # the run's running total; in the state so a resume can't reset it
    critique: Critique | None
    revisions: int  # critic -> writer rewrites in the current review round
    review_rounds: int  # human "revise" decisions so far
    feedback: list[str]  # human feedback, oldest first (replaced as a whole list: idempotent)
    decision: Literal["approved", "rejected"]
    status: RunStatus  # set by finalize


class ResearcherInput(TypedDict):
    """What each parallel researcher receives (via Send): only its own task, not the whole state."""

    task: ResearchTask
    ticker: str
    quarters: int


@dataclass(frozen=True)
class AgentContext:
    """Long-lived tools for one run, passed to nodes by LangGraph. Never part of the state:
    connections can't be saved in a checkpoint, and they aren't knowledge about the run."""

    settings: Settings
    sec: EdgarClient
    llm: Llm
    run_id: str = "local"  # the LangGraph thread_id; also the key in the runs table
    runs: "RunStore | None" = None  # None in tests and one-off runs: finalize then skips the database


# Every class that is saved inside a checkpoint. When a checkpoint is loaded, LangGraph rebuilds only
# the classes on this list; anything else in a stored checkpoint is refused. That stops a tampered
# checkpoint row from making the app construct arbitrary objects.
CHECKPOINT_TYPES: list[tuple[str, str]] = [
    (cls.__module__, cls.__name__)
    for cls in (Plan, ResearchTask, Finding, Source, Report, Claim, QuarterValue, MetricResult,
                Spend, Critique, ReviewDecision)
]
