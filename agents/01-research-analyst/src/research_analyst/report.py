"""Turns a run's state into the Markdown a reader sees.

Used by the CLI (to print) and by the finalize node (to store in the runs table).
When there is no written report (the writer failed or the budget ran out), it still renders
the verified numbers and findings: a data-only report instead of nothing.
"""

import re

from research_analyst.nodes.calculator import display
from research_analyst.state import AgentState

_CITED = re.compile(r"\[(m\d+)\]")
_FOOTER = "_Generated from SEC filings by an AI research agent. Not investment advice._"


def render(state: AgentState) -> str:
    report = state.get("report")
    metrics = state.get("metrics", {})
    if report is not None:
        parts = [f"# {report.title}", report.summary, report.body_markdown]
        text = " ".join([report.title, report.summary, report.body_markdown])
        cited = [mid for mid in dict.fromkeys(_CITED.findall(text)) if mid in metrics]
        if cited:
            parts.append("## Numbers used\n" + _metrics_table(metrics, cited))
        critique = state.get("critique")
        if critique is not None and (critique.problems or not critique.fully_checked):
            parts.append(_review_note(critique))
    else:
        plan = state.get("plan")
        name = f"{plan.company_name} ({plan.ticker})" if plan else "Unknown company"
        parts = [f"# Data-only report: {name}",
                 "The written analysis could not be produced, so this report shows only the verified "
                 "data the agent collected."]
        if state.get("errors"):
            parts.append("**Why:**\n" + "\n".join(f"- {e}" for e in state["errors"]))
        if metrics:
            parts.append("## Numbers\n" + _metrics_table(metrics, list(metrics)))
        for f in state.get("findings", []):
            if f.summary:
                parts.append(f"## {f.topic}\n{f.summary}")

    sources = _sources(state)
    if sources:
        parts.append("## Sources\n" + sources)
    parts.append(_FOOTER)
    return "\n\n".join(parts) + "\n"


def _metrics_table(metrics: dict, ids: list[str]) -> str:
    rows = ["| id | metric | period | value | how |", "|---|---|---|---|---|"]
    rows += [f"| {mid} | {metrics[mid].name} | {metrics[mid].period_label} | {display(metrics[mid])[0]} | "
             f"{metrics[mid].formula} |" for mid in ids]
    return "\n".join(rows)


def _review_note(critique) -> str:
    lines = ["## Review notes"]
    if not critique.fully_checked:
        lines.append("- Qualitative statements were not fully checked (the judge could not run).")
    lines += [f"- {p}" for p in critique.problems]
    return "\n".join(lines)


def _sources(state: AgentState) -> str:
    by_accession = {s.accession: s for f in state.get("findings", []) for s in f.sources if s.accession}
    return "\n".join(
        f"- {s.filed.isoformat() if s.filed else '?'} {s.kind}: {s.description} ({s.accession})"
        + (f" {s.url}" if s.url else "")
        for s in sorted(by_accession.values(), key=lambda s: s.filed.isoformat() if s.filed else "", reverse=True)
    )
