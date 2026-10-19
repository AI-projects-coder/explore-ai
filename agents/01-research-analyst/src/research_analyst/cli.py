"""Command-line entry point: run an analysis from the terminal.

    research-analyst "Analyze Apple's last 4 quarters"
    research-analyst "How is Nvidia doing?" --output nvda.md
"""

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from research_analyst.core.config import Settings, get_settings
from research_analyst.core.llm import Llm
from research_analyst.graph import run_analysis
from research_analyst.nodes.calculator import display
from research_analyst.state import AgentContext, AgentState
from research_analyst.tools.edgar import EdgarClient, SecAccessDenied

_CITED = re.compile(r"\[(m\d+)\]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-analyst", description="Research a US public company from SEC filings.")
    parser.add_argument("request", help="e.g. \"Analyze Apple's last 4 quarters\"")
    parser.add_argument("-o", "--output", type=Path, help="also save the report to this Markdown file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = get_settings()
    except ValidationError as e:
        print(f"Configuration error (check your .env):\n{e}", file=sys.stderr)
        return 2

    started = time.monotonic()
    try:
        state, llm = asyncio.run(_run(args.request, settings))
    except SecAccessDenied as e:
        print(f"SEC refused access: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130

    for error in state.get("errors", []):
        print(f"! {error}", file=sys.stderr)
    tokens = sum(llm.usage.input_tokens.values()) + sum(llm.usage.output_tokens.values())
    print(f"({time.monotonic() - started:.0f}s, {llm.usage.calls} LLM calls, {tokens:,} tokens)", file=sys.stderr)

    if "report" not in state:
        return 1
    markdown = render_report(state)
    print(markdown)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Saved to {args.output}", file=sys.stderr)
    return 0


async def _run(request: str, settings: Settings) -> tuple[AgentState, Llm]:
    llm = Llm(settings)
    async with EdgarClient(settings) as sec:
        context = AgentContext(settings=settings, sec=sec, llm=llm)
        state = await run_analysis(request, context, on_progress=lambda m: print(m, file=sys.stderr))
    return state, llm


def render_report(state: AgentState) -> str:
    """The report as Markdown, plus the numbers it cites and the filings behind them."""
    report, metrics = state["report"], state.get("metrics", {})
    parts = [f"# {report.title}", report.summary, report.body_markdown]

    cited = [mid for mid in dict.fromkeys(_CITED.findall(report.body_markdown)) if mid in metrics]
    if cited:
        rows = ["| id | metric | period | value | how |", "|---|---|---|---|---|"]
        rows += [f"| {mid} | {metrics[mid].name} | {metrics[mid].period_label} | {display(metrics[mid])[0]} | "
                 f"{metrics[mid].formula} |" for mid in cited]
        parts.append("## Numbers used\n" + "\n".join(rows))

    sources = {s.accession: s for f in state.get("findings", []) for s in f.sources if s.accession}
    if sources:
        lines = [f"- {s.filed.isoformat() if s.filed else '?'} {s.kind}: {s.description} ({s.accession})"
                 + (f" {s.url}" if s.url else "")
                 for s in sorted(sources.values(), key=lambda s: s.filed or 0, reverse=True)]
        parts.append("## Sources\n" + "\n".join(lines))

    parts.append("_Generated from SEC filings by an AI research agent. Not investment advice._")
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    sys.exit(main())
