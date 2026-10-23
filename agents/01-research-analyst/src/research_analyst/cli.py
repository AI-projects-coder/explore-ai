"""Command-line entry point: run an analysis from the terminal.

    research-analyst init-db                                  # once: create the tables
    research-analyst run "Analyze Apple's last 4 quarters"    # start; pauses for your review
    research-analyst resume run-1a2b3c4d                      # continue after a crash or a "decide later"
    research-analyst list                                     # recent runs and their status
    research-analyst show run-1a2b3c4d                        # print a run's report
"""

import argparse
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path

from langgraph.types import Command
from pydantic import ValidationError

from research_analyst.core.config import Settings, get_settings
from research_analyst.core.llm import Llm
from research_analyst.graph import advance, build_graph, run_config
from research_analyst.report import render
from research_analyst.state import AgentContext, ReviewDecision
from research_analyst.storage import DatabaseUnavailable, open_persistence, run_migrations
from research_analyst.tools.edgar import EdgarClient, SecAccessDenied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-analyst", description="Research a US public company from SEC filings.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="start a new analysis")
    p_run.add_argument("request", help="e.g. \"Analyze Apple's last 4 quarters\"")
    p_run.add_argument("-o", "--output", type=Path, help="also save the final report to this Markdown file")
    p_resume = sub.add_parser("resume", help="continue a run after a crash or a paused review")
    p_resume.add_argument("run_id")
    p_resume.add_argument("-o", "--output", type=Path)
    p_list = sub.add_parser("list", help="show recent runs")
    p_list.add_argument("-n", "--limit", type=int, default=20)
    p_show = sub.add_parser("show", help="print a run's report")
    p_show.add_argument("run_id")
    sub.add_parser("init-db", help="create or upgrade the database tables")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = get_settings()
    except ValidationError as e:
        print(f"Configuration error (check your .env):\n{e}", file=sys.stderr)
        return 2

    try:
        if args.command == "init-db":
            return asyncio.run(_init_db(settings))
        if args.command == "list":
            return asyncio.run(_list(settings, args.limit))
        if args.command == "show":
            return asyncio.run(_show(settings, args.run_id))
        run_id = f"run-{uuid.uuid4().hex[:8]}" if args.command == "run" else args.run_id
        request = args.request if args.command == "run" else None
        return asyncio.run(_session(settings, run_id, request, args.output))
    except DatabaseUnavailable as e:
        print(f"Database problem: {e}", file=sys.stderr)
        return 2
    except SecAccessDenied as e:
        print(f"SEC refused access: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Everything up to the last finished step is saved; "
              "continue with: research-analyst resume <run id>", file=sys.stderr)
        return 130


async def _session(settings: Settings, run_id: str, request: str | None, output: Path | None) -> int:
    """Start or continue one run until it finishes or you choose to decide later."""
    started = time.monotonic()
    progress = lambda m: print(m, file=sys.stderr)  # noqa: E731
    async with open_persistence(settings) as (checkpointer, runs), EdgarClient(settings) as sec:
        graph = build_graph(checkpointer)
        llm = Llm(settings)
        context = AgentContext(settings=settings, sec=sec, llm=llm, run_id=run_id, runs=runs)
        snapshot = await graph.aget_state(run_config(run_id, context))

        if request is not None:
            await runs.create(run_id, request)
            print(f"Run {run_id}", file=sys.stderr)
            payload = {"request": request}
        elif snapshot.created_at is None:
            print(f"No run with id {run_id}.", file=sys.stderr)
            return 1
        elif not snapshot.next:
            print(f"Run {run_id} already finished ({snapshot.values.get('status')}). "
                  f"See it with: research-analyst show {run_id}", file=sys.stderr)
            return 0
        else:
            llm.set_baseline(snapshot.values.get("spend"))  # the budget continues, it doesn't restart
            await runs.set_status(run_id, "running")
            print(f"Resuming {run_id} at: {', '.join(snapshot.next)}", file=sys.stderr)
            payload = None  # continue from the last saved step

        while True:
            if snapshot.interrupts and payload is None:  # paused for review: ask before continuing
                decision = _ask_review(snapshot.values, settings)
                if decision is None:
                    await runs.set_status(run_id, "awaiting_review")
                    print(f"Saved. Continue later with: research-analyst resume {run_id}", file=sys.stderr)
                    return 0
                await runs.set_status(run_id, "running")
                payload = Command(resume=decision.model_dump())
            snapshot = await advance(graph, run_id, payload, context, progress)
            payload = None
            if not snapshot.interrupts:
                break

    state = snapshot.values
    for error in dict.fromkeys(state.get("errors", [])):
        print(f"! {error}", file=sys.stderr)
    spent = state.get("spend")
    if spent:
        cost = f", about ${spent.usd:.2f}" if spent.usd else ""
        print(f"({time.monotonic() - started:.0f}s this session; run total {spent.llm_calls} LLM calls, "
              f"{spent.tokens:,} tokens{cost})", file=sys.stderr)

    status = state.get("status")
    if status == "rejected":
        print(f"Run {run_id} rejected. Nothing was published.", file=sys.stderr)
        return 0
    markdown = render(state)
    print(markdown)
    if output:
        output.write_text(markdown, encoding="utf-8")
        print(f"Saved to {output}", file=sys.stderr)
    return 0 if status == "approved" else 1


def _ask_review(state: dict, settings: Settings) -> ReviewDecision | None:
    """Show the draft and ask for a decision. None = decide later."""
    print(render(state), file=sys.stderr)
    rounds, limit = state.get("review_rounds", 0), settings.review_max_revisions
    can_revise = rounds < limit
    options = "[a]pprove, " + ("[r]evise, " if can_revise else "") + "re[j]ect, [q] decide later"
    while True:
        choice = _input(f"\nReview ({rounds}/{limit} revisions used). {options} > ").strip().lower()
        if choice in ("a", "approve"):
            return ReviewDecision(action="approve")
        if choice in ("j", "reject"):
            return ReviewDecision(action="reject")
        if choice in ("q", "quit", ""):
            return None
        if choice in ("r", "revise") and can_revise:
            feedback = _input("What should change? > ").strip()
            if feedback:
                return ReviewDecision(action="revise", feedback=feedback)
            print("Please describe what should change.", file=sys.stderr)
            continue
        print(f"Please choose: {options}", file=sys.stderr)


def _input(prompt: str) -> str:
    """input() prints its prompt to stdout, which would end up inside a redirected report. Use stderr."""
    print(prompt, end="", file=sys.stderr, flush=True)
    return input()


async def _init_db(settings: Settings) -> int:
    await asyncio.to_thread(run_migrations, settings)  # our runs table (Alembic)
    async with open_persistence(settings) as (checkpointer, _):
        await checkpointer.setup()  # LangGraph's checkpoint tables
    print("Database ready.", file=sys.stderr)
    return 0


async def _list(settings: Settings, limit: int) -> int:
    async with open_persistence(settings) as (_, runs):
        rows = await runs.list(limit)
    if not rows:
        print("No runs yet.", file=sys.stderr)
        return 0
    for r in rows:
        print(f"{r['run_id']}  {r['status']:<16} {r['ticker'] or '-':<6} "
              f"{r['created_at']:%Y-%m-%d %H:%M}  {r['request'][:60]}")
    return 0


async def _show(settings: Settings, run_id: str) -> int:
    async with open_persistence(settings) as (checkpointer, runs):
        row = await runs.get(run_id)
        if row is None:
            print(f"No run with id {run_id}.", file=sys.stderr)
            return 1
        print(f"{run_id}: {row['status']}", file=sys.stderr)
        if row["report_markdown"]:
            print(row["report_markdown"])
            return 0
        # not finished yet: show the current draft from the latest checkpoint
        snapshot = await build_graph(checkpointer).aget_state({"configurable": {"thread_id": run_id}})
        if snapshot.values.get("report") or snapshot.values.get("metrics"):
            print(render(snapshot.values))
        return 0


if __name__ == "__main__":
    sys.exit(main())
