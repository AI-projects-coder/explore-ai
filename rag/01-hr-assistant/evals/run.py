"""Run the golden dataset against the REAL pipeline (Vertex AI + Gemini + Postgres) and score it.

Usage:  uv run python evals/run.py [--ids id1,id2] [--category access] [--concurrency 4] [--no-judge]
Exit code 1 if any must_pass case fails or any access leak happens (so CI can block a release).
"""
import argparse
import asyncio
import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from judge import JUDGE_SYSTEM_PROMPT, Judge, JudgeVerdict

from hr_assistant.core.auth import DEV_USERS, User
from hr_assistant.core.config import get_settings
from hr_assistant.db.session import get_async_session_factory
from hr_assistant.embeddings.vertex import VertexEmbedder
from hr_assistant.generation.answer import Answer, AnswerService, NoAnswerMessages
from hr_assistant.generation.llm import GeminiLLM
from hr_assistant.generation.prompts import SYSTEM_PROMPT
from hr_assistant.retrieval.search import RetrievedChunk, Retriever

EVALS_DIR = Path(__file__).parent
RESULTS_DIR = EVALS_DIR / "results"


# ---------------------------------------------------------------- matching helpers


def _normalize(text: str) -> str:
    return text.lower().replace(",", "")


def contains(text: str, expected: str) -> bool:
    """True if any "|" alternative appears. Numbers match as whole numbers: "4" is not found in "14"."""
    haystack = _normalize(text)
    for alt in _normalize(expected).split("|"):
        alt = alt.strip()
        if re.fullmatch(r"\d+(\.\d+)?%?", alt):
            if re.search(rf"(?<![\d.]){re.escape(alt)}(?![\d])", haystack):
                return True
        elif alt in haystack:
            return True
    return False


def section_found(chunks: list[RetrievedChunk], expected: str) -> bool:
    """A section counts as retrieved if it is in a chunk's heading path, or a heading inside a merged chunk."""
    return any(
        alt in c.heading_path or f"## {alt}" in c.content for alt in expected.split("|") for c in chunks
    )


def actual_outcome(answer: Answer) -> str:
    if answer.answerable:
        return "answered"
    if answer.no_answer_reason in ("off_topic", "not_found", "no_context"):
        return answer.no_answer_reason
    return "not_found"  # broken model output: the user sees the not-found message


def outcome_ok(expected: str | list[str], actual: str) -> bool:
    allowed = {expected} if isinstance(expected, str) else set(expected)
    if actual == "no_context":  # retrieval found nothing: the message covers both cases
        return bool(allowed & {"not_found", "off_topic"})
    return actual in allowed


# ---------------------------------------------------------------- scoring one case


@dataclass
class CaseResult:
    id: str
    category: str
    tier: str
    user: str
    question: str
    passed: bool
    failures: list[str] = field(default_factory=list)  # "<layer>: <detail>"
    expected_outcome: str | list[str] = ""
    actual_outcome: str = ""
    no_answer_reason: str | None = None
    answer: str = ""
    retrieved: list[str] = field(default_factory=list)  # "title > section (score)"
    top_score: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    faithful: bool | None = None  # None = not judged (no answer, or --no-judge)
    unsupported_claims: list[str] = field(default_factory=list)
    judge_facts_stated: bool | None = None  # the judge's opinion, compared with must_contain later
    judge_claims: list[dict] = field(default_factory=list)  # every verdict, kept for human review
    known_issue: bool = False  # a failure we accepted on purpose: reported, but not a regression


def score(case: dict, chunks: list[RetrievedChunk], answer: Answer, verdict: JudgeVerdict | None) -> CaseResult:
    result = CaseResult(
        id=case["id"],
        category=case["category"],
        tier=case["tier"],
        user=case["user"],
        question=case["question"],
        passed=True,
        expected_outcome=case["expected_outcome"],
        actual_outcome=actual_outcome(answer),
        no_answer_reason=answer.no_answer_reason,
        answer=answer.text,
        retrieved=[f"{c.heading_path} ({c.score:.3f})" for c in chunks],
        top_score=chunks[0].score if chunks else 0.0,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        latency_ms=answer.latency_ms,
        known_issue=case.get("known_issue", False),
    )
    fail = result.failures.append

    # Access control first: the one failure that is never acceptable.
    forbidden = set(case.get("forbidden_documents", []))
    leaked = {c.title for c in chunks} & forbidden | {c.title for c in answer.citations} & forbidden
    if leaked:
        fail(f"LEAK: retrieved forbidden documents {sorted(leaked)}")

    if not outcome_ok(case["expected_outcome"], result.actual_outcome):
        fail(f"outcome: expected {case['expected_outcome']}, got {result.actual_outcome}"
             f" (reason={answer.no_answer_reason})")

    for section in case.get("expected_sections", []):
        if not section_found(chunks, section):
            fail(f"retrieval: section {section!r} not in top {len(chunks)}")

    if result.actual_outcome == "answered":
        for fact in case.get("must_contain", []):
            if not contains(answer.text, fact):
                fail(f"answer: missing {fact!r}")
    for bad in case.get("must_not_contain", []):
        if contains(answer.text, bad):
            fail(f"answer: contains forbidden {bad!r}")

    if verdict is not None:
        result.faithful = verdict.faithful
        result.unsupported_claims = [f"{c.claim}  ({c.reason})" for c in verdict.unsupported]
        result.judge_facts_stated = verdict.expected_facts_stated
        result.judge_claims = [c.model_dump() for c in verdict.claims]
        for claim in verdict.unsupported:
            fail(f"faithfulness: unsupported claim {claim.claim!r}")

    result.passed = not result.failures
    return result


# ---------------------------------------------------------------- running


async def run_case(
    case: dict, retriever: Retriever, service: AnswerService, judge: Judge | None, gate: asyncio.Semaphore
) -> CaseResult:
    user = User(case["user"], DEV_USERS[case["user"]])
    async with gate:
        # Retrieval is deterministic, so this is the same list the answer service sees.
        chunks = await retriever.search(case["question"], user)
        answer = await service.answer(case["question"], user)
        verdict = None
        if judge and answer.answerable:  # only answers can be (un)faithful: refusals skip the paid call
            verdict = await judge.judge(case["question"], chunks, answer.text, case.get("must_contain", []))
    return score(case, chunks, answer, verdict)


def report(results: list[CaseResult], config: dict, elapsed_s: float) -> dict:
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)
    with_sections = [r for r in results if r.category not in ("access", "not_found", "off_topic")]
    judged = [r for r in results if r.faithful is not None]
    # Judge vs. rules: on cases with must_contain, does the judge agree with our deterministic check?
    comparable = [r for r in judged if r.judge_facts_stated is not None]
    agree = [r for r in comparable if r.judge_facts_stated == (not any(f.startswith("answer: missing") for f in r.failures))]
    layers = Counter(f.split(":")[0] for r in results for f in r.failures)
    summary = {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / len(results), 3),
        "must_pass_failures": [r.id for r in results if r.tier == "must_pass" and not r.passed],
        "known_issues_failing": [r.id for r in results if r.known_issue and not r.passed],
        "unexpected_failures": [r.id for r in results if not r.known_issue and not r.passed],
        "leaks": [r.id for r in results if any(f.startswith("LEAK") for f in r.failures)],
        "outcome_accuracy": round(sum(not any(f.startswith("outcome") for f in r.failures) for r in results)
                                  / len(results), 3),
        "retrieval_recall": round(sum(not any(f.startswith("retrieval") for f in r.failures) for r in with_sections)
                                  / max(len(with_sections), 1), 3),
        "faithfulness": round(sum(r.faithful for r in judged) / len(judged), 3) if judged else None,
        "judged": len(judged),
        "judge_agrees_with_rules": f"{len(agree)}/{len(comparable)}",
        "failures_by_layer": dict(layers),
        "by_category": {c: f"{sum(r.passed for r in rs)}/{len(rs)}" for c, rs in sorted(by_category.items())},
        "tokens_in": sum(r.input_tokens for r in results),
        "tokens_out": sum(r.output_tokens for r in results),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results)),
        "elapsed_s": round(elapsed_s, 1),
    }
    return {"run_at": datetime.now(UTC).isoformat(), "config": config, "summary": summary,
            "results": [asdict(r) for r in results]}


def print_report(data: dict) -> None:
    s = data["summary"]
    print(f"\n{'=' * 70}\nEVAL RUN  {data['run_at']}\nconfig: {data['config']}\n{'=' * 70}")
    print(f"passed            {s['passed']}/{s['cases']}  ({s['pass_rate']:.0%})   "
          f"unexpected failures: {len(s['unexpected_failures'])}   known issues: {len(s['known_issues_failing'])}")
    print(f"outcome accuracy  {s['outcome_accuracy']:.0%}")
    print(f"retrieval recall  {s['retrieval_recall']:.0%}   (cases that expect sections)")
    if s["faithfulness"] is not None:
        print(f"faithfulness      {s['faithfulness']:.0%}   ({s['judged']} answers judged; "
              f"judge agrees with must_contain on {s['judge_agrees_with_rules']})")
    print(f"ACCESS LEAKS      {len(s['leaks'])}  {s['leaks'] or ''}")
    print(f"must_pass failed  {len(s['must_pass_failures'])}  {s['must_pass_failures'] or ''}")
    print(f"tokens            in={s['tokens_in']} out={s['tokens_out']}   avg latency {s['avg_latency_ms']} ms")
    print(f"failures by layer {s['failures_by_layer'] or '-'}")
    print("by category       " + "  ".join(f"{c}={v}" for c, v in s["by_category"].items()))
    failed = [r for r in data["results"] if not r["passed"]]
    if failed:
        print(f"\n{'-' * 70}\nFAILURES ({len(failed)})")
        for r in failed:
            tag = "  [KNOWN ISSUE]" if r["known_issue"] else ""
            print(f"\n[{r['tier']}] {r['id']}{tag}  ({r['user']}): {r['question']}")
            for f in r["failures"]:
                print(f"   ✗ {f}")
            print(f"   answer:    {r['answer'][:160]}")
            for claim in r["unsupported_claims"]:
                print(f"   judge:     ✗ {claim[:200]}")
            print(f"   retrieved: {'; '.join(x.split(' > ')[-1] for x in r['retrieved'][:5])}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="comma-separated case ids")
    parser.add_argument("--category")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (faster, cheaper)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    cases = yaml.safe_load((EVALS_DIR / "golden.yaml").read_text())["cases"]
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids.split(","))]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    s = get_settings()
    retriever = Retriever(
        embedder=VertexEmbedder.from_settings(s),
        session_factory=get_async_session_factory(),
        top_k=s.retrieval_top_k,
        min_score=s.retrieval_min_score,
        ef_search=s.hnsw_ef_search,
    )
    service = AnswerService(
        retriever=retriever,
        llm=GeminiLLM.from_settings(s),
        messages=NoAnswerMessages(s.not_found_message, s.off_topic_message, s.no_context_message),
    )
    judge = None if args.no_judge else Judge(project=s.gcp_project)
    config = {  # everything that can change the results: stored with every run so runs are comparable
        "llm_model": s.llm_model, "thinking": s.llm_thinking_level, "temperature": s.llm_temperature,
        "embedding_model": s.embedding_model, "task_types": s.embedding_task_types,
        "top_k": s.retrieval_top_k, "min_score": s.retrieval_min_score,
        "chunk_max_tokens": s.chunk_max_tokens, "chunk_min_tokens": s.chunk_min_tokens,
        "judge": judge.model if judge else None,
        # Prompts are configuration too: a short fingerprint shows when they changed between runs.
        "answer_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8],
        "judge_prompt": hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest()[:8],
        "golden_set": hashlib.sha256((EVALS_DIR / "golden.yaml").read_bytes()).hexdigest()[:8],
    }

    started = time.perf_counter()
    gate = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(run_case(c, retriever, service, judge, gate) for c in cases))
    data = report(list(results), config, time.perf_counter() - started)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print_report(data)
    print(f"\nsaved: {out}")
    return 1 if data["summary"]["leaks"] or data["summary"]["must_pass_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
