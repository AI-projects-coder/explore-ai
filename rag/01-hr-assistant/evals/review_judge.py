"""Print judge verdicts from the latest eval run for a HUMAN to check. Usage: uv run python evals/review_judge.py [n]

For each verdict, decide: do I agree? Count disagreements. If you disagree with more than ~1 in 10,
the judge prompt needs work before its faithfulness score can be trusted.
"""
import json
import random
import sys
from pathlib import Path

# The latest FULL run (single-case reruns with --ids would only give one or two verdicts to review).
runs = sorted((Path(__file__).parent / "results").glob("*.json"), reverse=True)
latest = next(p for p in runs if len(json.loads(p.read_text())["results"]) >= 20)
results = [r for r in json.loads(latest.read_text())["results"] if r["judge_claims"]]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
random.seed(7)  # same sample every time you run it, so you can come back to it

print(f"run: {latest.name}   reviewing {min(n, len(results))} of {len(results)} judged answers\n")
for i, r in enumerate(random.sample(results, min(n, len(results))), 1):
    print(f"{'=' * 80}\n#{i}  {r['id']}   (check the source: uv run python scripts/review_case.py {r['id']})")
    print(f"Q: {r['question']}\nA: {r['answer']}")
    for c in r["judge_claims"]:
        mark = "✅ supported  " if c["supported"] else "❌ UNSUPPORTED"
        print(f"   {mark} [excerpt {c['excerpt']}] {c['claim']}\n                  reason: {c['reason']}")
    print("   YOUR VERDICT: agree / disagree (why?)")
