"""Compare two eval runs: what changed in the config, the metrics, and per case.

Usage:  uv run python evals/compare.py                    # baseline.json (if it exists) vs the latest full run
        uv run python evals/compare.py OLD.json NEW.json  # specific runs

Make a run the baseline:  cp evals/results/<run>.json evals/baseline.json
"""
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def full_runs() -> list[Path]:
    return [p for p in sorted(RESULTS.glob("*.json")) if len(json.loads(p.read_text())["results"]) >= 20]


def main() -> None:
    if len(sys.argv) == 3:
        old_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
    elif (Path(__file__).parent / "baseline.json").exists():
        old_path, new_path = Path(__file__).parent / "baseline.json", full_runs()[-1]
    else:
        old_path, new_path = full_runs()[-2:]
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    print(f"OLD {old_path.name}\nNEW {new_path.name}\n")

    changed = {k: (old["config"].get(k), v) for k, v in new["config"].items() if old["config"].get(k) != v}
    print("CONFIG CHANGES   " + (", ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items()) or "none"))
    if len(changed) > 1:
        print("  ! more than one setting changed: you can't tell which one caused a difference")

    o, n = old["summary"], new["summary"]
    print("\nMETRIC                OLD        NEW        CHANGE")
    for key, fmt in [("pass_rate", "{:.0%}"), ("outcome_accuracy", "{:.0%}"), ("retrieval_recall", "{:.0%}"),
                     ("faithfulness", "{:.0%}"), ("tokens_in", "{:,}"), ("tokens_out", "{:,}"),
                     ("avg_latency_ms", "{:,}")]:
        a, b = o.get(key), n.get(key)
        if a is None or b is None:
            continue
        delta = b - a
        pct = f"({delta / a:+.0%})" if a and key.startswith(("tokens", "avg")) else ""
        print(f"{key:20}  {fmt.format(a):>9}  {fmt.format(b):>9}   {delta:+.3g} {pct}")
    print(f"{'leaks':20}  {len(o['leaks']):>9}  {len(n['leaks']):>9}   {'!!! NEW LEAKS' if n['leaks'] else ''}")

    old_cases = {r["id"]: r for r in old["results"]}
    fixed = [r["id"] for r in new["results"] if r["passed"] and not old_cases.get(r["id"], {}).get("passed", True)]
    broken = [r for r in new["results"] if not r["passed"] and old_cases.get(r["id"], {}).get("passed", False)]
    print(f"\nFIXED   ({len(fixed)}): {', '.join(fixed) or '-'}")
    print(f"BROKEN  ({len(broken)}): {', '.join(r['id'] for r in broken) or '-'}")
    for r in broken:
        print(f"   {r['id']}: {'; '.join(r['failures'])}")
        print(f"      was: {old_cases[r['id']]['answer'][:150]}")
        print(f"      now: {r['answer'][:150]}")


if __name__ == "__main__":
    main()
