"""See the clean quarterly numbers for a real company.

    uv run python scripts/try_xbrl.py AAPL
    uv run python scripts/try_xbrl.py JPM revenue
"""

import asyncio
import sys

from research_analyst.core.config import get_settings
from research_analyst.tools.edgar import EdgarClient, SecError
from research_analyst.tools.xbrl import XbrlError, available_metrics, quarterly_series


def fmt(q) -> str:
    v = f"{q.value / 1e9:,.2f}B" if q.unit == "USD" else f"{q.value:g}"
    return f"{q.label}: {v}{'*' if q.source == 'derived' else ''}"


async def main(ticker: str, metrics: list[str]) -> None:
    async with EdgarClient(get_settings()) as sec:
        try:
            company = await sec.resolve_ticker(ticker)
            facts = await sec.company_facts(company.cik)
        except SecError as e:
            print(f"SEC problem ({type(e).__name__}): {e}")
            return

    print(f"{company.name}  (* = derived, not reported directly)\n")
    for m in metrics:
        try:
            print(f"{m:20}", " | ".join(fmt(q) for q in quarterly_series(facts, m, quarters=4)))
        except XbrlError as e:
            print(f"{m:20} -- {e}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["AAPL"]
    asyncio.run(main(args[0], args[1:] or available_metrics()))
