"""See the SEC client work on real data.

    uv run python scripts/try_edgar.py AAPL
"""

import asyncio
import sys

from research_analyst.core.config import get_settings
from research_analyst.tools.edgar import EdgarClient, SecError


async def main(ticker: str) -> None:
    async with EdgarClient(get_settings()) as sec:
        try:
            company = await sec.resolve_ticker(ticker)
            print(f"1. ticker {ticker!r} -> {company.name}, CIK {company.cik}")

            facts = await sec.company_facts(company.cik)
            print(f"2. financial facts: {len(facts['facts'].get('us-gaap', {}))} different numbers reported")

            subs = await sec.submissions(company.cik)
            forms = subs["filings"]["recent"]["form"]
            dates = subs["filings"]["recent"]["filingDate"]
            recent_8k = [d for f, d in zip(forms, dates) if f == "8-K"][:3]
            print(f"3. latest 8-K filings (company news): {recent_8k}")

            # same company again: comes from the cache, no new download
            await sec.company_facts(company.cik)
            print("4. second request for the same data: served from cache")
        except SecError as e:
            print(f"SEC problem ({type(e).__name__}): {e}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "AAPL"))
