"""See the calculator on real data: growth, margins, free cash flow, TTM.

    uv run python scripts/try_metrics.py AAPL
"""

import asyncio
import sys

from research_analyst.core.config import get_settings
from research_analyst.finance import metrics as m
from research_analyst.tools.edgar import EdgarClient, SecError
from research_analyst.tools.xbrl import XbrlError, quarterly_series


def show(r: m.MetricResult) -> None:
    value = f"{r.value:+.1%}" if r.unit == "ratio" else f"{r.value / 1e9:,.2f}B"
    print(f"  {r.period_label:18} {value:>10}    = {r.formula}")


async def main(ticker: str) -> None:
    async with EdgarClient(get_settings()) as sec:
        try:
            company = await sec.resolve_ticker(ticker)
            facts = await sec.company_facts(company.cik)
        except SecError as e:
            print(f"SEC problem ({type(e).__name__}): {e}")
            return
    print(company.name)

    def series(metric):
        try:
            return quarterly_series(facts, metric, quarters=8)
        except XbrlError as e:
            print(f"  ({metric}: {e})")
            return []

    revenue, net_income = series("revenue"), series("net_income")
    ocf, capex = series("operating_cash_flow"), series("capital_expenditure")

    print("\nRevenue growth vs same quarter last year (YoY):")
    for r in m.yoy_growth(revenue)[-4:]:
        show(r)
    print("\nRevenue growth vs previous quarter (QoQ):")
    for r in m.qoq_growth(revenue)[-4:]:
        show(r)
    print("\nNet margin:")
    for ni, rev in m.pair_by_period(net_income, revenue)[-4:]:
        show(m.margin(ni, rev))
    print("\nFree cash flow:")
    for o, c in m.pair_by_period(ocf, capex)[-4:]:
        show(m.free_cash_flow(o, c))
    print("\nRevenue, trailing twelve months:")
    try:
        show(m.trailing_twelve_months(revenue))
    except m.CalculationError as e:
        print(f"  refused: {e}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "AAPL"))
