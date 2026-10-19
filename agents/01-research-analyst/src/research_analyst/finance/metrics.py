"""Deterministic finance functions (growth, margins, QoQ/YoY). No LLM, no eval of generated code.

Every function takes clean QuarterValues (from tools/xbrl.py) and returns MetricResults that carry
their inputs and a readable formula, so any number in the report can be traced and re-checked.

Each calculation refuses, with a clear CalculationError, instead of returning a misleading number:
  - comparing quarters that are not really neighbours (series can have gaps),
  - mixing different periods or units,
  - growth from a zero or negative base ("from -5 to +10" is not "+300%").
"""

from collections.abc import Iterable
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

from research_analyst.tools.xbrl import QuarterValue

_NEXT_QUARTER_DAYS = range(80, 101)  # period ends of consecutive quarters (12-14 week quarters)
_NEXT_YEAR_DAYS = range(350, 381)  # same quarter one year later (52/53-week years)


class CalculationError(Exception):
    """The inputs can't produce a meaningful number. The message says why, in plain words."""


class MetricResult(BaseModel):
    """One calculated number, with everything needed to verify it."""

    model_config = ConfigDict(frozen=True)

    name: str  # e.g. "revenue_yoy_growth", "net_margin"
    period_label: str  # e.g. "Q3 FY2026"
    period_end: date
    value: float
    unit: Literal["ratio", "USD", "USD/shares"]  # ratio: 0.061 means 6.1%
    formula: str  # human-readable, e.g. "(109.42B - 94.04B) / 94.04B"
    inputs: tuple[QuarterValue, ...]  # the exact numbers used, with their SEC filings


# --- growth -------------------------------------------------------------------

def growth(current: QuarterValue, base: QuarterValue, kind: Literal["qoq", "yoy"]) -> MetricResult:
    """Percentage change from `base` to `current`. qoq: previous quarter; yoy: same quarter last year."""
    _same_metric(current, base)
    gap = (current.period_end - base.period_end).days
    expected = _NEXT_QUARTER_DAYS if kind == "qoq" else _NEXT_YEAR_DAYS
    if gap not in expected:
        what = "the previous quarter" if kind == "qoq" else "the same quarter a year earlier"
        raise CalculationError(f"{base.label} is not {what} of {current.label} ({gap} days apart)")
    if base.value <= 0:
        raise CalculationError(
            f"{kind.upper()} growth of {current.metric} is not meaningful: the base ({base.label}) is {_fmt(base)}"
        )
    return MetricResult(
        name=f"{current.metric}_{kind}_growth",
        period_label=current.label,
        period_end=current.period_end,
        value=(current.value - base.value) / base.value,
        unit="ratio",
        formula=f"({_fmt(current)} - {_fmt(base)}) / {_fmt(base)}",
        inputs=(base, current),
    )


def qoq_growth(series: list[QuarterValue]) -> list[MetricResult]:
    """Growth vs the previous quarter, for every quarter whose previous quarter is in the series."""
    return _pairwise(series, "qoq")


def yoy_growth(series: list[QuarterValue]) -> list[MetricResult]:
    """Growth vs the same quarter a year earlier (removes seasonality, e.g. Apple's holiday quarter)."""
    return _pairwise(series, "yoy")


def _pairwise(series: list[QuarterValue], kind: Literal["qoq", "yoy"]) -> list[MetricResult]:
    """Match each quarter with its base BY DATE, not by list position. Unmatched quarters are skipped."""
    expected = _NEXT_QUARTER_DAYS if kind == "qoq" else _NEXT_YEAR_DAYS
    out = []
    for current in series:
        base = next((b for b in series if (current.period_end - b.period_end).days in expected), None)
        if base is None:
            continue
        try:
            out.append(growth(current, base, kind))
        except CalculationError:
            continue  # e.g. negative base: leave this quarter out rather than report a misleading number
    return out


# --- ratios and combinations ----------------------------------------------------

def margin(numerator: QuarterValue, revenue: QuarterValue) -> MetricResult:
    """numerator / revenue for the same quarter, e.g. net_income / revenue = net margin."""
    if revenue.metric != "revenue":
        raise CalculationError(f"a margin divides by revenue, not {revenue.metric}")
    _flow(numerator)
    _same_period(numerator, revenue)
    if revenue.value <= 0:
        raise CalculationError(f"margin is not meaningful: revenue for {revenue.label} is {_fmt(revenue)}")
    name = {
        "gross_profit": "gross_margin",
        "operating_income": "operating_margin",
        "net_income": "net_margin",
        "operating_cash_flow": "operating_cash_flow_margin",
    }.get(numerator.metric, f"{numerator.metric}_to_revenue")
    return MetricResult(
        name=name,
        period_label=revenue.label,
        period_end=revenue.period_end,
        value=numerator.value / revenue.value,
        unit="ratio",
        formula=f"{_fmt(numerator)} / {_fmt(revenue)}",
        inputs=(numerator, revenue),
    )


def free_cash_flow(operating_cash_flow: QuarterValue, capital_expenditure: QuarterValue) -> MetricResult:
    """Operating cash flow - capital expenditure: the cash left after investing in the business."""
    if (operating_cash_flow.metric, capital_expenditure.metric) != ("operating_cash_flow", "capital_expenditure"):
        raise CalculationError("free cash flow needs operating_cash_flow and capital_expenditure")
    _same_period(operating_cash_flow, capital_expenditure)
    return MetricResult(
        name="free_cash_flow",
        period_label=operating_cash_flow.label,
        period_end=operating_cash_flow.period_end,
        value=operating_cash_flow.value - capital_expenditure.value,
        unit="USD",
        formula=f"{_fmt(operating_cash_flow)} - {_fmt(capital_expenditure)}",
        inputs=(operating_cash_flow, capital_expenditure),
    )


def trailing_twelve_months(series: list[QuarterValue]) -> MetricResult:
    """Sum of the latest 4 consecutive quarters (TTM). Only for amounts that add up."""
    if len(series) < 4:
        raise CalculationError(f"TTM needs 4 quarters, got {len(series)}")
    last4 = sorted(series, key=lambda q: q.period_end)[-4:]
    for q in last4:
        _flow(q)
        if q.unit != "USD":
            raise CalculationError(f"TTM only adds up amounts in USD, not {q.unit}")
    for a, b in zip(last4, last4[1:]):
        _same_metric(a, b)
        if (b.period_end - a.period_end).days not in _NEXT_QUARTER_DAYS:
            raise CalculationError(f"TTM needs 4 consecutive quarters; {a.label} -> {b.label} has a gap")
    return MetricResult(
        name=f"{last4[-1].metric}_ttm",
        period_label=f"TTM to {last4[-1].label}",
        period_end=last4[-1].period_end,
        value=sum(q.value for q in last4),
        unit="USD",
        formula=" + ".join(_fmt(q) for q in last4),
        inputs=tuple(last4),
    )


def pair_by_period(a: Iterable[QuarterValue], b: Iterable[QuarterValue]) -> list[tuple[QuarterValue, QuarterValue]]:
    """Match two series quarter by quarter (same dates), e.g. net_income with revenue for margins."""
    b_by_end = {q.period_end: q for q in b}
    return [(x, b_by_end[x.period_end]) for x in a if x.period_end in b_by_end]


# --- checks -------------------------------------------------------------------------

def _same_metric(a: QuarterValue, b: QuarterValue) -> None:
    if a.metric != b.metric or a.unit != b.unit:
        raise CalculationError(f"can't compare {a.metric} ({a.unit}) with {b.metric} ({b.unit})")


def _same_period(a: QuarterValue, b: QuarterValue) -> None:
    if (a.period_start, a.period_end) != (b.period_start, b.period_end):
        raise CalculationError(f"{a.metric} for {a.label} and {b.metric} for {b.label} cover different periods")


def _flow(q: QuarterValue) -> None:
    if q.period_start is None:
        raise CalculationError(f"{q.metric} is a value on one day, not an amount over a quarter")


def _fmt(q: QuarterValue) -> str:
    if q.unit == "USD" and abs(q.value) >= 1e9:
        return f"{q.value / 1e9:,.2f}B"
    if q.unit == "USD" and abs(q.value) >= 1e6:
        return f"{q.value / 1e6:,.2f}M"
    return f"{q.value:,.2f}"
