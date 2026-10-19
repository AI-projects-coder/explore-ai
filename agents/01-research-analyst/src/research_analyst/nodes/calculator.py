"""Calculator node: findings -> metrics, via finance.metrics only.

No LLM. It turns every number the writer may use into a MetricResult with an id ("m1", "m2", ...):
the reported values themselves, plus growth, margins, free cash flow and trailing-twelve-months.
The writer may only use numbers from this table, and must cite their ids.
"""

from research_analyst.finance import metrics as fm
from research_analyst.finance.metrics import CalculationError, MetricResult
from research_analyst.state import AgentState
from research_analyst.tools.xbrl import QuarterValue

YOY_GROWTH = ("revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow", "eps_diluted")
QOQ_GROWTH = ("revenue", "net_income")
MARGINS = ("gross_profit", "operating_income", "net_income")
TTM = ("revenue", "net_income", "operating_cash_flow")


def calculator(state: AgentState) -> dict:
    plan = state.get("plan")
    if plan is None:
        return {}
    series = _series_by_metric(state.get("findings", []))
    if not series:
        if any(t.kind == "financials" for t in plan.tasks):
            return {"metrics": {}, "errors": ["No financial data was available, so no numbers were calculated."]}
        return {"metrics": {}}

    # Results are only kept for the quarters the user asked about; older data was fetched for YoY bases.
    in_window = _window(series, plan.quarters)
    results: list[MetricResult] = []

    for metric, qs in series.items():
        results += [_reported(q) for q in qs if q.period_end in in_window[metric]]
    for metric in YOY_GROWTH:
        results += _keep(fm.yoy_growth(series.get(metric, [])), in_window.get(metric))
    for metric in QOQ_GROWTH:
        results += _keep(fm.qoq_growth(series.get(metric, [])), in_window.get(metric))
    revenue = series.get("revenue", [])
    for metric in MARGINS:
        for num, rev in fm.pair_by_period(series.get(metric, []), revenue):
            if rev.period_end in in_window["revenue"]:
                results += _try(fm.margin, num, rev)
    for ocf, capex in fm.pair_by_period(series.get("operating_cash_flow", []), series.get("capital_expenditure", [])):
        if ocf.period_end in in_window["operating_cash_flow"]:
            results += _try(fm.free_cash_flow, ocf, capex)
    for metric in TTM:
        if metric in series:
            results += _try(fm.trailing_twelve_months, series[metric])

    # Stable ids: same data -> same ids, so a re-run (retry, duplicate delivery) produces the same table.
    results.sort(key=lambda r: (r.name, r.period_end))
    return {"metrics": {f"m{i}": r for i, r in enumerate(results, start=1)}}


def display(r: MetricResult) -> tuple[str, float]:
    """How a metric is shown in the report, and the number as written: ('+16.4%', 16.4), ('$109.42B', 109.42)."""
    if r.unit == "ratio":
        n = round(r.value * 100, 1)
        sign = "+" if r.name.endswith("_growth") and n > 0 else ""
        return f"{sign}{n:.1f}%", n
    if r.unit == "USD/shares":
        n = round(r.value, 2)
        return f"{'-' if n < 0 else ''}${abs(n):,.2f}", n
    for divisor, suffix in ((1e9, "B"), (1e6, "M")):
        if abs(r.value) >= divisor:
            n = round(r.value / divisor, 2)
            return f"{'-' if n < 0 else ''}${abs(n):,.2f}{suffix}", n
    n = round(r.value)
    return f"{'-' if n < 0 else ''}${abs(n):,}", n


def _series_by_metric(findings) -> dict[str, list[QuarterValue]]:
    """All QuarterValues from all findings, one per (metric, quarter), oldest first."""
    by_key: dict[tuple[str, object], QuarterValue] = {}
    for f in findings:
        for q in f.data:
            by_key[(q.metric, q.period_end)] = q
    series: dict[str, list[QuarterValue]] = {}
    for q in sorted(by_key.values(), key=lambda q: q.period_end):
        series.setdefault(q.metric, []).append(q)
    return series


def _window(series: dict[str, list[QuarterValue]], quarters: int) -> dict[str, set]:
    return {metric: {q.period_end for q in qs[-quarters:]} for metric, qs in series.items()}


def _reported(q: QuarterValue) -> MetricResult:
    return MetricResult(
        name=q.metric,
        period_label=q.label,
        period_end=q.period_end,
        value=q.value,
        unit=q.unit,
        formula=q.note or f"reported in {q.form} ({q.accession})",
        inputs=(q,),
    )


def _keep(results: list[MetricResult], window: set | None) -> list[MetricResult]:
    return [r for r in results if window and r.period_end in window]


def _try(fn, *args) -> list[MetricResult]:
    """A calculation that can't produce a meaningful number is left out, not reported as zero."""
    try:
        return [fn(*args)]
    except CalculationError:
        return []
