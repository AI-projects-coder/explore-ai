"""Turns raw XBRL companyfacts JSON into clean quarterly series (revenue, net income, ...).

SEC's data is not a table of quarters. The traps this module handles:
  1. Filings report "this quarter" AND "year to date" totals -> a quarter is a ~3-month period.
  2. Every number is repeated in later filings, labeled with the *later* filing's fy/fp -> identify a
     period by its dates; take the value from the latest filing (restatements win); take the label
     only from the filing that was *about* that period (its latest date).
  3. Some quarters are never reported on their own (Q4 always; for cash flow often Q2-Q4)
     -> derive them from year-to-date totals: Q3 = 9 months - 6 months, Q4 = full year - 9 months.
     Only for amounts that add up (not EPS).
  4. Concepts get renamed, and banks use different names -> try several names per metric.
  5. Flows (revenue: over a period) vs snapshots (assets: on one day) are handled differently.
  6. A metric can silently stop being reported (renamed) -> refuse stale data instead of
     presenting a 2014 number as the latest.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Period lengths in days. Fiscal quarters are 12-14 weeks (52/53-week years), so use ranges.
_DURATIONS = {  # number of quarters a period covers -> allowed lengths
    1: range(80, 101),
    2: range(170, 194),
    3: range(260, 291),
    4: range(350, 381),
}
_ALLOWED_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}
_STALE_AFTER_DAYS = 400  # metric's newest data this much older than the company's newest filing = stale


@dataclass(frozen=True)
class Metric:
    kind: Literal["flow", "snapshot"]  # flow: adds up over a period; snapshot: value on one day
    concepts: tuple[str, ...]  # XBRL names to try, in priority order
    unit: str = "USD"
    derivable: bool = True  # quarter = difference of year-to-date totals; only for amounts that add up


METRICS: dict[str, Metric] = {
    "revenue": Metric("flow", (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenuesNetOfInterestExpense",  # banks
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    )),
    "gross_profit": Metric("flow", ("GrossProfit",)),
    "operating_income": Metric("flow", ("OperatingIncomeLoss",)),
    "net_income": Metric("flow", ("NetIncomeLoss", "ProfitLoss")),
    "operating_cash_flow": Metric("flow", (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    )),
    "capital_expenditure": Metric("flow", ("PaymentsToAcquirePropertyPlantAndEquipment",)),
    # EPS does NOT add up across quarters (the share count changes), so missing quarters stay missing.
    "eps_diluted": Metric("flow", ("EarningsPerShareDiluted",), unit="USD/shares", derivable=False),
    "total_assets": Metric("snapshot", ("Assets",)),
    "cash": Metric("snapshot", ("CashAndCashEquivalentsAtCarryingValue", "CashAndDueFromBanks")),
    "long_term_debt": Metric("snapshot", ("LongTermDebtNoncurrent", "LongTermDebt")),
    "shareholders_equity": Metric("snapshot", ("StockholdersEquity",)),
}


class XbrlError(Exception):
    """Base class for problems turning companyfacts into a series."""


class UnknownMetric(XbrlError):
    """The caller (usually the LLM) asked for a metric we don't define."""


class MetricNotAvailable(XbrlError):
    """The company doesn't report this metric (or stopped reporting it under the names we know)."""


class QuarterValue(BaseModel):
    """One clean number for one fiscal quarter, with where it came from."""

    model_config = ConfigDict(frozen=True)

    metric: str
    fiscal_year: int | None  # None when no filing labeled this period
    fiscal_quarter: int | None  # 1-4
    period_start: date | None  # None for snapshots (a value on one day)
    period_end: date
    value: float
    unit: str
    source: Literal["reported", "derived"]
    concept: str
    form: str
    accession: str | None
    filed: date
    note: str | None = None  # how a derived value was computed

    @property
    def label(self) -> str:
        if self.fiscal_year is None or self.fiscal_quarter is None:
            return f"quarter ending {self.period_end.isoformat()}"
        return f"Q{self.fiscal_quarter} FY{self.fiscal_year}"


def available_metrics() -> list[str]:
    return sorted(METRICS)


def quarterly_series(facts: dict[str, Any], metric: str, quarters: int = 4) -> list[QuarterValue]:
    """The last `quarters` fiscal quarters of `metric`, oldest first.

    A quarter that can't be reported or safely derived is left out (e.g. Q4 EPS),
    so callers comparing neighbours must check the dates, not just the list positions.
    """
    spec = METRICS.get(metric)
    if spec is None:
        raise UnknownMetric(f"unknown metric {metric[:40]!r}; available: {', '.join(available_metrics())}")
    if not 1 <= quarters <= 40:
        raise ValueError("quarters must be between 1 and 40")

    us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    rows = list(_parse_rows(us_gaap, spec))
    if not rows:
        raise MetricNotAvailable(f"the company does not report {metric}")

    series = _flow_quarters(metric, spec, rows) if spec.kind == "flow" else _snapshot_quarters(metric, spec, rows)
    if not series:
        raise MetricNotAvailable(f"no quarterly {metric} data could be built")
    series.sort(key=lambda q: q.period_end)

    newest_filing = _latest_period_end(us_gaap)
    if newest_filing and (newest_filing - series[-1].period_end).days > _STALE_AFTER_DAYS:
        raise MetricNotAvailable(
            f"latest {metric} is for {series[-1].period_end}, but the company has filed data up to "
            f"{newest_filing}; it probably reports {metric} under a name we don't know"
        )
    return series[-quarters:]


# --- parsing ------------------------------------------------------------------

@dataclass(frozen=True)
class _Row:
    priority: int  # index of the concept in Metric.concepts (0 = preferred name)
    concept: str
    start: date | None
    end: date
    value: float
    form: str
    filed: date
    accession: str | None
    fy: int | None
    fp: str | None
    own_period: bool = False  # True when this row is about the period its filing covers

    @property
    def quarters(self) -> int | None:
        """How many quarters this period spans (1-4), or None if it's no standard length."""
        if self.start is None:
            return None
        days = (self.end - self.start).days + 1
        return next((n for n, r in _DURATIONS.items() if days in r), None)


def _parse_rows(us_gaap: dict[str, Any], spec: Metric) -> list[_Row]:
    """Every usable row of every concept for this metric. Malformed rows are skipped, not fatal."""
    rows: list[_Row] = []
    for priority, concept in enumerate(spec.concepts):
        units = (us_gaap.get(concept) or {}).get("units") or {}
        for raw in units.get(spec.unit) or []:
            try:
                if not isinstance(raw, dict) or raw.get("form") not in _ALLOWED_FORMS:
                    continue
                start = date.fromisoformat(raw["start"]) if raw.get("start") else None
                if (start is None) != (spec.kind == "snapshot"):
                    continue  # a flow needs a period; a snapshot must not have one
                rows.append(_Row(
                    priority=priority,
                    concept=concept,
                    start=start,
                    end=date.fromisoformat(raw["end"]),
                    value=raw["val"],
                    form=raw["form"],
                    filed=date.fromisoformat(raw["filed"]),
                    accession=raw.get("accn"),
                    fy=raw.get("fy") if isinstance(raw.get("fy"), int) else None,
                    fp=raw.get("fp"),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return _mark_own_periods(rows)


def _mark_own_periods(rows: list[_Row]) -> list[_Row]:
    """A filing is 'about' its latest date; every earlier date in it is a comparison or opening balance.

    Only rows about their filing's own period carry a trustworthy fy/fp label.
    """
    latest: dict[str, date] = {}
    for r in rows:
        if r.accession:
            latest[r.accession] = max(latest.get(r.accession, r.end), r.end)
    return [
        _Row(**{**r.__dict__, "own_period": r.accession is not None and r.end == latest[r.accession]})
        for r in rows
    ]


def _latest_period_end(us_gaap: dict[str, Any]) -> date | None:
    """The newest period end in any 10-Q/10-K, across every concept: how current the company's data is."""
    newest = ""
    for concept in us_gaap.values():
        for unit_rows in ((concept or {}).get("units") or {}).values():
            for raw in unit_rows or []:
                if not isinstance(raw, dict):
                    continue
                end = str(raw.get("end") or "")
                if raw.get("form") in _ALLOWED_FORMS and end > newest:  # ISO dates sort as strings
                    newest = end
    try:
        return date.fromisoformat(newest) if newest else None
    except ValueError:
        return None


@dataclass(frozen=True)
class _Fact:
    """Every filing that reported one period, collapsed into one number."""

    value_row: _Row  # the latest filing: restatements win
    label_row: _Row | None  # the filing that was about this period, if any

    @property
    def fiscal_year(self) -> int | None:
        return self.label_row.fy if self.label_row else None

    @property
    def fiscal_quarter(self) -> int | None:
        fp = self.label_row.fp if self.label_row else None
        return {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}.get(fp or "")


def _collapse(rows: Iterable[_Row], key) -> dict[Any, _Fact]:
    """Group rows by period. Within a period, only the preferred concept counts."""
    groups: dict[Any, list[_Row]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = {}
    for k, group in groups.items():
        best = min(r.priority for r in group)
        group = [r for r in group if r.priority == best]
        own = [r for r in group if r.own_period]
        out[k] = _Fact(
            value_row=max(group, key=lambda r: r.filed),
            label_row=min(own, key=lambda r: r.filed) if own else None,
        )
    return out


# --- flows --------------------------------------------------------------------

def _flow_quarters(metric: str, spec: Metric, rows: list[_Row]) -> list[QuarterValue]:
    periods = _collapse((r for r in rows if r.quarters), key=lambda r: (r.start, r.end))
    single = {k: f for k, f in periods.items() if f.value_row.quarters == 1}

    out = {end: _to_value(metric, spec, f) for (_, end), f in single.items()}
    if spec.derivable:
        for q in _derived_quarters(metric, spec, periods, single):
            out.setdefault(q.period_end, q)  # a reported quarter always beats a derived one
    return list(out.values())


def _derived_quarters(metric, spec, periods: dict, single: dict) -> Iterable[QuarterValue]:
    """Fill quarters that were only reported inside year-to-date totals.

    For one fiscal year (all periods starting on the same day), line up the running totals by end date:
        3 months: 30   6 months: 70   9 months: 105   12 months: 150
    and each quarter is the difference between neighbours: Q2 = 40, Q3 = 35, Q4 = 45.
    """
    year_starts = {s for (s, _), f in periods.items() if f.value_row.quarters > 1}
    single_by_start = {s: f for (s, _), f in single.items()}

    for year_start in year_starts:
        # running total -> (value, fact it came from); reported year-to-date rows first
        totals: dict[date, tuple[float, _Fact]] = {
            e: (f.value_row.value, f) for (s, e), f in periods.items() if s == year_start
        }
        # where a year-to-date row is missing, add up consecutive reported quarters instead
        running, cursor = 0.0, year_start
        while (q := single_by_start.get(cursor)) is not None:
            running += q.value_row.value
            totals.setdefault(q.value_row.end, (running, q))
            cursor = q.value_row.end + timedelta(days=1)

        prev_end, prev_total = year_start - timedelta(days=1), 0.0
        for end in sorted(totals):
            total, fact = totals[end]
            start = prev_end + timedelta(days=1)
            if (end - start).days + 1 in _DURATIONS[1] and (start, end) not in single:
                yield _to_value(
                    metric, spec, fact, start=start, value=total - prev_total,
                    note=f"derived: total to {end} ({_short(total)}) - total to {prev_end} ({_short(prev_total)})",
                )
            prev_end, prev_total = end, total


# --- snapshots ------------------------------------------------------------------

def _snapshot_quarters(metric: str, spec: Metric, rows: list[_Row]) -> list[QuarterValue]:
    return [_to_value(metric, spec, f) for f in _collapse(rows, key=lambda r: r.end).values()]


# --- shared ---------------------------------------------------------------------

def _to_value(metric: str, spec: Metric, fact: _Fact, *, start: date | None = None,
              value: float | None = None, note: str | None = None) -> QuarterValue:
    """Reported value by default; pass start/value/note for a derived quarter."""
    v = fact.value_row
    return QuarterValue(
        metric=metric,
        fiscal_year=fact.fiscal_year,
        fiscal_quarter=fact.fiscal_quarter,
        period_start=start if start is not None else v.start,
        period_end=v.end,
        value=value if value is not None else v.value,
        unit=spec.unit,
        source="derived" if note else "reported",
        concept=v.concept,
        form=v.form,
        accession=v.accession,
        filed=v.filed,
        note=note,
    )


def _short(value: float) -> str:
    return f"{value / 1e9:,.1f}B" if abs(value) >= 1e9 else f"{value:,.0f}"
