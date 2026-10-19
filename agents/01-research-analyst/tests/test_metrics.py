"""Finance calculations: correct math, and refusing to produce misleading numbers."""

from datetime import date, timedelta

import pytest

from research_analyst.finance.metrics import (
    CalculationError,
    free_cash_flow,
    growth,
    margin,
    pair_by_period,
    qoq_growth,
    trailing_twelve_months,
    yoy_growth,
)
from research_analyst.tools.xbrl import QuarterValue


def qv(metric, end, value, *, quarter=1, year=2025, unit="USD", snapshot=False, days=91):
    end = date.fromisoformat(end)
    return QuarterValue(
        metric=metric, fiscal_year=year, fiscal_quarter=quarter,
        period_start=None if snapshot else end - timedelta(days=days - 1), period_end=end,
        value=value, unit=unit, source="reported", concept="X", form="10-Q",
        accession=f"acc-{metric}-{end}", filed=end + timedelta(days=30),
    )


def year_of(metric, values, *, start_year=2024, unit="USD"):
    """Consecutive calendar quarters starting Q1 of start_year."""
    ends = ["03-31", "06-30", "09-30", "12-31"]
    return [
        qv(metric, f"{start_year + i // 4}-{ends[i % 4]}", v, quarter=i % 4 + 1, year=start_year + i // 4, unit=unit)
        for i, v in enumerate(values)
    ]


# --- growth -----------------------------------------------------------------------

def test_qoq_growth_math_and_traceability():
    prev, cur = qv("revenue", "2025-03-31", 100), qv("revenue", "2025-06-30", 110, quarter=2)
    r = growth(cur, prev, "qoq")
    assert r.value == pytest.approx(0.10)
    assert r.name == "revenue_qoq_growth"
    assert r.unit == "ratio"
    assert r.inputs == (prev, cur)          # the exact inputs, with their SEC accession numbers
    assert r.formula == "(110.00 - 100.00) / 100.00"


def test_negative_growth():
    r = growth(qv("revenue", "2025-06-30", 90), qv("revenue", "2025-03-31", 100), "qoq")
    assert r.value == pytest.approx(-0.10)


def test_yoy_growth_matches_same_quarter_last_year():
    s = year_of("revenue", [100, 50, 60, 70, 120, 55, 66, 77])
    results = yoy_growth(s)
    assert [r.period_end for r in results] == [q.period_end for q in s[4:]]
    assert [round(r.value, 4) for r in results] == [0.2, 0.1, 0.1, 0.1]


def test_qoq_rejects_quarters_that_are_not_neighbours():
    # Q1 and Q3: Q2 missing. Comparing them would be "half-year growth" mislabeled as QoQ.
    with pytest.raises(CalculationError, match="not the previous quarter"):
        growth(qv("revenue", "2025-09-30", 110), qv("revenue", "2025-03-31", 100), "qoq")


def test_series_growth_pairs_by_date_and_skips_gaps():
    s = year_of("revenue", [100, 110, 120, 130])
    del s[2]  # Q3 missing, like Apple's Q4 EPS
    results = qoq_growth(s)
    assert [(r.inputs[0].period_end, r.inputs[1].period_end) for r in results] == [
        (date(2024, 3, 31), date(2024, 6, 30)),  # Q1 -> Q2 only; Q2 -> Q4 is NOT a quarter
    ]


def test_yoy_rejects_wrong_distance():
    with pytest.raises(CalculationError, match="same quarter a year earlier"):
        growth(qv("revenue", "2025-06-30", 110), qv("revenue", "2025-03-31", 100), "yoy")


@pytest.mark.parametrize("base", [0, -50])
def test_growth_from_zero_or_negative_base_is_refused(base):
    # net income from -50 to +10 is not "+120%": the sign flip makes the percentage meaningless
    with pytest.raises(CalculationError, match="not meaningful"):
        growth(qv("net_income", "2025-06-30", 10), qv("net_income", "2025-03-31", base), "qoq")


def test_series_growth_skips_meaningless_quarters():
    s = year_of("net_income", [-50, 10, 20])
    assert [r.period_end for r in qoq_growth(s)] == [date(2024, 9, 30)]


def test_growth_refuses_mixed_metrics():
    with pytest.raises(CalculationError, match="can't compare"):
        growth(qv("revenue", "2025-06-30", 10), qv("net_income", "2025-03-31", 5), "qoq")


def test_growth_accepts_14_week_quarters():
    r = growth(qv("revenue", "2025-07-05", 110), qv("revenue", "2025-03-29", 100), "qoq")  # 98 days
    assert r.value == pytest.approx(0.10)


# --- margins ------------------------------------------------------------------------

@pytest.mark.parametrize("metric,name", [
    ("gross_profit", "gross_margin"),
    ("operating_income", "operating_margin"),
    ("net_income", "net_margin"),
])
def test_margins(metric, name):
    r = margin(qv(metric, "2025-06-30", 25), qv("revenue", "2025-06-30", 100))
    assert (r.name, r.value, r.unit) == (name, 0.25, "ratio")


def test_negative_margin_is_allowed():
    # a loss-making quarter has a negative margin; that's a real, meaningful number
    assert margin(qv("net_income", "2025-06-30", -10), qv("revenue", "2025-06-30", 100)).value == -0.10


def test_margin_refuses_different_periods():
    with pytest.raises(CalculationError, match="different periods"):
        margin(qv("net_income", "2025-06-30", 25), qv("revenue", "2025-03-31", 100))


def test_margin_refuses_zero_revenue():
    with pytest.raises(CalculationError, match="not meaningful"):
        margin(qv("net_income", "2025-06-30", 25), qv("revenue", "2025-06-30", 0))


def test_margin_must_divide_by_revenue():
    with pytest.raises(CalculationError, match="revenue"):
        margin(qv("net_income", "2025-06-30", 25), qv("gross_profit", "2025-06-30", 100))


def test_margin_refuses_snapshot_numerator():
    with pytest.raises(CalculationError, match="one day"):
        margin(qv("total_assets", "2025-06-30", 500, snapshot=True), qv("revenue", "2025-06-30", 100))


# --- free cash flow -------------------------------------------------------------------

def test_free_cash_flow():
    r = free_cash_flow(qv("operating_cash_flow", "2025-06-30", 30e9), qv("capital_expenditure", "2025-06-30", 2e9))
    assert (r.value, r.unit) == (28e9, "USD")
    assert r.formula == "30.00B - 2.00B"


def test_free_cash_flow_refuses_swapped_or_wrong_inputs():
    with pytest.raises(CalculationError):
        free_cash_flow(qv("capital_expenditure", "2025-06-30", 2), qv("operating_cash_flow", "2025-06-30", 30))


def test_free_cash_flow_refuses_different_periods():
    with pytest.raises(CalculationError, match="different periods"):
        free_cash_flow(qv("operating_cash_flow", "2025-06-30", 30), qv("capital_expenditure", "2025-03-31", 2))


# --- trailing twelve months -------------------------------------------------------------

def test_ttm_sums_latest_four_consecutive_quarters():
    r = trailing_twelve_months(year_of("revenue", [1, 10, 20, 30, 40]))
    assert r.value == 100
    assert [q.value for q in r.inputs] == [10, 20, 30, 40]
    assert r.name == "revenue_ttm"


def test_ttm_refuses_gaps():
    s = year_of("revenue", [10, 20, 30, 40, 50])
    del s[3]
    with pytest.raises(CalculationError, match="consecutive"):
        trailing_twelve_months(s)


def test_ttm_refuses_too_few_quarters():
    with pytest.raises(CalculationError, match="4 quarters"):
        trailing_twelve_months(year_of("revenue", [1, 2, 3]))


def test_ttm_refuses_eps_and_snapshots():
    with pytest.raises(CalculationError, match="USD"):
        trailing_twelve_months(year_of("eps_diluted", [1, 1, 1, 1], unit="USD/shares"))
    assets = [qv("total_assets", e, 5, snapshot=True) for e in ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]]
    with pytest.raises(CalculationError, match="one day"):
        trailing_twelve_months(assets)


# --- pairing -------------------------------------------------------------------------------

def test_pair_by_period_matches_dates_not_positions():
    ni = year_of("net_income", [1, 2, 3])
    rev = year_of("revenue", [10, 20, 30])[1:]  # revenue for Q1 missing
    assert [(a.value, b.value) for a, b in pair_by_period(ni, rev)] == [(2, 20), (3, 30)]
