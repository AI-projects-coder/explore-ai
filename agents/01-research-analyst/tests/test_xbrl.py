"""XBRL companyfacts -> clean quarterly series.

Each test builds a tiny, hand-made companyfacts file that reproduces one trap from real SEC data.
The fiscal year in most tests runs Oct 1 2024 -> Sep 30 2025 (like Apple's).
"""

from datetime import date

import pytest

from research_analyst.tools.xbrl import (
    MetricNotAvailable,
    UnknownMetric,
    available_metrics,
    quarterly_series,
)

REV = "RevenueFromContractWithCustomerExcludingAssessedTax"

# Fiscal year 2025: Q1 Oct-Dec, Q2 Jan-Mar, Q3 Apr-Jun, Q4 Jul-Sep
FY_START = "2024-10-01"
Q1_END, Q2_END, Q3_END, FY_END = "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30"
Q2_START, Q3_START, Q4_START = "2025-01-01", "2025-04-01", "2025-07-01"


def row(start, end, val, *, fy=2025, fp="Q1", form="10-Q", filed="2025-01-30", accn="A-Q1"):
    r = {"end": end, "val": val, "fy": fy, "fp": fp, "form": form, "filed": filed, "accn": accn}
    if start is not None:
        r["start"] = start
    return r


def facts(concepts: dict[str, list[dict]], unit: str = "USD") -> dict:
    return {"facts": {"us-gaap": {name: {"units": {unit: rows}} for name, rows in concepts.items()}}}


def full_year_revenue():
    """What a real company files for revenue in one year: 3 10-Qs (quarter + year-to-date) and a 10-K."""
    return [
        row(FY_START, Q1_END, 100, fp="Q1", filed="2025-01-30", accn="A-Q1"),
        row(Q2_START, Q2_END, 120, fp="Q2", filed="2025-04-30", accn="A-Q2"),
        row(FY_START, Q2_END, 220, fp="Q2", filed="2025-04-30", accn="A-Q2"),   # 6 months to date
        row(Q3_START, Q3_END, 130, fp="Q3", filed="2025-07-30", accn="A-Q3"),
        row(FY_START, Q3_END, 350, fp="Q3", filed="2025-07-30", accn="A-Q3"),   # 9 months to date
        row(FY_START, FY_END, 500, fp="FY", form="10-K", filed="2025-10-30", accn="A-K"),
    ]


def by_quarter(series):
    return {q.label: q.value for q in series}


# --- trap 1 + 3: year-to-date totals are not quarters; Q4 must be derived ------

def test_keeps_quarters_and_derives_q4():
    s = quarterly_series(facts({REV: full_year_revenue()}), "revenue", quarters=4)
    assert by_quarter(s) == {"Q1 FY2025": 100, "Q2 FY2025": 120, "Q3 FY2025": 130, "Q4 FY2025": 150}
    q4 = s[-1]
    assert q4.source == "derived"
    assert (q4.period_start, q4.period_end) == (date(2025, 7, 1), date(2025, 9, 30))
    assert "derived" in q4.note
    assert all(q.source == "reported" for q in s[:3])


def test_series_is_oldest_first_and_limited():
    s = quarterly_series(facts({REV: full_year_revenue()}), "revenue", quarters=2)
    assert [q.label for q in s] == ["Q3 FY2025", "Q4 FY2025"]


# --- trap 3b: cash flow is often reported ONLY as year-to-date ------------------

def test_derives_every_missing_quarter_from_year_to_date_totals():
    ocf = [
        row(FY_START, Q1_END, 30, fp="Q1", accn="A-Q1"),
        row(FY_START, Q2_END, 70, fp="Q2", filed="2025-04-30", accn="A-Q2"),
        row(FY_START, Q3_END, 105, fp="Q3", filed="2025-07-30", accn="A-Q3"),
        row(FY_START, FY_END, 150, fp="FY", form="10-K", filed="2025-10-30", accn="A-K"),
    ]
    s = quarterly_series(facts({"NetCashProvidedByUsedInOperatingActivities": ocf}), "operating_cash_flow")
    assert by_quarter(s) == {"Q1 FY2025": 30, "Q2 FY2025": 40, "Q3 FY2025": 35, "Q4 FY2025": 45}
    assert [q.source for q in s] == ["reported", "derived", "derived", "derived"]


def test_derives_q4_by_adding_quarters_when_no_nine_month_total():
    rows = [
        row(FY_START, Q1_END, 100, fp="Q1", accn="A-Q1"),
        row(Q2_START, Q2_END, 120, fp="Q2", filed="2025-04-30", accn="A-Q2"),
        row(Q3_START, Q3_END, 130, fp="Q3", filed="2025-07-30", accn="A-Q3"),
        row(FY_START, FY_END, 500, fp="FY", form="10-K", filed="2025-10-30", accn="A-K"),
    ]
    s = quarterly_series(facts({REV: rows}), "revenue")
    assert by_quarter(s)["Q4 FY2025"] == 150


def test_no_q4_when_the_pieces_do_not_add_up_to_a_quarter():
    # Q2 is missing entirely: FY - Q1 would be a 9-month value, not a quarter. Leave the gap.
    rows = [
        row(FY_START, Q1_END, 100, fp="Q1", accn="A-Q1"),
        row(Q3_START, Q3_END, 130, fp="Q3", filed="2025-07-30", accn="A-Q3"),
        row(FY_START, FY_END, 500, fp="FY", form="10-K", filed="2025-10-30", accn="A-K"),
    ]
    s = quarterly_series(facts({REV: rows}), "revenue")
    assert all(q.source == "reported" for q in s)
    assert len(s) == 2


def test_reported_quarter_is_never_replaced_by_a_derived_one():
    rows = full_year_revenue() + [
        row(Q4_START, FY_END, 149, fp="FY", form="10-K", filed="2025-10-30", accn="A-K"),  # Q4 reported directly
    ]
    q4 = quarterly_series(facts({REV: rows}), "revenue")[-1]
    assert (q4.value, q4.source) == (149, "reported")


def test_eps_is_never_derived():
    # EPS doesn't add up across quarters (share count changes), so FY - 9M EPS would be wrong
    rows = [
        row(FY_START, Q1_END, 1.0, fp="Q1", accn="A-Q1"),
        row(FY_START, Q3_END, 3.1, fp="Q3", filed="2025-07-30", accn="A-Q3"),
        row(FY_START, FY_END, 4.0, fp="FY", form="10-K", filed="2025-10-30", accn="A-K"),
    ]
    s = quarterly_series(facts({"EarningsPerShareDiluted": rows}, unit="USD/shares"), "eps_diluted")
    assert [q.label for q in s] == ["Q1 FY2025"]


# --- trap 2: numbers are repeated in later filings, with the later filing's labels ---

def test_repeated_number_appears_once_with_the_original_label():
    rows = full_year_revenue() + [
        # next year's Q1 10-Q repeats last year's Q1 for comparison, labeled fy=2026
        row(FY_START, Q1_END, 100, fy=2026, fp="Q1", filed="2026-01-30", accn="B-Q1"),
        row("2025-10-01", "2025-12-31", 110, fy=2026, fp="Q1", filed="2026-01-30", accn="B-Q1"),
    ]
    s = quarterly_series(facts({REV: rows}), "revenue", quarters=10)
    labels = [q.label for q in s]
    assert labels.count("Q1 FY2025") == 1
    assert "Q1 FY2026" in labels
    assert len(s) == 5


def test_restated_value_wins_but_label_comes_from_the_original_filing():
    rows = full_year_revenue() + [
        row(FY_START, Q1_END, 98, fy=2026, fp="Q1", filed="2026-01-30", accn="B-Q1"),  # restated Q1
        row("2025-10-01", "2025-12-31", 110, fy=2026, fp="Q1", filed="2026-01-30", accn="B-Q1"),
    ]
    q1 = next(q for q in quarterly_series(facts({REV: rows}), "revenue", 10) if q.period_end == date(2024, 12, 31))
    assert q1.value == 98            # newest filing's number
    assert q1.label == "Q1 FY2025"   # not "Q1 FY2026"
    assert q1.accession == "B-Q1"    # citation points at where the value came from


def test_missing_fy_gives_a_date_label_instead_of_crashing():
    rows = [row(FY_START, Q1_END, 100, fy=None)]
    s = quarterly_series(facts({REV: rows}), "revenue")
    assert s[0].label == "quarter ending 2024-12-31"


# --- trap 4: concepts get renamed --------------------------------------------------

def test_falls_back_to_older_concept_names_across_a_rename():
    # the company used "Revenues" until FY2024 Q4, then switched to the new name
    old = [row("2024-07-01", "2024-09-30", 90, fy=2024, fp="Q4", form="10-K", accn="OLD")]
    new = [row(FY_START, Q1_END, 100, accn="A-Q1")]
    s = quarterly_series(facts({REV: new, "Revenues": old}), "revenue", quarters=10)
    assert [(q.value, q.concept) for q in s] == [(90, "Revenues"), (100, REV)]


def test_preferred_concept_wins_when_both_report_the_same_period():
    s = quarterly_series(facts({
        REV: [row(FY_START, Q1_END, 100)],
        "Revenues": [row(FY_START, Q1_END, 105)],
    }), "revenue")
    assert (s[0].value, s[0].concept) == (100, REV)


# --- trap 5: snapshots (one day) ------------------------------------------------------

def test_snapshot_values_and_opening_balances():
    rows = [
        # Q1 10-Q: balance sheet at quarter end (its own date) + prior year-end for comparison
        {"end": Q1_END, "val": 400, "fy": 2025, "fp": "Q1", "form": "10-Q", "filed": "2025-01-30", "accn": "A-Q1"},
        {"end": "2024-09-30", "val": 380, "fy": 2025, "fp": "Q1", "form": "10-Q", "filed": "2025-01-30", "accn": "A-Q1"},
        # the year-end balance was first reported in last year's 10-K
        {"end": "2024-09-30", "val": 380, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-10-30", "accn": "Z-K"},
    ]
    s = quarterly_series(facts({"Assets": rows}), "total_assets")
    assert [(q.label, q.value, q.period_start) for q in s] == [("Q4 FY2024", 380, None), ("Q1 FY2025", 400, None)]


def test_opening_balance_alone_is_not_mislabeled():
    # a date that only appears as an earlier date inside a filing gets a date label, not the filing's fp
    rows = [
        {"end": Q1_END, "val": 400, "fy": 2025, "fp": "Q1", "form": "10-Q", "filed": "2025-01-30", "accn": "A-Q1"},
        {"end": "2023-09-30", "val": 300, "fy": 2025, "fp": "Q1", "form": "10-Q", "filed": "2025-01-30", "accn": "A-Q1"},
    ]
    s = quarterly_series(facts({"Assets": rows}), "total_assets")
    assert s[0].label == "quarter ending 2023-09-30"


# --- trap 6: stale data ------------------------------------------------------------------

def test_refuses_stale_metric():
    # revenue stopped in 2014 (renamed to something we don't know); the company files to 2025
    stale = [row("2014-01-01", "2014-03-31", 23, fy=2014, accn="OLD")]
    current = [row(FY_START, Q1_END, 999, accn="A-Q1")]
    with pytest.raises(MetricNotAvailable, match="2014"):
        quarterly_series(facts({REV: stale, "SomeOtherConcept": current}), "revenue")


# --- messy input ------------------------------------------------------------------------

def test_ignores_non_periodic_forms_and_malformed_rows():
    rows = full_year_revenue() + [
        row(FY_START, Q1_END, 999, form="8-K", filed="2025-02-01", accn="X"),   # not a 10-Q/10-K
        {"start": FY_START, "end": "not-a-date", "val": 1, "form": "10-Q", "filed": "2025-01-30"},
        {"start": FY_START, "end": Q1_END, "form": "10-Q", "filed": "2025-01-30"},  # no value
        "garbage",
    ]
    s = quarterly_series(facts({REV: rows}), "revenue")
    assert by_quarter(s)["Q1 FY2025"] == 100


def test_accepts_14_week_quarters():
    # 52/53-week fiscal years: some quarters are 98 days
    s = quarterly_series(facts({REV: [row("2024-09-29", "2025-01-04", 100)]}), "revenue")
    assert s[0].value == 100


@pytest.mark.parametrize("data", [{}, {"facts": None}, {"facts": {"us-gaap": None}}, facts({})])
def test_missing_data_is_metric_not_available(data):
    with pytest.raises(MetricNotAvailable):
        quarterly_series(data, "revenue")


def test_unknown_metric_lists_the_valid_ones():
    with pytest.raises(UnknownMetric, match="revenue"):
        quarterly_series(facts({}), "ebitda")


@pytest.mark.parametrize("n", [0, 41])
def test_quarters_out_of_range(n):
    with pytest.raises(ValueError):
        quarterly_series(facts({REV: full_year_revenue()}), "revenue", quarters=n)


def test_available_metrics():
    assert {"revenue", "net_income", "total_assets", "eps_diluted"} <= set(available_metrics())
