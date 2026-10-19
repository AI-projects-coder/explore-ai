"""8-K listing, HTML-to-text, and the SEC client's document download."""

from datetime import date

import httpx
import pytest

from research_analyst.core.config import Settings
from research_analyst.tools.edgar import EdgarClient, InvalidInput
from research_analyst.tools.filings import (
    Filing8K,
    filing_text,
    html_to_text,
    lookback_start,
    recent_8ks,
    strip_cover_page,
)


def submissions(*rows):
    cols = ("form", "accessionNumber", "filingDate", "items", "primaryDocument")
    return {"filings": {"recent": {c: [r[i] for r in rows] for i, c in enumerate(cols)}}}


SUBS = submissions(
    ("10-Q", "0000320193-26-000020", "2026-08-01", "", "q.htm"),
    ("8-K", "0000320193-26-000018", "2026-07-30", "2.02,9.01", "earn.htm"),
    ("8-K/A", "0001140361-26-035325", "2026-09-01", "5.02", "ceo_a.htm"),
    ("8-K", "0000320193-25-000001", "2025-01-30", "2.02,9.01", "old.htm"),
)


# --- recent_8ks -----------------------------------------------------------------------

def test_lists_only_8ks_in_the_window_newest_first():
    out = recent_8ks(SUBS, 320193, since=date(2026, 1, 1), limit=10)
    assert [f.accession for f in out] == ["0001140361-26-035325", "0000320193-26-000018"]  # amendments count too


def test_limit():
    assert len(recent_8ks(SUBS, 320193, since=date(2020, 1, 1), limit=2)) == 2


def test_items_and_description():
    f = recent_8ks(SUBS, 320193, since=date(2026, 7, 1), limit=10)[1]
    assert f.items == ("2.02", "9.01")
    assert f.description == "quarterly results (earnings)"  # 9.01 (exhibits) is a formality
    assert f.url == "https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/earn.htm"


def test_unknown_item_code_is_still_described():
    f = Filing8K(accession="a", filed=date(2026, 1, 1), items=("6.99",), primary_document="x", url="u")
    assert f.description == "item 6.99"


@pytest.mark.parametrize("bad", [{}, {"filings": None}, {"filings": {"recent": {"form": "8-K"}}}])
def test_malformed_submissions_give_no_filings(bad):
    assert recent_8ks(bad, 1, since=date(2020, 1, 1), limit=5) == []


def test_malformed_dates_are_skipped():
    subs = submissions(("8-K", "0000320193-26-000018", "not-a-date", "8.01", "x.htm"))
    assert recent_8ks(subs, 1, since=date(2020, 1, 1), limit=5) == []


def test_lookback_covers_the_analysis_period():
    assert lookback_start(4, date(2026, 10, 20)) == date(2025, 10, 17)  # 4 x 92 days


# --- HTML to text ------------------------------------------------------------------------

def test_html_to_text_keeps_visible_text_only():
    html = """<html><head><title>T</title><style>p{}</style></head><body>
      <div style="display: none"><ix:header>hidden xbrl</ix:header></div>
      <p>Apple&nbsp;announced   a <b>new</b> CEO.</p><script>alert(1)</script>
      <table><tr><td>Item</td><td>5.02</td></tr></table><br>Done</body></html>"""
    text = html_to_text(html)
    assert "Apple announced a new CEO." in text
    for junk in ("hidden xbrl", "alert", "p{}", "T\n"):
        assert junk not in text
    assert "Done" in text


def test_html_to_text_survives_broken_html():
    assert "text" in html_to_text("<div><p>text</div></span></p>")


def test_strip_cover_page():
    assert strip_cover_page("FORM 8-K\nAddress\nItem 5.02 Departure of officers\nbody") == "Item 5.02 Departure of officers\nbody"
    assert strip_cover_page("no items here") == "no items here"


# --- SEC client: filing index and documents (fake SEC server) ----------------------------------

FOLDER = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000018"
INDEX = {"directory": {"item": [{"name": "earn.htm"}, {"name": "a8-kex991.htm"}, {"name": "x.zip"}, {"name": "ex99_logo.jpg"}]}}


def client(routes: dict[str, httpx.Response]) -> EdgarClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(str(request.url), httpx.Response(404))

    async def no_sleep(_):
        return None

    s = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io")
    return EdgarClient(s, transport=httpx.MockTransport(handler), sleep=no_sleep)


async def test_filing_text_reads_cover_and_exhibit_99():
    routes = {
        f"{FOLDER}/index.json": httpx.Response(200, json=INDEX),
        f"{FOLDER}/earn.htm": httpx.Response(200, text="<p>FORM 8-K boilerplate</p><p>Item 2.02 Results</p>"),
        f"{FOLDER}/a8-kex991.htm": httpx.Response(200, text="<p>Press release: record quarter</p>"),
    }
    f = Filing8K(accession="0000320193-26-000018", filed=date(2026, 7, 30), items=("2.02",),
                 primary_document="earn.htm", url="u")
    async with client(routes) as sec:
        text = await filing_text(sec, 320193, f)
    assert "Item 2.02 Results" in text
    assert "boilerplate" not in text            # cover page stripped
    assert "Press release: record quarter" in text  # exhibit 99 included; images and zips are not


async def test_filing_text_is_size_capped():
    routes = {
        f"{FOLDER}/index.json": httpx.Response(200, json={"directory": {"item": []}}),
        f"{FOLDER}/earn.htm": httpx.Response(200, text="<p>Item 8.01 " + "word " * 50_000 + "</p>"),
    }
    f = Filing8K(accession="0000320193-26-000018", filed=date(2026, 7, 30), items=(), primary_document="earn.htm", url="u")
    async with client(routes) as sec:
        assert len(await filing_text(sec, 320193, f)) <= 20_000


@pytest.mark.parametrize("accession,name", [
    ("../../etc", "x.htm"),
    ("0000320193-26-000018", "../secrets.htm"),
    ("0000320193-26-000018", "dir/x.htm"),
    ("0000320193-26-000018", ".hidden"),
])
async def test_paths_never_reach_a_url(accession, name):
    async with client({}) as sec:
        with pytest.raises(InvalidInput):
            await sec.filing_document(320193, accession, name)
