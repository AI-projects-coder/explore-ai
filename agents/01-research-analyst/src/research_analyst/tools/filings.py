"""Recent 8-K filings: list them and fetch their text for the researchers.

An 8-K is how a US company announces a material event: earnings, a new CEO, an acquisition...
Each one lists "item" codes saying what kind of event it is. The real content is often in
exhibit 99 (the press release), not in the short cover document.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from html.parser import HTMLParser
from typing import Any

from research_analyst.tools.edgar import EdgarClient

# The 8-K item codes we translate for the LLM (and for readers of the report).
ITEM_NAMES = {
    "1.01": "entry into a material agreement",
    "1.02": "termination of a material agreement",
    "1.03": "bankruptcy or receivership",
    "2.01": "completed acquisition or sale of assets",
    "2.02": "quarterly results (earnings)",
    "2.03": "new debt or financial obligation",
    "2.05": "restructuring or exit costs",
    "2.06": "material impairment",
    "3.01": "delisting notice",
    "4.01": "change of auditor",
    "4.02": "earlier financial statements can no longer be relied on",
    "5.01": "change in control",
    "5.02": "executive or director change",
    "5.03": "change to articles or bylaws",
    "5.07": "shareholder vote results",
    "7.01": "Regulation FD disclosure",
    "8.01": "other events",
    "9.01": "financial statements and exhibits",
}

_EXHIBIT_99 = re.compile(r"ex-?99", re.IGNORECASE)
_FIRST_ITEM = re.compile(r"^Item\s+\d\.\d{2}", re.MULTILINE | re.IGNORECASE)
_TEXT_DOCUMENT = re.compile(r"\.(htm|html|txt)$", re.IGNORECASE)
_MAX_CHARS_PER_DOCUMENT = 12_000
_MAX_CHARS_PER_FILING = 20_000


@dataclass(frozen=True)
class Filing8K:
    accession: str
    filed: date
    items: tuple[str, ...]
    primary_document: str
    url: str

    @property
    def description(self) -> str:
        return "; ".join(ITEM_NAMES.get(i, f"item {i}") for i in self.items if i != "9.01") or "exhibits only"


def recent_8ks(submissions: dict[str, Any], cik: int, *, since: date, limit: int) -> list[Filing8K]:
    """8-Ks filed on or after `since`, newest first, at most `limit`. Malformed rows are skipped."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    columns = ("form", "accessionNumber", "filingDate", "items", "primaryDocument")
    if not all(isinstance(recent.get(c), list) for c in columns):
        return []
    out = []
    for form, accession, filed, items, primary in zip(*(recent[c] for c in columns)):
        if form not in ("8-K", "8-K/A"):
            continue
        try:
            filed_on = date.fromisoformat(filed)
        except (TypeError, ValueError):
            continue
        if filed_on < since:
            continue
        out.append(Filing8K(
            accession=accession,
            filed=filed_on,
            items=tuple(i.strip() for i in (items or "").split(",") if i.strip()),
            primary_document=primary,
            url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{primary}",
        ))
    out.sort(key=lambda f: f.filed, reverse=True)
    return out[:limit]


def lookback_start(quarters: int, today: date) -> date:
    """News window: the period the analysis covers (about 92 days per quarter)."""
    return today - timedelta(days=92 * quarters)


async def filing_text(sec: EdgarClient, cik: int, filing: Filing8K) -> str:
    """Readable text of an 8-K: the cover document plus any exhibit 99 (press releases), size-capped."""
    names = await sec.filing_index(cik, filing.accession)
    exhibits = [n for n in names if _EXHIBIT_99.search(n) and _TEXT_DOCUMENT.search(n)]
    parts = []
    for name in [filing.primary_document, *exhibits]:
        text = html_to_text(await sec.filing_document(cik, filing.accession, name))
        if name == filing.primary_document:
            text = strip_cover_page(text)
        parts.append(f"--- {name} ---\n{text[:_MAX_CHARS_PER_DOCUMENT]}")
    return "\n\n".join(parts)[:_MAX_CHARS_PER_FILING]


def strip_cover_page(text: str) -> str:
    """Drop the standard 8-K form header (address, checkboxes...): the content starts at the first 'Item X.XX'."""
    m = _FIRST_ITEM.search(text)
    return text[m.start():] if m else text


class _TextExtractor(HTMLParser):
    """Collects visible text. Skips scripts, styles and hidden blocks (inline XBRL hides metadata there)."""

    _SKIP = {"script", "style", "head", "ix:header"}
    _VOID = {"br", "hr", "img", "meta", "link", "input", "area", "base", "col", "wbr", "source"}
    _BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[bool] = []  # one entry per open element: is it hidden?
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._BLOCK:
            self._chunks.append("\n")
        if tag in self._VOID:
            return
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        hidden = tag in self._SKIP or "display:none" in style
        self._stack.append(hidden or (bool(self._stack) and self._stack[-1]))

    def handle_endtag(self, tag: str) -> None:
        if tag in self._VOID:
            return
        if self._stack:
            self._stack.pop()
        if tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not (self._stack and self._stack[-1]):
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks).replace("\xa0", " ")
        lines = (" ".join(line.split()) for line in raw.splitlines())
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()
