"""HTTP client for SEC EDGAR: ticker -> CIK, companyfacts, submissions. Polite rate limiting + retries.

Usage:
    async with EdgarClient(get_settings()) as sec:
        apple = await sec.resolve_ticker("AAPL")
        facts = await sec.company_facts(apple.cik)
"""

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from research_analyst.core.config import Settings

_TICKER = re.compile(r"[A-Z0-9]{1,6}(-[A-Z0-9]{1,3})?")
_CIK = re.compile(r"(?:CIK)?(\d{1,10})")
_ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")  # e.g. 0000320193-26-000018
_DOCUMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")  # a file name, never a path

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0

_TICKERS_TTL_SECONDS = 24 * 3600  # the ticker list barely changes
_FILINGS_TTL_SECONDS = 3600  # companyfacts/submissions change when a company files


# --- errors -----------------------------------------------------------------

class SecError(Exception):
    """Base class for every SEC problem; callers catch this one."""


class InvalidInput(SecError):
    """A ticker or CIK that can't be valid. Rejected before any request is made."""


class CompanyNotFound(SecError):
    """The ticker or CIK is well-formed but SEC has no such company (or no data for it)."""


class SecAccessDenied(SecError):
    """SEC returned 403: bad User-Agent or our IP is blocked. Never retried (retrying makes a block worse)."""


class SecUnavailable(SecError):
    """SEC kept failing (429/5xx/timeouts/bad JSON) after all retries, or answered with something unexpected."""


# --- input helpers ----------------------------------------------------------

def normalize_ticker(raw: str) -> str:
    """'$brk.b ' -> 'BRK-B'. The LLM chooses tickers, so this runs before any URL is built."""
    if not isinstance(raw, str):
        raise InvalidInput("ticker must be a string")
    t = raw.strip().upper().removeprefix("$").replace(".", "-").replace("/", "-")
    if not _TICKER.fullmatch(t):
        raise InvalidInput(f"not a valid ticker: {raw[:20]!r}")  # never echo huge input back
    return t


def format_cik(raw: int | str) -> str:
    """320193 / '320193' / 'CIK0000320193' -> '0000320193' (the 10-digit form SEC URLs use)."""
    if isinstance(raw, bool):  # bool is a subclass of int: True would become CIK 1
        raise InvalidInput("CIK must be a number")
    if isinstance(raw, int):
        if not 0 < raw < 10**10:
            raise InvalidInput(f"CIK out of range: {raw}")
        return f"{raw:010d}"
    if isinstance(raw, str):
        m = _CIK.fullmatch(raw.strip().upper())
        if m and int(m.group(1)) > 0:
            return f"{int(m.group(1)):010d}"
        raise InvalidInput(f"not a valid CIK: {raw[:20]!r}")
    raise InvalidInput("CIK must be an int or a string")


@dataclass(frozen=True)
class Company:
    cik: int
    ticker: str
    name: str


# --- rate limiter -----------------------------------------------------------

class RateLimiter:
    """Spaces requests at least 1/per_second apart, across every concurrent task in this process.

    Each caller reserves the next free time slot under a lock, then sleeps outside the lock,
    so waiting callers don't block each other from reserving.
    """

    def __init__(
        self,
        per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._interval = 1.0 / per_second
        self._clock = clock
        self._sleep = sleep
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        if slot > now:
            await self._sleep(slot - now)


# --- client -----------------------------------------------------------------

class EdgarClient:
    """One instance per process (or per run). Reuses connections; caches and de-duplicates downloads."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,  # tests pass a fake transport
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._data_url = settings.sec_data_url.rstrip("/")
        self._www_url = settings.sec_www_url.rstrip("/")
        self._max_retries = settings.sec_max_retries
        self._clock = clock
        self._sleep = sleep
        self._limiter = RateLimiter(settings.sec_requests_per_second, clock=clock, sleep=sleep)
        self._http = httpx.AsyncClient(
            headers={"User-Agent": settings.sec_user_agent},
            timeout=httpx.Timeout(settings.sec_timeout_seconds),
            follow_redirects=True,
            transport=transport,
        )
        # key -> (created_at, task). Storing the *task* (not the result) is what de-duplicates:
        # six researchers asking for the same file at once all await the one download in flight.
        self._cache: dict[str, tuple[float, asyncio.Task[Any]]] = {}

    async def __aenter__(self) -> "EdgarClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- public API ---------------------------------------------------------

    async def resolve_ticker(self, ticker: str) -> Company:
        t = normalize_ticker(ticker)
        companies = await self._cached("tickers", _TICKERS_TTL_SECONDS, self._load_tickers)
        try:
            return companies[t]
        except KeyError:
            raise CompanyNotFound(f"ticker {t!r} not found in SEC's company list") from None

    async def company_facts(self, cik: int | str) -> dict[str, Any]:
        """Every XBRL number the company has reported (several MB for large companies)."""
        c = format_cik(cik)
        url = f"{self._data_url}/api/xbrl/companyfacts/CIK{c}.json"
        return await self._cached(f"facts:{c}", _FILINGS_TTL_SECONDS,
                                  lambda: self._get_json(url, not_found=f"no XBRL financial data for CIK {c}"))

    async def submissions(self, cik: int | str) -> dict[str, Any]:
        """Company profile and its recent filings (forms, dates, accession numbers)."""
        c = format_cik(cik)
        url = f"{self._data_url}/submissions/CIK{c}.json"
        return await self._cached(f"subs:{c}", _FILINGS_TTL_SECONDS,
                                  lambda: self._get_json(url, not_found=f"no SEC filings for CIK {c}"))

    async def filing_index(self, cik: int | str, accession: str) -> list[str]:
        """File names inside one filing (primary document, exhibits, ...)."""
        folder = self._filing_folder(cik, accession)
        raw = await self._get_json(f"{folder}/index.json", not_found=f"filing {accession} not found")
        try:
            return [item["name"] for item in raw["directory"]["item"]]
        except (KeyError, TypeError) as e:
            raise SecUnavailable(f"filing index has an unexpected format ({type(e).__name__})") from None

    async def filing_document(self, cik: int | str, accession: str, name: str) -> str:
        """One document of a filing, as text (usually HTML)."""
        if not _DOCUMENT.fullmatch(name):  # names come from SEC's index, but never trust them in a URL
            raise InvalidInput(f"not a valid document name: {name[:40]!r}")
        url = f"{self._filing_folder(cik, accession)}/{name}"
        return await self._get(url, not_found=f"document {name} not found", parse=lambda r: r.text)

    # --- internals ----------------------------------------------------------

    def _filing_folder(self, cik: int | str, accession: str) -> str:
        if not _ACCESSION.fullmatch(accession):
            raise InvalidInput(f"not a valid accession number: {accession[:30]!r}")
        return f"{self._www_url}/Archives/edgar/data/{int(format_cik(cik))}/{accession.replace('-', '')}"

    async def _load_tickers(self) -> dict[str, Company]:
        raw = await self._get_json(f"{self._www_url}/files/company_tickers.json", not_found="SEC ticker list missing")
        try:
            return {
                row["ticker"].upper(): Company(cik=int(row["cik_str"]), ticker=row["ticker"].upper(), name=row["title"])
                for row in raw.values()
            }
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise SecUnavailable(f"SEC ticker list has an unexpected format ({type(e).__name__})") from None

    async def _cached(self, key: str, ttl: float, fetch: Callable[[], Awaitable[Any]]) -> Any:
        entry = self._cache.get(key)
        fresh = entry is not None and self._clock() - entry[0] < ttl
        failed = entry is not None and entry[1].done() and (entry[1].cancelled() or entry[1].exception() is not None)
        if not fresh or failed:  # failures are never cached: the next caller tries again
            entry = (self._clock(), asyncio.ensure_future(fetch()))
            self._cache[key] = entry
        # shield: if one caller is cancelled, the shared download keeps going for the others
        return await asyncio.shield(entry[1])

    async def _get_json(self, url: str, *, not_found: str) -> Any:
        return await self._get(url, not_found=not_found, parse=lambda r: r.json())

    async def _get(self, url: str, *, not_found: str, parse: Callable[[httpx.Response], Any]) -> Any:
        """GET with rate limiting and retries. Maps every failure to a SecError subclass.

        `parse` turns a 200 response into the result; if it raises ValueError (e.g. an HTML
        maintenance page where JSON was expected), the response counts as a temporary failure.
        """
        problem = "no attempt made"
        for attempt in range(self._max_retries + 1):
            retry_after: float | None = None
            await self._limiter.wait()
            try:
                resp = await self._http.get(url)
            except httpx.TransportError as e:  # timeouts, DNS, connection resets
                problem = type(e).__name__
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        return parse(resp)
                    except ValueError:  # e.g. an HTML error page served with 200
                        problem = "response could not be parsed"
                elif status == 404:
                    raise CompanyNotFound(not_found)
                elif status == 403:
                    raise SecAccessDenied("SEC refused access (403): check SEC_USER_AGENT, or we are rate-limit blocked")
                elif status in _RETRYABLE_STATUS:
                    problem = f"HTTP {status}"
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                else:
                    raise SecUnavailable(f"unexpected HTTP {status} from SEC")

            if attempt < self._max_retries:
                await self._sleep(self._backoff(attempt, retry_after))

        raise SecUnavailable(f"SEC still failing after {self._max_retries + 1} attempts ({problem})")

    @staticmethod
    def _backoff(attempt: int, retry_after: float | None) -> float:
        """1s, 2s, 4s, ... capped at 30s, plus jitter so parallel callers don't retry in lockstep."""
        if retry_after is not None:
            return min(retry_after, _BACKOFF_MAX_SECONDS)
        delay = min(_BACKOFF_BASE_SECONDS * 2**attempt, _BACKOFF_MAX_SECONDS)
        return delay + random.uniform(0, delay / 2)


def _parse_retry_after(value: str | None) -> float | None:
    """Retry-After in seconds. (It may also be an HTTP date; we ignore that form and use our own backoff.)"""
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
