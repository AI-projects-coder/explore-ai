"""SEC EDGAR client.

Part 1: pure helpers (errors, ticker normalization, CIK formatting). No network.
Part 2: rate limiter, retries, caching. A fake SEC server (httpx.MockTransport) stands in for the real one.
"""

import asyncio

import httpx
import pytest

from research_analyst.core.config import Settings
from research_analyst.tools.edgar import (
    Company,
    CompanyNotFound,
    EdgarClient,
    InvalidInput,
    RateLimiter,
    SecAccessDenied,
    SecError,
    SecUnavailable,
    format_cik,
    normalize_ticker,
)


# --- errors -----------------------------------------------------------------

@pytest.mark.parametrize("cls", [CompanyNotFound, InvalidInput, SecAccessDenied, SecUnavailable])
def test_all_errors_share_one_base(cls):
    # callers (the researcher node) can catch every SEC problem with one `except SecError`
    assert issubclass(cls, SecError)
    assert issubclass(SecError, Exception)


def test_errors_carry_a_readable_message():
    assert str(CompanyNotFound("ticker 'ZZZZ' not found")) == "ticker 'ZZZZ' not found"


# --- normalize_ticker -------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("AAPL", "AAPL"),
    ("aapl", "AAPL"),        # LLMs and users don't care about case
    ("  msft \n", "MSFT"),   # stray whitespace
    ("$NVDA", "NVDA"),       # "$AAPL" is how tickers are written in finance text
    ("BRK.B", "BRK-B"),      # SEC's ticker list uses '-' for share classes
    ("brk/b", "BRK-B"),
    ("GOOGL", "GOOGL"),
    ("BF-B", "BF-B"),
])
def test_normalize_ticker_accepts_common_forms(raw, expected):
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    "   ",
    "$",
    "../../etc/passwd",      # must never reach a URL
    "AAPL; DROP",
    "AAPL MSFT",             # two tickers in one string
    "A" * 11,                # longer than any real ticker
    "AA--B",
    "-AAPL",
    "AAPL-",
    "ÄPPL",                  # non-ASCII
])
def test_normalize_ticker_rejects_garbage(raw):
    with pytest.raises(InvalidInput):
        normalize_ticker(raw)


def test_normalize_ticker_rejects_non_strings():
    with pytest.raises(InvalidInput):
        normalize_ticker(None)  # type: ignore[arg-type]


def test_invalid_ticker_message_does_not_echo_huge_input():
    # the message goes back to the LLM and into logs: keep it short
    with pytest.raises(InvalidInput) as exc:
        normalize_ticker("X" * 10_000)
    assert len(str(exc.value)) < 200


# --- format_cik -------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (320193, "0000320193"),
    ("320193", "0000320193"),
    ("0000320193", "0000320193"),
    ("CIK0000320193", "0000320193"),   # the form used in SEC URLs and file names
    ("cik320193", "0000320193"),
    (" 320193 ", "0000320193"),
    (9_999_999_999, "9999999999"),     # largest 10-digit CIK
])
def test_format_cik(raw, expected):
    assert format_cik(raw) == expected


@pytest.mark.parametrize("raw", [
    0,
    -5,
    10_000_000_000,     # 11 digits
    "",
    "abc",
    "32O193",           # letter O, not zero
    "320193.0",
    3.2e5,              # floats are not CIKs
    True,               # bool is a subclass of int in Python
    None,
])
def test_format_cik_rejects_invalid(raw):
    with pytest.raises(InvalidInput):
        format_cik(raw)


# =============================================================================
# Part 2: network behaviour against a fake SEC server
# =============================================================================

UA = "explore-ai-test dev@mycompany.io"
TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
}
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"


class FakeTime:
    """A clock that only moves when someone 'sleeps'. Tests run instantly and can see every wait."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)  # let other tasks run, like a real sleep would


class FakeSec:
    """Answers requests from a script: url -> list of responses, used in order (the last one repeats)."""

    def __init__(self, routes: dict[str, list]):
        self.routes = routes
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        script = self.routes.get(str(request.url))
        if script is None:
            return httpx.Response(404)
        step = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(step, Exception):
            raise step
        return step

    def count(self, url: str) -> int:
        return sum(str(r.url) == url for r in self.calls)


def ok(data) -> httpx.Response:
    return httpx.Response(200, json=data)


@pytest.fixture
def fake_time():
    return FakeTime()


def make_client(fake: FakeSec, fake_time: FakeTime, **settings) -> EdgarClient:
    s = Settings(_env_file=None, gcp_project="p", sec_user_agent=UA, **settings)
    return EdgarClient(s, transport=httpx.MockTransport(fake), clock=fake_time.clock, sleep=fake_time.sleep)


# --- happy path -------------------------------------------------------------

async def test_resolve_ticker_and_fetch_facts(fake_time):
    fake = FakeSec({
        "https://www.sec.gov/files/company_tickers.json": [ok(TICKERS)],
        FACTS_URL: [ok({"entityName": "Apple Inc.", "facts": {}})],
    })
    async with make_client(fake, fake_time) as sec:
        apple = await sec.resolve_ticker(" $aapl ")
        assert apple == Company(cik=320193, ticker="AAPL", name="Apple Inc.")
        facts = await sec.company_facts(apple.cik)
    assert facts["entityName"] == "Apple Inc."


async def test_every_request_sends_the_user_agent(fake_time):
    fake = FakeSec({FACTS_URL: [ok({})]})
    async with make_client(fake, fake_time) as sec:
        await sec.company_facts(320193)
    assert fake.calls[0].headers["User-Agent"] == UA


async def test_share_class_tickers_resolve(fake_time):
    fake = FakeSec({"https://www.sec.gov/files/company_tickers.json": [ok(TICKERS)]})
    async with make_client(fake, fake_time) as sec:
        assert (await sec.resolve_ticker("brk.b")).cik == 1067983


# --- bad input never reaches the network -------------------------------------

async def test_invalid_input_makes_no_request(fake_time):
    fake = FakeSec({})
    async with make_client(fake, fake_time) as sec:
        with pytest.raises(InvalidInput):
            await sec.resolve_ticker("../../etc/passwd")
        with pytest.raises(InvalidInput):
            await sec.company_facts("abc")
    assert fake.calls == []


async def test_unknown_ticker_is_company_not_found(fake_time):
    fake = FakeSec({"https://www.sec.gov/files/company_tickers.json": [ok(TICKERS)]})
    async with make_client(fake, fake_time) as sec:
        with pytest.raises(CompanyNotFound, match="ZZZZ"):
            await sec.resolve_ticker("ZZZZ")


# --- error mapping: which failures are retried, which are not -----------------

async def test_404_is_company_not_found_and_not_retried(fake_time):
    fake = FakeSec({FACTS_URL: [httpx.Response(404)]})
    async with make_client(fake, fake_time) as sec:
        with pytest.raises(CompanyNotFound):
            await sec.company_facts(320193)
    assert fake.count(FACTS_URL) == 1


async def test_403_is_access_denied_and_never_retried(fake_time):
    fake = FakeSec({FACTS_URL: [httpx.Response(403)]})
    async with make_client(fake, fake_time) as sec:
        with pytest.raises(SecAccessDenied):
            await sec.company_facts(320193)
    assert fake.count(FACTS_URL) == 1  # retrying a block makes it longer


@pytest.mark.parametrize("first_failure", [
    httpx.Response(503),
    httpx.Response(429),
    httpx.Response(500),
    httpx.ReadTimeout("slow"),
    httpx.ConnectError("reset"),
    httpx.Response(200, text="<html>maintenance</html>"),  # 200 but not JSON
])
async def test_temporary_failures_are_retried_then_succeed(fake_time, first_failure):
    fake = FakeSec({FACTS_URL: [first_failure, ok({"entityName": "Apple Inc."})]})
    async with make_client(fake, fake_time) as sec:
        facts = await sec.company_facts(320193)
    assert facts["entityName"] == "Apple Inc."
    assert fake.count(FACTS_URL) == 2


async def test_gives_up_after_max_retries(fake_time):
    fake = FakeSec({FACTS_URL: [httpx.Response(503)]})
    async with make_client(fake, fake_time, sec_max_retries=3) as sec:
        with pytest.raises(SecUnavailable, match="4 attempts"):
            await sec.company_facts(320193)
    assert fake.count(FACTS_URL) == 4  # 1 try + 3 retries


async def test_unexpected_status_is_not_retried(fake_time):
    fake = FakeSec({FACTS_URL: [httpx.Response(400)]})
    async with make_client(fake, fake_time) as sec:
        with pytest.raises(SecUnavailable, match="400"):
            await sec.company_facts(320193)
    assert fake.count(FACTS_URL) == 1


async def test_backoff_grows_between_retries(fake_time):
    fake = FakeSec({FACTS_URL: [httpx.Response(503)]})
    async with make_client(fake, fake_time, sec_max_retries=3, sec_requests_per_second=10) as sec:
        with pytest.raises(SecUnavailable):
            await sec.company_facts(320193)
    backoffs = [s for s in fake_time.sleeps if s >= 1]  # ignore the small rate-limiter waits
    assert len(backoffs) == 3
    assert 1 <= backoffs[0] <= 1.5 and 2 <= backoffs[1] <= 3 and 4 <= backoffs[2] <= 6


async def test_retry_after_header_is_respected(fake_time):
    fake = FakeSec({FACTS_URL: [httpx.Response(429, headers={"Retry-After": "7"}), ok({})]})
    async with make_client(fake, fake_time) as sec:
        await sec.company_facts(320193)
    assert 7 in fake_time.sleeps


async def test_garbled_ticker_list_is_sec_unavailable(fake_time):
    fake = FakeSec({"https://www.sec.gov/files/company_tickers.json": [ok({"0": {"oops": 1}})]})
    async with make_client(fake, fake_time) as sec:
        with pytest.raises(SecUnavailable, match="unexpected format"):
            await sec.resolve_ticker("AAPL")


# --- caching -----------------------------------------------------------------

async def test_ticker_list_downloaded_once(fake_time):
    url = "https://www.sec.gov/files/company_tickers.json"
    fake = FakeSec({url: [ok(TICKERS)]})
    async with make_client(fake, fake_time) as sec:
        await sec.resolve_ticker("AAPL")
        await sec.resolve_ticker("BRK-B")
        await sec.resolve_ticker("AAPL")
    assert fake.count(url) == 1


async def test_parallel_requests_for_same_company_download_once(fake_time):
    # six researchers start at the same moment and all need Apple's facts
    fake = FakeSec({FACTS_URL: [ok({"entityName": "Apple Inc."})]})
    async with make_client(fake, fake_time) as sec:
        results = await asyncio.gather(*[sec.company_facts(320193) for _ in range(6)])
    assert all(r["entityName"] == "Apple Inc." for r in results)
    assert fake.count(FACTS_URL) == 1


async def test_failures_are_not_cached(fake_time):
    fake = FakeSec({FACTS_URL: [httpx.Response(404), ok({"entityName": "Apple Inc."})]})
    async with make_client(fake, fake_time) as sec:
        with pytest.raises(CompanyNotFound):
            await sec.company_facts(320193)
        assert (await sec.company_facts(320193))["entityName"] == "Apple Inc."


async def test_cache_expires(fake_time):
    fake = FakeSec({FACTS_URL: [ok({"v": 1}), ok({"v": 2})]})
    async with make_client(fake, fake_time) as sec:
        assert (await sec.company_facts(320193))["v"] == 1
        fake_time.now += 2 * 3600  # two hours later: the company may have filed something new
        assert (await sec.company_facts(320193))["v"] == 2


# --- rate limiter --------------------------------------------------------------

async def test_rate_limiter_spaces_parallel_callers(fake_time):
    # Four callers arrive at the same instant (the clock is frozen). Each must be told to wait
    # until its own slot: 5/s means one request every 0.2s.
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(round(seconds, 3))

    limiter = RateLimiter(5, clock=lambda: 1000.0, sleep=record)
    await asyncio.gather(*[limiter.wait() for _ in range(4)])
    assert sorted(waits) == [0.2, 0.4, 0.6]  # the first caller goes immediately


async def test_rate_limiter_does_not_wait_when_idle(fake_time):
    limiter = RateLimiter(5, clock=fake_time.clock, sleep=fake_time.sleep)
    await limiter.wait()
    fake_time.now += 10
    await limiter.wait()
    assert fake_time.sleeps == []
