"""One retry policy for every Google API call (embeddings and LLM): same rules everywhere."""
import logging

import httpx
from google.genai import errors
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

# Temporary failures worth retrying. Anything else (e.g. 400 = bad request) is our bug: fail fast.
RETRYABLE_STATUS = {429, 500, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in RETRYABLE_STATUS
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


google_api_retry = retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
