"""Per-request context: a request id on every log line and on the response, plus one access-log line."""
import logging
import re
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import RequestResponseEndpoint

from hr_assistant.core.logs import request_id_var, user_id_var

logger = logging.getLogger("hr_assistant.access")

# Accept the caller's id (the BFF sends one) only if it looks like an id: this stops log injection,
# e.g. a header containing newlines or a fake JSON fragment.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


async def request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
    rid_token, uid_token = request_id_var.set(request_id), user_id_var.set(None)
    started = time.perf_counter()
    status_code = 500  # if the app crashes before responding
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logger.info(
            "%s %s %d",
            request.method,
            request.url.path,
            status_code,
            extra={
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,  # the path only: query strings can contain personal data
                "status": status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "user_id": getattr(request.state, "user_id", None),  # set by get_current_user
            },
        )
        request_id_var.reset(rid_token)
        user_id_var.reset(uid_token)
