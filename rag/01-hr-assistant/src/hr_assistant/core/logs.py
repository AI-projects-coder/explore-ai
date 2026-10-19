"""Logging setup: one JSON object per line in dev/prod (Cloud Logging parses it), readable text locally.

Every line carries the request id and user id of the request it belongs to, via contextvars: set once
at the start of a request, visible to all code running for that request, without passing them around.

Log structured facts with `extra`, and give each important line a stable `event` name, e.g.
    logger.warning("LLM output truncated", extra={"event": "llm_truncated", "max_output_tokens": 1024})
Alerts and dashboards then filter on jsonPayload.event instead of parsing sentences.
Never put question or answer text in logs: HR questions are personal data.
"""
import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)

# Every LogRecord has these attributes; anything else was passed via extra={...} and becomes a JSON field.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message", "asctime", "taskName", "request_id", "user_id",
}


class ContextFilter(logging.Filter):
    """Copy the current request's ids onto every record (unless the call passed its own)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "request_id", None) is None:
            record.request_id = request_id_var.get()
        if getattr(record, "user_id", None) is None:
            record.user_id = user_id_var.get()
        return True


def _extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "severity": record.levelname,  # Cloud Logging maps this to the log level
            "message": record.getMessage(),
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "logger": record.name,
            "request_id": record.request_id,
            "user_id": record.user_id,
            **_extras(record),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """For humans on a laptop: time, level, logger, short request id, message, then key=value extras."""

    def format(self, record: logging.LogRecord) -> str:
        time = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        rid = (record.request_id or "-")[:8]
        extras = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        line = f"{time} {record.levelname:<7} {record.name} [{rid}] {record.getMessage()}"
        line = f"{line}  | {extras}" if extras else line
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(*, json_logs: bool, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)  # Cloud Run collects stdout
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Route uvicorn's own logs through the same handler (same format), and replace its access log
    # with ours (api/middleware.py), which also has the request id, user id and duration.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "google_genai"):  # one INFO line per call to Vertex is noise, not signal
        logging.getLogger(noisy).setLevel(logging.WARNING)
