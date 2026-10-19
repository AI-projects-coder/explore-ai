"""Settings: defaults, required values, and the guards that make bad config fail at startup."""

import pytest
from pydantic import ValidationError

from research_analyst.core.config import Settings, get_settings

VALID_UA = "explore-ai-research-analyst dev@mycompany.io"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from a known environment: only the required values are set."""
    for name in ("GCP_LOCATION", "PLANNER_MODEL", "RESEARCHER_MODEL", "WRITER_MODEL",
                 "SEC_REQUESTS_PER_SECOND", "MAX_PLAN_ITEMS", "ENVIRONMENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.setenv("SEC_USER_AGENT", VALID_UA)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def load(**overrides) -> Settings:
    # _env_file=None: tests must never depend on a developer's local .env
    return Settings(_env_file=None, **overrides)


def test_defaults():
    s = load()
    assert s.gcp_project == "test-project"
    assert s.gcp_location == "global"  # the Pro preview model is only served from the global endpoint
    assert s.planner_model and s.researcher_model and s.writer_model
    assert s.environment == "local"
    assert 0 < s.sec_requests_per_second <= 10


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("RESEARCHER_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("MAX_PLAN_ITEMS", "3")
    s = load()
    assert s.researcher_model == "gemini-2.5-flash"
    assert s.max_plan_items == 3


@pytest.mark.parametrize("missing", ["GCP_PROJECT", "SEC_USER_AGENT"])
def test_required_values_fail_fast(monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValidationError):
        load()


@pytest.mark.parametrize("ua", [
    "explore-ai-research-analyst",                         # no contact at all
    "explore-ai-research-analyst your-email@example.com",  # the .env.example placeholder
    "dev@mycompany.io",                                    # email but no app name
])
def test_bad_sec_user_agent_rejected(monkeypatch, ua):
    monkeypatch.setenv("SEC_USER_AGENT", ua)
    with pytest.raises(ValidationError):
        load()


def test_sec_user_agent_is_trimmed(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", f"  {VALID_UA}  ")
    assert load().sec_user_agent == VALID_UA


@pytest.mark.parametrize("field,value", [
    ("sec_requests_per_second", 11),   # SEC's published limit is 10/s
    ("sec_requests_per_second", 0),
    ("sec_max_retries", -1),
    ("max_plan_items", 0),
    ("max_plan_items", 11),
    ("max_quarters", 0),
    ("researcher_max_tool_calls", 0),
    ("llm_timeout_seconds", 0),
    ("graph_recursion_limit", 5),
    ("environment", "staging"),
])
def test_out_of_range_values_rejected(field, value):
    with pytest.raises(ValidationError):
        load(**{field: value})


def test_validation_errors_do_not_echo_input():
    with pytest.raises(ValidationError) as exc:
        load(sec_requests_per_second="SHOULD-NOT-LEAK")
    assert "SHOULD-NOT-LEAK" not in str(exc.value)


def test_unknown_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("SOME_OTHER_TOOL_SETTING", "x")
    load()


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
