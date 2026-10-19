"""Settings loaded from environment variables / .env."""

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Compiled once when the module loads, not on every validation
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class Settings(BaseSettings):
    """App configuration, loaded from environment variables (and .env locally)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    # Environment / GCP
    environment: Literal["local", "dev", "prod"] = "local"
    gcp_project: str
    gcp_location: str = "global"  # the Pro preview model is only served from the global endpoint

    # Models (one per node, so evals can swap them independently)
    planner_model: str = "gemini-3.1-pro-preview"
    researcher_model: str = "gemini-3-flash-preview"
    writer_model: str = "gemini-3.1-pro-preview"

    # LLM safety
    llm_max_output_tokens: int = Field(default=8192, gt=0, le=65536)  # thinking tokens count toward this
    writer_max_output_tokens: int = Field(default=32768, gt=0, le=65536)  # report + claims + thinking: needs more room
    llm_timeout_seconds: float = Field(default=120, gt=0, le=600)  # a hung call must not block a run forever

    # SEC EDGAR
    sec_user_agent: str  # SEC blocks requests without a contact: "AppName you@domain.com"
    sec_data_url: str = "https://data.sec.gov"  # XBRL companyfacts, submissions
    sec_www_url: str = "https://www.sec.gov"  # ticker list, filing documents
    sec_requests_per_second: float = Field(default=5, gt=0, le=10)  # SEC's hard limit is 10/s
    sec_timeout_seconds: float = Field(default=30, gt=0, le=120)
    sec_max_retries: int = Field(default=3, ge=0, le=10)

    # Agent bounds: never trust the LLM to limit itself
    max_plan_items: int = Field(default=6, ge=1, le=10)  # caps the researcher fan-out
    max_quarters: int = Field(default=12, ge=1, le=40)
    researcher_max_tool_calls: int = Field(default=8, ge=1, le=25)  # caps one researcher's tool loop
    graph_recursion_limit: int = Field(default=50, ge=10, le=500)  # LangGraph's last-resort step limit

    @field_validator("sec_user_agent")
    @classmethod
    def _sec_contact(cls, v: str) -> str:
        v = v.strip()
        email = _EMAIL.search(v)
        if not email:
            raise ValueError("SEC_USER_AGENT must contain a contact email")
        if email.group().lower().endswith("@example.com"):
            raise ValueError("SEC_USER_AGENT is still the example placeholder; use your real email")
        if not v.replace(email.group(), "").strip():
            raise ValueError("SEC_USER_AGENT must include an app name before the email")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
