"""Settings loaded from environment variables / .env."""

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
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

    # Persistence (v1): LangGraph checkpoints + our runs table
    database_url: str = "postgresql://agent:agent@localhost:5434/research_analyst"
    db_pool_max_size: int = Field(default=5, ge=1, le=50)

    # Quality loop (v1)
    critic_model: str = "gemini-3-flash-preview"  # judges qualitative claims; numbers are checked by code
    critic_max_revisions: int = Field(default=2, ge=0, le=5)  # critic -> writer loops per review round
    review_max_revisions: int = Field(default=3, ge=0, le=10)  # human "revise" rounds per run

    # Budget per run (v1). Tokens and calls are exact; USD is an estimate from model_prices.
    budget_max_tokens: int = Field(default=150_000, gt=0)
    budget_max_llm_calls: int = Field(default=20, gt=0)
    budget_max_seconds: int = Field(default=600, gt=0)  # per working session; time paused for review doesn't count
    budget_max_usd: float | None = Field(default=None, gt=0)  # set it only together with model_prices
    model_prices: dict[str, tuple[float, float]] = {}  # model -> (USD per 1M input tokens, per 1M output tokens)

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


    @model_validator(mode="after")
    def _prices_for_usd_budget(self) -> "Settings":
        """A USD limit needs a price for every model, or the estimate would silently be too low."""
        if self.budget_max_usd is not None:
            models = {self.planner_model, self.researcher_model, self.writer_model, self.critic_model}
            missing = sorted(models - set(self.model_prices))
            if missing:
                raise ValueError(f"BUDGET_MAX_USD is set but MODEL_PRICES has no price for: {missing}")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
