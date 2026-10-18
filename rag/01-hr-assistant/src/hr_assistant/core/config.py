from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, loaded from environment variables (and .env locally)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True
    )

    # Database
    database_url: str

    # GCP / Vertex AI
    gcp_project: str
    gcp_location: str = "asia-south1"

    # Models
    llm_model: str = "gemini-3.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = Field(default=1536, gt=0, le=2000)

    # Chunking
    chunk_max_tokens: int = Field(default=400, gt=0, le=1500)
    chunk_min_tokens: int = Field(default=50, ge=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)

    # Embeddings
    embedding_batch_size: int = Field(default=100, gt=0, le=250)
    embedding_task_types: bool = True

    # Ingestion safety: abort if a run would delete more than this fraction of active documents
    ingest_max_delete_fraction: float = Field(default=0.5, ge=0, le=1)

    # Retrieval
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_score: float = Field(default=0.0, ge=-1, le=1)
    hnsw_ef_search: int = Field(default=100, ge=1, le=1000)

    # Generation
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_thinking_level: Literal["low", "high"] | None = "low"
    llm_max_output_tokens: int = Field(default=1024, gt=0, le=8192)
        # What users see when there is no answer. Never mention specific documents (they may be restricted).
    not_found_message: str = (
        "I couldn't find this in the HR policies available to you. "
        "Please contact the HR helpdesk at hr-helpdesk@kalpa.example."
    )
    off_topic_message: str = (
        "I can only help with questions about Kalpa Systems' HR policies, "
        "such as leave, working from home, or workplace conduct."
    )
    no_context_message: str = (  # used when retrieval finds nothing, so we can't tell the two cases apart
        "I couldn't find this in the HR policies available to you. I can help with questions about "
        "Kalpa Systems' HR policies, such as leave, working from home, or workplace conduct. "
        "For other HR matters, please contact hr-helpdesk@kalpa.example."
    )


    # API and authentication
    environment: Literal["local", "dev", "prod"] = "local"
    jwt_secret: SecretStr = Field(min_length=32)
    jwt_issuer: str = "hr-assistant"
    jwt_ttl_minutes: int = Field(default=30, gt=0, le=24 * 60)

    @property
    def dev_auth_enabled(self) -> bool:
        """The dev login exists only on a developer's machine, by construction."""
        return self.environment == "local"

    @model_validator(mode="after")
    def _no_placeholder_secret_outside_local(self) -> "Settings":
        if self.environment != "local" and "change-me" in self.jwt_secret.get_secret_value():
            raise ValueError("JWT_SECRET is still the example placeholder; set a real secret")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
