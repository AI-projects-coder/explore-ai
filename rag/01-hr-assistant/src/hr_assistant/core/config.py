from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, loaded from environment variables (and .env locally)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
