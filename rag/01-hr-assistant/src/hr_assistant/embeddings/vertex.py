import logging
from dataclasses import dataclass
from typing import Protocol

from google import genai
from google.genai import errors, types


from hr_assistant.core.config import Settings
from hr_assistant.core.retry import google_api_retry
from hr_assistant.db.models import EMBEDDING_DIM

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    token_count: int


class Embedder(Protocol):
    model: str

    def embed_documents(self, texts: list[str]) -> EmbeddingResult: ...
    
    def embed_query(self, text: str) -> list[float]: ...

    async def embed_query_async(self, text: str) -> list[float]: ...


class VertexEmbedder:
    def __init__(
        self,
        *,
        project: str,
        location: str,
        model: str,
        batch_size: int = 100,
        use_task_types: bool = True,
        timeout_s: int = 60,
    ) -> None:
        self.model = model
        self._batch_size = batch_size
        self._use_task_types = use_task_types
        # One client for the whole run: reuses the HTTP connection and auth token.
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(timeout=timeout_s * 1000),  # milliseconds
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "VertexEmbedder":
        if settings.embedding_dim != EMBEDDING_DIM:
            raise ValueError(
                f"EMBEDDING_DIM={settings.embedding_dim} but the database column is vector({EMBEDDING_DIM})"
            )
        return cls(
            project=settings.gcp_project,
            location=settings.gcp_location,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            use_task_types=settings.embedding_task_types,
        )

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        vectors: list[list[float]] = []
        token_count = 0
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._embed(batch, "RETRIEVAL_DOCUMENT")
            if len(response.embeddings) != len(batch):
                raise ValueError(f"sent {len(batch)} texts, got {len(response.embeddings)} vectors")
            for offset, embedding in enumerate(response.embeddings):
                stats = embedding.statistics
                if stats and stats.truncated:
                    raise ValueError(f"text #{start + offset} exceeded the model's input limit and was truncated")
                token_count += int(stats.token_count or 0) if stats else 0
                vectors.append(self._checked(embedding.values))
        return EmbeddingResult(vectors=vectors, token_count=token_count)

    def embed_query(self, text: str) -> list[float]:
        response = self._embed([text], "RETRIEVAL_QUERY")
        return self._checked(response.embeddings[0].values)

    async def embed_query_async(self, text: str) -> list[float]:
        """Same as embed_query, but doesn't block the event loop (used by the async API)."""
        response = await self._embed_async([text], "RETRIEVAL_QUERY")
        return self._checked(response.embeddings[0].values)

    def _config(self, task_type: str) -> types.EmbedContentConfig:
        return types.EmbedContentConfig(
            task_type=task_type if self._use_task_types else None,
            output_dimensionality=EMBEDDING_DIM,
        )

    @google_api_retry
    def _embed(self, batch: list[str], task_type: str) -> types.EmbedContentResponse:
        return self._client.models.embed_content(model=self.model, contents=batch, config=self._config(task_type))

    @google_api_retry
    async def _embed_async(self, batch: list[str], task_type: str) -> types.EmbedContentResponse:
        return await self._client.aio.models.embed_content(
            model=self.model, contents=batch, config=self._config(task_type)
        )


    @staticmethod
    def _checked(values: list[float] | None) -> list[float]:
        if values is None or len(values) != EMBEDDING_DIM:
            raise ValueError(f"expected {EMBEDDING_DIM} dimensions, got {len(values or [])}")
        return list(values)
