import logging
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hr_assistant.core.auth import User
from hr_assistant.embeddings.vertex import Embedder

logger = logging.getLogger(__name__)

# Explicit SQL on purpose: this query IS the access-control boundary, so it should be easy to read and review.
_SEARCH_SQL = text("""
    SELECT c.id, c.document_id, d.title, c.heading_path, c.content, c.page_start, c.page_end,
           1 - (c.embedding <=> CAST(:query_vector AS vector)) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.allowed_groups && CAST(:user_groups AS text[])   -- access control
      AND d.status = 'active'                                -- defense in depth
      AND c.embedding_model = :embedding_model               -- never compare across models
    ORDER BY c.embedding <=> CAST(:query_vector AS vector)
    LIMIT :top_k
""")

# set_config(name, value, is_local=true) is SET LOCAL, but accepts bind parameters (no string formatting).
_SEARCH_SETTINGS_SQL = text("""
    SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true),
           set_config('hnsw.ef_search', :ef_search, true)
""")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    heading_path: str
    content: str
    page_start: int | None
    page_end: int | None
    score: float


class Retriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        session_factory: async_sessionmaker[AsyncSession],
        top_k: int = 5,
        min_score: float = 0.0,
        ef_search: int = 100,
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        self._top_k = top_k
        self._min_score = min_score
        self._ef_search = ef_search

    async def search(self, question: str, user: User) -> list[RetrievedChunk]:
        if not user.groups or not question.strip():
            return []  # default-deny: no groups means no access, never "no filter"

        started = time.perf_counter()
        query_vector = await self._embedder.embed_query_async(question)

        async with self._session_factory() as session, session.begin():
            await session.execute(_SEARCH_SETTINGS_SQL, {"ef_search": str(self._ef_search)})
            rows = (
                await session.execute(
                    _SEARCH_SQL,
                    {
                        "query_vector": str(query_vector),
                        "user_groups": sorted(user.groups),
                        "embedding_model": self._embedder.model,
                        "top_k": self._top_k,
                    },
                )
            ).all()

        # relaxed_order may return rows slightly out of order, so sort by score here.
        results = sorted(
            (
                RetrievedChunk(
                    chunk_id=r.id,
                    document_id=r.document_id,
                    title=r.title,
                    heading_path=r.heading_path,
                    content=r.content,
                    page_start=r.page_start,
                    page_end=r.page_end,
                    score=float(r.score),
                )
                for r in rows
            ),
            key=lambda c: c.score,
            reverse=True,
        )
        kept = [c for c in results if c.score >= self._min_score]
        logger.info(
            "retrieved %d chunks",
            len(kept),
            extra={
                "event": "retrieval",
                "groups": sorted(user.groups),
                "results": len(results),
                "kept": len(kept),
                "top_score": round(results[0].score, 3) if results else None,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return kept
