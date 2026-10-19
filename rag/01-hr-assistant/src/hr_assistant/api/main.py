"""The online API. Run locally: uv run uvicorn hr_assistant.api.main:app --reload"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from hr_assistant.generation.answer import AnswerService, NoAnswerMessages

from hr_assistant.api.middleware import request_context
from hr_assistant.api.routes import router
from hr_assistant.core.config import get_settings
from hr_assistant.core.logs import configure_logging
from hr_assistant.db.session import get_async_engine, get_async_session_factory
from hr_assistant.embeddings.vertex import VertexEmbedder
from hr_assistant.generation.answer import AnswerService
from hr_assistant.generation.llm import GeminiLLM
from hr_assistant.retrieval.search import Retriever

logger = logging.getLogger(__name__)


async def check_embedding_model(session_factory: async_sessionmaker[AsyncSession], configured: str) -> None:
    """The offline/online contract: questions must be embedded with the model that embedded the chunks."""
    async with session_factory() as session:
        indexed = set((await session.scalars(text("SELECT DISTINCT embedding_model FROM chunks"))).all())
    if not indexed:
        logger.warning("the index is empty: every question will get the no-answer message",
                       extra={"event": "index_empty"})
    elif configured not in indexed:
        raise RuntimeError(f"EMBEDDING_MODEL={configured!r} but the index was built with {sorted(indexed)}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the expensive objects once, at startup; share them across all requests."""
    settings = get_settings()
    session_factory = get_async_session_factory()
    await check_embedding_model(session_factory, settings.embedding_model)

    app.state.session_factory = session_factory
    app.state.answer_service = AnswerService(
        retriever=Retriever(
            embedder=VertexEmbedder.from_settings(settings),
            session_factory=session_factory,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
            ef_search=settings.hnsw_ef_search,
        ),
        llm=GeminiLLM.from_settings(settings),
         messages=NoAnswerMessages(
            not_found=settings.not_found_message,
            off_topic=settings.off_topic_message,
            no_context=settings.no_context_message,
        ),
    )
    logger.info("API started", extra={"event": "api_started", "environment": settings.environment,
                                       "dev_auth": settings.dev_auth_enabled})
    yield
    await get_async_engine().dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    is_prod = settings.environment == "prod"
    app = FastAPI(
        title="HR Policy Assistant",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if is_prod else "/docs",  # no public API explorer in production
        redoc_url=None,
        openapi_url=None if is_prod else "/openapi.json",
    )
    app.middleware("http")(request_context)
    app.include_router(router)
    return app


def _create_server_app() -> FastAPI:
    """The process entry point uvicorn loads: configure logging once, then build the app.
    (Tests call create_app() directly, so they keep pytest's own log capture.)"""
    settings = get_settings()
    configure_logging(json_logs=settings.json_logs, level=settings.log_level)
    return create_app()


app = _create_server_app()
