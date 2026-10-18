"""Shared test setup: a throwaway Postgres database seeded with known documents and fake vectors.

Requires the local database container: `docker compose up -d`.
"""
import os

# Point the whole app at the TEST database before anything reads settings.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://hr:hr@localhost:5433/hr_assistant_test"
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ.setdefault("GCP_PROJECT", "test-project")  # tests never call GCP; just satisfies Settings
os.environ["JWT_SECRET"] = "test-only-secret-not-used-anywhere-else-0123456789"
os.environ["ENVIRONMENT"] = "local"

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from hr_assistant.core.config import get_settings  # noqa: E402
from hr_assistant.db.models import Chunk, Document  # noqa: E402
from hr_assistant.retrieval.search import Retriever  # noqa: E402
from tests.support import SEED_DOCUMENTS, TEST_EMBEDDING_MODEL, FakeEmbedder, vector_for  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
@pytest.fixture(scope="session")
def database() -> str:
    """Create a fresh test database, run the real migrations, seed it; drop it when the session ends."""
    url = make_url(TEST_DATABASE_URL)
    assert url.database and url.database.endswith("_test"), f"refusing to use non-test database {url.database!r}"

    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{url.database}" WITH (FORCE)'))
        conn.execute(text(f'CREATE DATABASE "{url.database}"'))

    get_settings.cache_clear()
    alembic_cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(TEST_DATABASE_URL)
    with Session(engine) as session, session.begin():
        for doc in SEED_DOCUMENTS:
            document = Document(
                source_type="pdf",
                source_uri=f"test://{doc['title']}",
                title=doc["title"],
                content_hash="test",
                allowed_groups=doc["groups"],
                status=doc["status"],
            )
            session.add(document)
            session.flush()
            for index, name in enumerate(doc["chunks"]):
                session.add(
                    Chunk(
                        document_id=document.id,
                        chunk_index=index,
                        heading_path=f"{doc['title']} > {name}",
                        content=f"content of {name}",
                        content_hash=name,
                        allowed_groups=doc["groups"],
                        embedding=vector_for(name),
                        embedding_model=TEST_EMBEDDING_MODEL,
                    )
                )
    engine.dispose()

    yield TEST_DATABASE_URL

    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{url.database}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
async def retriever(database: str, embedder: FakeEmbedder):
    """A real Retriever against the real test database; only the embedder is fake."""
    engine = create_async_engine(database)
    yield Retriever(embedder=embedder, session_factory=async_sessionmaker(engine), top_k=5)
    await engine.dispose()
