"""The ingestion pipeline against the real test database: NEW / UNCHANGED / CHANGED / ACCESS / DELETED.

Loader, parser and embedder are fakes (fast, free, offline); the writer and the SQL are real.
Each test uses its own uri_prefix, so tests never see each other's documents.
"""
import hashlib
import uuid
from dataclasses import dataclass, field

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from hr_assistant.core.config import get_settings
from hr_assistant.db.models import EMBEDDING_DIM, Chunk, Document, RunStatus
from hr_assistant.embeddings.vertex import EmbeddingResult
from hr_assistant.ingestion.models import Block, BlockType, ParsedDocument, SourceDocument
from hr_assistant.ingestion.pipeline import run_ingestion

PIPELINE_MODEL = "pipeline-test-model"  # different from the retrieval seed data's model


def source(prefix: str, name: str, text: str, groups: list[str] | None = None, title: str | None = None):
    data = text.encode()
    return SourceDocument(
        source_type="pdf",
        source_uri=f"{prefix}{name}",
        title=title or name,
        allowed_groups=sorted(groups or ["employee"]),
        data=data,
        content_hash=hashlib.sha256(data).hexdigest(),
    )


@dataclass
class FakeLoader:
    uri_prefix: str
    sources: list[SourceDocument] = field(default_factory=list)
    source_type: str = "pdf"

    def load(self):
        return list(self.sources), []


class FakeParser:
    """'## Heading' lines become headings, everything else a paragraph. Raises on the word BROKEN."""

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        text = data.decode()
        if "BROKEN" in text:
            raise ValueError("corrupt PDF")
        blocks = [
            Block(BlockType.HEADING, line[3:], 1, 1) if line.startswith("## ") else Block(BlockType.TEXT, line, 1)
            for line in text.split("\n\n")
        ]
        return ParsedDocument(blocks, 1)


class CountingEmbedder:
    model = PIPELINE_MODEL

    def __init__(self) -> None:
        self.texts_embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        self.texts_embedded += texts
        vectors = []
        for t in texts:
            v = [0.0] * EMBEDDING_DIM
            v[int(hashlib.sha256(t.encode()).hexdigest(), 16) % EMBEDDING_DIM] = 1.0
            vectors.append(v)
        return EmbeddingResult(vectors, token_count=len(texts) * 10)


@pytest.fixture
def env(database):
    engine = create_engine(database)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    # Small min_tokens so each "## section" stays its own chunk.
    settings = get_settings().model_copy(update={"chunk_min_tokens": 0})
    loader = FakeLoader(uri_prefix=f"test-{uuid.uuid4().hex[:8]}://")
    embedder = CountingEmbedder()

    def run(**kwargs):
        return run_ingestion(
            loader=loader,
            parser=FakeParser(),
            embedder=embedder,
            session_factory=factory,
            settings=settings,
            trigger="manual",
            **kwargs,
        )

    def chunks_of(uri: str) -> list[Chunk]:
        with factory() as s:
            doc_id = s.scalar(select(Document.id).where(Document.source_uri == uri))
            return list(s.scalars(select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)))

    def document(uri: str) -> Document:
        with factory() as s:
            return s.scalar(select(Document).where(Document.source_uri == uri))

    yield SimpleEnv(loader, embedder, run, chunks_of, document)
    engine.dispose()


@dataclass
class SimpleEnv:
    loader: FakeLoader
    embedder: CountingEmbedder
    run: object
    chunks_of: object
    document: object


POLICY_V1 = "## 1. Casual Leave\n\nCarry forward up to 4 days.\n\n## 2. Sick Leave\n\nTen days per year."
POLICY_V2 = "## 1. Casual Leave\n\nCarry forward up to 6 days.\n\n## 2. Sick Leave\n\nTen days per year."


def test_first_run_ingests_new_documents(env):
    p = env.loader.uri_prefix
    env.loader.sources = [source(p, "leave.pdf", POLICY_V1)]

    run = env.run()

    assert (run.status, run.docs_new, run.docs_failed) == (RunStatus.SUCCEEDED, 1, 0)
    assert len(env.chunks_of(f"{p}leave.pdf")) == 2
    assert run.embed_tokens == 20


def test_second_run_with_same_data_does_nothing(env):
    env.loader.sources = [source(env.loader.uri_prefix, "leave.pdf", POLICY_V1)]
    env.run()
    embedded_after_first_run = len(env.embedder.texts_embedded)

    run = env.run()

    assert (run.docs_new, run.docs_changed, run.docs_unchanged) == (0, 0, 1)
    assert len(env.embedder.texts_embedded) == embedded_after_first_run  # zero new embedding calls
    assert run.embed_tokens == 0


def test_changed_document_re_embeds_only_changed_chunks(env):
    p = env.loader.uri_prefix
    env.loader.sources = [source(p, "leave.pdf", POLICY_V1)]
    env.run()
    env.embedder.texts_embedded.clear()

    env.loader.sources = [source(p, "leave.pdf", POLICY_V2)]
    run = env.run()

    assert run.docs_changed == 1
    assert len(env.embedder.texts_embedded) == 1  # only "Casual Leave" changed
    assert "6 days" in env.embedder.texts_embedded[0]
    contents = " ".join(c.content for c in env.chunks_of(f"{p}leave.pdf"))
    assert "6 days" in contents and "4 days" not in contents  # old text is gone


def test_permission_change_updates_chunks_without_re_embedding(env):
    p = env.loader.uri_prefix
    env.loader.sources = [source(p, "salary.pdf", POLICY_V1, groups=["employee"])]
    env.run()
    env.embedder.texts_embedded.clear()

    env.loader.sources = [source(p, "salary.pdf", POLICY_V1, groups=["hr", "manager"])]
    run = env.run()

    assert run.details["access_updated"] == 1
    assert env.embedder.texts_embedded == []
    assert env.document(f"{p}salary.pdf").allowed_groups == ["hr", "manager"]
    assert all(c.allowed_groups == ["hr", "manager"] for c in env.chunks_of(f"{p}salary.pdf"))


def test_title_change_re_embeds_because_the_title_is_part_of_the_embedded_text(env):
    p = env.loader.uri_prefix
    env.loader.sources = [source(p, "leave.pdf", POLICY_V1, title="Leave Policy 2026")]
    env.run()
    env.embedder.texts_embedded.clear()

    env.loader.sources = [source(p, "leave.pdf", POLICY_V1, title="Leave Policy 2027")]
    run = env.run()

    assert run.docs_changed == 1
    assert len(env.embedder.texts_embedded) == 2


def test_document_removed_from_source_is_deleted_but_kept_for_audit(env):
    p = env.loader.uri_prefix
    env.loader.sources = [source(p, f"doc{i}.pdf", POLICY_V1) for i in range(3)]
    env.run()

    env.loader.sources = env.loader.sources[:2]  # doc2 removed (1 of 3 = 33%, under the 50% limit)
    run = env.run()

    assert run.docs_deleted == 1
    assert env.chunks_of(f"{p}doc2.pdf") == []  # chunks hard-deleted: never retrievable again
    assert env.document(f"{p}doc2.pdf").status == "deleted"  # row kept: audit trail


def test_mass_deletion_is_aborted_and_nothing_is_deleted(env):
    p = env.loader.uri_prefix
    env.loader.sources = [source(p, f"doc{i}.pdf", POLICY_V1) for i in range(4)]
    env.run()

    env.loader.sources = env.loader.sources[:1]  # 3 of 4 would be deleted: looks like a broken listing
    run = env.run()

    assert run.status == RunStatus.ABORTED
    assert "would delete 3 of 4" in run.error
    assert all(env.document(f"{p}doc{i}.pdf").status == "active" for i in range(4))


def test_mass_deletion_can_be_explicitly_allowed(env):
    p = env.loader.uri_prefix
    env.loader.sources = [source(p, f"doc{i}.pdf", POLICY_V1) for i in range(4)]
    env.run()

    env.loader.sources = env.loader.sources[:1]
    run = env.run(allow_mass_delete=True)

    assert (run.status, run.docs_deleted) == (RunStatus.SUCCEEDED, 3)


def test_empty_listing_never_deletes_everything(env):
    env.loader.sources = [source(env.loader.uri_prefix, "only.pdf", POLICY_V1)]
    env.run()

    env.loader.sources = []  # e.g. expired credentials returned an empty list
    run = env.run()

    assert run.status == RunStatus.ABORTED
    assert env.document(f"{env.loader.uri_prefix}only.pdf").status == "active"


def test_one_broken_document_does_not_stop_the_others(env):
    p = env.loader.uri_prefix
    env.loader.sources = [
        source(p, "good1.pdf", POLICY_V1),
        source(p, "broken.pdf", "BROKEN"),
        source(p, "good2.pdf", POLICY_V2),
    ]

    run = env.run()

    assert run.status == RunStatus.FAILED  # failed overall, so alerting notices...
    assert (run.docs_new, run.docs_failed) == (2, 1)  # ...but the good documents were still ingested
    assert "corrupt PDF" in run.details["failures"][f"{p}broken.pdf"]
    assert env.document(f"{p}broken.pdf") is None


# ---------------------------------------------------------------- one run per source at a time (advisory lock)


def _lock_engine(database: str):
    # NullPool: close() really closes the connection. With a normal pool, close() only returns it to the
    # pool and a session-level advisory lock taken on it would STAY HELD. (The pitfall ingestion_lock avoids.)
    return create_engine(database, poolclass=NullPool)


def _hold_lock(engine, prefix: str):
    """Simulate another ingestion run (e.g. another container) holding the lock for this source."""
    conn = engine.connect()
    assert conn.execute(text("SELECT pg_try_advisory_lock(hashtext(:k))"), {"k": f"ingest:{prefix}"}).scalar()
    conn.commit()
    return conn


def _advisory_locks(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'"
            " AND database = (SELECT oid FROM pg_database WHERE datname = current_database())"
        )).scalar()


def test_run_is_skipped_while_another_run_holds_the_lock(env, database):
    engine = _lock_engine(database)
    p = env.loader.uri_prefix
    env.loader.sources = [source(p, "leave.pdf", POLICY_V1)]
    other_run = _hold_lock(engine, p)
    try:
        run = env.run()
    finally:
        other_run.close()  # closing the connection releases the lock, like a finished (or crashed) process

    assert run.status == RunStatus.ABORTED
    assert "already in progress" in run.error
    assert env.embedder.texts_embedded == []  # no duplicate, paid work
    assert env.document(f"{p}leave.pdf") is None

    assert env.run().status == RunStatus.SUCCEEDED  # once the other run is gone, the next one works
    engine.dispose()


def test_lock_is_released_after_a_normal_run(env, database):
    engine = _lock_engine(database)
    env.loader.sources = [source(env.loader.uri_prefix, "leave.pdf", POLICY_V1)]

    env.run()

    assert _advisory_locks(engine) == 0
    engine.dispose()


def test_lock_is_released_even_if_the_run_crashes(env, database):
    engine = _lock_engine(database)

    def broken_listing():
        raise RuntimeError("source unreachable")

    env.loader.load = broken_listing
    with pytest.raises(RuntimeError):
        env.run()

    assert _advisory_locks(engine) == 0  # a crash must never block future runs
    engine.dispose()


def test_different_sources_do_not_block_each_other(env, database):
    engine = _lock_engine(database)
    env.loader.sources = [source(env.loader.uri_prefix, "leave.pdf", POLICY_V1)]
    other_source = _hold_lock(engine, "gs://some-other-bucket/")
    try:
        run = env.run()
    finally:
        other_source.close()

    assert run.status == RunStatus.SUCCEEDED  # the lock is per source (uri_prefix), not global
    engine.dispose()
