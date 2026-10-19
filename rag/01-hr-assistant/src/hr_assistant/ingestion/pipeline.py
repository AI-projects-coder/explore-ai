"""Offline pipeline: diff the source against the database and apply NEW / CHANGED / UNCHANGED / DELETED."""
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from hr_assistant.core.config import Settings
from hr_assistant.db.models import Document, DocumentStatus, IngestionRun, RunStatus
from hr_assistant.embeddings.vertex import Embedder
from hr_assistant.ingestion import writer
from hr_assistant.ingestion.chunker import chunk_document
from hr_assistant.ingestion.models import Parser, SourceDocument
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text

logger = logging.getLogger(__name__)


class Loader(Protocol):
    source_type: str
    uri_prefix: str

    def load(self) -> tuple[list[SourceDocument], list[str]]: ...


class MassDeletionError(Exception):
    """The run would delete too many documents; most likely the source listing is broken."""


@dataclass
class RunStats:
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    access_updated: int = 0
    deleted: int = 0
    failed: int = 0
    embed_tokens: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    problems: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)


def _run_ingestion_locked(
    *,
    loader: Loader,
    parser: Parser,
    embedder: Embedder,
    session_factory: sessionmaker[Session],
    settings: Settings,
    trigger: str,
    allow_mass_delete: bool = False,
) -> IngestionRun:
    with session_factory.begin() as session:
        run = IngestionRun(trigger=trigger)
        session.add(run)
    logger.info("ingestion run %s started (trigger=%s)", run.id, trigger)

    stats = RunStats()
    status, error = RunStatus.SUCCEEDED, None
    try:
        sources, stats.problems = loader.load()
        to_delete = _plan_deletions(loader, sources, session_factory, settings, allow_mass_delete)

        for source in sources:
            try:
                _ingest_one(source, parser, embedder, session_factory, settings, stats)
            except Exception as exc:  # one bad document must not stop the others
                logger.exception("failed to ingest %s", source.source_uri)
                stats.failed += 1
                stats.failures[source.source_uri] = repr(exc)

        for doc_id in to_delete:
            with session_factory.begin() as session:
                doc = session.get(Document, doc_id)
                writer.mark_deleted(session, doc)
                logger.info("deleted %s", doc.source_uri)
            stats.deleted += 1

        if stats.failed:
            status = RunStatus.FAILED
    except MassDeletionError as exc:
        status, error = RunStatus.ABORTED, str(exc)
        logger.error("run aborted: %s", exc)
    except Exception as exc:
        status, error = RunStatus.FAILED, repr(exc)
        raise
    finally:
        with session_factory.begin() as session:
            run = session.merge(run)
            run.status = status
            run.error = error
            run.finished_at = datetime.now(UTC)
            run.docs_new, run.docs_changed = stats.new, stats.changed
            run.docs_unchanged, run.docs_deleted, run.docs_failed = stats.unchanged, stats.deleted, stats.failed
            run.embed_tokens = stats.embed_tokens
            run.details = asdict(stats)
        logger.info("ingestion run %s finished: %s", run.id, status)
    return run


def _plan_deletions(
    loader: Loader,
    sources: list[SourceDocument],
    session_factory: sessionmaker[Session],
    settings: Settings,
    allow_mass_delete: bool,
) -> list[uuid.UUID]:
    """Active documents from this loader's scope that are no longer in the source. Checked BEFORE any work."""
    seen = {s.source_uri for s in sources}
    with session_factory() as session:
        active = writer.active_documents(session, loader.source_type, loader.uri_prefix)
    to_delete = [d.id for d in active if d.source_uri not in seen]

    if to_delete and not allow_mass_delete:
        fraction = len(to_delete) / len(active)
        if not sources or fraction > settings.ingest_max_delete_fraction:
            raise MassDeletionError(
                f"would delete {len(to_delete)} of {len(active)} active documents ({fraction:.0%}); "
                f"limit is {settings.ingest_max_delete_fraction:.0%}. Re-run with --allow-mass-delete if intended."
            )
    return to_delete


def _ingest_one(
    source: SourceDocument,
    parser: Parser,
    embedder: Embedder,
    session_factory: sessionmaker[Session],
    settings: Settings,
    stats: RunStats,
) -> None:
    # 1. Read current state (short transaction).
    with session_factory() as session:
        existing = writer.find_document(session, source)
        same_content = (
            existing is not None
            and existing.status == DocumentStatus.ACTIVE
            and existing.content_hash == source.content_hash
            and existing.title == source.title
        )
        if same_content:
            if sorted(existing.allowed_groups) != source.allowed_groups:
                writer.update_access(session, existing, source.allowed_groups)
                session.commit()
                stats.access_updated += 1
                logger.info("access updated %s -> %s", source.source_uri, source.allowed_groups)
            else:
                stats.unchanged += 1
            return
        reuse = writer.reusable_embeddings(session, existing.id, embedder.model) if existing else {}

    # 2. Slow work with NO transaction open: parse, chunk, embed only what is new.
    parsed = parser.parse(source.data, source.source_uri)
    drafts = chunk_document(
        parsed,
        source.title,
        max_tokens=settings.chunk_max_tokens,
        min_tokens=settings.chunk_min_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    to_embed = [d for d in drafts if d.content_hash not in reuse]
    fresh: dict[str, list[float]] = {}
    if to_embed:
        result = embedder.embed_documents([d.embed_text for d in to_embed])
        fresh = {d.content_hash: v for d, v in zip(to_embed, result.vectors, strict=True)}
        stats.embed_tokens += result.token_count
    vectors = [reuse[d.content_hash] if d.content_hash in reuse else fresh[d.content_hash] for d in drafts]

    # 3. Atomic swap (one short transaction).
    with session_factory.begin() as session:
        current = writer.find_document(session, source)
        writer.save_document(session, source, current, drafts, vectors, embedder.model)

    stats.chunks_embedded += len(to_embed)
    stats.chunks_reused += len(drafts) - len(to_embed)
    if existing is None:
        stats.new += 1
    else:
        stats.changed += 1
    logger.info(
        "%s %s: %d chunks (%d embedded, %d reused)",
        "new" if existing is None else "changed",
        source.source_uri,
        len(drafts),
        len(to_embed),
        len(drafts) - len(to_embed),
    )


@contextmanager
def ingestion_lock(session_factory: sessionmaker[Session], scope: str) -> Iterator[bool]:
    """Only one run per source scope at a time. Yields False (immediately) if another run holds the lock.

    A Postgres SESSION-level advisory lock, held on ONE dedicated connection for the whole run:
    - if the process crashes, the connection closes and Postgres releases the lock by itself;
    - we commit right after taking it, so no transaction stays open during the slow work (VACUUM-friendly).
    """
    key = f"ingest:{scope}"
    engine = session_factory.kw["bind"]  # the sessionmaker's engine
    with engine.connect() as conn:  # a dedicated connection: NOT shared via a Session/pool between commits
        acquired = bool(conn.execute(text("SELECT pg_try_advisory_lock(hashtext(:key))"), {"key": key}).scalar())
        conn.commit()
        try:
            yield acquired
        finally:
            if acquired:
                conn.execute(text("SELECT pg_advisory_unlock(hashtext(:key))"), {"key": key})
                conn.commit()


def run_ingestion(
    *,
    loader: Loader,
    parser: Parser,
    embedder: Embedder,
    session_factory: sessionmaker[Session],
    settings: Settings,
    trigger: str,
    allow_mass_delete: bool = False,
) -> IngestionRun:
    with ingestion_lock(session_factory, loader.uri_prefix) as acquired:
        if not acquired:
            with session_factory.begin() as session:
                run = IngestionRun(
                    trigger=trigger,
                    status=RunStatus.ABORTED,
                    finished_at=datetime.now(UTC),
                    error=f"another ingestion run for {loader.uri_prefix} is already in progress",
                    details=asdict(RunStats()),
                )
                session.add(run)
            logger.warning("ingestion skipped: %s", run.error)
            return run
        return _run_ingestion_locked(
            loader=loader,
            parser=parser,
            embedder=embedder,
            session_factory=session_factory,
            settings=settings,
            trigger=trigger,
            allow_mass_delete=allow_mass_delete,
        )
