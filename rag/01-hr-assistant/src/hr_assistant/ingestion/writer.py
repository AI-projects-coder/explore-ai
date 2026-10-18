"""All database writes for ingestion. Every function runs inside a transaction the caller owns."""
import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from hr_assistant.db.models import Chunk, Document, DocumentStatus
from hr_assistant.ingestion.models import ChunkDraft, SourceDocument


def find_document(session: Session, source: SourceDocument) -> Document | None:
    return session.scalar(
        select(Document).where(
            Document.source_type == source.source_type,
            Document.source_uri == source.source_uri,
        )
    )


def reusable_embeddings(session: Session, document_id: uuid.UUID, embedding_model: str) -> dict[str, list[float]]:
    """content_hash -> vector for the document's current chunks, if made by the same model."""
    rows = session.execute(
        select(Chunk.content_hash, Chunk.embedding).where(
            Chunk.document_id == document_id,
            Chunk.embedding_model == embedding_model,
        )
    )
    return {content_hash: list(embedding) for content_hash, embedding in rows}


def save_document(
    session: Session,
    source: SourceDocument,
    existing: Document | None,
    chunks: list[ChunkDraft],
    vectors: list[list[float]],
    embedding_model: str,
) -> Document:
    """Atomic swap: upsert the document row, delete its old chunks, insert the new ones."""
    doc = existing or Document(source_type=source.source_type, source_uri=source.source_uri)
    doc.title = source.title
    doc.content_hash = source.content_hash
    doc.source_version = source.source_version
    doc.allowed_groups = list(source.allowed_groups)
    doc.status = DocumentStatus.ACTIVE
    doc.last_indexed_at = datetime.now(UTC)
    session.add(doc)
    session.flush()  # a new document gets its id here (needed for the chunks' foreign key)

    session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    session.add_all(
        Chunk(
            document_id=doc.id,
            chunk_index=draft.chunk_index,
            heading_path=draft.heading_path,
            content=draft.content,
            content_hash=draft.content_hash,
            page_start=draft.page_start,
            page_end=draft.page_end,
            token_count=draft.token_count,
            allowed_groups=list(source.allowed_groups),
            embedding=vector,
            embedding_model=embedding_model,
        )
        for draft, vector in zip(chunks, vectors, strict=True)
    )
    return doc


def update_access(session: Session, doc: Document, allowed_groups: list[str]) -> None:
    """Permissions changed but content didn't: update both tables, no re-embedding."""
    doc.allowed_groups = list(allowed_groups)
    session.execute(update(Chunk).where(Chunk.document_id == doc.id).values(allowed_groups=list(allowed_groups)))


def active_documents(session: Session, source_type: str, uri_prefix: str) -> list[Document]:
    return list(
        session.scalars(
            select(Document).where(
                Document.source_type == source_type,
                Document.source_uri.startswith(uri_prefix),
                Document.status == DocumentStatus.ACTIVE,
            )
        )
    )


def mark_deleted(session: Session, doc: Document) -> None:
    """Chunks are hard-deleted (never retrievable again); the document row stays for audit."""
    session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    doc.status = DocumentStatus.DELETED
