"""Saving interactions and feedback. Used by the API."""
import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hr_assistant.core.auth import User
from hr_assistant.db.models import Feedback, Interaction
from hr_assistant.generation.answer import Answer

logger = logging.getLogger(__name__)


async def save_interaction(
    session_factory: async_sessionmaker[AsyncSession], user: User, question: str, answer: Answer
) -> uuid.UUID | None:
    """Log the Q&A. A logging failure must never cost the user their answer, so errors return None."""
    try:
        async with session_factory() as session, session.begin():
            row = Interaction(
                user_id=user.id,
                question=question,
                answer=answer.text,
                answerable=answer.answerable,
                no_answer_reason=answer.no_answer_reason,
                citations=[
                    {"number": c.number, "document_id": str(c.document_id), "heading_path": c.heading_path,
                     "page_start": c.page_start, "page_end": c.page_end}
                    for c in answer.citations
                ],
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
                latency_ms=answer.latency_ms,
            )
            session.add(row)
            await session.flush()  # assigns row.id
            return row.id
    except Exception:
        logger.exception("could not save interaction for user=%s", user.id)
        return None


async def save_feedback(
    session: AsyncSession, user: User, interaction_id: uuid.UUID, rating: str, comment: str | None
) -> bool:
    """False if the interaction doesn't exist OR belongs to someone else (callers return 404 for both)."""
    owner = await session.scalar(select(Interaction.user_id).where(Interaction.id == interaction_id))
    if owner != user.id:
        return False
    stmt = insert(Feedback).values(interaction_id=interaction_id, user_id=user.id, rating=rating, comment=comment)
    await session.execute(
        stmt.on_conflict_do_update(  # clicking again changes the rating instead of adding a row
            index_elements=["interaction_id", "user_id"],
            set_={"rating": stmt.excluded.rating, "comment": stmt.excluded.comment, "updated_at": func.now()},
        )
    )
    await session.commit()
    return True
