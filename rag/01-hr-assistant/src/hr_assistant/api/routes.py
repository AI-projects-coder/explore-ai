import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from google.genai import errors
from sqlalchemy import text

from hr_assistant.api.deps import get_answer_service, get_current_user
from hr_assistant.api.schemas import AskRequest, AskResponse, DevTokenRequest, TokenResponse
from hr_assistant.core.auth import DEV_USERS, User, issue_token
from hr_assistant.core.config import Settings, get_settings
from hr_assistant.generation.answer import AnswerService
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hr_assistant.api.deps import get_answer_service, get_current_user, get_db_session, get_session_factory
from hr_assistant.api.schemas import AskRequest, AskResponse, DevTokenRequest, FeedbackRequest, TokenResponse
from hr_assistant.db.interactions import save_feedback, save_interaction

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up. No dependencies, so a database outage doesn't restart every container."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    """Readiness: can we serve traffic right now?"""
    try:
        async with request.app.state.session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        logger.exception("readiness check failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "database unavailable") from None
    return {"status": "ready"}


@router.post("/auth/dev-token", response_model=TokenResponse)
async def dev_token(body: DevTokenRequest, settings: Settings = Depends(get_settings)) -> TokenResponse:
    if not settings.dev_auth_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not Found")  # outside local, this endpoint doesn't exist
    groups = DEV_USERS.get(body.user_id)
    if groups is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown dev user; choose one of {sorted(DEV_USERS)}")
    token = issue_token(User(body.user_id, groups), settings)
    return TokenResponse(access_token=token, expires_in=settings.jwt_ttl_minutes * 60)


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    user: User = Depends(get_current_user),
    service: AnswerService = Depends(get_answer_service),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> AskResponse:
    try:
        answer = await service.answer(body.question, user)
    except (errors.APIError, httpx.HTTPError):
        logger.exception("upstream AI service failed for user=%s", user.id)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The assistant is temporarily unavailable. Please try again."
        ) from None
    interaction_id = await save_interaction(session_factory, user, body.question, answer)
    return AskResponse.from_answer(answer, interaction_id)

@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(
    body: FeedbackRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    if not await save_feedback(session, user, body.interaction_id, body.rating, body.comment):
        # Same 404 for "doesn't exist" and "not yours": never confirm that someone else's id exists.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not Found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
