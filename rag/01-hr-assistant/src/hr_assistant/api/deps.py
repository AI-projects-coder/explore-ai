"""FastAPI dependencies: the ONLY place a request becomes a User."""
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from hr_assistant.core.auth import InvalidTokenError, User, verify_token
from hr_assistant.core.config import Settings, get_settings
from hr_assistant.generation.answer import AnswerService
from collections.abc import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",  # same message for every failure: don't tell an attacker what was wrong
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> User:
    if credentials is None:
        raise _UNAUTHORIZED
    try:
        return verify_token(credentials.credentials, settings)
    except InvalidTokenError:
        raise _UNAUTHORIZED from None


def get_answer_service(request: Request) -> AnswerService:
    return request.app.state.answer_service

def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    return request.app.state.session_factory


async def get_db_session(
    factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> AsyncIterator[AsyncSession]:
    async with factory() as session:  # one session per request, closed afterwards
        yield session
