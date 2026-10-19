"""FastAPI dependencies: the ONLY place a request becomes a User."""
from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hr_assistant.core.auth import InvalidTokenError, User, verify_token
from hr_assistant.core.config import Settings, get_settings
from hr_assistant.core.logs import user_id_var
from hr_assistant.generation.answer import AnswerService

_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",  # same message for every failure: don't tell an attacker what was wrong
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> User:
    # async (not a plain def) so it runs in the request's own context: the user id set here is then
    # visible to every log line of the request. A plain def would run in a worker thread's copy.
    if credentials is None:
        raise _UNAUTHORIZED
    try:
        user = verify_token(credentials.credentials, settings)
    except InvalidTokenError:
        raise _UNAUTHORIZED from None
    user_id_var.set(user.id)
    request.state.user_id = user.id  # for the access-log line written by the middleware
    return user


def get_answer_service(request: Request) -> AnswerService:
    return request.app.state.answer_service


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    return request.app.state.session_factory


async def get_db_session(
    factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> AsyncIterator[AsyncSession]:
    async with factory() as session:  # one session per request, closed afterwards
        yield session
