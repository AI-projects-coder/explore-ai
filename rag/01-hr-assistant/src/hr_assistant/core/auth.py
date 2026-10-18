"""Who is asking: the User, and signed tokens that carry it (JWT, HS256)."""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from hr_assistant.core.config import Settings


@dataclass(frozen=True)
class User:
    """Who is asking. Built by the server from a verified token, never from the request body."""

    id: str
    groups: frozenset[str]


# Local development only: stands in for the company's identity provider (e.g. Google Workspace groups).
DEV_USERS: dict[str, frozenset[str]] = {
    "intern_ravi": frozenset({"employee"}),
    "manager_priya": frozenset({"employee", "manager"}),
    "hr_anita": frozenset({"employee", "hr"}),
    "vendor_sam": frozenset({"contractor"}),
}

_ALGORITHM = "HS256"


class InvalidTokenError(Exception):
    """Any problem with a token. Deliberately one type: callers must not reveal which check failed."""


def issue_token(user: User, settings: Settings) -> str:
    now = datetime.now(UTC)
    claims = {
        "sub": user.id,
        "groups": sorted(user.groups),
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_ttl_minutes),
    }
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=_ALGORITHM)


def verify_token(token: str, settings: Settings) -> User:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[_ALGORITHM],  # never trust the algorithm named inside the token
            issuer=settings.jwt_issuer,
            options={"require": ["sub", "groups", "iss", "exp"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidTokenError from exc

    groups = claims["groups"]
    if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
        raise InvalidTokenError
    return User(id=claims["sub"], groups=frozenset(groups))
