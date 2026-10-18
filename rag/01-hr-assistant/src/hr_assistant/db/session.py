from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from hr_assistant.core.config import get_settings
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


@lru_cache
def get_engine() -> Engine:
    # pool_pre_ping: test a pooled connection before using it, so a dropped connection
    # (DB restart, network blip, Cloud SQL maintenance) is replaced instead of failing the query.
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    # expire_on_commit=False: objects stay readable after commit (e.g. the run summary we print).
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@lru_cache
def get_async_engine() -> AsyncEngine:
    # Same URL and driver (psycopg 3) as the sync engine; SQLAlchemy uses its async mode here.
    return create_async_engine(get_settings().database_url, pool_pre_ping=True, pool_size=5, max_overflow=10)


@lru_cache
def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_async_engine(), expire_on_commit=False)
