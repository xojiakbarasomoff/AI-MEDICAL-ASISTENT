from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(get_settings().database_url)


@lru_cache
def _get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one AsyncSession per request, closed afterward.

    Routes depend on this via Depends(get_db_session); tests override it
    (app.dependency_overrides[get_db_session] = ...) to hand back a
    transactional, rolled-back session instead of hitting the real engine.
    """
    async with _get_sessionmaker()() as session:
        yield session
