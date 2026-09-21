# backend/app/infrastructure/db.py
"""SQLAlchemy async engine, session factory, and declarative Base.

All SQLAlchemy ORM models must inherit from Base defined here.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.infrastructure.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


_engine = None
_async_session_factory = None


def get_database_url() -> str:
    url = get_settings().database_url
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not configured in settings. "
            "Copy .env.sample to .env and configure it."
        )
    return url


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            get_database_url(),
            echo=False,  # Memory content and authentication digests must not enter SQL logs.
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an async DB session per request."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
