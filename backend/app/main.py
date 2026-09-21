"""Recalium FastAPI application.

Entrypoint for the ASGI server. Contains:
- App factory with lifespan (DB engine init, startup assertion, shutdown)
- Static file serving for the React frontend build
- API router registration
"""
from __future__ import annotations

import hmac
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as _JSONResponse
from starlette.responses import FileResponse as _FileResponse
from starlette.exceptions import HTTPException as _StarletteHTTPException

from app.api.bridge import bridge_app, bridge_mcp
from app.api.routes import router as api_router
from app.api.routes.search import router as search_router
from app.api.routes.canonical import router as canonical_router
from app.api.routes.review_queue import router as review_queue_router
from app.api.routes.audit import router as audit_router
from app.api.routes.backup import router as backup_router
from app.api.routes.telemetry import router as telemetry_router
from app.api.routes.status import router as status_router
from app.api.routes.portability import router as portability_router
from app.api.routes.tags import router as tags_router
from app.api.routes.facts import router as facts_router
from app.infrastructure.db import get_engine, get_session_factory
from app.infrastructure.settings import get_settings
from app.mcp_server.server import create_mcp_server, mcp_app as _mcp_app

logger = logging.getLogger(__name__)

# Basic logging setup — will be reconfigured in lifespan with proper level from settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")


_AUTH_EXEMPT_PREFIXES = ("/health", "/api/docs", "/api/redoc", "/openapi.json")
_AUTH_REQUIRED_PREFIXES = ("/api/", "/mcp/")


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer token auth for non-localhost deployments (PRIV-06)."""
    async def dispatch(self, request, call_next):
        settings = get_settings()
        if not settings.requires_auth:
            return await call_next(request)
        path = request.url.path
        requires_auth = any(path.startswith(p) for p in _AUTH_REQUIRED_PREFIXES)
        is_exempt = any(path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES)
        if requires_auth and not is_exempt:
            auth_header = request.headers.get("Authorization", "")
            expected = f"Bearer {settings.app_auth_bearer}"
            if not auth_header or not hmac.compare_digest(auth_header, expected):
                return _JSONResponse({"detail": "Authentication required"}, status_code=401)
        return await call_next(request)


def _assert_no_keys_in_schema() -> None:
    """Startup assertion: scan all column names for plaintext key patterns.

    Fails loudly if any ORM model has a column named with a key/secret/token suffix
    that could hold a full API key. This enforces the BYOK contract (D-12, Pitfall 5).

    Allowed exceptions:
    - Columns ending in _fingerprint (stores last 4 chars only)
    - Columns ending in _configured (boolean flag)
    - Columns ending in _validation_status (string status)
    - Columns ending in _validated_at (timestamp)
    """
    from app.infrastructure.db import Base

    # Force all model imports so metadata is populated
    import app.domain.archive.models  # noqa: F401
    import app.domain.settings.models  # noqa: F401
    import app.domain.jobs.models  # noqa: F401
    import app.domain.audit.models  # noqa: F401
    import app.domain.derived_memory.models  # noqa: F401
    import app.domain.canonical_memory.models  # noqa: F401
    import app.domain.review_queue.models  # noqa: F401
    import app.domain.telemetry.models  # noqa: F401

    forbidden_suffixes = ("_key", "_secret", "_token", "_password")
    allowed_suffixes = ("_fingerprint", "_configured", "_validation_status", "_validated_at")

    violations: list[str] = []
    for table in Base.metadata.tables.values():
        for column in table.columns:
            col_name = column.name.lower()
            for forbidden in forbidden_suffixes:
                if col_name.endswith(forbidden):
                    # Check if it's an allowed exception
                    if not any(col_name.endswith(allowed) for allowed in allowed_suffixes):
                        violations.append(f"{table.name}.{column.name}")

    if violations:
        raise RuntimeError(
            f"SECURITY VIOLATION: Columns that may store plaintext API keys found in schema: "
            f"{violations}. "
            "Keys must live in .env only. DB stores only fingerprints and booleans."
        )

    logger.info("Startup assertion passed: no plaintext key columns in schema.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: initialize DB pool on startup, close on shutdown."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s [%(name)s] %(message)s")

    logger.info(f"Starting Recalium in {settings.app_env} mode")

    # Security: enforce auth bearer when binding beyond localhost
    if settings.requires_auth and not settings.app_auth_bearer:
        raise RuntimeError(
            "SECURITY: APP_BIND_HOST is non-localhost but APP_AUTH_BEARER is empty. "
            "Set APP_AUTH_BEARER in .env before exposing Recalium beyond localhost."
        )

    # Initialize DB connection pool
    engine = get_engine()
    get_session_factory()  # Warm up session factory

    # Startup assertion: no plaintext key columns in schema
    _assert_no_keys_in_schema()

    # Start pipeline worker task
    import asyncio as _asyncio
    from app.worker.loop import worker_loop
    _worker_task = _asyncio.create_task(worker_loop(), name="pipeline-worker")
    logger.info("Pipeline worker task started")

    # Start cache invalidation listener (F8: event-driven retrieval-cache clearing)
    from app.domain.retrieval.service import cache_invalidation_listener
    _cache_listener_task = _asyncio.create_task(
        cache_invalidation_listener(), name="cache-invalidation-listener"
    )
    logger.info("Cache invalidation listener task started")

    # Start backup scheduler task (daily at midnight UTC, BKUP-01)
    _backup_task = _asyncio.create_task(_backup_scheduler(), name="backup-scheduler")
    logger.info("Backup scheduler task started")

    # Start file watcher task if configured (INGT-04)
    _watcher_task = None
    if settings.watch_dir:
        from app.domain.ingest.watcher import file_watcher_loop
        _watcher_task = _asyncio.create_task(
            file_watcher_loop(settings.watch_dir, settings.watch_poll_interval),
            name="file-watcher",
        )
        logger.info("File watcher task started: watching %s", settings.watch_dir)
    else:
        logger.info("File watcher disabled (WATCH_DIR not set)")

    # GPT5.6 #21: surface embedding-model drift/staleness at startup so a model
    # change doesn't silently make old vectors unsearchable.
    try:
        from app.domain.derived_memory.service import (
            ACTIVE_EMBEDDING_MODEL,
            embedding_model_health,
        )
        _factory = get_session_factory()
        async with _factory() as _health_session:
            _emb_health = await embedding_model_health(_health_session)
        if settings.embed_model and settings.embed_model != ACTIVE_EMBEDDING_MODEL:
            logger.warning(
                "EMBED_MODEL=%s is configured but the active embedding model is %s. "
                "The setting is not applied automatically — switching models requires a "
                "re-embed migration; retrieval will keep using %s.",
                settings.embed_model, ACTIVE_EMBEDDING_MODEL, ACTIVE_EMBEDDING_MODEL,
            )
        if _emb_health["stale_count"] > 0:
            logger.warning(
                "%d embedding(s) use a model other than the active %s and are excluded "
                "from semantic retrieval until re-embedded. Models present: %s",
                _emb_health["stale_count"], ACTIVE_EMBEDDING_MODEL, _emb_health["models"],
            )
    except Exception as e:
        logger.warning("Embedding-model health check skipped: %s", e)

    logger.info("DB pool initialized. Application ready.")
    logger.info("MCP retrieve_memory tool registered (SSE transport on /mcp/sse)")
    async with bridge_mcp.session_manager.run():
        yield

    # Shutdown pipeline worker cleanly
    _worker_task.cancel()
    try:
        await _worker_task
    except _asyncio.CancelledError:
        pass
    logger.info("Pipeline worker task stopped")

    # Shutdown cache invalidation listener cleanly
    _cache_listener_task.cancel()
    try:
        await _cache_listener_task
    except _asyncio.CancelledError:
        pass
    logger.info("Cache invalidation listener task stopped")

    # Shutdown backup scheduler cleanly
    _backup_task.cancel()
    try:
        await _backup_task
    except _asyncio.CancelledError:
        pass
    logger.info("Backup scheduler task stopped")

    # Shutdown file watcher cleanly
    if _watcher_task is not None:
        _watcher_task.cancel()
        try:
            await _watcher_task
        except _asyncio.CancelledError:
            pass
        logger.info("File watcher task stopped")

    # Shutdown: dispose DB connection pool
    await engine.dispose()
    logger.info("DB pool disposed. Application shutdown complete.")


async def _backup_scheduler() -> None:
    """Run a daily backup at midnight UTC (BKUP-01).

    Waits until the next midnight UTC, creates a backup, then repeats every 24h.
    Cancelled cleanly on shutdown.
    """
    import asyncio as _asyncio
    from datetime import datetime, timezone, timedelta
    from app.domain.backup.service import create_backup, delete_old_backups, DEFAULT_BACKUP_DIR

    while True:
        now = datetime.now(timezone.utc)
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        wait_seconds = (next_midnight - now).total_seconds()
        logger.info("Backup scheduler: next backup in %.0f seconds", wait_seconds)
        try:
            await _asyncio.sleep(wait_seconds)
        except _asyncio.CancelledError:
            break

        try:
            result = await create_backup(backup_dir=DEFAULT_BACKUP_DIR)
            deleted = await delete_old_backups(backup_dir=DEFAULT_BACKUP_DIR, retention_days=30)
            logger.info(
                "Scheduled backup complete: %s, deleted %d old backups",
                result["filename"],
                deleted,
            )
        except Exception as exc:
            logger.error("Scheduled backup failed: %s", exc)


class SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to index.html for unmatched routes (GPT5.6 #7).

    Enables client-side routing: deep links like /facts or /search return the SPA
    shell instead of 404. Unknown backend routes must NOT receive the HTML shell:
    a request to an unmatched ``/api/*`` or ``/mcp/*`` path returns a JSON 404 so
    clients get an honest API error instead of a 200 HTML page (GPT5.6 #7 remainder).
    """

    # Path prefixes that are backend contracts, never client-side SPA routes.
    # StaticFiles normalizes the request path relative to the mount, so these are
    # compared without a leading slash.
    _BACKEND_PREFIXES = ("api/", "mcp/")
    _BACKEND_EXACT = ("api", "mcp")

    async def get_response(self, path: str, scope):  # type: ignore[override]
        try:
            return await super().get_response(path, scope)
        except _StarletteHTTPException as exc:
            if exc.status_code == 404:
                normalized = path.lstrip("/")
                if normalized.startswith(self._BACKEND_PREFIXES) or normalized in self._BACKEND_EXACT:
                    # Unknown backend route: return an honest JSON 404 rather than
                    # masking it as a 200 SPA shell.
                    return _JSONResponse({"detail": "Not Found"}, status_code=404)
                index = os.path.join(str(self.directory), "index.html")
                if os.path.isfile(index):
                    return _FileResponse(index)
            raise


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Recalium API",
        description="Local-first MCP-native personal memory platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.is_development else None,
        redoc_url="/api/redoc" if settings.is_development else None,
    )

    app.add_middleware(AuthMiddleware)

    @app.middleware("http")
    async def add_api_version_header(request, call_next):
        response = await call_next(request)
        response.headers["X-API-Version"] = "1"
        return response

    # API routes under /api prefix
    app.include_router(api_router, prefix="/api")

    # Phase 3 routes (have their own /api prefix)
    app.include_router(search_router)
    app.include_router(canonical_router)
    app.include_router(review_queue_router)
    app.include_router(audit_router)

    # Phase 4 routes (have their own /api prefix)
    app.include_router(backup_router)
    app.include_router(telemetry_router)
    app.include_router(status_router)

    # Phase 5 routes
    app.include_router(portability_router, prefix="/api", tags=["portability"])
    app.include_router(tags_router)
    app.include_router(facts_router)

    app.mount("/bridge", bridge_app)

    # MCP SSE transport — bound to /mcp prefix.
    # SECURITY: Upstream proxy/uvicorn must bind to 127.0.0.1 only (DNS rebinding prevention).
    app.mount("/mcp", _mcp_app.sse_app())

    # Serve React SPA static files (built frontend)
    # In development, Vite dev server handles static; in production, serve from dist.
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        # Mount at / so React Router handles client-side routing
        app.mount("/", SPAStaticFiles(directory=static_dir, html=True), name="static")
        logger.info(f"Serving static files from {static_dir}")
    else:
        logger.info("No static/ directory found — assuming Vite dev server in use.")

    return app


# ASGI application instance (used by uvicorn)
app = create_app()
