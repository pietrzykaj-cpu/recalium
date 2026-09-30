"""Process- and database-wide ownership for Recalium local inference."""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from sqlalchemy import text

from app.infrastructure.db import get_engine


class LocalInferenceError(RuntimeError):
    """Base failure for local-inference ownership and residency policy."""


class LocalInferenceIdentityError(LocalInferenceError):
    """The configured local model is absent or has an unexpected identity."""


class LocalInferenceResidencyError(LocalInferenceError):
    """Ollama residency is unsafe for a new Recalium-owned request."""


class LocalInferenceCleanupError(LocalInferenceError):
    """Post-request model cleanup was not confirmed; the lane is quarantined."""


@dataclass(frozen=True, slots=True)
class ResidentModel:
    model: str
    digest: str


class AdvisoryConnection(Protocol):
    async def execute(self, statement: object, parameters: Mapping[str, object]) -> Any: ...

    async def close(self) -> None: ...


class ResidencyProbe(Protocol):
    async def installed_digest(
        self,
        *,
        base_url: str,
        model: str,
        headers: dict[str, str] | None,
    ) -> str | None: ...

    async def resident_models(
        self,
        *,
        base_url: str,
        headers: dict[str, str] | None,
    ) -> tuple[ResidentModel, ...]: ...


class HttpOllamaResidencyProbe:
    """Read Ollama model metadata without invoking inference."""

    async def _get(
        self,
        *,
        base_url: str,
        path: str,
        headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = await client.get(f"{base_url.rstrip('/')}{path}", headers=headers)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise LocalInferenceResidencyError("Ollama metadata response is not an object")
        return payload

    async def installed_digest(
        self,
        *,
        base_url: str,
        model: str,
        headers: dict[str, str] | None,
    ) -> str | None:
        payload = await self._get(base_url=base_url, path="/api/tags", headers=headers)
        models = payload.get("models")
        if not isinstance(models, list):
            raise LocalInferenceIdentityError("Ollama tags response has no model list")
        matches: set[str] = set()
        for item in models:
            if not isinstance(item, Mapping):
                continue
            item_model = item.get("model") or item.get("name")
            digest = item.get("digest")
            if item_model == model and isinstance(digest, str) and digest:
                matches.add(digest.lower())
        if len(matches) > 1:
            raise LocalInferenceIdentityError(
                f"Ollama reports multiple digests for approved model {model!r}"
            )
        return next(iter(matches), None)

    async def resident_models(
        self,
        *,
        base_url: str,
        headers: dict[str, str] | None,
    ) -> tuple[ResidentModel, ...]:
        payload = await self._get(base_url=base_url, path="/api/ps", headers=headers)
        models = payload.get("models")
        if not isinstance(models, list):
            raise LocalInferenceResidencyError("Ollama residency response has no model list")
        resident: list[ResidentModel] = []
        for item in models:
            if not isinstance(item, Mapping):
                raise LocalInferenceResidencyError("Ollama residency item is not an object")
            model = item.get("model") or item.get("name")
            digest = item.get("digest")
            if not isinstance(model, str) or not model:
                raise LocalInferenceResidencyError("Ollama residency item lacks a model name")
            if not isinstance(digest, str):
                raise LocalInferenceResidencyError("Ollama residency item lacks a digest")
            resident.append(ResidentModel(model=model, digest=digest.lower()))
        return tuple(sorted(resident, key=lambda item: (item.model, item.digest)))


async def _default_connection_factory() -> AdvisoryConnection:
    return await get_engine().connect()


class LocalInferenceCoordinator:
    """A capacity-one Recalium lane for local Ollama execution.

    Ordering is process lock -> PostgreSQL advisory lock -> residency preflight ->
    caller critical section -> cleanup confirmation -> advisory unlock -> process unlock.
    External Ollama clients are outside this ownership guarantee.
    """

    ADVISORY_LOCK_KEY = int.from_bytes(
        hashlib.sha256(b"recalium:local-inference:v1").digest()[:8],
        "big",
        signed=True,
    )

    def __init__(
        self,
        *,
        connection_factory: Callable[[], Awaitable[AdvisoryConnection]] = (
            _default_connection_factory
        ),
        residency_probe: ResidencyProbe | None = None,
        residency_attempts: int = 5,
        residency_poll_seconds: float = 0.25,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if residency_attempts < 1:
            raise ValueError("residency_attempts must be positive")
        if residency_poll_seconds < 0:
            raise ValueError("residency_poll_seconds must not be negative")
        self._connection_factory = connection_factory
        self._residency_probe = residency_probe or HttpOllamaResidencyProbe()
        self._residency_attempts = residency_attempts
        self._residency_poll_seconds = residency_poll_seconds
        self._sleep = sleep
        self._process_lock = asyncio.Lock()
        self._unhealthy_reason: str | None = None
        self._quarantined_connections: list[AdvisoryConnection] = []

    async def _assert_identity(
        self,
        *,
        base_url: str,
        expected_model: str,
        expected_digest: str | None,
        headers: dict[str, str] | None,
    ) -> None:
        installed_digest = await self._residency_probe.installed_digest(
            base_url=base_url,
            model=expected_model,
            headers=headers,
        )
        if installed_digest is None:
            raise LocalInferenceIdentityError(
                f"Approved local model {expected_model!r} is not installed"
            )
        if expected_digest is not None and installed_digest != expected_digest.lower():
            raise LocalInferenceIdentityError(
                f"Approved local model {expected_model!r} digest does not match policy"
            )

    async def _wait_until_unloaded(
        self,
        *,
        base_url: str,
        expected_model: str,
        expected_digest: str | None,
        headers: dict[str, str] | None,
        phase: str,
    ) -> None:
        for attempt in range(self._residency_attempts):
            resident = await self._residency_probe.resident_models(
                base_url=base_url,
                headers=headers,
            )
            if not resident:
                return
            unexpected = tuple(
                item
                for item in resident
                if item.model != expected_model
                or (expected_digest is not None and item.digest != expected_digest.lower())
            )
            if unexpected:
                names = ", ".join(item.model for item in unexpected)
                raise LocalInferenceResidencyError(
                    f"Ollama {phase} found unexpected resident model(s): {names}"
                )
            if attempt + 1 < self._residency_attempts:
                await self._sleep(self._residency_poll_seconds)
        raise LocalInferenceResidencyError(
            f"Ollama {phase} did not unload approved model {expected_model!r}"
        )

    async def _release_advisory(self, connection: AdvisoryConnection) -> None:
        result = await connection.execute(
            text("SELECT pg_advisory_unlock(:lock_key)"),
            {"lock_key": self.ADVISORY_LOCK_KEY},
        )
        if result.scalar_one() is not True:
            raise LocalInferenceError("PostgreSQL did not release the local-inference lock")

    @asynccontextmanager
    async def acquire(
        self,
        *,
        base_url: str,
        expected_model: str,
        expected_digest: str | None,
        headers: dict[str, str] | None,
    ) -> AsyncIterator[None]:
        """Own the sole Recalium local-inference lane for one request."""
        # Deferred to avoid making the infrastructure coordinator participate in
        # the model-context package's public-export import cycle.
        from app.domain.model_context.policy import (
            is_local_inference_endpoint_url,
        )

        if not is_local_inference_endpoint_url(base_url):
            raise PermissionError("Local inference ownership requires a local Ollama endpoint")
        if not expected_model.strip():
            raise LocalInferenceIdentityError("Expected local model must not be blank")

        async with self._process_lock:
            if self._unhealthy_reason is not None:
                raise LocalInferenceCleanupError(
                    f"Local inference lane is unhealthy: {self._unhealthy_reason}"
                )

            connection: AdvisoryConnection | None = None
            advisory_acquired = False
            quarantined = False
            try:
                connection = await self._connection_factory()
                await connection.execute(
                    text("SELECT pg_advisory_lock(:lock_key)"),
                    {"lock_key": self.ADVISORY_LOCK_KEY},
                )
                advisory_acquired = True
                await self._assert_identity(
                    base_url=base_url,
                    expected_model=expected_model,
                    expected_digest=expected_digest,
                    headers=headers,
                )
                await self._wait_until_unloaded(
                    base_url=base_url,
                    expected_model=expected_model,
                    expected_digest=expected_digest,
                    headers=headers,
                    phase="preflight",
                )

                body_error: BaseException | None = None
                try:
                    yield
                except BaseException as exc:  # noqa: BLE001 - cancellation needs cleanup
                    body_error = exc

                try:
                    await self._wait_until_unloaded(
                        base_url=base_url,
                        expected_model=expected_model,
                        expected_digest=expected_digest,
                        headers=headers,
                        phase="cleanup",
                    )
                except BaseException as cleanup_error:  # noqa: BLE001 - quarantine on any exit
                    self._unhealthy_reason = str(cleanup_error)
                    self._quarantined_connections.append(connection)
                    quarantined = True
                    raise LocalInferenceCleanupError(
                        "Local inference cleanup was not confirmed; lane quarantined"
                    ) from body_error or cleanup_error

                if body_error is not None:
                    raise body_error
            finally:
                if connection is not None and not quarantined:
                    try:
                        if advisory_acquired:
                            await self._release_advisory(connection)
                    finally:
                        await connection.close()


_local_inference_coordinator: LocalInferenceCoordinator | None = None


def get_local_inference_coordinator() -> LocalInferenceCoordinator:
    """Return the application-process coordinator; PostgreSQL provides cross-process scope."""
    global _local_inference_coordinator
    if _local_inference_coordinator is None:
        _local_inference_coordinator = LocalInferenceCoordinator()
    return _local_inference_coordinator
