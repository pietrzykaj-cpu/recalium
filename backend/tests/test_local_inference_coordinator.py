"""Deterministic unit certification for the capacity-one local inference lane."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from app.infrastructure.local_inference import (
    HttpOllamaResidencyProbe,
    LocalInferenceCleanupError,
    LocalInferenceCoordinator,
    LocalInferenceIdentityError,
    LocalInferenceResidencyError,
    ResidentModel,
)


@pytest.fixture(autouse=True)
def _clean_db_between_tests() -> None:
    """Override the integration autouse fixture; the advisory connection is faked."""


class ScalarResult:
    def __init__(self, value: object) -> None:
        self.value = value

    def scalar_one(self) -> object:
        return self.value


class SharedAdvisoryGate:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()


class FakeConnection:
    def __init__(self, gate: SharedAdvisoryGate | None = None) -> None:
        self.gate = gate or SharedAdvisoryGate()
        self.acquired = False
        self.closed = False
        self.statements: list[str] = []

    async def execute(self, statement: object, parameters: dict[str, object]) -> ScalarResult:
        sql = str(statement)
        self.statements.append(sql)
        assert parameters["lock_key"] == LocalInferenceCoordinator.ADVISORY_LOCK_KEY
        if "pg_advisory_lock(" in sql and "unlock" not in sql:
            await self.gate.lock.acquire()
            self.acquired = True
            return ScalarResult(None)
        if "pg_advisory_unlock(" in sql:
            if self.acquired:
                self.gate.lock.release()
                self.acquired = False
            return ScalarResult(True)
        raise AssertionError(f"Unexpected SQL: {sql}")

    async def close(self) -> None:
        if self.acquired:
            self.gate.lock.release()
            self.acquired = False
        self.closed = True


class FakeConnectionFactory:
    def __init__(
        self,
        *,
        gate: SharedAdvisoryGate | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.gate = gate or SharedAdvisoryGate()
        self.failure = failure
        self.connections: list[FakeConnection] = []

    async def __call__(self) -> FakeConnection:
        if self.failure is not None:
            raise self.failure
        connection = FakeConnection(self.gate)
        self.connections.append(connection)
        return connection


@dataclass
class FakeProbe:
    digest: str = "digest"
    snapshots: list[tuple[ResidentModel, ...]] | None = None

    def __post_init__(self) -> None:
        if self.snapshots is None:
            self.snapshots = [(), ()]
        self.snapshot_calls = 0

    async def installed_digest(
        self,
        *,
        base_url: str,
        model: str,
        headers: dict[str, str] | None,
    ) -> str | None:
        assert base_url == "http://127.0.0.1:11434"
        assert model == "qwen3:4b"
        return self.digest

    async def resident_models(
        self,
        *,
        base_url: str,
        headers: dict[str, str] | None,
    ) -> tuple[ResidentModel, ...]:
        assert self.snapshots is not None
        index = min(self.snapshot_calls, len(self.snapshots) - 1)
        self.snapshot_calls += 1
        return self.snapshots[index]


def coordinator(
    *,
    factory: FakeConnectionFactory | None = None,
    probe: FakeProbe | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> LocalInferenceCoordinator:
    return LocalInferenceCoordinator(
        connection_factory=factory or FakeConnectionFactory(),
        residency_probe=probe or FakeProbe(),
        residency_attempts=2,
        residency_poll_seconds=0,
        sleep=sleep,
    )


def lane(instance: LocalInferenceCoordinator):
    return instance.acquire(
        base_url="http://127.0.0.1:11434",
        expected_model="qwen3:4b",
        expected_digest="digest",
        headers=None,
    )


@pytest.mark.asyncio
async def test_same_process_second_request_cannot_enter_while_first_owns_lane() -> None:
    instance = coordinator()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        async with lane(instance):
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        async with lane(instance):
            second_entered.set()

    first_task = asyncio.create_task(first())
    await first_entered.wait()
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    release_first.set()
    await first_task
    await second_task
    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_independent_coordinators_share_cross_process_advisory_exclusion() -> None:
    gate = SharedAdvisoryGate()
    first = coordinator(factory=FakeConnectionFactory(gate=gate))
    second = coordinator(factory=FakeConnectionFactory(gate=gate))
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def own_first() -> None:
        async with lane(first):
            first_entered.set()
            await release_first.wait()

    async def own_second() -> None:
        async with lane(second):
            second_entered.set()

    first_task = asyncio.create_task(own_first())
    await first_entered.wait()
    second_task = asyncio.create_task(own_second())
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    release_first.set()
    await first_task
    await second_task
    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_success_releases_advisory_connection_and_process_lane() -> None:
    factory = FakeConnectionFactory()
    instance = coordinator(factory=factory)

    async with lane(instance):
        pass

    assert factory.connections[0].statements == [
        "SELECT pg_advisory_lock(:lock_key)",
        "SELECT pg_advisory_unlock(:lock_key)",
    ]
    assert factory.connections[0].closed is True
    assert factory.gate.lock.locked() is False
    async with lane(instance):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("transport"), ValueError("parser"), TimeoutError()])
async def test_failure_releases_both_locks(failure: BaseException) -> None:
    factory = FakeConnectionFactory()
    instance = coordinator(factory=factory)

    with pytest.raises(type(failure)):
        async with lane(instance):
            raise failure

    async with lane(instance):
        pass
    assert all(connection.closed for connection in factory.connections)
    assert not factory.gate.lock.locked()


@pytest.mark.asyncio
async def test_cancellation_releases_lane_for_next_request() -> None:
    instance = coordinator()
    entered = asyncio.Event()

    async def owner() -> None:
        async with lane(instance):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(owner())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with lane(instance):
        pass


@pytest.mark.asyncio
async def test_advisory_connection_failure_fails_closed() -> None:
    instance = coordinator(factory=FakeConnectionFactory(failure=RuntimeError("db down")))

    with pytest.raises(RuntimeError, match="db down"):
        async with lane(instance):
            raise AssertionError("must not enter")


@pytest.mark.asyncio
async def test_advisory_lock_acquisition_failure_fails_closed() -> None:
    connection = FakeConnection()

    async def fail_lock(statement: object, parameters: dict[str, object]) -> ScalarResult:
        assert "pg_advisory_lock(" in str(statement)
        raise RuntimeError("advisory unavailable")

    connection.execute = fail_lock  # type: ignore[method-assign]

    async def connection_factory() -> FakeConnection:
        return connection

    instance = coordinator(factory=connection_factory)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="advisory unavailable"):
        async with lane(instance):
            raise AssertionError("must not enter")
    assert connection.closed is True


@pytest.mark.asyncio
async def test_model_digest_mismatch_fails_before_critical_section() -> None:
    instance = coordinator(probe=FakeProbe(digest="wrong"))

    with pytest.raises(LocalInferenceIdentityError, match="digest"):
        async with lane(instance):
            raise AssertionError("must not enter")


@pytest.mark.asyncio
async def test_unexpected_resident_model_fails_closed() -> None:
    resident = ResidentModel(model="other:latest", digest="other")
    instance = coordinator(probe=FakeProbe(snapshots=[(resident,)]))

    with pytest.raises(LocalInferenceResidencyError, match="unexpected"):
        async with lane(instance):
            raise AssertionError("must not enter")


@pytest.mark.asyncio
async def test_expected_resident_model_is_waited_out_before_entry() -> None:
    resident = ResidentModel(model="qwen3:4b", digest="digest")
    probe = FakeProbe(snapshots=[(resident,), (), ()])
    instance = coordinator(probe=probe)

    async with lane(instance):
        pass

    assert probe.snapshot_calls >= 3


@pytest.mark.asyncio
async def test_cleanup_failure_quarantines_lane_and_blocks_next_request() -> None:
    resident = ResidentModel(model="qwen3:4b", digest="digest")
    factory = FakeConnectionFactory()
    probe = FakeProbe(snapshots=[(), (resident,), (resident,)])
    instance = coordinator(factory=factory, probe=probe)

    with pytest.raises(LocalInferenceCleanupError, match="cleanup"):
        async with lane(instance):
            pass

    assert factory.gate.lock.locked()
    assert factory.connections[0].closed is False
    with pytest.raises(LocalInferenceCleanupError, match="unhealthy"):
        async with lane(instance):
            raise AssertionError("must not enter")


@pytest.mark.asyncio
async def test_nonlocal_endpoint_fails_before_advisory_connection() -> None:
    factory = FakeConnectionFactory()
    instance = coordinator(factory=factory)

    with pytest.raises(PermissionError, match="local"):
        async with instance.acquire(
            base_url="https://remote.example.com",
            expected_model="qwen3:4b",
            expected_digest="digest",
            headers=None,
        ):
            raise AssertionError("must not enter")
    assert factory.connections == []


@pytest.mark.asyncio
async def test_http_probe_reads_installed_identity_and_residency_metadata() -> None:
    probe = HttpOllamaResidencyProbe()
    probe._get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {
                "models": [
                    {
                        "name": "qwen3:4b",
                        "model": "qwen3:4b",
                        "digest": "ABCDEF",
                    }
                ]
            },
            {
                "models": [
                    {
                        "name": "qwen3:4b",
                        "model": "qwen3:4b",
                        "digest": "ABCDEF",
                    }
                ]
            },
        ]
    )

    digest = await probe.installed_digest(
        base_url="http://127.0.0.1:11434",
        model="qwen3:4b",
        headers=None,
    )
    resident = await probe.resident_models(
        base_url="http://127.0.0.1:11434",
        headers=None,
    )

    assert digest == "abcdef"
    assert resident == (ResidentModel(model="qwen3:4b", digest="abcdef"),)
    assert [call.kwargs["path"] for call in probe._get.await_args_list] == [
        "/api/tags",
        "/api/ps",
    ]
