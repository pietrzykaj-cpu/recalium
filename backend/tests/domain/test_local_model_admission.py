"""Real static validation + synthetic PostgreSQL: rejected admission is atomic."""
import pytest
from unittest.mock import AsyncMock
from sqlalchemy import select

from app.domain.derivations import service as g
from app.domain.derivations.models import ArchiveDerivationHead as Head, DerivationGeneration as Gen
from app.domain.jobs.models import Job
from app.infrastructure import local_model as lm
from tests.domain.test_local_model_assets import assets
from tests.domain.test_derivation_generations import managed, claimed
from tests.domain.test_completed_reprocess import completed, state


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["completed", "automatic", "manual_failure"])
@pytest.mark.parametrize("damage", ["missing", "corrupt", "wrong_revision"])
async def test_unready_no_generation_head_or_job_reset(db_session, managed, assets, monkeypatch, path, damage):
    if path == "completed":
        _, jid, _ = await completed(db_session)
    else:
        _, job = await claimed(db_session)
        jid = job.id
        if path == "manual_failure":
            job.status = "failed"
            await db_session.commit()
    job = await db_session.get(Job, jid)
    token = g.Claim(job.id, job.raw_archive_id, job.generation_id, job.attempts, job.claimed_at)
    before = await state(db_session)
    await db_session.rollback()
    if damage == "missing":
        (assets / "modules.json").unlink()
    elif damage == "corrupt":
        (assets / "model.safetensors").write_bytes(b"x" * len(b"synthetic weights"))
    else:
        wrong = assets.parent / "unapproved"
        assets.rename(wrong)
        monkeypatch.setattr(lm, "get_settings", lambda: type("Settings", (), {"local_model_assets_dir": str(wrong)})())
    monkeypatch.setattr(g, "require_local_model_ready_async", lm.require_local_model_ready_async)
    with pytest.raises(lm.LocalModelNotReady, match="^local_model_not_ready$"):
        if path == "completed":
            await g.admit_completed_generation(db_session, jid, "legacy")
        else:
            await g.admit_generation(db_session, jid, manual=path == "manual_failure", expected_claim=token)
    assert not db_session.in_transaction()
    assert await state(db_session) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["completed", "automatic"])
async def test_valid_assets_admit_without_transaction_during_validation(db_session, managed, assets, monkeypatch, path):
    if path == "completed":
        _, jid, _ = await completed(db_session)
    else:
        _, job = await claimed(db_session)
        jid = job.id
    job = await db_session.get(Job, jid)
    token = g.Claim(job.id, job.raw_archive_id, job.generation_id, job.attempts, job.claimed_at)
    checks = []
    async def ready():
        assert not db_session.in_transaction()
        checks.append(True)
        return await lm.require_local_model_ready_async()
    monkeypatch.setattr(g, "require_local_model_ready_async", ready)
    if path == "completed":
        result = await g.admit_completed_generation(db_session, jid, "legacy")
    else:
        result = await g.admit_generation(db_session, jid, expected_claim=token)
    assert checks == [True]
    assert result.status == "building"
    head = await db_session.get(Head, token.archive_id)
    assert head.candidate_generation_id == result.id and head.active_generation_id is None
    assert len((await db_session.scalars(select(Gen))).all()) == 1


@pytest.mark.asyncio
async def test_completed_api_unavailable_is_static(client, db_session, managed, assets, monkeypatch):
    _, jid, _ = await completed(db_session)
    before = await state(db_session)
    (assets / "modules.json").unlink()
    monkeypatch.setattr(g, "require_local_model_ready_async", lm.require_local_model_ready_async)
    response = await client.post(f"/api/jobs/{jid}/reprocess", json={"expected_generation": "legacy"})
    assert response.status_code == 503
    assert response.json() == {"detail": "local_model_not_ready"}
    assert await state(db_session) == before


@pytest.mark.asyncio
async def test_automatic_worker_unready_bounded_failure_no_models(db_session, managed, assets, monkeypatch):
    from app.worker import managed_dispatcher as worker
    _, job = await claimed(db_session)
    jid = job.id
    model_work = AsyncMock(side_effect=AssertionError("model work before readiness"))
    monkeypatch.setattr(worker, "dispatch_claim", model_work)
    (assets / "modules.json").unlink()
    monkeypatch.setattr(g, "require_local_model_ready_async", lm.require_local_model_ready_async)
    await worker.dispatch_managed_job(db_session, job)
    final = await db_session.get(Job, jid, populate_existing=True)
    assert final.status == "failed" and final.error_message == "local_model_not_ready"
    assert final.attempts == 1 and final.generation_id is None
    assert not (await db_session.scalars(select(Gen))).all()
    assert not (await db_session.scalars(select(Head))).all()
    model_work.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_active_history_untouched_when_successor_unready(db_session, managed, assets, monkeypatch):
    from app.domain.derived_memory.models import Fact
    _, jid, gid = await completed(db_session, published=True)
    before = await state(db_session)
    (assets / "modules.json").unlink()
    monkeypatch.setattr(g, "require_local_model_ready_async", lm.require_local_model_ready_async)
    with pytest.raises(lm.LocalModelNotReady):
        await g.admit_completed_generation(db_session, jid, gid)
    assert await state(db_session) == before
    assert (await db_session.get(Gen, gid)).status == "active"
    assert (await db_session.scalars(select(Fact))).all()
