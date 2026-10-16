"""Reprocess API endpoint tests — PIPE-05.

Tests will FAIL (RED) until POST /api/jobs/{job_id}/reprocess endpoint is created.
"""
from __future__ import annotations

import hashlib
import uuid

import pytest
from tests.domain.test_derivation_generations import managed


async def _make_archive_item(session):
    """Helper: insert a minimal RawArchiveItem to satisfy FK constraint on jobs."""
    from app.domain.archive.models import RawArchiveItem
    content = f"test content {uuid.uuid4()}"
    item = RawArchiveItem(
        id=uuid.uuid4(),
        source_type="test",
        raw_content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )
    session.add(item)
    await session.flush()
    return item


@pytest.mark.parametrize("status", ["failed", "retryable_failed", "pending_provider"])
async def test_reprocess_endpoint_returns_200(client, db_session, status):
    """PIPE-05: POST /api/jobs/{job_id}/reprocess resets a failed job to pending."""
    from app.domain.jobs.models import Job

    archive_item = await _make_archive_item(db_session)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status=status,
        attempts=3,
        max_attempts=3,
        error_message="Old error",
    )
    db_session.add(job)
    await db_session.commit()

    response = await client.post(f"/api/jobs/{job.id}/reprocess")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "pending"
    assert data["job_id"] == str(job.id)
    await db_session.refresh(job)
    assert job.attempts == 0 and job.error_message is None
    assert job.max_attempts == 3 and job.raw_archive_id == archive_item.id


@pytest.mark.parametrize("status", ["claimed", "completed", "pending", "terminal_failed", "unsupported"])
async def test_reprocess_conflict_preserves_entire_row(client, db_session, status):
    from datetime import datetime, timezone
    from app.domain.jobs.models import Job

    archive = await _make_archive_item(db_session)
    job = Job(id=uuid.uuid4(), job_type="process_archive_item", raw_archive_id=archive.id,
              status=status, attempts=2, max_attempts=5, error_message="PRIVATE-OLD-ERROR",
              claimed_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc))
    db_session.add(job)
    await db_session.commit()
    job_id = job.id
    before = dict((await db_session.execute(Job.__table__.select())).mappings().one())
    response = await client.post(f"/api/jobs/{job_id}/reprocess")
    assert response.status_code == 409
    assert response.json() == {"detail": "Job cannot be reprocessed in its current state"}
    after = dict((await db_session.execute(Job.__table__.select())).mappings().one())
    assert after == before


async def test_reprocess_unknown_job_returns_404(client):
    """PIPE-05: POST /api/jobs/{unknown_id}/reprocess returns 404."""
    response = await client.post(f"/api/jobs/{uuid.uuid4()}/reprocess")
    assert response.status_code == 404


async def test_reprocess_pending_provider_returns_200(client, db_session):
    """PIPE-05: pending_provider jobs can also be manually re-queued via reprocess endpoint."""
    from app.domain.jobs.models import Job

    archive_item = await _make_archive_item(db_session)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="pending_provider",
        attempts=0,
    )
    db_session.add(job)
    await db_session.commit()

    response = await client.post(f"/api/jobs/{job.id}/reprocess")
    assert response.status_code == 200


async def test_completed_explicit_legacy_success_and_lost_ack_bounded(client, db_session, managed):
    from tests.domain.test_completed_reprocess import completed, state
    _, jid, _ = await completed(db_session)
    success = await client.post(f'/api/jobs/{jid}/reprocess', json={'expected_generation':'legacy'})
    assert success.status_code == 200
    body = success.json()
    assert set(body) == {'job_id','status','generation_id'}
    assert body['job_id'] == str(jid) and body['status'] == 'pending'
    before = await state(db_session)
    replay = await client.post(f'/api/jobs/{jid}/reprocess', json={'expected_generation':'legacy'})
    assert replay.status_code == 409
    assert replay.json() == {'detail': {'reason':'expected_generation_mismatch',
        'current_generation':body['generation_id'], 'current_status':'pending'}}
    assert await state(db_session) == before
    for response in (success,replay):
        assert all(secret not in response.text for secret in ('PRIVATE_OLD_FACT','PRIVATE_SPAN','PRIVATE_JOB_ERROR','unique alpha source'))


@pytest.mark.parametrize('payload', [{}, {'expected_generation':None}, {'expected_generation':'not-a-uuid'}, {'generation_id':'legacy'}])
async def test_completed_ambiguous_or_invalid_body_rejected(client, db_session, managed, payload):
    from tests.domain.test_completed_reprocess import completed, state
    _, jid, _ = await completed(db_session)
    before = await state(db_session)
    assert (await client.post(f'/api/jobs/{jid}/reprocess', json=payload)).status_code == 422
    assert await state(db_session) == before


@pytest.mark.parametrize('expected', ['legacy', str(uuid.UUID(int=1))])
async def test_completed_explicit_off_is_conflict(client, db_session, expected):
    from tests.domain.test_completed_reprocess import completed, state
    _, jid, _ = await completed(db_session)
    before = await state(db_session)
    response = await client.post(f'/api/jobs/{jid}/reprocess', json={'expected_generation':expected})
    assert response.status_code == 409
    assert await state(db_session) == before


@pytest.mark.parametrize('status', ['failed','retryable_failed','pending_provider','claimed','pending','terminal_failed','unsupported'])
async def test_existing_pr18_no_body_on_contract(client, db_session, managed, status):
    from app.domain.jobs.models import Job
    from tests.domain.test_completed_reprocess import state
    archive = await _make_archive_item(db_session)
    job=Job(raw_archive_id=archive.id,job_type='process_archive_item',status=status,attempts=2,max_attempts=4,error_message='PRIVATE_JOB_ERROR')
    db_session.add(job)
    await db_session.commit()
    jid=job.id
    before=await state(db_session)
    response=await client.post(f'/api/jobs/{jid}/reprocess')
    if status in {'failed','retryable_failed','pending_provider'}:
        assert response.status_code == 200 and response.json() == {'job_id':str(jid),'status':'pending'}
    else:
        assert response.status_code == 409
        assert await state(db_session) == before


async def test_explicit_unknown_on_remains_404(client, managed):
    assert (await client.post(f'/api/jobs/{uuid.uuid4()}/reprocess',json={'expected_generation':'legacy'})).status_code == 404


async def test_managed_uuid_success_and_stale_response(client, db_session, managed):
    from tests.domain.test_completed_reprocess import completed, state
    _, jid, gid=await completed(db_session,True)
    before=await state(db_session)
    stale=await client.post(f'/api/jobs/{jid}/reprocess',json={'expected_generation':str(uuid.uuid4())})
    assert stale.status_code == 409 and stale.json()['detail'] == {'reason':'expected_generation_mismatch','current_generation':str(gid),'current_status':'completed'}
    assert await state(db_session) == before
    result=await client.post(f'/api/jobs/{jid}/reprocess',json={'expected_generation':str(gid)})
    assert result.status_code == 200 and result.json()['generation_id'] != str(gid)


@pytest.mark.parametrize('status',['failed','retryable_failed','pending_provider'])
async def test_optional_binding_does_not_disable_existing_failure_path(client, db_session, managed, status):
    from app.domain.jobs.models import Job
    archive=await _make_archive_item(db_session)
    job=Job(raw_archive_id=archive.id,job_type='process_archive_item',status=status,attempts=3)
    db_session.add(job)
    await db_session.commit()
    response=await client.post(f'/api/jobs/{job.id}/reprocess',json={'expected_generation':'legacy'})
    assert response.status_code == 200 and response.json()['status'] == 'pending'


@pytest.mark.parametrize('status', ['failed', 'retryable_failed', 'pending_provider', 'claimed', 'pending', 'terminal_failed', 'completed', 'unsupported'])
@pytest.mark.parametrize('expected', ['legacy', str(uuid.UUID(int=1))])
async def test_off_explicit_fence_disabled_for_every_status(client, db_session, status, expected):
    from app.domain.jobs.models import Job
    from tests.domain.test_completed_reprocess import state
    archive = await _make_archive_item(db_session)
    job = Job(raw_archive_id=archive.id, job_type='process_archive_item', status=status,
              attempts=2, max_attempts=5, error_message='PRIVATE_JOB_ERROR')
    db_session.add(job)
    await db_session.commit()
    jid = job.id
    before = await state(db_session)
    response = await client.post(f'/api/jobs/{jid}/reprocess', json={'expected_generation': expected})
    assert response.status_code == 409
    assert response.json() == {'detail': 'Generation-aware reprocess is disabled'}
    assert await state(db_session) == before


async def test_off_explicit_unknown_disabled_but_no_body_unknown_stays_404(client):
    jid = uuid.uuid4()
    response = await client.post(f'/api/jobs/{jid}/reprocess', json={'expected_generation': 'legacy'})
    assert response.status_code == 409
    assert response.json() == {'detail': 'Generation-aware reprocess is disabled'}
    assert (await client.post(f'/api/jobs/{jid}/reprocess')).status_code == 404
