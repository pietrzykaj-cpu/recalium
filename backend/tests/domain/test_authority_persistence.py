"""Authority Phase 1B persistence tests; run only against a disposable database."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy import select

from app.domain.authority.models import AuthorityEdgeRow, AuthorityRecordRow
from app.domain.authority.repository import (
    activate_record,
    create_record,
    evaluate_scope,
    list_scope_records,
    mark_disputed,
    supersede,
    withdraw,
)
from app.domain.authority.service import AuthorityValidationError
from app.domain.authority.contracts import AuthorityRecord


NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def rec(status="proposed", *, space="recalium", workstream="bootstrap", key="storage"):
    return AuthorityRecord(
        id=str(uuid4()), space_id=space, workstream_id=workstream, authority_key=key,
        record_kind="decision", content="synthetic decision", lifecycle_status=status,
        created_at=NOW, created_by="test", provenance={"source": "synthetic"},
    )


@pytest.mark.asyncio
async def test_round_trip_and_current_state(db_session):
    a = rec("active")
    await create_record(db_session, a)
    await db_session.commit()
    loaded = (await list_scope_records(db_session, space_id="recalium", workstream_id="bootstrap", authority_key="storage"))[0]
    assert loaded.provenance == {"source": "synthetic"}
    assert (await evaluate_scope(db_session, space_id="recalium", workstream_id="bootstrap", authority_key="storage")).current_record.id == a.id


@pytest.mark.asyncio
async def test_supersession_persists_atomically_and_reload_is_stable(db_session):
    a, b = rec("active"), rec()
    await create_record(db_session, a)
    await create_record(db_session, b)
    await db_session.commit()
    await supersede(db_session, b.id, a.id, created_at=NOW, created_by="test", reason="replacement", provenance={"source": "synthetic"})
    await db_session.commit()
    result = await evaluate_scope(db_session, space_id="recalium", workstream_id="bootstrap", authority_key="storage")
    assert result.current_record.id == b.id and [r.id for r in result.historical_records] == [a.id]
    db_session.expire_all()
    reloaded = await evaluate_scope(db_session, space_id="recalium", workstream_id="bootstrap", authority_key="storage")
    assert reloaded.model_dump() == result.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", [withdraw, mark_disputed])
async def test_successor_withdrawal_or_dispute_does_not_revive_predecessor(db_session, operation):
    a, b = rec("active"), rec()
    await create_record(db_session, a)
    await create_record(db_session, b)
    await db_session.commit()
    await supersede(db_session, b.id, a.id, created_at=NOW, created_by="test")
    await db_session.commit()
    await operation(db_session, b.id, reason="synthetic")
    await db_session.commit()
    assert (await evaluate_scope(db_session, space_id="recalium", workstream_id="bootstrap", authority_key="storage")).status == "empty"


@pytest.mark.asyncio
async def test_competition_and_proposed_are_deterministic(db_session):
    a, b = rec("active"), rec("active")
    await create_record(db_session, a)
    await create_record(db_session, b)
    await db_session.commit()
    assert (await evaluate_scope(db_session, space_id="recalium", workstream_id="bootstrap", authority_key="storage")).status == "ambiguous"
    c = rec()
    await create_record(db_session, c)
    await db_session.commit()
    assert len((await evaluate_scope(db_session, space_id="recalium", workstream_id="bootstrap", authority_key="storage")).competing_records) == 2


@pytest.mark.asyncio
async def test_invalid_operations_leave_no_partial_state(db_session):
    a, b = rec("active"), rec()
    await create_record(db_session, a)
    await create_record(db_session, b)
    await db_session.commit()
    with pytest.raises(AuthorityValidationError):
        await supersede(db_session, b.id, a.id, created_at=NOW, created_by="test")
        await supersede(db_session, b.id, a.id, created_at=NOW, created_by="test")
    await db_session.rollback()
    rows = (await db_session.execute(select(AuthorityEdgeRow))).scalars().all()
    assert rows == []
    row = await db_session.get(AuthorityRecordRow, UUID(b.id))
    assert row.lifecycle_status == "proposed"


@pytest.mark.asyncio
async def test_cross_scope_and_cycle_rejected(db_session):
    a, b = rec("active"), rec(space="other")
    await create_record(db_session, a)
    await create_record(db_session, b)
    await db_session.commit()
    with pytest.raises(AuthorityValidationError):
        await supersede(db_session, b.id, a.id, created_at=NOW, created_by="test")
    await db_session.rollback()
    k = rec(key="other")
    await create_record(db_session, k)
    await db_session.commit()
    with pytest.raises(AuthorityValidationError):
        await supersede(db_session, k.id, a.id, created_at=NOW, created_by="test")
    await db_session.rollback()
    w = rec(workstream="other")
    await create_record(db_session, w)
    await db_session.commit()
    with pytest.raises(AuthorityValidationError):
        await supersede(db_session, w.id, a.id, created_at=NOW, created_by="test")
    await db_session.rollback()
    x, y, z = rec("active"), rec(), rec()
    await create_record(db_session, x)
    await create_record(db_session, y)
    await create_record(db_session, z)
    await db_session.commit()
    await supersede(db_session, y.id, x.id, created_at=NOW, created_by="test")
    await db_session.commit()
    await supersede(db_session, z.id, y.id, created_at=NOW, created_by="test")
    await db_session.commit()
    with pytest.raises(AuthorityValidationError):
        await supersede(db_session, x.id, z.id, created_at=NOW, created_by="test")


@pytest.mark.asyncio
async def test_activation_is_persisted(db_session):
    item = rec()
    await create_record(db_session, item)
    await db_session.commit()
    await activate_record(db_session, item.id)
    await db_session.commit()
    assert (await evaluate_scope(db_session, space_id="recalium", workstream_id="bootstrap", authority_key="storage")).current_record.id == item.id
