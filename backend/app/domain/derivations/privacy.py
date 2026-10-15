"""All-generation privacy hooks. These are deliberately NOT current-read queries."""
from sqlalchemy import select, text
from app.domain.derivations.models import ArchiveDerivationHead, DerivationGeneration

async def schema_present(session):
    return bool(await session.scalar(text("SELECT to_regclass('derivation_fts_projections')")))

async def lock_existing_heads(session, archive_ids):
    # Caller already holds all source rows in sorted order.
    if not await schema_present(session):
        return
    ids = sorted(set(archive_ids), key=str)
    await session.execute(select(ArchiveDerivationHead).where(
        ArchiveDerivationHead.raw_archive_id.in_(ids)).order_by(ArchiveDerivationHead.raw_archive_id)
        .with_for_update().execution_options(populate_existing=True, autoflush=False))
    await session.execute(select(DerivationGeneration).where(
        DerivationGeneration.raw_archive_id.in_(ids)).order_by(DerivationGeneration.id)
        .with_for_update().execution_options(populate_existing=True, autoflush=False))

async def suppress_generations(session, archive_id):
    if not await schema_present(session):
        return
    await session.execute(text("""UPDATE derivation_fts_projections SET source_status='source_removed'
        WHERE raw_archive_id=:aid"""), {"aid":archive_id})
    await session.execute(text("""UPDATE derivation_generations SET status='abandoned'
        WHERE raw_archive_id=:aid AND status IN ('building','prepared')"""), {"aid":archive_id})
    # Omit retirement column: it survives deletion and the DB trigger forbids reset.
    await session.execute(text("""UPDATE archive_derivation_heads
        SET active_generation_id=NULL, candidate_generation_id=NULL WHERE raw_archive_id=:aid"""),
        {"aid":archive_id})

async def erase_projections(session, archive_id, marker):
    if await schema_present(session):
        await session.execute(text("""UPDATE derivation_fts_projections SET text_content=:marker,
            source_status='source_removed' WHERE raw_archive_id=:aid"""),
            {"aid":archive_id,"marker":marker})

async def assert_restore_safe(session):
    # Full cross-backup retirement reconciliation belongs to the separate restore gate.
    # Until certified, fail closed BEFORE pg_restore rather than silently reset R1.
    from app.domain.derivations.visibility import enabled
    if enabled():
        # Also exclude the first-publication race between this preflight and an
        # old backup's destructive restore. No managed restore is certified here.
        raise RuntimeError("Managed mode requires a separately certified restore plan")
    if await schema_present(session) and await session.scalar(select(
        ArchiveDerivationHead.raw_archive_id).where(
        ArchiveDerivationHead.legacy_retired_at.is_not(None)).limit(1)):
        raise RuntimeError("Managed retirement history requires a separately certified restore plan")
