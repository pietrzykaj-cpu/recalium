"""Consistent reader snapshots and precomputed local query vectors."""
from contextvars import ContextVar
from sqlalchemy import text
from app.domain.derivations.visibility import enabled

prepared_query = ContextVar("derivation_prepared_query", default=None)

async def prepare_query(query, mode, *, canonical_only=False):
    from app.domain.retrieval.service import _sanitize_fts_query
    query = _sanitize_fts_query(query)
    vector, degraded = None, False
    if mode != "keyword" and not canonical_only:
        from app.domain.derived_memory.service import embed_text
        try:
            vector = await embed_text(query)
        except RuntimeError:
            degraded = True
    return query, vector, degraded

async def begin_reader_snapshot(session):
    if not enabled():
        return
    txn = session.sync_session.get_transaction()
    if txn is not None and session.info.get("derivation_snapshot") is txn:
        return
    if session.new or session.dirty or session.deleted:
        raise RuntimeError("Managed reader requires completed caller writes")
    # Ends only a prior read/prepare transaction, never silently commits mutations.
    await session.rollback()
    await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
    session.info["derivation_snapshot"] = session.sync_session.get_transaction()
