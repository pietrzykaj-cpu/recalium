"""Shared R2 gate and R1 visibility. History/curation/erasure opt out explicitly."""
from sqlalchemy import and_, or_, select, exists, event, text
from sqlalchemy.orm import Session, with_loader_criteria
from app.infrastructure.settings import get_settings
from app.domain.derivations.models import ArchiveDerivationHead

HISTORY = {"derivation_history": True}

def enabled():
    return getattr(get_settings(), "generation_idempotency_enabled", False) is True

def current(model):
    head = ArchiveDerivationHead
    retired_or_managed = exists(select(head.raw_archive_id).where(
        head.raw_archive_id == model.raw_archive_id,
        or_(head.active_generation_id.is_not(None), head.legacy_retired_at.is_not(None)),
    ).correlate_except(head))
    active = exists(select(head.raw_archive_id).where(
        head.raw_archive_id == model.raw_archive_id,
        head.active_generation_id.is_not(None),
        head.active_generation_id == model.generation_id,
    ).correlate_except(head))
    return or_(and_(model.generation_id.is_(None), ~retired_or_managed), active)

def current_sql(alias):
    if not enabled():
        return ""
    if not alias.isidentifier():
        raise ValueError("SQL visibility alias must be a trusted identifier")
    return f""" AND (({alias}.generation_id IS NULL AND NOT EXISTS (
        SELECT 1 FROM archive_derivation_heads dh
        WHERE dh.raw_archive_id = {alias}.raw_archive_id
        AND (dh.active_generation_id IS NOT NULL OR dh.legacy_retired_at IS NOT NULL)))
        OR EXISTS (SELECT 1 FROM archive_derivation_heads dh
        WHERE dh.raw_archive_id = {alias}.raw_archive_id
        AND dh.active_generation_id IS NOT NULL AND dh.active_generation_id = {alias}.generation_id)) """

@event.listens_for(Session, "do_orm_execute")
def ordinary_generation_visibility(state):
    if not enabled() or state.execution_options.get("derivation_history"):
        return
    if state.is_select or state.is_update:
        from app.domain.derived_memory.models import Fact, Summary, Embedding
        for model in (Fact, Summary, Embedding):
            state.statement = state.statement.options(
                with_loader_criteria(model, current(model), include_aliases=True)
            )
