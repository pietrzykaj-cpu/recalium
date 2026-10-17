"""B1/B2 regression tests: real static assets, synthetic PG, no ML/network."""
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.derivations import service as g
from app.domain.derivations.models import ArchiveDerivationHead as Head, DerivationGeneration as Gen
from app.domain.derived_memory import service as writers
from app.domain.derived_memory.models import Fact
from app.domain.jobs.models import Job
from app.infrastructure import local_model as lm
from app.infrastructure.settings import get_settings
from tests.domain.test_local_model_assets import assets
from tests.domain.test_derivation_generations import managed, claimed
from tests.domain.test_completed_reprocess import completed, state

pytestmark = pytest.mark.asyncio


async def seed_visible(session, on):
    if on:
        aid, jid, gid = await completed(session, published=True)
    else:
        source, job = await claimed(session)
        aid, jid, gid = source.id, job.id, None
        await writers.write_facts(session, aid, [{"fact_text": "alpha legacy evidence"}])
        await writers.write_fts_entry(session, aid, "alpha legacy evidence")
        job = await session.get(Job, jid)
        job.status = "completed"
        await session.commit()
    ids = {str(f.id) for f in (await session.scalars(select(Fact))).all()}
    assert ids
    return aid, jid, gid, ids


def unavailable(assets, monkeypatch, damage):
    monkeypatch.setattr(writers, "_embed_model", None)
    monkeypatch.setattr(writers, "_SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(writers, "_SentenceTransformer", None)
    if damage == "missing":
        (assets / "modules.json").unlink()
    else:
        (assets / "model.safetensors").write_bytes(b"x" * len(b"synthetic weights"))


@pytest.mark.parametrize("on", [True, False], ids=["managed-G1", "legacy-OFF"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
@pytest.mark.parametrize("mode", ["default", "hybrid", "semantic"])
async def test_b1_api_degrades_preserving_existing_reads(client, db_session, managed, assets, monkeypatch, on, damage, mode):
    from app.domain.retrieval.service import invalidate_cache
    _, _, _, expected = await seed_visible(db_session, on)
    monkeypatch.setattr(get_settings(), "generation_idempotency_enabled", on)
    before = await state(db_session)
    unavailable(assets, monkeypatch, damage)
    invalidate_cache()
    params = {"q": "alpha"}
    if mode != "default":
        params["mode"] = mode
    response = await client.get("/api/search", params=params)
    assert response.status_code == 200
    body = response.json()
    assert body["degraded_mode"] is True
    if mode != "semantic":
        assert body["retrieval_mode"] == "keyword"
        assert expected <= {item["id"] for item in body["items"]}
    # Preserve existing semantic-only semantics; explicit keyword still works.
    keywords = (await client.get("/api/search", params={"q": "alpha", "mode": "keyword"})).json()
    assert expected <= {item["id"] for item in keywords["items"]}
    assert keywords["degraded_mode"] is False
    assert await state(db_session) == before


@pytest.mark.parametrize("on", [True, False], ids=["managed-G1", "legacy-OFF"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
async def test_b1_mcp_default_returns_keywords_not_internal_error(db_session, test_engine, managed, assets, monkeypatch, on, damage):
    from app.mcp_server import server as mcp
    from app.domain.retrieval.service import invalidate_cache
    _, _, _, expected = await seed_visible(db_session, on)
    monkeypatch.setattr(get_settings(), "generation_idempotency_enabled", on)
    before = await state(db_session)
    await db_session.rollback()
    unavailable(assets, monkeypatch, damage)
    invalidate_cache()
    monkeypatch.setattr(mcp, "get_session_factory", lambda: async_sessionmaker(test_engine, expire_on_commit=False))
    body = await mcp.retrieve_memory(query="alpha")
    assert "error" not in body
    assert body["degraded_mode"] is True and body["retrieval_mode"] == "keyword"
    assert expected <= {item["id"] for item in body["items"]}
    assert await state(db_session) == before


@pytest.mark.parametrize("path", ["managed-legacy", "managed-successor", "legacy-OFF"])
async def test_b2_constructor_failure_after_readiness_never_publishes(db_session, managed, assets, monkeypatch, path):
    from app.domain.policy import gate
    from app.domain.jobs.service import reprocess_job, claim_next_job
    from app.worker import dispatcher as d
    from app.worker import managed_dispatcher as worker
    on = path != "legacy-OFF"
    aid, jid, old_gid, visible_ids = await seed_visible(db_session, path == "managed-successor")
    if path == "managed-successor":
        await reprocess_job(db_session, jid, expected_generation=old_gid)
        await db_session.rollback()
        job = await claim_next_job(db_session)
    else:
        job = await db_session.get(Job, jid)
        job.status = "claimed"
        from datetime import datetime, timezone
        job.claimed_at = datetime.now(timezone.utc)
        job.attempts = 1
        await db_session.commit()
    monkeypatch.setattr(get_settings(), "generation_idempotency_enabled", on)
    before = await state(db_session)
    await db_session.rollback()
    assert lm.require_local_model_ready() == assets.resolve()
    monkeypatch.setattr(g, "require_local_model_ready_async", lm.require_local_model_ready_async)
    constructor = Mock(side_effect=RuntimeError("PRIVATE_LOADER /secret/path https://private/token"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    monkeypatch.setattr(gate, "_gate_model", None)
    monkeypatch.setattr(gate, "_prototype_embeddings", None)
    # Run real classify_async -> _embedding_classify -> shared loader -> constructor.
    # Only downstream provider/enrichment work is replaced by no-call spies.
    monkeypatch.setattr(d, "_gate", gate.SensitivityGate())
    summarize = AsyncMock(side_effect=AssertionError("provider must not run"))
    extract = AsyncMock(side_effect=AssertionError("provider must not run"))
    monkeypatch.setattr(d, "_run_summarize_job", summarize)
    monkeypatch.setattr(d, "_run_extract_job", extract)
    async def finish_without_optional_enrichment(session, claim, **kwargs):
        await g.finish_claim(session, claim)
    monkeypatch.setattr(worker, "enrich_and_finish", finish_without_optional_enrichment)
    job = await db_session.get(Job, jid, populate_existing=True)
    await d.dispatch_job(db_session, job)
    final = await db_session.get(Job, jid, populate_existing=True)
    assert final.status == "retryable_failed"
    assert final.attempts == 1
    assert "LocalModelNotReady" in final.error_message if on else final.error_message == "Sensitivity gate error: local_model_not_ready"
    assert all(secret not in final.error_message for secret in ["PRIVATE_LOADER", "/secret", "https://", "token"])
    constructor.assert_called_once()
    summarize.assert_not_awaited()
    extract.assert_not_awaited()
    after = await state(db_session)
    assert after["derivation_stage_completions"] == before["derivation_stage_completions"]
    assert after["derivation_fts_projections"] == before["derivation_fts_projections"]
    assert after["facts"] == before["facts"]
    assert {str(f.id) for f in (await db_session.scalars(select(Fact))).all()} == visible_ids
    if on:
        candidate = await db_session.get(Gen, final.generation_id)
        head = await db_session.get(Head, aid)
        assert candidate.status == "building" and candidate.published_at is None
        assert head.candidate_generation_id == candidate.id
        assert head.active_generation_id == old_gid
        if old_gid is None:
            assert head.legacy_retired_at is None
        else:
            assert (await db_session.get(Gen, old_gid)).status == "active"
            assert after["archive_derivation_heads"] == before["archive_derivation_heads"]
            old_before = next(row for row in before["derivation_generations"] if row["id"] == old_gid)
            assert next(row for row in after["derivation_generations"] if row["id"] == old_gid) == old_before
    else:
        assert after["derivation_generations"] == before["derivation_generations"] == []
        assert after["archive_derivation_heads"] == before["archive_derivation_heads"] == []


async def test_b2_unrelated_gate_exception_keeps_existing_fallback(monkeypatch):
    from app.domain.policy.gate import SensitivityGate
    gate = SensitivityGate()
    monkeypatch.setattr(gate, "_embedding_classify", Mock(side_effect=ValueError("synthetic classifier error")))
    result = await gate.classify_async("alpha technical source")
    assert result.category == "unclassified" and result.blocked is True


async def test_n1_atomic_prototypes_failure_static_then_clean_retry(assets, monkeypatch, caplog):
    from app.domain.policy import gate
    assert lm.require_local_model_ready() == assets.resolve()
    first, second = Mock(), Mock()
    first.encode.side_effect = RuntimeError("PRIVATE_PROTOTYPE /secret/path https://private/token")
    def ready_prototypes(*args, **kwargs):
        assert gate._gate_model is None and gate._prototype_embeddings is None
        return object()
    second.encode.side_effect = ready_prototypes
    constructor = Mock(side_effect=[first, second])
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    monkeypatch.setattr(gate, "_gate_model", None)
    monkeypatch.setattr(gate, "_prototype_embeddings", None)
    with pytest.raises(lm.LocalModelNotReady) as exc:
        gate._get_gate_model()
    assert str(exc.value) == "local_model_not_ready"
    assert exc.value.__cause__ is None and exc.value.__suppress_context__
    assert gate._gate_model is None and gate._prototype_embeddings is None
    first.encode.assert_called_once()
    assert "PRIVATE_PROTOTYPE" not in caplog.text and "/secret/path" not in caplog.text
    assert gate._get_gate_model() is second
    assert constructor.call_count == 2
    assert second.encode.call_count == len(gate._CATEGORY_PROTOTYPES)
    assert set(gate._prototype_embeddings) == set(gate._CATEGORY_PROTOTYPES)
    assert all(value is not None for value in gate._prototype_embeddings.values())
    assert gate._get_gate_model() is second
    assert constructor.call_count == 2  # no poisoned first model, no third load


@pytest.mark.parametrize("path", ["managed-legacy", "managed-successor", "legacy-OFF"])
async def test_n1_prototype_failure_uses_b2_no_publication_then_init_recovers(db_session, managed, assets, monkeypatch, path):
    from app.domain.policy import gate
    from app.domain.jobs.service import reprocess_job, claim_next_job
    from app.worker import dispatcher as d
    from app.worker import managed_dispatcher as worker
    on = path != "legacy-OFF"
    aid, jid, old_gid, visible_ids = await seed_visible(db_session, path == "managed-successor")
    if path == "managed-successor":
        await reprocess_job(db_session, jid, expected_generation=old_gid)
        await db_session.rollback()
        await claim_next_job(db_session)
    else:
        from datetime import datetime, timezone
        job = await db_session.get(Job, jid)
        job.status, job.attempts = "claimed", 1
        job.claimed_at = datetime.now(timezone.utc)
        await db_session.commit()
    monkeypatch.setattr(get_settings(), "generation_idempotency_enabled", on)
    before = await state(db_session)
    await db_session.rollback()
    assert lm.require_local_model_ready() == assets.resolve()
    monkeypatch.setattr(g, "require_local_model_ready_async", lm.require_local_model_ready_async)
    first, second = Mock(), Mock()
    first.encode.side_effect = RuntimeError("PRIVATE_PROTOTYPE /secret/path https://private/token")
    second.encode.side_effect = lambda *a, **kw: object()
    constructor = Mock(side_effect=[first, second])
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    monkeypatch.setattr(gate, "_gate_model", None)
    monkeypatch.setattr(gate, "_prototype_embeddings", None)
    monkeypatch.setattr(d, "_gate", gate.SensitivityGate())
    summarize = AsyncMock(side_effect=AssertionError("provider must not run"))
    extract = AsyncMock(side_effect=AssertionError("provider must not run"))
    monkeypatch.setattr(d, "_run_summarize_job", summarize)
    monkeypatch.setattr(d, "_run_extract_job", extract)
    async def finish_without_optional_enrichment(session, claim, **kwargs):
        await g.finish_claim(session, claim)
    monkeypatch.setattr(worker, "enrich_and_finish", finish_without_optional_enrichment)
    job = await db_session.get(Job, jid, populate_existing=True)
    await d.dispatch_job(db_session, job)
    final = await db_session.get(Job, jid, populate_existing=True)
    assert final.status == "retryable_failed" and final.attempts == 1
    if on:
        assert "LocalModelNotReady" in final.error_message
    else:
        assert final.error_message == "Sensitivity gate error: local_model_not_ready"
    assert all(secret not in final.error_message for secret in ["PRIVATE_PROTOTYPE", "/secret", "https://"])
    constructor.assert_called_once()  # construction succeeded; first encode failed
    first.encode.assert_called_once()
    assert gate._gate_model is None and gate._prototype_embeddings is None
    summarize.assert_not_awaited()
    extract.assert_not_awaited()
    after = await state(db_session)
    assert after["derivation_stage_completions"] == before["derivation_stage_completions"]
    assert after["derivation_fts_projections"] == before["derivation_fts_projections"]
    assert after["facts"] == before["facts"]
    assert {str(f.id) for f in (await db_session.scalars(select(Fact))).all()} == visible_ids
    if on:
        candidate = await db_session.get(Gen, final.generation_id)
        head = await db_session.get(Head, aid)
        assert candidate.status == "building" and candidate.published_at is None
        assert head.active_generation_id == old_gid and head.candidate_generation_id == candidate.id
        if old_gid is None:
            assert head.legacy_retired_at is None
        else:
            assert after["archive_derivation_heads"] == before["archive_derivation_heads"]
            assert next(r for r in after["derivation_generations"] if r["id"] == old_gid) == next(
                r for r in before["derivation_generations"] if r["id"] == old_gid)
    else:
        assert after["derivation_generations"] == before["derivation_generations"] == []
        assert after["archive_derivation_heads"] == before["archive_derivation_heads"] == []
    # Retry initialization only, not another dispatch/provider/generation action.
    assert gate._get_gate_model() is second
    assert constructor.call_count == 2
    assert second.encode.call_count == len(gate._CATEGORY_PROTOTYPES)
    assert set(gate._prototype_embeddings) == set(gate._CATEGORY_PROTOTYPES)
    assert await state(db_session) == after


async def test_n1_post_init_classification_error_keeps_existing_fallback(assets, monkeypatch):
    from app.domain.policy import gate
    model = Mock()
    prototypes = [object() for _ in gate._CATEGORY_PROTOTYPES]
    model.encode.side_effect = prototypes + [ValueError("ordinary document classification failure")]
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    monkeypatch.setattr(gate, "_gate_model", None)
    monkeypatch.setattr(gate, "_prototype_embeddings", None)
    result = await gate.SensitivityGate().classify_async("alpha technical source")
    assert result.category == "unclassified" and result.blocked is True
    assert gate._gate_model is model
    assert list(gate._prototype_embeddings.values()) == prototypes
    assert model.encode.call_count == len(prototypes) + 1
    constructor.assert_called_once()
