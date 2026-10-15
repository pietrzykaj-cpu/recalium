"""R2-ON dispatcher; the default-OFF legacy dispatcher remains authoritative."""
import logging
from dataclasses import replace
from app.domain.derivations import service as g
from app.infrastructure.settings import get_settings

logger = logging.getLogger(__name__)

async def dispatch_managed_job(session, job):
    from app.worker import dispatcher as legacy
    from app.domain.policy.resolver import resolve_effective_policy
    jid, aid, attempt, at = job.id, job.raw_archive_id, job.attempts, job.claimed_at
    generation = await g.admit_generation(session, jid)
    claim = g.Claim(jid, aid, generation.id, attempt, at)
    inputs = await g.snapshot(session, claim)
    claim = replace(claim, policy_intent=(
        inputs["metadata"].get("processing_mode"), inputs["metadata"].get("sensitivity_hint")))
    # Includes the local gate model: no session transaction or lock may span it.
    try:
        decision = await legacy._gate.classify_async(inputs["raw"])
    except Exception as exc:
        await g.fail_current_claim(session, jid, attempt, at,
            generation_id=claim.generation_id, error=legacy._stage_failure("gate","failure",exc))
        return
    policy = resolve_effective_policy(gate_allows=decision.is_allowed,
        data_class=decision.category, processing_mode=inputs["metadata"].get("processing_mode"),
        sensitivity_hint=inputs["metadata"].get("sensitivity_hint"))
    settings = get_settings()
    providers = {"summarize":legacy._resolve_provider(settings.summarize_provider),
                 "extract":legacy._resolve_provider(settings.extract_provider)}
    permissions = {op:{"provider":p,
        "local":p == "ollama" and legacy._is_local_ollama_url(settings.ollama_base_url),
        "allowed":legacy._provider_allowed(p, allow_external=policy.allow_external)}
        for op,p in providers.items()}
    await g.write_policy(session, claim, decision, policy, permissions)
    await g.raw_baseline(session, claim)
    inputs = await g.snapshot(session, claim)
    if inputs["status"] != "active" and (policy.allow_external or any(v["allowed"] for v in permissions.values())) and not legacy._has_llm_provider():
        await g.finish_claim(session, claim, pending_provider=True)
        return

    if inputs["summary_stage"] is None:
        if permissions["summarize"]["allowed"]:
            try:
                text = await legacy._run_summarize_job(inputs["raw"], allow_external=policy.allow_external)
            except Exception as exc:
                category = legacy._summary_failure_category(exc)
                failure = legacy._stage_failure("summary",category or "unclassified",exc)
                if category is None or attempt < inputs["max_attempts"]:
                    await g.fail_current_claim(session, jid, attempt, at, generation_id=claim.generation_id, error=failure)
                    return
                # Persistence/audit errors below are outside the provider catch;
                # they can never authorize final-attempt fallthrough.
                await g.persist_summary(session, claim, None, outcome="partial_failure",
                    partial=(category, legacy._bounded_exception_class(exc), providers["summarize"]))
            else:
                bounded = providers["summarize"] == "ollama" and legacy._bound_summary_input(inputs["raw"])[1]
                await g.persist_summary(session, claim, text, outcome="succeeded" if text else "empty",
                    model_used=legacy._resolve_model(providers["summarize"],settings.summarize_model),
                    method="llm_summarization_bounded" if bounded else "llm_summarization")
        else:
            await g.persist_summary(session, claim, None, outcome="policy_skipped")

    inputs = await g.snapshot(session, claim)
    partial = inputs["summary_stage"]
    marker = (f"summary_failed(category={partial[1]}, exception_class={partial[2]})"
              if partial and partial[0] == "partial_failure" else None)
    if not inputs["extraction_complete"]:
        try:
            batch = (await legacy._run_extract_job(inputs["raw"], allow_external=policy.allow_external)
                     if permissions["extract"]["allowed"] else [])
            batch = [{**f,"derivation_method":"llm_extraction","derivation_model":legacy._provider_name()} for f in batch]
            await g.persist_extraction(session, claim, batch,
                outcome="succeeded" if permissions["extract"]["allowed"] else "policy_skipped")
        except Exception as exc:
            error = legacy._stage_failure("extraction","failure",exc)
            await g.fail_current_claim(session, jid, attempt, at, generation_id=claim.generation_id,
                error=(marker+"; "+error) if marker else error, retryable=marker is None)
            return
    inputs = await g.snapshot(session, claim)
    if inputs["status"] != "active":
        await g.prepare_fts(session, claim, inputs["summary"] if inputs["summary"] is not None else inputs["raw"][:10000],
                            "summary" if inputs["summary"] is not None else "raw_fallback")
        await g.publish_generation(session, claim)

    # Enrichment is independent and non-fatal. Each model call follows an ended
    # prepare transaction; each write revalidates its current generation.
    try:
        inputs = await g.snapshot(session, claim)
        if inputs["embedding"] is None:
            from app.domain.derived_memory.service import embed_text
            vector = await embed_text(inputs["summary"] if inputs["summary"] is not None else inputs["raw"][:10000])
            await g.persist_embedding(session, claim, vector)
    except Exception as exc:
        await session.rollback()
        logger.warning("Managed embedding failed (non-fatal): %s", type(exc).__name__)
    try:
        inputs = await g.snapshot(session, claim)
        if inputs["embedding"] is not None:
            from app.domain.conflict_detection import detect_and_group_duplicates
            from app.domain.derived_memory.models import Embedding
            from sqlalchemy import select
            row = await session.scalar(select(Embedding).where(Embedding.generation_id == claim.generation_id))
            # No model call in duplicate detection. Fence all involved sources.
            await detect_and_group_duplicates(session, raw_archive_id=aid,
                embedding_id=row.id, embedding=inputs["embedding"])
    except Exception as exc:
        await session.rollback()
        logger.warning("Managed conflict enrichment failed (non-fatal): %s",type(exc).__name__)
    try:
        from app.worker.managed_links import run_managed_links
        await run_managed_links(session, claim, allow_external=policy.allow_external)
    except Exception as exc:
        await session.rollback()
        logger.warning("Managed link enrichment failed (non-fatal): %s",type(exc).__name__)
    await g.finish_claim(session, claim, partial_marker=marker)
