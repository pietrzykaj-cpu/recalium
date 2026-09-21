"""Dispatcher tests — BYOK-07, BYOK-08.

Tests will FAIL (RED) until app.worker.dispatcher is created.
"""
from __future__ import annotations

import uuid

import pytest

pytest.importorskip("app.worker.dispatcher", reason="worker.dispatcher not yet implemented")

from app.worker.dispatcher import dispatch_job  # noqa: E402


async def _make_archive_item(session):
    """Helper: insert a minimal RawArchiveItem to satisfy FK constraint on jobs."""
    import hashlib
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


async def test_invalid_key_causes_retryable_failed(db_session_phase2, monkeypatch):
    """BYOK-07: Invalid/rate-limited key → job enters retryable_failed with error_message."""
    from app.domain.jobs.models import Job
    from app.domain.policy.gate import SensitivityDecision

    archive_item = await _make_archive_item(db_session_phase2)
    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=archive_item.id,
        status="claimed",
        attempts=1,
    )
    db_session_phase2.add(job)
    await db_session_phase2.commit()

    # Mock gate to return "general" (not blocked) so the LLM path is exercised.
    # Without this, missing sentence_transformers causes gate to default to "unclassified"
    # (blocked), which would skip the LLM call and complete the job instead.
    async def mock_classify_async(text: str) -> SensitivityDecision:
        return SensitivityDecision(category="general", confidence=0.9, blocked=False, method="nli")

    monkeypatch.setattr(
        "app.worker.dispatcher._gate.classify_async",
        mock_classify_async,
    )

    # Simulate invalid key by monkeypatching OpenAI to raise AuthenticationError
    from openai import AuthenticationError
    import httpx
    async def fake_summarize(*args, **kwargs):
        fake_response = httpx.Response(401, text="Unauthorized")
        raise AuthenticationError("Invalid API key", response=fake_response, body=None)

    monkeypatch.setattr(
        "app.worker.dispatcher._run_summarize_job",
        fake_summarize,
    )

    # Select an available provider as well as reporting availability.
    monkeypatch.setattr("app.worker.dispatcher._resolve_provider", lambda configured: "openai")

    # Also mock _has_llm_provider to return True so the LLM path is entered
    monkeypatch.setattr("app.worker.dispatcher._has_llm_provider", lambda: True)

    await dispatch_job(db_session_phase2, job)
    await db_session_phase2.refresh(job)

    assert job.status == "retryable_failed"
    assert job.error_message is not None
    assert len(job.error_message) > 0


async def test_completed_subjob_not_rerun(db_session_phase2):
    """BYOK-08: Switching provider does not re-run already-completed sub-jobs."""
    # A job whose summary sub-job is already completed should not re-run summarize
    # even if provider is changed.
    from app.domain.jobs.models import Job
    from app.domain.derived_memory.models import Summary

    archive_item = await _make_archive_item(db_session_phase2)
    raw_id = archive_item.id

    # Create a completed summary for this archive item
    summary = Summary(
        raw_archive_id=raw_id,
        summary_text="Existing summary",
        model_used="gpt-4o-mini",
        derivation_method="llm_summarization",
    )
    db_session_phase2.add(summary)

    job = Job(
        id=uuid.uuid4(),
        job_type="process_archive_item",
        raw_archive_id=raw_id,
        status="claimed",
        attempts=1,
    )
    db_session_phase2.add(job)
    await db_session_phase2.commit()

    call_count = {"n": 0}

    async def mock_summarize(*args, **kwargs):
        call_count["n"] += 1
        return {"summary": "new"}

    # If implementation correctly skips completed sub-jobs, mock should not be called
    import app.worker.dispatcher as disp
    original = getattr(disp, "_run_summarize_job", None)
    disp._run_summarize_job = mock_summarize
    try:
        await dispatch_job(db_session_phase2, job)
    finally:
        if original is not None:
            disp._run_summarize_job = original

    assert call_count["n"] == 0, "Summarize ran again even though summary exists"


# ── Chunked extraction (F3) ───────────────────────────────────────────────────

from app.worker.dispatcher import _split_conversation, _dedupe_facts  # noqa: E402


def test_split_short_text_single_chunk():
    text = "User: Hi\n\nAssistant: Hello!"
    assert _split_conversation(text, max_chunk_chars=1500) == [text]


def test_split_on_turn_boundaries_chunks_are_verbatim_slices():
    """Chunks must be contiguous verbatim substrings of the source so
    extracted source_spans remain verbatim in the FULL document."""
    turns = []
    for i in range(6):
        turns.append(f"User: question number {i} " + "x" * 300)
        turns.append(f"Assistant: answer number {i} " + "y" * 300)
    text = "\n\n".join(turns)

    chunks = _split_conversation(text, max_chunk_chars=1500)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk in text  # verbatim slice
        assert len(chunk) <= 1500 or "\n" not in chunk.strip()
    # No content lost
    assert sum(len(c) for c in chunks) >= len(text) - 2 * len(chunks)


def test_split_oversized_single_turn_hard_splits():
    text = "Assistant: " + "z" * 5000
    chunks = _split_conversation(text, max_chunk_chars=1500)
    assert len(chunks) >= 3
    assert all(c in text for c in chunks)
    assert "".join(chunks) == text


def test_dedupe_facts_removes_near_identical_statements():
    facts = [
        {"fact_text": "The capital of France is Paris.", "source_span": "a"},
        {"fact_text": "the capital of france is paris", "source_span": "b"},
        {"fact_text": "Rust has one owner per value.", "source_span": "c"},
    ]
    result = _dedupe_facts(facts)
    assert len(result) == 2
    assert result[0]["fact_text"] == "The capital of France is Paris."
