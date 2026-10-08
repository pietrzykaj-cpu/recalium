"""Dispatcher tests — BYOK-07, BYOK-08.

Tests will FAIL (RED) until app.worker.dispatcher is created.
"""
from __future__ import annotations

import uuid
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("app.worker.dispatcher", reason="worker.dispatcher not yet implemented")

from app.worker.dispatcher import dispatch_job, _run_summarize_job as _summary_provider_call  # noqa: E402


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


@pytest.fixture
def bounded_summary_pipeline(monkeypatch):
    """Exercise dispatcher writes without DB, embedding, or provider runtime."""
    from app.worker import dispatcher as d
    from app.domain.policy.gate import SensitivityDecision

    archive = SimpleNamespace(raw_content="small input", metadata_json={})
    session = MagicMock()
    session.execute = AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: archive))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    job = SimpleNamespace(id=uuid.uuid4(), raw_archive_id=uuid.uuid4(),
                          attempts=1, max_attempts=3, status="claimed", error_message=None)
    settings = SimpleNamespace(
        summarize_provider="ollama", extract_provider="ollama",
        summarize_model="auto", extract_model="auto", ollama_model="qwen3:4b",
        ollama_base_url="http://localhost:11434", ollama_api_key="",
        openai_api_key="test-only", anthropic_api_key="test-only",
    )
    monkeypatch.setattr(d, "get_settings", lambda: settings)
    monkeypatch.setattr(d._gate, "classify_async", AsyncMock(return_value=SensitivityDecision(
        category="general", confidence=0.9, blocked=False, method="test",
    )))
    services = "app.domain.derived_memory.service."
    for name in ("get_existing_summary", "get_existing_embedding", "write_summary",
                 "write_facts", "write_fts_entry", "write_embedding", "write_tags"):
        monkeypatch.setattr(services + name, AsyncMock(return_value=None))
    monkeypatch.setattr(services + "embed_text", AsyncMock(return_value=[0.1] * 384))
    monkeypatch.setattr("app.domain.archive.service.suppress_new_derivations_if_deleted", AsyncMock())
    monkeypatch.setattr(d, "_run_link_detection_job", AsyncMock())
    monkeypatch.setattr(d, "complete_job", AsyncMock())
    monkeypatch.setattr(d, "fail_job", AsyncMock())
    monkeypatch.setattr(d, "_run_summarize_job", AsyncMock(return_value="A summary"))
    monkeypatch.setattr(d, "_run_extract_job", AsyncMock(return_value=[]))
    from app.domain.derived_memory import service
    return session, job, archive, settings, service


@pytest.mark.parametrize("provider,size,expected", [
    ("ollama", 2047, "llm_summarization"),
    ("ollama", 2048, "llm_summarization"),
    ("ollama", 2049, "llm_summarization_bounded"),
    ("openai", 6000, "llm_summarization"),
    ("anthropic", 6000, "llm_summarization"),
])
async def test_bounded_summary_provenance(monkeypatch, bounded_summary_pipeline, provider, size, expected):
    from app.worker import dispatcher as d
    session, job, archive, settings, service = bounded_summary_pipeline
    archive.raw_content = "x" * size
    settings.summarize_provider = provider
    await d.dispatch_job(session, job)
    service.write_summary.assert_awaited_once()
    assert service.write_summary.call_args.kwargs["derivation_method"] == expected
    assert service.write_summary.call_args.kwargs["summary_text"] == "A summary"
    d._run_extract_job.assert_awaited_once_with(archive.raw_content, allow_external=True)
    d.complete_job.assert_awaited_once()
    d.fail_job.assert_not_awaited()


async def test_bounded_summary_provenance_uses_utf8_bytes(bounded_summary_pipeline):
    from app.worker import dispatcher as d
    session, job, archive, settings, service = bounded_summary_pipeline
    archive.raw_content = "😀" * 600
    assert len(archive.raw_content) <= d.SUMMARY_INPUT_MAX_BYTES
    assert len(archive.raw_content.encode("utf-8")) > d.SUMMARY_INPUT_MAX_BYTES
    await d.dispatch_job(session, job)
    service.write_summary.assert_awaited_once()
    assert service.write_summary.call_args.kwargs["derivation_method"] == "llm_summarization_bounded"


async def test_bounded_summary_failure_still_returns_before_extraction(bounded_summary_pipeline):
    from app.worker import dispatcher as d
    session, job, archive, settings, service = bounded_summary_pipeline
    archive.raw_content = "x" * 6000
    d._run_summarize_job.side_effect = ValueError("synthetic summary failure")
    await d.dispatch_job(session, job)
    d._run_summarize_job.assert_awaited_once()
    d._run_extract_job.assert_not_awaited()
    service.write_summary.assert_not_awaited()
    d.fail_job.assert_awaited_once()
    assert d.fail_job.call_args.kwargs["retryable"] is True
    session.rollback.assert_awaited_once()
    session.refresh.assert_awaited_once_with(job)
    d.complete_job.assert_not_awaited()


async def test_bounded_summary_existing_summary_is_not_reprocessed(bounded_summary_pipeline):
    from app.worker import dispatcher as d
    session, job, archive, settings, service = bounded_summary_pipeline
    archive.raw_content = "x" * 6000
    service.get_existing_summary.return_value = SimpleNamespace(summary_text="Existing summary")
    await d.dispatch_job(session, job)
    d._run_summarize_job.assert_not_awaited()
    service.write_summary.assert_not_awaited()
    d.complete_job.assert_awaited_once()


@pytest.fixture
def partial_summary_pipeline(monkeypatch, bounded_summary_pipeline):
    """Use real job transitions and the real summary request; no sockets or DB."""
    import httpx
    from app.worker import dispatcher as d
    from app.domain.jobs import service as jobs

    session, job, archive, settings, service = bounded_summary_pipeline
    job.attempts = job.max_attempts
    archive.raw_content = "RAW_PRIVATE_CANARY"
    client = AsyncMock()
    client.post.return_value = httpx.Response(
        500, json={"error": "RESPONSE_PRIVATE_CANARY"},
        request=httpx.Request("POST", "http://localhost/api/chat"),
    )
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=client)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    control = SimpleNamespace(before=None, after=None, entered=0)

    @asynccontextmanager
    async def acquire(**kwargs):
        control.entered += 1
        if control.before is not None:
            raise control.before
        try:
            yield
        finally:
            if control.after is not None:
                raise control.after

    monkeypatch.setattr(d, "get_local_inference_coordinator", lambda: SimpleNamespace(acquire=acquire))
    monkeypatch.setattr(d, "_run_summarize_job", AsyncMock(side_effect=_summary_provider_call))
    monkeypatch.setattr(d, "complete_job", AsyncMock(side_effect=jobs.complete_job))
    monkeypatch.setattr(d, "fail_job", AsyncMock(side_effect=jobs.fail_job))
    d._run_extract_job.return_value = [{
        "fact_text": "Invented fact", "source_span": "Invented",
        "confidence_tier": "high", "tags": [], "entities": [],
    }]
    service.write_facts.return_value = [SimpleNamespace(id=uuid.uuid4())]
    return (*bounded_summary_pipeline, client, control)


def _summary_audits(session):
    return [call.args[0] for call in session.add.call_args_list
            if getattr(call.args[0], "event_type", None) == "summary_failed"]


async def test_partial_summary_nonfinal_retries_without_extraction(partial_summary_pipeline):
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    job.attempts = 1
    await d.dispatch_job(session, job)
    assert job.status == "retryable_failed"
    assert job.error_message == "summary_failed(category=provider_response, exception_class=HTTPStatusError)"
    session.rollback.assert_awaited_once()
    session.refresh.assert_awaited_once_with(job)
    d.fail_job.assert_awaited_once()
    assert d.fail_job.call_args.kwargs["retryable"] is True
    d._run_extract_job.assert_not_awaited()
    service.write_facts.assert_not_awaited()
    service.write_summary.assert_not_awaited()
    d.complete_job.assert_not_awaited()
    assert _summary_audits(session) == []
    client.post.assert_awaited_once()


async def test_partial_summary_final_audits_then_completes(partial_summary_pipeline, caplog):
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    async def extraction(*args, **kwargs):
        # The fallthrough event must have been committed BEFORE extraction.
        assert len(_summary_audits(session)) == 1
        assert session.commit.await_count == 3  # gate, policy, summary_failed
        return [{"fact_text": "Invented", "tags": [], "entities": []}]
    d._run_extract_job.side_effect = extraction
    await d.dispatch_job(session, job)
    d._run_extract_job.assert_awaited_once()
    service.write_facts.assert_awaited_once()
    service.write_summary.assert_not_awaited()
    service.write_fts_entry.assert_awaited_once()
    service.write_embedding.assert_awaited_once()
    d.complete_job.assert_awaited_once()
    d.fail_job.assert_not_awaited()
    assert job.status == "completed"
    assert job.error_message == "partial: summary_failed(category=provider_response, exception_class=HTTPStatusError)"
    event, = _summary_audits(session)
    assert event.actor == "pipeline_worker"
    assert event.raw_archive_id == job.raw_archive_id
    assert event.operation_metadata == {
        "job_id": str(job.id), "stage": "summary", "attempt": 3,
        "max_attempts": 3, "final_attempt": True, "failure_category": "provider_response",
        "exception_class": "HTTPStatusError", "provider": "ollama",
        "continuing_to_extraction": True,
    }
    evidence = json.dumps(event.operation_metadata) + job.error_message + caplog.text
    for private in (archive.raw_content, "RESPONSE_PRIVATE_CANARY", d.SUMMARIZATION_SYSTEM_PROMPT,
                    "500 Internal Server Error", "http://localhost/api/chat"):
        assert private not in evidence
    client.post.assert_awaited_once()
    assert control.entered == 1


async def test_partial_summary_dual_failure_is_terminal(partial_summary_pipeline, caplog):
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    d._run_extract_job.side_effect = ValueError("EXTRACTION_PRIVATE_CANARY")
    await d.dispatch_job(session, job)
    assert job.status == "failed"
    assert job.error_message == (
        "summary_failed(category=provider_response, exception_class=HTTPStatusError); "
        "extraction_failed(category=failure, exception_class=ValueError)"
    )
    assert "EXTRACTION_PRIVATE_CANARY" not in job.error_message + caplog.text
    assert d.fail_job.call_args.kwargs["retryable"] is False
    d.complete_job.assert_not_awaited()
    service.write_facts.assert_not_awaited()


async def test_partial_summary_successful_retry_clears_old_error(partial_summary_pipeline):
    import httpx
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    job.error_message = "OLD_SUMMARY_ERROR"
    client.post.return_value = httpx.Response(200, json={"message": {"content": "Invented summary"}},
        request=httpx.Request("POST", "http://localhost/api/chat"))
    await d.dispatch_job(session, job)
    assert job.status == "completed"
    assert job.error_message is None
    service.write_summary.assert_awaited_once()
    assert _summary_audits(session) == []


@pytest.mark.parametrize("attempt,expected", [(1, "retryable_failed"), (3, "failed")])
async def test_partial_summary_success_extraction_failure_unchanged(partial_summary_pipeline, attempt, expected):
    import httpx
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    job.attempts = attempt
    client.post.return_value = httpx.Response(200, json={"message": {"content": "Invented summary"}},
        request=httpx.Request("POST", "http://localhost/api/chat"))
    d._run_extract_job.side_effect = ValueError("existing extraction semantics")
    await d.dispatch_job(session, job)
    assert job.status == expected
    assert job.error_message == "ValueError: existing extraction semantics"
    assert d.fail_job.call_args.kwargs["retryable"] is True
    assert _summary_audits(session) == []
    d.complete_job.assert_not_awaited()


async def test_partial_summary_existing_summary_skips_provider(partial_summary_pipeline):
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    service.get_existing_summary.return_value = SimpleNamespace(summary_text="Existing summary")
    job.error_message = "OLD_ERROR"
    await d.dispatch_job(session, job)
    client.post.assert_not_awaited()
    d._run_summarize_job.assert_not_awaited()
    assert job.status == "completed"
    assert job.error_message is None


@pytest.mark.parametrize("postcommit", [False, True])
async def test_partial_summary_persistence_never_falls_through(partial_summary_pipeline, postcommit):
    import httpx
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    client.post.return_value = httpx.Response(200, json={"message": {"content": "Invented summary"}},
        request=httpx.Request("POST", "http://localhost/api/chat"))
    async def write(*args, **kwargs):
        if postcommit:
            await session.commit()
            await session.refresh(SimpleNamespace(id=uuid.uuid4()))
        raise RuntimeError("PRIVATE_PERSISTENCE_FAILURE")
    service.write_summary.side_effect = write
    if postcommit:
        session.refresh.side_effect = [RuntimeError("PRIVATE_REFRESH_FAILURE"), None]
    await d.dispatch_job(session, job)
    assert job.status == "failed"
    assert job.error_message == "summary_failed(category=unclassified, exception_class=RuntimeError)"
    assert _summary_audits(session) == []
    d._run_extract_job.assert_not_awaited()
    d.complete_job.assert_not_awaited()


@pytest.mark.parametrize("kind", ["identity", "residency", "cleanup", "coordinator_http",
                                  "coordinator_json", "authorization", "unknown"])
async def test_partial_summary_control_plane_never_falls_through(partial_summary_pipeline, kind):
    import httpx
    from app.worker import dispatcher as d
    from app.infrastructure.local_inference import (
        LocalInferenceIdentityError, LocalInferenceResidencyError, LocalInferenceCleanupError,
    )
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    errors = {
        "identity": LocalInferenceIdentityError("PRIVATE_IDENTITY"),
        "residency": LocalInferenceResidencyError("PRIVATE_RESIDENCY"),
        "cleanup": LocalInferenceCleanupError("PRIVATE_CLEANUP"),
        "coordinator_http": httpx.HTTPStatusError("PRIVATE_PROBE",
            request=httpx.Request("GET", "http://localhost/api/tags"), response=httpx.Response(500)),
        "coordinator_json": json.JSONDecodeError("PRIVATE_PROBE", "PRIVATE_DOCUMENT", 0),
        "authorization": PermissionError("PRIVATE_AUTH"),
        "unknown": ValueError("PRIVATE_UNKNOWN"),
    }
    if kind == "cleanup":
        control.after = errors[kind]  # overrides the eligible HTTP body failure
    else:
        control.before = errors[kind]
    await d.dispatch_job(session, job)
    assert job.status == "failed"
    assert "category=unclassified" in job.error_message
    assert "PRIVATE_" not in job.error_message
    assert _summary_audits(session) == []
    d._run_extract_job.assert_not_awaited()
    d.complete_job.assert_not_awaited()
    assert client.post.await_count == (1 if kind == "cleanup" else 0)


async def test_partial_summary_audit_failure_is_terminal(partial_summary_pipeline, caplog):
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    session.commit.side_effect = [None, None, RuntimeError("PRIVATE_AUDIT_FAILURE"), None]
    await d.dispatch_job(session, job)
    assert job.status == "failed"
    assert job.error_message.endswith("audit_failed(category=persistence, exception_class=RuntimeError)")
    assert "summary_failed" in job.error_message
    assert "PRIVATE_AUDIT_FAILURE" not in job.error_message + caplog.text
    assert d.fail_job.call_args.kwargs["retryable"] is False
    d._run_extract_job.assert_not_awaited()
    d.complete_job.assert_not_awaited()
    assert session.rollback.await_count == 2


async def test_partial_summary_marker_survives_downstream_refresh(partial_summary_pipeline):
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    service.write_fts_entry.side_effect = RuntimeError("nonfatal FTS failure")
    async def refresh(value):
        value.error_message = "OLD_DATABASE_ERROR"
    session.refresh.side_effect = refresh
    await d.dispatch_job(session, job)
    assert session.rollback.await_count == 2
    assert session.refresh.await_count == 2
    assert job.status == "completed"
    assert job.error_message == "partial: summary_failed(category=provider_response, exception_class=HTTPStatusError)"


@pytest.mark.parametrize("kind,category,eligible", [
    ("timeout", "provider_transport", True), ("invalid_json", "response_parsing", True),
    ("thinking_only", "response_parsing", True), ("missing_message", "response_parsing", True),
    ("unauthorized", "unclassified", False), ("forbidden", "unclassified", False),
    ("local_protocol", "unclassified", False), ("diagnostic_internal", "unclassified", False),
])
async def test_partial_summary_request_boundary(partial_summary_pipeline, monkeypatch, kind, category, eligible):
    import httpx
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    request = httpx.Request("POST", "http://localhost/api/chat")
    if kind == "timeout":
        client.post.side_effect = httpx.ReadTimeout("PRIVATE_TIMEOUT", request=request)
    elif kind == "local_protocol":
        client.post.side_effect = httpx.LocalProtocolError("PRIVATE_PROTOCOL", request=request)
    elif kind == "invalid_json":
        client.post.return_value = httpx.Response(200, content=b"PRIVATE_BAD_JSON", request=request)
    elif kind == "thinking_only":
        client.post.return_value = httpx.Response(200, json={"message": {"content": "<think>PRIVATE_REASONING"}}, request=request)
    elif kind == "missing_message":
        client.post.return_value = httpx.Response(200, json={}, request=request)
    elif kind in ("unauthorized", "forbidden"):
        client.post.return_value = httpx.Response(401 if kind == "unauthorized" else 403, request=request)
    else:
        client.post.return_value = httpx.Response(200, json={"message": {"content": "Summary"}}, request=request)
        monkeypatch.setattr(d, "parse_ollama_completion_diagnostics", MagicMock(side_effect=ValueError("PRIVATE_INTERNAL")))
    await d.dispatch_job(session, job)
    assert job.status == ("completed" if eligible else "failed")
    assert f"category={category}" in job.error_message
    assert "PRIVATE_" not in job.error_message
    assert len(_summary_audits(session)) == int(eligible)
    assert d._run_extract_job.await_count == int(eligible)
    client.post.assert_awaited_once()


def test_partial_summary_exception_class_is_bounded():
    from app.worker import dispatcher as d
    for name in ("X" * 65, "PRIVATE-response-text", "非ASCII"):
        exc = type(name, (Exception,), {})("PRIVATE_MESSAGE")
        assert d._bounded_exception_class(exc) == "Exception"
    assert d._bounded_exception_class(ValueError("PRIVATE_MESSAGE")) == "ValueError"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("kind,eligible", [
    ("transport", True), ("response", True), ("parse", True), ("malformed", True),
    ("authentication", False), ("permission", False), ("unknown", False),
    ("status_408", True), ("status_429", True), ("status_500", True), ("status_599", True),
    ("status_400", False), ("status_404", False), ("status_409", False), ("status_422", False),
    ("ambiguous_status", False),
])
async def test_partial_summary_sdk_request_boundary(partial_summary_pipeline, monkeypatch, provider, kind, eligible):
    """SDK-shaped doubles exercise the real caller, without installing/calling SDKs."""
    import sys
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    settings.summarize_provider = provider

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, status):
            super().__init__("PRIVATE_SDK_STATUS")
            self.status_code = status

    class APIResponseValidationError(Exception):
        pass

    errors = {
        "transport": APIConnectionError("PRIVATE_TRANSPORT"),
        "response": APIStatusError(503), "parse": APIResponseValidationError("PRIVATE_PARSE"),
        "authentication": APIStatusError(401), "permission": APIStatusError(403),
        "unknown": RuntimeError("PRIVATE_UNKNOWN"),
        "ambiguous_status": APIStatusError("500"),
    }
    if kind.startswith("status_"):
        errors[kind] = APIStatusError(int(kind.removeprefix("status_")))
    create = AsyncMock(return_value=SimpleNamespace(choices=[], content=[]))
    if kind != "malformed":
        create.side_effect = errors[kind]
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                                 messages=SimpleNamespace(create=create))
    sdk = SimpleNamespace(APIConnectionError=APIConnectionError, APIStatusError=APIStatusError,
                          APIResponseValidationError=APIResponseValidationError)
    sdk_name = "AsyncOpenAI" if provider == "openai" else "AsyncAnthropic"
    setattr(sdk, sdk_name, lambda **kwargs: sdk_client)
    monkeypatch.setitem(sys.modules, provider, sdk)
    await d.dispatch_job(session, job)
    assert job.status == ("completed" if eligible else "failed")
    assert "PRIVATE_" not in job.error_message
    assert d._run_extract_job.await_count == int(eligible)
    events = _summary_audits(session)
    assert len(events) == int(eligible)
    if eligible:
        assert events[0].operation_metadata["provider"] == provider
    create.assert_awaited_once()
    client.post.assert_not_awaited()  # no Ollama fallback


@pytest.mark.parametrize("status,eligible", [
    (408, True), (429, True), (500, True), (503, True), (599, True),
    (400, False), (401, False), (403, False), (404, False), (409, False), (422, False),
    (302, False), (600, False),
])
async def test_partial_summary_http_status_allowlist(partial_summary_pipeline, status, eligible):
    import httpx
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    client.post.return_value = httpx.Response(status, request=httpx.Request("POST", "http://localhost/api/chat"))
    await d.dispatch_job(session, job)
    assert job.status == ("completed" if eligible else "failed")
    category = "provider_response" if eligible else "unclassified"
    assert f"category={category}" in job.error_message
    assert d._run_extract_job.await_count == int(eligible)
    assert len(_summary_audits(session)) == int(eligible)
    client.post.assert_awaited_once()


def test_partial_summary_http_status_unknown_values_fail_closed():
    from app.worker import dispatcher as d
    for status in (None, True, False, "408", "429", "500", 500.0, object()):
        assert d._eligible_summary_http_status(status) is False
    for status in range(100, 700):
        assert d._eligible_summary_http_status(status) is (status in (408, 429) or 500 <= status <= 599)


@pytest.mark.parametrize("status", [408, 429, 503])
async def test_partial_summary_coordinator_http_status_never_qualifies(partial_summary_pipeline, status):
    import httpx
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    control.before = httpx.HTTPStatusError("PRIVATE_COORDINATOR_FAILURE",
        request=httpx.Request("GET", "http://localhost/api/tags"), response=httpx.Response(status))
    await d.dispatch_job(session, job)
    assert job.status == "failed"
    assert "category=unclassified" in job.error_message
    d._run_extract_job.assert_not_awaited()
    client.post.assert_not_awaited()
    assert _summary_audits(session) == []


async def test_partial_summary_marked_persistence_error_still_fails_closed(partial_summary_pipeline):
    import httpx
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    request = httpx.Request("POST", "http://localhost/api/chat")
    client.post.return_value = httpx.Response(200, json={"message": {"content": "Summary"}}, request=request)
    exc = httpx.ReadTimeout("PRIVATE_WRITE_ERROR", request=request)
    d._mark_summary_output_failure(exc, "provider_transport")
    service.write_summary.side_effect = exc
    await d.dispatch_job(session, job)
    assert job.status == "failed"
    assert "category=unclassified" in job.error_message
    assert _summary_audits(session) == []
    d._run_extract_job.assert_not_awaited()


async def test_partial_summary_remote_policy_denial_fails_closed(partial_summary_pipeline):
    from app.worker import dispatcher as d
    session, job, archive, settings, service, client, control = partial_summary_pipeline
    settings.ollama_base_url = "https://remote.invalid"
    # Exercise the provider callable's policy check, not a fake transport error.
    settings.openai_api_key = settings.anthropic_api_key = ""
    archive.metadata_json = {"processing_mode": "local_only"}
    await d.dispatch_job(session, job)
    client.post.assert_not_awaited()
    d._run_summarize_job.assert_not_awaited()
    assert _summary_audits(session) == []
