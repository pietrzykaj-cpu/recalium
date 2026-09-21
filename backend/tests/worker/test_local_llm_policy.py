"""Local inference must never grant permission to a different remote operation."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from app.worker import dispatcher as d
from app.domain.policy.gate import SensitivityDecision


@pytest.fixture
def settings(monkeypatch):
    value = SimpleNamespace(
        summarize_provider="ollama", extract_provider="ollama",
        summarize_model="auto", extract_model="auto", ollama_model="qwen3:4b",
        ollama_base_url="http://host.docker.internal:11434",
        ollama_api_key=None, openai_api_key="test-only", anthropic_api_key="test-only",
    )
    monkeypatch.setattr(d, "get_settings", lambda: value)
    return value


@pytest.mark.parametrize("url,local", [
    ("http://host.docker.internal:11434", True),
    ("http://localhost:11434/", True), ("http://127.0.0.1:11434", True),
    ("http://[::1]:11434", True), ("https://ollama.example.com", False),
    ("http://192.168.1.20:11434", False), ("http://localhost.evil.test", False),
    ("http://localhost@evil.test", False), ("http://evil.test@localhost", False),
    ("http://localhost:99999", False), ("file://localhost", False),
    ("http://localhost/proxy", False), ("http://[", False), (None, False),
])
def test_local_endpoint_boundary(url, local):
    assert d._is_local_ollama_url(url) is local


@pytest.fixture
def pipeline(monkeypatch):
    archive = SimpleNamespace(raw_content="My favourite animal is the silver fox.", metadata_json={})
    session = MagicMock()
    session.execute = AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: archive))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    job = SimpleNamespace(id=uuid.uuid4(), raw_archive_id=uuid.uuid4())
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
    return session, job, archive


@pytest.mark.parametrize("category", ["general", "personal_profile", "relationship", "unclassified"])
@pytest.mark.parametrize("mode", ["deferred", "local_only"])
@pytest.mark.parametrize("summarize,extract", [
    ("ollama", "ollama"), ("ollama", "openai"), ("anthropic", "ollama"),
    ("openai", "anthropic"),
])
async def test_dispatch_authorizes_operations_independently(
    monkeypatch, settings, pipeline, category, mode, summarize, extract,
):
    session, job, archive = pipeline
    settings.summarize_provider, settings.extract_provider = summarize, extract
    archive.metadata_json = {"processing_mode": mode}
    external = category == "general" and mode != "local_only"
    monkeypatch.setattr(d._gate, "classify_async", AsyncMock(return_value=SensitivityDecision(
        category=category, confidence=0.9, blocked=category != "general", method="test",
    )))
    await d.dispatch_job(session, job)
    assert d._run_summarize_job.await_count == int(external or summarize == "ollama")
    assert d._run_extract_job.await_count == int(external or extract == "ollama")
    for operation, provider, call in (
        ("summarize", summarize, d._run_summarize_job),
        ("extract", extract, d._run_extract_job),
    ):
        if call.await_count:
            assert call.call_args.kwargs["allow_external"] is external
    audit = next(c.args[0] for c in session.add.call_args_list
                 if c.args[0].event_type == "policy_decision").operation_metadata
    assert audit["allow_external"] is external
    for operation, provider in (("summarize", summarize), ("extract", extract)):
        assert audit["operations"][operation] == {
            "provider": provider, "local": provider == "ollama",
            "allowed": external or provider == "ollama",
        }
    d.complete_job.assert_awaited_once()
    d.fail_job.assert_not_awaited()


async def test_audit_failure_stops_local_llm(monkeypatch, settings, pipeline):
    session, job, archive = pipeline
    session.commit.side_effect = [None, RuntimeError("audit unavailable")]
    monkeypatch.setattr(d._gate, "classify_async", AsyncMock(return_value=SensitivityDecision(
        category="unclassified", confidence=0.1, blocked=True, method="test",
    )))
    await d.dispatch_job(session, job)
    d._run_summarize_job.assert_not_awaited()
    d._run_extract_job.assert_not_awaited()
    d.fail_job.assert_awaited_once()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama"])
async def test_actual_call_boundary_blocks_remote(monkeypatch, settings, provider):
    settings.summarize_provider = settings.extract_provider = provider
    settings.ollama_base_url = "https://remote.example.com"
    ollama = AsyncMock(side_effect=AssertionError("Ollama egress"))
    monkeypatch.setattr(d, "_ollama_chat", ollama)
    monkeypatch.setattr("openai.AsyncOpenAI", MagicMock(side_effect=AssertionError("OpenAI egress")))
    monkeypatch.setattr("anthropic.AsyncAnthropic", MagicMock(side_effect=AssertionError("Anthropic egress")))
    assert await d._run_summarize_job("private", allow_external=False) is None
    assert await d._run_extract_job("private", allow_external=False) == []
    assert await d._extract_chunk("private", allow_external=False) == []
    ollama.assert_not_awaited()


async def test_actual_local_calls_and_proxy_isolation(monkeypatch, settings):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"message": {"content": '{"facts": []}'}}
    client = AsyncMock()
    client.post.return_value = response
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=client)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("httpx.AsyncClient", factory)
    await d._run_summarize_job("private", allow_external=False)
    assert await d._run_extract_job("private", allow_external=False) == []
    assert client.post.await_count == 2
    for call in factory.call_args_list:
        assert call.kwargs["trust_env"] is False
        assert call.kwargs["follow_redirects"] is False
    assert client.post.call_args.args[0] == "http://host.docker.internal:11434/api/chat"


async def test_remote_ollama_skipped_by_dispatch(monkeypatch, settings, pipeline):
    session, job, archive = pipeline
    settings.ollama_base_url = "https://remote.example.com"
    monkeypatch.setattr(d._gate, "classify_async", AsyncMock(return_value=SensitivityDecision(
        category="personal_profile", confidence=0.9, blocked=True, method="test",
    )))
    await d.dispatch_job(session, job)
    d._run_summarize_job.assert_not_awaited()
    d._run_extract_job.assert_not_awaited()
    d.complete_job.assert_awaited_once()


async def test_sensitive_hint_blocks_remote_summary_only(monkeypatch, settings, pipeline):
    session, job, archive = pipeline
    archive.metadata_json = {"sensitivity_hint": "private"}
    settings.summarize_provider = "openai"
    monkeypatch.setattr(d._gate, "classify_async", AsyncMock(return_value=SensitivityDecision(
        category="general", confidence=0.9, blocked=False, method="test",
    )))
    await d.dispatch_job(session, job)
    d._run_summarize_job.assert_not_awaited()
    d._run_extract_job.assert_awaited_once()
    assert d._run_extract_job.call_args.kwargs["allow_external"] is False
