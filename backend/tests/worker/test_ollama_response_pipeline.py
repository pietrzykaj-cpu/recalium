import hashlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy import select
from app.domain.archive.models import RawArchiveItem
from app.domain.derived_memory.models import Fact
from app.domain.jobs.models import Job
from app.domain.policy.gate import SensitivityDecision
from app.worker import dispatcher as d


async def test_invalid_ollama_envelope_marks_job_retryable(db_session_phase2, monkeypatch):
    session = db_session_phase2
    content = f'Fictional purple mug test {uuid.uuid4()}'
    item = RawArchiveItem(id=uuid.uuid4(), source_type='test', raw_content=content,
                          content_hash=hashlib.sha256(content.encode()).hexdigest(),
                          metadata_json={'processing_mode': 'local_only'})
    session.add(item)
    await session.flush()
    job = Job(id=uuid.uuid4(), job_type='process_archive_item', raw_archive_id=item.id,
              status='claimed', attempts=1)
    session.add(job)
    await session.commit()
    settings = SimpleNamespace(ollama_model='qwen3:4b', ollama_base_url='http://localhost:11434',
                               ollama_api_key='', extract_provider='ollama', extract_model='auto',
                               summarize_provider='ollama', summarize_model='auto',
                               openai_api_key='', anthropic_api_key='')
    monkeypatch.setattr(d, 'get_settings', lambda: settings)
    monkeypatch.setattr(d._gate, 'classify_async', AsyncMock(return_value=SensitivityDecision(
        category='unclassified', confidence=0.1, blocked=True, method='test')))
    monkeypatch.setattr(d, '_run_summarize_job', AsyncMock(return_value='A clean summary.'))
    monkeypatch.setattr(d, '_ollama_chat', AsyncMock(return_value='{"fact_text": "a bare fact"}'))
    await d.dispatch_job(session, job)
    await session.refresh(job)
    assert job.status == 'retryable_failed'
    assert 'facts schema' in job.error_message
    assert 'a bare fact' not in job.error_message
    assert (await session.scalars(select(Fact).where(Fact.raw_archive_id == job.raw_archive_id))).all() == []
