"""Regression coverage for Qwen final answers and the extraction envelope."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from app.worker import dispatcher as d

TEXT = 'My favourite test mug is purple. I always keep it beside my laptop when I work.'
FACT = {'fact_text': 'The test mug is purple.', 'source_span': 'My favourite test mug is purple.',
        'confidence_tier': 'high', 'entities': [], 'tags': ['preference']}


@pytest.fixture
def settings(monkeypatch):
    value = SimpleNamespace(ollama_model='qwen3:4b', ollama_base_url='http://host.docker.internal:11434',
                            ollama_api_key='', extract_provider='ollama', extract_model='auto',
                            summarize_provider='ollama', summarize_model='auto',
                            openai_api_key='', anthropic_api_key='')
    monkeypatch.setattr(d, 'get_settings', lambda: value)
    return value


def mock_http(monkeypatch, content, status=200):
    response = httpx.Response(status, json={'message': {'content': content, 'thinking': 'private trace'}},
                              request=httpx.Request('POST', 'http://localhost/api/chat'))
    client = AsyncMock()
    client.post.return_value = response
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=client)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr('httpx.AsyncClient', factory)
    return client, factory


@pytest.mark.parametrize('raw,expected', [
    ('A concise summary.', 'A concise summary.'),
    ('<think>Reasoning</think>\nFinal answer.', 'Final answer.'),
    ('Okay, let me consider this.\n</think>\n\nFinal answer.', 'Final answer.'),
    ('<think>first</think>\n<think>second</think>\nFinal.', 'Final.'),
    ('The text mentions the literal tag </think> in a sentence.',
     'The text mentions the literal tag </think> in a sentence.'),
    ('{"facts": [{"fact_text": "The tag is </think>."}]}',
     '{"facts": [{"fact_text": "The tag is </think>."}]}'),
])
async def test_final_answer_boundary(monkeypatch, settings, raw, expected):
    mock_http(monkeypatch, raw)
    assert await d._ollama_chat('system', TEXT, allow_external=False) == expected


@pytest.mark.parametrize('raw', ['', '<think>unfinished reasoning', '<think>only reasoning</think>',
                                 'unclosed-prefix reasoning\n</think>\n'])
async def test_missing_final_answer_is_failure(monkeypatch, settings, raw):
    mock_http(monkeypatch, raw)
    with pytest.raises(ValueError):
        await d._ollama_chat('system', TEXT, allow_external=False)


async def test_qwen_request_schema_and_locality(monkeypatch, settings):
    client, factory = mock_http(monkeypatch, '{"facts": []}')
    await d._extract_chunk(TEXT, allow_external=False)
    payload = client.post.call_args.kwargs['json']
    assert payload['think'] is False
    assert payload['messages'][0]['content'] == d.FACT_EXTRACTION_SYSTEM_PROMPT
    assert payload['messages'][1]['content'] == TEXT
    assert isinstance(payload['format'], dict)
    assert 'facts' in payload['format']['required']
    assert payload['format']['properties']['facts']['type'] == 'array'
    assert factory.call_args.kwargs['trust_env'] is False
    assert factory.call_args.kwargs['follow_redirects'] is False


async def test_other_models_do_not_receive_qwen_switch(monkeypatch, settings):
    settings.ollama_model = 'llama3.2'
    client, _ = mock_http(monkeypatch, 'Final.')
    await d._ollama_chat('system', TEXT)
    assert client.post.call_args.kwargs['json']['messages'][0]['content'] == 'system'


async def test_400_is_not_blindly_retried(monkeypatch, settings):
    client, _ = mock_http(monkeypatch, '', status=400)
    with pytest.raises(httpx.HTTPStatusError):
        await d._ollama_chat('system', TEXT)
    assert client.post.await_count == 1


@pytest.mark.parametrize('raw', [json.dumps(FACT), '{}', 'not JSON', '{"facts": null}',
                                 '{"facts": {}}', '{"facts": [42]}',
                                 '{"facts": [{"fact_text": "only text"}]}',
                                 '[{"facts": []}]', '{"facts": []} trailing junk',
                                 'Example {"facts": []}\nA final answer follows.'])
async def test_invalid_extraction_is_not_successful_empty_result(monkeypatch, settings, raw):
    monkeypatch.setattr(d, '_ollama_chat', AsyncMock(return_value=raw))
    with pytest.raises(ValueError):
        await d._extract_chunk(TEXT, allow_external=False)


async def test_legitimate_empty_extraction(monkeypatch, settings):
    monkeypatch.setattr(d, '_ollama_chat', AsyncMock(return_value='{"facts": []}'))
    assert await d._extract_chunk(TEXT, allow_external=False) == []


async def test_explicit_json_fence_is_supported(monkeypatch, settings):
    monkeypatch.setattr(d, '_ollama_chat', AsyncMock(return_value='```json\n{"facts": []}\n```'))
    assert await d._extract_chunk(TEXT, allow_external=False) == []


async def test_facts_survive_and_span_checks_remain(monkeypatch, settings):
    facts = [FACT, {**FACT, 'fact_text': 'Unsupported claim.', 'source_span': 'invented span'}]
    mock_http(monkeypatch, '<think>example {"facts": []}</think>\n' + json.dumps({'facts': facts}))
    result = await d._run_extract_job(TEXT, allow_external=False)
    assert len(result) == 2
    assert result[0]['source_span'] == FACT['source_span']
    assert result[1]['source_span'] == '' and result[1]['confidence_tier'] == 'low'


async def test_link_classifier_consumes_final_answer(monkeypatch, settings):
    mock_http(monkeypatch, 'Let me reason.\n</think>\nsupports')
    assert await d._classify_link_pair('A', 'B') == 'supports'
