"""Regression coverage for Qwen final answers and the extraction envelope."""
import json
import logging
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.worker import dispatcher as d

TEXT = 'My favourite test mug is purple. I always keep it beside my laptop when I work.'
FACT = {'fact_text': 'The test mug is purple.', 'source_span': 'My favourite test mug is purple.',
        'confidence_tier': 'high', 'entities': [], 'tags': ['preference']}


class FakeCoordinator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @asynccontextmanager
    async def acquire(self, **kwargs: object) -> AsyncIterator[None]:
        self.calls.append(kwargs)
        yield


@pytest.fixture(autouse=True)
def _clean_db_between_tests() -> None:
    """Override the repository DB fixture for this mocked worker module."""


@pytest.fixture(autouse=True)
def coordinator(monkeypatch):
    value = FakeCoordinator()
    monkeypatch.setattr(d, 'get_local_inference_coordinator', lambda: value)
    return value


@pytest.fixture
def settings(monkeypatch):
    value = SimpleNamespace(ollama_model='qwen3:4b', ollama_base_url='http://host.docker.internal:11434',
                            ollama_api_key='', extract_provider='ollama', extract_model='auto',
                            summarize_provider='ollama', summarize_model='auto',
                            openai_api_key='', anthropic_api_key='')
    monkeypatch.setattr(d, 'get_settings', lambda: value)
    return value


def mock_http(monkeypatch, content, status=200, metadata=None):
    payload = {'message': {'content': content, 'thinking': 'private trace'}}
    payload.update(metadata or {})
    response = httpx.Response(status, json=payload,
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


async def test_qwen_request_schema_and_locality(monkeypatch, settings, coordinator):
    client, factory = mock_http(monkeypatch, '{"facts": []}')
    await d._extract_chunk(TEXT, allow_external=False)
    payload = client.post.call_args.kwargs['json']
    assert payload['think'] is False
    assert payload['keep_alive'] == '0s'
    assert payload['messages'][0]['content'] == d.FACT_EXTRACTION_SYSTEM_PROMPT
    assert payload['messages'][1]['content'] == TEXT
    assert isinstance(payload['format'], dict)
    assert 'facts' in payload['format']['required']
    assert payload['format']['properties']['facts']['type'] == 'array'
    assert factory.call_args.kwargs['trust_env'] is False
    assert factory.call_args.kwargs['follow_redirects'] is False
    assert coordinator.calls == [{
        'base_url': settings.ollama_base_url,
        'expected_model': settings.ollama_model,
        'expected_digest': None,
        'headers': None,
    }]


async def test_other_models_do_not_receive_qwen_switch(monkeypatch, settings):
    settings.ollama_model = 'llama3.2'
    client, _ = mock_http(monkeypatch, 'Final.')
    await d._ollama_chat('system', TEXT)
    assert client.post.call_args.kwargs['json']['messages'][0]['content'] == 'system'


async def test_remote_ollama_bypasses_local_coordinator(monkeypatch, settings, coordinator):
    settings.ollama_base_url = 'https://ollama.example.com'
    client, factory = mock_http(monkeypatch, 'Final.')

    assert await d._ollama_chat('system', TEXT, allow_external=True) == 'Final.'

    assert coordinator.calls == []
    assert factory.call_args.kwargs['trust_env'] is True
    assert 'keep_alive' not in client.post.call_args.kwargs['json']


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


def completion_records(caplog):
    return [record for record in caplog.records
            if record.name == d.logger.name
            and record.msg == 'Ollama completion diagnostics: %s']


@pytest.mark.parametrize('reason', ['stop', 'length'])
async def test_completion_diagnostics_are_observable_without_content_changes(
    monkeypatch, settings, caplog, reason,
):
    metadata = {
        'done': True, 'done_reason': reason, 'prompt_eval_count': 91, 'eval_count': 37,
        'total_duration': 9000, 'load_duration': 500,
        'prompt_eval_duration': 2000, 'eval_duration': 6500,
        'context': [101, 202], 'tool_calls': [{'private': 'tool-secret'}],
        'reasoning': 'reasoning-secret', 'prompt': 'prompt-secret',
        'unknown_provider_field': 'unknown-secret',
    }
    settings.ollama_api_key = 'key-secret'
    mock_http(monkeypatch, '<think>embedded-secret</think>\nFinal-secret.', metadata=metadata)
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        result = await d._ollama_chat('system-secret', TEXT, allow_external=False)
    assert type(result) is str
    assert result == 'Final-secret.'
    records = completion_records(caplog)
    assert len(records) == 1
    assert records[0].args == {
        'done': True, 'done_reason': reason, 'prompt_eval_count': 91, 'eval_count': 37,
        'total_duration_ns': 9000, 'load_duration_ns': 500,
        'prompt_eval_duration_ns': 2000, 'eval_duration_ns': 6500,
    }
    for private in ['private trace', 'tool-secret', 'reasoning-secret', 'prompt-secret',
                    'unknown-secret', 'key-secret', 'embedded-secret', 'Final-secret.',
                    'system-secret', TEXT, '[101, 202]']:
        assert private not in caplog.text


@pytest.mark.parametrize('metadata', [None, {'reasoning': 'secret', 'context': [1]}, {
    'done': 1, 'done_reason': 'bad\nreason', 'eval_count': True,
    'prompt_eval_count': '91', 'total_duration': -1, 'load_duration': 1.5,
    'eval_duration': '6500', 'prompt_eval_duration': False,
}])
async def test_missing_or_invalid_diagnostics_remain_harmless(
    monkeypatch, settings, caplog, metadata,
):
    mock_http(monkeypatch, 'Final.', metadata=metadata)
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        assert await d._ollama_chat('system', TEXT, allow_external=False) == 'Final.'
    assert completion_records(caplog) == []


async def test_partial_diagnostics_are_filtered(monkeypatch, settings, caplog):
    mock_http(monkeypatch, 'Final.', metadata={
        'done': False, 'done_reason': 42, 'eval_count': 0, 'total_duration': -1,
    })
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        assert await d._ollama_chat('system', TEXT, allow_external=False) == 'Final.'
    assert completion_records(caplog)[0].args == {'done': False, 'eval_count': 0}


async def test_diagnostics_survive_existing_content_failure(monkeypatch, settings, caplog):
    mock_http(monkeypatch, '<think>private trace</think>', metadata={
        'done': True, 'done_reason': 'length', 'eval_count': 512,
    })
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        with pytest.raises(ValueError, match='^Ollama returned no final answer$'):
            await d._ollama_chat('system', TEXT, allow_external=False)
    assert completion_records(caplog)[0].args == {
        'done': True, 'done_reason': 'length', 'eval_count': 512,
    }
    assert 'private trace' not in caplog.text


async def test_http_failure_is_not_a_completion(monkeypatch, settings, caplog):
    client, _ = mock_http(monkeypatch, 'Final.', status=503, metadata={'done': True})
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        with pytest.raises(httpx.HTTPStatusError):
            await d._ollama_chat('system', TEXT, allow_external=False)
    assert completion_records(caplog) == []
    assert client.post.await_count == 1


@pytest.mark.parametrize('operation,content,expected', [
    ('summary', 'Summary.', 'Summary.'),
    ('extract', '{"facts": []}', []),
    ('link', 'supports', 'supports'),
])
async def test_worker_operations_keep_diagnostics_observable(
    monkeypatch, settings, caplog, operation, content, expected,
):
    client, _ = mock_http(monkeypatch, content, metadata={'done_reason': 'stop'})
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        if operation == 'summary':
            result = await d._run_summarize_job(TEXT, allow_external=False)
        elif operation == 'extract':
            result = await d._extract_chunk(TEXT, allow_external=False)
        else:
            result = await d._classify_link_pair('A', 'B')
    assert result == expected
    assert client.post.await_count == 1
    payload = client.post.call_args.kwargs['json']
    assert payload['model'] == settings.ollama_model
    assert payload['stream'] is False
    assert payload['think'] is False
    assert payload['keep_alive'] == '0s'
    assert payload['options'] == {'temperature': 0, 'num_ctx': 4096}
    assert 'num_batch' not in payload['options']
    assert 'num_ubatch' not in payload['options']
    assert len(completion_records(caplog)) == 1
    assert completion_records(caplog)[0].args == {'done_reason': 'stop'}


async def test_worker_keeps_legacy_reasoning_metadata_tolerance(monkeypatch, settings, caplog):
    mock_http(monkeypatch, 'unused', metadata={
        'done_reason': 'stop',
        'message': {'content': 'Final.', 'thinking': {'private': 'secret'},
                    'reasoning': ['private-secret'], 'tool_calls': ['tool-secret']},
    })
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        assert await d._ollama_chat('system', TEXT, allow_external=False) == 'Final.'
    assert completion_records(caplog)[0].args == {'done_reason': 'stop'}
    assert 'secret' not in caplog.text


@pytest.mark.parametrize('size', [2047, 2048])
def test_summary_bound_keeps_input_at_or_below_limit(size):
    text = 'x' * size
    result, bounded = d._bound_summary_input(text)
    assert d.SUMMARY_INPUT_MAX_BYTES == 2048
    assert result == text
    assert result.encode('utf-8') == text.encode('utf-8')
    assert bounded is False


@pytest.mark.parametrize('text', [
    'H' + 'x' * 2047 + 'T',
    'HEAD\n' + 'middle ' * 500_000 + '\nTAIL',
    'Zażółć gęślą jaźń. ' * 200,
    '漢字中文日本語' * 300,
    '😀🚀🌍' * 500,
    'Zażółć 漢字 😀 ' * 300,
], ids=['max-plus-one', 'several-mb', 'polish', 'cjk', 'emoji', 'mixed'])
def test_summary_bound_preserves_utf8_edges_and_exact_byte_accounting(text):
    result, bounded = d._bound_summary_input(text)
    assert bounded is True
    assert d._bound_summary_input(text) == (result, True)
    encoded = result.encode('utf-8')
    assert encoded.decode('utf-8', errors='strict') == result
    assert len(encoded) <= d.SUMMARY_INPUT_MAX_BYTES
    assert '\ufffd' not in result
    marker = re.search(r'\n\[\.\.\. (\d+) bytes omitted \.\.\.\]\n', result)
    assert marker is not None
    head, tail = result[:marker.start()], result[marker.end():]
    assert head and tail
    assert text.startswith(head) and text.endswith(tail)
    omitted = int(marker.group(1))
    assert omitted > 0
    assert len(text.encode('utf-8')) == len(head.encode('utf-8')) + omitted + len(tail.encode('utf-8'))
    assert abs(len(head.encode('utf-8')) - len(tail.encode('utf-8'))) <= 4


@pytest.mark.parametrize('text', [TEXT, 'HEAD\n' + 'long text ' * 800 + '\nTAIL'], ids=['small', 'oversize'])
async def test_summary_request_is_bounded_only_when_needed(
    monkeypatch, settings, coordinator, caplog, text,
):
    client, _ = mock_http(monkeypatch, 'Summary.', metadata={'done_reason': 'stop'})
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        assert await d._run_summarize_job(text, allow_external=False) == 'Summary.'
    client.post.assert_awaited_once()
    payload = client.post.call_args.kwargs['json']
    assert payload == {
        'model': settings.ollama_model,
        'messages': [
            {'role': 'system', 'content': d.SUMMARIZATION_SYSTEM_PROMPT},
            {'role': 'user', 'content': d._bound_summary_input(text)[0]},
        ],
        'stream': False, 'think': False, 'keep_alive': '0s',
        'options': {'temperature': 0, 'num_ctx': 4096},
    }
    assert 'num_batch' not in payload['options']
    assert 'num_ubatch' not in payload['options']
    if len(text.encode('utf-8')) > d.SUMMARY_INPUT_MAX_BYTES:
        sent_message = payload['messages'][1]['content']
        assert sent_message != text
        assert len(sent_message.encode('utf-8')) <= d.SUMMARY_INPUT_MAX_BYTES
        assert re.search(r'\n\[\.\.\. \d+ bytes omitted \.\.\.\]\n', sent_message)
        assert sent_message.startswith('HEAD\n')
        assert sent_message.endswith('\nTAIL')
    assert len(coordinator.calls) == 1
    assert completion_records(caplog)[0].args == {'done_reason': 'stop'}
    assert text not in caplog.text


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
async def test_external_summary_receives_full_original_input(monkeypatch, settings, provider):
    text = 'Zażółć 漢字 😀 ' * 600
    settings.summarize_provider = provider
    settings.openai_api_key = settings.anthropic_api_key = 'test-only'
    create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='Summary.'))],
        content=[SimpleNamespace(text='Summary.')],
    ))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        messages=SimpleNamespace(create=create),
    )
    sdk_name = 'AsyncOpenAI' if provider == 'openai' else 'AsyncAnthropic'
    monkeypatch.setitem(sys.modules, provider, SimpleNamespace(**{sdk_name: lambda **_: client}))
    monkeypatch.setattr(d, '_bound_summary_input', MagicMock(side_effect=AssertionError('Must not clip')))
    monkeypatch.setattr(d, '_ollama_chat', AsyncMock(side_effect=AssertionError('No fallback')))
    assert await d._run_summarize_job(text) == 'Summary.'
    create.assert_awaited_once()
    assert create.call_args.kwargs['messages'][-1] == {'role': 'user', 'content': text}
    d._bound_summary_input.assert_not_called()
    d._ollama_chat.assert_not_awaited()


async def test_summary_clipping_does_not_change_extraction_chunks(monkeypatch, settings):
    text = 'User: ' + 'Zażółć 漢字 😀 ' * 400 + '\nAssistant: ' + 'tail ' * 500
    chunks = d._split_conversation(text)
    client, _ = mock_http(monkeypatch, '{"facts": []}')
    monkeypatch.setattr(d, '_bound_summary_input', MagicMock(side_effect=AssertionError('Must not clip')))
    assert await d._run_extract_job(text, allow_external=False) == []
    users = [call.kwargs['json']['messages'][1]['content'] for call in client.post.call_args_list]
    assert users == chunks
    assert ''.join(users) == text
    assert all('bytes omitted' not in user for user in users)
    d._bound_summary_input.assert_not_called()


async def test_summary_clipping_does_not_change_link_payload(monkeypatch, settings):
    source, target = 'source ' * 600, 'target ' * 600
    client, _ = mock_http(monkeypatch, 'supports')
    monkeypatch.setattr(d, '_bound_summary_input', MagicMock(side_effect=AssertionError('Must not clip')))
    assert await d._classify_link_pair(source, target) == 'supports'
    client.post.assert_awaited_once()
    assert client.post.call_args.kwargs['json']['messages'][1]['content'] == (
        f'Fact A: {source}\nFact B: {target}\n\n'
        'How does Fact B relate to Fact A? '
        'Reply with exactly one word: supports, elaborates, contradicts, or unrelated.'
    )
    d._bound_summary_input.assert_not_called()


async def test_bounded_summary_http_failure_has_no_retry_or_fallback(monkeypatch, settings, coordinator):
    client, _ = mock_http(monkeypatch, '', status=503)
    with pytest.raises(httpx.HTTPStatusError):
        await d._run_summarize_job('x' * 6000, allow_external=False)
    client.post.assert_awaited_once()
    assert len(coordinator.calls) == 1
