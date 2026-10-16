"""Synthetic configuration/log tests; never start DB/worker/runtime tasks."""
import importlib
import logging
from pathlib import Path

import pytest
from app.infrastructure.settings import Settings, get_settings


def test_compose_explicit_default_false_and_env_precedence(monkeypatch, tmp_path):
    compose = Path(__file__).resolve().parents[3] / 'docker-compose.yml'
    contents = compose.read_text()
    app = contents.split('  recalium-app:', 1)[1].split('\nnetworks:', 1)[0]
    assert '      GENERATION_IDEMPOTENCY_ENABLED: ${GENERATION_IDEMPOTENCY_ENABLED:-false}' in app
    # Only harmless synthetic dotenv content. Never read root/backend live .env.
    stray = tmp_path / 'stray-settings.env'
    stray.write_text('GENERATION_IDEMPOTENCY_ENABLED=true\n')
    monkeypatch.setenv('GENERATION_IDEMPOTENCY_ENABLED', 'false')
    assert Settings(_env_file=stray, database_url=get_settings().database_url).generation_idempotency_enabled is False
    monkeypatch.delenv('GENERATION_IDEMPOTENCY_ENABLED')
    assert Settings(_env_file=None, database_url=get_settings().database_url).generation_idempotency_enabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [False, True])
async def test_startup_log_is_one_boolean_only(monkeypatch, caplog, value):
    main = importlib.import_module('app.main')
    settings = get_settings()
    monkeypatch.setattr(settings, 'generation_idempotency_enabled', value)
    monkeypatch.setattr(settings, 'app_bind_host', '0.0.0.0')
    monkeypatch.setattr(settings, 'app_auth_bearer', '')
    # Deliberate auth guard stops lifespan BEFORE DB initialization/task creation.
    with caplog.at_level(logging.INFO, logger='app.main'):
        with pytest.raises(RuntimeError, match='SECURITY'):
            async with main.lifespan(main.app):
                pytest.fail('Startup must stop before side effects')
    records = [r for r in caplog.records if r.getMessage().startswith('generation_idempotency_enabled=')]
    assert len(records) == 1 and records[0].levelno == logging.INFO
    assert records[0].getMessage() == f'generation_idempotency_enabled={value}'
    assert records[0].args == (value,)
