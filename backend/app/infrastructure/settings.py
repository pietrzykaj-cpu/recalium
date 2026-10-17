"""Application settings loaded from environment variables via pydantic-settings.

All configuration comes from .env / environment variables.
API keys are NEVER stored here — they are read from environment at validation time only.
"""
from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class Settings(BaseSettings):
    """Runtime configuration. All values from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars (e.g. CI variables)
    )

    # Database
    database_url: str
    postgres_user: str = "recalium"
    postgres_password: str = "recalium"
    postgres_db: str = "recalium"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Application
    app_env: str = "production"
    log_level: str = "info"
    app_port: int = 8000
    # R2: one dormant writer/reader cutover; production flip requires its own gate.
    generation_idempotency_enabled: bool = False

    # Network exposure settings
    app_bind_host: str = "127.0.0.1"
    # When set to anything other than 127.0.0.1, auth is required.
    app_auth_bearer: str = ""
    # Bearer token required when app_bind_host != "127.0.0.1". Must be set if exposing beyond localhost.

    # Watched import folder (INGT-04)
    watch_dir: str = ""  # Empty string = watcher disabled
    watch_poll_interval: int = 10  # Seconds between directory polls

    # BYOK keys (optional — empty string means not configured)
    # These are read at runtime for validation only; never persisted to DB.
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = ""
    ollama_api_key: str = ""
    ollama_model: str = "llama3.2"
    # Default Ollama model for summarization/extraction; override via OLLAMA_MODEL env var

    # Per-function model names (F1) — overridable via env; "auto" uses the
    # provider default (gpt-4o-mini for OpenAI, claude-3-haiku for Anthropic,
    # OLLAMA_MODEL for Ollama).
    summarize_model: str = "auto"
    extract_model: str = "auto"
    embed_model: str = "all-MiniLM-L6-v2"
    # Immutable approved MiniLM snapshot; no remote/cache resolution fallback.
    local_model_assets_dir: str = "/app/local-models/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    # Per-function provider routing (F2, BYOK-08) — "auto" | "openai" |
    # "anthropic" | "ollama". "auto" falls back to the first configured key
    # (openai → anthropic → ollama). An explicit provider without a key skips
    # the job (pending_provider) rather than silently falling through.
    summarize_provider: str = "auto"
    extract_provider: str = "auto"
    embed_provider: str = "sentence-transformers"

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        if v not in ("development", "production"):
            raise ValueError(f"app_env must be 'development' or 'production', got: {v!r}")
        return v

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def requires_auth(self) -> bool:
        return self.app_bind_host not in _LOCALHOST_HOSTS


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
