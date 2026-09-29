"""Approved local-model policy profiles."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.domain.model_context.profiles import (
    LocalModelRole,
    resolve_local_model_profile,
)


@pytest.fixture(autouse=True)
def _clean_db_between_tests() -> None:
    """Override the integration autouse fixture; these policy tests need no DB."""


def test_continuity_reasoning_resolves_exact_qwen_profile() -> None:
    profile = resolve_local_model_profile(LocalModelRole.CONTINUITY_REASONING)

    assert profile.role is LocalModelRole.CONTINUITY_REASONING
    assert profile.provider == "ollama"
    assert profile.model == "qwen3:4b"
    assert profile.model_digest == (
        "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
    )
    assert profile.num_ctx == 4096
    assert profile.num_predict == 512
    assert profile.temperature == 0
    assert profile.seed == 20260915
    assert profile.think is False
    assert profile.keep_alive == "0s"
    assert profile.stream is False
    assert profile.connect_seconds == 5.0
    assert profile.read_seconds == 120.0
    assert profile.write_seconds == 10.0
    assert profile.pool_seconds == 5.0
    assert profile.overall_seconds == 135.0
    assert profile.parser == "ollama_final_content"


@pytest.mark.parametrize("role", ["", "continuity_certification", "unknown"])
def test_unknown_role_fails_closed_without_fallback(role: str) -> None:
    with pytest.raises(ValueError, match="approved local model role"):
        resolve_local_model_profile(role)


def test_profile_is_immutable() -> None:
    profile = resolve_local_model_profile(LocalModelRole.CONTINUITY_REASONING)

    with pytest.raises(FrozenInstanceError):
        profile.model = "other"  # type: ignore[misc]


def test_resolver_returns_the_registered_profile_not_a_caller_copy() -> None:
    first = resolve_local_model_profile("continuity_reasoning")
    second = resolve_local_model_profile(LocalModelRole.CONTINUITY_REASONING)

    assert first is second
