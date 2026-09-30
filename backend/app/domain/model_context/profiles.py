"""Immutable application-owned profiles for certified local model execution."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Literal


class LocalModelRole(StrEnum):
    """Semantic execution roles; roles never carry authority or permissions."""

    CONTINUITY_REASONING = "continuity_reasoning"


@dataclass(frozen=True, slots=True)
class ApprovedLocalModelProfile:
    """An immutable, application-approved local execution profile."""

    role: LocalModelRole
    provider: Literal["ollama"]
    model: str
    model_digest: str
    num_ctx: int
    num_predict: int
    temperature: float
    seed: int
    think: bool
    keep_alive: Literal["0s"]
    stream: Literal[False]
    connect_seconds: float
    read_seconds: float
    write_seconds: float
    pool_seconds: float
    overall_seconds: float
    parser: Literal["ollama_final_content"]

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Approved local model name must not be blank")
        digest = self.model_digest.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Approved local model digest must be a SHA-256 hex digest")
        if type(self.num_ctx) is not int or self.num_ctx < 1:
            raise ValueError("Approved local model num_ctx must be a positive integer")
        if type(self.num_predict) is not int or self.num_predict < 1:
            raise ValueError("Approved local model num_predict must be a positive integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not isfinite(float(self.temperature))
        ):
            raise ValueError("Approved local model temperature must be finite")
        if type(self.seed) is not int:
            raise ValueError("Approved local model seed must be an integer")
        for name in (
            "connect_seconds",
            "read_seconds",
            "write_seconds",
            "pool_seconds",
            "overall_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"Approved local model {name} must be positive and finite")


_QWEN_CONTINUITY_REASONING = ApprovedLocalModelProfile(
    role=LocalModelRole.CONTINUITY_REASONING,
    provider="ollama",
    model="qwen3:4b",
    model_digest="359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7",
    num_ctx=4096,
    num_predict=512,
    temperature=0,
    seed=20260915,
    think=False,
    keep_alive="0s",
    stream=False,
    connect_seconds=5.0,
    read_seconds=120.0,
    write_seconds=10.0,
    pool_seconds=5.0,
    overall_seconds=135.0,
    parser="ollama_final_content",
)

_APPROVED_LOCAL_MODEL_PROFILES = MappingProxyType(
    {LocalModelRole.CONTINUITY_REASONING: _QWEN_CONTINUITY_REASONING}
)


def resolve_local_model_profile(role: LocalModelRole | str) -> ApprovedLocalModelProfile:
    """Resolve an exact approved role with no default or provider fallback."""
    try:
        normalized_role = LocalModelRole(role)
        return _APPROVED_LOCAL_MODEL_PROFILES[normalized_role]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unknown approved local model role: {role!r}") from exc
