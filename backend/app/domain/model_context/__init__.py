"""Provider-neutral model-context adapter boundary."""

from app.domain.model_context.continuity import (
    ContinuityBudgetExceeded,
    ContinuityConsumptionError,
    build_continuity_consumption_payload,
    canonical_consumption_json,
)
from app.domain.model_context.contracts import (
    ContextSegment,
    ContinuityConsumptionPayload,
    ProviderChatRequest,
    ProviderMessage,
)
from app.domain.model_context.execution import (
    ModelExecutionResult,
    execute_ollama_succession_conversation,
)

__all__ = [
    "ContextSegment",
    "ContinuityBudgetExceeded",
    "ContinuityConsumptionError",
    "ContinuityConsumptionPayload",
    "ModelExecutionResult",
    "ProviderChatRequest",
    "ProviderMessage",
    "build_continuity_consumption_payload",
    "canonical_consumption_json",
    "execute_ollama_succession_conversation",
]
