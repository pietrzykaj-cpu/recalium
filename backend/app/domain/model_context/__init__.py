"""Provider-neutral model-context adapter boundary."""

from app.domain.model_context.continuity import (
    ContinuityBudgetExceeded,
    ContinuityConsumptionError,
    build_continuity_consumption_payload,
    build_continuity_provider_request,
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
    OllamaExecutionTimeouts,
    execute_local_ollama_provider_request,
    execute_ollama_succession_conversation,
)
from app.domain.model_context.ollama import OllamaContinuityOptions

__all__ = [
    "ContextSegment",
    "ContinuityBudgetExceeded",
    "ContinuityConsumptionError",
    "ContinuityConsumptionPayload",
    "ModelExecutionResult",
    "OllamaContinuityOptions",
    "OllamaExecutionTimeouts",
    "ProviderChatRequest",
    "ProviderMessage",
    "build_continuity_consumption_payload",
    "build_continuity_provider_request",
    "canonical_consumption_json",
    "execute_local_ollama_provider_request",
    "execute_ollama_succession_conversation",
]
