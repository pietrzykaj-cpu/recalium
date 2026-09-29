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
from app.domain.model_context.profiles import (
    ApprovedLocalModelProfile,
    LocalModelRole,
    resolve_local_model_profile,
)

__all__ = [
    "ApprovedLocalModelProfile",
    "ContextSegment",
    "ContinuityBudgetExceeded",
    "ContinuityConsumptionError",
    "ContinuityConsumptionPayload",
    "LocalModelRole",
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
    "resolve_local_model_profile",
]
