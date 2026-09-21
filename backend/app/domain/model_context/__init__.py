"""Provider-neutral model-context adapter boundary."""
from app.domain.model_context.contracts import ContextSegment, ProviderChatRequest, ProviderMessage
from app.domain.model_context.execution import ModelExecutionResult, execute_ollama_succession_conversation

__all__ = [
    "ContextSegment",
    "ModelExecutionResult",
    "ProviderChatRequest",
    "ProviderMessage",
    "execute_ollama_succession_conversation",
]
