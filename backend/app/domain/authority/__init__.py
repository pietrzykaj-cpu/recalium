"""Pure in-memory authority and currentness domain primitives."""

from .contracts import AuthorityEdge, AuthorityRecord, AuthorityStateResult
from .service import (
    AuthorityGraph,
    AuthorityValidationError,
    evaluate_current_state,
    validate_supersession_edge,
)

__all__ = [
    "AuthorityEdge",
    "AuthorityRecord",
    "AuthorityStateResult",
    "AuthorityGraph",
    "AuthorityValidationError",
    "evaluate_current_state",
    "validate_supersession_edge",
]
