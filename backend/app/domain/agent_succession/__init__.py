"""Model-independent succession envelopes."""
from app.domain.agent_succession.contracts import AgentSuccessionEnvelope, CurrentAgent, Predecessor
from app.domain.agent_succession.service import build_agent_succession_envelope, render_agent_succession_context

__all__ = ["AgentSuccessionEnvelope", "CurrentAgent", "Predecessor", "build_agent_succession_envelope", "render_agent_succession_context"]
