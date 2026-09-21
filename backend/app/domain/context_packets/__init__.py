"""Typed, non-persistent context packets for model-independent continuity."""

from app.domain.context_packets.contracts import ContextPacket
from app.domain.context_packets.service import build_context_packet

__all__ = ["ContextPacket", "build_context_packet"]
