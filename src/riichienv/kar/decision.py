from __future__ import annotations

from dataclasses import dataclass

from .observation import PublicSnapshot
from .tokenize import TokenSequence


@dataclass(frozen=True)
class ReplayDecision:
    snapshot: PublicSnapshot
    prefix: TokenSequence
    conditional: TokenSequence
    action_target_global: int
    trigger_event_id: int
    action_provenance: str = "explicit"
