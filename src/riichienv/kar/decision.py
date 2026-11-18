from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from riichienv import Action, ActionType, Observation, convert

from .actions import MappedAction, legal_candidates
from .observation import ObserverLedger, PublicSnapshot
from .tokenize import TokenSequence


@dataclass(frozen=True)
class ReplayOpportunity:
    """An owned observation and complete action identities at the opportunity, not its result."""

    snapshot: PublicSnapshot
    candidates: tuple[int, ...]
    native_actions: tuple[Action, ...]
    observation: Observation | None = None
    mapped: tuple[MappedAction, ...] = ()

    @classmethod
    def observe(cls, ledger: ObserverLedger, observation: Observation) -> ReplayOpportunity:
        ledger.check_observation(observation)
        snapshot = ledger.snapshot()
        source = snapshot.pending_kan_actor if snapshot.pending_kan_actor is not None else snapshot.last_discard_actor
        mapped = legal_candidates(observation, source_player=source)
        return cls(snapshot, tuple(item.global_id for item in mapped), tuple(observation.legal_actions()),
                   observation, mapped)

    @classmethod
    def from_kan_preview(cls, snapshot: PublicSnapshot, actions: Sequence[Action]) -> ReplayOpportunity:
        source = snapshot.pending_kan_actor
        if source is None or source == snapshot.observer_player_id:
            raise ValueError("Kan preview needs the matching public robbery window")
        ron = 542 + (source - snapshot.observer_player_id) % 4
        ids = {ActionType.PASS: 221, ActionType.RON: ron}
        if any(action.action_type not in ids or action.actor != snapshot.observer_player_id for action in actions):
            raise ValueError("Unexpected native action in the kan preview")
        return cls(snapshot, tuple(sorted({ids[action.action_type] for action in actions})), tuple(actions))

    def select(self, event: dict[str, Any]) -> tuple[int, Action]:
        selected = self.observation.select_action_from_mjai(event) if self.observation is not None else None
        if selected is None:
            raise ValueError(f"Observed {event['type']} has no matching frozen native legal action")
        matching = [item.global_id for item in self.mapped if len(item.actions) == 1
                    and item.actions[0].to_dict() == selected.to_dict()]
        if len(matching) != 1:
            raise ValueError("Observed native action does not have a unique complete action identity")
        return matching[0], selected

    def win_target(self, event: dict[str, Any]) -> int:
        actor, target = event["actor"], event["target"]
        kind = ActionType.TSUMO if actor == target else ActionType.RON
        actions = [action for action in self.native_actions if action.action_type == kind]
        if not actions:
            raise ValueError("Observed win is absent from the pre-resolution native legal actions")
        if "pai" in event:
            tile = convert.tid_to_mjai(convert.mjai_to_tid(event["pai"]))
            if not any(action.tile is not None and convert.tid_to_mjai(action.tile) == tile for action in actions):
                raise ValueError("Winning tile disagrees with the frozen native legal win")
        if kind == ActionType.TSUMO:
            return 219
        snapshot = self.snapshot
        source = snapshot.pending_kan_actor if snapshot.pending_kan_actor is not None else snapshot.last_discard_actor
        if source != target:
            raise ValueError("Ron source disagrees with the frozen response window")
        return 542 + (target - actor) % 4


@dataclass(frozen=True)
class ResponseEvidence:
    window_trigger_event_id: int
    label_evidence_event_id: int


@dataclass(frozen=True)
class ReplayDecision:
    snapshot: PublicSnapshot
    prefix: TokenSequence
    conditional: TokenSequence
    action_target_global: int
    trigger_event_id: int
    action_provenance: str = "explicit"
    response_evidence: ResponseEvidence | None = None
