from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from riichienv import convert
from riichienv._riichienv import NativeObserverState

from .contracts import canonical_bytes
from .event_semantics import PUBLIC_CLOCK_VERSION, PublicEventClock
from .rule_features import (
    current_rank_features,
    full_suji_exclusion,
    genbutsu,
    known_count_base34,
    no_visible_possible_ryanmen_shape,
    public_dora_types,
)


def _tile(value: str) -> int:
    return convert.mjai_to_tid(value)


def _public_tile(value: str) -> str:
    return convert.tid_to_mjai(_tile(value))


@dataclass(frozen=True)
class PublicMeld:
    kind: str
    tiles: tuple[int, ...]
    source: int | None = None


@dataclass(frozen=True)
class PublicSnapshot:
    observer_player_id: int
    cutoff_event_id: int
    initial_seat_order_player_ids: tuple[int, ...]
    hand: tuple[int, ...]
    scores: tuple[int, ...]
    discard_history: tuple[tuple[int, ...], ...]
    current_rivers: tuple[tuple[int, ...], ...]
    melds: tuple[tuple[PublicMeld, ...], ...]
    dora_indicators: tuple[int, ...]
    riichi_declared: tuple[bool, ...]
    riichi_accepted: tuple[bool, ...]
    remaining_draws: int | None
    last_discard_actor: int | None
    last_discard_tile: int | None
    oya: int
    honba: int
    riichi_sticks: int
    history_complete: bool
    physical_ledger_complete: bool
    events: tuple[str, ...]
    pending_kan_actor: int | None
    round_wind: int = 0
    public_hand_sizes: tuple[int, ...] = (13, 13, 13, 13)
    draw_counts: tuple[int, ...] = (0, 0, 0, 0)
    riichi_discard_ordinals: tuple[int | None, ...] = (None, None, None, None)
    current_draw_actor: int | None = None
    drawn_tile: int | None = None
    decision_phase: str = "unknown"
    post_call_actor: int | None = None
    forced_tedashi: bool = False
    pending_kan_kind: str | None = None
    pending_kan_tile: int | None = None
    history_positions: tuple[int, ...] = ()
    visibility_complete: bool = True
    public_clock_version: str | None = None
    # A process-local ledger lifetime, never a public event or model input.
    _history_session: object | None = field(default=None, repr=False, compare=False)

    def to_bytes(self) -> bytes:
        record = asdict(self)
        record.pop("_history_session")
        return canonical_bytes(record)

    def known_counts(self) -> tuple[int, ...] | None:
        """Count physical visible regions only when both ledger and visibility are complete.

        Public discard history has its own completeness flag and is not required
        for this count: called-away tiles live in melds, not the current rivers.
        """
        if not self.physical_ledger_complete or not self.visibility_complete:
            return None
        return known_count_base34(
            [t // 4 for t in self.hand],
            [[t // 4 for t in river] for river in self.current_rivers],
            [[t // 4 for meld in player for t in meld.tiles] for player in self.melds],
            [t // 4 for t in self.dora_indicators],
        )


PASS_NEXT_DRAW = "inferred_unique_next_draw_pass"
PASS_KAN_REPLACEMENT = "inferred_unique_kan_replacement_pass"
PASS_CALL_PRIORITY = "inferred_unique_call_priority_pass"
PASS_TERMINAL_NO_RON = "inferred_unique_terminal_no_ron_pass"
SELF_PASS_PROVENANCES = frozenset(
    {
        "native_committed_pass",
        PASS_NEXT_DRAW,
        PASS_KAN_REPLACEMENT,
        PASS_CALL_PRIORITY,
        PASS_TERMINAL_NO_RON,
    }
)


@dataclass(frozen=True)
class SelfPassTrigger:
    observer_player_id: int
    trigger_event_id: int
    round_event_id: int
    trigger_type: str
    target: int
    pai: str


class ObserverLedger:
    def __init__(self, observer_player_id: int, initial_seat_order_player_ids: tuple[int, ...]):
        if observer_player_id not in range(4) or sorted(initial_seat_order_player_ids) != list(range(4)):
            raise ValueError("Four stable player IDs and an explicit initial seat order are required")
        self.observer = observer_player_id
        self._history_session = object()  # ObserverLedger owns one complete game's visible log.
        self.initial_order = tuple(initial_seat_order_player_ids)
        self.event_id = -1
        self._native = NativeObserverState(observer_player_id)
        self._state: dict[str, Any] = {}
        self._refresh_state()
        self.events: list[str] = []
        self.history_positions: list[int] = []
        self.logical_clock = 0
        self._event_clock = PublicEventClock()
        self._round_event_id = -1
        self._public_events: list[dict[str, Any]] = []
        self._public_event_indices: list[int] = []
        self._pass_triggers: dict[int, SelfPassTrigger] = {}
        self._active_pass_trigger: int | None = None
        self._pass_confirmations: dict[int, int] = {}
        self._recorded_passes: set[int] = set()

    # These are detached read-only projections of the Rust-owned facts. Python
    # owns only model history, clocks, cutoff/provenance and private Pass records.
    _ENGINE_FIELDS = frozenset(
        {
            "have_start",
            "have_round",
            "own_hand_complete",
            "hand",
            "scores",
            "history",
            "rivers",
            "melds",
            "dora",
            "declared",
            "accepted",
            "remaining",
            "last_actor",
            "last_tile",
            "oya",
            "honba",
            "riichi_sticks",
            "pending",
            "round_wind",
            "public_hand_sizes",
            "draw_counts",
            "riichi_ordinals",
            "current_draw_actor",
            "drawn_tile",
            "decision_phase",
            "post_call_actor",
            "interrupted",
            "visibility_complete",
        }
    )

    def _refresh_state(self) -> None:
        state = json.loads(self._native.state_json())
        state["melds"] = tuple(
            tuple(PublicMeld(m["kind"], tuple(m["tiles"]), m["source"]) for m in player) for player in state["melds"]
        )
        for name in ("history", "rivers"):
            state[name] = tuple(tuple(player) for player in state[name])
        for name in (
            "hand",
            "scores",
            "dora",
            "declared",
            "accepted",
            "public_hand_sizes",
            "draw_counts",
            "riichi_ordinals",
        ):
            state[name] = tuple(state[name])
        if state["pending"] is not None:
            kind, actor, tiles = state["pending"]
            state["pending"] = kind, actor, tuple(tiles)
        self._state = state

    def __getattr__(self, name: str) -> Any:
        if name in self._ENGINE_FIELDS:
            return self._state[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._ENGINE_FIELDS:
            raise AttributeError(f"{name} is owned by the native observer; apply an observed event")
        object.__setattr__(self, name, value)

    def pass_trigger(self) -> SelfPassTrigger:
        """Capture owner-only response context, not a submission or approval."""
        trigger = self._pass_triggers.get(self._active_pass_trigger if self._active_pass_trigger is not None else -1)
        if (
            trigger is None
            or trigger.target == self.observer
            or self.decision_phase not in {"discard_response", "kan_response"}
        ):
            raise ValueError("A self Pass needs this observer's current public response trigger")
        if trigger.trigger_type == "kakan" and not any(
            meld.kind == "pon" and meld.tiles[0] // 4 == _tile(trigger.pai) // 4 for meld in self.melds[trigger.target]
        ):
            raise ValueError("A kakan response requires the existing committed pon")
        return trigger

    def _self_pass_confirmation(self, trigger: SelfPassTrigger, provenance: str) -> int | None:
        if (
            not isinstance(trigger, SelfPassTrigger)
            or any(
                type(value) is not int
                for value in (
                    trigger.observer_player_id,
                    trigger.trigger_event_id,
                    trigger.round_event_id,
                    trigger.target,
                )
            )
            or trigger.observer_player_id != self.observer
            or trigger.target == self.observer
            or self._pass_triggers.get(trigger.trigger_event_id) != trigger
        ):
            raise ValueError("Foreign or unknown private Pass trigger")
        if not isinstance(provenance, str) or provenance not in SELF_PASS_PROVENANCES:
            raise ValueError("Private Pass requires confirmed, supported provenance")
        kan = trigger.trigger_type in {"ankan", "kakan"}
        invalid = "An unestablished/overridden call cannot be inferred as private Pass"
        if (
            provenance in {PASS_NEXT_DRAW, PASS_CALL_PRIORITY}
            and trigger.trigger_type != "dahai"
            or provenance == PASS_KAN_REPLACEMENT
            and not kan
        ):
            raise ValueError(invalid)
        confirmed = self._pass_confirmations.get(trigger.trigger_event_id)
        if confirmed is None:
            if self._active_pass_trigger != trigger.trigger_event_id:
                raise ValueError("Private Pass window ended without a legal closing event")
            return None
        closing = self._public_events[confirmed]
        kind = closing["type"]
        if (
            provenance == PASS_NEXT_DRAW
            and (kind != "tsumo" or closing.get("actor") != (trigger.target + 1) % 4)
            or provenance == PASS_KAN_REPLACEMENT
            and (kind != "tsumo" or closing.get("actor") != trigger.target)
            or provenance == PASS_CALL_PRIORITY
            and kind not in {"chi", "pon", "kan", "daiminkan"}
            or provenance == PASS_TERMINAL_NO_RON
            and kind not in ({"hora"} if kan else {"hora", "ryukyoku"})
        ):
            raise ValueError(invalid)
        return confirmed

    def self_pass_ready(self, trigger: SelfPassTrigger, *, provenance: str) -> bool:
        """Check the anchor without mutation; only a valid, still-open trigger may wait."""
        return self._self_pass_confirmation(trigger, provenance) is not None

    def record_self_pass(self, trigger: SelfPassTrigger, *, provenance: str) -> bool:
        """Materialize a confirmed owner action at its public closing-event anchor.

        Public event IDs/clocks do not advance. The caller must have submitted a
        real Pass successfully, or use the frozen unique-next-draw inference.
        A previously frozen snapshot is never mutated.
        """
        confirmed = self._self_pass_confirmation(trigger, provenance)
        if confirmed is None:
            raise ValueError("Private Pass trigger has no observed closing event")
        if trigger.trigger_event_id in self._recorded_passes:
            return False
        marker = {
            "type": "self_pass",
            "actor": self.observer,
            "target": trigger.target,
            "pai": trigger.pai,
            "trigger_type": trigger.trigger_type,
            "trigger_event_id": trigger.trigger_event_id,
            "round_event_id": trigger.round_event_id,
            "confirmed_at_event_id": confirmed,
            "confirmed": True,
            "provenance": provenance,
        }
        anchor = self._public_event_indices[confirmed]
        index = anchor + 1
        self.events.insert(index, canonical_bytes(marker).decode("utf-8"))
        self.history_positions.insert(index, self.history_positions[anchor])
        for public_id in range(confirmed + 1, len(self._public_event_indices)):
            self._public_event_indices[public_id] += 1
        self._recorded_passes.add(trigger.trigger_event_id)
        return True

    def _apply_self_pass(self, event: dict[str, Any]) -> None:
        if event.get("actor") != self.observer or event.get("confirmed") is not True:
            raise ValueError("A private Pass belongs only to its confirmed owner")
        source_id = event.get("trigger_event_id")
        trigger = self._pass_triggers.get(source_id) if type(source_id) is int else None
        if trigger is None or any(event.get(key) != getattr(trigger, key) for key in ("target", "pai", "trigger_type")):
            raise ValueError("Private Pass source facts differ from the observed trigger")
        confirmed = event.get("confirmed_at_event_id")
        if type(confirmed) is not int or self._pass_confirmations.get(trigger.trigger_event_id) != confirmed:
            raise ValueError("Private Pass cannot use a future or foreign confirmation")
        provenance = event.get("provenance")
        if not isinstance(provenance, str):
            raise ValueError("Private Pass requires an explicit provenance string")
        self.record_self_pass(trigger, provenance=provenance)

    def _update_pass_window(self, public: dict[str, Any]) -> None:
        typ = public["type"]
        current = self._pass_triggers.get(self._active_pass_trigger if self._active_pass_trigger is not None else -1)
        if current is not None:
            closed = False
            if typ == "tsumo":
                target = (current.target + 1) % 4 if current.trigger_type == "dahai" else current.target
                closed = public["actor"] == target
            elif typ in {"chi", "pon", "kan", "daiminkan"}:
                closed = (
                    current.trigger_type == "dahai"
                    and public["target"] == current.target
                    and public["pai"] == current.pai
                )
            elif typ == "hora":
                closed = public.get("target") == current.target and public.get("actor") != current.target
            elif typ == "ryukyoku":
                closed = current.trigger_type == "dahai"
            if closed:
                self._pass_confirmations[current.trigger_event_id] = self.event_id
                self._active_pass_trigger = None
        if typ in {"start_kyoku", "end_kyoku", "end_game"}:
            self._active_pass_trigger = None
        if typ == "start_kyoku":
            self._round_event_id = self.event_id
        if typ in {"dahai", "ankan", "kakan"}:
            pai = public["consumed"][0] if typ == "ankan" else public["pai"]
            trigger = SelfPassTrigger(self.observer, self.event_id, self._round_event_id, typ, public["actor"], pai)
            self._pass_triggers[self.event_id] = trigger
            self._active_pass_trigger = self.event_id

    def apply_event(self, event: dict[str, Any]) -> None:
        if event["type"] == "self_pass":
            self._apply_self_pass(event)
            return
        public = json.loads(self._native.apply_event_json(canonical_bytes(event).decode("utf-8")))
        self._refresh_state()
        self.event_id += 1
        self.logical_clock = self._event_clock.advance(public)
        self.history_positions.append(self.logical_clock)
        self._public_event_indices.append(len(self.events))
        self.events.append(canonical_bytes(public).decode("utf-8"))
        self._public_events.append(public)
        self._update_pass_window(public)

    def snapshot(self) -> PublicSnapshot:
        return PublicSnapshot(
            self.observer,
            self.event_id,
            self.initial_order,
            tuple(sorted(self.hand)),
            tuple(self.scores),
            tuple(tuple(x) for x in self.history),
            tuple(tuple(x) for x in self.rivers),
            tuple(tuple(x) for x in self.melds),
            tuple(self.dora),
            tuple(self.declared),
            tuple(self.accepted),
            self.remaining,
            self.last_actor,
            self.last_tile,
            self.oya,
            self.honba,
            self.riichi_sticks,
            self.have_start and self.have_round,
            self.have_round and self.own_hand_complete,
            tuple(self.events),
            None if self.pending is None else self.pending[1],
            round_wind=self.round_wind,
            public_hand_sizes=tuple(self.public_hand_sizes),
            draw_counts=tuple(self.draw_counts),
            riichi_discard_ordinals=tuple(self.riichi_ordinals),
            current_draw_actor=self.current_draw_actor,
            drawn_tile=self.drawn_tile,
            decision_phase=self.decision_phase,
            post_call_actor=self.post_call_actor,
            forced_tedashi=(self.observer == self.oya and not self.history[self.observer] and not self.interrupted),
            pending_kan_kind=None if self.pending is None else self.pending[0],
            pending_kan_tile=None if self.pending is None else self.pending[2][0],
            history_positions=tuple(self.history_positions),
            visibility_complete=self.visibility_complete,
            public_clock_version=PUBLIC_CLOCK_VERSION,
            _history_session=self._history_session,
        )

    def check_observation(self, observation: Any) -> None:
        if observation.player_id != self.observer:
            raise ValueError("Observation belongs to a different player")
        observed = Counter(_tile(convert.tid_to_mjai(t)) for t in observation.hand)
        if self.own_hand_complete and observed != Counter(self.hand):
            raise ValueError("RiichiEnv hand does not match cutoff observer ledger")


def project_observer(
    events: list[dict[str, Any]], seat: int, cutoff_event_id: int, initial_seat_order_player_ids: tuple[int, ...]
) -> PublicSnapshot:
    if not 0 <= cutoff_event_id < len(events):
        raise ValueError("cutoff_event_id outside event log")
    ledger = ObserverLedger(seat, initial_seat_order_player_ids)
    for event in events[: cutoff_event_id + 1]:
        ledger.apply_event(event)
    return ledger.snapshot()


def extract_rule_features(snapshot: PublicSnapshot) -> dict[str, Any]:
    """Preserve per-feature dependencies: history, physical counts and given dora."""
    if snapshot.remaining_draws is None:
        raise ValueError("Rule features require an initialized round snapshot")
    counts = snapshot.known_counts()
    dora = public_dora_types(t // 4 for t in snapshot.dora_indicators)
    flags = []
    for tile in range(34):
        histories = [
            {t // 4 for t in snapshot.discard_history[(snapshot.observer_player_id + relative) % 4]}
            if snapshot.history_complete
            else None
            for relative in range(1, 4)
        ]
        values = [genbutsu(tile, history) for history in histories]
        values.extend(full_suji_exclusion(tile, history) for history in histories)
        values.append(no_visible_possible_ryanmen_shape(tile, counts))
        values.append(tile in dora if snapshot.dora_indicators else None)
        flags.append(tuple(0 if value is None else 2 if value else 1 for value in values) + (0,) * 8)
    rank = current_rank_features(snapshot.scores, snapshot.initial_seat_order_player_ids, snapshot.observer_player_id)
    return {
        "hand_count_flags": tuple(flags),
        "own_current_rank": rank.rank,
        "gap_to_higher": rank.gap_to_higher,
        "gap_to_higher_present": rank.gap_to_higher_present,
        "gap_to_lower": rank.gap_to_lower,
        "gap_to_lower_present": rank.gap_to_lower_present,
    }
