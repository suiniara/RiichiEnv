from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from riichienv import convert

from .observation import ObserverLedger, PublicSnapshot, SelfPassTrigger

if TYPE_CHECKING:
    from .actions import MappedAction

LIVE_EVENT_VERSION = "native_rule_event_projection_v6"
_NATIVE_PASS_PROVENANCE = "native_committed_pass"
_DRAW_ALIASES = {
    "exhaustive_draw": "exhaustive_draw",
    "nagashimangan": "nagashi_mangan",
    "nagashi_mangan": "nagashi_mangan",
    "kyushu_kyuhai": "kyushukyuhai",
    "kyushukyuhai": "kyushukyuhai",
    "sufuurenta": "four_winds",
    "four_winds": "four_winds",
    "suucha_riichi": "four_riichi",
    "four_riichi": "four_riichi",
    "suukansansen": "four_kans",
    "four_kans": "four_kans",
}


def _points(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(type(item) is not int for item in value):
        raise ValueError("Native public payment must contain four integer seat values")
    return tuple(value)


class LiveEventProjection:
    """Simulation boundary. Private owner snapshots are revealed only after a legal public resolution.

    Agents receive one white-listed snapshot, never this aggregate simulation object.
    """

    def __init__(self):
        self.ledgers = [ObserverLedger(seat, (0, 1, 2, 3)) for seat in range(4)]
        self.events: list[dict[str, Any]] = []
        self.raw_cursor = 0
        self.round_instance = -1
        self.public_pot = 0
        self.public_scores: tuple[int, ...] | None = None
        self.ended = False
        self._record_serial = 0
        self._pending_horas: list[dict[str, Any]] = []
        self._pending_hora_record: int | None = None
        self._pending_pass_receipts: dict[SelfPassTrigger, None] = {}

    def _next_record(self) -> int:
        value = self._record_serial
        self._record_serial += 1
        return value

    @staticmethod
    def _ankan_key(event: dict[str, Any]) -> tuple[Any, ...]:
        consumed = event.get("consumed", ())
        if (
            event.get("type") != "ankan"
            or type(event.get("actor")) is not int
            or not 0 <= event["actor"] < 4
            or len(consumed) != 4
        ):
            raise ValueError("A declared ankan needs an actor and all four consumed tiles")
        tiles = tuple(sorted(convert.mjai_to_tid(value) for value in consumed))
        if len({value // 4 for value in tiles}) != 1:
            raise ValueError("Ankan declaration contains different tile types")
        return event["actor"], tiles

    def snapshot(self, seat: int) -> PublicSnapshot:
        if type(seat) is not int or not 0 <= seat < 4:
            raise ValueError("Invalid observer")
        if self._pending_horas:
            raise ValueError("Public win resolution is incomplete; no decision snapshot is available")
        return self.ledgers[seat].snapshot()

    def capture_self_passes(self, choices: Mapping[int, MappedAction]) -> tuple[SelfPassTrigger, ...]:
        """Side-effect-free receipts. Confirm only after atomic native submission succeeds."""
        return tuple(
            self.ledgers[seat].pass_trigger() for seat, choice in choices.items() if choice.spec.action_type == "PASS"
        )

    def _append(self, event: dict[str, Any], record: int) -> None:
        if event["type"] == "self_pass":
            raise ValueError("Private self Pass must never be broadcast as a public event")
        event = {**event, "source_record_index": record}
        for ledger in self.ledgers:
            ledger.apply_event(event)
        self.events.append(event)
        if event["type"] in {"start_kyoku", "reach_accepted", "hora", "ryukyoku", "end_game"}:
            self.public_scores = self.ledgers[0].scores
            self.public_pot = self.ledgers[0].riichi_sticks

    def _win_group(self, group: Sequence[dict[str, Any]], record: int) -> None:
        if self.public_scores is None or not group:
            raise ValueError("Win precedes a complete public round")
        actors = [event.get("actor") for event in group]
        if any(type(actor) is not int or not 0 <= actor < 4 for actor in actors) or len(set(actors)) != len(actors):
            raise ValueError("Duplicate/invalid native winners")
        targets = {event.get("target") for event in group}
        if len(targets) != 1:
            raise ValueError("Simultaneous win group has inconsistent public sources")
        total = tuple(sum(_points(event.get("deltas"))[seat] for event in group) for seat in range(4))
        if sum(total) != self.public_pot * 1000:
            raise ValueError("Native win payments do not match the public riichi pot")
        scores = tuple(score + delta for score, delta in zip(self.public_scores, total))
        normalized = []
        for raw in sorted(group, key=lambda value: (value["actor"] - value["target"]) % 4):
            actor, target = raw["actor"], raw["target"]
            view = self.snapshot(actor)
            if actor == target:
                if len(group) != 1 or view.current_draw_actor != actor or view.drawn_tile is None:
                    raise ValueError("Native tsumo lacks its visible owner draw")
                tile = view.drawn_tile
            else:
                source = view.pending_kan_actor if view.pending_kan_actor is not None else view.last_discard_actor
                tile = view.pending_kan_tile if view.pending_kan_actor is not None else view.last_discard_tile
                if source != target or tile is None:
                    raise ValueError("Native ron lacks its public discard/kan trigger")
            if raw.get("pai") is not None and convert.mjai_to_tid(raw["pai"]) // 4 != tile // 4:
                raise ValueError("Native winning tile conflicts with the visible trigger")
            hand = raw.get("public_hand")
            if not isinstance(hand, list) or any(not isinstance(item, str) or item == "?" for item in hand):
                raise ValueError("Native win lacks its engine-resolved public hand; rebuild the extension")
            if _points(raw.get("scores")) != scores:
                raise ValueError("Native win resolution scores disagree with the complete window")
            event = {
                "type": "hora",
                "actor": actor,
                "target": target,
                "pai": convert.tid_to_mjai(tile),
                "deltas": list(total),
                "scores": list(scores),
                "public_hand": list(hand),
                "visibility_complete": True,
                "public_reveal_provenance": "native_round_resolution_v1",
            }
            if view.riichi_accepted[actor]:
                markers = raw.get("ura_markers")
                if not isinstance(markers, list) or len(markers) != len(view.dora_indicators):
                    raise ValueError("Declared winner lacks actual public ura indicators")
                event["ura_markers"] = list(markers)
            normalized.append(event)
        for event in normalized:
            self._append(event, record)
        self.public_scores, self.public_pot = scores, 0

    def _draw(self, raw: dict[str, Any], record: int) -> None:
        raw_reason = raw.get("reason")
        reason = _DRAW_ALIASES.get(raw_reason) if isinstance(raw_reason, str) else None

        if reason is None or self.public_scores is None:
            raise ValueError("Unknown/illegal draw or incomplete public score state")
        delta = _points(raw.get("deltas"))
        if sum(delta):
            raise ValueError("A draw does not claim the public riichi pot")
        scores = _points(raw.get("scores"))
        if scores != tuple(score + change for score, change in zip(self.public_scores, delta)):
            raise ValueError("Native draw resolution breaks the public score chain")
        shown = raw.get("public_hands")
        if not isinstance(shown, list) or len(shown) != 4 or raw.get("visibility_complete") is not True:
            raise ValueError("Native draw lacks its engine-resolved public hands; rebuild the extension")
        if any(
            hand is not None and (not isinstance(hand, list) or any(not isinstance(t, str) or t == "?" for t in hand))
            for hand in shown
        ):
            raise ValueError("Invalid native public draw hands")
        actor = raw.get("actor")
        if reason == "kyushukyuhai" and (type(actor) is not int or not 0 <= actor < 4 or shown[actor] is None):
            raise ValueError("Native nine-terminals resolution lacks its declarer reveal")
        self._append(
            {
                "type": "ryukyoku",
                "reason": reason,
                **({"actor": actor} if reason == "kyushukyuhai" else {}),
                "deltas": list(delta),
                "scores": list(scores),
                "public_hands": shown,
                "visibility_complete": True,
                "public_reveal_provenance": "native_round_resolution_v1",
            },
            record,
        )
        self.public_scores = scores

    def observe_native(  # noqa: PLR0915
        self,
        raw_events: Sequence[dict[str, Any]],
        *,
        declared_ankan: dict[str, Any] | None = None,
        pending_kan_window: bool = False,
        confirmed_self_passes: Sequence[SelfPassTrigger] = (),
    ) -> None:
        if any(event.get("type") == "self_pass" for event in raw_events):
            raise ValueError("Observer-private Pass cannot enter the native public stream")
        if self.ended and (raw_events or declared_ankan is not None):
            raise ValueError("Events after completed match")
        # A receipt must come from a trigger already observed at submission.
        # Reject new invalid receipts before consuming events or queuing any.
        for trigger in confirmed_self_passes:
            if (
                not isinstance(trigger, SelfPassTrigger)
                or type(trigger.observer_player_id) is not int
                or not 0 <= trigger.observer_player_id < 4
            ):
                raise ValueError("Invalid confirmed private Pass receipt")
            self.ledgers[trigger.observer_player_id].self_pass_ready(trigger, provenance=_NATIVE_PASS_PROVENANCE)
        if declared_ankan is not None:
            # Compatibility validation only. The engine's rule_log must already
            # contain the declaration; Python never synthesizes engine events.
            key = self._ankan_key(declared_ankan)
            if not any(event.get("type") == "ankan" and self._ankan_key(event) == key for event in raw_events):
                raise ValueError("Native rule_log must include the ankan declaration; rebuild the extension")
        index = 0
        while index < len(raw_events):
            raw = copy.deepcopy(raw_events[index])
            record = self._next_record()
            kind = raw["type"]
            if self._pending_horas and kind != "hora":
                group = self._pending_horas
                group_record = self._pending_hora_record
                self._pending_horas, self._pending_hora_record = [], None
                assert group_record is not None
                self._win_group(group, group_record)
            if kind == "hora":
                if not self._pending_horas:
                    self._pending_hora_record = record
                self._pending_horas.append(raw)
                index += 1
                continue
            if kind == "start_kyoku":
                scores = _points(raw["scores"])
                if self.public_scores is not None and scores != self.public_scores:
                    raise ValueError("Native new-round public scores break the projected chain")
                self.public_scores = scores
                self.public_pot = raw.get("kyotaku", raw.get("kyoutaku", 0))
                self.round_instance += 1
            elif kind == "ryukyoku":
                self._draw(raw, record)
                index += 1
                continue
            elif kind == "end_game":
                if self.public_scores is None:
                    raise ValueError("Missing complete public final scores")
                delta = _points(raw.get("end_game_pot_deltas"))
                scores = _points(raw.get("scores"))
                if sum(delta) != self.public_pot * 1000 or scores != tuple(
                    score + change for score, change in zip(self.public_scores, delta)
                ):
                    raise ValueError("Native match-end resolution breaks the public score/pot chain")
                self.ended = True
            self._append(raw, record)
            index += 1
        self.raw_cursor += len(raw_events)
        # Public events are fully validated first. Markers can be inserted at an
        # earlier observed anchor, even if this callback has an incomplete win
        # group for a later window. An open window keeps its receipt across calls.
        self._pending_pass_receipts.update(dict.fromkeys(confirmed_self_passes))
        for trigger in tuple(self._pending_pass_receipts):
            ledger = self.ledgers[trigger.observer_player_id]
            if ledger.self_pass_ready(trigger, provenance=_NATIVE_PASS_PROVENANCE):
                ledger.record_self_pass(trigger, provenance=_NATIVE_PASS_PROVENANCE)
                # Both newly inserted and already-recorded receipts are complete.
                del self._pending_pass_receipts[trigger]
