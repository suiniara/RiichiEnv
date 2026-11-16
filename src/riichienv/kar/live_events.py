from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from riichienv import HandEvaluator, convert

from .observation import ObserverLedger, PublicSnapshot
from .payments import end_game_pot
from .settlement import _meld

LIVE_EVENT_VERSION = "native_public_resolution_projection_v2"
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
        self._delayed_ankan: tuple[Any, ...] | None = None

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
        return self.ledgers[seat].snapshot()

    def _append(self, event: dict[str, Any], record: int) -> None:
        event = {**event, "source_record_index": record}
        for ledger in self.ledgers:
            ledger.apply_event(event)
        self.events.append(event)

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
        for raw in group:
            actor, target = raw["actor"], raw["target"]
            view = self.snapshot(actor)
            if actor == target:
                if len(group) != 1 or view.current_draw_actor != actor or view.drawn_tile is None:
                    raise ValueError("Native tsumo lacks its visible owner draw")
                tile = view.drawn_tile
                hand = list(view.hand)
            else:
                source = view.pending_kan_actor if view.pending_kan_actor is not None else view.last_discard_actor
                tile = view.pending_kan_tile if view.pending_kan_actor is not None else view.last_discard_tile
                if source != target or tile is None:
                    raise ValueError("Native ron lacks its public discard/kan trigger")
                hand = [*view.hand, tile]
            if raw.get("pai") is not None and convert.mjai_to_tid(raw["pai"]) // 4 != tile // 4:
                raise ValueError("Native winning tile conflicts with the visible trigger")
            event = {
                "type": "hora",
                "actor": actor,
                "target": target,
                "pai": convert.tid_to_mjai(tile),
                "deltas": list(total),
                "scores": list(scores),
                "public_hand": [convert.tid_to_mjai(item) for item in hand],
                "visibility_complete": True,
                "public_reveal_provenance": "winner_prior_owner_snapshot_after_public_resolution",
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
        scores = tuple(score + change for score, change in zip(self.public_scores, delta))
        shown: list[list[str] | None] = [None for _ in range(4)]
        if reason in {"exhaustive_draw", "nagashi_mangan"}:
            # RiichiEnv's frozen exhaustive-draw mechanism automatically declares
            # structural tenpai and reveals those hands. Never infer it from deltas.
            for seat in range(4):
                view = self.snapshot(seat)
                evaluator = HandEvaluator(list(view.hand), [_meld(meld) for meld in view.melds[seat]])
                if evaluator.is_tenpai():
                    shown[seat] = [convert.tid_to_mjai(tile) for tile in view.hand]
        elif reason == "kyushukyuhai":
            actor = self.snapshot(0).current_draw_actor
            if actor is None:
                raise ValueError("Nine-terminals public reveal lacks a declaring actor")
            shown[actor] = [convert.tid_to_mjai(tile) for tile in self.snapshot(actor).hand]
        self._append(
            {
                "type": "ryukyoku",
                "reason": reason,
                "deltas": list(delta),
                "scores": list(scores),
                "public_hands": shown,
                "visibility_complete": True,
                "public_reveal_provenance": "frozen_native_public_draw_resolution",
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
    ) -> None:
        if self.ended and (raw_events or declared_ankan is not None):
            raise ValueError("Events after completed match")
        if declared_ankan is not None:
            key = self._ankan_key(declared_ankan)
            matching = any(event.get("type") == "ankan" and self._ankan_key(event) == key for event in raw_events)
            if not matching:
                if not pending_kan_window or self._delayed_ankan is not None:
                    raise ValueError("Missing native ankan log without a new validated robbery window")
                self._append(copy.deepcopy(declared_ankan), self._next_record())
                self._delayed_ankan = key
        index = 0
        while index < len(raw_events):
            raw = copy.deepcopy(raw_events[index])
            record = self._next_record()
            kind = raw["type"]
            if kind == "ankan" and self._delayed_ankan is not None:
                if self._ankan_key(raw) != self._delayed_ankan:
                    raise ValueError("Late native ankan acknowledgement is for another declaration")
                self._append({"type": "kan_commit", "actor": raw["actor"], "kan_type": "ankan"}, record)
                self._delayed_ankan = None
                index += 1
                continue
            if kind == "hora":
                end = index + 1
                while end < len(raw_events) and raw_events[end]["type"] == "hora":
                    end += 1
                self._win_group(raw_events[index:end], record)
                self._delayed_ankan = None
                index = end
                continue
            if kind in {"start_kyoku", "ryukyoku", "end_kyoku", "end_game"}:
                self._delayed_ankan = None
            if kind == "start_kyoku":
                scores = _points(raw["scores"])
                if self.public_scores is not None and scores != self.public_scores:
                    raise ValueError("Native new-round public scores break the projected chain")
                self.public_scores = scores
                self.public_pot = raw.get("kyotaku", raw.get("kyoutaku", 0))
                self.round_instance += 1
            elif kind == "reach_accepted":
                if self.public_scores is None:
                    raise ValueError("Riichi acceptance precedes public scores")
                actor = raw["actor"]
                self.public_scores = tuple(
                    score - (1000 if seat == actor else 0) for seat, score in enumerate(self.public_scores)
                )
                self.public_pot += 1
            elif kind == "ryukyoku":
                self._draw(raw, record)
                index += 1
                continue
            elif kind == "end_game":
                if self.public_scores is None:
                    raise ValueError("Missing complete public final scores")
                delta = end_game_pot(self.public_scores, self.public_pot, (0, 1, 2, 3))
                self.public_scores = tuple(score + change for score, change in zip(self.public_scores, delta))
                self.public_pot = 0
                raw.update(scores=list(self.public_scores), end_game_pot_deltas=list(delta))
                self.ended = True
            self._append(raw, record)
            index += 1
        self.raw_cursor += len(raw_events)
