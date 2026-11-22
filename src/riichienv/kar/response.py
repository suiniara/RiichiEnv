from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import MJSOUL_RULESET, Response, Ruleset


@dataclass(frozen=True)
class ResponseResolution:
    response_applicable: bool
    response_observed: bool
    response_target: int | None
    response_resolution_event_ids: tuple[int, ...] = ()
    error: str | None = None


_END_REASONS = {
    "exhaustive": Response.EXHAUSTIVE_DRAW,
    "exhaustive_draw": Response.EXHAUSTIVE_DRAW,
    "fanpai": Response.EXHAUSTIVE_DRAW,
    "four_winds": Response.ABORT_FOUR_WINDS,
    "suufon_renda": Response.ABORT_FOUR_WINDS,
    "sufuurenta": Response.ABORT_FOUR_WINDS,
    "four_riichi": Response.ABORT_FOUR_RIICHI,
    "suucha_riichi": Response.ABORT_FOUR_RIICHI,
    "four_kans": Response.ABORT_FOUR_KANS,
    "suukaikan": Response.ABORT_FOUR_KANS,
    "suukansansen": Response.ABORT_FOUR_KANS,
    "triple_ron": Response.ABORT_TRIPLE_RON,
    "sanchahou": Response.ABORT_TRIPLE_RON,
    "sanchaho": Response.ABORT_TRIPLE_RON,
    "nagashi_mangan": Response.NAGASHI_MANGAN,
    "nagashimangan": Response.NAGASHI_MANGAN,
}


def draw_response(reason: object) -> Response | None:
    return _END_REASONS.get(reason) if isinstance(reason, str) else None


def resolve_discard_response(  # noqa: PLR0911
    events: list[dict[str, Any]], trigger_event_id: int, ruleset: Ruleset = MJSOUL_RULESET
) -> ResponseResolution:
    if not 0 <= trigger_event_id < len(events):
        raise ValueError("Trigger outside log")
    trigger = events[trigger_event_id]
    if trigger["type"] != "dahai":
        return ResponseResolution(False, False, None)
    actor = trigger["actor"]
    winners: set[int] = set()
    resolution_ids: list[int] = []
    for index in range(trigger_event_id + 1, len(events)):
        event = events[index]
        typ = event["type"]
        if typ == "self_pass":
            continue
        if typ == "hora":
            if event["target"] != actor or event["actor"] == actor or event["actor"] in winners:
                return ResponseResolution(True, False, None, error="conflicting_or_duplicate_ron")
            winners.add(event["actor"])
            resolution_ids.append(index)
            continue
        if winners:
            if typ not in {"end_kyoku", "end_game"}:
                return ResponseResolution(True, False, None, error="missing_ron_resolution_boundary")
            mask = sum(1 << ((winner - actor) % 4 - 1) for winner in winners)
            if mask == 7 and ruleset.sanchaho_is_draw:
                return ResponseResolution(True, False, None, error="triple_ron_conflicts_with_ruleset")
            return ResponseResolution(True, True, 7 + mask, tuple(resolution_ids))
        if typ in {"reach_accepted", "dora"}:
            continue
        if typ in {"chi", "pon", "daiminkan", "kan"}:
            if event.get("target") != actor or event.get("pai") != trigger["pai"]:
                return ResponseResolution(True, False, None, error="call_does_not_match_trigger")
            relative = (event["actor"] - actor) % 4
            if relative == 0 or (typ == "chi" and relative != 1):
                return ResponseResolution(True, False, None, error="invalid_call_source")
            target = 1 if typ == "chi" else 1 + relative if typ == "pon" else 4 + relative
            return ResponseResolution(True, True, target, (index,))
        if typ == "tsumo":
            if event["actor"] != (actor + 1) % 4:
                return ResponseResolution(True, False, None, error="not_normal_next_player_draw")
            return ResponseResolution(True, True, int(Response.NEXT_DRAW), (index,))
        if typ == "ryukyoku":
            target = draw_response(event.get("reason"))
            if target is None:
                return ResponseResolution(True, False, None, error="unknown_draw_reason")
            if target == Response.ABORT_TRIPLE_RON and not ruleset.sanchaho_is_draw:
                return ResponseResolution(True, False, None, error="abort_triple_ron_disabled")
            return ResponseResolution(True, True, int(target), (index,))
        return ResponseResolution(True, False, None, error="incomplete_or_conflicting_response_window")
    return ResponseResolution(True, False, None, error="truncated_response_window")
