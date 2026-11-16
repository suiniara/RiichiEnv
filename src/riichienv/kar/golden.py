from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from riichienv import convert

from .actions import legal_candidates
from .completion import certify_completion, observed_raw_final_scores
from .contracts import MJSOUL_RULESET, content_hash, game_split
from .mjsoul import decode_record_bytes, load_protocol
from .observation import ObserverLedger, PublicSnapshot
from .response import resolve_discard_response
from .response_mask import public_rule_response_mask, validate_response_target
from .rule_features import (
    OTHER_WAITS_BY_RANK,
    RYANMEN_PAIRS_BY_RANK,
    current_rank_features,
    full_suji_exclusion,
    genbutsu,
    no_visible_possible_ryanmen_shape,
)
from .settlement import verify_settlements
from .tokenize import encode_prefix

GOLDEN_VERSION = "mjsoul_real_golden_v1"
_MAX_RAW_BYTES = 64 * 1024 * 1024


class _EvidenceError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("golden_registry.v1.json").read_text(encoding="utf-8"))


def _require(condition: bool, reason: str, status: str = "FAIL") -> None:
    if not condition:
        raise _EvidenceError(status, reason)


def _integer(value: Any, name: str, upper: int | None = None) -> int:
    _require(not isinstance(value, bool) and isinstance(value, int) and value >= 0, f"invalid_{name}")
    if upper is not None:
        _require(value < upper, f"invalid_{name}")
    return value


def _digest(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"invalid_{name}")
    _require(all(char in "0123456789abcdef" for char in value.lower()), f"invalid_{name}")
    return value.lower()


def _tile(value: str) -> int:
    return convert.mjai_to_tid(value)


def _snapshot(events: Sequence[Mapping[str, Any]], observer: int, cutoff: int) -> PublicSnapshot:
    _require(0 <= cutoff < len(events), "cutoff_outside_game")
    ledger = ObserverLedger(observer, (0, 1, 2, 3))
    for event in events[: cutoff + 1]:
        ledger.apply_event(dict(event))
    return ledger.snapshot()


def _significant(events: Sequence[Mapping[str, Any]], start: int) -> int | None:
    return next(
        (index for index in range(start, len(events)) if events[index]["type"] not in {"dora", "reach_accepted"}), None
    )


def _previous_discard(events: Sequence[Mapping[str, Any]], index: int) -> int | None:
    for candidate in range(index - 1, -1, -1):
        typ = events[candidate]["type"]
        if typ == "dahai":
            return candidate
        if typ not in {"reach", "reach_accepted", "dora"}:
            return None
    return None


def _decision_anchor(events: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    for index, event in enumerate(events):
        if event["type"] != "dahai":
            continue
        cutoff = index - 1
        if cutoff >= 0 and events[cutoff]["type"] == "reach" and events[cutoff]["actor"] == event["actor"]:
            cutoff -= 1
        if cutoff >= 1:
            return event["actor"], cutoff
    raise _EvidenceError("BLOCKED", "no_real_discard_decision_anchor")


def _native_candidates(events: Sequence[Mapping[str, Any]], observer: int, cutoff: int) -> tuple[int, ...]:
    env = MJSOUL_RULESET.create_env(seed=0)
    for event in events[: cutoff + 1]:
        env.apply_event(dict(event))
    snapshot = _snapshot(events, observer, cutoff)
    observation = env.get_observations([observer])[observer]
    return tuple(
        candidate.global_id
        for candidate in legal_candidates(
            observation,
            drawn_tile=env.drawn_tile,
            source_player=snapshot.last_discard_actor,
        )
    )


def _positive_result(  # noqa: PLR0911, PLR0915
    game: Any,
    case_id: str,
    observer: int,
    event_ids: tuple[int, ...],
) -> dict[str, Any]:
    events = game.events
    index = event_ids[0]
    event = events[index]
    if case_id in {"red_draw_normal_5p_hand", "normal_draw_red_5p_hand"}:
        _require(len(event_ids) == 1 and event["type"] == "tsumo" and event["actor"] == observer, "wrong_draw_locator")
        red = case_id == "red_draw_normal_5p_hand"
        _require(event["pai"] == ("5pr" if red else "5p"), "draw_tile_not_matching_case")
        before, after = _snapshot(events, observer, index - 1), _snapshot(events, observer, index)
        retained = 53 if red else 52
        _require(retained in before.hand, "opposite_five_not_in_dark_hand")
        _require(not after.forced_tedashi and not after.riichi_accepted[observer], "discard_variants_not_available")
        expected = (42, 60) if red else (40, 62)
        legal = _native_candidates(events, observer, index)
        _require(set(expected).issubset(legal), "required_five_variants_not_legal")
        return {
            "draw_tile": event["pai"],
            "required_non_riichi_ids": list(expected),
            "riichi_variants_only_if_structurally_legal": True,
        }
    if case_id in {"riichi_discard_ron", "multiple_ron", "triple_ron", "pon_then_discard", "exhaustive_last_discard"}:
        _require(event["type"] == "dahai", "case_does_not_start_at_discard")
        _require(observer == event["actor"], "wrong_discard_observer")
        resolution = resolve_discard_response(list(events), index)
        _require(resolution.response_observed and resolution.response_target is not None, "response_window_unresolved")
        target = resolution.response_target
        assert target is not None
        resolved = tuple(resolution.response_resolution_event_ids)
        if case_id == "riichi_discard_ron":
            _require(
                8 <= target <= 14
                and index > 0
                and events[index - 1]["type"] == "reach"
                and events[index - 1]["actor"] == event["actor"],
                "not_ron_on_riichi_declaration_discard",
            )
            _require(observer == event["actor"], "wrong_riichi_observer")
            before, pending = _snapshot(events, observer, index - 2), _snapshot(events, observer, index)
            _require(
                before.scores == pending.scores and before.riichi_sticks == pending.riichi_sticks,
                "riichi_charged_before_resolution",
            )
            expected_ids = (index, *resolved)
        elif case_id in {"multiple_ron", "triple_ron"}:
            _require(8 <= target <= 14, "not_discard_ron")
            mask = target - 7
            _require(mask == 7 if case_id == "triple_ron" else mask.bit_count() >= 2, "insufficient_ron_winners")
            expected_ids = (index, *resolved)
        elif case_id == "pon_then_discard":
            _require(target in {2, 3, 4} and len(resolved) == 1, "not_accepted_pon")
            call = events[resolved[0]]
            following = _significant(events, resolved[0] + 1)
            _require(
                following is not None
                and events[following]["type"] == "dahai"
                and events[following]["actor"] == call["actor"],
                "pon_not_followed_by_callers_discard",
            )
            expected_ids = (index, resolved[0], following)
        else:
            _require(target in {15, 20}, "last_discard_not_direct_draw_settlement")
            snapshot = _snapshot(events, event["actor"], index)
            _require(snapshot.remaining_draws == 0, "live_wall_not_exhausted")
            expected_ids = (index, *resolved)
        _require(event_ids == expected_ids, "resolution_locator_mismatch")
        return {
            "response_target": target,
            "resolution_event_ids": list(resolved),
            "riichi_deposit_not_charged_early": case_id == "riichi_discard_ron",
        }
    if case_id == "daiminkan_rinshan_tsumo":
        _require(event["type"] in {"daiminkan", "kan"}, "not_daiminkan")
        draw = _significant(events, index + 1)
        win = _significant(events, draw + 1) if draw is not None else None
        _require(draw is not None and win is not None, "rinshan_sequence_truncated")
        _require(
            events[draw]["type"] == "tsumo" and events[draw]["actor"] == event["actor"], "not_callers_replacement_draw"
        )
        _require(
            events[win]["type"] == "hora"
            and events[win]["actor"] == event["actor"]
            and events[win]["target"] == event["actor"],
            "not_rinshan_self_draw_win",
        )
        discard = _previous_discard(events, index)
        _require(discard is not None, "missing_trigger_discard")
        assert discard is not None
        response = resolve_discard_response(list(events), discard)
        _require(response.response_target in {5, 6, 7}, "original_discard_not_daiminkan_response")
        _require(event_ids == (index, draw, win), "rinshan_locator_mismatch")
        return {"original_discard": discard, "response_target": response.response_target, "rinshan_win": win}
    if case_id in {"kakan_chankan", "kokushi_ankan_ron", "kakan_commit"}:
        expected_kind = "ankan" if case_id == "kokushi_ankan_ron" else "kakan"
        _require(event["type"] == expected_kind, "wrong_kan_declaration")
        next_index = index + 1
        while next_index < len(events) and events[next_index]["type"] == "reach_accepted":
            next_index += 1
        if next_index >= len(events):
            next_index = None
        if case_id == "kakan_commit":
            next_index = index + 1
            while next_index < len(events) and events[next_index]["type"] == "reach_accepted":
                next_index += 1
            _require(
                next_index < len(events)
                and (
                    events[next_index]["type"] == "dora"
                    or (events[next_index]["type"] == "tsumo" and events[next_index]["actor"] == event["actor"])
                ),
                "no_kan_commit_evidence",
            )
            before = _snapshot(events, observer, index)
            after = _snapshot(events, observer, next_index)
            actor, base = event["actor"], _tile(event["pai"]) // 4
            old = [
                slot
                for slot, meld in enumerate(before.melds[actor])
                if meld.kind == "pon" and meld.tiles[0] // 4 == base
            ]
            _require(
                len(old) == 1
                and after.melds[actor][old[0]].kind == "kakan"
                and len(before.melds[actor]) == len(after.melds[actor]),
                "kakan_changed_wrong_meld_slot",
            )
            _require(event_ids == (index, next_index), "commit_locator_mismatch")
            return {"original_meld_slot": old[0], "meld_count": len(after.melds[actor]), "added_physical_tiles": 1}
        _require(
            next_index is not None
            and events[next_index]["type"] == "hora"
            and events[next_index]["target"] == event["actor"]
            and events[next_index]["actor"] != event["actor"],
            "not_chankan",
        )
        _require(event_ids == (index, next_index), "chankan_locator_mismatch")
        before = _snapshot(events, observer, index)
        actor = event["actor"]
        if expected_kind == "kakan":
            base = _tile(event["pai"]) // 4
            _require(
                any(m.kind == "pon" and m.tiles[0] // 4 == base for m in before.melds[actor]),
                "kakan_committed_before_rob_window",
            )
            _require(
                not any(m.kind == "kakan" and m.tiles[0] // 4 == base for m in before.melds[actor]),
                "kakan_already_committed",
            )
        return {"declaration_kind": expected_kind, "winner": events[next_index]["actor"], "committed_before_win": False}
    if case_id == "opponent_ankan_red_hidden":
        _require(event["type"] == "ankan" and event["actor"] != observer, "not_opponents_ankan")
        _require(any(value in {"5mr", "5pr", "5sr"} for value in event["consumed"]), "source_ankan_has_no_red_five")
        first = _snapshot(events, observer, index)
        changed = copy.deepcopy(list(events))
        changed[index]["consumed"] = [value.removesuffix("r") for value in event["consumed"]]
        second = _snapshot(changed, observer, index)
        _require(first.to_bytes() == second.to_bytes(), "opponent_closed_kan_red_identity_leaked")
        _require(event_ids == (index,), "ankan_locator_mismatch")
        return {"opponent_red_composition_hidden": True, "base34": _tile(event["consumed"][0]) // 4}
    if case_id in {"four_winds", "four_riichi", "four_kans", "nagashi_mangan"}:
        expected = {"four_winds": 16, "four_riichi": 17, "four_kans": 18, "nagashi_mangan": 20}[case_id]
        _require(event["type"] == "ryukyoku" and event_ids == (index,), "wrong_draw_locator")
        from .response import draw_response  # noqa: PLC0415

        _require(draw_response(event.get("reason")) == expected, "wrong_draw_category")
        return {"response_target": expected, "recorded_reason": event["reason"]}
    raise _EvidenceError("FAIL", "unknown_real_positive_case")


def _derived_result(  # noqa: PLR0911, PLR0915
    game: Any,
    case_id: str,
    observer: int,
    event_ids: tuple[int, ...],
) -> dict[str, Any]:
    events = game.events
    cutoff = event_ids[0]
    baseline = _snapshot(events, observer, cutoff)
    _require(
        baseline.history_complete and baseline.physical_ledger_complete,
        "real_baseline_observation_incomplete",
        "BLOCKED",
    )
    if case_id == "four_observers_same_split":
        _require(len(event_ids) == 1, "wrong_derived_locator")
        for seat in range(4):
            snapshot = _snapshot(events, seat, cutoff)
            for serialized in snapshot.events:
                event = json.loads(serialized)
                if event["type"] == "start_kyoku":
                    _require(
                        all(hand == ["?"] * 13 for player, hand in enumerate(event["tehais"]) if player != seat),
                        "other_initial_hand_visible",
                    )
                if event["type"] == "tsumo" and event["actor"] != seat:
                    _require(event["pai"] == "?", "other_draw_visible")
        splits = [game_split(game.game_id, "golden-contract-v1") for _ in range(4)]
        _require(len(set(splits)) == 1, "observer_views_cross_split")
        return {"observer_count": 4, "one_game_split": splits[0], "opponent_private_tiles_masked": True}
    if case_id in {"future_hidden_ura_prefix_invariance", "candidate_order_invariance", "public_mask_public_only"}:
        candidates = _native_candidates(events, observer, cutoff)
        _require(bool(candidates), "anchor_has_no_complete_legal_candidates", "BLOCKED")
        prefix = encode_prefix(baseline, candidates)
        if case_id == "candidate_order_invariance":
            other = encode_prefix(baseline, tuple(reversed(candidates)) + candidates[:1])
            _require(prefix.to_bytes() == other.to_bytes(), "candidate_order_changes_prefix")
            return {
                "prefix_sha256": hashlib.sha256(prefix.to_bytes()).hexdigest(),
                "candidate_count": len(candidates),
                "invariant": True,
            }
        if case_id == "future_hidden_ura_prefix_invariance":
            changed = copy.deepcopy(list(events))
            for event in changed[: cutoff + 1]:
                if event["type"] == "start_kyoku":
                    for seat in range(4):
                        if seat != observer:
                            event["tehais"][seat] = ["C"] * 13
                    event["hidden_wall"] = "never-a-model-input"
                    event["ura_markers"] = ["5pr"] * 5
                if event["type"] == "tsumo" and event["actor"] != observer:
                    event["pai"] = "5pr"
            for event in changed[cutoff + 1 :]:
                event.update(final_rank=4, future_response_target=19, hidden_wall="future-only")
            alternate = encode_prefix(_snapshot(changed, observer, cutoff), candidates)
            _require(prefix.to_bytes() == alternate.to_bytes(), "hidden_or_future_data_changes_prefix")
            return {
                "prefix_sha256": hashlib.sha256(prefix.to_bytes()).hexdigest(),
                "invariant": True,
                "mutations": ["opponent_initial_hands", "opponent_draws", "unrevealed_ura", "future_labels"],
            }
        discard = next((candidate for candidate in candidates if candidate < 148), None)
        _require(discard is not None, "anchor_has_no_discard", "BLOCKED")
        assert discard is not None
        allowed = public_rule_response_mask(baseline, discard)
        _require(len(allowed) == 21 and any(allowed) and not allowed[19], "invalid_public_mask")
        parameters = inspect.signature(public_rule_response_mask).parameters
        _require(
            set(parameters) == {"snapshot", "candidate_discard", "ruleset"}, "public_mask_accepts_forbidden_inputs"
        )
        row = (
            resolve_discard_response(list(events), cutoff + 1)
            if cutoff + 1 < len(events) and events[cutoff + 1]["type"] == "dahai"
            else None
        )
        if row is not None and row.response_observed:
            assert row.response_target is not None
            validate_response_target(allowed, row.response_target)
        return {"allowed": list(allowed), "forbidden_input_parameters_absent": True}
    if case_id == "missing_response_not_next_draw":
        _require(events[cutoff]["type"] == "dahai", "anchor_not_discard")
        response = resolve_discard_response(list(events[: cutoff + 1]), cutoff)
        _require(
            response.response_applicable and not response.response_observed and response.response_target is None,
            "truncated_response_fabricated",
        )
        return {"applicable": True, "observed": False, "target": None, "mutation": "truncate_after_trigger"}
    if case_id == "non_discard_not_applicable":
        _require(
            events[cutoff]["type"] in {"chi", "pon", "daiminkan", "kan", "ankan", "kakan", "hora"},
            "anchor_not_action_family",
        )
        result = resolve_discard_response(list(events), cutoff)
        _require(not result.response_applicable and result.response_target is None, "non_discard_has_fake_response")
        return {"applicable": False, "target": None, "action_type": events[cutoff]["type"]}
    if case_id in {"initial_tie_and_oya_invariance", "unknown_zero_presence"}:
        _require(baseline.scores == (25000,) * 4, "anchor_not_initial_tie")
        results = [
            current_rank_features(baseline.scores, baseline.initial_seat_order_player_ids, seat) for seat in range(4)
        ]
        _require([result.rank for result in results] == [1, 2, 3, 4], "wrong_initial_tie_order")
        _require(
            not results[0].gap_to_higher_present and results[0].gap_to_lower_present and results[0].gap_to_lower == 0,
            "zero_gap_and_absence_confused",
        )
        _require(
            not results[3].gap_to_lower_present and results[3].gap_to_higher_present and results[3].gap_to_higher == 0,
            "zero_gap_and_absence_confused",
        )
        if case_id == "initial_tie_and_oya_invariance":
            altered = copy.deepcopy(list(events))
            for event in altered[: cutoff + 1]:
                if event["type"] == "start_kyoku":
                    event["oya"] = (event["oya"] + 1) % 4
            other = _snapshot(altered, observer, cutoff)
            rank = current_rank_features(other.scores, other.initial_seat_order_player_ids, observer)
            _require(rank == results[observer], "dealer_rotation_changes_tie_rank")
        return {
            "ranks": [result.rank for result in results],
            "real_zero_present": True,
            "missing_neighbor_absent": True,
        }
    if case_id == "called_tile_counted_once":
        event = events[cutoff]
        _require(
            event["type"] in {"chi", "pon", "daiminkan", "kan"} and event["actor"] == observer,
            "anchor_not_observers_call",
        )
        before, after = _snapshot(events, observer, cutoff - 1), baseline
        target, base = event["target"], _tile(event["pai"]) // 4
        _require(before.known_counts() == after.known_counts(), "called_tile_double_counted")
        _require(
            len(before.current_rivers[target]) == len(after.current_rivers[target]) + 1,
            "called_tile_not_removed_from_river",
        )
        _require(
            genbutsu(base, {tile // 4 for tile in after.discard_history[target]}) is True,
            "called_discard_lost_from_history",
        )
        return {"called_base34": base, "known_physical_counts_unchanged": True, "discard_history_retained": True}
    if case_id == "full_suji_both_sides":
        checked = []
        for suit in range(3):
            for rank, waits in ((4, (1, 7)), (5, (2, 8)), (6, (3, 9))):
                target = suit * 9 + rank - 1
                left, right = (suit * 9 + wait - 1 for wait in waits)
                values = [
                    full_suji_exclusion(target, {left}),
                    full_suji_exclusion(target, {right}),
                    full_suji_exclusion(target, {left, right}),
                ]
                _require(values == [False, False, True], "half_suji_misclassified_as_full")
                checked.append({"target_base34": target, "results": values})
        return {"vectors": checked, "mutations": "derived_visible_discard_sets_not_original_positive_events"}
    if case_id == "ryanmen_edge_exclusions":
        _require(
            RYANMEN_PAIRS_BY_RANK[3] == ((4, 5),) and RYANMEN_PAIRS_BY_RANK[7] == ((5, 6),), "penchan_in_ryanmen_table"
        )
        _require(OTHER_WAITS_BY_RANK[3] == (6,) and OTHER_WAITS_BY_RANK[7] == (4,), "incorrect_other_wait_table")
        baseline_counts = baseline.known_counts()
        _require(baseline_counts is not None, "real_baseline_visible_counts_incomplete", "BLOCKED")
        assert baseline_counts is not None
        for suit in range(3):
            counts = list(baseline_counts)
            counts[suit * 9 + 3] = 4
            _require(no_visible_possible_ryanmen_shape(suit * 9 + 2, counts) is True, "target_three_retains_penchan")
            counts = list(baseline_counts)
            counts[suit * 9 + 4] = 4
            _require(no_visible_possible_ryanmen_shape(suit * 9 + 6, counts) is True, "target_seven_retains_penchan")
        return {
            "target_3_pairs": [[4, 5]],
            "target_7_pairs": [[5, 6]],
            "suits_checked": 3,
            "mutations": "derived_visible_count_vectors_not_original_positive_events",
        }
    raise _EvidenceError("FAIL", "unknown_derived_case")


def _result(game: Any, case_id: str, observer: int, ids: tuple[int, ...]) -> dict[str, Any]:
    registry = _registry()
    if case_id in registry["required_real_positive"]:
        return _positive_result(game, case_id, observer, ids)
    if case_id in registry["required_derived_from_real"]:
        return _derived_result(game, case_id, observer, ids)
    raise _EvidenceError("FAIL", "case_not_in_frozen_registry")


def _candidate(game: Any, case_id: str, observer: int, ids: tuple[int, ...]) -> dict[str, Any]:
    expected = _result(game, case_id, observer, ids)
    return {
        "status": "CANDIDATE",
        "case_id": case_id,
        "game_id": game.game_id,
        "source_sha256": game.source_sha256,
        "protocol_sha256": game.protocol_sha256,
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "observer": observer,
        "event_ids": list(ids),
        "expected": expected,
        "evidence_type": "real_positive" if case_id in _registry()["required_real_positive"] else "derived_from_real",
        "source_role": "golden_only",
        "source_provenance": "unverified",
        "baseline_event_stream_sha256": content_hash(game.events),
    }


def find_golden_candidates(game: Any) -> list[dict[str, Any]]:  # noqa: PLR0915
    events = game.events
    found: dict[str, dict[str, Any]] = {}

    def add(case: str, observer: int, ids: tuple[int, ...]) -> None:
        if case in found:
            return
        try:
            found[case] = _candidate(game, case, observer, ids)
        except (ValueError, KeyError, IndexError):
            return

    for index, event in enumerate(events):
        typ = event["type"]
        if typ == "tsumo" and event.get("pai") in {"5p", "5pr"}:
            add(
                "red_draw_normal_5p_hand" if event["pai"] == "5pr" else "normal_draw_red_5p_hand",
                event["actor"],
                (index,),
            )
        elif typ == "dahai":
            response = resolve_discard_response(list(events), index)
            if response.response_observed and response.response_target is not None:
                target, resolved = response.response_target, tuple(response.response_resolution_event_ids)
                if 8 <= target <= 14:
                    if index > 0 and events[index - 1]["type"] == "reach":
                        add("riichi_discard_ron", event["actor"], (index, *resolved))
                    if (target - 7).bit_count() >= 2:
                        add("multiple_ron", event["actor"], (index, *resolved))
                    if target == 14:
                        add("triple_ron", event["actor"], (index, *resolved))
                if target in {2, 3, 4}:
                    following = _significant(events, resolved[0] + 1)
                    if following is not None:
                        add("pon_then_discard", event["actor"], (index, resolved[0], following))
                if target in {15, 20}:
                    add("exhaustive_last_discard", event["actor"], (index, *resolved))
            add("missing_response_not_next_draw", event["actor"], (index,))
        elif typ in {"daiminkan", "kan"}:
            draw = _significant(events, index + 1)
            win = _significant(events, draw + 1) if draw is not None else None
            if draw is not None and win is not None:
                add("daiminkan_rinshan_tsumo", event["actor"], (index, draw, win))
        elif typ in {"ankan", "kakan"}:
            observer = (event["actor"] + 1) % 4
            if typ == "ankan":
                add("opponent_ankan_red_hidden", observer, (index,))
            following = _significant(events, index + 1)
            if following is not None and events[following]["type"] == "hora":
                add("kokushi_ankan_ron" if typ == "ankan" else "kakan_chankan", observer, (index, following))
            if typ == "kakan":
                commit = index + 1
                while commit < len(events) and events[commit]["type"] == "reach_accepted":
                    commit += 1
                if commit < len(events):
                    add("kakan_commit", observer, (index, commit))
        elif typ == "ryukyoku":
            reason = event.get("reason")
            if reason in {"four_winds", "four_riichi", "four_kans", "nagashi_mangan"}:
                add(reason, 0, (index,))
        if typ in {"chi", "pon", "daiminkan", "kan"}:
            add("called_tile_counted_once", event["actor"], (index,))
        if typ in {"chi", "pon", "daiminkan", "kan", "ankan", "kakan", "hora"}:
            add("non_discard_not_applicable", event["actor"], (index,))
    try:
        observer, cutoff = _decision_anchor(events)
    except _EvidenceError:
        return list(found.values())
    for case in _registry()["required_derived_from_real"]:
        if case not in {"missing_response_not_next_draw", "non_discard_not_applicable", "called_tile_counted_once"}:
            add(case, observer, (cutoff,))
    return list(found.values())


def _protocol(path: str, digest: str):
    source = Path(path)
    _require(hashlib.sha256(source.read_bytes()).hexdigest() == digest, "protocol_file_hash_mismatch")
    return load_protocol(source)


def _source_game(game: Any, evidence: Mapping[str, Any]) -> Any:
    _require(evidence.get("source_provenance") == "real_raw_record", "real_raw_origin_not_declared", "BLOCKED")
    _require(evidence.get("source_role") == "golden_only", "golden_source_cannot_be_training_member")
    _require(
        not evidence.get("synthetic")
        and not str(getattr(game, "metadata", {}).get("record_provenance", "")).startswith("synthetic"),
        "synthetic_fixture_cannot_satisfy_real_coverage",
        "BLOCKED",
    )
    _require(evidence.get("game_id") == game.game_id, "game_id_mismatch")
    source_hash = _digest(evidence.get("source_sha256"), "source_sha256")
    protocol_hash = _digest(evidence.get("protocol_sha256"), "protocol_sha256")
    _require(source_hash == game.source_sha256 and protocol_hash == game.protocol_sha256, "source_identity_mismatch")
    _require(evidence.get("ruleset_hash") == MJSOUL_RULESET.ruleset_hash, "ruleset_hash_mismatch")
    _require(
        bool(evidence.get("source_path")) and bool(evidence.get("proto_path")),
        "raw_source_or_proto_path_missing",
        "BLOCKED",
    )
    source = Path(evidence["source_path"])
    proto = Path(evidence["proto_path"])
    _require(source.is_file() and proto.is_file(), "raw_source_or_proto_file_missing", "BLOCKED")
    _require(source.stat().st_size <= _MAX_RAW_BYTES, "raw_source_size_limit_exceeded")
    payload = source.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == source_hash, "raw_source_file_hash_mismatch")
    protocol = _protocol(str(proto.resolve()), protocol_hash)
    decoded = decode_record_bytes(payload, protocol, container=evidence.get("container", "majsoul_download"))
    _require(
        decoded.game_id == game.game_id and decoded.source_sha256 == source_hash, "decoded_source_identity_mismatch"
    )
    _require(content_hash(decoded.events) == content_hash(game.events), "normalized_event_stream_mismatch")
    records = [(record.index, record.name, record.source_sha256) for record in game.records]
    actual = [(record.index, record.name, record.source_sha256) for record in decoded.records]
    _require(records == actual, "raw_server_record_identity_mismatch")
    _require(
        evidence.get("baseline_event_stream_sha256") == content_hash(decoded.events), "baseline_event_hash_mismatch"
    )
    return decoded


def _replay_evidence(game: Any, replay: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(isinstance(replay, Mapping), "native_replay_receipt_missing", "BLOCKED")
    _require(replay.get("errors") == [], "native_replay_errors_or_missing_evidence")
    _require(replay.get("events_applied") == len(game.events), "incomplete_native_replay", "BLOCKED")
    discards = sum(event["type"] == "dahai" for event in game.events)
    _require(
        replay.get("discard_decisions_checked") == discards and replay.get("complete_response_windows") == discards,
        "incomplete_action_or_response_validation",
        "BLOCKED",
    )
    calculated = verify_settlements(game.events, observed_final_scores=observed_raw_final_scores(game))
    _require(calculated["status"] == "PASS", "independent_settlement_not_verified", calculated["status"])
    supplied = replay.get("independent_settlement")
    _require(
        isinstance(supplied, Mapping) and supplied.get("status") == "PASS", "settlement_receipt_missing", "BLOCKED"
    )
    assert isinstance(supplied, Mapping)
    keys = (
        "version",
        "ruleset_hash",
        "settlements",
        "prefix_score_checks",
        "errors",
        "computed_final_scores",
        "computed_riichi_sticks",
        "explicit_end_game",
        "independent_rule_requires_match_end",
        "end_game_pot_deltas",
        "server_han_fu_fans_payment_labels_used_as_calculation_inputs",
        "native_apply_event_scores_used_as_calculation_inputs",
    )
    _require(
        content_hash({key: supplied.get(key) for key in keys})
        == content_hash({key: calculated.get(key) for key in keys}),
        "independent_settlement_receipt_conflict",
    )
    compact = {
        key: copy.deepcopy(replay.get(key))
        for key in (
            "events_total",
            "events_applied",
            "discard_decisions_checked",
            "complete_response_windows",
            "errors",
            "independent_settlement",
        )
    }
    compact["events_total"] = len(game.events)
    completion = certify_completion(game, compact)
    _require(completion["status"] == "PASS", completion.get("reason", "completion_not_verified"), completion["status"])
    attached = replay.get("completion")
    if isinstance(attached, Mapping) and attached.get("status") == "PASS":
        _require(
            attached.get("certificate_sha256") == completion["certificate_sha256"],
            "attached_completion_certificate_conflict",
        )
    return compact, completion


def verify_golden_evidence(game: Any, replay_result: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    case_id = evidence.get("case_id") if isinstance(evidence, Mapping) else None
    try:
        _require(isinstance(evidence, Mapping), "malformed_golden_evidence")
        _require(isinstance(case_id, str), "case_id_missing")
        assert isinstance(case_id, str)
        registry = _registry()
        expected_type = "real_positive" if case_id in registry["required_real_positive"] else "derived_from_real"
        _require(
            case_id in registry["required_real_positive"] + registry["required_derived_from_real"], "unknown_case_id"
        )
        _require(evidence.get("evidence_type") == expected_type, "case_evidence_type_mismatch")
        observer = _integer(evidence.get("observer"), "observer", 4)
        supplied_ids = evidence.get("event_ids")
        _require(isinstance(supplied_ids, (list, tuple)) and bool(supplied_ids), "event_ids_missing")
        assert isinstance(supplied_ids, (list, tuple))
        ids = tuple(_integer(index, "event_id", len(game.events)) for index in supplied_ids)
        _require(tuple(sorted(set(ids))) == ids, "noncanonical_event_ids")
        decoded = _source_game(game, evidence)
        replay, completion = _replay_evidence(decoded, replay_result)
        expected = _result(decoded, case_id, observer, ids)
        _require(content_hash(evidence.get("expected")) == content_hash(expected), "expected_output_mismatch")
        if case_id == "kokushi_ankan_ron":
            win_index = ids[-1]
            rows = replay["independent_settlement"]["settlements"]
            winner = decoded.events[win_index]["actor"]
            derived = [
                win
                for row in rows
                if win_index in row.get("event_ids", [])
                for win in row.get("derived_wins", [])
                if win["player_id"] == winner
            ]
            _require(
                len(derived) == 1 and bool({42, 49}.intersection(derived[0]["computed_yaku"])),
                "robbed_ankan_not_independently_kokushi",
            )
        result = {
            "status": "PASS",
            "verified": True,
            "verification_version": GOLDEN_VERSION,
            "case_id": case_id,
            "evidence_type": expected_type,
            "source_role": "golden_only",
            "source_provenance": "real_raw_record",
            "game_id": decoded.game_id,
            "source_sha256": decoded.source_sha256,
            "protocol_sha256": decoded.protocol_sha256,
            "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
            "observer": observer,
            "event_ids": list(ids),
            "expected": expected,
            "source_path": str(Path(evidence["source_path"]).resolve()),
            "proto_path": str(Path(evidence["proto_path"]).resolve()),
            "container": evidence.get("container", "majsoul_download"),
            "baseline_event_stream_sha256": content_hash(decoded.events),
            "completion_certificate_sha256": completion["certificate_sha256"],
            "replay_evidence": replay,
            "authenticity_scope": "user_declared_local_raw_origin_byte_bound_not_provider_authenticated",
        }
        return {**result, "evidence_sha256": content_hash(result)}
    except _EvidenceError as exc:
        return {"status": exc.status, "verified": False, "case_id": case_id, "reason": str(exc)}
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        return {
            "status": "FAIL",
            "verified": False,
            "case_id": case_id,
            "reason": "malformed_or_inconsistent_golden_evidence",
            "detail": str(exc),
        }


def _rule_exclusions() -> list[dict[str, Any]]:
    if MJSOUL_RULESET.sanchaho_is_draw:
        return []
    return [
        {
            "case_id": "abort_triple_ron",
            "status": "NOT_APPLICABLE",
            "reason": "frozen_sanchaho_is_draw_false",
            "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
            "disabled_response_id": 19,
            "retained_response_id": 14,
        }
    ]


def golden_coverage(evidences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:  # noqa: PLR0915
    registry = _registry()
    required = registry["required_real_positive"] + registry["required_derived_from_real"]
    verified: dict[str, dict[str, Any]] = {}
    failed = []
    blocked = []
    seen: set[str] = set()
    replay_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for evidence in evidences:
        if not isinstance(evidence, Mapping):
            failed.append({"case_id": None, "status": "FAIL", "reason": "malformed_golden_record"})
            continue
        case_id = evidence.get("case_id")
        if not isinstance(case_id, str) or case_id not in required:
            failed.append({"case_id": case_id, "reason": "unknown_case_id"})
            continue
        if evidence.get("status") != "PASS" or evidence.get("verified") is not True:
            blocked.append({"case_id": case_id, "reason": "unverified_candidate_does_not_count"})
            continue
        payload = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        if evidence.get("verification_version") != GOLDEN_VERSION or evidence.get("evidence_sha256") != content_hash(
            payload
        ):
            failed.append({"case_id": case_id, "reason": "golden_record_integrity_mismatch"})
            continue
        digest = evidence["evidence_sha256"]
        if digest in seen:
            continue
        seen.add(digest)
        try:
            source = Path(evidence["source_path"])
            _require(source.is_file(), "raw_source_file_missing", "BLOCKED")
            _require(source.stat().st_size <= _MAX_RAW_BYTES, "raw_source_size_limit_exceeded")
            protocol = _protocol(str(Path(evidence["proto_path"]).resolve()), evidence["protocol_sha256"])
            game = decode_record_bytes(
                source.read_bytes(), protocol, container=evidence.get("container", "majsoul_download")
            )
            # Saved counters and a self-hash are not native-legality evidence.
            # Re-run the actual result-only replay, once per raw/protocol pair.
            key = (game.source_sha256, game.protocol_sha256)
            if key not in replay_cache:
                from .audit import replay_game  # noqa: PLC0415

                replay_cache[key] = replay_game(game)
            current_replay = replay_cache[key]
            _require(
                current_replay["status"] == "PASS", "current_full_native_replay_not_verified", current_replay["status"]
            )
            result = verify_golden_evidence(game, current_replay, evidence)
        except _EvidenceError as exc:
            result = {"case_id": case_id, "status": exc.status, "reason": str(exc)}
        except (KeyError, ValueError, TypeError, OSError) as exc:
            result = {"case_id": case_id, "status": "FAIL", "reason": str(exc)}
        if result["status"] == "PASS":
            verified[case_id] = result
        elif result["status"] == "FAIL":
            failed.append(result)
        else:
            blocked.append(result)
    missing_real = [case for case in registry["required_real_positive"] if case not in verified]
    missing_derived = [case for case in registry["required_derived_from_real"] if case not in verified]
    status = "FAIL" if failed else "BLOCKED" if missing_real or missing_derived else "PASS"
    report = {
        "status": status,
        "registry_version": GOLDEN_VERSION,
        "registry_sha256": content_hash(registry),
        "schema_id": registry["schema_id"],
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "required_count": len(required),
        "verified_count": len(verified),
        "verified_cases": sorted(verified),
        "missing_real_positive": missing_real,
        "missing_derived_from_real": missing_derived,
        "failed": failed,
        "blocked_evidence": blocked,
        "not_applicable": _rule_exclusions(),
        "golden_game_ids": sorted({item["game_id"] for item in verified.values()}),
        "training_exclusion_required": True,
        "scope": "source_bound_case_coverage_not_training_cohort_or_full_disk_audit",
        "source_authenticity_is_provider_signed": False,
    }
    return {**report, "coverage_sha256": content_hash(report)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Revalidate and combine source-bound real Golden evidence")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = []
    inputs = []
    for path in args.manifest:
        payload = path.read_bytes()
        if len(payload) > 16 * 1024 * 1024:
            raise ValueError("Golden manifest exceeds bounded evidence size")
        document = json.loads(payload)
        evidence.extend(document["evidence"])
        inputs.append({"source": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()})
    coverage = golden_coverage(evidence)
    report = {"coverage": coverage, "source_manifests": inputs}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    return {"PASS": 0, "BLOCKED": 2, "FAIL": 1}[coverage["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
