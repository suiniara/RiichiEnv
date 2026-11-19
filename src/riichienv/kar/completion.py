from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from riichienv._riichienv import native_round_transition

from .contracts import ENGINE_COMMIT, MJSOUL_RULESET, content_hash
from .payments import end_game_pot
from .settlement import SETTLEMENT_VERSION

COMPLETION_VERSION = "frozen_mjsoul_hanchan_completion_v1"
INITIAL_SEAT_ORDER = (0, 1, 2, 3)


class _EvidenceError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if value is None:
        raise _EvidenceError("BLOCKED", f"missing_{name}")
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _EvidenceError("FAIL", f"invalid_{name}")
    return value


def _vector(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        raise _EvidenceError("BLOCKED", f"missing_{name}")
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise _EvidenceError("FAIL", f"invalid_{name}")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise _EvidenceError("FAIL", f"invalid_{name}")
    return tuple(value)


def _digest(value: Any, name: str) -> str:
    if not value:
        raise _EvidenceError("BLOCKED", f"missing_{name}")
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise _EvidenceError("FAIL", f"invalid_{name}")
    return value.lower()


def _passed(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _EvidenceError("BLOCKED", f"missing_{name}")
    if value.get("status") == "FAIL":
        raise _EvidenceError("FAIL", f"{name}_failed")
    if value.get("status") != "PASS":
        raise _EvidenceError("BLOCKED", f"{name}_not_verified")
    return value


def _rows(value: Any, name: str) -> Sequence[Mapping[str, Any]]:
    if value is None:
        raise _EvidenceError("BLOCKED", f"missing_{name}")
    if not isinstance(value, (list, tuple)) or any(not isinstance(row, Mapping) for row in value):
        raise _EvidenceError("FAIL", f"invalid_{name}")
    return value


def _event_structure(events: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[list[int]]]:  # noqa: PLR0915
    starts: list[int] = []
    terminals: list[list[int]] = []
    active = False
    resolved = False
    ended = False
    kan_kind: str | None = None
    kan_actor: int | None = None
    group: list[int] = []
    for index, event in enumerate(events):
        typ = event.get("type")
        if ended:
            raise _EvidenceError("FAIL", "events_after_end_game")
        if typ == "start_game":
            if index != 0:
                raise _EvidenceError("FAIL", "duplicate_start_game")
        elif typ == "start_kyoku":
            if active:
                raise _EvidenceError("FAIL", "new_round_before_previous_close")
            active, resolved, group = True, False, []
            kan_kind, kan_actor = None, None
            starts.append(index)
        elif typ in {"hora", "ryukyoku"}:
            if not active or (resolved and (typ != "hora" or events[group[-1]].get("type") != "hora")):
                raise _EvidenceError("FAIL", "conflicting_terminal_events")
            if kan_kind is not None:
                if typ != "hora" or kan_kind == "daiminkan" or event.get("target") != kan_actor:
                    raise _EvidenceError("FAIL", "terminal_before_required_replacement_draw")
                kan_kind, kan_actor = None, None
            resolved = True
            group.append(index)
        elif typ == "end_kyoku":
            if not active or not resolved:
                raise _EvidenceError("FAIL", "round_close_without_terminal_resolution")
            terminals.append(group)
            active, resolved = False, False
        elif typ == "end_game":
            if active or not terminals:
                raise _EvidenceError("FAIL", "end_game_before_round_close")
            ended = True
        elif typ == "public_hand_reveal":
            if not active or not resolved:
                raise _EvidenceError("FAIL", "hand_reveal_outside_terminal_resolution")
        else:
            if not active or resolved:
                raise _EvidenceError("FAIL", "action_outside_open_round")
            if typ in {"ankan", "kakan", "daiminkan", "kan"}:
                if kan_kind is not None:
                    raise _EvidenceError("FAIL", "nested_unresolved_kan")
                kan_kind = "daiminkan" if typ == "kan" else typ
                kan_actor = event.get("actor")
            elif typ == "tsumo" and kan_kind is not None:
                if event.get("actor") != kan_actor:
                    raise _EvidenceError("FAIL", "wrong_replacement_draw_actor")
                kan_kind, kan_actor = None, None
            elif typ == "dahai" and kan_kind is not None:
                raise _EvidenceError("FAIL", "discard_before_required_replacement_draw")
            elif typ not in {"tsumo", "dahai", "chi", "pon", "reach", "reach_accepted", "dora"}:
                raise _EvidenceError("BLOCKED", "unsupported_completion_event")
    if active or kan_kind is not None or not terminals:
        raise _EvidenceError("BLOCKED", "unresolved_final_round_or_kan")
    if events[-1].get("type") not in {"end_kyoku", "end_game"}:
        raise _EvidenceError("BLOCKED", "missing_final_round_close")
    return starts, terminals


def _initial_round(events: Sequence[Mapping[str, Any]], starts: list[int]) -> None:
    if not starts:
        raise _EvidenceError("BLOCKED", "missing_initial_round")
    first = events[starts[0]]
    expected = ("E", 1, 0, 0, 0)
    actual = (
        first.get("bakaze"),
        first.get("kyoku"),
        first.get("oya"),
        first.get("honba"),
        first.get("kyotaku", first.get("kyoutaku")),
    )
    if actual != expected or _vector(first.get("scores"), "initial_scores") != (25000,) * 4:
        raise _EvidenceError("FAIL", "not_frozen_initial_round")
    for index in starts:
        event = events[index]
        dealer = _integer(event.get("oya"), "round_dealer")
        ordinal = _integer(event.get("kyoku"), "round_ordinal", minimum=1)
        _integer(event.get("honba"), "round_honba")
        _integer(event.get("kyotaku", event.get("kyoutaku")), "round_riichi_sticks")
        if dealer >= 4 or ordinal != dealer + 1:
            raise _EvidenceError("FAIL", "invalid_round_identity")
        hands = event.get("tehais")
        if hands is None:
            raise _EvidenceError("BLOCKED", "missing_round_private_tiles")
        if not isinstance(hands, (list, tuple)) or len(hands) != 4:
            raise _EvidenceError("FAIL", "invalid_four_player_round")
        if any(not isinstance(hand, (list, tuple)) or len(hand) != 13 for hand in hands):
            raise _EvidenceError("FAIL", "invalid_initial_hand_count")
        if any(not isinstance(tile, str) or tile == "?" for hand in hands for tile in hand):
            raise _EvidenceError("BLOCKED", "incomplete_round_private_tiles")


def _coverage(independent: Mapping[str, Any], starts: list[int], groups: list[list[int]]) -> list[Mapping[str, Any]]:
    prefixes = _rows(independent.get("prefix_score_checks"), "prefix_score_checks")
    if len(prefixes) != len(starts):
        raise _EvidenceError("BLOCKED", "prefix_score_coverage_incomplete")
    for row, index in zip(prefixes, starts):
        _passed(row, "prefix_score_check")
        if row.get("event_id") != index:
            raise _EvidenceError("FAIL", "prefix_score_event_mismatch")
        if _vector(row.get("expected_scores"), "prefix_expected_scores") != _vector(
            row.get("observed_scores"), "prefix_observed_scores"
        ):
            raise _EvidenceError("FAIL", "prefix_score_evidence_conflict")
    rows = list(_rows(independent.get("settlements"), "settlements"))
    if len(rows) != len(groups):
        raise _EvidenceError("BLOCKED", "settlement_coverage_incomplete")
    for row, indices in zip(rows, groups):
        _passed(row, "settlement")
        if row.get("event_ids") != indices:
            raise _EvidenceError("FAIL", "settlement_event_coverage_mismatch")
    return rows


def _end_required(
    last_round: Mapping[str, Any],
    last_row: Mapping[str, Any],
    terminal_events: Sequence[Mapping[str, Any]],
    scores: tuple[int, ...],
) -> bool:
    wind_name = last_round.get("bakaze")
    if wind_name not in ("E", "S", "W", "N"):
        raise _EvidenceError("FAIL", "invalid_final_round_wind")
    wind = ("E", "S", "W", "N").index(wind_name)
    oya = _integer(last_round.get("oya"), "final_dealer")
    if oya >= 4 or last_round.get("kyoku") != oya + 1:
        raise _EvidenceError("FAIL", "invalid_final_round_identity")
    if last_row.get("kind") == "win":
        winners = [_integer(event.get("actor"), "winner") for event in terminal_events]
        derived = _rows(last_row.get("derived_wins"), "derived_wins")
        if any(winner >= 4 for winner in winners) or len(set(winners)) != len(winners):
            raise _EvidenceError("FAIL", "invalid_winner_group")
        if len(derived) != len(winners) or {row.get("player_id") for row in derived} != set(winners):
            raise _EvidenceError("FAIL", "derived_winner_evidence_mismatch")
        renchan, abortive = oya in winners, False
    elif last_row.get("kind") == "draw":
        reason = terminal_events[0].get("reason")
        ordinary = reason in {"exhaustive", "exhaustive_draw", "fanpai", "nagashi_mangan"}
        if ordinary:
            tenpai = last_row.get("tenpai")
            if (
                not isinstance(tenpai, (tuple, list))
                or len(tenpai) != 4
                or any(not isinstance(value, bool) for value in tenpai)
            ):
                raise _EvidenceError("BLOCKED", "missing_dealer_tenpai_evidence")
            renchan, abortive = tenpai[oya], False
        elif reason in {"four_winds", "four_riichi", "four_kans", "kyushukyuhai", "kyushu_kyuhai"}:
            renchan, abortive = True, True
        else:
            raise _EvidenceError("BLOCKED", "unsupported_terminal_draw_reason")
    else:
        raise _EvidenceError("FAIL", "unknown_terminal_settlement_kind")
    kind = "abortive_draw" if abortive else "win" if last_row.get("kind") == "win" else "exhaustive_draw"
    winner_count = len(terminal_events) if kind == "win" else 1
    honba = _integer(last_round.get("honba"), "final_honba")
    try:
        transition = native_round_transition(
            scores, oya, wind, honba, 2, kind, renchan, winner_count, INITIAL_SEAT_ORDER,
            rule=MJSOUL_RULESET.engine_rule(),
        )
    except (ValueError, TypeError, OverflowError) as error:
        raise _EvidenceError("FAIL", "invalid_native_termination_context") from error
    return transition[4]


def observed_raw_final_scores(game: Any) -> tuple[int, ...] | None:
    """Read seat-addressed raw points only as independent-ledger comparison targets."""
    rows = getattr(game, "metadata", {}).get("final_standings", [])
    if not rows:
        return None
    if len(rows) != 4 or {row.get("seat") for row in rows} != set(range(4)):
        raise ValueError("Header standings must identify four stable seats")
    if any("part_point_1" not in row for row in rows):
        return None
    by_seat = {row["seat"]: row["part_point_1"] for row in rows}
    if any(isinstance(score, bool) or not isinstance(score, int) for score in by_seat.values()):
        raise ValueError("Header raw points must be integers")
    return tuple(by_seat[seat] for seat in range(4))


def _header(game: Any, game_id: str) -> tuple[Mapping[str, Any], tuple[int, ...]]:
    metadata = getattr(game, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise _EvidenceError("BLOCKED", "missing_header_metadata")
    if metadata.get("uuid", game_id) != game_id:
        raise _EvidenceError("FAIL", "header_game_id_mismatch")
    if metadata.get("mode") != 2 or metadata.get("category") != 2:
        raise _EvidenceError("BLOCKED", "unsupported_source_game_configuration")
    seats = metadata.get("account_seats")
    if seats is None:
        raise _EvidenceError("BLOCKED", "missing_header_player_seats")
    if (
        not isinstance(seats, (tuple, list))
        or len(seats) != 4
        or any(isinstance(seat, bool) or not isinstance(seat, int) for seat in seats)
        or set(seats) != set(INITIAL_SEAT_ORDER)
    ):
        raise _EvidenceError("FAIL", "invalid_header_player_seats")
    start, end = metadata.get("start_time"), metadata.get("end_time")
    if start in (None, 0) or end in (None, 0):
        raise _EvidenceError("BLOCKED", "missing_header_completion_time")
    start = _integer(start, "start_time", minimum=1)
    end = _integer(end, "end_time", minimum=1)
    if end < start:
        raise _EvidenceError("FAIL", "header_time_reversed")
    rows = _rows(metadata.get("final_standings"), "header_final_standings")
    if len(rows) != 4:
        raise _EvidenceError("BLOCKED", "incomplete_header_final_standings")
    by_seat: dict[int, int] = {}
    for row in rows:
        seat = _integer(row.get("seat"), "standing_seat")
        if seat >= 4 or seat in by_seat:
            raise _EvidenceError("FAIL", "invalid_header_standing_seats")
        point = row.get("part_point_1")
        if point is None:
            raise _EvidenceError("BLOCKED", "header_raw_point_semantics_unavailable")
        if isinstance(point, bool) or not isinstance(point, int):
            raise _EvidenceError("FAIL", "invalid_header_raw_points")
        by_seat[seat] = point
    return metadata, tuple(by_seat[seat] for seat in INITIAL_SEAT_ORDER)


def _record_evidence(game: Any, terminal: Mapping[str, Any], end_event: Mapping[str, Any]) -> dict[str, Any]:
    index = _integer(terminal.get("source_record_index"), "terminal_record_index")
    digest = _digest(terminal.get("source_record_sha256"), "terminal_record_sha256")
    records = getattr(game, "records", None)
    if not isinstance(records, (tuple, list)) or not records:
        raise _EvidenceError("BLOCKED", "missing_server_records")
    matching = [record for record in records if getattr(record, "index", None) == index]
    if len(matching) != 1 or getattr(records[-1], "index", None) != index:
        raise _EvidenceError("FAIL", "terminal_record_not_unique_final_server_result")
    if _digest(getattr(matching[0], "source_sha256", None), "record_sha256") != digest:
        raise _EvidenceError("FAIL", "terminal_record_hash_mismatch")
    if getattr(matching[0], "name", None) not in {"lq.RecordHule", "lq.RecordNoTile", "lq.RecordLiuJu"}:
        raise _EvidenceError("FAIL", "last_server_result_is_not_terminal")
    if "source_record_index" in end_event and end_event["source_record_index"] != index:
        raise _EvidenceError("FAIL", "end_boundary_record_mismatch")
    if (
        "source_record_sha256" in end_event
        and _digest(end_event["source_record_sha256"], "end_boundary_record_sha256") != digest
    ):
        raise _EvidenceError("FAIL", "end_boundary_record_hash_mismatch")
    return {"end_record_index": index, "end_record_sha256": digest}


def _certificate(game: Any, replay: Mapping[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
    events = getattr(game, "events", None)
    if not isinstance(events, (tuple, list)) or not events or any(not isinstance(event, Mapping) for event in events):
        raise _EvidenceError("BLOCKED", "missing_normalized_events")
    if events[0].get("type") != "start_game":
        raise _EvidenceError("BLOCKED", "missing_start_game")
    game_id = getattr(game, "game_id", None)
    if not isinstance(game_id, str) or not game_id:
        raise _EvidenceError("BLOCKED", "missing_game_id")
    if events[0].get("id") is None:
        raise _EvidenceError("BLOCKED", "missing_start_game_id")
    if events[0]["id"] != game_id:
        raise _EvidenceError("FAIL", "start_game_id_mismatch")
    source_hash = _digest(getattr(game, "source_sha256", None), "source_sha256")
    protocol_hash = _digest(getattr(game, "protocol_sha256", None), "protocol_sha256")
    metadata, observed = _header(game, game_id)
    if replay.get("errors") is None:
        raise _EvidenceError("BLOCKED", "missing_replay_error_evidence")
    if not isinstance(replay["errors"], (list, tuple)) or replay["errors"]:
        raise _EvidenceError("FAIL", "replay_has_errors")
    applied = _integer(replay.get("events_applied"), "events_applied")
    if applied != len(events):
        raise _EvidenceError(
            "BLOCKED" if applied < len(events) else "FAIL", "replay_did_not_consume_exact_event_stream"
        )
    if replay.get("events_total", len(events)) != len(events):
        raise _EvidenceError("FAIL", "replay_event_total_mismatch")
    discards = sum(event.get("type") == "dahai" for event in events)
    checked = _integer(replay.get("discard_decisions_checked"), "discard_decisions_checked")
    windows = _integer(replay.get("complete_response_windows"), "complete_response_windows")
    if checked != discards or windows != discards:
        raise _EvidenceError("BLOCKED", "discard_or_response_coverage_incomplete")
    starts, groups = _event_structure(events)
    _initial_round(events, starts)
    independent = _passed(replay.get("independent_settlement"), "independent_settlement")
    if (
        independent.get("version") != SETTLEMENT_VERSION
        or independent.get("ruleset_hash") != MJSOUL_RULESET.ruleset_hash
    ):
        raise _EvidenceError("BLOCKED", "unsupported_independent_settlement_contract")
    for field in (
        "server_han_fu_fans_payment_labels_used_as_calculation_inputs",
        "native_apply_event_scores_used_as_calculation_inputs",
    ):
        if independent.get(field) is True:
            raise _EvidenceError("FAIL", "settlement_not_independent")
        if independent.get(field) is not False:
            raise _EvidenceError("BLOCKED", "missing_settlement_independence_evidence")
    if independent.get("errors") is None:
        raise _EvidenceError("BLOCKED", "missing_settlement_error_evidence")
    if not isinstance(independent["errors"], (list, tuple)) or independent["errors"]:
        raise _EvidenceError("FAIL", "settlement_has_errors")
    rows = _coverage(independent, starts, groups)
    explicit = events[-1].get("type") == "end_game"
    if independent.get("explicit_end_game") is not explicit:
        raise _EvidenceError("FAIL", "explicit_end_evidence_conflict")
    scores = _vector(independent.get("computed_final_scores"), "computed_final_scores")
    remaining_pot = _integer(independent.get("computed_riichi_sticks"), "computed_riichi_sticks")
    reported_pot_delta = _vector(independent.get("end_game_pot_deltas"), "end_game_pot_deltas")
    if explicit:
        if remaining_pot != 0 or any(value < 0 for value in reported_pot_delta) or sum(reported_pot_delta) % 1000:
            raise _EvidenceError("FAIL", "invalid_explicit_end_pot_evidence")
        pot_before = sum(reported_pot_delta) // 1000
        before_award = tuple(score - delta for score, delta in zip(scores, reported_pot_delta))
        if end_game_pot(before_award, pot_before, INITIAL_SEAT_ORDER) != reported_pot_delta:
            raise _EvidenceError("FAIL", "incorrect_explicit_end_pot_allocation")
        pot_delta = reported_pot_delta
        expected = scores
    else:
        if any(reported_pot_delta):
            raise _EvidenceError("FAIL", "legacy_pot_was_already_awarded_without_end_evidence")
        pot_before, before_award = remaining_pot, scores
        pot_delta = end_game_pot(before_award, pot_before, INITIAL_SEAT_ORDER)
        expected = tuple(score + delta for score, delta in zip(before_award, pot_delta))
    if sum(before_award) + pot_before * 1000 != 100000 or sum(expected) != 100000:
        raise _EvidenceError("FAIL", "independent_point_and_pot_conservation_conflict")
    required = _end_required(events[starts[-1]], rows[-1], [events[index] for index in groups[-1]], before_award)
    flag = independent.get("independent_rule_requires_match_end")
    if flag is None:
        raise _EvidenceError("BLOCKED", "missing_independent_termination_evidence")
    if not isinstance(flag, bool) or flag != required:
        raise _EvidenceError("FAIL", "independent_termination_evidence_conflict")
    if not required:
        raise _EvidenceError("FAIL" if explicit else "BLOCKED", "hanchan_termination_not_reached")
    if expected != observed:
        raise _EvidenceError("FAIL", "header_raw_scores_disagree_with_independent_final_scores")
    record = _record_evidence(game, events[groups[-1][-1]], events[-1])
    ranking = sorted(INITIAL_SEAT_ORDER, key=lambda seat: (-expected[seat], seat))
    ranks = [ranking.index(seat) + 1 for seat in INITIAL_SEAT_ORDER]
    evidence = {
        "status": "PASS",
        "certificate_version": COMPLETION_VERSION,
        "evidence_kind": "explicit_server_gameend" if explicit else "derived_frozen_hanchan_termination",
        "game_id": game_id,
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "reference_engine_commit": ENGINE_COMMIT,
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "settlement_version": SETTLEMENT_VERSION,
        "explicit_end_game": explicit,
        "initial_seat_order_player_ids": INITIAL_SEAT_ORDER,
        "round_count": len(starts),
        "start_event_id": 0,
        "initial_round_event_id": starts[0],
        "end_event_id": len(events) - 1,
        "terminal_event_ids": groups[-1],
        **record,
        "scores_before_final_pot": before_award,
        "riichi_sticks_before_final_award": pot_before,
        "independent_pot_deltas": pot_delta,
        "expected_final_scores": expected,
        "observed_final_scores": observed,
        "final_ranks": ranks,
        "event_stream_sha256": content_hash(events),
        "replay_evidence_sha256": content_hash(
            {
                "events_applied": applied,
                "discard_decisions_checked": checked,
                "complete_response_windows": windows,
                "errors": replay["errors"],
                "independent_settlement": independent,
            }
        ),
        "header_evidence_sha256": content_hash(
            {
                "game_id": game_id,
                "start_time": metadata["start_time"],
                "end_time": metadata["end_time"],
                "account_seats": metadata["account_seats"],
                "raw_scores": observed,
            }
        ),
        "gameplay_events_were_modified": False,
    }
    return {**evidence, "certificate_sha256": content_hash(evidence)}


def certify_completion(game: Any, replay_result: Mapping[str, Any]) -> dict[str, Any]:
    try:
        if not isinstance(replay_result, Mapping):
            raise _EvidenceError("BLOCKED", "missing_replay_result")
        return _certificate(game, replay_result)
    except _EvidenceError as error:
        return {
            "status": error.status,
            "certificate_version": COMPLETION_VERSION,
            "reason": str(error),
            "certificate_sha256": None,
        }
    except (KeyError, TypeError, ValueError) as error:
        return {
            "status": "FAIL",
            "certificate_version": COMPLETION_VERSION,
            "reason": "malformed_completion_evidence",
            "detail": str(error),
            "certificate_sha256": None,
        }
