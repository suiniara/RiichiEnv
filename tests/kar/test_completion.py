import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from riichienv import Conditions, HandEvaluator, convert
from riichienv.kar.audit import replay_game
from riichienv.kar.completion import COMPLETION_VERSION, certify_completion
from riichienv.kar.contracts import content_hash
from riichienv.kar.mjsoul import ServerRecord, decode_record_bytes, load_protocol
from riichienv.kar.settlement import verify_settlements

KOKUSHI = ["1m", "9m", "1p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "C"]
READY = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "2s", "3s", "5p", "5p"]


def _take(pool, value):
    tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == value)
    pool.remove(tile)
    return tile


def _attach_records(events):
    records = []
    names = {
        "start_kyoku": "lq.RecordNewRound",
        "tsumo": "lq.RecordDealTile",
        "dahai": "lq.RecordDiscardTile",
        "hora": "lq.RecordHule",
        "ryukyoku": "lq.RecordNoTile",
        "reach": "lq.RecordDiscardTile",
        "reach_accepted": "lq.RecordDealTile",
    }
    for event in events:
        typ = event["type"]
        if typ in {"start_game", "end_game"}:
            continue
        if typ == "end_kyoku":
            index, digest = records[-1].index, records[-1].source_sha256
        else:
            index = len(records)
            digest = content_hash({"synthetic_record_index": index, "event": event})
            records.append(ServerRecord(index, names[typ], digest, {}))
        event["source_record_index"] = index
        event["source_record_sha256"] = digest
    return tuple(records)


def _summary(events):
    count = sum(event["type"] == "dahai" for event in events)
    return {
        "status": "BLOCKED",
        "events_total": len(events),
        "events_applied": len(events),
        "discard_decisions_checked": count,
        "complete_response_windows": count,
        "errors": [],
        "independent_settlement": verify_settlements(events),
    }


def _game(events, records, final_scores, *, game_id="synthetic/completion"):
    return SimpleNamespace(
        game_id=game_id,
        events=tuple(events),
        records=records,
        source_sha256=content_hash({"synthetic_event_source": events}),
        protocol_sha256="b" * 64,
        metadata={
            "category": 2,
            "mode": 2,
            "mode_id": 9,
            "account_seats": [0, 1, 2, 3],
            "start_time": 100,
            "end_time": 200,
            "record_provenance": "synthetic_fixture",
            "final_standings": [{"seat": seat, "part_point_1": score} for seat, score in enumerate(final_scores)],
        },
    )


def _win_game(*, early=False, explicit=False):
    hand, tile = (READY, "1s") if early else (KOKUSHI, "9p")
    pool = list(range(136))
    dora = _take(pool, "C")
    draw = _take(pool, "6s")
    own = [_take(pool, value) for value in hand]
    target = _take(pool, tile)
    hands = [[target], own, [], []]
    for seat in (0, 2, 3):
        while len(hands[seat]) < 13:
            hands[seat].append(pool.pop(0))
    result = HandEvaluator(own).calc(target, [dora], Conditions(player_wind=1))
    assert result.is_win
    delta = [-result.ron_agari, result.ron_agari, 0, 0]
    scores = [25000 + change for change in delta]
    events = [
        {"type": "start_game", "id": "synthetic/completion"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "oya": 0,
            "honba": 0,
            "kyotaku": 0,
            "scores": [25000] * 4,
            "dora_marker": "C",
            "tehais": [[convert.tid_to_mjai(t) for t in player] for player in hands],
        },
        {"type": "tsumo", "actor": 0, "pai": convert.tid_to_mjai(draw)},
        {"type": "dahai", "actor": 0, "pai": tile, "tsumogiri": False},
        {"type": "hora", "actor": 1, "target": 0, "pai": tile, "scores": scores, "deltas": delta},
        {"type": "end_kyoku"},
    ]
    if explicit:
        events.append({"type": "end_game"})
    records = _attach_records(events)
    game = _game(events, records, scores)
    return game, _summary(game.events)


def test_legacy_completion_has_a_deterministic_certificate_without_inventing_end_event():
    game, replay = _win_game()
    original_game, original_replay = copy.deepcopy(game), copy.deepcopy(replay)
    result = certify_completion(game, replay)
    assert result["status"] == "PASS"
    assert result["evidence_kind"] == "derived_frozen_hanchan_termination"
    assert result["explicit_end_game"] is False
    assert result["expected_final_scores"] == (-7000, 57000, 25000, 25000)
    assert result["final_ranks"] == [4, 1, 2, 3]
    assert result["certificate_version"] == COMPLETION_VERSION
    assert result["certificate_sha256"] == content_hash(
        {key: value for key, value in result.items() if key != "certificate_sha256"}
    )
    assert result == certify_completion(game, replay)
    assert game == original_game
    assert replay == original_replay
    assert game.events[-1]["type"] == "end_kyoku"


def test_explicit_end_also_requires_the_independent_hanchan_rule():
    game, replay = _win_game(explicit=True)
    result = certify_completion(game, replay)
    assert result["status"] == "PASS"
    assert result["explicit_end_game"] is True
    assert result["evidence_kind"] == "explicit_server_gameend"
    early, early_replay = _win_game(early=True, explicit=True)
    assert certify_completion(early, early_replay)["status"] == "FAIL"


def test_complete_header_and_matching_early_scores_are_not_completion_evidence():
    game, replay = _win_game(early=True)
    assert replay["independent_settlement"]["status"] == "PASS"
    assert replay["independent_settlement"]["independent_rule_requires_match_end"] is False
    result = certify_completion(game, replay)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "hanchan_termination_not_reached"
    replay["independent_settlement"]["independent_rule_requires_match_end"] = True
    assert certify_completion(game, replay)["status"] == "FAIL"


@pytest.mark.parametrize(
    "field", ["events_applied", "errors", "discard_decisions_checked", "complete_response_windows"]
)
def test_missing_replay_evidence_is_blocked(field):
    game, replay = _win_game()
    replay.pop(field)
    result = certify_completion(game, replay)
    assert result["status"] == "BLOCKED"
    assert result["certificate_sha256"] is None


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda report: report.update(events_applied=5), "BLOCKED"),
        (lambda report: report.update(events_applied=99), "FAIL"),
        (lambda report: report.update(events_total=99), "FAIL"),
        (lambda report: report.update(errors=[{"reason": "illegal_action"}]), "FAIL"),
        (lambda report: report.update(complete_response_windows=0), "BLOCKED"),
        (lambda report: report["independent_settlement"].update(status="BLOCKED"), "BLOCKED"),
        (lambda report: report["independent_settlement"].update(status="FAIL"), "FAIL"),
        (lambda report: report["independent_settlement"].update(version="unknown"), "BLOCKED"),
        (lambda report: report["independent_settlement"].update(ruleset_hash="other"), "BLOCKED"),
        (
            lambda report: report["independent_settlement"].update(
                native_apply_event_scores_used_as_calculation_inputs=True
            ),
            "FAIL",
        ),
        (lambda report: report["independent_settlement"].update(prefix_score_checks=[]), "BLOCKED"),
        (lambda report: report["independent_settlement"]["prefix_score_checks"][0].update(event_id=0), "FAIL"),
        (lambda report: report["independent_settlement"]["settlements"][0].update(status="BLOCKED"), "BLOCKED"),
        (lambda report: report["independent_settlement"]["settlements"][0].update(event_ids=[0]), "FAIL"),
        (lambda report: report["independent_settlement"].update(explicit_end_game=True), "FAIL"),
        (lambda report: report["independent_settlement"].update(computed_final_scores=None), "BLOCKED"),
    ],
)
def test_upstream_failures_and_incomplete_coverage_cannot_mint_a_certificate(mutate, expected):
    game, replay = _win_game()
    mutate(replay)
    assert certify_completion(game, replay)["status"] == expected


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda metadata: metadata.pop("end_time"), "BLOCKED"),
        (lambda metadata: metadata.update(end_time=0), "BLOCKED"),
        (lambda metadata: metadata.update(end_time=99), "FAIL"),
        (lambda metadata: metadata.update(start_time=True), "FAIL"),
        (lambda metadata: metadata.update(mode=1), "BLOCKED"),
        (lambda metadata: metadata.update(category=1), "BLOCKED"),
        (lambda metadata: metadata.update(account_seats=[0, 1, 2, 2]), "FAIL"),
        (lambda metadata: metadata.update(final_standings=[]), "BLOCKED"),
        (lambda metadata: metadata["final_standings"][0].update(seat=1), "FAIL"),
        (lambda metadata: metadata["final_standings"][0].pop("part_point_1"), "BLOCKED"),
        (lambda metadata: metadata["final_standings"][0].update(part_point_1=True), "FAIL"),
        (lambda metadata: metadata["final_standings"][0].update(part_point_1=-6000), "FAIL"),
        (lambda metadata: metadata.update(uuid="different-game"), "FAIL"),
    ],
)
def test_header_is_only_a_cross_check_not_a_source_for_computed_scores(mutate, expected):
    game, replay = _win_game()
    original_scores = replay["independent_settlement"]["computed_final_scores"]
    mutate(game.metadata)
    assert certify_completion(game, replay)["status"] == expected
    assert replay["independent_settlement"]["computed_final_scores"] == original_scores


def test_adjusted_header_points_are_never_used_as_raw_points():
    game, replay = _win_game()
    for row in game.metadata["final_standings"]:
        row["total_point"] = 999999
        row["part_point_2"] = -999999
    assert certify_completion(game, replay)["status"] == "PASS"


def test_cutoff_at_hora_with_a_complete_header_stays_blocked():
    game, replay = _win_game()
    game.events = game.events[:-1]
    replay["events_total"] = replay["events_applied"] = len(game.events)
    assert certify_completion(game, replay)["status"] == "BLOCKED"
    assert game.metadata["end_time"] == 200


def test_an_unresolved_next_round_cannot_reuse_a_previous_certificate():
    game, replay = _win_game()
    next_round = copy.deepcopy(game.events[1])
    next_round.update(kyoku=2, oya=1)
    game.events = (*game.events, next_round)
    replay["events_total"] = replay["events_applied"] = len(game.events)
    assert certify_completion(game, replay)["status"] == "BLOCKED"


def test_uuid_source_hash_and_terminal_record_must_be_traceable():
    game, replay = _win_game()
    game.events[0]["id"] = "another-game"
    assert certify_completion(game, replay)["reason"] == "start_game_id_mismatch"
    game, replay = _win_game()
    game.source_sha256 = "not-a-hash"
    assert certify_completion(game, replay)["status"] == "FAIL"
    game, replay = _win_game()
    game.events[-2]["source_record_sha256"] = "c" * 64
    assert certify_completion(game, replay)["reason"] == "terminal_record_hash_mismatch"
    game, replay = _win_game()
    game.records = (*game.records, ServerRecord(99, "lq.RecordNewRound", "d" * 64, {}))
    assert certify_completion(game, replay)["status"] == "FAIL"


@pytest.mark.parametrize("field,value", [("kyoku", True), ("honba", False), ("kyotaku", False)])
def test_boolean_round_counters_cannot_masquerade_as_zero_or_one(field, value):
    game, replay = _win_game()
    game.events[1][field] = value
    assert certify_completion(game, replay)["status"] == "FAIL"


def test_end_boundary_must_reference_the_same_server_result_bytes():
    game, replay = _win_game()
    game.events[-1]["source_record_sha256"] = "c" * 64
    assert certify_completion(game, replay)["reason"] == "end_boundary_record_hash_mismatch"


def _draw_bust_with_pot(*, explicit=False):  # noqa: PLR0915
    hands = [
        ["1m", "2m", "9m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "S"],
        ["4m", "5m", "6m", "4p", "5p", "6p", "4s", "5s", "6s", "S", "S", "S", "W"],
        ["7m", "8m", "9m", "7p", "8p", "9p", "7s", "8s", "9s", "W", "W", "W", "N"],
        ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "N", "N", "N", "E"],
    ]
    scores = [25000] * 4
    events = [{"type": "start_game", "id": "synthetic/completion"}]
    for round_number in range(9):
        dealer = 0 if round_number == 0 else 1
        events.append(
            {
                "type": "start_kyoku",
                "bakaze": "E",
                "kyoku": dealer + 1,
                "oya": dealer,
                "honba": round_number,
                "kyotaku": 0,
                "scores": scores.copy(),
                "dora_marker": "C",
                "tehais": copy.deepcopy(hands),
            }
        )
        pool = list(range(136))
        _take(pool, "C")
        for hand in hands:
            for value in hand:
                _take(pool, value)
        for number in range(70):
            actor, tile = (dealer + number) % 4, convert.tid_to_mjai(pool.pop(0))
            events.append({"type": "tsumo", "actor": actor, "pai": tile})
            declaration = round_number == 8 and number == 0
            if declaration:
                events.append({"type": "reach", "actor": actor})
            events.append({"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": number != 0})
            if declaration:
                events.append({"type": "reach_accepted", "actor": actor})
                scores[actor] -= 1000
        scores = [score + change for score, change in zip(scores, (-3000, 1000, 1000, 1000))]
        events.extend(
            [
                {
                    "type": "ryukyoku",
                    "reason": "exhaustive_draw",
                    "scores": scores.copy(),
                    "deltas": [-3000, 1000, 1000, 1000],
                },
                {"type": "end_kyoku"},
            ]
        )
    if explicit:
        events.append({"type": "end_game"})
    records = _attach_records(events)
    final = [-2000, 33000, 35000, 34000]
    game = _game(events, records, final)
    replay = _summary(game.events)
    if explicit:
        replay["independent_settlement"] = verify_settlements(game.events, observed_final_scores=final)
    assert replay["independent_settlement"]["status"] == "PASS"
    return game, replay


@pytest.mark.parametrize("explicit", [False, True])
def test_terminal_pot_is_independently_awarded_once_using_initial_tie_order(explicit):
    game, replay = _draw_bust_with_pot(explicit=explicit)
    result = certify_completion(game, replay)
    assert result["status"] == "PASS"
    assert result["independent_pot_deltas"] == (0, 0, 1000, 0)
    assert result["expected_final_scores"] == (-2000, 33000, 35000, 34000)
    assert result["final_ranks"] == [4, 3, 1, 2]
    assert result["riichi_sticks_before_final_award"] == 1
    assert result["explicit_end_game"] is explicit


def test_legacy_header_cannot_choose_a_different_recipient_of_the_pot():
    game, replay = _draw_bust_with_pot()
    game.metadata["final_standings"][2]["part_point_1"] -= 1000
    game.metadata["final_standings"][3]["part_point_1"] += 1000
    assert certify_completion(game, replay)["status"] == "FAIL"


def test_explicit_pot_report_cannot_charge_the_same_sticks_twice():
    game, replay = _draw_bust_with_pot(explicit=True)
    replay["independent_settlement"]["computed_riichi_sticks"] = 1
    assert certify_completion(game, replay)["reason"] == "invalid_explicit_end_pot_evidence"


def test_real_old_probe_is_certified_without_modifying_its_recorded_events():
    assets = Path(__file__).parents[3] / "kar-m0-assets"
    source = assets / "200331-00a872c6-afb1-487b-ba2d-e58b9379f60e.bin"
    if not source.is_file():
        pytest.skip("Private real probe is not distributed with the source checkout")
    game = decode_record_bytes(
        source.read_bytes(), load_protocol(assets / "mahjongsoul.proto"), container="majsoul_download"
    )
    original = copy.deepcopy(game.events)
    result = certify_completion(game, replay_game(game))
    assert result["status"] == "PASS"
    assert result["round_count"] == 9
    assert result["explicit_end_game"] is False
    assert result["evidence_kind"] == "derived_frozen_hanchan_termination"
    assert result["expected_final_scores"] == (30700, 12800, 28900, 27600)
    assert result["final_ranks"] == [1, 4, 2, 3]
    assert result["end_record_index"] == 777
    assert game.events == original
    assert game.complete is False
