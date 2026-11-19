import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from riichienv.kar import audit, mjsoul
from riichienv.kar.audit import check_overlay, replay_game, response_support_evidence, run_audit
from riichienv.kar.contracts import SCHEMA_OVERLAY_SHA256, game_split


def test_missing_data_blocks_training_but_new_complete_schema_is_frozen():
    result = run_audit()
    assert result["status"] == "BLOCKED"
    assert result["training_allowed"] is False
    assert result["cloud_resources_started"] is False
    assert result["N_eligible_half_games"] == 0
    assert result["N_train_half_games"] == 0
    assert result["model_tier_basis"] == "N_train_half_games"
    registry = next(check for check in result["checks"] if check["name"] == "complete_schema")
    assert registry["status"] == "PASS"
    assert registry["schema_id"] == "kar-riichienv-v1"
    assert len(registry["schema_hash"]) == 64
    assert registry["compatibility"] == "not_claimed"
    json.dumps(result, allow_nan=False)


def test_unknown_proto_is_not_replaced_with_reference_proto(tmp_path):
    result = run_audit(proto_path=tmp_path / "missing.proto", records=(tmp_path / "missing.bin",))
    assert result["games"][0]["status"] == "BLOCKED"
    assert result["training_allowed"] is False


def test_bad_overlay_is_a_failure_not_a_missing_schema(tmp_path):
    overlay = tmp_path / "overlay.json"
    overlay.write_text('{"schema_id":"kar-v1.2.2"}', encoding="utf-8")
    assert check_overlay(overlay)["status"] == "FAIL"
    result = run_audit(overlay_path=overlay)
    assert result["status"] == "FAIL"
    assert result["training_allowed"] is False


def test_delivered_overlay_hash_is_not_full_schema_hash():
    path = Path(__file__).parents[3] / "kar-v122-complete-package-20261115" / "token_schema.v1.2.2.overlay.json"
    if not path.is_file():
        pytest.skip("Original design delivery is outside the source checkout")
    result = check_overlay(path)
    assert result["status"] == "PASS"
    assert result["sha256"] == SCHEMA_OVERLAY_SHA256
    assert result["is_complete_schema"] is False


def _start_events(hands=None):
    own = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    return [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000] * 4,
            "dora_marker": "C",
            "tehais": hands or [own, ["?"] * 13, ["?"] * 13, ["?"] * 13],
        },
    ]


def test_illegal_win_cannot_pass_with_unchanged_scores():
    events = _start_events() + [
        {"type": "tsumo", "actor": 0, "pai": "1s"},
        {"type": "hora", "actor": 0, "target": 0, "pai": "1s", "scores": [25000] * 4, "deltas": [0] * 4},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    result = replay_game(SimpleNamespace(events=events))
    assert result["status"] == "FAIL"
    assert "win is absent" in result["errors"][0]["detail"]


def test_immediate_exhaustive_draw_cannot_pass_as_zero_settlement():
    events = _start_events() + [
        {"type": "ryukyoku", "reason": "exhaustive_draw", "scores": [25000] * 4, "deltas": [0] * 4},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    result = replay_game(SimpleNamespace(events=events))
    assert result["status"] == "FAIL"
    assert "no matching public discard window" in result["errors"][0]["detail"]


@pytest.mark.parametrize("winners", [(1, 2), (1, 2, 3)])
def test_multiple_ron_is_checked_before_native_terminal_state_clears_actions(winners):
    hands = [
        ["3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "9p", "1s", "2s", "9s"],
        ["1m", "2m", "3p", "4p", "5p", "3s", "4s", "5s", "E", "E", "E", "P", "P"],
        ["1m", "2m", "6p", "7p", "8p", "6s", "7s", "8s", "F", "F", "F", "P", "P"],
        ["1m", "2m", "6m", "7m", "8m", "6p", "7p", "8p", "C", "C", "C", "N", "N"],
    ]
    payments = {1: 6400, 2: 6400, 3: 1600}
    delta = [
        -sum(payments[player] for player in winners),
        *[payments[player] if player in winners else 0 for player in range(1, 4)],
    ]
    scores = [25000 + change for change in delta]
    events = _start_events(hands) + [
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
    ]
    events.extend(
        {"type": "hora", "actor": winner, "target": 0, "pai": "3m", "scores": scores, "deltas": delta}
        for winner in winners
    )
    events.append({"type": "end_kyoku"})
    result = replay_game(SimpleNamespace(events=events))
    assert result["errors"] == []
    assert result["status"] == "BLOCKED"
    assert result["independent_settlement"]["status"] == "PASS"
    assert result["native_settlement_gaps"]
    assert 7 + sum(1 << (player - 1) for player in winners) in result["response_categories_observed"]


def test_test_games_do_not_supply_response_value_support():
    salt = "support-regression"
    train = set()
    held_out = set()
    for index in range(10000):
        game_id = f"support-game-{index}"
        split = game_split(game_id, salt)
        if split == "train" and len(train) < 499:
            train.add(game_id)
        elif split == "test" and not held_out:
            held_out.add(game_id)
        if len(train) == 499 and held_out:
            break
    assert len(train) == 499 and len(held_out) == 1
    result = response_support_evidence({0: train | held_out}, salt)
    assert result["response_support_independent_games"]["0"] == 499
    assert result["response_support_counts_by_split"]["test"]["0"] == 1
    assert result["response_value_supported"]["0"] is False


@pytest.mark.parametrize(
    "provenance,mode,eligible",
    [("synthetic", 2, 0), ("unverified", 2, 0), ("real_raw_record", 1, 0), ("real_raw_record", 2, 1)],
)
def test_eligibility_requires_explicit_real_origin_and_hanchan_metadata(
    tmp_path, monkeypatch, provenance, mode, eligible
):
    proto = tmp_path / "fixture.proto"
    proto.write_text("fixture", encoding="utf-8")
    raw = tmp_path / "fixture.bin"
    raw.write_bytes(b"fixture")
    game = SimpleNamespace(
        game_id="gate-unit-fixture",
        source_sha256="fixture",
        protocol_sha256="a" * 64,
        protocol_descriptor_sha256="b" * 64,
        metadata={"category": 2, "mode": mode, "mode_id": 12, "account_seats": [0, 1, 2, 3]},
    )
    monkeypatch.setattr(mjsoul, "load_protocol", lambda _: mjsoul.Protocol("a" * 64, "b" * 64, "fixture", None))
    monkeypatch.setattr(mjsoul, "decode_record_bytes", lambda *args, **kwargs: game)
    monkeypatch.setattr(audit, "replay_game", lambda _: {"status": "PASS", "response_categories_observed": [0]})
    report = run_audit(proto_path=proto, records=(raw,), cohort_mode_ids=(12,), record_provenance=provenance)
    assert report["N_eligible_half_games"] == eligible
    assert report["training_allowed"] is False


def _call_events(kind):
    hands = [
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
        ["1m", "1m", "2m", "3m", "4p", "5p", "6p", "1s", "2s", "3s", "E", "E", "F"],
        ["4m", "5m", "6m", "7m", "8m", "9m", "5p", "6p", "7p", "8p", "9p", "S", "S"],
        ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "W", "W", "N", "N"],
    ]
    return _start_events(hands) + [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {
            "type": kind,
            "actor": 1,
            "target": 0,
            "pai": "1m",
            "consumed": ["1m", "1m"] if kind == "pon" else ["2m", "3m"],
        },
    ]


@pytest.mark.parametrize("kind", ["chi", "pon"])
def test_call_requires_discard_not_an_extra_draw(kind):
    events = _call_events(kind) + [{"type": "tsumo", "actor": 1, "pai": "9p"}]
    result = replay_game(SimpleNamespace(events=events))
    assert result["status"] == "FAIL"
    assert "mandatory discard" in result["errors"][-1]["detail"]
    assert result["events_applied"] == len(events) - 1


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "tsumo", "actor": 1, "pai": "9p"}],
        [{"type": "tsumo", "actor": 0, "pai": "9p"}, {"type": "tsumo", "actor": 0, "pai": "9s"}],
    ],
)
def test_opening_wrong_actor_and_duplicate_draw_are_invalid(events):
    result = replay_game(SimpleNamespace(events=_start_events() + events))
    assert result["status"] == "FAIL"
    assert result["errors"]


def test_normal_draw_cannot_skip_the_next_player():
    events = _call_events("pon")[:-1] + [{"type": "tsumo", "actor": 2, "pai": "9p"}]
    result = replay_game(SimpleNamespace(events=events))
    assert result["status"] == "FAIL"
    assert "public turn" in result["errors"][-1]["detail"]


def test_daiminkan_draws_for_caller_not_discarders_next_player():
    events = _call_events("pon")[:-1]
    events[1]["tehais"][1][:2] = ["F", "F"]
    events[1]["tehais"][2][:3] = ["1m"] * 3
    events.extend(
        [
            {"type": "daiminkan", "actor": 2, "target": 0, "pai": "1m", "consumed": ["1m"] * 3},
            {"type": "tsumo", "actor": 2, "pai": "9p"},
        ]
    )
    result = replay_game(SimpleNamespace(events=events))
    assert result["errors"] == []
    assert result["events_applied"] == len(events)
    assert result["status"] == "BLOCKED"
