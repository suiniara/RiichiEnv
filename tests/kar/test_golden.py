import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from riichienv import convert
from riichienv.kar.audit import replay_game
from riichienv.kar.contracts import MJSOUL_RULESET, content_hash
from riichienv.kar.golden import find_golden_candidates, golden_coverage, verify_golden_evidence
from riichienv.kar.mjsoul import decode_record_bytes, load_protocol

HAND = ["1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p", "P"]


def _take(pool, value):
    tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == value)
    pool.remove(tile)
    return tile


def _hands(*, red_in_own=False, pair_in_one=False):
    pool = list(range(136))
    _take(pool, "C")
    own = list(HAND)
    own[-2] = "5pr" if red_in_own else "5p"
    reserved = "5p" if red_in_own else "5pr"
    _take(pool, reserved)
    own_ids = [_take(pool, value) for value in own]
    initial = [own_ids, [], [], []]
    if pair_in_one:
        initial[1] = [_take(pool, "5p"), _take(pool, "5p")]
    for seat in range(1, 4):
        while len(initial[seat]) < 13:
            initial[seat].append(pool.pop(0))
    return [[convert.tid_to_mjai(tile) for tile in hand] for hand in initial]


def _events(hands=None):
    return [
        {"type": "start_game", "id": "synthetic/golden"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000] * 4,
            "dora_marker": "C",
            "tehais": hands or _hands(),
        },
    ]


def _game(events):
    return SimpleNamespace(
        game_id="synthetic/golden",
        events=tuple(events),
        records=(),
        source_sha256=content_hash(events),
        protocol_sha256="b" * 64,
        protocol_descriptor_sha256="c" * 64,
        metadata={"record_provenance": "synthetic_fixture"},
    )


def _five_game(red=True, forced=False):
    events = _events(_hands(red_in_own=not red))
    if not forced:
        for actor in range(4):
            tile = ("9s", "8s", "7s", "6s")[actor]
            events.extend(
                [
                    {"type": "tsumo", "actor": actor, "pai": tile},
                    {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": actor != 0},
                ]
            )
    events.append({"type": "tsumo", "actor": 0, "pai": "5pr" if red else "5p"})
    return _game(events)


@pytest.mark.parametrize(
    "red,case,ids", [(True, "red_draw_normal_5p_hand", [42, 60]), (False, "normal_draw_red_5p_hand", [40, 62])]
)
def test_red_and_normal_five_detectors_only_locate_candidates(red, case, ids):
    found = find_golden_candidates(_five_game(red))
    row = next(row for row in found if row["case_id"] == case)
    assert row["status"] == "CANDIDATE"
    assert row["source_provenance"] == "unverified"
    assert row["expected"]["required_non_riichi_ids"] == ids
    assert row["evidence_type"] == "real_positive"
    assert golden_coverage([row])["status"] == "BLOCKED"
    assert golden_coverage([row])["verified_count"] == 0


@pytest.mark.parametrize("red,case", [(True, "red_draw_normal_5p_hand"), (False, "normal_draw_red_5p_hand")])
def test_dealers_forced_initial_tedashi_does_not_claim_tsumogiri_variants(red, case):
    assert case not in {row["case_id"] for row in find_golden_candidates(_five_game(red, forced=True))}


@pytest.mark.parametrize("winners,expected", [((1, 2), {"multiple_ron"}), ((1, 2, 3), {"multiple_ron", "triple_ron"})])
def test_multi_ron_candidates_use_the_complete_discard_response(winners, expected):
    events = _events() + [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
    ]
    events.extend({"type": "hora", "actor": actor, "target": 0, "pai": "1m"} for actor in winners)
    events.append({"type": "end_kyoku"})
    rows = find_golden_candidates(_game(events))
    found = {row["case_id"] for row in rows}
    assert found.intersection({"multiple_ron", "triple_ron"}) == expected
    row = next(row for row in rows if row["case_id"] == "multiple_ron")
    assert row["expected"]["response_target"] == 7 + sum(1 << (actor - 1) for actor in winners)


def test_riichi_declaration_ron_does_not_charge_the_pot_early():
    events = _events() + [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {"type": "hora", "actor": 1, "target": 0, "pai": "1m"},
        {"type": "end_kyoku"},
    ]
    rows = find_golden_candidates(_game(events))
    row = next(row for row in rows if row["case_id"] == "riichi_discard_ron")
    assert row["expected"]["riichi_deposit_not_charged_early"] is True


def _pon_and_kakan(*, commit=True):
    events = _events(_hands(pair_in_one=True)) + [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": False},
        {"type": "pon", "actor": 1, "target": 0, "pai": "5p", "consumed": ["5p", "5p"]},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
    ]
    for actor in (2, 3, 0):
        tile = ("9s", "8s", "7s", "6s")[actor]
        events.extend(
            [
                {"type": "tsumo", "actor": actor, "pai": tile},
                {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": True},
            ]
        )
    events.extend([{"type": "tsumo", "actor": 1, "pai": "5pr"}, {"type": "kakan", "actor": 1, "pai": "5pr"}])
    if commit:
        events.append({"type": "dora", "dora_marker": "8s"})
    events.extend([{"type": "hora", "actor": 2, "target": 1, "pai": "5pr"}, {"type": "end_kyoku"}])
    return _game(events)


def test_kakan_commit_and_chankan_are_not_conflated():
    robbed = {row["case_id"] for row in find_golden_candidates(_pon_and_kakan(commit=False))}
    committed = {row["case_id"] for row in find_golden_candidates(_pon_and_kakan(commit=True))}
    assert "kakan_chankan" in robbed
    assert "kakan_commit" not in robbed
    assert "kakan_commit" in committed
    assert "kakan_chankan" not in committed


def test_pon_then_discard_keeps_pon_as_the_original_response():
    rows = find_golden_candidates(_pon_and_kakan())
    row = next(row for row in rows if row["case_id"] == "pon_then_discard")
    assert row["expected"]["response_target"] == 2
    assert "called_tile_counted_once" in {row["case_id"] for row in rows}


@pytest.mark.parametrize("reason", ["four_winds", "four_riichi", "four_kans", "nagashi_mangan"])
def test_draw_candidates_require_the_recorded_fixed_category(reason):
    events = _events() + [{"type": "ryukyoku", "reason": reason}, {"type": "end_kyoku"}]
    rows = find_golden_candidates(_game(events))
    assert reason in {row["case_id"] for row in rows}


def test_empty_or_synthetic_coverage_never_passes_and_rule_exclusion_is_explicit():
    report = golden_coverage([])
    assert report["status"] == "BLOCKED"
    assert len(report["missing_real_positive"]) == 16
    assert report["not_applicable"] == [
        {
            "case_id": "abort_triple_ron",
            "status": "NOT_APPLICABLE",
            "reason": "frozen_sanchaho_is_draw_false",
            "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
            "disabled_response_id": 19,
            "retained_response_id": 14,
        }
    ]
    assert "kokushi_ankan_ron" in report["missing_real_positive"]
    assert "nagashi_mangan" in report["missing_real_positive"]
    assert report["training_exclusion_required"] is True
    assert report["source_authenticity_is_provider_signed"] is False


def test_synthetic_provenance_cannot_be_relabelled_into_a_real_positive():
    game = _five_game()
    evidence = next(row for row in find_golden_candidates(game) if row["case_id"] == "red_draw_normal_5p_hand")
    evidence.update(status="PASS", verified=True, source_provenance="real_raw_record")
    report = verify_golden_evidence(game, {}, evidence)
    assert report["status"] == "BLOCKED"
    assert "synthetic" in report["reason"]
    assert report["verified"] is False


@pytest.mark.parametrize("change", ["case_id", "observer", "event_ids", "type", "source_hash"])
def test_invalid_manifest_identity_or_locator_does_not_pass(change):
    game = _five_game()
    evidence = next(row for row in find_golden_candidates(game) if row["case_id"] == "red_draw_normal_5p_hand")
    if change == "case_id":
        evidence["case_id"] = "make_all_checks_green"
    elif change == "observer":
        evidence["observer"] = True
    elif change == "event_ids":
        evidence["event_ids"] = [-1]
    elif change == "type":
        evidence["evidence_type"] = "derived_from_real"
    else:
        evidence["source_sha256"] = "z" * 64
    result = verify_golden_evidence(game, {}, evidence)
    assert result["status"] != "PASS"
    assert result["verified"] is False


def test_forged_pass_receipts_fail_integrity_and_unknown_cases_are_not_ignored():
    forged = {
        "case_id": "triple_ron",
        "status": "PASS",
        "verified": True,
        "verification_version": "mjsoul_real_golden_v1",
        "evidence_sha256": "0" * 64,
    }
    report = golden_coverage([forged])
    assert report["status"] == "FAIL"
    assert report["verified_count"] == 0
    assert report["failed"][0]["reason"] == "golden_record_integrity_mismatch"
    assert golden_coverage([{"case_id": "not_in_registry", "status": "PASS"}])["status"] == "FAIL"


def _real_fixture():
    root = Path(__file__).parents[3] / "kar-m0-assets"
    raw = root / "200331-00a872c6-afb1-487b-ba2d-e58b9379f60e.bin"
    proto = root / "mahjongsoul.proto"
    if not raw.is_file() or not proto.is_file():
        pytest.skip("Private source-backed raw probe is not distributed with the checkout")
    game = decode_record_bytes(raw.read_bytes(), load_protocol(proto), container="majsoul_download")
    return game, raw, proto


@pytest.fixture(scope="module")
def real_evidence():
    game, raw, proto = _real_fixture()
    replay = replay_game(game)
    candidates = find_golden_candidates(game)
    return game, replay, candidates, raw, proto


def _source_evidence(candidate, raw, proto):
    return {
        **copy.deepcopy(candidate),
        "source_path": str(raw),
        "proto_path": str(proto),
        "source_provenance": "real_raw_record",
    }


def test_real_source_is_redecoded_and_case_verified_without_claiming_full_coverage(real_evidence):
    game, replay, candidates, raw, proto = real_evidence
    evidence = _source_evidence(next(row for row in candidates if row["case_id"] == "pon_then_discard"), raw, proto)
    original = copy.deepcopy(evidence)
    result = verify_golden_evidence(game, replay, evidence)
    assert result["status"] == "PASS", result
    assert result["evidence_type"] == "real_positive"
    assert evidence == original
    assert result["source_role"] == "golden_only"
    coverage = golden_coverage([result, result])
    assert coverage["status"] == "BLOCKED"
    assert coverage["verified_count"] == 1
    assert coverage["verified_cases"] == ["pon_then_discard"]
    assert "kokushi_ankan_ron" in coverage["missing_real_positive"]


def test_real_baseline_derived_contracts_are_separate_from_real_positives(real_evidence):
    game, replay, candidates, raw, proto = real_evidence
    available = {row["case_id"]: row for row in candidates}
    derived = json.loads(
        Path(__file__)
        .parents[2]
        .joinpath("src", "riichienv", "kar", "golden_registry.v1.json")
        .read_text(encoding="utf-8")
    )["required_derived_from_real"]
    assert set(derived).issubset(available)
    verified = [verify_golden_evidence(game, replay, _source_evidence(available[case], raw, proto)) for case in derived]
    assert all(row["status"] == "PASS" for row in verified), verified
    assert all(row["evidence_type"] == "derived_from_real" for row in verified)
    result = golden_coverage(verified)
    assert result["status"] == "BLOCKED"
    assert result["missing_derived_from_real"] == []
    assert len(result["missing_real_positive"]) == 16


@pytest.mark.parametrize("change", ["expected", "source", "game", "receipt", "completion", "baseline"])
def test_real_evidence_rejects_modified_expectation_bytes_and_receipts(real_evidence, change):
    game, replay, candidates, raw, proto = real_evidence
    evidence = _source_evidence(next(row for row in candidates if row["case_id"] == "pon_then_discard"), raw, proto)
    supplied = copy.deepcopy(replay)
    if change == "expected":
        evidence["expected"]["response_target"] = 0
    elif change == "source":
        evidence["source_sha256"] = "0" * 64
    elif change == "game":
        evidence["game_id"] = "other-game"
    elif change == "receipt":
        supplied["events_applied"] -= 1
    elif change == "completion":
        supplied["completion"] = {"status": "PASS", "certificate_sha256": "0" * 64}
    else:
        evidence["baseline_event_stream_sha256"] = "0" * 64
    result = verify_golden_evidence(game, supplied, evidence)
    assert result["status"] != "PASS"


def test_coverage_reruns_native_legality_instead_of_trusting_saved_counters(real_evidence, monkeypatch):
    from riichienv.kar import audit  # noqa: PLC0415

    game, replay, candidates, raw, proto = real_evidence
    candidate = next(row for row in candidates if row["case_id"] == "pon_then_discard")
    verified = verify_golden_evidence(game, replay, _source_evidence(candidate, raw, proto))
    assert verified["status"] == "PASS"
    calls = []

    def invalid_current_replay(decoded):
        calls.append(decoded.source_sha256)
        return {"status": "FAIL", "errors": [{"reason": "native_contract_failed"}]}

    monkeypatch.setattr(audit, "replay_game", invalid_current_replay)
    report = golden_coverage([verified, verified])
    assert report["status"] == "FAIL"
    assert report["verified_count"] == 0
    assert len(calls) == 1
    assert report["failed"][0]["reason"] == "current_full_native_replay_not_verified"


def test_coverage_reopens_raw_files_instead_of_only_trusting_saved_pass(real_evidence, tmp_path):
    game, replay, candidates, raw, proto = real_evidence
    copied = tmp_path / raw.name
    copied.write_bytes(raw.read_bytes())
    candidate = next(row for row in candidates if row["case_id"] == "pon_then_discard")
    result = verify_golden_evidence(game, replay, _source_evidence(candidate, copied, proto))
    assert result["status"] == "PASS", result
    copied.write_bytes(b"not the verified source")
    report = golden_coverage([result])
    assert report["status"] == "FAIL"
    assert report["verified_count"] == 0
