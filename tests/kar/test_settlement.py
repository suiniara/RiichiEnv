import copy
from collections import Counter
from pathlib import Path

import pytest

from riichienv import Conditions, HandEvaluator, Meld, MeldType, convert
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.mjsoul import decode_record_bytes, load_protocol
from riichienv.kar.settlement import _Verifier, verify_settlements

READY = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "2s", "3s", "5p", "5p"]


def _allocate(available, value):
    for tile in available:
        if convert.tid_to_mjai(tile) == value:
            available.remove(tile)
            return tile
    raise AssertionError(f"No physical copy of {value}")


def _layout(winner, hand, *, target=None, win_tile="1s", extra=()):
    available = list(range(136))
    _allocate(available, "C")
    reserved = [_allocate(available, value) for value in extra]
    own = [_allocate(available, value) for value in hand]
    winning = _allocate(available, win_tile)
    hands = []
    for seat in range(4):
        if seat == winner:
            tiles = own
        else:
            tiles = [winning] if seat == target else []
            while len(tiles) < 13:
                tiles.append(available.pop(0))
        hands.append([convert.tid_to_mjai(tile) for tile in tiles])
    return hands, available, winning, reserved


def _start(hands, *, scores=None, oya=0, honba=0, kyoku=1):
    return [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": kyoku,
            "honba": honba,
            "kyotaku": 0,
            "oya": oya,
            "scores": scores or [25000] * 4,
            "dora_marker": "C",
            "tehais": copy.deepcopy(hands),
        },
    ]


def _win_fixture(*, tsumo=False, riichi=False, double=False, expired=False):  # noqa: PLR0915
    winner = 1
    hands, pool, winning, reserved = _layout(winner, READY, target=None if tsumo else 0, extra=("9m",))
    events = _start(hands)
    points = [25000] * 4
    cycles = 1 if not riichi or double else 2
    cycles += int(expired)
    declare_cycle = 0 if double else 1
    for cycle in range(cycles):
        for actor in range(4):
            tile = convert.tid_to_mjai(pool.pop())
            events.append({"type": "tsumo", "actor": actor, "pai": tile})
            declaration = riichi and actor == winner and cycle == declare_cycle
            if declaration:
                events.append({"type": "reach", "actor": actor})
            events.append({"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": not (actor == 0 and cycle == 0)})
            if declaration:
                events.append({"type": "reach_accepted", "actor": actor})
                points[actor] -= 1000
    tile = convert.tid_to_mjai(pool.pop())
    events.append({"type": "tsumo", "actor": 0, "pai": tile})
    if tsumo:
        events.append({"type": "dahai", "actor": 0, "pai": tile, "tsumogiri": True})
        events.append({"type": "tsumo", "actor": winner, "pai": convert.tid_to_mjai(winning)})
        target = winner
    else:
        events.append({"type": "dahai", "actor": 0, "pai": convert.tid_to_mjai(winning), "tsumogiri": False})
        target = 0
    condition = Conditions(
        tsumo=tsumo,
        riichi=riichi,
        double_riichi=riichi and double,
        ippatsu=riichi and not expired,
        player_wind=winner,
        round_wind=0,
    )
    ura = reserved if riichi else []
    expected = HandEvaluator([convert.mjai_to_tid(value) for value in READY]).calc(
        winning,
        [convert.mjai_to_tid("C")],
        condition,
        ura,
    )
    assert expected.is_win
    delta = [0] * 4
    if tsumo:
        for payer in (0, 2, 3):
            pay = expected.tsumo_agari_oya if payer == 0 else expected.tsumo_agari_ko
            delta[payer] -= pay
            delta[winner] += pay
    else:
        delta[target] -= expected.ron_agari
        delta[winner] += expected.ron_agari
    delta[winner] += 1000 if riichi else 0
    scores = [score + change for score, change in zip(points, delta)]
    event = {
        "type": "hora",
        "actor": winner,
        "target": target,
        "pai": "1s",
        "scores": scores,
        "deltas": delta,
        "han": expected.han,
        "fu": expected.fu,
    }
    if riichi:
        event["ura_markers"] = [convert.tid_to_mjai(tile) for tile in ura]
    events.extend([event, {"type": "end_kyoku"}])
    return events, expected


@pytest.mark.parametrize("tsumo", [False, True])
@pytest.mark.parametrize(
    "riichi,double,expired", [(False, False, False), (True, False, False), (True, True, False), (True, False, True)]
)
def test_independent_hand_and_context_agree_with_explicit_scalar_reference(tsumo, riichi, double, expired):
    events, expected = _win_fixture(tsumo=tsumo, riichi=riichi, double=double, expired=expired)
    report = verify_settlements(events)
    assert report["status"] == "PASS"
    detail = report["settlements"][0]["derived_wins"][0]
    assert (detail["computed_han"], detail["computed_fu"]) == (expected.han, expected.fu)
    assert detail["conditions"]["double_riichi"] is (riichi and double)
    assert detail["conditions"]["ippatsu"] is (riichi and not expired)
    assert detail["conditions"]["tsumo_first_turn"] is False
    assert report["computed_final_scores"] == tuple(events[-2]["scores"])
    assert report["native_apply_event_scores_used_as_calculation_inputs"] is False


def test_reported_han_fu_and_fans_cannot_change_the_calculation():
    events, _ = _win_fixture(riichi=True)
    original = verify_settlements(events)
    changed = copy.deepcopy(events)
    changed[-2].update(han=999, fu=1, fans=[{"id": 999, "val": 999}], point_rong=999999, points=999999)
    poisoned = verify_settlements(changed)
    assert poisoned["status"] == "PASS"
    assert poisoned["settlements"][0]["derived_wins"] == original["settlements"][0]["derived_wins"]
    assert poisoned["computed_final_scores"] == original["computed_final_scores"]


@pytest.mark.parametrize("field", ["deltas", "scores"])
def test_poisoned_money_is_a_failure_not_a_new_input(field):
    events, _ = _win_fixture()
    original = verify_settlements(events)
    changed = copy.deepcopy(events)
    changed[-2][field][0] += 100
    changed[-2][field][1] -= 100
    report = verify_settlements(changed)
    assert report["status"] == "FAIL"
    assert report["computed_final_scores"] == original["computed_final_scores"]
    assert report["settlements"][0]["reason"] == "independent_settlement_mismatch"


@pytest.mark.parametrize("ura", [None, [], ["9m", "8m"]])
def test_unknown_ura_never_uses_reported_han_or_next_round_to_restore_scores(ura):
    events, _ = _win_fixture(riichi=True)
    correct_scores = list(events[-2]["scores"])
    if ura is None:
        events[-2].pop("ura_markers")
    else:
        events[-2]["ura_markers"] = ura
    next_round = _start(events[1]["tehais"], scores=correct_scores, oya=1, kyoku=2)[1]
    events.append(next_round)
    report = verify_settlements(events)
    assert report["status"] == "BLOCKED"
    assert report["computed_final_scores"] is None
    assert "ura" in report["settlements"][0]["reason"]
    assert report["prefix_score_checks"][-1]["status"] == "BLOCKED"


def test_early_header_cannot_seal_a_missing_payment_comparison():
    events, _ = _win_fixture()
    final = tuple(events[-2].pop("scores"))
    events[-2].pop("deltas")
    report = verify_settlements(events, observed_final_scores=final)
    assert report["status"] == "BLOCKED"
    assert report["computed_final_scores"] == final
    assert report["independent_rule_requires_match_end"] is False
    assert report["settlements"][0]["reason"] == "settlement_comparison_target_missing"


@pytest.mark.parametrize("target", ["correct", "wrong", "absent"])
def test_legacy_endpoint_can_compare_missing_payment_without_reanchoring_points(target):
    from riichienv.kar.completion import observed_raw_final_scores  # noqa: PLC0415

    assets = Path(__file__).parents[3] / "kar-m0-assets"
    raw = assets / "200331-00a872c6-afb1-487b-ba2d-e58b9379f60e.bin"
    if not raw.is_file():
        pytest.skip("Private raw source is not distributed with the code")
    game = decode_record_bytes(
        raw.read_bytes(), load_protocol(assets / "mahjongsoul.proto"), container="majsoul_download"
    )
    events = copy.deepcopy(list(game.events))
    last_win = next(event for event in reversed(events) if event["type"] == "hora")
    last_win.pop("scores", None)
    last_win.pop("deltas", None)
    expected = observed_raw_final_scores(game)
    assert expected is not None
    observed = expected if target == "correct" else None if target == "absent" else (expected[0] + 100, *expected[1:])
    result = verify_settlements(events, observed_final_scores=observed)
    assert result["computed_final_scores"] == expected
    assert result["independent_rule_requires_match_end"] is True
    assert result["explicit_end_game"] is False
    assert result["status"] == ("PASS" if target == "correct" else "BLOCKED")
    assert result["final_score_comparison"]["is_completion_evidence"] is False
    if target == "correct":
        assert result["settlements"][-1]["comparison_source"] == "final_raw_scores_after_independent_termination"


def test_header_points_are_comparison_only_and_cannot_prove_completion():
    events, _ = _win_fixture()
    baseline = verify_settlements(events)
    report = verify_settlements(events, observed_final_scores=[25000] * 4)
    assert report["status"] == "PASS"
    assert report["computed_final_scores"] == baseline["computed_final_scores"]
    assert report["final_score_comparison"]["status"] == "DIAGNOSTIC_ONLY"
    assert report["final_score_comparison"]["scores_match"] is False
    assert report["explicit_end_game"] is False
    invalid = verify_settlements(events, observed_final_scores=[True, 25000, 25000, 25000])
    assert invalid["status"] == "FAIL"
    assert invalid["computed_final_scores"] == baseline["computed_final_scores"]


def test_wrong_round_score_and_honba_cannot_reanchor_the_independent_ledger():
    events, _ = _win_fixture()
    scores = list(events[-2]["scores"])
    next_round = _start(events[1]["tehais"], scores=scores, oya=1, kyoku=2)[1]
    bad_honba = copy.deepcopy(next_round)
    bad_honba["honba"] = 1
    assert verify_settlements(events + [bad_honba])["status"] == "FAIL"
    bad_score = copy.deepcopy(next_round)
    bad_score["scores"][0] += 1000
    bad_score["scores"][1] -= 1000
    report = verify_settlements(events + [bad_score])
    assert report["status"] == "FAIL"
    assert report["computed_final_scores"] == tuple(scores)


def test_furiten_applies_to_the_whole_wait_set_not_only_the_claimed_tile():
    events, _ = _win_fixture()
    first_draw = next(index for index, event in enumerate(events) if event["type"] == "tsumo" and event["actor"] == 1)
    events[first_draw]["pai"] = "4s"
    events[first_draw + 1]["pai"] = "4s"
    assert events[-2]["pai"] == "1s"
    report = verify_settlements(events)
    assert report["status"] == "FAIL"
    assert "furiten" in report["settlements"][0]["reason"]


def test_consistent_money_does_not_make_east_one_a_complete_hanchan():
    events, _ = _win_fixture()
    report = verify_settlements(events + [{"type": "end_game"}])
    assert report["status"] == "FAIL"
    assert "hanchan termination" in report["errors"][-1]["reason"]


@pytest.mark.parametrize(
    "wind,oya,points,renchan,abortive,expected",
    [
        (0, 0, (-1000, 31000, 35000, 35000), False, False, True),
        (1, 3, (20000, 24000, 25000, 31000), True, False, True),
        (1, 3, (31000, 18000, 20000, 31000), True, False, False),
        (1, 3, (25000, 25000, 25000, 25000), False, False, False),
        (2, 3, (25000, 25000, 25000, 25000), False, False, True),
        (1, 3, (20000, 24000, 25000, 31000), True, True, False),
    ],
)
def test_match_end_is_derived_from_frozen_rules_not_server_end_marker(wind, oya, points, renchan, abortive, expected):
    state = _Verifier(MJSOUL_RULESET)
    state.wind, state.oya, state.points = wind, oya, points
    state._next_round(renchan, True, abortive=abortive)
    assert state.rule_requires_match_end is expected


def test_end_pot_uses_independent_points_and_initial_tie_order():
    state = _Verifier(MJSOUL_RULESET)
    state.wind, state.oya = 2, 3
    state.points, state.pot = (24750,) * 4, 1
    state._next_round(False, True)
    state.observe({"type": "end_game"}, 0)
    assert state.points == (25750, 24750, 24750, 24750)
    assert state.pot == 0


@pytest.mark.parametrize("ready_count", range(5))
def test_exhaustive_draw_uses_private_shape_tenpai_and_not_server_tingpai(ready_count):
    hands = [
        ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "S"],
        ["4m", "5m", "6m", "4p", "5p", "6p", "4s", "5s", "6s", "S", "S", "S", "W"],
        ["7m", "8m", "9m", "7p", "8p", "9p", "7s", "8s", "9s", "W", "W", "W", "N"],
        ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "N", "N", "N", "E"],
    ]
    for seat in range(ready_count, 4):
        hands[seat][2] = ("9m", "1m", "4m", "7m")[seat]
    pool = list(range(136))
    _allocate(pool, "C")
    for hand in hands:
        for value in hand:
            _allocate(pool, value)
    events = _start(hands)
    for number in range(70):
        actor, tile = number % 4, convert.tid_to_mjai(pool.pop(0))
        events.extend(
            [
                {"type": "tsumo", "actor": actor, "pai": tile},
                {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": number != 0},
            ]
        )
    delta = (
        [0] * 4
        if ready_count in (0, 4)
        else [3000 // ready_count if seat < ready_count else -3000 // (4 - ready_count) for seat in range(4)]
    )
    events.extend(
        [
            {
                "type": "ryukyoku",
                "reason": "exhaustive_draw",
                "scores": [25000 + d for d in delta],
                "deltas": delta,
                "tingpai": [not (seat < ready_count) for seat in range(4)],
            },
            {"type": "end_kyoku"},
        ]
    )
    report = verify_settlements(events)
    assert report["status"] == "PASS"
    assert report["settlements"][0]["tenpai"] == [seat < ready_count for seat in range(4)]
    assert report["settlements"][0]["expected_deltas"] == tuple(delta)


def test_abortive_draw_without_payment_labels_is_checked_by_next_round_start():
    hands, _, _, _ = _layout(1, READY, target=0)
    events = _start(hands) + [{"type": "ryukyoku", "reason": "four_winds"}, {"type": "end_kyoku"}]
    next_round = _start(hands, honba=1)[1]
    report = verify_settlements(events + [next_round])
    assert report["settlements"][0]["status"] == "PASS"
    assert report["settlements"][0]["comparison_source"] == "next_round_start_scores"
    assert "reason" not in report["settlements"][0]


def test_real_probe_matches_all_nine_independent_payments_if_assets_are_available():
    root = Path(__file__).parents[3] / "kar-m0-assets"
    raw = root / "200331-00a872c6-afb1-487b-ba2d-e58b9379f60e.bin"
    if not raw.is_file():
        pytest.skip("Private raw probe is not distributed with the source checkout")
    game = decode_record_bytes(
        raw.read_bytes(), load_protocol(root / "mahjongsoul.proto"), container="majsoul_download"
    )
    report = verify_settlements(game.events)
    assert report["status"] == "PASS"
    assert len(report["settlements"]) == 9
    assert report["computed_final_scores"] == (30700, 12800, 28900, 27600)
    assert report["independent_rule_requires_match_end"] is True
    assert report["explicit_end_game"] is False
    assert all(row["status"] == "PASS" for row in report["settlements"])
    assert sum(Counter(row["kind"] for row in report["settlements"]).values()) == 9


def _specific_layout(known, reserve):
    pool = list(range(136))
    _allocate(pool, "C")
    reserved = [_allocate(pool, value) for value in reserve]
    hands = [[] for _ in range(4)]
    for seat, values in known.items():
        hands[seat] = [_allocate(pool, value) for value in values]
    for hand in hands:
        while len(hand) < 13:
            hand.append(pool.pop(0))
    return [[convert.tid_to_mjai(tile) for tile in hand] for hand in hands], pool, reserved


def _append_junk_turn(events, pool, actor, *, first=False, avoid=()):
    index = next(index for index in range(len(pool) - 1, -1, -1) if convert.tid_to_mjai(pool[index]) not in avoid)
    tile = convert.tid_to_mjai(pool.pop(index))
    events.extend(
        [
            {"type": "tsumo", "actor": actor, "pai": tile},
            {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": not first},
        ]
    )


def _single_win_event(actor, target, tile, base, *, points=None):
    delta = [0] * 4
    if actor == target:
        for payer in range(4):
            if payer != actor:
                payment = base.tsumo_agari_oya if payer == 0 else base.tsumo_agari_ko
                delta[payer] -= payment
                delta[actor] += payment
    else:
        delta[target] -= base.ron_agari
        delta[actor] += base.ron_agari
    before = points or [25000] * 4
    return {
        "type": "hora",
        "actor": actor,
        "target": target,
        "pai": tile,
        "deltas": delta,
        "scores": [score + change for score, change in zip(before, delta)],
    }


@pytest.mark.parametrize("tsumo", [False, True])
def test_last_live_tile_sets_haitei_or_houtei_from_the_draw_counter(tsumo):
    winner = 1 if tsumo else 2
    hands, pool, reserved = _specific_layout({winner: READY}, ("1s",))
    events = _start(hands)
    for number in range(69):
        actor = number % 4
        _append_junk_turn(
            events, pool, actor, first=number == 0, avoid=("1s", "4s") if not tsumo and actor == winner else ()
        )
    events.append({"type": "tsumo", "actor": 1, "pai": "1s"})
    if not tsumo:
        events.append({"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": True})
    conditions = Conditions(tsumo=tsumo, haitei=tsumo, houtei=not tsumo, player_wind=winner)
    base = HandEvaluator([convert.mjai_to_tid(tile) for tile in READY]).calc(
        reserved[0], [convert.mjai_to_tid("C")], conditions
    )
    events.extend([_single_win_event(winner, winner if tsumo else 1, "1s", base), {"type": "end_kyoku"}])
    report = verify_settlements(events)
    assert report["status"] == "PASS"
    derived = report["settlements"][0]["derived_wins"][0]["conditions"]
    assert derived["haitei"] is tsumo
    assert derived["houtei"] is (not tsumo)
    assert derived["rinshan"] is False


@pytest.mark.parametrize("two_kans", [False, True])
def test_rinshan_and_multiple_kans_add_the_winning_tile_exactly_once(two_kans):
    hand = ["1m"] * 4 + (
        ["9m"] * 4 + ["2p", "3p", "4p", "5s", "5s"]
        if two_kans
        else ["2p", "3p", "4p", "6s", "7s", "8s", "E", "E", "5s"]
    )
    reserve = ("6s", "7s", "8s", "7p", "9p") if two_kans else ("5s", "E", "7p")
    hands, pool, _ = _specific_layout({1: hand}, reserve)
    events = _start(hands)
    _append_junk_turn(events, pool, 0, first=True)
    initial, first = ("6s", "7s") if two_kans else ("5s", "E")
    events.extend(
        [
            {"type": "tsumo", "actor": 1, "pai": initial},
            {"type": "ankan", "actor": 1, "consumed": ["1m"] * 4},
            {"type": "dora", "dora_marker": "7p"},
            {"type": "tsumo", "actor": 1, "pai": first},
        ]
    )
    melds = [Meld(MeldType.Ankan, [0, 1, 2, 3], False)]
    indicators = [convert.mjai_to_tid(value) for value in ("C", "7p")]
    if two_kans:
        events.extend(
            [
                {"type": "ankan", "actor": 1, "consumed": ["9m"] * 4},
                {"type": "dora", "dora_marker": "9p"},
                {"type": "tsumo", "actor": 1, "pai": "8s"},
            ]
        )
        melds.append(Meld(MeldType.Ankan, [32, 33, 34, 35], False))
        indicators.append(convert.mjai_to_tid("9p"))
        pre = ["2p", "3p", "4p", "5s", "5s", "6s", "7s"]
        win = "8s"
    else:
        pre = ["2p", "3p", "4p", "6s", "7s", "8s", "E", "E", "5s", "5s"]
        win = "E"
    base = HandEvaluator([convert.mjai_to_tid(value) for value in pre], melds).calc(
        convert.mjai_to_tid(win), indicators, Conditions(tsumo=True, rinshan=True, player_wind=1)
    )
    events.extend([_single_win_event(1, 1, win, base), {"type": "end_kyoku"}])
    report = verify_settlements(events)
    assert report["status"] == "PASS"
    derived = report["settlements"][0]["derived_wins"][0]
    assert derived["conditions"]["rinshan"] is True
    assert derived["conditions"]["tsumo_first_turn"] is False
    assert (derived["computed_han"], derived["computed_fu"]) == (base.han, base.fu)
    if two_kans:
        assert derived["computed_fu"] == 90


@pytest.mark.parametrize("target", [0, 3, None])
def test_liability_source_is_derived_from_the_third_public_dragon_call(target):
    hand = ["P", "P", "F", "F", "C", "C", "1m", "2m", "3m", "1p", "2p", "3p", "5m"]
    known = {1: hand, 0: ["P"], 2: ["F"], 3: ["C"]}
    if target is not None:
        known[target].append("5m")
    hands, pool, reserved = _specific_layout(known, ("5m",) if target is None else ())
    events = _start(hands)
    for source, dragon, discard in ((0, "P", "1p"), (2, "F", "2p"), (3, "C", "3p")):
        if source == 3:
            _append_junk_turn(events, pool, 2)
        tile = convert.tid_to_mjai(pool.pop())
        events.extend(
            [
                {"type": "tsumo", "actor": source, "pai": tile},
                {"type": "dahai", "actor": source, "pai": dragon, "tsumogiri": False},
                {"type": "pon", "actor": 1, "target": source, "pai": dragon, "consumed": [dragon, dragon]},
                {"type": "dahai", "actor": 1, "pai": discard, "tsumogiri": False},
            ]
        )
    for actor in (2, 3, 0):
        if actor == target:
            tile = convert.tid_to_mjai(pool.pop())
            events.extend(
                [
                    {"type": "tsumo", "actor": actor, "pai": tile},
                    {"type": "dahai", "actor": actor, "pai": "5m", "tsumogiri": False},
                ]
            )
            break
        _append_junk_turn(events, pool, actor)
    if target is None:
        events.append({"type": "tsumo", "actor": 1, "pai": convert.tid_to_mjai(reserved[0])})
        delta = [0, 32000, 0, -32000]
    else:
        delta = [0, 32000, 0, -16000]
        delta[target] -= 16000
    events.extend(
        [
            {
                "type": "hora",
                "actor": 1,
                "target": 1 if target is None else target,
                "pai": "5m",
                "scores": [25000 + change for change in delta],
                "deltas": delta,
            },
            {"type": "end_kyoku"},
        ]
    )
    report = verify_settlements(events)
    assert report["status"] == "PASS"
    detail = report["settlements"][0]["derived_wins"][0]
    assert detail["liable_player"] == 3
    assert detail["liable_yakuman_units"] == 1
    assert report["settlements"][0]["expected_deltas"] == tuple(delta)


def test_chankan_keeps_ippatsu_until_the_kan_is_committed():
    hands, pool, reserved = _specific_layout({1: READY, 0: ["1s", "1s"], 3: ["1s"]}, ("1s", "9m"))
    events = _start(hands)
    for actor in (0, 1, 2):
        _append_junk_turn(events, pool, actor, first=actor == 0)
    tile = convert.tid_to_mjai(pool.pop())
    discard = next(value for value in hands[0] if value != "1s")
    events.extend(
        [
            {"type": "tsumo", "actor": 3, "pai": tile},
            {"type": "dahai", "actor": 3, "pai": "1s", "tsumogiri": False},
            {"type": "pon", "actor": 0, "target": 3, "pai": "1s", "consumed": ["1s", "1s"]},
            {"type": "dahai", "actor": 0, "pai": discard, "tsumogiri": False},
        ]
    )
    tile = convert.tid_to_mjai(pool.pop())
    events.extend(
        [
            {"type": "tsumo", "actor": 1, "pai": tile},
            {"type": "reach", "actor": 1},
            {"type": "dahai", "actor": 1, "pai": tile, "tsumogiri": True},
            {"type": "reach_accepted", "actor": 1},
        ]
    )
    for actor in (2, 3):
        _append_junk_turn(events, pool, actor)
    events.extend([{"type": "tsumo", "actor": 0, "pai": "1s"}, {"type": "kakan", "actor": 0, "pai": "1s"}])
    base = HandEvaluator([convert.mjai_to_tid(value) for value in READY]).calc(
        reserved[0],
        [convert.mjai_to_tid("C")],
        Conditions(riichi=True, ippatsu=True, chankan=True, player_wind=1),
        [reserved[1]],
    )
    event = _single_win_event(1, 0, "1s", base, points=[25000, 24000, 25000, 25000])
    event["deltas"][1] += 1000
    event["scores"][1] += 1000
    event["ura_markers"] = ["9m"]
    events.extend([event, {"type": "end_kyoku"}])
    report = verify_settlements(events)
    assert report["status"] == "PASS"
    condition = report["settlements"][0]["derived_wins"][0]["conditions"]
    assert condition["chankan"] is True
    assert condition["ippatsu"] is True
    assert condition["double_riichi"] is False


@pytest.mark.parametrize("kokushi", [True, False])
def test_only_kokushi_can_rob_a_concealed_kan(kokushi):
    hand = ["9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "C"] if kokushi else READY
    tile = "1m" if kokushi else "1s"
    hands, _, _ = _specific_layout({1: hand, 0: [tile] * 3}, (tile,))
    events = _start(hands) + [
        {"type": "tsumo", "actor": 0, "pai": tile},
        {"type": "ankan", "actor": 0, "consumed": [tile] * 4},
        {
            "type": "hora",
            "actor": 1,
            "target": 0,
            "pai": tile,
            "deltas": [-32000, 32000, 0, 0],
            "scores": [-7000, 57000, 25000, 25000],
        },
        {"type": "end_kyoku"},
    ]
    report = verify_settlements(events)
    assert report["status"] == ("PASS" if kokushi else "FAIL")
    if kokushi:
        assert report["settlements"][0]["derived_wins"][0]["computed_yaku"] == [42]
    else:
        assert "Only kokushi" in report["settlements"][0]["reason"]


def test_ron_on_a_riichi_declaration_discard_does_not_charge_the_deposit():
    target_hand = ["1s", "2p", "3p", "4m", "5m", "6m", "7m", "8m", "9m", "E", "E", "E", "P"]
    hands, pool, reserved = _specific_layout({0: target_hand, 1: READY}, ("1p",))
    events = _start(hands)
    for actor in range(4):
        _append_junk_turn(events, pool, actor, first=actor == 0)
    events.extend(
        [
            {"type": "tsumo", "actor": 0, "pai": convert.tid_to_mjai(reserved[0])},
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "1s", "tsumogiri": False},
            {
                "type": "hora",
                "actor": 1,
                "target": 0,
                "pai": "1s",
                "deltas": [-1000, 1000, 0, 0],
                "scores": [24000, 26000, 25000, 25000],
            },
            {"type": "end_kyoku"},
        ]
    )
    report = verify_settlements(events)
    assert report["status"] == "PASS"
    assert report["computed_final_scores"] == (24000, 26000, 25000, 25000)
    assert report["computed_riichi_sticks"] == 0


def test_nagashi_eligibility_and_payments_do_not_take_the_riichi_pot():
    hands = [
        ["1m", "2m", "9m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "S"],
        ["4m", "5m", "1m", "4p", "5p", "6p", "4s", "5s", "6s", "S", "S", "S", "W"],
        ["7m", "8m", "9m", "7p", "8p", "9p", "7s", "8s", "9s", "W", "W", "W", "N"],
        ["2m", "3m", "7m", "2p", "3p", "4p", "2s", "3s", "4s", "N", "N", "N", "E"],
    ]
    pool = list(range(136))
    _allocate(pool, "C")
    for hand in hands:
        for value in hand:
            _allocate(pool, value)
    terminal = [tile for tile in pool if tile // 4 >= 27 or tile // 4 % 9 in {0, 8}]
    simple = [tile for tile in pool if tile not in terminal]
    events = _start(hands)
    for number in range(70):
        actor = number % 4
        tile = convert.tid_to_mjai(terminal.pop() if actor == 1 else simple.pop())
        events.append({"type": "tsumo", "actor": actor, "pai": tile})
        if number == 2:
            events.append({"type": "reach", "actor": 2})
        events.append({"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": number != 0})
        if number == 2:
            events.append({"type": "reach_accepted", "actor": 2})
    events.extend(
        [
            {
                "type": "ryukyoku",
                "reason": "nagashi_mangan",
                "deltas": [-4000, 8000, -2000, -2000],
                "scores": [21000, 33000, 22000, 23000],
            },
            {"type": "end_kyoku"},
        ]
    )
    report = verify_settlements(events)
    assert report["status"] == "PASS"
    assert report["settlements"][0]["nagashi_winners"] == [1]
    assert report["computed_riichi_sticks"] == 1
    assert report["computed_final_scores"] == (21000, 33000, 22000, 23000)
