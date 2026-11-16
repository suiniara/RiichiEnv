from dataclasses import replace

import pytest

from riichienv import convert
from riichienv.kar.contracts import MJSOUL_RULESET, Response
from riichienv.kar.observation import ObserverLedger, project_observer
from riichienv.kar.response import resolve_discard_response
from riichienv.kar.response_mask import public_rule_response_mask, validate_response_target


def round_events(own=None):
    hand = own or ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
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
            "dora_marker": "9s",
            "tehais": [hand, ["1s"] * 13, ["2s"] * 13, ["3s"] * 13],
        },
    ]


def ledger(events):
    view = ObserverLedger(0, (0, 1, 2, 3))
    for event in events:
        view.apply_event(event)
    return view


def test_future_and_opponent_hidden_information_cannot_change_cutoff():
    original = round_events() + [{"type": "tsumo", "actor": 0, "pai": "5pr"}]
    changed = round_events() + [{"type": "tsumo", "actor": 0, "pai": "5pr"}, {"type": "hora", "actor": 2, "target": 0}]
    changed[1]["tehais"][1] = ["C"] * 13
    changed[1]["paishan"] = "never observable"
    changed[1]["ura_doras"] = ["1m"]
    a = project_observer(original, 0, 2, (0, 1, 2, 3))
    b = project_observer(changed, 0, 2, (0, 1, 2, 3))
    assert a.to_bytes() == b.to_bytes()
    assert b"never observable" not in b.to_bytes()


def test_opponent_draws_are_hidden():
    a = ledger(round_events() + [{"type": "tsumo", "actor": 1, "pai": "1m"}]).snapshot()
    b = ledger(round_events() + [{"type": "tsumo", "actor": 1, "pai": "5pr"}]).snapshot()
    assert a.to_bytes() == b.to_bytes()


def test_called_discard_keeps_history_but_moves_physical_region():
    events = round_events() + [
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False},
        {"type": "pon", "actor": 2, "target": 1, "pai": "5p", "consumed": ["5p", "5pr"]},
    ]
    state = ledger(events).snapshot()
    assert state.current_rivers[1] == ()
    assert len(state.discard_history[1]) == 1
    assert state.known_counts()[13] == 3


def test_kakan_is_only_committed_after_rob_window_and_never_creates_fifth_meld():
    events = round_events() + [
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False},
        {"type": "pon", "actor": 2, "target": 1, "pai": "5p", "consumed": ["5p", "5pr"]},
        {"type": "kakan", "actor": 2, "pai": "5p"},
    ]
    view = ledger(events)
    assert view.snapshot().melds[2][0].kind == "pon"
    assert view.snapshot().known_counts()[13] == 3
    view.apply_event({"type": "hora", "actor": 3, "target": 2})
    assert view.snapshot().melds[2][0].kind == "pon"
    committed = ledger(events)
    committed.apply_event({"type": "tsumo", "actor": 2, "pai": "1s"})
    assert len(committed.snapshot().melds[2]) == 1
    assert committed.snapshot().melds[2][0].kind == "kakan"
    assert committed.snapshot().known_counts()[13] == 4


def test_opponent_ankan_does_not_reveal_red_composition():
    a = ledger(round_events() + [{"type": "ankan", "actor": 2, "consumed": ["5p"] * 4}]).snapshot()
    b = ledger(round_events() + [{"type": "ankan", "actor": 2, "consumed": ["5pr", "5p", "5p", "5p"]}]).snapshot()
    assert a.to_bytes() == b.to_bytes()


def test_riichi_does_not_charge_before_acceptance():
    view = ledger(round_events())
    view.apply_event({"type": "reach", "actor": 0})
    view.apply_event({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False})
    assert view.snapshot().scores[0] == 25000
    view.apply_event({"type": "reach_accepted", "actor": 0})
    assert view.snapshot().scores[0] == 24000
    with pytest.raises(ValueError):
        view.apply_event({"type": "reach_accepted", "actor": 0})


@pytest.mark.parametrize("call,actor,expected", [("chi", 1, 1), ("pon", 2, 3), ("daiminkan", 3, 7)])
def test_accepted_call_closes_original_discard_window(call, actor, expected):
    events = [
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": call, "actor": actor, "target": 0, "pai": "1m"},
        {"type": "tsumo", "actor": actor, "pai": "2m"},
        {"type": "hora", "actor": actor, "target": actor},
    ]
    result = resolve_discard_response(events, 0)
    assert result.response_target == expected
    assert result.response_observed is True


def test_ron_waits_for_complete_winner_set_and_mjsoul_allows_triple_ron():
    events = [{"type": "dahai", "actor": 0, "pai": "1m"}]
    events += [{"type": "hora", "actor": p, "target": 0} for p in (1, 2, 3)]
    assert not resolve_discard_response(events, 0).response_observed
    events.append({"type": "end_kyoku"})
    assert resolve_discard_response(events, 0).response_target == Response.RON_MASK_7
    assert not resolve_discard_response(events, 0, replace(MJSOUL_RULESET, sanchaho_is_draw=True)).response_observed


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("exhaustive", 15),
        ("exhaustive_draw", 15),
        ("four_winds", 16),
        ("four_riichi", 17),
        ("four_kans", 18),
        ("nagashi_mangan", 20),
    ],
)
def test_direct_end_is_not_next_draw(reason, expected):
    result = resolve_discard_response(
        [{"type": "dahai", "actor": 0, "pai": "1m"}, {"type": "ryukyoku", "reason": reason}], 0
    )
    assert result.response_target == expected


def test_unknown_or_missing_response_is_not_fabricated():
    trigger = {"type": "dahai", "actor": 0, "pai": "1m"}
    for suffix in ([], [{"type": "end_kyoku"}], [{"type": "ryukyoku"}], [{"type": "ryukyoku", "reason": "triple_ron"}]):
        assert resolve_discard_response([trigger, *suffix], 0).response_observed is False
    assert resolve_discard_response([{"type": "ankan"}], 0).response_applicable is False


def test_public_mask_uses_visible_furiten_and_mjsoul_rule_only():
    view = ledger(round_events())
    view.apply_event({"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False})
    state = view.snapshot()
    allowed = public_rule_response_mask(state, 4)
    assert allowed[Response.ABORT_TRIPLE_RON] is False
    assert allowed[Response.RON_MASK_1] is False
    assert allowed[Response.RON_MASK_2] is True
    assert allowed[Response.RON_MASK_7] is False
    assert public_rule_response_mask(ledger(round_events()).snapshot(), 4)[Response.RON_MASK_7] is True
    with pytest.raises(ValueError):
        validate_response_target(allowed, Response.RON_MASK_1)
    assert not hasattr(state, "opponent_legal_actions")


def test_last_tile_masks_normal_draw_but_allows_end():
    state = replace(ledger(round_events()).snapshot(), remaining_draws=0)
    allowed = public_rule_response_mask(state, 4)
    assert allowed[Response.NEXT_DRAW] is False
    assert allowed[Response.EXHAUSTIVE_DRAW] is True
    assert allowed[5:8] == (False, False, False)
    with pytest.raises(ValueError):
        validate_response_target((False,) * 21, 0)
    with pytest.raises(ValueError):
        public_rule_response_mask(state, 219)


def test_unknown_history_cannot_imply_public_furiten():
    state = ledger(round_events()).snapshot()
    discards = ((0,), (0,), (), ())
    state = replace(state, history_complete=False, discard_history=discards)
    assert public_rule_response_mask(state, 4)[Response.RON_MASK_1] is True


def test_unknown_observer_hand_does_not_make_known_counts():
    state = ledger(round_events(["?"] * 13)).snapshot()
    assert state.physical_ledger_complete is False
    assert state.known_counts() is None
    assert convert.mjai_to_tid("5pr") // 4 == 13
