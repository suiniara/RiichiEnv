"""Native terminal enrichment must not rewrite externally supplied MJAI."""

from copy import deepcopy

import pytest

from riichienv import Action, ActionType, GameRule, RiichiEnv
from tests.kar.test_decision_time_parity import _wall


@pytest.mark.parametrize("method", ["apply_event", "observe_event"])
@pytest.mark.parametrize("kind", ["tsumo", "kyushu"])
def test_raw_replay_preserves_terminal_fields_from_a_real_native_round(method, kind):
    if kind == "tsumo":
        hand = ("1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "E", "S")
        draw, action_type = "S", ActionType.TSUMO
    else:
        hand = ("1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C")
        draw, action_type = "2m", ActionType.KYUSHU_KYUHAI
    wall = _wall({0: hand}, (draw,))
    assert sorted(wall) == list(range(136))
    native = RiichiEnv("4p-red-single", seed=72, rule=GameRule.default_mjsoul())
    native.reset(wall=wall, oya=0)
    native.step({0: Action(action_type, native.drawn_tile if kind == "tsumo" else None, [], 0)})
    assert native.done()
    source = deepcopy(list(native.rule_log))
    terminal_kind = "hora" if kind == "tsumo" else "ryukyoku"
    terminal = next(event for event in source if event["type"] == terminal_kind)
    assert "scores" in terminal and terminal["visibility_complete"] is True
    legacy = next(event for event in native.mjai_log if event["type"] == terminal_kind)
    assert "scores" not in legacy and "visibility_complete" not in legacy
    assert source[-1]["type"] == "end_game" and "end_game_pot_deltas" in source[-1]

    replay = RiichiEnv("4p-red-single", seed=73, rule=GameRule.default_mjsoul())
    for event in source:
        event["caller_annotation"] = {"keep": True}
        # The provenance deliberately still says native: payload metadata must
        # never decide whether the public raw-event API strips supplied fields.
        if method == "apply_event":
            replay.apply_event(event)
        else:
            replay.observe_event(event, player_id=0)
        assert replay.mjai_log[-1] == event
        assert replay.rule_log[-1] == event
