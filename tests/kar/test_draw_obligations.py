from types import SimpleNamespace

import pytest

from riichienv import convert
from riichienv.kar.audit import replay_game
from riichienv.kar.settlement import verify_settlements


def _missing_rinshan_game():
    pool = list(range(136))

    def take(value):
        tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == value)
        pool.remove(tile)
        return tile

    take("C")
    take("6s")
    known = {
        0: ["5pr"],
        1: ["1m", "9m", "1p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "C"],
        2: ["5p", "5p", "5p", "9p"],
    }
    hands = [[take(value) for value in known.get(seat, [])] for seat in range(4)]
    for hand in hands:
        while len(hand) < 13:
            hand.append(pool.pop(0))
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
            "tehais": [[convert.tid_to_mjai(tile) for tile in hand] for hand in hands],
        },
        {"type": "tsumo", "actor": 0, "pai": "6s"},
        {"type": "dahai", "actor": 0, "pai": "5pr", "tsumogiri": False},
        {"type": "daiminkan", "actor": 2, "target": 0, "pai": "5pr", "consumed": ["5p"] * 3},
        {"type": "dahai", "actor": 2, "pai": "9p", "tsumogiri": False},
        {
            "type": "hora",
            "actor": 1,
            "target": 2,
            "pai": "9p",
            "deltas": [0, 32000, -32000, 0],
            "scores": [25000, 57000, -7000, 25000],
        },
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]


@pytest.mark.parametrize("component", ["native_audit", "independent_ledger"])
def test_missing_kan_replacement_draw_is_a_failure_even_if_later_money_matches(component):
    events = _missing_rinshan_game()
    result = replay_game(SimpleNamespace(events=events)) if component == "native_audit" else verify_settlements(events)
    assert result["status"] == "FAIL"
    assert result["errors"][-1]["event_id"] == 5
    assert "required draw" in result["errors"][-1].get("detail", result["errors"][-1].get("reason", ""))
    if component == "native_audit":
        assert result["events_applied"] == 5
