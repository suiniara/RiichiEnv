"""Engine-owned observer facts retain the model's strict information boundary."""

import json

import pytest

from riichienv._riichienv import NativeObserverState
from riichienv.kar.observation import ObserverLedger


def opening():
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "oya": 0,
        "honba": 0,
        "scores": [25000] * 4,
        "dora_marker": "9s",
        "tehais": [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
            ["?"] * 13,
            ["?"] * 13,
            ["?"] * 13,
        ],
    }


def test_native_observer_does_not_retain_hidden_opponent_or_wall_information():
    a, b = NativeObserverState(0), NativeObserverState(0)
    first, changed = opening(), opening()
    changed["tehais"][1] = ["C"] * 13
    changed["paishan"] = "hidden-wall-sentinel"
    changed["ura_doras"] = ["5pr"]
    assert a.apply_event_json(json.dumps(first)) == b.apply_event_json(json.dumps(changed))
    assert a.apply_event_json(json.dumps({"type": "tsumo", "actor": 1, "pai": "5pr"})) == (
        b.apply_event_json(json.dumps({"type": "tsumo", "actor": 1, "pai": "E"}))
    )
    assert a.state_json() == b.state_json()
    assert "hidden-wall-sentinel" not in b.state_json()


@pytest.mark.parametrize(
    "bad_event",
    [
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "C", "tsumogiri": False},
        {"type": "kan_commit", "actor": 0, "kan_type": "ankan"},
    ],
)
def test_rejected_native_event_does_not_advance_facts_or_training_prefix(bad_event):
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    ledger.apply_event({"type": "start_game"})
    ledger.apply_event(opening())
    before = ledger.snapshot().to_bytes()
    with pytest.raises(ValueError):
        ledger.apply_event(bad_event)
    assert ledger.snapshot().to_bytes() == before


def test_python_cannot_override_native_observer_facts():
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    ledger.apply_event(opening())
    with pytest.raises(AttributeError, match="owned by the native observer"):
        ledger.remaining = 0
    with pytest.raises(AttributeError, match="owned by the native observer"):
        ledger.decision_phase = "closed"
    assert ledger.snapshot().remaining_draws == 70
    assert ledger.snapshot().decision_phase == "await_draw"
