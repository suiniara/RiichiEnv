import pytest
from test_kan_preview import _ankan_env
from test_observation_response import round_events

from riichienv import Phase, convert
from riichienv.kar.actions import execute_candidate, execute_joint_responses, legal_candidates
from riichienv.kar.kan_transactions import pending_kan_commits
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.tokenize import _history


@pytest.mark.parametrize(
    "event,committed,indicators,expected",
    [
        ({"type": "dora"}, 1, 1, False),
        ({"type": "dora"}, 1, 2, True),
        ({"type": "tsumo", "actor": 2}, 1, 1, True),
        ({"type": "kan_commit", "actor": 2}, 1, 1, True),
    ],
)
def test_only_public_evidence_for_the_current_kan_commits(event, committed, indicators, expected):
    assert pending_kan_commits(event, actor=2, committed_kans=committed, indicators=indicators) == expected


def test_old_deferred_dora_does_not_accept_a_new_kakan_in_snapshot_or_tokens():
    hand = ["1m", "2m", "3m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "C"]
    events = round_events(hand) + [
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False},
        {"type": "pon", "actor": 2, "target": 1, "pai": "4m", "consumed": ["4m", "4m"]},
        {"type": "dahai", "actor": 3, "pai": "9p", "tsumogiri": False},
        {"type": "daiminkan", "actor": 2, "target": 3, "pai": "9p", "consumed": ["9p"] * 3},
        {"type": "tsumo", "actor": 2, "pai": "4m"},
        {"type": "kakan", "actor": 2, "pai": "4m"},
        {"type": "dora", "dora_marker": "2s"},
    ]
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    for event in events:
        ledger.apply_event(event)
    view = ledger.snapshot()
    assert view.pending_kan_actor == 2 and view.decision_phase == "kan_response"
    assert view.melds[2][0].kind == "pon"
    assert not any(row["kind"] == 21 for row in _history(view))
    ledger.apply_event({"type": "tsumo", "actor": 2, "pai": "1s"})
    assert ledger.snapshot().pending_kan_actor is None
    assert ledger.snapshot().melds[2][0].kind == "kakan"
    assert sum(row["kind"] == 21 for row in _history(ledger.snapshot())) == 1


@pytest.mark.parametrize("ron", [False, True])
def test_robbable_ankan_uses_validated_declaration_and_late_acknowledgement(ron):
    env, declaration = _ankan_env()
    projection = LiveEventProjection()
    projection.observe_native(list(env.mjai_log))
    mapped = next(
        value
        for value in legal_candidates(env.get_observations([0])[0])
        if value.spec.action_type == "ANKAN" and value.actions[0].tile == declaration.tile
    )
    before = len(env.mjai_log)
    responses = execute_candidate(env, 0, mapped)
    delta = list(env.mjai_log[before:])
    assert env.phase == Phase.WaitResponse and not delta
    projected = {
        "type": "ankan",
        "actor": 0,
        "consumed": [convert.tid_to_mjai(tile) for tile in mapped.actions[0].consume_tiles],
    }
    projection.observe_native(delta, declared_ankan=projected, pending_kan_window=True)
    view = projection.snapshot(1)
    assert view.pending_kan_actor == 0 and view.pending_kan_kind == "ankan" and view.decision_phase == "kan_response"
    choices = {}
    for seat, observation in responses.items():
        options = legal_candidates(observation, source_player=view.pending_kan_actor)
        choices[seat] = next(value for value in options if value.spec.action_type == ("RON" if ron else "PASS"))
    before = len(env.mjai_log)
    execute_joint_responses(env, choices)
    projection.observe_native(list(env.mjai_log[before:]))
    assert sum(event["type"] == "ankan" for event in projection.events) == 1
    if ron:
        assert not projection.snapshot(0).melds[0]
        assert any(event["type"] == "hora" and event["target"] == 0 for event in projection.events)
    else:
        assert projection.snapshot(0).pending_kan_actor is None
        assert projection.snapshot(0).melds[0][0].kind == "ankan"
        assert sum(event["type"] == "kan_commit" for event in projection.events) == 1
        rows = _history(projection.snapshot(1))
        assert sum(row["kind"] == 18 for row in rows) == 1 and sum(row["kind"] == 19 for row in rows) == 1
    assert all("source_record_index" in event for event in projection.events)


def test_empty_native_stream_without_a_validated_kan_window_is_not_accepted():
    projection = LiveEventProjection()
    declaration = {"type": "ankan", "actor": 0, "consumed": ["9p"] * 4}
    with pytest.raises(ValueError):
        projection.observe_native([], declared_ankan=declaration, pending_kan_window=False)
