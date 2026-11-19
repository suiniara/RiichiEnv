"""Exercise incremental visible history through the frozen FP32 predictor."""

# ruff: noqa: E402 -- Torch is optional until ML tests are collected
import json
from dataclasses import replace

import pytest

pytest.importorskip("torch")

from riichienv import Phase
from riichienv.kar import tokenize
from riichienv.kar.actions import execute_joint_responses, legal_candidates
from riichienv.kar.contracts import canonical_bytes
from riichienv.kar.learning import inference
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.model import KARTransformer
from tests.kar.test_decision_time_parity import _Episode


@pytest.fixture
def encodes(monkeypatch):
    model = KARTransformer(ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="history-parity"))
    cached = inference.FrozenModelPredictor(model, "a" * 64)
    cold = inference.FrozenModelPredictor(model, "a" * 64)
    starts = []
    real_history = tokenize._history
    real_prefix = inference.encode_prefix

    def track(snapshot, **kwargs):
        starts.append(kwargs["start_index"])
        return real_history(snapshot, **kwargs)

    def reference(snapshot, candidates, **_kwargs):
        return real_prefix(snapshot, candidates)

    def both(snapshot, ids, context):
        ids = tuple(ids)
        discards = tuple(action for action in ids if action < 148)[:2]
        assert discards
        expected_weights = cached.binding.model_weights_sha256
        with monkeypatch.context() as patch:
            patch.setattr(tokenize, "_history", track)
            actual_policy = cached.policy(snapshot, ids, context)
            actual_scores = cached.candidates(snapshot, ids, discards, context)
        with monkeypatch.context() as patch:
            patch.setattr(inference, "encode_prefix", reference)
            expected_policy = cold.policy(snapshot, ids, context)
            expected_scores = cold.candidates(snapshot, ids, discards, context)
        assert len(starts) and cached._prefix is not None and cold._prefix is not None
        assert cached._prefix.to_bytes() == cold._prefix.to_bytes()
        assert cached._prefix.to_record() == cold._prefix.to_record()
        assert actual_policy == expected_policy and actual_scores == expected_scores
        assert inference.model_weights_sha256(model) == expected_weights
        return starts[-1]

    return cached, both


def _own(ep):
    assert ep.env.phase == Phase.WaitAct
    seat = ep.env.current_player
    snapshot = ep.live.snapshot(seat)
    ids = tuple(item.global_id for item in legal_candidates(ep.obs[seat], source_player=snapshot.last_discard_actor))
    return snapshot, ids


def test_frozen_predictor_processes_only_each_observers_new_events(encodes):
    _, both = encodes
    ep = _Episode({}, ("E",))
    prior = {}
    for turn in range(13):
        ep.resolve()
        view, ids = _own(ep)
        seat = view.observer_player_id
        start = both(view, ids, f"mixed-observers/{turn}")
        assert start == prior.get(seat, 0)
        prior[seat] = len(view.events)
        ep.act()
    assert set(prior) == set(range(4))


def test_history_cache_crosses_round_but_not_an_identical_new_match(encodes):
    predictor, both = encodes
    ep = _Episode({}, ("E",))
    first, first_ids = _own(ep)
    assert both(first, first_ids, "first-round") == 0
    ep.finish_round()
    for _ in range(5):
        ep.resolve()
        view, ids = _own(ep)
        if view.observer_player_id == first.observer_player_id:
            break
        ep.act()
    else:
        pytest.fail("No choice for the original observer in the next round")
    assert ep.live.round_instance == 1
    assert both(view, ids, "next-round") == len(first.events)
    # A repeated seed/opening has identical model-visible bytes, but is not
    # the same half-game. Neither the token cache nor the K/V can cross it.
    assert both(first, first_ids, "same-context-for-restart") == 0
    restarted, restarted_ids = _own(_Episode({}, ("E",)))
    assert restarted.to_bytes() == first.to_bytes() and restarted_ids == first_ids
    with pytest.raises(ValueError, match="another observer/decision"):
        predictor.candidates(
            restarted, restarted_ids, tuple(a for a in first_ids if a < 148)[:1], "same-context-for-restart"
        )
    assert both(restarted, restarted_ids, "same-context-for-restart") == 0


def test_late_self_pass_changed_prefix_and_rewind_rebuild_in_real_predictor(encodes):
    _, both = encodes
    ep = _Episode({1: ("E", "E")}, ("E",))
    ep.act(tile="E")
    assert 1 in ep.env.active_players
    choices = {seat: ep._choose(seat, "PASS")[0] for seat in ep.env.active_players}
    receipts = tuple(receipt for receipt in ep.live.capture_self_passes(choices) if receipt.observer_player_id == 1)
    assert len(receipts) == 1
    raw = len(ep.env.rule_log)
    ep.obs = execute_joint_responses(ep.env, choices)
    ep.live.observe_native(list(ep.env.rule_log)[raw:])  # original batch has no receipt yet
    first, first_ids = _own(ep)
    assert first.observer_player_id == 1
    assert both(first, first_ids, "before-more-events") == 0
    for _ in range(4):
        ep.act()
    ep.resolve()
    before, ids = _own(ep)
    assert before.observer_player_id == 1
    assert both(before, ids, "before-receipt") == len(first.events)
    ep.live.observe_native([], confirmed_self_passes=receipts)
    after, _ = _own(ep)
    markers = [json.loads(raw) for raw in after.events if json.loads(raw)["type"] == "self_pass"]
    marker = next(row for row in markers if row["trigger_event_id"] == receipts[0].trigger_event_id)
    assert marker["confirmed_at_event_id"] < before.cutoff_event_id
    assert after.events[:-1] != before.events and len(after.events) == len(before.events) + 1
    assert both(after, ids, "late-receipt") == 0
    altered = list(after.events)
    index = next(i for i, raw in enumerate(altered) if json.loads(raw)["type"] == "dahai")
    event = json.loads(altered[index])
    event["tsumogiri"] = not event["tsumogiri"]
    altered[index] = canonical_bytes(event).decode()
    assert both(replace(after, events=tuple(altered)), ids, "edited-prefix") == 0
    assert both(first, first_ids, "rewind") == 0
