"""Bounded native Pass submission and public stream segmentation regressions."""
import json
from dataclasses import replace

import pytest

from riichienv import Phase, convert
from riichienv.kar.actions import execute_candidate, execute_joint_responses, legal_candidates
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.tokenize import TokenHistoryCache, encode_prefix
from tests.kar.test_self_pass_history import _fixed_native, _markers

KOKUSHI_MISSING_1M = ("9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "C")


def _pass_choices(responses, actor):
    return {seat: next(item for item in legal_candidates(obs, source_player=actor) if item.global_id == 221)
            for seat, obs in responses.items() if obs.legal_actions()}


def _open_robbable_ankan(env, actor, waiter, projections):
    kan = next(item for item in legal_candidates(env.get_observation(actor), drawn_tile=env.drawn_tile)
               if item.spec.action_type == "ANKAN" and item.spec.base34 == 0)
    declare = {"type": "ankan", "actor": actor,
               "consumed": [convert.tid_to_mjai(tile) for tile in kan.actions[0].consume_tiles]}
    begin = len(env.mjai_log)
    responses = execute_candidate(env, actor, kan)
    assert env.phase == Phase.WaitResponse and not env.mjai_log[begin:]
    assert any(item.spec.action_type == "RON" for item in legal_candidates(responses[waiter], source_player=actor))
    for projection in projections:
        projection.observe_native([], declared_ankan=declare, pending_kan_window=True)
    return _pass_choices(responses, actor)


def _native_pass_resolution(env, choices):
    begin = len(env.mjai_log)
    execute_joint_responses(env, choices)
    raw = list(env.mjai_log[begin:])
    assert [event["type"] for event in raw] == ["ankan", "dora", "tsumo"]
    return raw


@pytest.mark.parametrize("receipt_only", [False, True], ids=["with-ankan", "empty-callback"])
def test_successful_ankan_pass_matches_whole_batch_at_every_final_field(receipt_only, record_property):  # noqa: PLR0915
    env, _, whole = _fixed_native({0: ["1m"] * 4, 1: KOKUSHI_MISSING_1M}, "6s")
    split = LiveEventProjection()
    split.observe_native(env.mjai_log)
    choices = _open_robbable_ankan(env, 0, 1, (whole, split))
    old = split.snapshot(1)
    ids = tuple(item.global_id for item in legal_candidates(env.get_observation(1), source_player=0))
    cache = TokenHistoryCache()
    old_prefix = encode_prefix(old, ids, history_cache=cache)
    old_bytes = old_prefix.to_bytes()
    receipt = next(item for item in split.capture_self_passes(choices) if item.observer_player_id == 1)
    assert receipt in whole.capture_self_passes(choices)
    assert not split._pending_pass_receipts and not _markers(split.snapshot(1))
    raw = _native_pass_resolution(env, choices)  # only now may a receipt be sent
    whole.observe_native(raw, confirmed_self_passes=(receipt,))
    record_property("raw_event_types", json.dumps([event["type"] for event in raw]))
    record_property("whole_marker_count", len(_markers(whole.snapshot(1))))

    # Collect any first-callback failure as reproduction evidence, but the
    # permanent assertion below requires both no exception and full parity.
    first = [] if receipt_only else raw[:1]
    issue = None
    try:
        split.observe_native(first, confirmed_self_passes=(receipt,))
    except ValueError as error:
        issue = str(error)
    queued = len(split._pending_pass_receipts)
    record_property("first_callback_error", issue or "none")
    record_property("queue_after_first_callback", queued)
    assert not _markers(split.snapshot(1))
    if receipt_only:
        split.observe_native(raw[:1])
    after_ack = split.snapshot(1)
    split.observe_native(raw[1:2])
    after_dora = split.snapshot(1)
    assert not _markers(after_ack) and not _markers(after_dora)
    record_property("queue_after_dora", len(split._pending_pass_receipts))
    split.observe_native(raw[2:])
    actual = split.snapshot(1)
    expected = whole.snapshot(1)
    final = encode_prefix(actual, ids, history_cache=cache)
    equal = final.to_bytes() == encode_prefix(expected, ids).to_bytes()
    record_property("split_final_marker_count", len(_markers(actual)))
    record_property("final_prefix_identical", equal)
    assert issue is None, f"{issue}; queued={queued}; final markers={len(_markers(actual))}; prefix equal={equal}"
    assert queued == 1 and not split._pending_pass_receipts
    assert equal and final.to_bytes() == encode_prefix(actual, ids).to_bytes()
    assert actual.events == expected.events and actual.history_positions == expected.history_positions
    assert split.events == whole.events and all(event["type"] != "self_pass" for event in split.events)
    (marker,) = _markers(actual)
    assert marker["confirmed_at_event_id"] == actual.cutoff_event_id == len(split.events) - 1
    assert split.events[marker["confirmed_at_event_id"]]["type"] == "tsumo"
    assert marker["provenance"] == "native_committed_pass"
    assert actual.history_positions[-1] == actual.history_positions[-2] == split.ledgers[1].logical_clock
    assert split.ledgers[1].logical_clock == split.ledgers[0].logical_clock
    assert len([row for row in final.rows if row["kind"] == 26]) == 1
    assert all(split.snapshot(seat) == whole.snapshot(seat) and not _markers(split.snapshot(seat))
               for seat in (0, 2, 3))
    assert old_prefix.to_bytes() == old_bytes == encode_prefix(old, ids).to_bytes()
    assert not _markers(old) and not _markers(after_ack) and not _markers(after_dora)
    split.observe_native([], confirmed_self_passes=(receipt, receipt))
    assert split.snapshot(1) == actual and not split._pending_pass_receipts


def _two_receipts():
    env, observations, projection = _fixed_native(
        {0: ["E"], 1: ["1m"] * 4, 2: KOKUSHI_MISSING_1M, 3: ["E"] * 2}, "6s"
    )
    discard = next(item for item in legal_candidates(observations[0], drawn_tile=env.drawn_tile)
                   if item.spec.base34 == 27 and item.spec.riichi is False and item.spec.tsumogiri is False)
    begin = len(env.mjai_log)
    responses = execute_candidate(env, 0, discard)
    projection.observe_native(env.mjai_log[begin:])
    assert any(item.spec.action_type == "PON" for item in legal_candidates(responses[3], source_player=0))
    choices = _pass_choices(responses, 0)
    ready = next(item for item in projection.capture_self_passes(choices) if item.observer_player_id == 3)
    begin = len(env.mjai_log)
    execute_joint_responses(env, choices)
    projection.observe_native(env.mjai_log[begin:])
    assert projection.events[-1]["type"] == "tsumo" and env.current_player == 1
    choices = _open_robbable_ankan(env, 1, 2, (projection,))
    waiting = next(item for item in projection.capture_self_passes(choices) if item.observer_player_id == 2)
    return projection, ready, waiting, _native_pass_resolution(env, choices)


@pytest.mark.parametrize("ack_failure", [False, True], ids=["mixed-readiness", "write-error"])
def test_other_valid_receipts_survive_waiting_or_a_recording_error(monkeypatch, ack_failure):
    projection, ready, waiting, raw = _two_receipts()
    receipts = waiting, ready  # an unready first receipt must not block the ready second receipt
    if ack_failure:
        def fail(*_args, **_kwargs):
            raise ValueError("induced recording failure, not a missing anchor")

        with monkeypatch.context() as scoped:
            scoped.setattr(projection.ledgers[3], "record_self_pass", fail)
            with pytest.raises(ValueError, match="induced recording failure"):
                projection.observe_native(raw[:1], confirmed_self_passes=receipts)
            assert set(projection._pending_pass_receipts) == set(receipts)
        projection.observe_native([])
    else:
        projection.observe_native(raw[:1], confirmed_self_passes=receipts)
    assert tuple(projection._pending_pass_receipts) == (waiting,)
    assert not _markers(projection.snapshot(2))
    (first_marker,) = _markers(projection.snapshot(3))
    assert first_marker["trigger_event_id"] == ready.trigger_event_id
    projection.observe_native(raw[1:])
    assert not projection._pending_pass_receipts
    (second_marker,) = _markers(projection.snapshot(2))
    assert second_marker["trigger_event_id"] == waiting.trigger_event_id
    assert _markers(projection.snapshot(3)) == [first_marker]


@pytest.mark.parametrize("change", [{"pai": "9m"}, {"trigger_event_id": 999}, {"observer_player_id": 1}])
def test_illegal_incoming_receipt_is_not_waiting_and_cannot_poison_pending_queue(change):
    projection, ready, waiting, raw = _two_receipts()
    projection.observe_native(raw[:1], confirmed_self_passes=(waiting,))
    old_cursor = projection.raw_cursor
    with pytest.raises(ValueError, match="Foreign or unknown private Pass trigger"):
        projection.observe_native(raw[1:2], confirmed_self_passes=(replace(waiting, **change),))
    assert projection.raw_cursor == old_cursor and tuple(projection._pending_pass_receipts) == (waiting,)
    projection.observe_native(raw[1:], confirmed_self_passes=(ready,))
    assert not projection._pending_pass_receipts
    assert [len(_markers(projection.snapshot(seat))) for seat in range(4)] == [0, 0, 1, 1]


def test_readiness_does_not_hide_bad_provenance_or_a_window_that_ended_without_an_anchor():
    projection, _, waiting, _ = _two_receipts()
    owner = projection.ledgers[waiting.observer_player_id]
    assert not owner.self_pass_ready(waiting, provenance="native_committed_pass")
    with pytest.raises(ValueError, match="supported provenance"):
        owner.self_pass_ready(waiting, provenance="guessed_hidden_intent")
    with pytest.raises(ValueError, match="cannot be inferred"):
        owner.self_pass_ready(waiting, provenance="inferred_unique_next_draw_pass")
    # Deliberately malformed terminal sequence, not a legal native resolution.
    owner.apply_event({"type": "end_kyoku"})
    with pytest.raises(ValueError, match="ended without a legal closing event"):
        owner.self_pass_ready(waiting, provenance="native_committed_pass")
    assert not _markers(owner.snapshot())
