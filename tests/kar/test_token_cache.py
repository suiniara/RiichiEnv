import copy
import hashlib
import json
from array import array
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from riichienv import ActionType
from riichienv.kar import tokenize
from riichienv.kar.actions import legal_candidates
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.mjsoul import decode_record_bytes, load_protocol
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.schema import TOKEN_BYTE_SIZE
from riichienv.kar.tokenize import (
    TokenHistoryCache,
    TokenizationUnavailableError,
    TokenSequence,
    append_action,
    encode_prefix,
)


def _initial(seed=42, observer=0):
    env = MJSOUL_RULESET.create_env(seed=seed)
    ledger = ObserverLedger(observer, (0, 1, 2, 3))
    for event in env.mjai_log:
        ledger.apply_event(event)
    observation = env.get_observations([observer])[observer]
    candidates = [candidate.global_id for candidate in legal_candidates(observation, drawn_tile=env.drawn_tile)]
    return env, ledger, candidates


def _snapshots(seed=42, steps=36):
    env = MJSOUL_RULESET.create_env(seed=seed)
    ledgers = [ObserverLedger(seat, (0, 1, 2, 3)) for seat in range(4)]
    offset = 0
    result = []
    for _ in range(steps):
        for event in env.mjai_log[offset:]:
            for ledger in ledgers:
                ledger.apply_event(event)
        offset = len(env.mjai_log)
        observations = env.get_observations(list(env.active_players))
        for actor, observation in observations.items():
            native = observation.legal_actions()
            if not native:
                continue
            snapshot = ledgers[actor].snapshot()
            source = (
                snapshot.pending_kan_actor if snapshot.pending_kan_actor is not None else snapshot.last_discard_actor
            )
            candidates = [
                candidate.global_id
                for candidate in legal_candidates(observation, drawn_tile=env.drawn_tile, source_player=source)
            ]
            result.append((snapshot, candidates))
        if env.done():
            break
        actions = {}
        for actor, observation in observations.items():
            native = observation.legal_actions()
            actions[actor] = next(
                (a for a in native if a.action_type == ActionType.PASS),
                next((a for a in native if a.action_type == ActionType.DISCARD), native[0]),
            )
        env.step(actions)
    return result


@pytest.mark.parametrize(
    "seed,expected_prefix,expected_action",
    [
        (
            0,
            "82ded4262f3eb210ab98f13c34d2fbb4803c96b447f1785dfa0071736852057f",
            "68bcbd0d9e947af7b574d5b83a9b6c66ce60aa8c34433cc248ed22994cf70b63",
        ),
        (
            42,
            "6852fe036cd33a166844234f531f2b195f9d04412b749928c227c3432215fdee",
            "6100f3d29395365bb6dd93ead314f18d1823f5a1955881a3b4f99ccb67cc5b56",
        ),
    ],
)
def test_pre_cache_fixed_binary_fingerprints_are_unchanged(seed, expected_prefix, expected_action):
    _, ledger, candidates = _initial(seed)
    sequence = encode_prefix(ledger.snapshot(), candidates, history_cache=TokenHistoryCache())
    assert hashlib.sha256(sequence.to_bytes()).hexdigest() == expected_prefix
    assert hashlib.sha256(append_action(sequence, candidates[0]).to_bytes()).hexdigest() == expected_action
    assert len(sequence.to_bytes()) == 44 + 64 + len(sequence.rows) * TOKEN_BYTE_SIZE


def test_incremental_and_fresh_prefixes_match_for_multiple_observers_and_decisions():
    caches = [TokenHistoryCache() for _ in range(4)]
    previous_sequences = []
    stream = _snapshots()
    assert len(stream) >= 20
    for snapshot, candidates in stream:
        sequence = encode_prefix(snapshot, candidates, history_cache=caches[snapshot.observer_player_id])
        assert sequence.to_bytes() == encode_prefix(snapshot, candidates).to_bytes()
        assert sequence.to_record() == encode_prefix(snapshot, candidates).to_record()
        assert caches[snapshot.observer_player_id].processed_events == len(snapshot.events)
        assert caches[snapshot.observer_player_id].history_rows == sequence.history_rows
        previous_sequences.append((sequence, sequence.to_bytes()))
    assert all(sequence.to_bytes() == original for sequence, original in previous_sequences)


def test_history_is_parsed_and_encoded_once_and_to_bytes_never_reencodes(monkeypatch):
    _, ledger, candidates = _initial()
    cache = TokenHistoryCache()
    snapshot = ledger.snapshot()
    calls = []
    original_encode = tokenize.encode_token
    original_loads = tokenize.json.loads
    parsed = []

    def count_encode(row):
        calls.append(row["block"])
        return original_encode(row)

    def count_loads(value):
        parsed.append(value)
        return original_loads(value)

    monkeypatch.setattr(tokenize, "encode_token", count_encode)
    monkeypatch.setattr(tokenize.json, "loads", count_loads)
    sequence = encode_prefix(snapshot, candidates, history_cache=cache)
    first_count = len(calls)
    assert calls.count(0) == sequence.history_rows + 1
    assert len(parsed) == len(snapshot.events)
    sequence.to_bytes()
    sequence.to_bytes()
    assert len(calls) == first_count
    repeated = encode_prefix(snapshot, candidates, history_cache=cache)
    assert len(calls) - first_count == 63
    assert len(parsed) == len(snapshot.events)
    assert repeated.to_bytes() == sequence.to_bytes()
    previous = len(calls)
    for action in candidates[:2]:
        append_action(repeated, action).to_bytes()
    assert len(calls) - previous == min(2, len(candidates))


def test_only_new_visible_events_are_parsed_on_append(monkeypatch):
    frames = [frame for frame in _snapshots(steps=20) if frame[0].observer_player_id == 0]
    first, second = frames[0], frames[1]
    cache = TokenHistoryCache()
    encode_prefix(*first, history_cache=cache)
    original = tokenize.json.loads
    parsed = []

    def load(value):
        parsed.append(value)
        return original(value)

    monkeypatch.setattr(tokenize.json, "loads", load)
    sequence = encode_prefix(*second, history_cache=cache)
    assert parsed == list(second[0].events[len(first[0].events) :])
    monkeypatch.setattr(tokenize.json, "loads", original)
    assert sequence.to_bytes() == encode_prefix(*second).to_bytes()


def test_changed_old_public_event_resets_cache_instead_of_reusing_blob():
    _, ledger, candidates = _initial()
    cache = TokenHistoryCache()
    snapshot = ledger.snapshot()
    original = encode_prefix(snapshot, candidates, history_cache=cache).to_bytes()
    events = list(snapshot.events)
    index = next(i for i, value in enumerate(events) if json.loads(value)["type"] == "start_kyoku")
    event = json.loads(events[index])
    event["scores"][0] += 1000
    events[index] = json.dumps(event, sort_keys=True)
    changed = replace(snapshot, events=tuple(events))
    rebuilt = encode_prefix(changed, candidates, history_cache=cache)
    assert rebuilt.to_bytes() == encode_prefix(changed, candidates).to_bytes()
    assert rebuilt.to_bytes() != original
    assert encode_prefix(snapshot, candidates, history_cache=cache).to_bytes() == original


def test_shortened_prefix_observer_and_initial_order_cannot_share_stale_state():
    frames = _snapshots(steps=24)
    cache = TokenHistoryCache()
    for snapshot, candidates in reversed(frames):
        assert (
            encode_prefix(snapshot, candidates, history_cache=cache).to_bytes()
            == encode_prefix(snapshot, candidates).to_bytes()
        )
    snapshot, candidates = frames[0]
    changed = replace(snapshot, initial_seat_order_player_ids=(3, 2, 1, 0))
    assert (
        encode_prefix(changed, candidates, history_cache=cache).to_bytes()
        == encode_prefix(changed, candidates).to_bytes()
    )
    cache.reset()
    assert cache.processed_events == cache.history_rows == 0
    assert cache.schema_hash is None


def test_schema_binding_is_separate_and_existing_sequences_keep_their_original_hash(monkeypatch):
    _, ledger, candidates = _initial()
    snapshot = ledger.snapshot()
    cache = TokenHistoryCache()
    original = encode_prefix(snapshot, candidates, history_cache=cache)
    original_bytes = original.to_bytes()
    monkeypatch.setattr(tokenize, "_active_hash", lambda: "1" * 64)
    newer = encode_prefix(snapshot, candidates, history_cache=cache)
    assert cache.schema_hash == newer.schema_hash == "1" * 64
    assert newer.to_bytes() == encode_prefix(snapshot, candidates).to_bytes()
    assert original.to_bytes() == original_bytes
    assert newer.to_bytes() != original_bytes
    with pytest.raises(ValueError, match="schema"):
        append_action(original, candidates[0])


def test_schema_identity_change_cannot_reuse_history_rows(monkeypatch):
    _, ledger, candidates = _initial()
    snapshot = ledger.snapshot()
    cache = TokenHistoryCache()
    first = encode_prefix(snapshot, candidates, history_cache=cache)
    monkeypatch.setattr(tokenize, "ACTIVE_SCHEMA_ID", "different-schema-identity")
    second = encode_prefix(snapshot, candidates, history_cache=cache)
    assert second.schema_id == "different-schema-identity"
    assert second.rows[0] is not first.rows[0]
    assert second.to_bytes() == encode_prefix(snapshot, candidates).to_bytes()
    with pytest.raises(ValueError, match="schema"):
        append_action(first, candidates[0])


def test_candidate_sets_and_action_suffixes_do_not_contaminate_history():
    _, ledger, candidates = _initial()
    cache = TokenHistoryCache()
    snapshot = ledger.snapshot()
    prefix = encode_prefix(snapshot, candidates, history_cache=cache)
    original = prefix.to_bytes()
    append_action(prefix, candidates[0])
    changed = encode_prefix(snapshot, candidates[:1], history_cache=cache)
    assert changed.to_bytes() == encode_prefix(snapshot, candidates[:1]).to_bytes()
    assert changed.rows[:-1] != ()
    assert prefix.to_bytes() == original
    assert encode_prefix(snapshot, candidates, history_cache=cache).to_bytes() == original


def test_rows_and_nested_vectors_are_deeply_immutable_and_records_are_plain_copies():
    _, ledger, candidates = _initial()
    sequence = encode_prefix(ledger.snapshot(), candidates, history_cache=TokenHistoryCache())
    original = sequence.to_bytes()
    with pytest.raises(TypeError):
        sequence.rows[0]["response_target"] = 0
    state = sequence.rows[sequence.history_rows + 1]
    with pytest.raises(TypeError):
        state["numeric"][0] = 99
    with pytest.raises(FrozenInstanceError):
        sequence.rows = ()
    record = sequence.to_record()
    assert isinstance(record["rows"], list)
    assert isinstance(record["rows"][0], dict)
    assert isinstance(record["rows"][0]["tiles"], list)
    json.dumps(record, allow_nan=False)
    record["rows"][0]["tiles"][0] = 38
    assert sequence.to_bytes() == original


def test_public_sequence_constructor_copies_mutable_vectors_and_rejects_invalid_rows():
    _, ledger, candidates = _initial()
    prefix = encode_prefix(ledger.snapshot(), candidates)
    rows = prefix.to_record()["rows"]
    row = rows[prefix.history_rows + 1]
    row["numeric"] = array("d", row["numeric"])
    copy_sequence = TokenSequence(
        tuple(rows), prefix.candidates, prefix.history_rows, prefix.decide_index, prefix.logical_position
    )
    original = copy_sequence.to_bytes()
    row["numeric"][3] = 100
    rows[0]["actor"] = 4
    assert copy_sequence.to_bytes() == original
    broken = prefix.to_record()["rows"]
    broken[-1]["response_target"] = 0
    with pytest.raises(ValueError):
        TokenSequence(
            tuple(broken), prefix.candidates, prefix.history_rows, prefix.decide_index, prefix.logical_position
        )
    broken = prefix.to_record()["rows"]
    broken[-1]["flags"][0] = 2
    with pytest.raises(ValueError):
        TokenSequence(
            tuple(broken), prefix.candidates, prefix.history_rows, prefix.decide_index, prefix.logical_position
        )


def test_failure_drops_partial_history_state_and_visibility_is_never_silently_filled():
    _, ledger, candidates = _initial()
    snapshot = ledger.snapshot()
    cache = TokenHistoryCache()
    reference = encode_prefix(snapshot, candidates, history_cache=cache).to_bytes()
    bad = replace(
        snapshot,
        events=(*snapshot.events, '{"type":"unrecognized"}'),
        history_positions=(*snapshot.history_positions, snapshot.history_positions[-1] + 1),
    )
    with pytest.raises(ValueError):
        encode_prefix(bad, candidates, history_cache=cache)
    assert cache.processed_events == 0
    assert encode_prefix(snapshot, candidates, history_cache=cache).to_bytes() == reference
    for field in ("visibility_complete", "history_complete", "physical_ledger_complete"):
        with pytest.raises(TokenizationUnavailableError):
            encode_prefix(replace(snapshot, **{field: False}), candidates, history_cache=cache)


def test_history_clock_changes_rebuild_and_invalid_clock_is_rejected():
    _, ledger, candidates = _initial()
    snapshot = ledger.snapshot()
    cache = TokenHistoryCache()
    encode_prefix(snapshot, candidates, history_cache=cache)
    changed = replace(snapshot, history_positions=tuple(position * 2 for position in snapshot.history_positions))
    assert (
        encode_prefix(changed, candidates, history_cache=cache).to_bytes()
        == encode_prefix(changed, candidates).to_bytes()
    )
    for positions in ((1,), (3, 2, 1), (True, 2, 3)):
        with pytest.raises(ValueError):
            encode_prefix(replace(snapshot, history_positions=positions), candidates, history_cache=cache)


def test_hidden_and_future_source_fields_do_not_invalidate_public_history(monkeypatch):
    env, ledger, candidates = _initial()
    cache = TokenHistoryCache()
    original = encode_prefix(ledger.snapshot(), candidates, history_cache=cache).to_bytes()
    changed = ObserverLedger(0, (0, 1, 2, 3))
    for index, event in enumerate(copy.deepcopy(env.mjai_log)):
        if event["type"] == "start_kyoku":
            event["tehais"][1:] = [["C"] * 13, ["P"] * 13, ["F"] * 13]
            event["future_result"] = {"rank": 1}
        event.update(source_record_index=1000 + index * 10, source_record_sha256="raw hash must not be a token")
        changed.apply_event(event)
    original_loads = tokenize.json.loads
    parsed = []

    def load(value):
        parsed.append(value)
        return original_loads(value)

    monkeypatch.setattr(tokenize.json, "loads", load)
    assert encode_prefix(changed.snapshot(), candidates, history_cache=cache).to_bytes() == original
    assert parsed == []


def test_real_asset_incremental_history_matches_fresh_paths_when_available():
    root = Path(__file__).parents[3] / "kar-m0-assets"
    raw = root / "200331-00a872c6-afb1-487b-ba2d-e58b9379f60e.bin"
    if not raw.is_file():
        pytest.skip("Private raw asset is not distributed with the checkout")
    game = decode_record_bytes(
        raw.read_bytes(), load_protocol(root / "mahjongsoul.proto"), container="majsoul_download"
    )
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    cache = TokenHistoryCache()
    checked = 0
    decisions = 0
    for event in game.events:
        ledger.apply_event(event)
        snapshot = ledger.snapshot()
        if snapshot.decision_phase not in {"own_act", "discard_response", "kan_response"}:
            continue
        if not snapshot.history_complete or not snapshot.physical_ledger_complete or not snapshot.visibility_complete:
            continue
        decisions += 1
        if decisions % 23:
            continue
        candidates = (0, 4)
        assert (
            encode_prefix(snapshot, candidates, history_cache=cache).to_bytes()
            == encode_prefix(snapshot, candidates).to_bytes()
        )
        checked += 1
    assert checked >= 10
