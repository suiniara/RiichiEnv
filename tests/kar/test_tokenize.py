import copy
import json
from dataclasses import replace

import pytest

from riichienv.kar.actions import legal_candidates
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.schema import TOKEN_BYTE_SIZE, decode_token, encode_token
from riichienv.kar.tokenize import append_action, encode_prefix


def _prefix(seed=42):
    env = MJSOUL_RULESET.create_env(seed=seed)
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    for event in env.mjai_log:
        ledger.apply_event(event)
    obs = env.get_observations([0])[0]
    candidates = legal_candidates(obs, drawn_tile=env.drawn_tile)
    return env, ledger, [candidate.global_id for candidate in candidates]


def test_prefix_layout_has_exactly_sixty_two_snapshot_slots():
    _, ledger, candidates = _prefix()
    sequence = encode_prefix(ledger.snapshot(), candidates)
    assert sequence.schema_id == "kar-riichienv-v1"
    assert len(sequence.rows) == sequence.history_rows + 64
    assert sequence.rows[0]["kind"] == 1
    assert sequence.rows[-1]["kind"] == 9
    state = sequence.rows[sequence.history_rows + 1 : -1]
    assert len(state) == 62
    assert [row["kind"] for row in state[:39]] == [2] + [3] * 4 + [4] * 34
    assert all(row["positions"] == sequence.logical_position for row in state)
    assert sequence.rows[-1]["positions"] == sequence.logical_position
    assert len(sequence.to_bytes()) == 44 + 64 + len(sequence.rows) * TOKEN_BYTE_SIZE


@pytest.mark.parametrize("seed", range(10))
def test_every_generated_token_round_trips_with_float32_normalization(seed):
    _, ledger, candidates = _prefix(seed)
    sequence = append_action(encode_prefix(ledger.snapshot(), candidates), candidates[0])
    for row in sequence.rows:
        encoded = encode_token(row)
        assert encode_token(decode_token(encoded)) == encoded
    assert len(sequence.rows) == sequence.history_rows + 65


def test_candidate_order_and_duplicates_do_not_change_prefix():
    _, ledger, candidates = _prefix()
    first = encode_prefix(ledger.snapshot(), candidates)
    reordered = encode_prefix(ledger.snapshot(), list(reversed(candidates)) + candidates[:1])
    assert first.to_bytes() == reordered.to_bytes()


def test_action_target_only_changes_the_action_suffix():
    _, ledger, candidates = _prefix()
    prefix = encode_prefix(ledger.snapshot(), candidates)
    assert len(candidates) >= 2
    first, second = append_action(prefix, candidates[0]), append_action(prefix, candidates[1])
    assert first.rows[:-1] == second.rows[:-1] == prefix.rows
    assert first.rows[-1]["action_ids"] != second.rows[-1]["action_ids"]
    assert first.rows[-1]["block"] == 3
    with pytest.raises(ValueError):
        append_action(first, candidates[0])
    with pytest.raises(ValueError):
        append_action(prefix, 546)


def test_hidden_initial_hands_and_other_draws_do_not_enter_tokens():
    env, ledger, candidates = _prefix()
    original = encode_prefix(ledger.snapshot(), candidates).to_bytes()
    altered = ObserverLedger(0, (0, 1, 2, 3))
    for event in copy.deepcopy(env.mjai_log):
        if event["type"] == "start_kyoku":
            event["tehais"][1:] = [["C"] * 13, ["P"] * 13, ["F"] * 13]
            event["paishan"] = "must not enter the model"
            event["final_rank"] = 1
        altered.apply_event(event)
    assert encode_prefix(altered.snapshot(), candidates).to_bytes() == original
    visible = json.dumps(encode_prefix(altered.snapshot(), candidates).to_record())
    assert "must not enter" not in visible
    assert "final_rank" not in visible


def test_source_record_id_gaps_and_hashes_do_not_change_the_visible_clock():
    env, _, candidates = _prefix()
    first, second = ObserverLedger(0, (0, 1, 2, 3)), ObserverLedger(0, (0, 1, 2, 3))
    for index, event in enumerate(env.mjai_log):
        a, b = copy.deepcopy(event), copy.deepcopy(event)
        a.update(source_record_index=index, source_record_sha256="a" * 64)
        b.update(source_record_index=1000 + 10 * index, source_record_sha256="b" * 64)
        first.apply_event(a)
        second.apply_event(b)
    assert (
        encode_prefix(first.snapshot(), candidates).to_bytes()
        == encode_prefix(second.snapshot(), candidates).to_bytes()
    )


def test_one_public_result_subrows_share_a_logical_time():
    env, _, candidates = _prefix()
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    for index, event in enumerate(env.mjai_log):
        ledger.apply_event({**event, "source_record_index": index})
    sequence = encode_prefix(ledger.snapshot(), candidates)
    initial = [row for row in sequence.rows if row["kind"] == 11 and row["stage"] == 2]
    assert len(initial) == 3
    assert len({row["positions"] for row in initial}) == 1
    assert sum(code != 0 for row in initial for code in row["tiles"]) == 13


def test_missing_visibility_or_private_state_is_not_silently_filled():
    _, ledger, candidates = _prefix()
    snapshot = ledger.snapshot()
    for changed in (
        replace(snapshot, visibility_complete=False),
        replace(snapshot, history_complete=False),
        replace(snapshot, physical_ledger_complete=False),
    ):
        with pytest.raises(ValueError):
            encode_prefix(changed, candidates)
    with pytest.raises(ValueError):
        encode_prefix(snapshot, [])
    with pytest.raises(ValueError):
        encode_prefix(snapshot, list(range(33)))


def test_rank_gap_zero_and_absent_are_distinct_in_full_tokens():
    _, ledger, candidates = _prefix()
    sequence = encode_prefix(ledger.snapshot(), candidates)
    state = sequence.rows[sequence.history_rows + 1]
    assert state["stage"] == 1
    assert tuple(state["numeric"][18:20]) == (0.0, 0.0)
    assert tuple(state["numeric_present"][18:20]) == (False, True)
