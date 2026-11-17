"""Forward-only LEGAL_SET wiring tests; mixed manifest sets are vector unit cases,
not claims that all of those actions can occur in one real response window.
"""
from dataclasses import replace

import pytest
import torch

from riichienv.kar.actions import action_spec, manifest
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.learning.batch import collate_inputs
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.tokenize import append_action, encode_prefix

TABLES = ("action_ids", "action_type_ids", "action_tile37", "action_tsumogiri", "action_riichi")
FAMILIES = {"DISCARD": 1, "ANKAN": 2, "KAKAN": 3, "TSUMO": 4, "KYUSHU_KYUHAI": 5,
            "PASS": 6, "CHI": 7, "PON": 8, "DAIMINKAN": 9, "RON": 10}


def _batch(ids, *, action=None):
    env = MJSOUL_RULESET.create_env(seed=42)
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    for event in env.mjai_log:
        ledger.apply_event(event)
    prefix = encode_prefix(ledger.snapshot(), ids)
    sequence = prefix if action is None else append_action(prefix, action)
    batch = collate_inputs([{"tokens": sequence.to_bytes()[108:], "candidates": prefix.candidates,
                             "decide_index": prefix.decide_index,
                             "action_index": -1 if action is None else prefix.decide_index + 1,
                             "decision_row_id": "fixed-input-context"}])
    return batch, prefix


@pytest.fixture
def controlled():
    model = KARTransformer(ModelConfig(layers=1, width=8, heads=2, feedforward=16, preset="vector-unit")).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for index, name in enumerate(TABLES):
            weight = model.embedding.scalar[name].weight
            weight.copy_(torch.arange(weight.shape[0])[:, None] * ((index + 1) * 1000)
                         + torch.arange(weight.shape[1])[None, :])
            weight[546 if name == "action_ids" else 0].zero_()
    return model


def _expected_action(action, width=8):
    # Independent explicit design §4.7 oracle, not the production factor helper.
    spec = action_spec(action)
    fields = [action, FAMILIES[spec.action_type], 0, 0, 0]
    if spec.action_type == "DISCARD":
        fields[2:] = [spec.tile37 + 1, 1 + int(spec.tsumogiri), 1 + int(spec.riichi)]
    vector = torch.zeros(width)
    for index, (name, row) in enumerate(zip(TABLES, fields)):
        if row == (546 if name == "action_ids" else 0):
            continue
        vector += row * ((index + 1) * 1000) + torch.arange(width)
    return vector


def _legal_vector(model, batch):
    captured = []
    handle = model.embedding.register_forward_hook(
        lambda _module, _args, output: captured.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            model._hidden(batch)
    finally:
        handle.remove()
    return captured[0][batch.fields["kind"] == 8][0]


@pytest.mark.parametrize("ids", [(0, 1, 2, 3), (221, 543), (148, 182, 219, 220, 221)])
def test_legal_set_vector_is_exact_mean_of_existing_five_action_tables(controlled, ids):
    batch, prefix = _batch(ids)
    expected = torch.stack([_expected_action(action) for action in ids]).mean(0)
    torch.testing.assert_close(_legal_vector(controlled, batch), expected, atol=0, rtol=0)
    row = next(row for row in prefix.rows if row["kind"] == 8)
    assert row["numeric"][16] == len(ids) / 32 and row["numeric_present"][16]
    assert set(controlled.embedding.scalar) >= set(TABLES)


def test_equal_size_different_sets_change_actual_input_vector_not_only_output_mask(controlled):
    first, _ = _batch((4, 8))
    second, _ = _batch((12, 16))
    assert all(torch.equal(first.fields[key], second.fields[key]) for key in first.fields)
    assert not torch.equal(_legal_vector(controlled, first), _legal_vector(controlled, second))


def test_candidate_permutation_and_padding_do_not_change_vector_or_denominator(controlled):
    batch, _ = _batch((0, 4, 221, 545))
    expected = torch.stack([_expected_action(action) for action in (0, 4, 221, 545)]).mean(0)
    shuffled = replace(batch, candidates=batch.candidates[:, torch.tensor([31, 2, 0, 30, 1, 3, *range(4, 30)])])
    torch.testing.assert_close(_legal_vector(controlled, batch), expected, atol=0, rtol=0)
    torch.testing.assert_close(_legal_vector(controlled, shuffled), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "family", ["ANKAN", "KAKAN", "TSUMO", "KYUSHU_KYUHAI", "PASS", "CHI", "PON", "DAIMINKAN", "RON"]
)
def test_non_discard_pooling_uses_manifest_family_and_na_factors(controlled, family):
    action = next(spec.global_id for spec in manifest() if spec.action_type == family)
    batch, _ = _batch((action,))
    torch.testing.assert_close(_legal_vector(controlled, batch), _expected_action(action), atol=0, rtol=0)


@pytest.mark.parametrize("bad", ["empty", "duplicate", "capacity"])
def test_candidate_pool_validates_nonempty_unique_and_capacity(controlled, bad):
    batch, _ = _batch((0, 4))
    candidates = batch.candidates.clone()
    if bad == "empty":
        candidates.fill_(546)
    elif bad == "duplicate":
        candidates[0, 1] = candidates[0, 0]
    else:
        candidates = torch.arange(33, dtype=torch.int64)[None, :]
    with pytest.raises(ValueError):
        _legal_vector(controlled, replace(batch, candidates=candidates))


def test_single_action_suffix_does_not_repeat_legal_pool(controlled):
    prefix_batch, _ = _batch((0, 4))
    dense, _ = _batch((0, 4), action=4)
    suffix = replace(dense, fields={key: value[:, -1:] for key, value in dense.fields.items()},
                     decide_index=torch.zeros(1, dtype=torch.int64), action_index=torch.zeros(1, dtype=torch.int64))
    captured = []
    with torch.no_grad():
        cache = controlled.encode_prefix(prefix_batch)
        handle = controlled.embedding.register_forward_hook(
            lambda _m, _a, output: captured.append(output.detach().clone())
        )
        try:
            controlled.score_actions(cache, suffix)
        finally:
            handle.remove()
    torch.testing.assert_close(captured[0][0, 0], _expected_action(4), atol=0, rtol=0)


def test_dense_and_prefix_cache_share_pool_and_teacher_action_cannot_reach_decide(miniature):
    miniature.eval()
    prefix, _ = _batch((0, 4, 8))
    first, _ = _batch((0, 4, 8), action=0)
    second, _ = _batch((0, 4, 8), action=4)
    with torch.no_grad():
        a, b = miniature(first), miniature(second)
        torch.testing.assert_close(a.policy_logits, b.policy_logits, atol=0, rtol=0)
        cache = miniature.encode_prefix(prefix)
        suffix = replace(first, fields={key: value[:, -1:] for key, value in first.fields.items()},
                         decide_index=torch.zeros(1, dtype=torch.int64), action_index=torch.zeros(1, dtype=torch.int64))
        response, rank = miniature.score_actions(cache, suffix)
        torch.testing.assert_close(cache.policy_logits, a.policy_logits, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(response, a.response_logits, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(rank, a.rank_logits, atol=2e-6, rtol=2e-5)


def test_cache_binds_complete_candidate_set_not_just_context_string(miniature):
    miniature.eval()
    prefix, _ = _batch((0, 4))
    action, _ = _batch((0, 8), action=0)
    suffix = replace(action, fields={key: value[:, -1:] for key, value in action.fields.items()},
                     decide_index=torch.zeros(1, dtype=torch.int64), action_index=torch.zeros(1, dtype=torch.int64))
    with torch.no_grad():
        cache = miniature.encode_prefix(prefix)
        with pytest.raises(ValueError, match="candidate|input|contract"):
            miniature.score_actions(cache, suffix)


def test_equal_length_sets_change_raw_policy_through_legal_input(miniature):
    miniature.eval()
    first, _ = _batch((4, 8))
    second, _ = _batch((12, 16))
    with torch.no_grad():
        assert not torch.equal(miniature(first).policy_logits, miniature(second).policy_logits)


def test_cache_rejects_old_input_version_and_changed_manifest_buffer(miniature):
    miniature.eval()
    prefix, _ = _batch((0, 4))
    dense, _ = _batch((0, 4), action=0)
    suffix = replace(dense, fields={key: value[:, -1:] for key, value in dense.fields.items()},
                     decide_index=torch.zeros(1, dtype=torch.int64), action_index=torch.zeros(1, dtype=torch.int64))
    with torch.no_grad():
        cache = miniature.encode_prefix(prefix)
        with pytest.raises(ValueError, match="input/model contract"):
            miniature.score_actions(replace(cache, input_pipeline_version="legacy-no-pool"), suffix)
        miniature.embedding.action_factors[0, 1] = 2
        with pytest.raises(ValueError, match="Stale"):
            miniature.score_actions(cache, suffix)
