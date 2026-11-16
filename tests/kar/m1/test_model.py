from dataclasses import replace

import pytest
import torch

from riichienv.kar.learning.batch import collate_inputs, feature_mask_draw, rule_feature_view
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.model import KARTransformer, block_attention_mask


@pytest.mark.parametrize(
    "preset,expected", [("tiny", 2085771), ("small", 5239435), ("medium", 14790795), ("large", 26536587)]
)
def test_exact_parameter_counts_and_independent_embeddings(preset, expected):
    model = KARTransformer(ModelConfig.from_preset(preset))
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    tables = [*model.embedding.scalar.values(), *model.embedding.tiles, *model.embedding.flags]
    assert sum(table.num_embeddings for table in tables) == 944
    assert len({table.weight.data_ptr() for table in tables}) == len(tables)
    assert model.embedding.numeric.bias is None


@pytest.mark.parameter_updates
def test_dense_head_shapes_and_backward(miniature, model_batch):
    outputs = miniature(model_batch)
    assert outputs.policy_logits.shape == (2, 546)
    assert outputs.response_logits.shape == (2, 21)
    assert outputs.rank_logits.shape == (2, 21, 4)
    assert all(
        value.dtype == torch.float32 for value in (outputs.policy_logits, outputs.response_logits, outputs.rank_logits)
    )
    sum(
        value.square().mean() for value in (outputs.policy_logits, outputs.response_logits, outputs.rank_logits)
    ).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in miniature.parameters()
    )


def test_teacher_action_never_changes_policy(miniature, make_sample):
    first = collate_inputs([make_sample(action_choice=0)])
    second = collate_inputs([make_sample(action_choice=1)])
    miniature.eval()
    a, b = miniature(first), miniature(second)
    torch.testing.assert_close(a.policy_logits, b.policy_logits, rtol=0, atol=0)
    assert not torch.equal(a.response_logits, b.response_logits)


def test_block_mask_semantics_and_padding():
    blocks = torch.tensor([[0, 0, 1, 1, 2, 3, 0]])
    valid = torch.tensor([[True] * 6 + [False]])
    mask = block_attention_mask(blocks, valid)[0, 0]
    assert mask[0].tolist() == [True, False, False, False, False, False, False]
    assert mask[2].tolist() == [True, True, True, True, False, False, False]
    assert mask[4].tolist() == [True, True, True, True, True, False, False]
    assert mask[5].tolist() == [True, True, True, True, True, True, False]
    assert mask[6].tolist() == [True, False, False, False, False, False, False]


def test_state_permutation_and_padding_do_not_change_readouts(miniature, make_sample):
    sample = make_sample()
    prefix = sample["prefix"]
    data = bytearray(sample["tokens"])
    size = 139
    start = prefix.history_rows + 1
    for left, right in ((start + 1, start + 2), (start + 5, start + 6)):
        a, b = bytes(data[left * size : (left + 1) * size]), bytes(data[right * size : (right + 1) * size])
        data[left * size : (left + 1) * size], data[right * size : (right + 1) * size] = b, a
    altered = {**sample, "tokens": bytes(data)}
    miniature.eval()
    a, b = miniature(collate_inputs([sample])), miniature(collate_inputs([altered]))
    torch.testing.assert_close(a.policy_logits, b.policy_logits, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(a.rank_logits, b.rank_logits, atol=2e-6, rtol=2e-5)
    padded = {**sample, "tokens": sample["tokens"] + bytes(139 * 4)}
    c = miniature(collate_inputs([padded]))
    torch.testing.assert_close(a.policy_logits, c.policy_logits, atol=2e-6, rtol=2e-5)


def test_cached_independent_actions_match_dense(miniature, make_sample):
    miniature.eval()
    prefix = collate_inputs([make_sample(prefix_only=True)])
    cache = miniature.encode_prefix(prefix)
    samples = [make_sample(action_choice=index) for index in range(3)]
    branches = [
        {**sample, "tokens": sample["tokens"][-139:], "decide_index": 0, "action_index": 0} for sample in samples
    ]
    response, ranks = miniature.score_actions(cache, collate_inputs(branches))
    dense = miniature(collate_inputs(samples))
    torch.testing.assert_close(cache.policy_logits[0], dense.policy_logits[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(response, dense.response_logits, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(ranks, dense.rank_logits, atol=2e-6, rtol=2e-5)
    alone = miniature.score_actions(cache, collate_inputs(branches[:1]))
    torch.testing.assert_close(alone[0][0], response[0], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("change", ["weights", "foreign", "context", "cache"])
def test_stale_or_cross_decision_cache_is_rejected(miniature, make_sample, change):
    miniature.eval()
    cache = miniature.encode_prefix(collate_inputs([make_sample(prefix_only=True)]))
    sample = make_sample()
    branch = collate_inputs([{**sample, "tokens": sample["tokens"][-139:], "decide_index": 0, "action_index": 0}])
    if change == "weights":
        with torch.no_grad():
            next(miniature.parameters()).add_(0.1)
    elif change == "foreign":
        miniature = KARTransformer(miniature.config).eval()
    elif change == "context":
        branch = replace(branch, context_ids=("different-observer-decision",))
    else:
        cache.key_values[0][0].add_(1)
    with pytest.raises(ValueError):
        miniature.score_actions(cache, branch)


def test_feature_mask_is_whole_sample_and_does_not_mutate_base(model_batch, miniature):
    parameters = sum(parameter.numel() for parameter in miniature.parameters())
    base = model_batch.fields
    on = rule_feature_view(model_batch, torch.tensor([True, False]))
    hands = base["kind"] == 4
    assert torch.all(on.fields["flags"][0, hands[0], :8] == 0)
    assert torch.equal(on.fields["flags"][1], base["flags"][1])
    assert torch.equal(on.fields["stage"], base["stage"])
    off = rule_feature_view(model_batch, torch.tensor([False, False]), enabled=False)
    assert torch.all(off.fields["stage"][base["kind"] == 2] == 0)
    assert torch.all(base["stage"][base["kind"] == 2] > 0)
    assert sum(parameter.numel() for parameter in miniature.parameters()) == parameters


def test_feature_rng_depends_on_sample_not_rank_or_order():
    ids = [f"decision-{index}" for index in range(200)]
    first = {identity: feature_mask_draw(17, 0, "game", identity) for identity in ids}
    second = {identity: feature_mask_draw(17, 0, "game", identity) for identity in reversed(ids)}
    assert first == second and 20 < sum(first.values()) < 80
    assert first != {identity: feature_mask_draw(17, 1, "game", identity) for identity in ids}
