"""CPU forward-only FP32 readout boundaries, including an outer BF16 autocast."""

import json
from dataclasses import replace

import pytest
import torch

from riichienv.kar.learning.batch import collate_inputs
from riichienv.kar.learning.inference import model_weights_sha256


@pytest.fixture
def frozen_model(miniature, record_property):
    model = miniature.eval()
    before = model_weights_sha256(model)
    record_property("model_weights_sha256_before", before)
    try:
        yield model
    finally:
        after = model_weights_sha256(model)
        record_property("model_weights_sha256_after", after)
        assert before == after
        assert all(parameter.grad is None for parameter in model.parameters())


def _batches(make_sample, seeds=(42, 0)):
    prefixes = [make_sample(seed, prefix_only=True) for seed in seeds]
    samples = [make_sample(seed, action_choice=1) for seed in seeds]
    branches = [
        {**sample, "tokens": sample["tokens"][-139:], "decide_index": 0, "action_index": 0} for sample in samples
    ]
    return collate_inputs(prefixes), collate_inputs(samples), collate_inputs(branches)


@pytest.mark.parametrize("use_bf16", [False, True], ids=["fp32", "cpu-bf16"])
@pytest.mark.parametrize("seeds", [(42,), (42, 0)], ids=["single", "batched"])
@torch.no_grad()
def test_dense_and_cached_three_heads_have_fp32_projection_boundaries(
    frozen_model, make_sample, record_property, use_bf16, seeds
):
    model = frozen_model
    prefix, dense_batch, branches = _batches(make_sample, seeds)
    projections = []

    def watch(name):
        def capture(_module, arguments, output):
            projections.append(
                {
                    "head": name,
                    "input_dtype": str(arguments[0].dtype),
                    "output_dtype": str(output.dtype),
                    "cpu_autocast_enabled": torch.is_autocast_enabled("cpu"),
                }
            )

        return capture

    handles = [
        head.register_forward_hook(watch(name))
        for name, head in (("policy", model.policy_head), ("response", model.response_head), ("rank", model.value_head))
    ]
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_bf16):
            dense = model(dense_batch)
            cache = model.encode_prefix(prefix)
            response, ranks = model.score_actions(cache, branches, torch.arange(len(seeds)))
            assert torch.is_autocast_enabled("cpu") == use_bf16
    finally:
        for handle in handles:
            handle.remove()

    dense_heads = (dense.policy_logits, dense.response_logits, dense.rank_logits)
    cached_heads = (cache.policy_logits, response, ranks)
    differences = [float((a.float() - b.float()).abs().max()) for a, b in zip(dense_heads, cached_heads)]
    record_property("projection_observations", json.dumps(projections, sort_keys=True))
    record_property("dense_cached_max_abs_differences", json.dumps(differences))
    record_property("prefix_kv_dtypes", json.dumps([str(tensor.dtype) for pair in cache.key_values for tensor in pair]))
    expected_shapes = ((len(seeds), 546), (len(seeds), 21), (len(seeds), 21, 4))
    for group in (dense_heads, cached_heads):
        assert tuple(tuple(value.shape) for value in group) == expected_shapes
        assert all(value.dtype == torch.float32 and not value.requires_grad for value in group)
        assert all(torch.isfinite(value).all() for value in group)
    assert len(projections) == 6
    assert all(not item["cpu_autocast_enabled"] for item in projections)
    assert all(item["input_dtype"] == item["output_dtype"] == "torch.float32" for item in projections)
    expected_kv_dtype = torch.bfloat16 if use_bf16 else torch.float32
    assert all(tensor.dtype == expected_kv_dtype for pair in cache.key_values for tensor in pair)
    if not use_bf16:
        for dense_head, cached_head in zip(dense_heads, cached_heads):
            torch.testing.assert_close(dense_head, cached_head, atol=2e-6, rtol=2e-5)
    # BF16 and FP32 backbones are intentionally not compared for bitwise equality.


@pytest.mark.parametrize("use_bf16", [False, True], ids=["fp32", "cpu-bf16"])
@torch.no_grad()
def test_cached_policy_matches_fp32_projection_of_identical_prefix_hidden(
    frozen_model, make_sample, record_property, use_bf16
):
    model = frozen_model
    prefix, _, _ = _batches(make_sample)
    seen = []
    handle = model.policy_head.register_forward_pre_hook(
        lambda _module, arguments: seen.append(arguments[0].detach().clone())
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_bf16):
        try:
            cache = model.encode_prefix(prefix)
        finally:
            handle.remove()
        assert len(seen) == 1
        decide = seen[0]
        with torch.autocast(device_type="cpu", enabled=False):
            reference = model.policy_head(decide.float())
        if use_bf16:
            # A post-projection float() has the right container dtype but loses precision.
            lossy = model.policy_head(decide.float()).float()
            lossy_error = float((lossy - reference).abs().max())
            record_property("post_cast_bf16_projection_max_abs_error", lossy_error)
            assert lossy_error > 0
    error = float((cache.policy_logits.float() - reference).abs().max())
    record_property("cached_policy_dtype", str(cache.policy_logits.dtype))
    record_property("same_hidden_fp32_reference_dtype", str(reference.dtype))
    record_property("same_hidden_fp32_max_abs_error", error)
    assert reference.dtype == torch.float32
    # Check the values independently of dtype so a post-projection cast cannot pass.
    torch.testing.assert_close(cache.policy_logits.float(), reference, atol=0, rtol=0)


@pytest.mark.parametrize("use_bf16", [False, True], ids=["fp32", "cpu-bf16"])
@pytest.mark.parametrize("invalid", ["teacher-prefix", "empty-branch", "multirow-branch"])
@torch.no_grad()
def test_precision_boundary_preserves_prefix_and_single_action_validation(frozen_model, make_sample, use_bf16, invalid):
    prefix, dense, branches = _batches(make_sample)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_bf16):
        if invalid == "teacher-prefix":
            with pytest.raises(ValueError, match="must exclude teacher ACTION"):
                frozen_model.encode_prefix(dense)
        else:
            cache = frozen_model.encode_prefix(prefix)
            invalid_branch = (
                dense
                if invalid == "multirow-branch"
                else replace(branches, fields={key: value[:, :0] for key, value in branches.fields.items()})
            )
            with pytest.raises(ValueError, match="exactly one valid ACTION"):
                frozen_model.score_actions(cache, invalid_branch, torch.arange(prefix.size))
