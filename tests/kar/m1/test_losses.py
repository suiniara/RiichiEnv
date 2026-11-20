from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from riichienv.kar.learning.losses import Labels, loss_sums
from riichienv.kar.learning.model import ModelOutputs


def _case(size=4):
    generator = torch.Generator().manual_seed(9)
    outputs = ModelOutputs(
        torch.randn(size, 546, generator=generator, requires_grad=True),
        torch.randn(size, 21, generator=generator, requires_grad=True),
        torch.randn(size, 21, 4, generator=generator, requires_grad=True),
        torch.ones(size, dtype=torch.bool),
    )
    observed = torch.tensor(([True, False] * size)[:size])
    labels = Labels(
        torch.arange(size, dtype=torch.int64),
        torch.ones(size, dtype=torch.bool),
        torch.where(observed, torch.arange(size, dtype=torch.int64), -255),
        observed.clone(),
        observed.clone(),
        torch.where(observed, torch.arange(size, dtype=torch.int64) % 4, 255),
        observed.clone(),
        torch.ones(size, 21, dtype=torch.bool),
    )
    legal = torch.ones(size, 546, dtype=torch.bool)
    return outputs, labels, legal


@pytest.mark.parameter_updates
def test_sentinel_safe_loss_matches_selected_valid_reference():
    outputs, labels, legal = _case()
    result = loss_sums(outputs, labels, legal)
    valid = labels.response_valid
    expected_action = F.cross_entropy(outputs.policy_logits, labels.action_target, reduction="sum")
    expected_response = F.cross_entropy(outputs.response_logits[valid], labels.response_target[valid], reduction="sum")
    selected = outputs.rank_logits[valid, labels.response_target[valid]]
    expected_value = F.cross_entropy(selected, labels.final_rank_target[valid], reduction="sum")
    for actual, expected in (
        (result.action, expected_action),
        (result.response, expected_response),
        (result.value, expected_value),
    ):
        torch.testing.assert_close(actual, expected)
    result.objective(global_examples=4, world_size=1, rho_e=0.5, rho_v=0.5).backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in (outputs.policy_logits, outputs.response_logits, outputs.rank_logits)
    )
    assert torch.count_nonzero(outputs.rank_logits.grad[:, 1]) == 0


@pytest.mark.parametrize("bad", [-1, 21, 255])
def test_active_bad_response_rejected_before_gather(bad):
    outputs, labels, legal = _case()
    event = labels.response_target.clone()
    event[0] = bad
    with pytest.raises(ValueError, match="response target"):
        loss_sums(outputs, replace(labels, response_target=event), legal)


@pytest.mark.parametrize("bad", [-1, 4, 255])
def test_active_bad_rank_rejected_before_gather(bad):
    outputs, labels, legal = _case()
    rank = labels.final_rank_target.clone()
    rank[0] = bad
    with pytest.raises(ValueError, match="rank target"):
        loss_sums(outputs, replace(labels, final_rank_target=rank), legal)


def test_auxiliary_rules_are_applied_before_softmax():
    outputs, labels, legal = _case()
    allowed = torch.zeros(4, 21, dtype=torch.bool)
    allowed[:, 0] = True
    event = labels.response_target.clone()
    event[2] = 0
    result = loss_sums(outputs, replace(labels, response_allowed=allowed, response_target=event), legal)
    assert result.response.item() == 0
    allowed[:, 0] = False
    with pytest.raises(ValueError, match="empty"):
        loss_sums(outputs, replace(labels, response_allowed=allowed), legal)


def test_masked_out_target_is_not_silently_discarded():
    outputs, labels, legal = _case()
    allowed = labels.response_allowed.clone()
    allowed[0, 0] = False
    with pytest.raises(ValueError, match="excluded"):
        loss_sums(outputs, replace(labels, response_allowed=allowed), legal)
    legal[:, 0] = False
    with pytest.raises(ValueError, match="excluded"):
        loss_sums(outputs, labels, legal)


@pytest.mark.parameter_updates
def test_no_valid_auxiliary_labels_produces_connected_zero_gradients_and_na_metrics():
    outputs, labels, legal = _case()
    no = torch.zeros(4, dtype=torch.bool)
    labels = replace(
        labels,
        response_applicable=no,
        response_observed=no,
        value_valid=no,
        response_allowed=torch.zeros(4, 21, dtype=torch.bool),
    )
    result = loss_sums(outputs, labels, legal)
    result.objective(global_examples=4, world_size=1, rho_e=0.5, rho_v=0.5).backward()
    assert result.response.item() == result.value.item() == 0
    assert result.metrics()["response_nll"] is None and result.metrics()["rank_nll"] is None
    for value in (outputs.response_logits, outputs.rank_logits):
        assert value.grad is not None and value.grad.count_nonzero() == 0


@pytest.mark.parameter_updates
def test_candidate_branch_auxiliary_labels_do_not_require_policy_action_labels():
    outputs, labels, legal = _case(3)
    action_valid = torch.tensor([True, False, False])
    response_applicable = torch.tensor([False, True, True])
    response_observed = torch.tensor([False, True, True])
    value_valid = torch.tensor([False, True, True])
    labels = replace(
        labels,
        action_valid=action_valid,
        response_applicable=response_applicable,
        response_observed=response_observed,
        value_valid=value_valid,
        response_target=torch.tensor([-255, 2, 2]),
        final_rank_target=torch.tensor([255, 1, 3]),
        response_weight=torch.tensor([0.0, 0.5, 0.5]),
        value_weight=torch.tensor([0.0, 0.5, 0.5]),
    )
    result = loss_sums(outputs, labels, legal)
    assert result.action_count == 1
    assert result.response_count == pytest.approx(1.0)
    assert result.value_count == pytest.approx(1.0)
    assert result.response.item() == pytest.approx(
        0.5 * F.cross_entropy(outputs.response_logits[1:2], torch.tensor([2]), reduction="sum").item()
        + 0.5 * F.cross_entropy(outputs.response_logits[2:3], torch.tensor([2]), reduction="sum").item()
    )


def test_value_valid_requires_response_valid():
    outputs, labels, legal = _case()
    with pytest.raises(ValueError, match="implies"):
        loss_sums(outputs, replace(labels, value_valid=torch.ones(4, dtype=torch.bool)), legal)


def test_nonfinite_outputs_abort_even_on_invalid_labels():
    outputs, labels, legal = _case()
    with torch.no_grad():
        outputs.rank_logits[1, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError):
        loss_sums(outputs, labels, legal)


def _slice(outputs, labels, legal, start, end):
    out = ModelOutputs(
        *(
            value[start:end]
            for value in (outputs.policy_logits, outputs.response_logits, outputs.rank_logits, outputs.action_present)
        )
    )
    lab = Labels(**{key: value[start:end] for key, value in labels.__dict__.items()})
    return out, lab, legal[start:end]


@pytest.mark.parameter_updates
def test_global_sum_scaling_equals_uneven_microbatch_accumulation():
    outputs, labels, legal = _case(5)
    full = loss_sums(outputs, labels, legal).objective(global_examples=5, world_size=1, rho_e=0.6, rho_v=0.6)
    micros = [
        loss_sums(*_slice(outputs, labels, legal, a, b)).objective(
            global_examples=5, world_size=1, rho_e=0.6, rho_v=0.6
        )
        for a, b in ((0, 2), (2, 5))
    ]
    torch.testing.assert_close(sum(micros), full)
    grads = torch.autograd.grad(
        full, (outputs.policy_logits, outputs.response_logits, outputs.rank_logits), retain_graph=True
    )
    accumulated = torch.autograd.grad(
        sum(micros), (outputs.policy_logits, outputs.response_logits, outputs.rank_logits)
    )
    for first, second in zip(grads, accumulated):
        torch.testing.assert_close(first, second)


@pytest.mark.parametrize("rho_e,rho_v", [(0, 0.1), (0.1, 0), (0.1, 0.2), (1.1, 0.1)])
def test_fixed_rhos_must_be_positive_and_consistent(rho_e, rho_v):
    outputs, labels, legal = _case()
    with pytest.raises(ValueError):
        loss_sums(outputs, labels, legal).objective(global_examples=4, world_size=1, rho_e=rho_e, rho_v=rho_v)
