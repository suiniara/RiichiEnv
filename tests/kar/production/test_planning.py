import json
from dataclasses import replace

import pytest

from riichienv.kar.budget import BudgetExceededError, BudgetLedger, BudgetPlan, CostForecast, choose_training_tier
from riichienv.kar.calibration_roles import CALIBRATION_SPLIT_SALT, calibration_is_fit
from riichienv.kar.learning.sampling import GlobalEpochSampler, validation_updates
from riichienv.kar.learning.training_config import ProductionTrainConfig, third_epoch_allowed, training_recipe_sha256


def forecast(phase="M2", cost=1.0):
    return CostForecast(phase, 100, 100.0, 0.0, cost, "synthetic unit-test estimate", "a" * 64)


def test_complete_global_epoch_has_no_real_sample_duplicates_or_missing_tail():
    lengths = [65 + index % 11 for index in range(103)]
    samplers = [
        GlobalEpochSampler(lengths, seed=17, effective_batch=16, microbatch=3, rank=rank, world_size=4)
        for rank in range(4)
    ]
    seen = []
    while not samplers[0].exhausted:
        groups = [sampler.peek() for sampler in samplers]
        assert len({group.microbatch_rounds for group in groups}) == 1
        assert sorted(index for group in groups for index in group.local_indices) == sorted(groups[0].global_indices)
        seen.extend(groups[0].global_indices)
        for sampler, group in zip(samplers, groups):
            sampler.commit(group)
    assert sorted(seen) == list(range(103)) and len(set(seen)) == 103
    assert samplers[0].updates_per_epoch == 7


def test_empty_ddp_rank_is_explicit_not_a_duplicated_training_example():
    groups = [
        GlobalEpochSampler([65, 66], seed=1, effective_batch=128, microbatch=1, rank=rank, world_size=4).peek()
        for rank in range(4)
    ]
    assert [len(group.local_indices) for group in groups] == [1, 1, 0, 0]
    assert all(group.microbatch_rounds == 1 and group.global_examples == 2 for group in groups)


def test_sampler_resume_is_at_an_exact_committed_global_boundary():
    first = GlobalEpochSampler([70] * 49, seed=4, effective_batch=16, microbatch=2)
    first.commit(first.peek())
    state = first.state_dict()
    resumed = GlobalEpochSampler([70] * 49, seed=4, effective_batch=16, microbatch=2)
    resumed.load_state_dict(state)
    assert resumed.peek() == first.peek()
    bad = dict(state, cursor=3)
    with pytest.raises(ValueError):
        resumed.load_state_dict(bad)
    assert validation_updates(1) == (1,) and validation_updates(8) == (2, 4, 6, 8)


def test_model_tier_uses_train_count_and_only_measured_affordable_fallbacks():
    values = {"large": forecast(cost=700.0), "medium": forecast(cost=660.0), "small": forecast(cost=100.0)}
    assert choose_training_tier(50000, values, budget_spent_usd=0.0) == "small"
    assert choose_training_tier(29999, {"small": forecast()}, budget_spent_usd=0.0) == "small"
    with pytest.raises(BudgetExceededError):
        choose_training_tier(50000, {}, budget_spent_usd=0.0)
    with pytest.raises(ValueError):
        BudgetPlan("M3", 300.0, 0.0, forecast("M3"))


def test_cost_ledger_cannot_loan_a_phase_budget_or_rewrite_receipts(tmp_path):
    plan = BudgetPlan("M1", 10.0, 1.0, forecast("M1", cost=1.0))
    ledger = BudgetLedger(tmp_path / "ledger.jsonl", plan)
    ledger.charge(2.0, category="storage", reference="synthetic receipt")
    assert ledger.spent_usd == 3.0
    resumed = BudgetLedger(tmp_path / "ledger.jsonl", plan)
    assert resumed.head == ledger.head
    with pytest.raises(BudgetExceededError):
        resumed.reserve_next(0.0, fixed_usd=8.0)
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    row = json.loads(lines[-1])
    row["usd"] = 0.0
    lines[-1] = json.dumps(row)
    (tmp_path / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        BudgetLedger(tmp_path / "ledger.jsonl", plan)


def test_config_roundtrip_and_independent_third_epoch_conditions():
    values = {
        "phase": "M2",
        "model_tier": "small",
        "N_train_half_games": 900,
        "N_eligible_half_games": 990,
        "planned_epochs": 3,
        "device": "cpu",
        "precision": "fp32",
    }
    recipe = training_recipe_sha256(values)
    estimate = replace(forecast(), recipe_sha256=recipe)
    config = ProductionTrainConfig(**values, budget=BudgetPlan("M2", 650.0, 0.0, estimate))
    assert config.recipe_sha256 == recipe
    assert ProductionTrainConfig.from_record(config.to_record()) == config
    game = next(f"epoch-fixture-{index}" for index in range(100) if calibration_is_fit(f"epoch-fixture-{index}"))
    fit = {
        "partition": "cal_fit",
        "cal_check_used": False,
        "full_cal_fit_evaluated": True,
        "calibration_split_salt": CALIBRATION_SPLIT_SALT,
        "game_ids": [game],
        "validation_objective": 2.0,
        "action_nll": 2.0,
        "response_nll": 1.0,
        "rank_nll": 1.0,
    }
    first = {
        "partition": "validation",
        "full_validation_split_evaluated": True,
        "validation_objective": 2.0,
        "model_selection": fit,
    }
    second = {**first, "model_selection": {**fit, "validation_objective": 1.9, "action_nll": 1.9}}
    assert third_epoch_allowed(first, second, config.third_epoch_nll_tolerance, budget_has_room=True)[0]
    assert not third_epoch_allowed(
        first,
        {**second, "model_selection": {**second["model_selection"], "rank_nll": 1.1}},
        config.third_epoch_nll_tolerance,
        budget_has_room=True,
    )[0]
    with pytest.raises(ValueError):
        third_epoch_allowed(
            first, {**second, "partition": "test"}, config.third_epoch_nll_tolerance, budget_has_room=True
        )
    assert third_epoch_allowed(
        first, {**second, "validation_objective": 999.0}, config.third_epoch_nll_tolerance, budget_has_room=True
    )[0]
    with pytest.raises(ValueError, match="cal_check"):
        third_epoch_allowed(
            first,
            {**second, "model_selection": {**second["model_selection"], "cal_check_used": True}},
            config.third_epoch_nll_tolerance,
            budget_has_room=True,
        )
