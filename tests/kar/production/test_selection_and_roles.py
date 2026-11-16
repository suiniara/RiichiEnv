import math
from dataclasses import replace

import pytest

from riichienv.kar.calibration import calibration_split, choose_validation_tau
from riichienv.kar.calibration_roles import CALIBRATION_SPLIT_SALT, calibration_is_fit
from riichienv.kar.contracts import content_hash, game_split
from riichienv.kar.runtime_contracts import RuntimeBinding
from riichienv.kar.selection_quality import TauSelectionPolicy, validate_finite_tau_selection


def selection_fixture(values):
    games = []
    index = 0
    while len(games) < len(values):
        identity = f"synthetic-gain-{index}"
        if game_split(identity, "kar-riichienv-v1-games") == "validation" and calibration_is_fit(identity):
            games.append(identity)
        index += 1
    split = calibration_split(games)
    binding = RuntimeBinding.current("a" * 64, "b" * 64)
    outcomes = {math.inf: {game: 0.0 for game in games}, 0.5: dict(zip(games, values))}
    return split, binding, outcomes


def test_positive_validation_mean_without_reproducible_gain_retains_infinity():
    split, binding, outcomes = selection_fixture([-1.0, 1.2])
    result = choose_validation_tau(
        outcomes,
        split=split,
        binding=binding,
        grid=(math.inf, 0.5),
        diagnostic_fixture=True,
        selection_policy=TauSelectionPolicy(bootstrap_replicates=100),
    )
    assert result["mean_utility"]["0.5"] > result["mean_utility"]["infinity"]
    assert result["selected_tau"] == "infinity" and not result["reproducible_gain"]


def test_one_scenario_is_not_repeatability_even_if_many_decisions_share_it():
    split, binding, outcomes = selection_fixture([1.0])
    result = choose_validation_tau(
        outcomes,
        split=split,
        binding=binding,
        grid=(math.inf, 0.5),
        diagnostic_fixture=True,
        selection_policy=TauSelectionPolicy(bootstrap_replicates=20),
    )
    assert result["selected_tau"] == "infinity"
    assert result["paired_validation_evidence"]["0.5"]["paired_utility_interval"]["independent_games"] == 1


def test_finite_validation_gate_and_criteria_are_immutable_metadata():
    split, binding, outcomes = selection_fixture([0.5, 0.5, 0.5])
    policy = TauSelectionPolicy(bootstrap_replicates=20)
    result = choose_validation_tau(
        outcomes, split=split, binding=binding, grid=(math.inf, 0.5), diagnostic_fixture=True, selection_policy=policy
    )
    assert result["selected_tau"] == "0.5" and result["reproducible_gain"]
    # Production-shaped consumer unit fixture, not an executed validation job.
    body = {key: value for key, value in result.items() if key != "tau_selection_sha256"}
    body["diagnostic_fixture"] = False
    production = {**body, "tau_selection_sha256": content_hash(body)}
    assert validate_finite_tau_selection(production) == policy
    body["paired_validation_evidence"]["0.5"]["paired_utility_interval"]["lower"] = 0.0
    rejected = {**body, "tau_selection_sha256": content_hash(body)}
    with pytest.raises(ValueError):
        validate_finite_tau_selection(rejected)


def test_production_tau_selection_cannot_omit_preregistered_gain_criteria():
    split, binding, outcomes = selection_fixture([0.5, 0.5])
    split = replace(split, validation_manifest_sha256="c" * 64, main_split_salt="kar-riichienv-v1-games")
    with pytest.raises(ValueError, match="Preregister"):
        choose_validation_tau(outcomes, split=split, binding=binding, grid=(math.inf, 0.5))


def test_independent_cal_check_is_never_used_for_epoch_loss_selection():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from test_training_contracts import SyntheticDataset  # noqa: PLC0415

    from riichienv.kar.learning.dataset import collate_examples  # noqa: PLC0415
    from riichienv.kar.learning.evaluation import cal_fit_loss_summary  # noqa: PLC0415
    from riichienv.kar.learning.model import ModelOutputs  # noqa: PLC0415

    dataset = SyntheticDataset()
    fit = next(f"role-fit-{index}" for index in range(100) if calibration_is_fit(f"role-fit-{index}"))
    check = next(f"role-check-{index}" for index in range(100) if not calibration_is_fit(f"role-check-{index}"))
    rows = [{**dataset[0], "game_id": fit}, {**dataset[1], "game_id": check}]
    outputs = ModelOutputs(
        torch.zeros((2, 546)), torch.zeros((2, 21)), torch.zeros((2, 21, 4)), torch.ones(2, dtype=torch.bool)
    )
    before = cal_fit_loss_summary(outputs, collate_examples(rows))
    rows[1] = {**rows[1], "final_rank_target": 3}
    after = cal_fit_loss_summary(outputs, collate_examples(rows))
    assert before == after and before["examples"] == 1 and before["games"] == [fit]
    assert CALIBRATION_SPLIT_SALT == "kar-cal-fit-check-v1"
