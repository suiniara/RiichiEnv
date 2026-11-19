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
    split = replace(
        calibration_split(games),
        validation_manifest_sha256="c" * 64,
        main_split_salt="kar-riichienv-v1-games",
    )
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
    split, binding, outcomes = selection_fixture([1.0] * 64)
    policy = TauSelectionPolicy(bootstrap_replicates=20)
    result = choose_validation_tau(
        outcomes, split=split, binding=binding, grid=(math.inf, 0.5), diagnostic_fixture=True, selection_policy=policy
    )
    assert result["selected_tau"] == "0.5" and result["reproducible_gain"]
    # Production-shaped consumer unit fixture, not an executed validation job.
    body = {key: value for key, value in result.items() if key != "tau_selection_sha256"}
    body["diagnostic_fixture"] = False
    production = {**body, "tau_selection_sha256": content_hash(body)}
    assert validate_finite_tau_selection(production, expected_game_ids=split.fit_games) == policy
    body["paired_validation_evidence"]["0.5"]["gain_bound"]["lower"] = 0.0
    rejected = {**body, "tau_selection_sha256": content_hash(body)}
    with pytest.raises(ValueError):
        validate_finite_tau_selection(rejected, expected_game_ids=split.fit_games)


def test_two_positive_rank_pairs_cannot_admit_a_null_policy():
    # Under the null the baseline is rank 3 (-0.5), while the candidate is
    # rank 2 (+0.5) or rank 4 (-1.5), independently with equal probability.
    # Both observed differences are +1 with probability 1/4. The old bootstrap
    # interval degenerates to [1, 1] and wrongly declares 95% positive evidence.
    split, binding, outcomes = selection_fixture([0.5, 0.5])
    outcomes[math.inf] = dict.fromkeys(split.fit_games, -0.5)
    result = choose_validation_tau(
        outcomes,
        split=split,
        binding=binding,
        grid=(math.inf, 0.5),
        diagnostic_fixture=True,
        selection_policy=TauSelectionPolicy(bootstrap_replicates=20),
    )
    evidence = result["paired_validation_evidence"]["0.5"]
    assert evidence["paired_utility_interval"]["lower"] == 1.0
    assert evidence["gain_bound"]["lower"] < 0
    assert evidence["gain_bound"]["finite_candidates"] == 1
    assert result["selected_tau"] == "infinity" and result["reproducible_gain"] is False


def test_simultaneous_gain_bound_counts_every_registered_finite_candidate():
    split, binding, outcomes = selection_fixture([1.0] * 64)
    policy = TauSelectionPolicy(bootstrap_replicates=20)
    single = choose_validation_tau(
        outcomes,
        split=split,
        binding=binding,
        grid=(math.inf, 0.5),
        diagnostic_fixture=True,
        selection_policy=policy,
    )
    expanded = {**outcomes, 1.0: outcomes[0.5], 0.25: outcomes[0.5]}
    multiple = choose_validation_tau(
        expanded,
        split=split,
        binding=binding,
        grid=(math.inf, 1.0, 0.5, 0.25),
        diagnostic_fixture=True,
        selection_policy=policy,
    )
    first = single["paired_validation_evidence"]["0.5"]["gain_bound"]
    second = multiple["paired_validation_evidence"]["0.5"]["gain_bound"]
    assert first["lower"] == pytest.approx(1.0 - math.sqrt(18.0 * math.log(20.0) / 64))
    assert second["lower"] == pytest.approx(1.0 - math.sqrt(18.0 * math.log(60.0) / 64))
    assert first["finite_candidates"] == 1 and second["finite_candidates"] == 3
    assert single["selected_tau"] == "0.5" and multiple["selected_tau"] == "infinity"


def test_infinity_only_grid_needs_no_finite_candidate_bound():
    split, binding, outcomes = selection_fixture([1.0, 1.0])
    result = choose_validation_tau(
        {math.inf: outcomes[math.inf]},
        split=split,
        binding=binding,
        grid=(math.inf,),
        diagnostic_fixture=True,
        selection_policy=TauSelectionPolicy(bootstrap_replicates=20),
    )
    assert result["selected_tau"] == "infinity" and result["reproducible_gain"] is False
    assert result["paired_validation_evidence"]["infinity"]["gain_bound"] is None


@pytest.mark.parametrize(
    "modified", ["version", "policy_version", "candidate_count", "game_count", "missing_candidate"]
)
def test_finite_consumer_recomputes_evidence_and_rejects_superseded_artifacts(modified):
    split, binding, outcomes = selection_fixture([1.0] * 64)
    result = choose_validation_tau(
        outcomes,
        split=split,
        binding=binding,
        grid=(math.inf, 0.5),
        diagnostic_fixture=True,
        selection_policy=TauSelectionPolicy(bootstrap_replicates=20),
    )
    body = {key: value for key, value in result.items() if key != "tau_selection_sha256"}
    body["diagnostic_fixture"] = False
    if modified == "version":
        body["version"] = "frozen_validation_tau_grid_v3"
    elif modified == "policy_version":
        old = {key: value for key, value in body["selection_policy"].items() if key != "selection_policy_sha256"}
        old["version"] = "preregistered_tau_selection_v1"
        body["selection_policy"] = {**old, "selection_policy_sha256": content_hash(old)}
    elif modified == "candidate_count":
        body["paired_validation_evidence"]["0.5"]["gain_bound"]["finite_candidates"] = 0
    elif modified == "game_count":
        body["paired_validation_evidence"]["0.5"]["gain_bound"]["independent_games"] += 1
    else:
        body["grid"].append("0.25")
    resigned = {**body, "tau_selection_sha256": content_hash(body)}
    with pytest.raises(ValueError):
        validate_finite_tau_selection(resigned, expected_game_ids=split.fit_games)


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
    from riichienv.kar.learning.losses import loss_sums  # noqa: PLC0415
    from riichienv.kar.learning.model import ModelOutputs  # noqa: PLC0415

    dataset = SyntheticDataset()
    fit = next(f"role-fit-{index}" for index in range(100) if calibration_is_fit(f"role-fit-{index}"))
    check = next(f"role-check-{index}" for index in range(100) if not calibration_is_fit(f"role-check-{index}"))
    rows = [{**dataset[0], "game_id": fit}, {**dataset[1], "game_id": check}]
    rank_logits = torch.tensor([4.0, 1.0, -0.5, -2.0]).expand(2, 21, 4).clone()
    outputs = ModelOutputs(
        torch.zeros((2, 546)),
        torch.zeros((2, 21)),
        rank_logits,
        torch.ones(2, dtype=torch.bool),
    )
    batch = collate_examples(rows)
    before = cal_fit_loss_summary(outputs, batch)
    assert before["examples"] == 1 and before["games"] == [fit] and before["rank_count"] == 1

    def change_label(index):
        changed = list(rows)
        changed[index] = {**changed[index], "final_rank": 3}
        collated = collate_examples(changed)
        assert int(collated.labels.final_rank_target[index]) == 3
        return collated

    def change_logits(index):
        changed = outputs.rank_logits.clone()
        changed[index] = changed[index].flip(-1)
        return replace(outputs, rank_logits=changed)

    # If both rows were used, changing either input would affect the loss.
    total = loss_sums(outputs, batch.labels, batch.inputs.legal_mask).value
    check_changed = change_label(1)
    assert loss_sums(outputs, check_changed.labels, batch.inputs.legal_mask).value > total
    assert loss_sums(change_logits(1), batch.labels, batch.inputs.legal_mask).value > total
    assert cal_fit_loss_summary(outputs, check_changed) == before
    assert cal_fit_loss_summary(change_logits(1), batch) == before
    assert cal_fit_loss_summary(outputs, change_label(0))["rank"] > before["rank"]
    assert cal_fit_loss_summary(change_logits(0), batch)["rank"] > before["rank"]
    assert CALIBRATION_SPLIT_SALT == "kar-cal-fit-check-v1"
