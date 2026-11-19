"""Production API boundary tests with synthetic evidence, never actual matches."""

# ruff: noqa: E402 -- Torch remains an optional ML dependency
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from riichienv.kar.authorization import update_implementation_sha256
from riichienv.kar.budget import BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.calibration import CalibrationRow, calibrate_static_rows, calibration_split, choose_validation_tau
from riichienv.kar.calibration_quality import BranchQualityCriterion, BranchQualityPolicy, CalibrationQualityPolicy
from riichienv.kar.contracts import content_hash, game_split
from riichienv.kar.learning import validation_policy
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.deployment import FrozenRelease, save_frozen_release
from riichienv.kar.learning.inference import FrozenModelPredictor
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.policy import PolicyOptions
from riichienv.kar.selection_quality import TauSelectionPolicy
from riichienv.kar.support import EventSupport
from riichienv.kar.versioning import artifact_contract


@pytest.fixture
def prior_release(tmp_path):
    """Write/load a genuine release using explicitly synthetic calibration rows."""
    games = [
        f"synthetic-tau-support-{index}"
        for index in range(1000)
        if game_split(f"synthetic-tau-support-{index}", "kar-riichienv-v1-games") == "validation"
    ][:20]
    split = replace(
        calibration_split(games), validation_manifest_sha256="a" * 64, main_split_salt="kar-riichienv-v1-games"
    )

    def build(state):
        model = KARTransformer(ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="synthetic-tau-support"))
        predictor = FrozenModelPredictor(model, "b" * 64)
        support = EventSupport(
            "b" * 64,
            (499 if state == "train_insufficient" else 500,) + (0,) * 20,
            "synthetic support counts; no real training evidence",
        )
        branch = BranchQualityPolicy(
            (
                BranchQualityCriterion("response", 0, 5.0, 2.0, 1.0, 1),
                BranchQualityCriterion(
                    "rank", 0, 0.0 if state == "failed" else 5.0, 2.0, 1.0, 2 if state == "unavailable" else 1
                ),
            ),
            "synthetic branch criteria fixed before fixture logits",
        )
        heads = ("action", "response", "rank")
        quality = CalibrationQualityPolicy(
            {head: 5.0 for head in heads},
            {head: 2.0 for head in heads},
            {head: 1.0 for head in heads},
            1,
            1,
            0.0,
            "synthetic overall criteria, not production thresholds",
            branch,
        )

        def row(game):
            return CalibrationRow(
                game,
                game + "/decision",
                (4, 8),
                (0.0,) * 546,
                4,
                response_logits=(8.0,) + (0.0,) * 20,
                response_allowed=(True,) * 21,
                rank_logits=((8.0, 0.0, 0.0, 0.0),) * 21,
                response_target=0,
                response_valid=True,
                rank_target=0,
                value_valid=True,
            )

        calibration = calibrate_static_rows(
            [row(split.fit_games[0])],
            [row(split.check_games[0])],
            split=split,
            binding=predictor.binding,
            support=support,
            temperatures=(1.0,),
            quality_policy=quality,
            inference_device_type="cpu",
            source_reference="synthetic production-shaped arrays, not actual model calibration",
        )
        assert calibration.empirical_check_valid
        directory = tmp_path / state
        save_frozen_release(
            directory,
            model,
            train_manifest_sha256="b" * 64,
            use_rule_features=True,
            training_contract=artifact_contract(),
            support=support,
            calibration=calibration,
            options=PolicyOptions("prior_only", math.inf),
        )
        return FrozenRelease(directory)

    return build


@pytest.mark.parametrize("state", ["train_insufficient", "failed", "unavailable"])
def test_finite_tau_rejected_after_metered_preparation_before_matches(prior_release, state, monkeypatch, tmp_path):
    subject = prior_release(state)
    assert subject.calibration.empirical_check_valid
    assert not any(subject.support.value_supported(subject.calibration.value_branch_statuses))
    if state == "train_insufficient":
        assert subject.calibration.value_branch_statuses[0] == "passed"
    else:
        assert subject.support.supported[0] and subject.calibration.value_branch_statuses[0] == state
    opponents = (subject,) * 3
    with pytest.raises(ValueError, match="Finite tau selection requires at least one"):
        validation_policy.validation_tau_plan(subject, opponents, seed=17)

    # Reconstruct an earlier-style finite plan from the permitted baseline plan
    # to exercise the execution entry's recheck, not just a direct planner call.
    baseline = validation_policy.validation_tau_plan(subject, opponents, seed=17, grid=(math.inf,))
    plan = {key: value for key, value in baseline.items() if key != "plan_sha256"}
    plan.update(grid=["infinity", 0.5], matches=2 * baseline["matches"])
    digest = content_hash(plan)
    forecast = CostForecast(
        "M2", plan["matches"], 1.0, 0.0, 0.0, "synthetic rejection-only fixture, no matches executed", digest
    )
    budget = BudgetPlan("M2", 650.0, 0.0, forecast)
    authorization = {
        "version": "explicit_frozen_evaluation_authorization_v1",
        "explicit_user_approval": True,
        "approval_reference": "synthetic guard-only test, no actual evaluation authorization",
        "allowed_operations": ["select_validation_tau"],
        "plan_sha256": digest,
        "source_sha256": update_implementation_sha256(),
    }
    authorization["authorization_sha256"] = content_hash(authorization)

    def forbidden(*args, **kwargs):
        raise AssertionError("A value-inactive finite grid must not run a match")

    monkeypatch.setattr(validation_policy, "run_match", forbidden)
    output = tmp_path / "rejected-selection"
    with pytest.raises(ValueError, match="Finite tau selection requires at least one"):
        validation_policy.select_validation_tau(
            subject,
            opponents,
            output,
            budget=budget,
            seed=17,
            plan_sha256=digest,
            authorization=authorization,
            grid=(math.inf, 0.5),
        )
    assert (output / "startup.json").is_file()
    ledger = BudgetLedger(output / "costs.jsonl", budget)
    assert not ledger.pending_compute and ledger.entries[-1]["reference"].endswith("/aborted")
    assert not (output / "validation_matches.jsonl").exists()


def test_qualified_branch_keeps_finite_grid_available(prior_release):
    subject = prior_release("passed")
    assert sum(subject.support.value_supported(subject.calibration.value_branch_statuses)) == 1
    plan = validation_policy.validation_tau_plan(subject, (subject,) * 3, seed=17)
    assert plan["grid"] == ["infinity", 1.0, 0.5, 0.25]
    assert plan["matches"] == 4 * len(subject.calibration.split.fit_games)


def test_explicit_baseline_grid_without_value_support_cannot_claim_finite_gain(prior_release):
    subject = prior_release("failed")
    policy = TauSelectionPolicy(bootstrap_replicates=20)
    plan = validation_policy.validation_tau_plan(
        subject, (subject,) * 3, seed=17, grid=(math.inf,), selection_policy=policy
    )
    assert plan["grid"] == ["infinity"]
    assert plan["matches"] == len(subject.calibration.split.fit_games)
    # Only synthetic supplied outcomes; this does not execute any complete game.
    selection = choose_validation_tau(
        {math.inf: {game: 0.5 for game in subject.calibration.split.fit_games}},
        split=subject.calibration.split,
        binding=subject.calibration.binding,
        grid=(math.inf,),
        policy_identity=plan["policy_identity"],
        selection_policy=policy,
    )
    assert selection["selected_tau"] == "infinity" and not selection["reproducible_gain"]
