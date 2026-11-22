import math
from dataclasses import replace

import pytest

from riichienv.kar.policy import BoundEvidence, PolicyOptions
from riichienv.kar.runtime import RuntimePolicy


def test_unfitted_default_is_prior_only_without_candidate_forward(fixed_predictor, public_view):
    view, ids = public_view
    result = RuntimePolicy(fixed_predictor, seed=5).decide(view, ids, "decision0")
    assert result.evidence_status == "unfitted_prior_only" and fixed_predictor.value_calls == 0
    assert result.telemetry["temperatures"] == [1.0, 1.0, 1.0] and result.telemetry["bound_E2"] is None


@pytest.mark.parametrize("mode", ["empirical_only", "certified"])
def test_infinite_tau_is_exact_prior_and_skips_all_auxiliary_work(
    mode, fixed_predictor, public_view, event_support, calibration_fixture
):
    artifact, _, _ = calibration_fixture
    view, ids = public_view
    baseline = RuntimePolicy(fixed_predictor, seed=6).decide(view, ids, "baseline")
    policy = RuntimePolicy(
        fixed_predictor,
        options=PolicyOptions(mode, math.inf),
        support=event_support,
        calibration=artifact,
        allow_diagnostic_calibration=True,
        seed=6,
    )
    result = policy.decide(view, ids, "infinite")
    assert result.probabilities == baseline.probabilities and result.action_id == baseline.action_id
    assert fixed_predictor.value_calls == 0


def test_missing_bounds_cannot_promote_500_game_support_or_nll_to_certificate(
    fixed_predictor, event_support, calibration_fixture, public_view
):
    artifact, _, _ = calibration_fixture
    view, ids = public_view
    policy = RuntimePolicy(
        fixed_predictor,
        options=PolicyOptions("certified", 0.5),
        support=event_support,
        calibration=artifact,
        allow_diagnostic_calibration=True,
    )
    result = policy.decide(view, ids, "decision")
    assert (
        result.evidence_status == "certificate_unavailable"
        and result.fallback_reason == "error_bound_certificate_unavailable"
    )
    assert fixed_predictor.value_calls == 0 and result.telemetry["bound_E2"] is None


def test_explicit_empirical_mode_scores_all_s_and_has_no_certificate(
    fixed_predictor, event_support, calibration_fixture, public_view
):
    artifact, _, _ = calibration_fixture
    view, ids = public_view
    policy = RuntimePolicy(
        fixed_predictor,
        options=PolicyOptions("empirical_only", 0.5),
        support=event_support,
        calibration=artifact,
        allow_diagnostic_calibration=True,
    )
    result = policy.decide(view, ids, "decision")
    assert result.evidence_status == "empirical_only" and fixed_predictor.value_calls == 1
    assert result.telemetry["candidate_forward_count"] == len([action for action in ids if action < 148])
    assert result.telemetry["bound_E2"] is None
    assert result.probabilities[min(ids)] > 1 / len(ids)


def _production_calibration(calibration_fixture, binding, support, public_view, *, with_candidate_branches=False):
    from riichienv.kar.calibration import (
        CalibrationRow,
        CandidateBranchCalibrationRow,
        calibrate_static_rows,
        calibration_split,
    )
    from riichienv.kar.contracts import game_split

    base, _, _ = calibration_fixture
    _, ids = public_view
    val_games = [
        f"runtime-prod-{index}"
        for index in range(1000)
        if game_split(f"runtime-prod-{index}", "kar-riichienv-v1-games") == "validation"
    ][:20]
    split = replace(
        calibration_split(val_games),
        validation_manifest_sha256="a" * 64,
        main_split_salt="kar-riichienv-v1-games",
    )

    def row(game, event=0):
        return CalibrationRow(
            game,
            game + f"/row{event}",
            ids,
            (0.0,) * 546,
            ids[0],
            response_logits=(0.0,) * 21,
            rank_logits=((0.0,) * 4,) * 21,
            response_allowed=(True,) * 21,
            response_target=event,
            response_valid=True,
            rank_target=0,
            value_valid=True,
        )

    branch_rows = (
        [
            CandidateBranchCalibrationRow(
                split.check_games[0],
                split.check_games[0] + "/row0",
                ids[0],
                0,
                (0.0,) * 21,
                ((0.0,) * 4,) * 21,
                (True,) * 21,
                0,
                0,
            )
        ]
        if with_candidate_branches
        else []
    )
    return calibrate_static_rows(
        [row(split.fit_games[0])],
        [row(split.check_games[0], event) for event in range(21)],
        split=split,
        binding=binding,
        support=support,
        temperatures=(1.0,),
        source_reference="synthetic production-shaped fixture",
        diagnostic_fixture=False,
        quality_policy=base.quality_policy,
        inference_device_type="cpu",
        candidate_branch_check_rows=branch_rows,
    )


def test_finite_policy_falls_back_when_candidate_actions_have_no_branch_supervision(
    fixed_predictor, binding, event_support, calibration_fixture, public_view
):
    artifact = _production_calibration(calibration_fixture, binding, event_support, public_view)
    view, ids = public_view
    policy = RuntimePolicy(
        fixed_predictor,
        options=PolicyOptions("empirical_only", 0.5),
        support=event_support,
        calibration=artifact,
    )
    result = policy.decide(view, ids, "candidate-support-gap")
    assert result.evidence_status == "certificate_unavailable"
    assert result.fallback_reason == "candidate_action_branch_supervision_unavailable"
    assert fixed_predictor.value_calls == 0


def test_certification_requires_bound_scope_and_positive_correct_e2(
    fixed_predictor, event_support, calibration_fixture, public_view, binding
):
    artifact, _, _ = calibration_fixture
    view, ids = public_view
    policy = RuntimePolicy(
        fixed_predictor,
        options=PolicyOptions("certified", 0.5),
        support=event_support,
        calibration=artifact,
        allow_diagnostic_calibration=True,
    )
    selected = tuple(action for action in ids if action < 148)
    evidence = BoundEvidence(
        binding.digest,
        "decision",
        selected,
        (0.0,) * len(selected),
        0.0,
        "synthetic exact-errors fixture",
        "this candidate set at this decision only",
        "deterministic synthetic construction only",
    )
    accepted = policy.decide(view, ids, "decision", bounds=evidence)
    assert accepted.evidence_status == "certified" and accepted.telemetry["bound_E2"] > 0
    rejected = policy.decide(view, ids, "decision", bounds=replace(evidence, b=(3.0,) * len(selected)))
    assert rejected.evidence_status == "certified_bound_fallback"
    foreign = policy.decide(view, ids, "other", bounds=evidence)
    assert (
        foreign.evidence_status == "certificate_unavailable"
        and foreign.fallback_reason == "error_bound_coverage_invalid"
    )


def test_foreign_model_visibility_gap_and_production_fixture_are_rejected(
    fixed_predictor, event_support, calibration_fixture, public_view
):
    artifact, _, _ = calibration_fixture
    view, ids = public_view
    with pytest.raises(ValueError):
        RuntimePolicy(fixed_predictor, support=event_support, calibration=artifact)
    policy = RuntimePolicy(fixed_predictor)
    with pytest.raises(ValueError):
        policy.decide(replace(view, visibility_complete=False), ids, "gap")
    fixed_predictor.binding = replace(fixed_predictor.binding, model_weights_sha256="1" * 64)
    with pytest.raises(ValueError):
        policy.decide(view, ids, "foreign-model")



def test_production_event_calibration_alone_cannot_enable_candidate_value_reranking(
    fixed_predictor, binding, event_support, calibration_fixture, public_view
):
    supported = replace(event_support, candidate_branch_source_decisions=100, candidate_action_source_counts=(50,) * 148)
    artifact = _production_calibration(calibration_fixture, binding, supported, public_view)
    view, ids = public_view
    policy = RuntimePolicy(
        fixed_predictor,
        options=PolicyOptions("empirical_only", 0.5),
        support=supported,
        calibration=artifact,
    )
    result = policy.decide(view, ids, "candidate-calibration-gap")
    assert result.evidence_status == "certificate_unavailable"
    assert result.fallback_reason == "candidate_action_conditional_calibration_unavailable"
    assert fixed_predictor.value_calls == 0


def test_production_candidate_branch_validation_enables_finite_tau_reranking(
    fixed_predictor, binding, event_support, calibration_fixture, public_view
):
    supported = replace(event_support, candidate_branch_source_decisions=100, candidate_action_source_counts=(50,) * 148)
    artifact = _production_calibration(
        calibration_fixture, binding, supported, public_view, with_candidate_branches=True
    )
    assert artifact.branch_quality_report["candidate_action_conditional_validation"]["status"] == "passed"
    view, ids = public_view
    policy = RuntimePolicy(
        fixed_predictor,
        options=PolicyOptions("empirical_only", 0.5),
        support=supported,
        calibration=artifact,
    )
    result = policy.decide(view, ids, "candidate-calibration-passed")
    assert result.evidence_status == "empirical_only"
    assert result.fallback_reason is None
    assert fixed_predictor.value_calls == 1
    assert result.telemetry["candidate_forward_count"] == len([action for action in ids if action < 148])
