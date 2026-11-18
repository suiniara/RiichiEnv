"""Synthetic preregistered criteria only: no measured model-calibration claims."""
import copy
import math
from dataclasses import replace

import pytest

from riichienv.kar.calibration import CalibrationArtifact, CalibrationRow, calibrate_static_rows, calibration_split
from riichienv.kar.calibration_quality import (
    BranchQualityCriterion,
    BranchQualityPolicy,
    CalibrationQualityPolicy,
)
from riichienv.kar.contracts import content_hash
from riichienv.kar.policy import PolicyOptions
from riichienv.kar.probability import masked_softmax
from riichienv.kar.runtime import RuntimePolicy
from riichienv.kar.support import EventSupport, candidate_value


def branch_policy(minimum_games=1):
    # Deliberately synthetic test thresholds, NOT production defaults.
    return BranchQualityPolicy(tuple(
        BranchQualityCriterion(head, event, 5.0 if head == "response" else 1.0, 2.0, 1.0, minimum_games)
        for head in ("response", "rank") for event in (0, 8)
    ), "synthetic criteria fixed before generating fixture rows;not real calibration")


def artifact_fixture(binding, support, public_view, *, bad=True, minimum_games=1):
    _, ids = public_view
    split = calibration_split([f"branch-validation-{index}" for index in range(20)])

    def row(game, index, event):
        ranks = [(8.0, 0.0, 0.0, 0.0)] * 21
        if bad:
            ranks[8] = (0.0, 8.0, 0.0, 0.0)
        return CalibrationRow(game, f"{game}/{index}", ids, (0.0,) * 546, ids[0],
                              response_logits=(0.0,) * 21, rank_logits=tuple(ranks),
                              response_allowed=(True,) * 21, response_target=event, response_valid=True,
                              rank_target=0, value_valid=True)

    fit = [row(split.fit_games[0], 0, 0)]
    check = [row(split.check_games[0], index, 0 if index < 100 else 8) for index in range(101)]
    quality = CalibrationQualityPolicy(
        {"action": 10.0, "response": 5.0, "rank": 1.0}, {head: 2.0 for head in ("action", "response", "rank")},
        {head: 1.0 for head in ("action", "response", "rank")}, 1, 1, 0.0,
        "synthetic overall criteria,not production thresholds", branch_policy(minimum_games),
    )
    return calibrate_static_rows(fit, check, split=split, binding=binding, support=support,
                                 temperatures=(1.0,), source_reference="synthetic logits only",
                                 diagnostic_fixture=True, quality_policy=quality)


def test_overall_pass_does_not_authorize_unregistered_branch_quality(
    calibration_fixture, fixed_predictor, event_support, public_view
):
    original, fit, check = calibration_fixture
    assert original.quality_report["status"] == "passed"
    artifact = calibrate_static_rows(
        fit, check, split=original.split, binding=original.binding, support=event_support,
        temperatures=original.temperatures[:1], source_reference="synthetic, no branch preregistration",
        diagnostic_fixture=True, quality_policy=replace(original.quality_policy, branch_policy=None),
    )
    assert artifact.quality_report["status"] == "passed"
    view, ids = public_view
    result = RuntimePolicy(fixed_predictor, options=PolicyOptions("empirical_only", 0.5),
                           support=event_support, calibration=artifact,
                           allow_diagnostic_calibration=True).decide(view, ids, "missing-branch-criteria")
    assert result.telemetry["candidate_forward_count"] == 0
    assert not result.telemetry["distribution_changed"]
    assert result.fallback_reason == "no_quality_qualified_response_branches"
    assert artifact.value_branch_statuses == ("unavailable",) * 21


def test_overall_pass_rare_rank_branch_failure_keeps_probability_but_cannot_supply_value(
    binding, event_support, public_view, fixed_predictor, monkeypatch, record_property
):
    artifact = artifact_fixture(binding, event_support, public_view)
    assert artifact.quality_report["status"] == "passed"
    assert artifact.check_metrics["rank"]["nll"] < 0.081
    assert artifact.check_metrics["branch_metrics"]["rank"]["8"]["nll"] > 8
    assert artifact.value_branch_statuses[0] == "passed" and artifact.value_branch_statuses[8] == "failed"
    record_property("overall_rank_nll", artifact.check_metrics["rank"]["nll"])
    record_property("rare_rank_nll", artifact.check_metrics["branch_metrics"]["rank"]["8"]["nll"])
    view, ids = public_view
    policy_logits = [0.0] * 546
    policy_logits[ids[-1]] = -20.0  # a real legal action outside S, not a Top-K truncation
    monkeypatch.setattr(fixed_predictor, "policy", lambda *_args: policy_logits)

    def decide():
        return RuntimePolicy(fixed_predictor, options=PolicyOptions("empirical_only", 0.5),
                             support=event_support, calibration=artifact,
                             allow_diagnostic_calibration=True, seed=3).decide(view, ids, "branch-failure")

    first = decide()
    original = fixed_predictor.candidates

    def poisoned(*args):
        values = original(*args)
        return {action: (response, tuple((-1000.0, -1000.0, -1000.0, 1000.0) if event == 8 else row
                                        for event, row in enumerate(ranks)))
                for action, (response, ranks) in values.items()}

    monkeypatch.setattr(fixed_predictor, "candidates", poisoned)
    second = decide()
    assert first.probabilities == second.probabilities and first.action_id == second.action_id
    assert first.evidence_status == "empirical_only" and first.telemetry["bound_E2"] is None
    for details in first.telemetry["candidates"].values():
        assert details["rule_allowed"][8] and details["train_supported"][8]
        assert details["branch_quality_statuses"][8] == "failed" and not details["value_supported"][8]
        assert details["response_probabilities"][8] > 0
        expected = masked_softmax((0.0,) * 21, details["rule_allowed"])
        assert details["response_probabilities"] == expected
        expected_mass = sum(p for p, ok in zip(expected, details["value_supported"]) if not ok)
        assert details["unsupported_mass"] == pytest.approx(expected_mass)
    assert first.probabilities[ids[-1]] == first.telemetry["prior"][str(ids[-1])]
    selected = first.telemetry["supported_discards"]
    assert sum(first.probabilities[action] for action in selected) == pytest.approx(first.telemetry["support_mass"])


@pytest.mark.parametrize("minimum_games,expected", [(1, "passed"), (2, "unavailable")])
def test_branch_evidence_counts_distinct_validation_games_not_rows(
    binding, event_support, public_view, minimum_games, expected
):
    artifact = artifact_fixture(binding, event_support, public_view, bad=False, minimum_games=minimum_games)
    assert artifact.check_metrics["branch_metrics"]["rank"]["0"]["count"] == 100
    assert artifact.check_metrics["branch_metrics"]["rank"]["0"]["independent_games"] == 1
    assert artifact.value_branch_statuses[0] == expected
    assert artifact.value_branch_statuses[8] == expected
    assert artifact.value_branch_statuses[1] == "unavailable"  # no criterion or evidence, never implicit pass


@pytest.mark.parametrize("state", ["passed", "failed", "unavailable"])
@pytest.mark.parametrize("train_games", [499, 500])
@pytest.mark.parametrize("rule_possible", [False, True])
def test_rule_impossibility_training_support_and_quality_are_distinct(binding, state, train_games, rule_possible):
    support = EventSupport(
        binding.train_manifest_sha256, (500,) * 8 + (train_games,) + (500,) * 12, "synthetic train evidence"
    )
    allowed = (True,) + (False,) * 7 + (rule_possible,) + (False,) * 12
    logits = [0.0] * 21
    logits[8] = 20.0
    statuses = ("passed",) * 8 + (state,) + ("passed",) * 12
    value = candidate_value(4, logits, ((8.0, 0.0, 0.0, 0.0),) * 21, allowed, support, branch_quality_statuses=statuses)
    assert value.rule_allowed[8] == rule_possible
    assert value.train_supported[8] == (train_games >= 500)
    assert value.branch_quality_statuses[8] == state
    assert value.value_supported[8] == (train_games >= 500 and state == "passed")
    assert (value.response_probabilities[8] > 0) == rule_possible
    expected_mass = value.response_probabilities[8] if state != "passed" or train_games < 500 else 0.0
    assert value.unsupported_mass == pytest.approx(expected_mass)
    assert sum(value.response_probabilities) == pytest.approx(1)
    if not rule_possible:
        assert value.unsupported_mass == 0


def test_missing_quality_is_conservative_even_when_all_train_counts_pass(binding):
    support = EventSupport(binding.train_manifest_sha256, (500,) * 21, "synthetic")
    value = candidate_value(4, (0.0,) * 21, ((8.0, 0.0, 0.0, 0.0),) * 21, (True,) * 21, support)
    assert value.conservative_value == pytest.approx(-1.5) and value.unsupported_mass == pytest.approx(1)


@pytest.mark.parametrize("missing", ["branch_quality_report", "value_branch_statuses", "branch_policy"])
def test_checksum_valid_missing_new_calibration_contract_is_rejected(binding, event_support, public_view, missing):
    artifact = artifact_fixture(binding, event_support, public_view)
    forged = copy.deepcopy(artifact.to_record())
    if missing == "branch_policy":
        forged["quality_policy"].pop(missing)
        unsigned = {key: value for key, value in forged["quality_policy"].items() if key != "quality_policy_sha256"}
        forged["quality_policy"]["quality_policy_sha256"] = content_hash(unsigned)
    else:
        forged.pop(missing)
    unsigned = {key: value for key, value in forged.items() if key != "calibration_sha256"}
    forged["calibration_sha256"] = content_hash(unsigned)
    with pytest.raises(ValueError):
        CalibrationArtifact.from_record(forged, binding, event_support, allow_diagnostic=True)


def test_branch_preregistration_round_trip_and_mutation_rejection():
    policy = branch_policy()
    assert BranchQualityPolicy.from_record(policy.to_record()) == policy
    changed = copy.deepcopy(policy.to_record())
    changed["criteria"][0]["minimum_independent_games"] += 1
    with pytest.raises(ValueError):
        BranchQualityPolicy.from_record(changed)
    with pytest.raises(ValueError):
        BranchQualityPolicy((policy.criteria[0], policy.criteria[0]), "synthetic")
    assert math.isfinite(policy.criteria[0].max_nll)


@pytest.mark.parametrize("missing", ["branch_quality_policy_sha256", "value_branch_statuses"])
def test_finite_policy_identity_cannot_omit_branch_contract(binding, event_support, public_view, missing):
    from riichienv.kar.calibration import calibrated_policy_identity, validate_policy_identity  # noqa: PLC0415

    artifact = artifact_fixture(binding, event_support, public_view)
    record = calibrated_policy_identity(binding, event_support, artifact)
    validate_policy_identity(record, binding)
    record.pop(missing)
    record["policy_identity_sha256"] = content_hash({key: value for key, value in record.items()
                                                  if key != "policy_identity_sha256"})
    with pytest.raises(ValueError, match="branch-quality contract"):
        validate_policy_identity(record, binding)
