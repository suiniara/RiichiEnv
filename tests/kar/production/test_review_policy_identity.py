from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from riichienv.kar.calibration import (
    CalibrationArtifact,
    CalibrationRow,
    calibrate_static_rows,
    calibrated_policy_identity,
    calibration_split,
    choose_validation_tau,
)
from riichienv.kar.calibration_quality import BranchQualityCriterion, BranchQualityPolicy, CalibrationQualityPolicy
from riichienv.kar.contracts import content_hash, game_split
from riichienv.kar.learning.deployment import validate_frozen_policy
from riichienv.kar.policy import PolicyOptions
from riichienv.kar.runtime_contracts import RuntimeBinding
from riichienv.kar.selection_quality import TauSelectionPolicy
from riichienv.kar.support import EventSupport

# ruff: noqa: E402 -- optional ML dependency is checked before the deployment import


def policy_fixture():
    identities = []
    index = 0
    while len(identities) < 200:
        value = f"synthetic-policy-contract-{index}"
        if game_split(value, "kar-riichienv-v1-games") == "validation":
            identities.append(value)
        index += 1
    split = replace(
        calibration_split(identities), validation_manifest_sha256="a" * 64, main_split_salt="kar-riichienv-v1-games"
    )
    binding = RuntimeBinding.current("b" * 64, "c" * 64)
    support = EventSupport("c" * 64, (500,) * 21, "synthetic contract support")
    quality = CalibrationQualityPolicy(
        {key: 5.0 for key in ("action", "response", "rank")},
        {key: 1.0 for key in ("action", "response", "rank")},
        {key: 0.5 for key in ("action", "response", "rank")},
        1,
        1,
        0.8,
        "synthetic preregistration",
        BranchQualityPolicy(
            tuple(BranchQualityCriterion(head, 0, 5.0, 1.0, 0.5, 1) for head in ("response", "rank")),
            "synthetic branch preregistration",
        ),
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

    artifact = calibrate_static_rows(
        [row(split.fit_games[0])],
        [row(split.check_games[0])],
        split=split,
        binding=binding,
        support=support,
        temperatures=(1.0,),
        source_reference="synthetic production-shaped logits,not actual model calibration",
        quality_policy=quality,
        inference_device_type="cpu",
    )
    identity = calibrated_policy_identity(binding, support, artifact)
    outcomes = {float("inf"): {game: -0.5 for game in split.fit_games}, 0.5: {game: 0.5 for game in split.fit_games}}
    selection = choose_validation_tau(
        outcomes,
        split=split,
        binding=binding,
        grid=(float("inf"), 0.5),
        policy_identity=identity,
        selection_policy=TauSelectionPolicy(bootstrap_replicates=20),
    )
    return binding, support, artifact, selection


def test_finite_release_rejects_changed_calibration_or_sampling_policy():
    binding, support, artifact, selection = policy_fixture()
    validate_frozen_policy(
        PolicyOptions("empirical_only", 0.5), binding, support, artifact, selection, inference_device_type="cpu"
    )
    changed = replace(artifact, temperatures=(2.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        validate_frozen_policy(
            PolicyOptions("empirical_only", 0.5), binding, support, changed, selection, inference_device_type="cpu"
        )
    with pytest.raises(ValueError):
        validate_frozen_policy(
            PolicyOptions("empirical_only", 0.5, diagnostic_argmax=True),
            binding,
            support,
            artifact,
            selection,
            inference_device_type="cpu",
        )


def test_calibration_origin_backend_cannot_be_restamped_by_export():
    binding, support, artifact, selection = policy_fixture()
    with pytest.raises(ValueError, match="backend"):
        validate_frozen_policy(PolicyOptions(), binding, support, artifact, None, inference_device_type="cuda")


def test_production_tau_selection_and_deployment_reject_a_cal_fit_subset():
    binding, support, artifact, selection = policy_fixture()
    subset = artifact.split.fit_games[:-1]
    outcomes = {float("inf"): dict.fromkeys(subset, -0.5), 0.5: dict.fromkeys(subset, 0.5)}
    arguments = {
        "binding": binding,
        "grid": (float("inf"), 0.5),
        "policy_identity": calibrated_policy_identity(binding, support, artifact),
        "selection_policy": TauSelectionPolicy(bootstrap_replicates=20),
    }
    with pytest.raises(ValueError, match="complete registered cal_fit"):
        choose_validation_tau(outcomes, split=artifact.split, **arguments)
    # Even a self-consistent, rehashed selection on a shortened source split
    # must not pass export against the actual frozen calibration split.
    shortened = replace(
        artifact.split,
        fit_games=subset,
        validation_games=tuple(sorted((*subset, *artifact.split.check_games))),
    )
    partial = choose_validation_tau(outcomes, split=shortened, **arguments)
    assert partial["reproducible_gain"] is True
    with pytest.raises(ValueError, match="complete registered cal_fit"):
        validate_frozen_policy(
            PolicyOptions("empirical_only", 0.5),
            binding,
            support,
            artifact,
            partial,
            inference_device_type="cpu",
        )


def test_deployment_does_not_promote_a_rehashed_legacy_selection():
    binding, support, artifact, selection = policy_fixture()
    body = {key: value for key, value in selection.items() if key != "tau_selection_sha256"}
    body["version"] = "frozen_validation_tau_grid_v3"
    legacy = {**body, "tau_selection_sha256": content_hash(body)}
    with pytest.raises(ValueError, match="production validation-policy selection"):
        validate_frozen_policy(
            PolicyOptions("empirical_only", 0.5),
            binding,
            support,
            artifact,
            legacy,
            inference_device_type="cpu",
        )


def test_preregistered_quality_policy_is_required_and_quality_failure_is_persisted():
    binding, support, artifact, selection = policy_fixture()
    bad = {key: dict(value) if hasattr(value, "items") else value for key, value in artifact.check_metrics.items()}
    bad["rank"] = {**bad["rank"], "nll": 500.0}
    rejected = replace(artifact, check_metrics=bad)
    document = rejected.to_record()
    assert document["quality_report"]["status"] == "failed" and not document["empirical_check_valid"]
    assert CalibrationArtifact.from_record(document, binding, support).quality_report["status"] == "failed"
    with pytest.raises(ValueError):
        validate_frozen_policy(
            PolicyOptions("empirical_only", 0.5), binding, support, rejected, selection, inference_device_type="cpu"
        )
    with pytest.raises(ValueError):
        replace(artifact, quality_policy=None).validate(binding, support)
