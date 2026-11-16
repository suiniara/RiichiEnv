from pathlib import Path

import pytest

from riichienv.kar.authorization import (
    ParameterUpdatesNotAuthorizedError,
    UpdateAuthorization,
    require_parameter_updates,
)
from riichienv.kar.readiness import CORE_REQUIREMENTS, feature_readiness, validate_readiness


def test_data_and_core_readiness_never_grant_update_permission():
    report = feature_readiness(
        {key: True for key in CORE_REQUIREMENTS},
        evidence={key: "functional-test-artifact" for key in CORE_REQUIREMENTS},
        source_hash="fixture-source",
    )
    validate_readiness(report)
    assert report["core_feature_ready"] and not report["full_feature_ready"]
    assert not report["parameter_updates_authorized"]
    with pytest.raises(ParameterUpdatesNotAuthorizedError):
        require_parameter_updates(None, operation="train_cpu")
    with pytest.raises(ParameterUpdatesNotAuthorizedError):
        require_parameter_updates(UpdateAuthorization({"training_allowed": True}), operation="train_cpu")


def test_default_entrypoints_fail_before_any_model_computation_or_output(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from riichienv.kar.learning import gpu_preflight, probe, train  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        raise AssertionError("Model/data execution reached before authorization")

    monkeypatch.setattr(train, "DecisionDataset", forbidden)
    monkeypatch.setattr(train, "KARTransformer", forbidden)
    monkeypatch.setattr(probe, "train_cpu", forbidden)
    monkeypatch.setattr(gpu_preflight, "DecisionDataset", forbidden)
    for call in (
        lambda: train.train_cpu(Path("nonexistent"), tmp_path / "train"),
        lambda: probe.cpu_probe(Path("nonexistent"), tmp_path / "probe"),
        lambda: gpu_preflight.run_preflight(Path("nonexistent")),
        lambda: train.optimizer_update(None, None, [], global_examples=1, rho_e=1.0, rho_v=1.0),
    ):
        with pytest.raises(ParameterUpdatesNotAuthorizedError):
            call()
    assert not list(tmp_path.iterdir())
