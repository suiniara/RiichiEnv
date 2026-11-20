"""Frozen arithmetic checks cover auxiliary heads and full validation forwards."""

# ruff: noqa: E402 -- Torch is optional outside explicit ML tests.
import sqlite3
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from test_training_contracts import SyntheticDataset

from riichienv.kar.budget import BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.learning import calibration_job
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.distributed import DistributedContext, DistributedStageError
from riichienv.kar.learning.evaluation import evaluate_complete
from riichienv.kar.learning.inference import FrozenModelPredictor
from riichienv.kar.learning.model import KARTransformer


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "ieee")
    torch.set_num_threads(1)
    torch.manual_seed(61)
    return KARTransformer(ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="backend-guard-test"))


@pytest.mark.parametrize("during_forward", [False, True])
def test_archive_refuses_bad_backend_before_recording_auxiliary_logits(model, monkeypatch, tmp_path, during_forward):
    dataset = SyntheticDataset()
    predictor = FrozenModelPredictor(model, dataset.manifest["manifest_sha256"])
    monkeypatch.setattr(calibration_job, "dataset_event_support", lambda _data: "synthetic-support")
    monkeypatch.setattr(
        calibration_job,
        "calibration_split_from_manifest",
        lambda _manifest: SimpleNamespace(fit_games=("synthetic-game-0",)),
    )
    forward_calls = []
    original = model.forward

    def change_during_forward(batch):
        forward_calls.append(torch.is_autocast_enabled("cpu"))
        outputs = original(batch)
        monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "bf16")
        return outputs

    monkeypatch.setattr(model, "forward", change_during_forward)
    if not during_forward:
        monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "bf16")
    ledger = BudgetLedger(
        tmp_path / "costs.jsonl",
        BudgetPlan("M2", 0.0, 0.0, CostForecast("M2", 3, 1.0, 0.0, 0.0, "synthetic guarded forward", "a" * 64)),
    )
    with (
        torch.no_grad(),
        ledger.record_compute("synthetic-precision-guard"),
        pytest.raises(ValueError, match="IEEE FP32"),
    ):
        calibration_job._archive_and_fit(
            predictor,
            dataset,
            {},
            tmp_path,
            ledger=ledger,
            quality_policy=None,
            temperatures=(1.0,),
            microbatch=2,
            plan_sha256="a" * 64,
        )
    with sqlite3.connect(tmp_path / "logits.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM logits").fetchone() == (0,)
    assert forward_calls == ([False] if during_forward else [])
    assert torch.backends.mkldnn.matmul.fp32_precision == "bf16"
    assert all(parameter.grad is None for parameter in model.parameters())


def test_uncalibrated_fp32_validation_checks_backend_but_bf16_recipe_remains_available(model, monkeypatch):
    dataset = SyntheticDataset()
    dataset.split = "validation"
    context = DistributedContext.open("cpu")
    model.train()
    monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "bf16")
    kwargs = {"microbatch": 2, "use_rule_features": True, "bootstrap_replicates": 5}
    try:
        with pytest.raises(DistributedStageError, match="IEEE FP32"):
            evaluate_complete(model, dataset, context, precision="fp32", **kwargs)
        assert model.training
        report = evaluate_complete(model, dataset, context, precision="bf16", **kwargs)
        assert report["full_validation_split_evaluated"] and report["evaluated_examples"] == len(dataset)
        assert model.training
    finally:
        context.close()
    assert torch.backends.mkldnn.matmul.fp32_precision == "bf16"
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fp32_validation_rejects_backend_changed_inside_forward(model, monkeypatch):
    dataset = SyntheticDataset()
    dataset.split = "validation"
    context = DistributedContext.open("cpu")
    original = model.forward

    def change_during_forward(batch):
        outputs = original(batch)
        monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "bf16")
        return outputs

    monkeypatch.setattr(model, "forward", change_during_forward)
    try:
        with pytest.raises(DistributedStageError, match="IEEE FP32"):
            evaluate_complete(model, dataset, context, microbatch=2, use_rule_features=True, bootstrap_replicates=5)
    finally:
        context.close()
