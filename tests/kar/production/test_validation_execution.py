# ruff: noqa: E402 -- optional Torch must be checked before ML imports
import json
import os
import socket
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from test_training_contracts import SyntheticDataset

from riichienv.kar.calibration import CalibrationRow, calibrate_static_rows, calibration_split
from riichienv.kar.calibration_quality import CalibrationQualityPolicy
from riichienv.kar.calibration_roles import calibration_is_fit
from riichienv.kar.contracts import content_hash, game_split
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.distributed import DistributedContext, DistributedStageError
from riichienv.kar.learning.evaluation import evaluate_complete
from riichienv.kar.learning.inference import FrozenModelPredictor
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.statistics import FullDiagnostics
from riichienv.kar.support import EventSupport


class _Validation(SyntheticDataset):
    split = "validation"


def _model():
    torch.manual_seed(61)
    return KARTransformer(ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="synthetic-evaluation"))


def _calibration(model, dataset):
    # Production-shaped boundary fixture using synthetic fixed logits only.
    salt = "kar-riichienv-v1-games"
    games = [f"synthetic-eval-{i}" for i in range(1000)]
    val = [game for game in games if game_split(game, salt) == "validation"]
    fit = next(game for game in val if calibration_is_fit(game))
    check = next(game for game in val if not calibration_is_fit(game))
    split = replace(calibration_split((fit, check)), validation_manifest_sha256="d" * 64, main_split_salt=salt)
    binding = FrozenModelPredictor(model, dataset.manifest["manifest_sha256"]).binding
    support = EventSupport(binding.train_manifest_sha256, (500,) * 21, "fixture only")
    sample = dataset[0]

    def row(game):
        return CalibrationRow(
            game, game + "/row", sample["candidates"], (0.0,) * 546, sample["action_target"],
            response_logits=(0.0,) * 21, response_allowed=sample["response_allowed"],
            response_target=0, response_valid=True, rank_logits=((2.0, 0.0, 0.0, 0.0),) * 21,
            rank_target=0, value_valid=True,
        )

    heads = ("action", "response", "rank")
    quality = CalibrationQualityPolicy(dict.fromkeys(heads, 10.0), dict.fromkeys(heads, 2.0),
                                       dict.fromkeys(heads, 1.0), 1, 1, 0.0, "fixture only")
    artifact = calibrate_static_rows(
        [row(fit)], [row(check)], split=split, binding=binding, support=support, temperatures=(1.0,),
        quality_policy=quality, inference_device_type="cpu", source_reference="synthetic FP32 boundary fixture",
    )
    return artifact, support


def test_calibrated_evaluation_is_fp32_under_ambient_bf16_and_refuses_requested_bf16(monkeypatch):
    model, dataset = _model(), _Validation()
    artifact, support = _calibration(model, dataset)
    model.train()
    context = DistributedContext.open("cpu")
    original = model.forward
    states = []

    def inspect(batch):
        states.append(torch.is_autocast_enabled("cpu"))
        return original(batch)

    monkeypatch.setattr(model, "forward", inspect)
    kwargs = {"microbatch": 2, "use_rule_features": True, "calibration": artifact, "support": support,
              "bootstrap_replicates": 5}
    try:
        expected = evaluate_complete(model, dataset, context, **kwargs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            observed = evaluate_complete(model, dataset, context, **kwargs)
        with pytest.raises(ValueError, match="Calibrated validation requires frozen FP32"):
            evaluate_complete(model, dataset, context, precision="bf16", **kwargs)
    finally:
        context.close()
    assert model.training and states == [False] * 4
    assert expected.pop("timings_seconds") and observed.pop("timings_seconds") and observed == expected


def _worker(rank, port, directory, fail):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="-1")
    torch.set_num_threads(1)
    report_calls = 0
    original = FullDiagnostics.report

    def counted(self, **kwargs):
        nonlocal report_calls
        report_calls += 1
        if fail and rank == 0:
            raise ArithmeticError("synthetic primary report failure")
        return original(self, **kwargs)

    FullDiagnostics.report = counted
    context = DistributedContext.open("cpu", backend="gloo", gloo_bind_host="127.0.0.1")
    try:
        report = evaluate_complete(_model(), _Validation(), context, microbatch=1, use_rule_features=True,
                                   bootstrap_replicates=17)
        outcome = {"examples": report["evaluated_examples"], "digest": content_hash(report),
                   "timings": report.get("timings_seconds", {})}
    except Exception as error:
        outcome = {"error_type": type(error).__name__, "error": str(error),
                   "distributed_error": isinstance(error, DistributedStageError)}
    finally:
        context.close()
    (Path(directory) / f"rank-{rank}.json").write_text(json.dumps({"calls": report_calls, **outcome}), encoding="utf-8")


@pytest.mark.parametrize("fail", [False, True])
def test_full_validation_only_primary_reports_broadcasts_and_coordinates_failure(tmp_path, fail, record_property):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    spawn = get_context("spawn")
    workers = [spawn.Process(target=_worker, args=(rank, port, str(tmp_path), fail)) for rank in range(2)]
    started = []
    try:
        for worker in workers:
            worker.start()
            started.append(worker)
        for worker in workers:
            worker.join(45)
        assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
    finally:
        for worker in started:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)
    first, second = [json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    record_property("rank_report_calls", json.dumps([first["calls"], second["calls"]]))
    assert (first["calls"], second["calls"]) == (1, 0), (first, second)
    if fail:
        assert first["distributed_error"] and second["distributed_error"]
        assert "synthetic primary report failure" in first["error"]
        assert "global report" in second["error"] and "another rank failed" in second["error"]
    else:
        assert first["examples"] == second["examples"] == 3
        assert first["digest"] == second["digest"] and first["timings"] == second["timings"]
        assert set(first["timings"]) == {"forward", "aggregation", "bootstrap"}
        assert all(duration >= 0 for duration in first["timings"].values())
        record_property("validation_timing_seconds", json.dumps(first["timings"]))
