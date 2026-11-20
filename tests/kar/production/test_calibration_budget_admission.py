"""Calibration admission/accounting only; no network forward or parameter update."""

# ruff: noqa: E402 -- Torch is optional outside the explicit ML test job.
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from test_training_contracts import SyntheticDataset

from riichienv.kar import budget as budget_module
from riichienv.kar.authorization import update_implementation_sha256
from riichienv.kar.budget import BudgetExceededError, BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.calibration_quality import CalibrationQualityPolicy
from riichienv.kar.calibration_roles import calibration_is_fit
from riichienv.kar.contracts import content_hash, game_split
from riichienv.kar.learning import calibration_job
from riichienv.kar.learning import dataset as dataset_module
from riichienv.kar.learning.dataset import DecisionDataset
from riichienv.kar.learning.export import SQL_SCHEMA, learning_contract
from riichienv.kar.learning.model import ModelOutputs
from riichienv.kar.learning.precision import capture_frozen_backend_precision, frozen_fp32
from riichienv.kar.runtime_contracts import RuntimeBinding
from riichienv.kar.versioning import game_count_contract


class ComputeEntryReachedError(RuntimeError):
    pass


@pytest.fixture
def calibration_request(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    salt = "kar-riichienv-v1-games"
    names = [f"calibration-admission-{index}" for index in range(1000)]
    validation_games = [game for game in names if game_split(game, salt) == "validation"]
    chosen = {
        "train": next(game for game in names if game_split(game, salt) == "train"),
        "fit": next(game for game in validation_games if calibration_is_fit(game)),
        "check": next(game for game in validation_games if not calibration_is_fit(game)),
    }
    available = {"train": 1, "validation": 2, "test": 0}
    scope = {
        **learning_contract(),
        "protocol_sha256": "d" * 64,
        "protocol_descriptor_sha256": "e" * 64,
        "available_split_counts": available,
        "split_salt": salt,
    }
    path = data / "decisions.sqlite3"
    games = []
    with sqlite3.connect(path) as db:
        db.executescript(SQL_SCHEMA)
        db.executemany("INSERT INTO metadata VALUES (?,?)", [("complete", "true"), ("scope", json.dumps(scope))])
        for role, game in chosen.items():
            split = "train" if role == "train" else "validation"
            games.append({"game_id": game, "split": split})
            db.execute("INSERT INTO games VALUES (?,?,?,?,?)", (game, split, "a" * 64, "[1,2,3,4]", "b" * 64))
            # This fixture tests the real dataset index/count contract. Computation
            # is blocked before these deliberately metadata-only rows are fetched.
            for ordinal in range(1 if split == "train" else 2):
                row = (
                    f"{game}/row{ordinal}",
                    game,
                    0,
                    0,
                    ordinal,
                    0,
                    0,
                    0,
                    bytes(63 * 139),
                    bytes(139),
                    bytes(64),
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    bytes([1] * 21),
                    "synthetic_metadata",
                    "{}",
                )
                db.execute("INSERT INTO samples VALUES (" + ",".join("?" for _ in row) + ")", row)
    manifest = {
        "status": "PASS",
        "complete": True,
        "scope": scope,
        **game_count_contract(available),
        "games": games,
        "rho_E": 1.0,
        "rho_V": 1.0,
        "database_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "counts": {
            "train": {"games": 1, "examples": 1},
            "validation": {"games": 2, "examples": 4},
            "test": {"games": 0, "examples": 0},
        },
    }
    manifest["manifest_sha256"] = content_hash(manifest)
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    selected = [{"game_id": chosen[role], "source_role": "main_cohort_candidate"} for role in ("fit", "check")]
    validation_manifest = {"games": selected, "scope": scope, "manifest_sha256": content_hash(selected)}
    predictor = SimpleNamespace(
        _unchanged=lambda: None,
        binding=RuntimeBinding.current(
            model_weights_sha256="c" * 64, train_manifest_sha256=manifest["manifest_sha256"]
        ),
        model=SimpleNamespace(parameters=lambda: iter((torch.empty(0),))),
        use_rule_features=True,
        backend_precision=capture_frozen_backend_precision("cpu"),
    )
    heads = ("action", "response", "rank")
    quality = CalibrationQualityPolicy(
        dict.fromkeys(heads, 5.0),
        dict.fromkeys(heads, 2.0),
        dict.fromkeys(heads, 1.0),
        1,
        1,
        0.0,
        "synthetic admission criteria, no actual model execution",
    )
    plan = calibration_job.calibration_job_plan(
        predictor,
        manifest,
        validation_manifest,
        quality_policy=quality,
        temperatures=(1.0,),
    )
    digest = plan["plan_sha256"]
    authorization = {
        "version": "explicit_frozen_evaluation_authorization_v1",
        "explicit_user_approval": True,
        "approval_reference": "synthetic admission regression with blocked computation",
        "allowed_operations": ["calibrate_frozen_logits"],
        "plan_sha256": digest,
        "source_sha256": update_implementation_sha256(),
    }
    authorization["authorization_sha256"] = content_hash(authorization)
    reached = []

    def stop_before_compute(_predictor, validation, *args, **kwargs):
        assert isinstance(validation, DecisionDataset)
        reached.append(len(validation))
        raise ComputeEntryReachedError("no network forward or calibration is executed")

    monkeypatch.setattr(calibration_job, "_archive_and_fit", stop_before_compute)
    output = tmp_path / "job"

    def run(work_units, *, phase="M2", recipe=None, cap=10.0, microbatch=1):
        forecast = CostForecast(phase, work_units, 1.0, 3600.0, 0.0, "synthetic row rate", recipe or digest, 1.0)
        budget = BudgetPlan(phase, cap, 0.0, forecast)
        return calibration_job.collect_and_calibrate(
            predictor,
            data,
            validation_manifest,
            output,
            budget=budget,
            plan_sha256=digest,
            authorization=authorization,
            temperatures=(1.0,),
            microbatch=microbatch,
            quality_policy=quality,
        )

    return run, output, reached


@pytest.mark.parametrize("phase", ["M1", "M2"])
@pytest.mark.parametrize("work_units,cap", [(1, 1.5), (3, 3.5)])
def test_incomplete_workload_is_rejected_after_metered_preparation_before_forward(
    calibration_request, phase, work_units, cap
):
    run, output, reached = calibration_request
    with pytest.raises(ValueError, match="every validation decision row"):
        run(work_units, phase=phase, cap=cap)
    assert reached == [] and (output / "startup.json").is_file()
    entries = [json.loads(line) for line in (output / "costs.jsonl").read_text().splitlines()]
    assert entries[-1]["closed"] and entries[-1]["reference"].endswith("/aborted")
    assert not (output / "logits.sqlite3").exists()


@pytest.mark.parametrize("phase,work_units", [("M1", 4), ("M2", 4), ("M2", 5)])
def test_forecast_covering_all_actual_rows_reaches_compute(calibration_request, phase, work_units):
    run, output, reached = calibration_request
    with pytest.raises(ComputeEntryReachedError):
        run(work_units, phase=phase)
    assert reached == [4] and (output / "costs.jsonl").is_file()


@pytest.mark.parametrize("overrides", [{"phase": "M3"}, {"recipe": "f" * 64}])
def test_complete_workload_still_requires_the_calibration_phase_and_recipe(calibration_request, overrides):
    run, output, reached = calibration_request
    with pytest.raises(ValueError, match="registered full-validation budget"):
        run(4, **overrides)
    assert reached == [] and not output.exists()


@pytest.mark.parametrize("fail_hash", [False, True])
def test_real_database_preparation_is_durably_metered(calibration_request, monkeypatch, fail_hash):
    run, output, reached = calibration_request
    clock = [0.0]
    monkeypatch.setattr(budget_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original_hash = dataset_module._file_sha

    def measured_hash(path):
        assert (output / "startup.json").is_file()
        entries = [json.loads(line) for line in (output / "costs.jsonl").read_text().splitlines()]
        assert entries[-1]["kind"] == "compute_reservation"
        clock[0] += 10.0
        if fail_hash:
            raise RuntimeError("synthetic hash read failure")
        return original_hash(path)

    def archive(*args, **kwargs):
        reached.append("forward-blocked")
        clock[0] += 1.0
        raise ComputeEntryReachedError("no model execution")

    monkeypatch.setattr(dataset_module, "_file_sha", measured_hash)
    monkeypatch.setattr(calibration_job, "_archive_and_fit", archive)
    with pytest.raises(RuntimeError, match="synthetic hash read failure" if fail_hash else "no model execution"):
        run(4, cap=20.0)
    entries = [json.loads(line) for line in (output / "costs.jsonl").read_text().splitlines()]
    settlements = [entry for entry in entries if entry["kind"] == "compute_settlement"]
    assert sum(entry["usd"] for entry in settlements) == (10.0 if fail_hash else 11.0)
    assert len({entry["attempt_id"] for entry in settlements}) == 1
    assert settlements[-1]["closed"] and settlements[-1]["reference"].endswith("/aborted")
    assert reached == ([] if fail_hash else ["forward-blocked"])


def test_preparation_overrun_is_charged_once_before_model_forward(calibration_request, monkeypatch):
    run, output, reached = calibration_request
    clock = [0.0]
    monkeypatch.setattr(budget_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original_hash = dataset_module._file_sha

    def measured_hash(path):
        clock[0] += 10.0
        return original_hash(path)

    monkeypatch.setattr(dataset_module, "_file_sha", measured_hash)
    with pytest.raises(BudgetExceededError, match="exhausted this phase"):
        run(4, cap=4.5)
    entries = [json.loads(line) for line in (output / "costs.jsonl").read_text().splitlines()]
    settlements = [entry for entry in entries if entry["kind"] == "compute_settlement"]
    assert sum(entry["usd"] for entry in settlements) == 10.0
    assert settlements[-1]["closed"] and reached == []


def test_last_partial_batch_reserves_actual_rows_with_an_exact_budget(monkeypatch, tmp_path):
    # Three valid synthetic decision rows, two batches, and a fixed-logit stub.
    # A simulated second per row makes an unnecessary fourth reservation fail.
    validation = SyntheticDataset()
    clock = [0.0]
    monkeypatch.setattr(budget_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    forecast = CostForecast("M2", 3, 1.0, 3600.0, 0.0, "synthetic full-row cost", "a" * 64, 1.0)
    budget = BudgetPlan("M2", 3.0, 0.0, forecast)
    ledger = BudgetLedger(tmp_path / "costs.jsonl", budget)
    sizes = []

    class FixedLogits:
        def parameters(self):
            return iter((torch.empty(0),))

        def __call__(self, batch):
            sizes.append(batch.size)
            clock[0] += batch.size
            return ModelOutputs(
                torch.zeros(batch.size, 546),
                torch.zeros(batch.size, 21),
                torch.zeros(batch.size, 21, 4),
                torch.ones(batch.size, dtype=torch.bool),
            )

    predictor = SimpleNamespace(
        model=FixedLogits(),
        _frozen_identity=lambda: "fixed-logit-stub",
        use_rule_features=True,
        binding=None,
        policy_logits_from_rows=lambda rows: torch.zeros(len(rows), 546),
        frozen_forward=lambda: frozen_fp32(capture_frozen_backend_precision("cpu")),
    )
    monkeypatch.setattr(calibration_job, "dataset_event_support", lambda _data: "synthetic-support")
    monkeypatch.setattr(
        calibration_job,
        "calibration_split_from_manifest",
        lambda _manifest: SimpleNamespace(fit_games=("synthetic-game-0",)),
    )
    partitions = []

    def inspect_archive(fit, check, **kwargs):
        partitions.append((len(list(fit)), len(list(check))))
        return "synthetic-artifact"

    monkeypatch.setattr(calibration_job, "calibrate_static_rows", inspect_archive)
    with ledger.record_compute("synthetic_calibration_row_accounting"):
        result = calibration_job._archive_and_fit(
            predictor,
            validation,
            {},
            tmp_path,
            ledger=ledger,
            quality_policy=None,
            temperatures=(1.0,),
            microbatch=2,
            plan_sha256="a" * 64,
        )
    assert result == ("synthetic-artifact", "synthetic-support")
    assert sizes == [2, 1] and partitions == [(1, 2)]
    assert ledger.spent_usd == 3.0 and not ledger.pending_compute
    with sqlite3.connect(tmp_path / "logits.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM logits").fetchone() == (3,)
