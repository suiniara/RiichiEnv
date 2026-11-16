# ruff: noqa: PLC0415 -- ML imports stay behind per-test importorskip
import json
from types import SimpleNamespace

import pytest

from riichienv.kar.budget import BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.contracts import content_hash


def test_preflight_wrong_dataset_is_denied_before_seed_cuda_or_model(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from riichienv.kar.authorization import (
        ParameterUpdatesNotAuthorizedError,
        UpdateAuthorization,
        update_implementation_sha256,
    )
    from riichienv.kar.learning import gpu_preflight
    from riichienv.kar.readiness import CORE_REQUIREMENTS, FULL_REMAINING_REQUIREMENTS, feature_readiness

    checks = {key: True for key in (*CORE_REQUIREMENTS, *FULL_REMAINING_REQUIREMENTS)}
    ready = feature_readiness(
        checks,
        evidence={key: "synthetic guard-only fixture" for key in checks},
        source_hash=update_implementation_sha256(),
        data_eligible=True,
        resource_ready=True,
    )
    document = {
        "version": "kar_explicit_update_authorization_v1",
        "explicit_user_approval": True,
        "approval_reference": "guard-only synthetic test",
        "readiness": ready,
        "allowed_operations": ["run_preflight"],
        "dataset_manifest_sha256": "a" * 64,
    }
    authorization = UpdateAuthorization({**document, "authorization_sha256": content_hash(document)})
    monkeypatch.setattr(
        gpu_preflight,
        "DecisionDataset",
        lambda *args, **kwargs: SimpleNamespace(manifest={"manifest_sha256": "b" * 64}),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("resource/model computation before data authorization")

    monkeypatch.setattr(gpu_preflight.torch, "manual_seed", forbidden)
    monkeypatch.setattr(gpu_preflight.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(gpu_preflight, "KARTransformer", forbidden)
    with pytest.raises(ParameterUpdatesNotAuthorizedError, match="dataset"):
        gpu_preflight.run_preflight(tmp_path / "dataset-B", authorization=authorization)


def test_m3_receipts_require_their_original_cost_history(tmp_path):
    pytest.importorskip("torch")
    from riichienv.kar.learning.simulation import _atomic_record, open_m3_ledger

    digest = "a" * 64
    source = "b" * 64
    estimate = CostForecast("M3", 20000, 100.0, 0.0, 0.0, "synthetic ledger-only evidence", digest)
    budget = BudgetPlan("M3", 250.0, 0.0, estimate)
    ledger_path = tmp_path / "costs.jsonl"
    ledger = BudgetLedger(ledger_path, budget)
    ledger.charge(5.0, category="compute", reference="already completed baseline")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    _atomic_record(
        receipts / "pair-00000-baseline.json",
        {
            "plan_sha256": digest,
            "source_sha256": source,
            "budget_ledger_head": ledger.head,
            "budget_ledger_entries": len(ledger.entries),
        },
    )
    config = SimpleNamespace(
        output=tmp_path, budget=budget, paired_seed_manifest={"paired_seed_manifest_sha256": digest}
    )
    assert open_m3_ledger(config, source).spent_usd == 5.0
    lines = ledger_path.read_text().splitlines()
    ledger_path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="history|rewound"):
        open_m3_ledger(config, source)
    ledger_path.unlink()
    with pytest.raises(ValueError, match="missing"):
        open_m3_ledger(config, source)
    assert not ledger_path.exists()


@pytest.mark.parametrize("job", ["calibrate", "select-tau", "benchmark"])
def test_frozen_job_failures_settle_last_work_without_any_model_execution(job, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from riichienv.kar import budget as budget_module
    from riichienv.kar.authorization import update_implementation_sha256
    from riichienv.kar.calibration_quality import CalibrationQualityPolicy
    from riichienv.kar.contracts import content_hash
    from riichienv.kar.learning import calibration_job, simulation, validation_policy

    clock = [0.0]
    monkeypatch.setattr(budget_module.time, "monotonic", lambda: clock[0])
    output = tmp_path / "job"
    digest = "c" * 64
    phase = "M2"

    def fail_work(*args, **kwargs):
        clock[0] = 10.0
        raise RuntimeError("simulated computation failure")

    if job == "benchmark":
        phase = "M1"
        paths = [tmp_path / f"release-{index}" for index in range(4)]
        from test_paired_benchmark import release_fixture

        for index, path in enumerate(paths):
            path.mkdir()
            (path / "release.json").write_text(
                json.dumps(release_fixture(str(index), tau=0.5 if index == 0 else "infinity"))
            )
        digest = simulation.benchmark_plan(paths[0], (paths[1], paths[2], paths[3]), matches=8, seed=1)["plan_sha256"]
        monkeypatch.setattr(simulation, "_benchmark_measure", fail_work)
    estimate = CostForecast(
        phase, 8 if job == "benchmark" else 4, 4.0, 3600.0, 0.0, "synthetic failure-clock fixture", digest, 1.0
    )
    budget = BudgetPlan(phase, 100.0 if phase == "M1" else 650.0, 0.0, estimate)
    operation = {
        "calibrate": "calibrate_frozen_logits",
        "select-tau": "select_validation_tau",
        "benchmark": "benchmark_matches",
    }[job]
    authorization = {
        "version": "explicit_frozen_evaluation_authorization_v1",
        "explicit_user_approval": True,
        "approval_reference": "synthetic cleanup-only test",
        "allowed_operations": [operation],
        "plan_sha256": digest,
        "source_sha256": update_implementation_sha256(),
    }
    authorization["authorization_sha256"] = content_hash(authorization)
    with pytest.raises(RuntimeError, match="simulated computation failure"):
        if job == "calibrate":
            predictor = SimpleNamespace(
                _unchanged=lambda: None, binding=SimpleNamespace(train_manifest_sha256="d" * 64)
            )
            fake = SimpleNamespace(
                manifest={"manifest_sha256": "d" * 64, "games": [{"game_id": "v", "split": "validation"}]}
            )
            monkeypatch.setattr(calibration_job, "DecisionDataset", lambda *args: fake)
            monkeypatch.setattr(
                calibration_job, "calibration_job_plan", lambda *args, **kwargs: {"plan_sha256": digest}
            )
            monkeypatch.setattr(
                calibration_job,
                "calibration_split_from_manifest",
                lambda *args: SimpleNamespace(validation_games=("v",)),
            )
            monkeypatch.setattr(calibration_job, "_archive_and_fit", fail_work)
            quality = CalibrationQualityPolicy(
                {head: 5.0 for head in ("action", "response", "rank")},
                {head: 2.0 for head in ("action", "response", "rank")},
                {head: 1.0 for head in ("action", "response", "rank")},
                1,
                1,
                0.0,
                "synthetic criteria",
            )
            calibration_job.collect_and_calibrate(
                predictor,
                tmp_path / "data",
                {},
                output,
                budget=budget,
                plan_sha256=digest,
                authorization=authorization,
                quality_policy=quality,
            )
        elif job == "select-tau":
            subject = SimpleNamespace(
                calibration=SimpleNamespace(split=SimpleNamespace(fit_games=("v",))),
                support=object(),
                options=SimpleNamespace(max_prior_amplification=None),
            )
            monkeypatch.setattr(
                validation_policy, "validation_tau_plan", lambda *args, **kwargs: {"plan_sha256": digest, "matches": 4}
            )
            monkeypatch.setattr(validation_policy, "_tau_outcomes", fail_work)
            validation_policy.select_validation_tau(
                subject, (object(),) * 3, output, budget=budget, seed=1, plan_sha256=digest, authorization=authorization
            )
        else:
            simulation.benchmark_matches(
                paths[0],
                (paths[1], paths[2], paths[3]),
                matches=8,
                seed=1,
                output=output,
                budget=budget,
                authorization=authorization,
            )
    ledger = BudgetLedger(output / "costs.jsonl", budget)
    assert ledger.spent_usd == 10.0 and ledger.entries[-1]["reference"].endswith("/aborted")
