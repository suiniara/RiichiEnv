"""Startup accounting with fake clocks/loaders; no model updates or matches."""

# ruff: noqa: E402 -- Torch is optional outside the explicit ML test job.
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from riichienv.kar import budget as budget_module
from riichienv.kar.authorization import update_implementation_sha256
from riichienv.kar.budget import BudgetExceededError, BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.calibration_quality import CalibrationQualityPolicy
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning import calibration_job, jobs_cli, validation_policy


def _quality():
    heads = ("action", "response", "rank")
    return CalibrationQualityPolicy(
        dict.fromkeys(heads, 5.0),
        dict.fromkeys(heads, 2.0),
        dict.fromkeys(heads, 1.0),
        1,
        1,
        0.0,
        "synthetic guard-only fixture; no fitted model or calibration evidence",
    )


def _budget(*, rate=3600.0):
    return BudgetPlan("M2", 20.0, 0.0, CostForecast("M2", 4, 1.0, rate, 0.0, "fake-clock fixture", "c" * 64))


def _authorization(operation):
    record = {
        "version": "explicit_frozen_evaluation_authorization_v1",
        "explicit_user_approval": True,
        "approval_reference": "synthetic startup regression; computation is blocked",
        "allowed_operations": [operation],
        "plan_sha256": "c" * 64,
        "source_sha256": update_implementation_sha256(),
    }
    return {**record, "authorization_sha256": content_hash(record)}


def _cli_files(tmp_path, monkeypatch, job, budget):
    validation = tmp_path / "validation.json"
    validation.write_text("{}", encoding="utf-8")
    config = {
        "plan_sha256": "c" * 64,
        "output": str(tmp_path / "output"),
        "checkpoint": "synthetic-checkpoint",
        "dataset": "synthetic-dataset",
        "validation_manifest": str(validation),
        "quality_policy": _quality().to_record(),
        "subject_release": "subject",
        "opponent_releases": ["opponent-1", "opponent-2", "opponent-3"],
        "seed": 17,
        "budget": budget,
    }
    config_path, auth_path = tmp_path / "config.json", tmp_path / "authorization.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    operation = "calibrate_frozen_logits" if job == "calibrate" else "select_validation_tau"
    auth_path.write_text(json.dumps(_authorization(operation)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["jobs", job, "--config", str(config_path), "--authorization", str(auth_path)])
    return tmp_path / "output"


@pytest.mark.parametrize("stage", ["calibration-loader", "second-release-loader", "tau-work"])
def test_cli_loaders_and_work_share_one_durable_attempt(tmp_path, monkeypatch, stage):
    budget = _budget()
    job = "calibrate" if stage == "calibration-loader" else "select-tau"
    output = _cli_files(tmp_path, monkeypatch, job, budget.to_record())
    clock, loaded = [0.0], []
    monkeypatch.setattr(budget_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def load(path, **kwargs):
        assert (output / "startup.json").is_file()
        assert len(BudgetLedger(output / "costs.jsonl", budget).pending_compute) == 1
        loaded.append(str(path))
        clock[0] += 2.0
        if stage == "calibration-loader" or stage == "second-release-loader" and len(loaded) == 2:
            raise RuntimeError("synthetic loader failure")
        return SimpleNamespace(
            calibration=SimpleNamespace(split=SimpleNamespace(fit_games=("synthetic-fit",))),
            support=object(),
            options=SimpleNamespace(max_prior_amplification=None),
        )

    def work(*args, **kwargs):
        assert loaded == ["subject", "opponent-1", "opponent-2", "opponent-3"]
        assert len(kwargs["ledger"].pending_compute) == 1
        clock[0] += 1.0
        raise RuntimeError("synthetic work failure")

    monkeypatch.setattr(jobs_cli, "load_training_model", load)
    monkeypatch.setattr(jobs_cli, "FrozenRelease", load)
    monkeypatch.setattr(
        validation_policy,
        "validation_tau_plan",
        lambda *args, **kwargs: {
            "plan_sha256": "c" * 64,
            "matches": 4,
        },
    )
    monkeypatch.setattr(validation_policy, "_tau_outcomes", work)
    with pytest.raises(RuntimeError, match="synthetic .* failure"):
        jobs_cli.main()
    ledger = BudgetLedger(output / "costs.jsonl", budget)
    assert ledger.spent_usd == {"calibration-loader": 2.0, "second-release-loader": 4.0, "tau-work": 9.0}[stage]
    assert len({entry["attempt_id"] for entry in ledger.entries if "attempt_id" in entry}) == 1
    assert not ledger.pending_compute and ledger.entries[-1]["reference"].endswith("/aborted")


@pytest.mark.parametrize("job", ["calibrate", "select-tau"])
def test_cli_rejects_unaffordable_budget_before_any_loader(tmp_path, monkeypatch, job):
    budget = _budget().to_record()
    budget["budget_cap_usd"] = 0.0
    output = _cli_files(tmp_path, monkeypatch, job, budget)

    def forbidden(*args, **kwargs):
        raise AssertionError("Budget admission must precede model/device loading")

    monkeypatch.setattr(jobs_cli, "load_training_model", forbidden)
    monkeypatch.setattr(jobs_cli, "FrozenRelease", forbidden)
    with pytest.raises(BudgetExceededError):
        jobs_cli.main()
    assert not output.exists()


def test_hard_exit_keeps_pending_attempt_and_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "interrupted"
    budget = _budget(rate=0.0)
    script = """
import json
import os
import sys
from pathlib import Path
from riichienv.kar.authorization import update_implementation_sha256
from riichienv.kar.budget import BudgetPlan
from riichienv.kar.calibration_quality import CalibrationQualityPolicy
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning.calibration_job import collect_and_calibrate
output = Path(sys.argv[1])
auth = {
    "version": "explicit_frozen_evaluation_authorization_v1", "explicit_user_approval": True,
    "approval_reference": "synthetic child process exits without model loading",
    "allowed_operations": ["calibrate_frozen_logits"], "plan_sha256": "c" * 64,
    "source_sha256": update_implementation_sha256(),
}
auth["authorization_sha256"] = content_hash(auth)
collect_and_calibrate(
    lambda: os._exit(17), Path("unused"), {}, output,
    budget=BudgetPlan.from_record(json.loads(sys.argv[2])), plan_sha256="c" * 64,
    authorization=auth, quality_policy=CalibrationQualityPolicy.from_record(json.loads(sys.argv[3])),
)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(output), json.dumps(budget.to_record()), json.dumps(_quality().to_record())],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 17, child.stderr
    ledger = BudgetLedger(output / "costs.jsonl", budget)
    assert len(ledger.pending_compute) == 1
    assert ledger.state_dict()["billing_status"] == "compute_reconciliation_required"
    # Existing partial artifacts are immutable regardless of settlement status.
    (output / "logits.sqlite3").write_bytes(b"synthetic partial archive; preserve for audit")

    def forbidden():
        raise AssertionError("An existing output must be rejected before the loader")

    for reconciled in (False, True):
        if reconciled:
            ledger.reconcile_compute(
                ledger.pending_compute[0]["attempt_id"],
                usd=0.0,
                receipt_id="synthetic-zero-price-process",
                reference="zero registered resource price, no model loaded",
            )
        before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
        with pytest.raises(FileExistsError):
            calibration_job.collect_and_calibrate(
                forbidden,
                tmp_path / "unused",
                {},
                output,
                budget=budget,
                plan_sha256="c" * 64,
                authorization=_authorization("calibrate_frozen_logits"),
                quality_policy=_quality(),
            )
        assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == before
    assert not BudgetLedger(output / "costs.jsonl", budget).pending_compute
