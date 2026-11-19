import json
from dataclasses import replace

import pytest

from riichienv.kar import budget as module
from riichienv.kar.budget import BudgetExceededError, BudgetLedger, BudgetPlan, CostForecast, FixedCostReceipt
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning.run_publication import completed_run_record, publish_run_manifest
from riichienv.kar.learning.training_config import (
    ProductionTrainConfig,
    main_model_registration,
    validate_main_model_registration,
)
from riichienv.kar.versioning import MODEL_TIER_BASIS, artifact_contract


def plan(phase="M1", fixed=0.0, rate=3600.0):
    estimate = CostForecast(
        phase, 1, 1.0, rate, fixed, "synthetic billing fixture,not measured resource evidence", "a" * 64, 1.0
    )
    return BudgetPlan(phase, 100.0 if phase == "M1" else 650.0 if phase == "M2" else 250.0, 0.0, estimate)


def test_fixed_estimates_remain_reserved_until_actual_invoice_reconciliation(tmp_path):
    budget = plan(fixed=70.0)
    ledger = BudgetLedger(tmp_path / "costs.jsonl", budget)
    ledger.charge(24.0, category="compute", reference="synthetic successful work")
    assert ledger.spent_usd == 24.0 and ledger.reserved_fixed_usd == 70.0 and ledger.committed_usd == 94.0
    with pytest.raises(BudgetExceededError):
        ledger.reserve_next(7.0)
    invoice = ledger.charge(70.0, category="storage", reference="actual test invoice", receipt_id="invoice-1")
    assert ledger.spent_usd == 94.0 and ledger.reserved_fixed_usd == 0
    assert ledger.charge(70.0, category="storage", reference="actual test invoice", receipt_id="invoice-1") == invoice
    assert BudgetLedger(tmp_path / "costs.jsonl", budget).spent_usd == 94.0


def test_actual_overrun_is_durably_recorded_not_hidden(tmp_path):
    ledger = BudgetLedger(tmp_path / "costs.jsonl", plan(fixed=70.0))
    with pytest.raises(BudgetExceededError):
        ledger.charge(40.0, category="compute", reference="slower than forecast")
    assert ledger.spent_usd == 40.0 and ledger.committed_usd == 110.0
    assert json.loads((tmp_path / "costs.jsonl").read_text().splitlines()[-1])["usd"] == 40.0


def test_fixed_receipts_roundtrip_without_becoming_estimated_actual_costs(tmp_path):
    budget = replace(
        plan(fixed=70.0), fixed_cost_receipts=(FixedCostReceipt("invoice-1", 50.0, "transfer", "actual test invoice"),)
    )
    assert BudgetPlan.from_record(budget.to_record()) == budget
    ledger = BudgetLedger(tmp_path / "costs.jsonl", budget)
    assert ledger.spent_usd == 50.0 and ledger.reserved_fixed_usd == 20.0
    assert ledger.state_dict()["billing_status"] == "fixed_receipts_pending"


def test_exception_scope_settles_failed_work_and_keeps_original_exception(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    ledger = BudgetLedger(tmp_path / "costs.jsonl", plan())
    with pytest.raises(RuntimeError, match="failed body"):
        with ledger.record_compute("calibration"):
            clock[0] = 10.0
            raise RuntimeError("failed body")
    assert ledger.spent_usd == 10.0
    assert ledger.entries[-1]["reference"] == "calibration/aborted"


def test_resume_cost_anchor_rejects_a_valid_truncated_prefix(tmp_path):
    budget = plan()
    path = tmp_path / "costs.jsonl"
    ledger = BudgetLedger(path, budget)
    ledger.charge(2.0, category="compute", reference="completed arm")
    state = ledger.state_dict()
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    resumed = BudgetLedger(path, budget)
    with pytest.raises(ValueError):
        resumed.verify_state(state)
    path.unlink()
    with pytest.raises(ValueError):
        BudgetLedger(path, budget, allow_create=False)
    assert not path.exists()


def test_main_model_registration_binds_output_and_ledger(tmp_path, monkeypatch):
    estimate = CostForecast("M2", 1, 1.0, 0.0, 0.0, "synthetic registry fixture", "b" * 64)
    config = ProductionTrainConfig(
        "M2", "small", 900, 990, 2, BudgetPlan("M2", 650.0, 0.0, estimate), device="cpu", precision="fp32"
    )
    root = tmp_path / "A"
    from test_stage_prerequisites import m1_fixture  # noqa: PLC0415

    from riichienv.kar.learning.prerequisites import m2_prerequisites  # noqa: PLC0415

    m1_path, _ = m1_fixture(tmp_path, monkeypatch)
    prerequisite = m2_prerequisites(m1_path, config=config, dataset_sha256="c" * 64)
    registry = main_model_registration("c" * 64, config, root, prerequisites=prerequisite)
    expected = root / "M2-costs.jsonl"
    assert (
        validate_main_model_registration(
            registry, dataset_sha256="c" * 64, config=config, output_dir=root / "training", ledger_path=None
        )
        == expected.resolve()
    )
    for output, ledger in ((tmp_path / "B", None), (root / "training", tmp_path / "fresh.jsonl")):
        with pytest.raises(ValueError):
            validate_main_model_registration(
                registry, dataset_sha256="c" * 64, config=config, output_dir=output, ledger_path=ledger
            )


def test_finished_checkpoint_publication_can_be_rebuilt_without_updates(tmp_path):
    contract = {
        **artifact_contract(),
        "synthetic": True,
        "training": {
            "model_tier": "small",
            "N_train_half_games": 900,
            "N_eligible_half_games": 990,
            "model_tier_basis": MODEL_TIER_BASIS,
        },
    }
    runtime = {"world_size": 1}
    state = {
        "finished": True,
        "completed_epochs": 2,
        "step": 7,
        "processed_examples": 900,
        "initial_weights_sha256": "a" * 64,
        "final_weights_sha256": "b" * 64,
        "epoch_order_sha256": {"0": "c" * 64},
        "third_epoch_decision": None,
        "validation_history": [{"report": {"partition": "validation", "full_validation_split_evaluated": True}}],
    }
    pointer = {
        "metadata": {"step": 7, "release_contract": contract},
        "checkpoint_sha256": "d" * 64,
        "checkpoint_file": "owned-generation.pt",
    }
    record = completed_run_record(
        state,
        phase="M2",
        planned_epochs=2,
        runtime=runtime,
        contract=contract,
        budget={"spent_usd": 1.0},
        dataset_sha256="e" * 64,
        final_weights_sha256="b" * 64,
        checkpoint=pointer,
        recovered=True,
    )
    target = tmp_path / "run_manifest.json"
    target.write_bytes(b"{partial publication")
    signed = publish_run_manifest(target, record)
    assert json.loads(target.read_text()) == signed and signed["recovered_completed_checkpoint"]
    assert signed["step"] == 7 and signed["final_weights_sha256"] == "b" * 64
    assert signed["N_train_half_games"] == 900 and signed["N_eligible_half_games"] == 990
    assert signed["model_tier_basis"] == MODEL_TIER_BASIS
    assert next(tmp_path.glob("run_manifest.previous-*.json")).read_bytes() == b"{partial publication"
    assert signed["run_manifest_sha256"] == content_hash(record)
