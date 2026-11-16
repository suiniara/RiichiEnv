import hashlib
import json

import pytest

from riichienv.kar.budget import BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning.config import FEATURE_VERSION, ModelConfig
from riichienv.kar.learning.prerequisites import m2_prerequisites, verify_prerequisites
from riichienv.kar.learning.training_config import (
    ProductionTrainConfig,
    main_model_registration,
    validate_main_model_registration,
)
from riichienv.kar.runtime_contracts import RuntimeBinding
from riichienv.kar.selection_quality import TAU_SELECTION_VERSION, TauSelectionPolicy, paired_tau_evidence


def sign(body, key):
    return {**body, key: content_hash(body)}


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def m1_fixture(tmp_path, monkeypatch, *, train_games=900):
    # Metadata-only temporary fixture. These bytes are deliberately not model
    # weights; tests verify provenance contracts without importing/loading Torch.
    from riichienv.kar.learning import prerequisites  # noqa: PLC0415

    implementation = {"source_sha256": "a" * 64, "native_extension_sha256": "b" * 64}
    monkeypatch.setattr(prerequisites, "implementation_release_identity", lambda: implementation)
    root = tmp_path / "m1"
    root.mkdir()
    results = {}
    forecast = CostForecast("M1", 1, 1.0, 0.0, 0.0, "synthetic metadata-only prerequisite fixture", "f" * 64)
    budget = BudgetPlan("M1", 100.0, 0.0, forecast)
    ledger = BudgetLedger(root / "M1-costs.jsonl", budget)
    for name, enabled in (("feature-on", True), ("feature-off", False)):
        directory = root / name
        directory.mkdir()
        config = ProductionTrainConfig(
            "M1",
            "small",
            train_games,
            train_games + 90,
            1,
            budget,
            use_rule_features=enabled,
            device="cpu",
            precision="fp32",
        )
        payload = ("synthetic,not a Torch checkpoint/" + name).encode()
        checkpoint = directory / "checkpoint.g-fixture.pt"
        checkpoint.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        write(directory / "checkpoint.json", {"checkpoint_file": checkpoint.name, "checkpoint_sha256": sha})
        body = {
            "status": "PASS",
            "full_epoch_training": True,
            "completed_epochs": 1,
            "planned_epochs": 1,
            "dataset_manifest_sha256": "c" * 64,
            "release_contract": {"training": config.to_record(), "implementation": implementation},
            "final_validation": {"report": {"partition": "validation", "full_validation_split_evaluated": True}},
            "checkpoint_sha256": sha,
            "budget_ledger": ledger.state_dict(),
            "final_weights_sha256": hashlib.sha256(name.encode()).hexdigest(),
        }
        run = sign(body, "run_manifest_sha256")
        write(directory / "run_manifest.json", run)
        results[name] = run
    body = {
        "status": "PASS",
        "phase": "M1",
        "complete_feature_ablation": True,
        "same_initialization": True,
        "same_sample_order": True,
        "same_updates": True,
        "results": results,
    }
    path = root / "M1-ablation.json"
    write(path, sign(body, "report_sha256"))
    return path, results


def m2_config(tier="small", games=900):
    estimate = CostForecast("M2", 1, 1.0, 0.0, 0.0, "synthetic prerequisite planning fixture", "e" * 64)
    return ProductionTrainConfig(
        "M2", tier, games, games + 90, 2, BudgetPlan("M2", 650.0, 0.0, estimate), device="cpu", precision="fp32"
    )


def growth_fixture(path, results, config):
    policy = TauSelectionPolicy(bootstrap_replicates=20)
    recipe = content_hash(
        {
            "model_config_hash": ModelConfig.from_preset("small").contract_hash,
            "feature_version": FEATURE_VERSION,
            "use_rule_features": config.use_rule_features,
            "inference_feature_dropout": 0,
        }
    )
    arm = results["feature-on" if config.use_rule_features else "feature-off"]
    binding = RuntimeBinding.current(arm["final_weights_sha256"], "c" * 64, inference_recipe_sha256=recipe)
    outcomes = {
        float("inf"): {"synthetic-val-A": 0.0, "synthetic-val-B": 0.0},
        0.5: {"synthetic-val-A": 0.5, "synthetic-val-B": 0.5},
    }
    choice = {
        "version": TAU_SELECTION_VERSION,
        "diagnostic_fixture": False,
        "binding_sha256": binding.digest,
        "selection_partition": "cal_fit",
        "selected_tau": ".5",
        "reproducible_gain": True,
        "game_ids": ["synthetic-val-A", "synthetic-val-B"],
        "mean_utility": {"infinity": 0.0, "0.5": 0.5},
        "selection_policy": policy.to_record(),
        "paired_validation_evidence": paired_tau_evidence(outcomes, (float("inf"), 0.5), policy),
    }
    choice = sign(choice, "tau_selection_sha256")
    job = sign(
        {"status": "PASS", "test_games_used": False, "cal_check_used_for_tuning": False, "tau_selection": choice},
        "selection_job_sha256",
    )
    target = path.parent / "synthetic-selection.json"
    write(target, job)
    return target


def test_completed_m1_is_revalidated_when_registration_is_used(tmp_path, monkeypatch):
    path, _ = m1_fixture(tmp_path, monkeypatch)
    config = m2_config()
    evidence = m2_prerequisites(path, config=config, dataset_sha256="c" * 64)
    verify_prerequisites(evidence, config=config, dataset_sha256="c" * 64)
    checkpoint = path.parent / "feature-on/checkpoint.g-fixture.pt"
    checkpoint.write_bytes(b"modified metadata-only test bytes")
    with pytest.raises(ValueError, match="weights changed"):
        verify_prerequisites(evidence, config=config, dataset_sha256="c" * 64)


def test_growth_needs_reproducible_gain_from_the_actual_m1_checkpoint(tmp_path, monkeypatch):
    path, results = m1_fixture(tmp_path, monkeypatch, train_games=30000)
    config = m2_config("medium", 30000)
    with pytest.raises(ValueError, match="enlarge"):
        m2_prerequisites(path, config=config, dataset_sha256="c" * 64)
    selection = growth_fixture(path, results, config)
    evidence = m2_prerequisites(path, config=config, dataset_sha256="c" * 64, tau_selection_job=selection)
    assert evidence["growth_validated"] is True
    foreign = results.copy()
    foreign["feature-on"] = {**foreign["feature-on"], "final_weights_sha256": "d" * 64}
    selection = growth_fixture(path, foreign, config)
    with pytest.raises(ValueError, match="checkpoint"):
        m2_prerequisites(path, config=config, dataset_sha256="c" * 64, tau_selection_job=selection)


def test_missing_prerequisites_cannot_make_a_draft_registration_executable(tmp_path):
    config = m2_config()
    root = tmp_path / "run"
    draft = main_model_registration("c" * 64, config, root)
    assert not draft["execution_ready"]
    with pytest.raises(ValueError, match="draft"):
        validate_main_model_registration(
            draft, dataset_sha256="c" * 64, config=config, output_dir=root / "training", ledger_path=None
        )
