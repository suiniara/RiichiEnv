import hashlib
import json
import math

import pytest

from riichienv.kar.budget import BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.calibration_roles import CALIBRATION_SPLIT_SALT, calibration_is_fit
from riichienv.kar.contracts import content_hash, game_split
from riichienv.kar.learning.config import ModelConfig, frozen_inference_recipe_sha256
from riichienv.kar.learning.precision import FrozenBackendPrecision
from riichienv.kar.learning.prerequisites import m2_prerequisites, verify_prerequisites
from riichienv.kar.learning.training_config import (
    ProductionTrainConfig,
    main_model_registration,
    validate_main_model_registration,
)
from riichienv.kar.runtime_contracts import RuntimeBinding
from riichienv.kar.selection_quality import (
    TAU_MATCH_PLAN_VERSION,
    TAU_SELECTION_VERSION,
    TauSelectionPolicy,
    cal_fit_scenario_manifest,
    paired_tau_evidence,
)
from riichienv.kar.versioning import artifact_contract


def sign(body, key):
    return {**body, key: content_hash(body)}


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fit_games():
    games = []
    index = 0
    while len(games) < 64:
        game = f"synthetic-m1-validation-{index}"
        if game_split(game, "kar-riichienv-v1-games") == "validation" and calibration_is_fit(game):
            games.append(game)
        index += 1
    return sorted(games)


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
            "release_contract": {
                **artifact_contract(),
                "training": config.to_record(),
                "implementation": implementation,
            },
            "final_validation": {
                "report": {
                    "partition": "validation",
                    "full_validation_split_evaluated": True,
                    "model_selection": {
                        "partition": "cal_fit",
                        "full_cal_fit_evaluated": True,
                        "cal_check_used": False,
                        "calibration_split_salt": CALIBRATION_SPLIT_SALT,
                        "game_ids": fit_games(),
                    },
                }
            },
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


def growth_fixture(path, results, config, *, subset=False):
    policy = TauSelectionPolicy(bootstrap_replicates=20)
    backend_precision = FrozenBackendPrecision(
        "cpu", "synthetic-metadata-only", "ieee", ("flash", "math"), False, True, None
    )
    recipe = frozen_inference_recipe_sha256(
        ModelConfig.from_preset("small"),
        use_rule_features=config.use_rule_features,
        backend_precision=backend_precision,
    )
    arm = results["feature-on" if config.use_rule_features else "feature-off"]
    binding = RuntimeBinding.current(arm["final_weights_sha256"], "c" * 64, inference_recipe_sha256=recipe)
    games = arm["final_validation"]["report"]["model_selection"]["game_ids"]
    if subset:
        games = games[:-1]
    outcomes = {math.inf: dict.fromkeys(games, -0.5), 0.5: dict.fromkeys(games, 0.5)}
    scenarios = cal_fit_scenario_manifest(
        games,
        validation_manifest_sha256="d" * 64,
        calibration_split_salt=CALIBRATION_SPLIT_SALT,
    )
    choice = {
        "version": TAU_SELECTION_VERSION,
        "diagnostic_fixture": False,
        "binding_sha256": binding.digest,
        "selection_partition": "cal_fit",
        "selected_tau": "0.5",
        "reproducible_gain": True,
        "validation_manifest_sha256": "d" * 64,
        "calibration_split_salt": CALIBRATION_SPLIT_SALT,
        "game_ids": games,
        "scenario_manifest": scenarios,
        "grid": ["infinity", "0.5"],
        "mean_utility": {"infinity": -0.5, "0.5": 0.5},
        "selection_policy": policy.to_record(),
        "paired_validation_evidence": paired_tau_evidence(outcomes, (float("inf"), 0.5), policy),
    }
    choice = sign(choice, "tau_selection_sha256")
    plan = sign(
        {
            "version": TAU_MATCH_PLAN_VERSION,
            "partition": "cal_fit",
            "selection_policy": policy.to_record(),
            "subject_binding_sha256": binding.digest,
            "backend_precision": backend_precision.to_record(),
            "validation_manifest_sha256": "d" * 64,
            "calibration_split_salt": CALIBRATION_SPLIT_SALT,
            "scenario_game_ids": games,
            "scenario_manifest": scenarios,
            "grid": ["infinity", 0.5],
            "matches": len(games) * 2,
        },
        "plan_sha256",
    )
    job = sign(
        {
            "status": "PASS",
            "test_games_used": False,
            "cal_check_used_for_tuning": False,
            "plan": plan,
            "tau_selection": choice,
        },
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


def test_growth_rejects_a_rehashed_plan_and_selection_on_a_cal_fit_subset(tmp_path, monkeypatch):
    path, results = m1_fixture(tmp_path, monkeypatch, train_games=30000)
    config = m2_config("medium", 30000)
    selection = growth_fixture(path, results, config, subset=True)
    with pytest.raises(ValueError, match="registered full cal_fit"):
        m2_prerequisites(path, config=config, dataset_sha256="c" * 64, tau_selection_job=selection)


def test_growth_uses_recomputed_gain_bound_and_bootstrap_is_descriptive(tmp_path, monkeypatch):
    path, results = m1_fixture(tmp_path, monkeypatch, train_games=30000)
    config = m2_config("medium", 30000)
    selection = growth_fixture(path, results, config)
    job = json.loads(selection.read_text(encoding="utf-8"))
    choice = job.pop("tau_selection")
    choice.pop("tau_selection_sha256")
    choice["paired_validation_evidence"]["0.5"]["paired_utility_interval"]["lower"] = -1.0
    job.pop("selection_job_sha256")
    job["tau_selection"] = sign(choice, "tau_selection_sha256")
    write(selection, sign(job, "selection_job_sha256"))
    assert m2_prerequisites(path, config=config, dataset_sha256="c" * 64, tau_selection_job=selection)[
        "growth_validated"
    ]
    choice["paired_validation_evidence"]["0.5"]["gain_bound"]["lower"] = 0.9
    job["tau_selection"] = sign(choice, "tau_selection_sha256")
    write(selection, sign(job, "selection_job_sha256"))
    with pytest.raises(ValueError, match="gain bound"):
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
