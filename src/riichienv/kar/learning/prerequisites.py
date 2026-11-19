from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..budget import BudgetLedger
from ..calibration_roles import CALIBRATION_SPLIT_SALT
from ..contracts import content_hash
from ..policy import fixed_tau_grid
from ..release import implementation_release_identity
from ..runtime_contracts import RuntimeBinding
from ..selection_quality import (
    TAU_MATCH_PLAN_VERSION,
    TAU_SELECTION_VERSION,
    TauSelectionPolicy,
    cal_fit_scenario_manifest,
    validate_finite_tau_selection,
)
from ..versioning import validate_artifact_contract
from .config import ModelConfig, frozen_inference_recipe_sha256

if TYPE_CHECKING:
    from .training_config import ProductionTrainConfig

PREREQUISITE_VERSION = "completed_m1_to_m2_gate_v2"


def _read_signed(path: Path, field: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise ValueError("Missing/oversized completed stage artifact")
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != content_hash(unsigned):
        raise ValueError("Completed stage artifact integrity mismatch")
    return value


def m2_prerequisites(  # noqa: PLR0915
    m1_report: Path,
    *,
    config: ProductionTrainConfig,
    dataset_sha256: str,
    tau_selection_job: Path | None = None,
) -> dict[str, Any]:
    report = _read_signed(m1_report, "report_sha256")
    required = ("complete_feature_ablation", "same_initialization", "same_sample_order", "same_updates")
    if (
        report.get("status") != "PASS"
        or report.get("phase") != "M1"
        or any(report.get(key) is not True for key in required)
    ):
        raise ValueError("M2 requires the completed same-initialization/full-epoch M1 feature comparison")
    if set(report.get("results", {})) != {"feature-on", "feature-off"}:
        raise ValueError("Both M1 feature arms must be complete")
    implementation = implementation_release_identity()
    snapshots = {}
    for name, enabled in (("feature-on", True), ("feature-off", False)):
        saved = report["results"][name]
        root = m1_report.parent / name
        run = _read_signed(root / "run_manifest.json", "run_manifest_sha256")
        if run != saved or run.get("status") != "PASS" or run.get("full_epoch_training") is not True:
            raise ValueError("M1 summary differs from its actual completed training arm")
        validate_artifact_contract(run["release_contract"], context="M1 prerequisite")
        training = run["release_contract"]["training"]
        if (
            run.get("completed_epochs") != 1
            or run.get("planned_epochs") != 1
            or training["phase"] != "M1"
            or training["model_tier"] != "small"
            or training["use_rule_features"] is not enabled
            or training["N_train_half_games"] != config.N_train_half_games
            or training["N_eligible_half_games"] != config.N_eligible_half_games
            or run.get("dataset_manifest_sha256") != dataset_sha256
        ):
            raise ValueError("M1 prerequisites used another dataset, feature mode, tier or epoch scope")
        validation = run["final_validation"]["report"]
        if validation.get("partition") != "validation" or validation.get("full_validation_split_evaluated") is not True:
            raise ValueError("M1 prerequisites lack complete validation")
        built = run["release_contract"]["implementation"]
        if any(built[key] != implementation[key] for key in ("source_sha256", "native_extension_sha256")):
            raise ValueError("Recheck M1 after implementation/native changes; do not inherit old acceptance")
        pointer = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
        generation = (root / pointer["checkpoint_file"]).resolve()
        if generation.parent != root.resolve() or not generation.name.startswith("checkpoint.g"):
            raise ValueError("M1 checkpoint escapes its recorded arm")
        if (
            pointer["checkpoint_sha256"] != run["checkpoint_sha256"]
            or hashlib.sha256(generation.read_bytes()).hexdigest() != run["checkpoint_sha256"]
        ):
            raise ValueError("M1 checkpoint weights changed")
        from .training_config import ProductionTrainConfig  # noqa: PLC0415

        arm_config = ProductionTrainConfig.from_record(training)
        ledger = BudgetLedger(m1_report.parent / "M1-costs.jsonl", arm_config.budget, allow_create=False)
        ledger.verify_state(run["budget_ledger"])
        snapshots[name] = run["run_manifest_sha256"]
    choice = None
    selection_sha = None
    if tau_selection_job is not None:
        job = _read_signed(tau_selection_job, "selection_job_sha256")
        if (
            job.get("status") != "PASS"
            or job.get("test_games_used") is not False
            or job.get("cal_check_used_for_tuning") is not False
        ):
            raise ValueError("Growth evidence must be the frozen validation-only tau job")
        choice = job["tau_selection"]
        unsigned = {key: value for key, value in choice.items() if key != "tau_selection_sha256"}
        if (
            choice.get("version") != TAU_SELECTION_VERSION
            or choice.get("tau_selection_sha256") != content_hash(unsigned)
            or choice.get("diagnostic_fixture") is not False
        ):
            raise ValueError("Growth evidence is not production paired validation selection")
        arm = report["results"]["feature-on" if config.use_rule_features else "feature-off"]
        recipe = frozen_inference_recipe_sha256(
            ModelConfig.from_preset("small"),
            use_rule_features=config.use_rule_features,
        )
        binding = RuntimeBinding.current(arm["final_weights_sha256"], dataset_sha256, inference_recipe_sha256=recipe)
        if choice.get("binding_sha256") != binding.digest:
            raise ValueError("Growth evidence used another M1 checkpoint or feature mode")
        model_selection = arm["final_validation"]["report"].get("model_selection", {})
        if (
            model_selection.get("partition") != "cal_fit"
            or model_selection.get("full_cal_fit_evaluated") is not True
            or model_selection.get("cal_check_used") is not False
            or model_selection.get("calibration_split_salt") != CALIBRATION_SPLIT_SALT
            or not model_selection.get("game_ids")
        ):
            raise ValueError("Growth evidence requires the actual M1 complete cal_fit scenario list")
        expected_games = model_selection["game_ids"]
        manifest = cal_fit_scenario_manifest(
            expected_games,
            validation_manifest_sha256=choice.get("validation_manifest_sha256"),
            calibration_split_salt=CALIBRATION_SPLIT_SALT,
        )
        policy = TauSelectionPolicy.from_record(choice["selection_policy"])
        plan = job.get("plan", {})
        plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
        grid = fixed_tau_grid([float(value) for value in plan.get("grid", [])])
        if (
            plan.get("version") != TAU_MATCH_PLAN_VERSION
            or plan.get("plan_sha256") != content_hash(plan_body)
            or plan.get("partition") != "cal_fit"
            or plan.get("selection_policy") != policy.to_record()
            or choice["selection_policy"] != policy.to_record()
            or plan.get("subject_binding_sha256") != binding.digest
            or plan.get("policy_identity") != choice.get("policy_identity")
            or plan.get("validation_manifest_sha256") != choice.get("validation_manifest_sha256")
            or plan.get("calibration_split_salt") != CALIBRATION_SPLIT_SALT
            or choice.get("calibration_split_salt") != CALIBRATION_SPLIT_SALT
            or plan.get("scenario_manifest") != manifest
            or choice.get("scenario_manifest") != manifest
            or plan.get("scenario_game_ids") != manifest["game_ids"]
            or choice.get("game_ids") != manifest["game_ids"]
            or choice.get("grid") != ["infinity" if math.isinf(value) else str(value) for value in grid]
            or plan.get("grid") != ["infinity" if math.isinf(value) else value for value in grid]
            or plan.get("matches") != len(expected_games) * len(grid)
        ):
            raise ValueError("Growth evidence must retain the registered full cal_fit scenarios, tau grid and criteria")
        selected = float(choice["selected_tau"])
        if math.isnan(selected) or selected <= 0:
            raise ValueError("Growth evidence contains an invalid selected tau")
        if math.isfinite(selected):
            validate_finite_tau_selection(choice, expected_game_ids=expected_games)
        selection_sha = job["selection_job_sha256"]
    growth = (
        choice is not None and choice.get("reproducible_gain") is True and math.isfinite(float(choice["selected_tau"]))
    )
    if config.model_tier != "small" and not growth:
        raise ValueError("Do not enlarge the model without reproducible validation policy gain")
    record = {
        "version": PREREQUISITE_VERSION,
        "dataset_sha256": dataset_sha256,
        "source_sha256": implementation["source_sha256"],
        "native_sha256": implementation["native_extension_sha256"],
        "model_tier": config.model_tier,
        "use_rule_features": config.use_rule_features,
        "m1_report": str(m1_report.resolve()),
        "m1_report_sha256": report["report_sha256"],
        "m1_runs": snapshots,
        "tau_selection_job": str(tau_selection_job.resolve()) if tau_selection_job else None,
        "tau_selection_job_sha256": selection_sha,
        "growth_validated": growth,
    }
    return {**record, "prerequisites_sha256": content_hash(record)}


def verify_prerequisites(record: dict[str, Any], *, config: ProductionTrainConfig, dataset_sha256: str) -> None:
    unsigned = {key: value for key, value in record.items() if key != "prerequisites_sha256"}
    if record.get("version") != PREREQUISITE_VERSION or record.get("prerequisites_sha256") != content_hash(unsigned):
        raise ValueError("Missing or modified completed M1 stage prerequisites")
    rebuilt = m2_prerequisites(
        Path(record["m1_report"]),
        config=config,
        dataset_sha256=dataset_sha256,
        tau_selection_job=Path(record["tau_selection_job"]) if record["tau_selection_job"] else None,
    )
    if rebuilt != record:
        raise ValueError("Completed M1/validation evidence changed after main-model registration")
