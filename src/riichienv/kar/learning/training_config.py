from __future__ import annotations

import math
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..budget import BudgetPlan
from ..calibration_roles import CALIBRATION_SPLIT_SALT, calibration_is_fit
from ..contracts import content_hash, model_tier
from ..versioning import (
    INPUT_PIPELINE_VERSION,
    MODEL_TIER_BASIS,
    RULE_FEATURE_MASK_VERSION,
    TRAIN_CONFIG_VERSION,
    TRAINING_PLAN_VERSION,
)

EFFECTIVE_BATCH = {"small": 128, "medium": 256, "large": 512}


def _config_versions() -> dict[str, str]:
    return {
        "training_config_version": TRAIN_CONFIG_VERSION,
        "input_pipeline_version": INPUT_PIPELINE_VERSION,
        "training_plan_version": TRAINING_PLAN_VERSION,
        "rule_feature_mask_version": RULE_FEATURE_MASK_VERSION,
        "model_tier_basis": MODEL_TIER_BASIS,
    }


@dataclass(frozen=True)
class ProductionTrainConfig:
    phase: str
    model_tier: str
    N_train_half_games: int
    N_eligible_half_games: int
    planned_epochs: int
    budget: BudgetPlan
    seed: int = 17
    device: str = "cuda"
    precision: str = "bf16"
    microbatch_size: int = 1
    cpu_threads: int = 2
    effective_batch_size: int = 0
    learning_rate: float = 3e-4
    use_rule_features: bool = True
    feature_mask_probability: float = 0.25
    bucket_window: int = 128
    checkpoint_updates: int = 100
    third_epoch_nll_tolerance: dict[str, float] = field(
        default_factory=lambda: {head: 0.0 for head in ("action", "response", "rank")}
    )
    distributed_backend: str | None = None
    gloo_bind_host: str | None = None
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if self.phase not in {"M1", "M2"} or self.budget.phase != self.phase or self.model_tier not in EFFECTIVE_BATCH:
            raise ValueError("Production training requires a registered phase,tier and independent budget")
        if (
            type(self.N_train_half_games) is not int
            or type(self.N_eligible_half_games) is not int
            or not 0 < self.N_train_half_games <= self.N_eligible_half_games
        ):
            raise ValueError("Model selection uses complete train games, never raw/all-split file counts")
        if self.phase == "M2" and tuple(EFFECTIVE_BATCH).index(self.model_tier) > tuple(EFFECTIVE_BATCH).index(
            model_tier(self.N_train_half_games)
        ):
            raise ValueError("Model tier exceeds the confirmed train-split ladder")
        if self.phase == "M1" and (self.model_tier != "small" or self.planned_epochs != 1):
            raise ValueError("M1 is Small only, at most one complete epoch per feature mode")
        if self.phase == "M2" and self.planned_epochs not in {2, 3}:
            raise ValueError("M2 registers two epochs, with at most one conditional third epoch")
        expected = EFFECTIVE_BATCH[self.model_tier]
        if self.effective_batch_size not in {0, expected}:
            raise ValueError("Effective batch size differs from the frozen tier recipe")
        object.__setattr__(self, "effective_batch_size", expected)
        if self.device not in {"cpu", "cuda"} or self.precision not in {"fp32", "bf16"}:
            raise ValueError("Unsupported device/precision recipe")
        if (
            self.microbatch_size < 1
            or self.microbatch_size > expected
            or self.cpu_threads < 1
            or self.checkpoint_updates < 1
        ):
            raise ValueError("Invalid batching/thread/checkpoint configuration")
        if self.learning_rate != 3e-4 or self.feature_mask_probability != 0.25:
            raise ValueError("The first production recipe does not silently tune LR or augmentation")
        if set(self.third_epoch_nll_tolerance) != {"action", "response", "rank"} or any(
            not math.isfinite(value) or value < 0 for value in self.third_epoch_nll_tolerance.values()
        ):
            raise ValueError("Third-epoch head deterioration limits must be preregistered")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ProductionTrainConfig:
        allowed = (
            set(cls.__dataclass_fields__) | set(_config_versions()) | {"recipe_sha256", "rule_feature_mask_probability"}
        )
        if set(record) - allowed:
            raise ValueError("Unknown training configuration field")
        if any(record.get(key) != value for key, value in _config_versions().items()):
            raise ValueError("Training plan/configuration/count-basis version differs")
        values = {key: value for key, value in record.items() if key in cls.__dataclass_fields__}
        values["budget"] = BudgetPlan.from_record(values["budget"])
        config = cls(**values)
        if record.get("rule_feature_mask_probability") != config.feature_mask_probability:
            raise ValueError("Rule feature mask probability alias differs")
        if "recipe_sha256" in record and record["recipe_sha256"] != config.recipe_sha256:
            raise ValueError("Training recipe digest differs")
        return config

    @property
    def recipe_sha256(self) -> str:
        record = asdict(self)
        record.pop("budget")
        return content_hash({**_config_versions(), **record})

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "budget": self.budget.to_record(),
            **_config_versions(),
            "rule_feature_mask_probability": self.feature_mask_probability,
            "recipe_sha256": self.recipe_sha256,
        }


def training_recipe_sha256(arguments: dict[str, Any]) -> str:
    """Compute the measurement/forecast binding before supplying prices or a budget."""
    record = {}
    for item in fields(ProductionTrainConfig):
        if item.name == "budget":
            continue
        if item.name in arguments:
            record[item.name] = arguments[item.name]
        elif item.default is not MISSING:
            record[item.name] = item.default
        elif item.default_factory is not MISSING:
            record[item.name] = item.default_factory()
        else:
            raise ValueError("Missing training recipe field: " + item.name)
    record["effective_batch_size"] = EFFECTIVE_BATCH[record["model_tier"]]
    return content_hash({**_config_versions(), **record})


def main_model_registration(
    dataset_sha256: str,
    config: ProductionTrainConfig,
    output_root: Path,
    *,
    prerequisites: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "version": "unique_main_model_registration_v3",
        "dataset_manifest_sha256": dataset_sha256,
        "selected_tier": config.model_tier,
        "recipe_sha256": config.recipe_sha256,
        "seed": config.seed,
        "output_root": str(output_root.resolve()),
        "training_output": str((output_root / "training").resolve()),
        "budget_ledger": str((output_root / "M2-costs.jsonl").resolve()),
        "prerequisites": prerequisites,
        "execution_ready": prerequisites is not None,
    }
    return {**record, "registration_sha256": content_hash(record)}


def registered_main_model_ledger(
    record: dict[str, Any],
    *,
    dataset_sha256: str,
    config: ProductionTrainConfig,
    output_dir: Path,
    ledger_path: Path | None,
) -> Path:
    """Resolve the signed canonical ledger before metered prerequisite verification."""
    unsigned = {key: value for key, value in record.items() if key != "registration_sha256"}
    if record.get("version") != "unique_main_model_registration_v3" or record.get(
        "registration_sha256"
    ) != content_hash(unsigned):
        raise ValueError("Main-model registration identity/version mismatch")
    if (
        record.get("dataset_manifest_sha256") != dataset_sha256
        or record.get("recipe_sha256") != config.recipe_sha256
        or record.get("selected_tier") != config.model_tier
    ):
        raise ValueError("Registration does not cover this model/data/recipe")
    root = Path(record["output_root"]).resolve()
    registered_output, registered_ledger = (root / "training").resolve(), (root / "M2-costs.jsonl").resolve()
    if (
        Path(record["training_output"]).resolve() != registered_output
        or Path(record["budget_ledger"]).resolve() != registered_ledger
    ):
        raise ValueError("Registered run and ledger are not the canonical unique M2 paths")
    if output_dir.resolve() != registered_output or (
        ledger_path is not None and ledger_path.resolve() != registered_ledger
    ):
        raise ValueError("Cannot reuse M2 registration in another output or fresh budget ledger")
    if record.get("execution_ready") is not True or not isinstance(record.get("prerequisites"), dict):
        raise ValueError("A draft main-model registration cannot bypass completed M1 prerequisites")
    return registered_ledger


def validate_main_model_registration(
    record: dict[str, Any],
    *,
    dataset_sha256: str,
    config: ProductionTrainConfig,
    output_dir: Path,
    ledger_path: Path | None,
) -> Path:
    registered_ledger = registered_main_model_ledger(
        record, dataset_sha256=dataset_sha256, config=config, output_dir=output_dir, ledger_path=ledger_path
    )
    from .prerequisites import verify_prerequisites  # noqa: PLC0415

    verify_prerequisites(record["prerequisites"], config=config, dataset_sha256=dataset_sha256)
    return registered_ledger


def third_epoch_allowed(
    previous: dict[str, Any], current: dict[str, Any], tolerance: dict[str, float], *, budget_has_room: bool
) -> tuple[bool, str]:
    if not budget_has_room:
        return False, "insufficient_registered_budget"
    if previous.get("partition") != "validation" or current.get("partition") != "validation":
        raise ValueError("Test/cal_check results must not select training epochs")
    if not previous.get("full_validation_split_evaluated") or not current.get("full_validation_split_evaluated"):
        return False, "whole_validation_evidence_missing"
    left, right = previous.get("model_selection"), current.get("model_selection")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False, "cal_fit_selection_evidence_missing"
    for report in (left, right):
        if (
            report.get("partition") != "cal_fit"
            or report.get("cal_check_used") is not False
            or report.get("full_cal_fit_evaluated") is not True
            or report.get("calibration_split_salt") != CALIBRATION_SPLIT_SALT
        ):
            raise ValueError("Third-epoch selection must exclude independent cal_check")
        if not report.get("game_ids") or any(not calibration_is_fit(game) for game in report["game_ids"]):
            raise ValueError("Model-selection evidence contains foreign calibration-role games")
    if left["game_ids"] != right["game_ids"]:
        raise ValueError("Training epochs must compare the same frozen cal_fit game clusters")
    old, new = left.get("validation_objective"), right.get("validation_objective")
    if old is None or new is None or not math.isfinite(old) or not math.isfinite(new) or new >= old:
        return False, "validation_objective_not_improving"
    for head in ("action", "response", "rank"):
        before, after = left.get(f"{head}_nll"), right.get(f"{head}_nll")
        if (
            before is None
            or after is None
            or not math.isfinite(before)
            or not math.isfinite(after)
            or after - before > tolerance[head]
        ):
            return False, f"{head}_head_deterioration_or_missing_evidence"
    return True, "validation_and_budget_qualified"
