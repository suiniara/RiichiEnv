from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ..budget import BudgetPlan
from ..calibration import CalibrationArtifact
from ..calibration_quality import CalibrationQualityPolicy
from ..evaluation_authorization import read_evaluation_authorization, require_evaluation_authorization
from ..policy import PolicyOptions
from ..selection_quality import TauSelectionPolicy
from ..support import EventSupport
from .calibration_job import collect_and_calibrate
from .deployment import FrozenRelease, frozen_export_plan, load_training_model, save_frozen_release
from .inference import FrozenModelPredictor
from .simulation import M3Config, benchmark_matches, run_m3
from .validation_policy import select_validation_tau


def _read(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > 32 * 1024**2:
        raise ValueError("Missing/oversized job configuration or immutable artifact")
    return json.loads(source.read_text(encoding="utf-8"))


def _opponents(record: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = record["opponent_releases"]
    if len(paths) != 3:
        raise ValueError("Three explicitly fixed opponents are required")
    return Path(paths[0]), Path(paths[1]), Path(paths[2])


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Frozen-model jobs; writing code never starts these jobs")
    parser.add_argument("job", choices=("m3", "benchmark", "calibrate", "select-tau", "release"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    record = _read(args.config)
    authorization = read_evaluation_authorization(args.authorization)
    operations = {
        "m3": "run_m3",
        "benchmark": "benchmark_matches",
        "calibrate": "calibrate_frozen_logits",
        "select-tau": "select_validation_tau",
        "release": "export_frozen_release",
    }
    digest = record["plan_sha256"]
    require_evaluation_authorization(authorization, operation=operations[args.job], plan_sha256=digest)
    device = record.get("device", "cpu")
    output = Path(record["output"])
    if args.job == "m3":
        plan = record["paired_seed_manifest"]
        if isinstance(plan, str):
            plan = _read(plan)
        if plan["paired_seed_manifest_sha256"] != digest:
            raise ValueError("M3 plan digest differs from approval")
        run_m3(
            M3Config(
                plan,
                Path(record["subject_release"]),
                _opponents(record),
                output,
                Path(record["test_registry"]),
                BudgetPlan.from_record(record["budget"]),
                device=device,
                max_decisions=record.get("max_decisions", 10000),
                bootstrap_seed=record.get("bootstrap_seed", 17),
                bootstrap_replicates=record.get("bootstrap_replicates", 10000),
                benchmark_report=_read(record["benchmark_report"]),
            ),
            authorization=authorization,
        )
    elif args.job == "benchmark":
        benchmark_matches(
            Path(record["subject_release"]),
            _opponents(record),
            matches=record["matches"],
            seed=record["seed"],
            output=output,
            budget=BudgetPlan.from_record(record["budget"]),
            authorization=authorization,
            device=device,
            mode=record.get("mode", "paired"),
            max_decisions=record.get("max_decisions", 10000),
        )
    elif args.job == "calibrate":
        if "quality_policy" not in record:
            raise ValueError("Register heldout calibration-quality criteria before computing logits")
        quality = CalibrationQualityPolicy.from_record(record["quality_policy"])
        budget = BudgetPlan.from_record(record["budget"])

        def load_predictor():
            model, metadata = load_training_model(Path(record["checkpoint"]), device=device)
            contract = metadata["release_contract"]
            return FrozenModelPredictor(
                model, contract["data_manifest_sha256"], use_rule_features=contract["training"]["use_rule_features"]
            )

        collect_and_calibrate(
            load_predictor,
            Path(record["dataset"]),
            _read(record["validation_manifest"]),
            output,
            budget=budget,
            plan_sha256=digest,
            authorization=authorization,
            temperatures=tuple(record.get("temperatures", [0.5, 1.0, 2.0])),
            microbatch=record.get("microbatch", 1),
            quality_policy=quality,
        )
    elif args.job == "select-tau":
        budget = BudgetPlan.from_record(record["budget"])
        opponent_paths = _opponents(record)
        grid = tuple(
            math.inf if value == "infinity" else float(value)
            for value in record.get("tau_grid", ["infinity", 1.0, 0.5, 0.25])
        )
        select_validation_tau(
            lambda: FrozenRelease(Path(record["subject_release"]), device=device),
            lambda: tuple(FrozenRelease(path, device=device) for path in opponent_paths),
            output,
            budget=budget,
            seed=record["seed"],
            plan_sha256=digest,
            authorization=authorization,
            grid=grid,
            selection_policy=TauSelectionPolicy.from_record(record["selection_policy"])
            if record.get("selection_policy")
            else None,
        )
    else:
        job = _read(record["calibration_job"]) if record.get("calibration_job") else None
        tau_job = _read(record["tau_selection_job"]) if record.get("tau_selection_job") else None
        policy_record = record.get("policy", {"mode": "prior_only", "tau": "infinity"})
        export_plan = frozen_export_plan(
            Path(record["checkpoint"]),
            device=device,
            policy=policy_record,
            calibration_job=job,
            tau_selection_job=tau_job,
        )
        if export_plan["plan_sha256"] != digest:
            raise ValueError("Release export inputs differ from the approved plan")
        model, metadata = load_training_model(Path(record["checkpoint"]), device=device)
        contract = metadata["release_contract"]
        support = None
        calibration = None
        predictor = FrozenModelPredictor(
            model, contract["data_manifest_sha256"], use_rule_features=contract["training"]["use_rule_features"]
        )
        if record.get("calibration_job"):
            job = _read(record["calibration_job"])
            support = EventSupport.from_record(job["support"], predictor.binding)
            calibration = CalibrationArtifact.from_record(job["calibration"], predictor.binding, support)
        policy = dict(record.get("policy", {"mode": "prior_only", "tau": "infinity"}))
        policy["tau"] = math.inf if policy["tau"] == "infinity" else float(policy["tau"])
        selection = _read(record["tau_selection_job"])["tau_selection"] if record.get("tau_selection_job") else None
        save_frozen_release(
            output,
            model,
            train_manifest_sha256=contract["data_manifest_sha256"],
            use_rule_features=contract["training"]["use_rule_features"],
            training_contract=contract,
            support=support,
            calibration=calibration,
            options=PolicyOptions(**policy),
            tau_selection=selection,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
