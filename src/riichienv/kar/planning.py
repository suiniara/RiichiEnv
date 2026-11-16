from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .authorization import update_implementation_sha256
from .budget import BudgetPlan, CostForecast, FixedCostReceipt
from .contracts import content_hash
from .experiments import M3_MATCHES, paired_seed_manifest
from .simulation_contracts import benchmark_plan_from_records, qualify_benchmark_report, validate_m3_forecast


def _read(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > 32 * 1024**2:
        raise ValueError("Missing/oversized planning metadata")
    return json.loads(source.read_text(encoding="utf-8"))


def _releases(config: dict[str, Any]) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    opponents = config["opponent_releases"]
    if len(opponents) != 3:
        raise ValueError("Preregister exactly three immutable opponents")
    return _read(Path(config["subject_release"]) / "release.json"), tuple(
        _read(Path(path) / "release.json") for path in opponents
    )


def plan_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    subject, opponents = _releases(config)
    plan = benchmark_plan_from_records(
        subject,
        opponents,
        matches=config.get("matches", 8),
        seed=config["seed"],
        device=config.get("device", "cpu"),
        source_sha256=update_implementation_sha256(),
        mode=config.get("mode", "paired"),
        max_decisions=config.get("max_decisions", 10000),
    )
    return {
        "plan_sha256": plan["plan_sha256"],
        "benchmark_plan": plan,
        "planning_only": True,
        "model_computation_performed": False,
        "authorization_granted": False,
        "note": "Bind this plan and the complete M1 budget in an independently approved benchmark job",
    }


def plan_m3(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = _read(config["benchmark_report"])
    evidence = qualify_benchmark_report(benchmark)
    subject, opponents = _releases(config)
    measured = evidence["plan"]
    if (
        subject["release_sha256"] != measured["subject_release_sha256"]
        or [value["release_sha256"] for value in opponents] != measured["opponent_release_ids"]
    ):
        raise ValueError("M3 must use the exact subject/opponents from the completed paired benchmark")
    plan = paired_seed_manifest(
        master_seed=config["master_seed"],
        subject_release_sha256=subject["release_sha256"],
        opponent_release_ids=tuple(value["release_sha256"] for value in opponents),
        inference_device=measured["device"],
        max_decisions=measured["max_decisions"],
        statistics_seed=config.get("bootstrap_seed", 17),
        bootstrap_replicates=config.get("bootstrap_replicates", 10000),
        benchmark_sha256=evidence["benchmark_sha256"],
    )
    forecast = CostForecast(
        "M3",
        M3_MATCHES,
        evidence["matches_per_second"],
        config["resource_usd_per_hour"],
        config["fixed_cost_usd"],
        "benchmark:" + evidence["benchmark_sha256"],
        plan["paired_seed_manifest_sha256"],
        config.get("safety_multiplier", 1.2),
    )
    budget = BudgetPlan(
        "M3",
        config["budget_cap_usd"],
        config["budget_spent_usd"],
        forecast,
        other_phase_spent_usd=config["other_phase_spent_usd"],
        fixed_cost_receipts=tuple(FixedCostReceipt(**value) for value in config.get("fixed_cost_receipts", ())),
    )
    validate_m3_forecast(plan, forecast, benchmark, source_sha256=update_implementation_sha256())
    record = {
        "plan_sha256": plan["paired_seed_manifest_sha256"],
        "paired_seed_manifest": plan,
        "benchmark_report": str(Path(config["benchmark_report"]).resolve()),
        "budget": budget.to_record(),
        "subject_release": str(Path(config["subject_release"]).resolve()),
        "opponent_releases": [str(Path(value).resolve()) for value in config["opponent_releases"]],
        "output": str(Path(config["run_output"]).resolve()),
        "test_registry": str(Path(config["test_registry"]).resolve()),
        "device": measured["device"],
        "max_decisions": measured["max_decisions"],
        "bootstrap_seed": plan["statistics"]["seed"],
        "bootstrap_replicates": plan["statistics"]["bootstrap_replicates"],
        "planning_only": True,
        "model_computation_performed": False,
        "authorization_granted": False,
    }
    return {**record, "planning_record_sha256": content_hash(record)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Metadata-only frozen plans; never loads model weights or starts a job"
    )
    parser.add_argument("kind", choices=("benchmark", "m3"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _read(args.config)
    record = plan_benchmark(config) if args.kind == "benchmark" else plan_m3(config)
    if not args.output.parent.is_dir():
        raise ValueError("Select an existing planning-output directory outside raw inputs")
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"plan_sha256": record["plan_sha256"], "planning_only": True, "authorization_granted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
