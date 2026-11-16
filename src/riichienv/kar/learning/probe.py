from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..authorization import UpdateAuthorization, load_update_authorization, require_parameter_updates
from ..contracts import content_hash
from .benchmark import benchmark_candidates, peak_process_memory_bytes
from .train import TrainConfig, train_cpu


def cpu_probe(dataset: Path, output: Path, *, authorization: UpdateAuthorization | None = None) -> dict[str, Any]:
    require_parameter_updates(authorization, operation="cpu_probe")
    output.mkdir(parents=True, exist_ok=False)
    config = TrainConfig(max_updates=16, effective_batch_size=8, microbatch_size=1)
    feature_on = train_cpu(dataset, output / "feature-on", config, authorization=authorization)
    feature_off = train_cpu(
        dataset, output / "feature-off", replace(config, use_rule_features=False), authorization=authorization
    )
    same_order = feature_on["sample_order_sha256"] == feature_off["sample_order_sha256"]
    same_weights = (
        feature_on["run_contract"]["initial_weights_sha256"] == feature_off["run_contract"]["initial_weights_sha256"]
    )
    if not same_order or not same_weights:
        raise RuntimeError("Feature ablation did not preserve initialization/sample order")
    # A separate diagnostic deliberately revisits eight train rows; never a validation/test curriculum.
    overfit = train_cpu(
        dataset,
        output / "overfit-eight",
        replace(config, max_updates=32, overfit_examples=8),
        authorization=authorization,
    )
    first, last = overfit["train_history"][0]["objective"], overfit["train_history"][-1]["objective"]
    if not last < first:
        raise RuntimeError("The bounded real-data overfit diagnostic did not reduce its training objective")
    benchmark = benchmark_candidates(dataset)
    report = {
        "status": "PASS",
        "milestone": "M1_CPU_PIPELINE",
        "complete_M1_probe": False,
        "model_parameter_count": feature_on["parameter_count"],
        "dataset_manifest_sha256": feature_on["run_contract"]["dataset_manifest_sha256"],
        "model_parameters_identical_between_feature_modes": feature_on["parameter_count"]
        == feature_off["parameter_count"],
        "same_seed_split_sample_order": same_order,
        "same_initial_weights": same_weights,
        "effective_batch_size": config.effective_batch_size,
        "official_full_probe_effective_batch_size": 128,
        "feature_on": feature_on,
        "feature_off": feature_off,
        "overfit_diagnostic": {
            "examples_revisited": 8,
            "updates": overfit["completed_updates"],
            "first_objective": first,
            "last_objective": last,
            "report": str(output / "overfit-eight" / "run_manifest.json"),
        },
        "inference_benchmark": benchmark,
        "peak_probe_process_memory_bytes": peak_process_memory_bytes(),
        "not_run": [
            "full_train_split_one_epoch_feature_on_and_off",
            "GPU_memory_and_cost_probe",
            "post_training_probability_calibration",
            "M2_main_training",
            "M3_paired_matches",
        ],
        "cloud_spent_usd": 0.0,
        "gpu_resources_started": False,
        "playing_strength_claimed": False,
    }
    report["report_sha256"] = content_hash(report)
    (output / "cpu-probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded Small M1 CPU ablation and overfit diagnostics")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-authorization", type=Path)
    args = parser.parse_args()
    report = cpu_probe(args.dataset, args.output, authorization=load_update_authorization(args.update_authorization))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"feature_on", "feature_off"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
