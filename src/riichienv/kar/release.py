from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .contracts import content_hash
from .runtime_contracts import runtime_contract
from .versioning import validate_artifact_contract

BUILD_IDENTITY_VERSION = "packaged_source_identity_v1"
RELEASE_VERSION = "complete_kar_release_v1"


def source_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.suffix in {".py", ".json"} and path.name != "_build_identity.json"
    }


def write_build_identity(package: Path, *, code_commit: str, dirty: bool, source_reference: str) -> dict[str, Any]:
    if (
        len(code_commit) != 40
        or any(character not in "0123456789abcdef" for character in code_commit)
        or not source_reference
    ):
        raise ValueError("Build identity needs an explicit real commit and build source")
    files = source_files(package)
    record = {
        "version": BUILD_IDENTITY_VERSION,
        "code_commit": code_commit,
        "source_worktree_dirty": dirty,
        "source_reference": source_reference,
        "source_files": files,
        "source_sha256": content_hash(files),
    }
    record["build_identity_sha256"] = content_hash(record)
    path = package / "_build_identity.json"
    if path.exists():
        raise ValueError("Refuse to replace build provenance without an explicit new build")
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def implementation_release_identity() -> dict[str, Any]:
    package = Path(__file__).parent
    files = source_files(package)
    built = package / "_build_identity.json"
    if built.exists():
        record = json.loads(built.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in record.items() if key != "build_identity_sha256"}
        if (
            record.get("version") != BUILD_IDENTITY_VERSION
            or record.get("source_files") != files
            or record.get("build_identity_sha256") != content_hash(unsigned)
        ):
            raise ValueError("Packaged code differs from its build identity")
    else:
        repository = next(
            (
                parent
                for parent in package.parents
                if (parent / ".git").exists() and (parent / "riichienv-core").is_dir()
            ),
            None,
        )
        if repository is None:
            raise ValueError(
                "Installed production package needs _build_identity.json; do not substitute an unknown commit"
            )
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout
        record = {
            "version": BUILD_IDENTITY_VERSION,
            "code_commit": commit,
            "source_worktree_dirty": bool(status),
            "source_reference": str(repository),
            "source_files": files,
            "source_sha256": content_hash(files),
        }
        record["build_identity_sha256"] = content_hash(record)
    native = importlib.import_module("riichienv._riichienv")
    path = Path(native.__file__)
    return {
        **record,
        "native_extension_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "python_implementation": sys.implementation.name,
        "platform": platform.platform(),
    }


def complete_release_contract(
    *,
    training: dict[str, Any],
    data_manifest: dict[str, Any],
    budget: dict[str, Any],
    calibration: dict[str, Any] | None = None,
    support: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    paired_seed_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .learning.training_config import ProductionTrainConfig  # noqa: PLC0415

    validate_artifact_contract(data_manifest["scope"], context="release dataset")
    config = ProductionTrainConfig.from_record(training)
    if (
        training["N_train_half_games"] != data_manifest.get("N_train_half_games")
        or training["N_eligible_half_games"] != data_manifest.get("N_eligible_half_games")
        or config.budget.to_record() != budget
        or config.budget.forecast.recipe_sha256 != config.recipe_sha256
    ):
        raise ValueError("Release training/data/forecast scale or recipe differs")
    record = {
        "release_version": RELEASE_VERSION,
        **runtime_contract(),
        "implementation": implementation_release_identity(),
        "initial_seat_order": [0, 1, 2, 3],
        "rank_tiebreak": "initial_seat_order_east_south_west_north",
        "rule_feature_mask_probability": training["feature_mask_probability"],
        "use_rule_features": training["use_rule_features"],
        "training": training,
        **{key: training[key] for key in (
            "model_tier", "N_train_half_games", "N_eligible_half_games", "planned_epochs", "effective_batch_size"
        )},
        "data_manifest_sha256": data_manifest["manifest_sha256"],
        "split_salt": data_manifest["scope"]["split_salt"],
        "rho_E": data_manifest["rho_E"],
        "rho_V": data_manifest["rho_V"],
        "lambda_E": 0.25,
        "lambda_V": 0.1,
        "weight_decay_parameter_groups": {"value_head.weight": 0.05, "other_matrices": 0.01, "bias_norm": 0.0},
        "budget": budget,
        "paired_seed_manifest": paired_seed_manifest,
        "calibration": calibration,
        "event_support": support,
        "policy": policy,
        "calibration_status": "heldout_artifact" if calibration is not None else "unfitted",
        "evidence_status": (policy or {}).get("evidence_status", "certificate_unavailable"),
        "feature_ablation_mode": "feature_on" if training["use_rule_features"] else "feature_off",
    }
    return {**record, "release_contract_sha256": content_hash(record)}


def main() -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Stamp immutable source provenance before packaging; no model computation"
    )
    parser.add_argument("--package", type=Path, default=Path(__file__).parent)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--dirty", action="store_true")
    parser.add_argument("--source-reference", required=True)
    args = parser.parse_args()
    record = write_build_identity(
        args.package, code_commit=args.commit, dirty=args.dirty, source_reference=args.source_reference
    )
    print(
        json.dumps(
            {"build_identity_sha256": record["build_identity_sha256"], "source_sha256": record["source_sha256"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
