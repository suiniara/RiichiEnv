from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..authorization import UpdateAuthorization, load_update_authorization, require_parameter_updates
from ..budget import BudgetLedger, CostForecast, choose_training_tier
from ..contracts import content_hash
from ..evaluation_authorization import exclusive_job
from ..versioning import validate_artifact_contract
from .distributed import DistributedContext, StartupCoordinator
from .prerequisites import m2_prerequisites
from .production_checkpoint import advance_training_startup, create_training_directory, read_training_startup
from .trainer import _startup_identity, train_production
from .training_config import ProductionTrainConfig, main_model_registration


def _read_run(path: Path, config: ProductionTrainConfig, dataset_sha256: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "run_manifest_sha256"}
    if (
        record.get("status") != "PASS"
        or record.get("run_manifest_sha256") != content_hash(unsigned)
        or record["release_contract"]["training"]["recipe_sha256"] != config.recipe_sha256
    ):
        raise ValueError("Prior completed phase evidence was modified or has another recipe")
    if record.get("dataset_manifest_sha256") != dataset_sha256:
        raise ValueError("Completed arm belongs to another training selection")
    validate_artifact_contract(record["release_contract"], context="completed phase")
    from ..release import implementation_release_identity  # noqa: PLC0415

    if (
        record["release_contract"]["implementation"]["source_sha256"]
        != implementation_release_identity()["source_sha256"]
    ):
        raise ValueError("A completed arm used another source build; do not mix code versions in an ablation")
    pointer = json.loads((path.parent / "checkpoint.json").read_text(encoding="utf-8"))
    generation = path.parent / pointer["checkpoint_file"]
    if generation.parent.resolve() != path.parent.resolve() or not generation.name.startswith("checkpoint.g"):
        raise ValueError("Checkpoint receipt points outside this run")
    if (
        pointer["checkpoint_sha256"] != record["checkpoint_sha256"]
        or hashlib.sha256(generation.read_bytes()).hexdigest() != record["checkpoint_sha256"]
    ):
        raise ValueError("Completed run weights changed")
    ledger = BudgetLedger(path.parent.parent / (config.phase + "-costs.jsonl"), config.budget, allow_create=False)
    ledger.verify_state(record["budget_ledger"])
    return record


@contextmanager
def _phase_job(  # noqa: PLR0915 -- coordinated bootstrap and lease cleanup are one lifecycle
    root: Path, config: ProductionTrainConfig, *, dataset_sha256: str
) -> Iterator[DistributedContext]:
    coordinator = StartupCoordinator.open(root / ".phase-startup")
    context = None
    lease = None
    ledger = None
    error = None
    entered = False
    try:
        if coordinator.primary:
            try:
                root.mkdir(parents=True, exist_ok=True)
                lease = exclusive_job(root / "phase.lock")
                lease.__enter__()
                entered = True
                directory, ledger_path = root / ".phase-startup", root / (config.phase + "-costs.jsonl")
                identity = _startup_identity(
                    {"manifest_sha256": dataset_sha256},
                    config,
                    coordinator,
                    output_dir=root,
                    ledger_path=ledger_path,
                    registration=None,
                )
                if not directory.exists():
                    create_training_directory(directory, identity=identity, ledger_path=ledger_path)
                startup = read_training_startup(
                    directory, identity=identity, ledger_path=ledger_path, checkpoint_exists=False
                )
                assert startup is not None
                ledger = BudgetLedger(
                    ledger_path,
                    config.budget,
                    allow_create=startup["phase"] == "preparing" and startup["ledger_origin"] is None,
                )
                if startup["budget_ledger"] is not None:
                    ledger.verify_state(startup["budget_ledger"])
                startup = advance_training_startup(
                    directory, startup, phase="initializing", budget_ledger=ledger.state_dict()
                )
                ledger.reserve_next(0.0)
            except Exception as caught:
                error = caught
        coordinator.coordinate(error, "phase startup accounting/lease")
        error = None
        try:
            if config.device == "cuda":
                # Make an ordinary CUDA setup failure visible before peers enter group initialization.
                import torch  # noqa: PLC0415

                torch.cuda.set_device(coordinator.local_rank)
        except Exception as caught:
            error = caught
        coordinator.coordinate(error, "phase device initialization")
        error = None
        try:
            context = DistributedContext.open(
                config.device, backend=config.distributed_backend, gloo_bind_host=config.gloo_bind_host
            )
        except Exception as caught:
            error = caught
        coordinator.coordinate(error, "phase distributed initialization")
        assert context is not None
        error = None
        if coordinator.primary:
            try:
                assert ledger is not None
                assert startup is not None
                ledger.finish_compute("phase_distributed_startup")
                advance_training_startup(directory, startup, phase="initializing", budget_ledger=ledger.state_dict())
                ledger = None  # The trainer takes over this same ledger with a new measured interval.
            except Exception as caught:
                error = caught
        context.coordinate(error, "phase startup settlement")
        yield context
    finally:
        if ledger is not None:
            # Only owns the bootstrap interval; the inner trainer opens a separate attempt in this same ledger.
            # Reopening here would double-charge inner training or claim another owner's pending attempt.
            try:
                ledger.finish_compute("aborted_phase_startup")
            except Exception:
                pass
        if lease is not None and entered:
            lease.__exit__(None, None, None)
        if context is not None:
            context.close()


def run_m1_ablation(
    dataset: Path,
    output: Path,
    on: ProductionTrainConfig,
    off: ProductionTrainConfig,
    *,
    authorization: UpdateAuthorization | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    require_parameter_updates(authorization, operation="run_m1_ablation")
    for config in (on, off):
        require_parameter_updates(
            authorization, operation="run_m1_ablation", phase=config.phase, recipe_sha256=config.recipe_sha256
        )
    if on.phase != "M1" or off.phase != "M1" or not on.use_rule_features or off.use_rule_features:
        raise ValueError("M1 registers exactly feature-on and feature-off, Small, one epoch each")
    left, right = on.to_record(), off.to_record()
    for key in (
        "seed",
        "model_tier",
        "N_train_half_games",
        "N_eligible_half_games",
        "planned_epochs",
        "effective_batch_size",
        "microbatch_size",
        "bucket_window",
        "device",
        "precision",
    ):
        if left[key] != right[key]:
            raise ValueError("Ablation must use identical initialization,data,order,batch and precision")
    if (
        on.budget.budget_cap_usd != off.budget.budget_cap_usd
        or on.budget.budget_spent_usd != off.budget.budget_spent_usd
    ):
        raise ValueError("Feature arms share one USD100 phase ledger, not two independent allowances")
    on.budget.admit(on.budget.forecast.usd + off.budget.forecast.usd)
    data_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    dataset_sha = data_manifest["manifest_sha256"]
    require_parameter_updates(authorization, operation="run_m1_ablation", dataset_hash=dataset_sha)
    with _phase_job(output, on, dataset_sha256=dataset_sha) as context:
        results = {}
        for name, config in (("feature-on", on), ("feature-off", off)):
            directory = output / name
            manifest = directory / "run_manifest.json"
            error = None
            prior = None
            try:
                if manifest.exists():
                    if not resume:
                        raise ValueError("Ablation output exists; explicit resume is required")
                    try:
                        prior = _read_run(manifest, config, dataset_sha)
                    except (ValueError, KeyError, OSError):
                        if not (directory / "checkpoint.json").is_file():
                            raise
                        # A committed checkpoint, not a partial publication, is the
                        # authority for reconstructing a completed arm on resume.
                        prior = None
            except Exception as caught:
                error = caught
            context.coordinate(error, "M1 completed arm integrity")
            result = (
                prior
                if prior is not None
                else train_production(
                    dataset,
                    directory,
                    config,
                    authorization=authorization,
                    resume=resume and directory.exists(),
                    ledger_path=output / "M1-costs.jsonl",
                )
            )
            results[name] = result
        first, second = results["feature-on"], results["feature-off"]
        if (
            first["initial_weights_sha256"] != second["initial_weights_sha256"]
            or first["epoch_order_sha256"] != second["epoch_order_sha256"]
            or first["processed_examples"] != second["processed_examples"]
        ):
            raise ValueError("Ablation initialization or actual committed sample order differs")
        result = {
            "status": "PASS",
            "phase": "M1",
            "complete_feature_ablation": True,
            "results": results,
            "same_initialization": True,
            "same_sample_order": True,
            "same_updates": first["step"] == second["step"],
            "conclusion_scope": "pipeline/conditional-metrics/feature-diagnostics,not a strength result",
        }
        result["report_sha256"] = content_hash(result)
        if context.primary:
            (output / "M1-ablation.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        return result


def run_m2(
    dataset: Path,
    output: Path,
    config: ProductionTrainConfig,
    forecasts: dict[str, CostForecast],
    *,
    main_model_registry: Path,
    m1_report: Path | None = None,
    m1_tau_selection_job: Path | None = None,
    authorization: UpdateAuthorization | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    require_parameter_updates(authorization, operation="run_m2")
    require_parameter_updates(authorization, operation="run_m2", phase=config.phase, recipe_sha256=config.recipe_sha256)
    if config.phase != "M2":
        raise ValueError("M2 requires its registered two/conditional-three epoch recipe")
    if m1_report is None:
        raise ValueError("Supply completed M1 feature-ablation evidence before M2")
    selected = choose_training_tier(
        config.N_train_half_games,
        forecasts,
        budget_spent_usd=config.budget.budget_spent_usd,
        fixed_phase_cap_usd=config.budget.budget_cap_usd,
    )
    if selected != config.model_tier or forecasts[selected].to_record() != config.budget.forecast.to_record():
        raise ValueError("Register the unique data/budget-selected tier before launch; do not train a tier matrix")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    require_parameter_updates(authorization, operation="run_m2", dataset_hash=manifest["manifest_sha256"])
    prerequisite_evidence = m2_prerequisites(
        m1_report, config=config, dataset_sha256=manifest["manifest_sha256"], tau_selection_job=m1_tau_selection_job
    )
    registration = main_model_registration(
        manifest["manifest_sha256"], config, output, prerequisites=prerequisite_evidence
    )
    with _phase_job(output, config, dataset_sha256=manifest["manifest_sha256"]) as context:
        error = None
        if context.primary:
            try:
                if not main_model_registry.parent.is_dir():
                    raise ValueError("Main-model registry parent must be explicitly selected")
                with exclusive_job(main_model_registry.with_suffix(".lock")):
                    if main_model_registry.exists():
                        if json.loads(main_model_registry.read_text(encoding="utf-8")) != registration:
                            raise ValueError("A different main model is already registered for this project")
                    else:
                        with main_model_registry.open("x", encoding="utf-8") as stream:
                            json.dump(registration, stream, indent=2)
            except Exception as caught:
                error = caught
        context.coordinate(error, "unique M2 model registration")
        target = output / "training"
        resumed = None
        resume_error = None
        if context.primary and resume and (target / "run_manifest.json").exists():
            try:
                resumed = _read_run(target / "run_manifest.json", config, manifest["manifest_sha256"])
            except (ValueError, KeyError, OSError) as caught:
                if not (target / "checkpoint.json").is_file():
                    resume_error = caught
        context.coordinate(resume_error, "M2 resume manifest inspection")
        resumed = context.broadcast_from_primary(resumed)
        if resumed is not None:
            return resumed
        return train_production(
            dataset,
            target,
            config,
            authorization=authorization,
            resume=resume,
            ledger_path=output / "M2-costs.jsonl",
            main_model_registry=main_model_registry,
        )


def torchrun_command(
    *,
    processes: int,
    dataset: Path,
    output: Path,
    config: Path,
    authorization: Path,
    resume: bool = False,
    main_model_registry: Path | None = None,
) -> list[str]:
    if processes < 1:
        raise ValueError("Positive worker count required")
    import sys  # noqa: PLC0415

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--max-restarts=0",
        f"--nproc-per-node={processes}",
        "-m",
        "riichienv.kar.learning.trainer",
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--config",
        str(config),
        "--authorization",
        str(authorization),
    ]
    if main_model_registry is not None:
        command.extend(["--main-model-registry", str(main_model_registry)])
    if resume:
        command.append("--resume")
    return command


def config_from_record(record: dict[str, Any]) -> ProductionTrainConfig:
    return ProductionTrainConfig.from_record(record)


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicit-authorization M1/M2 orchestration, no implicit training")
    parser.add_argument("phase", choices=("M1", "M2"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    auth = load_update_authorization(args.authorization)
    require_parameter_updates(auth, operation="run_m1_ablation" if args.phase == "M1" else "run_m2")
    record = json.loads(args.config.read_text(encoding="utf-8"))
    if args.phase == "M1":
        run_m1_ablation(
            args.dataset,
            args.output,
            config_from_record(record["on"]),
            config_from_record(record["off"]),
            authorization=auth,
            resume=args.resume,
        )
    else:
        run_m2(
            args.dataset,
            args.output,
            config_from_record(record["training"]),
            {key: CostForecast.from_record(value) for key, value in record["forecasts"].items()},
            main_model_registry=Path(record["main_model_registry"]),
            m1_report=Path(record["m1_report"]) if record.get("m1_report") else None,
            m1_tau_selection_job=Path(record["m1_tau_selection_job"]) if record.get("m1_tau_selection_job") else None,
            authorization=auth,
            resume=args.resume,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
