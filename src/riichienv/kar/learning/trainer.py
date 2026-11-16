from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from ..authorization import UpdateAuthorization, load_update_authorization, require_parameter_updates
from ..budget import BudgetLedger
from ..contracts import content_hash, model_tier
from ..evaluation_authorization import exclusive_job
from ..release import complete_release_contract
from .batch import feature_mask_draw, rule_feature_view
from .config import ModelConfig
from .dataset import DecisionDataset, TrainingBatch, collate_examples
from .distributed import DistributedContext, dist
from .evaluation import evaluate_complete
from .inference import model_weights_sha256
from .losses import Labels, loss_sums
from .model import KARTransformer
from .production_checkpoint import (
    rank_rng_state,
    read_production_checkpoint,
    restore_rank_rng,
    save_production_checkpoint,
)
from .run_publication import completed_run_record, publish_run_manifest
from .sampling import GlobalEpochSampler, UpdateGroup, validation_updates
from .train import create_optimizer, create_scheduler
from .training_config import ProductionTrainConfig, third_epoch_allowed, validate_main_model_registration


def _training_batch(
    dataset: DecisionDataset,
    indices: tuple[int, ...],
    fallback_index: int,
    config: ProductionTrainConfig,
    epoch: int,
    device: torch.device,
) -> tuple[TrainingBatch, bool]:
    real = bool(indices)
    batch = collate_examples([dataset[index] for index in indices] if real else [dataset[fallback_index]]).to(device)
    mask = torch.tensor(
        [
            feature_mask_draw(config.seed, epoch, game, row, probability=config.feature_mask_probability)
            for game, row in zip(batch.game_ids, batch.inputs.context_ids)
        ],
        dtype=torch.bool,
        device=device,
    )
    inputs = rule_feature_view(batch.inputs, mask, enabled=config.use_rule_features)
    labels = batch.labels
    if not real:
        no = torch.zeros(inputs.size, dtype=torch.bool, device=device)
        labels = Labels(
            labels.action_target,
            no,
            labels.response_target,
            no,
            no,
            labels.final_rank_target,
            no,
            labels.response_allowed,
        )
    return TrainingBatch(inputs, labels, batch.game_ids), real


def production_update(  # noqa: PLR0915
    model: torch.nn.Module,
    raw: KARTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    dataset: DecisionDataset,
    group: UpdateGroup,
    context: DistributedContext,
    config: ProductionTrainConfig,
    authorization: UpdateAuthorization | None,
) -> dict[str, Any]:
    require_parameter_updates(authorization, operation="production_update")
    require_parameter_updates(
        authorization,
        operation="production_update",
        dataset_hash=dataset.manifest["manifest_sha256"],
        phase=config.phase,
        recipe_sha256=config.recipe_sha256,
    )
    global_count = context.global_count(len(group.local_indices))
    if global_count != group.global_examples:
        raise ValueError("Sampler/DDP global_N mismatch")
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    started = time.perf_counter()
    data_seconds = 0.0
    tokens = 0
    padded = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    objective = 0.0
    examples = 0
    totals = {"action": 0.0, "response": 0.0, "rank": 0.0}
    for micro in range(group.microbatch_rounds):
        indices = group.local_indices[micro * config.microbatch_size : (micro + 1) * config.microbatch_size]
        batch = None
        error = None
        loaded_at = time.perf_counter()
        try:
            batch, real = _training_batch(
                dataset, indices, group.global_indices[0], config, group.epoch, context.device
            )
        except Exception as caught:
            error = caught
        context.coordinate(error, "microbatch materialization")
        data_seconds += time.perf_counter() - loaded_at
        assert batch is not None
        if real:
            tokens += int(batch.inputs.fields["valid"].sum())
            padded += batch.inputs.fields["valid"].numel()
        synchronize = model.no_sync() if isinstance(model, DistributedDataParallel) else nullcontext()
        with synchronize:
            error = None
            loss = None
            sums = None
            try:
                autocast = (
                    torch.autocast(device_type=context.device.type, dtype=torch.bfloat16)
                    if config.precision == "bf16"
                    else nullcontext()
                )
                with autocast:
                    outputs = model(batch.inputs)
                sums = loss_sums(outputs, batch.labels, batch.inputs.legal_mask)
                loss = sums.objective(
                    global_examples=global_count,
                    world_size=context.world_size,
                    rho_e=dataset.manifest["rho_E"],
                    rho_v=dataset.manifest["rho_V"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite AMP/FP32 objective")
            except Exception as caught:
                error = caught
            context.coordinate(error, "microbatch forward/loss")
            assert loss is not None and sums is not None
            error = None
            try:
                loss.backward()
            except Exception as caught:
                error = caught
            context.coordinate(error, "microbatch backward")
        if real:
            examples += batch.inputs.size
            objective += float(loss.detach())
            totals["action"] += float(sums.action.detach())
            totals["response"] += float(sums.response.detach())
            totals["rank"] += float(sums.value.detach())
    context.average_gradients(raw)
    error = None
    norm = None
    try:
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0, error_if_nonfinite=True)
    except Exception as caught:
        error = caught
    context.coordinate(error, "global norm clipping")
    error = None
    try:
        optimizer.step()
    except Exception as caught:
        error = caught
    # A failed step aborts the entire worker job; never retry partial weights.
    context.coordinate(error, "optimizer transaction")
    error = None
    try:
        scheduler.step()
    except Exception as caught:
        error = caught
    context.coordinate(error, "scheduler commit")
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    reports = context.gather(
        {
            "examples": examples,
            "objective": objective,
            "sums": totals,
            "seconds": time.perf_counter() - started,
            "tokens": tokens,
            "padded_tokens": padded,
            "data_seconds": data_seconds,
        }
    )
    if sum(row["examples"] for row in reports) != global_count:
        raise ValueError("Dummy rows must not enter normalization or metrics")
    assert norm is not None
    return {
        "global_examples": global_count,
        "objective": math.fsum(row["objective"] for row in reports) / context.world_size,
        "gradient_norm_before_clip": float(norm),
        "loss_sums": {head: math.fsum(row["sums"][head] for row in reports) for head in totals},
        "update_committed": True,
        "step_elapsed_seconds": max(row["seconds"] for row in reports),
        "real_tokens": sum(row["tokens"] for row in reports),
        "padded_real_tokens": sum(row["padded_tokens"] for row in reports),
        "precision": config.precision,
        "decision_samples_per_second": global_count / max(row["seconds"] for row in reports),
        "tokens_per_second": sum(row["tokens"] for row in reports) / max(row["seconds"] for row in reports),
        "data_wait_seconds_sum": math.fsum(row["data_seconds"] for row in reports),
        "padding_fraction": 1
        - sum(row["tokens"] for row in reports) / max(1, sum(row["padded_tokens"] for row in reports)),
    }


def _dataset_contract(dataset: DecisionDataset, validation: DecisionDataset, config: ProductionTrainConfig) -> None:
    manifest = dataset.manifest
    if not len(dataset) or not len(validation) or manifest["N_train_half_games"] != config.N_train_half_games:
        raise ValueError(
            "Production data must have actual nonempty train/validation and exact complete-train-game evidence"
        )
    available = manifest["scope"]["available_split_counts"]
    if sum(available.values()) != config.N_eligible_half_games:
        raise ValueError("Eligible-game evidence differs from M0 selection")
    if config.phase == "M2" and (
        manifest["is_probe_subset"]
        or manifest["counts"]["train"]["games"] != available["train"]
        or manifest["counts"]["validation"]["games"] != available["validation"]
    ):
        raise ValueError("M2 cannot silently train/evaluate a diagnostic subset")
    if ("small", "medium", "large").index(config.model_tier) > ("small", "medium", "large").index(
        model_tier(config.N_train_half_games)
    ):
        raise ValueError("Model tier exceeds the conservative train-split ladder")
    if not 0 < manifest["rho_V"] <= manifest["rho_E"] <= 1:
        raise ValueError("Three-head production training requires fixed positive manifest rho")


def _append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _model_components(
    train: DecisionDataset, config: ProductionTrainConfig, context: DistributedContext, lengths: list[int], updates: int
):
    raw = KARTransformer(ModelConfig.from_preset(config.model_tier)).to(context.device)
    optimizer = create_optimizer(raw, config.learning_rate)
    scheduler = create_scheduler(optimizer, updates * config.planned_epochs)
    sampler = GlobalEpochSampler(
        lengths,
        seed=config.seed,
        effective_batch=config.effective_batch_size,
        microbatch=config.microbatch_size,
        rank=context.rank,
        world_size=context.world_size,
        bucket_window=config.bucket_window,
    )
    runtime = {
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "device_type": context.device.type,
        "precision": config.precision,
        "world_size": context.world_size,
        "backend": str(dist.get_backend()) if context.world_size > 1 else "single",
        "deterministic_algorithms": config.deterministic_algorithms,
        "tf32": False,
        "master_parameter_dtype": "float32",
        "device_capability": list(torch.cuda.get_device_capability(context.device))
        if context.device.type == "cuda"
        else None,
        "cudnn_version": torch.backends.cudnn.version() if context.device.type == "cuda" else None,
        "scaler": None,
    }
    contract = complete_release_contract(
        training=config.to_record(), data_manifest=train.manifest, budget=config.budget.to_record()
    )
    state = {
        "step": 0,
        "processed_examples": 0,
        "validation_history": [],
        "epoch_end_validation": {},
        "third_epoch_decision": None,
        "initial_weights_sha256": model_weights_sha256(raw),
        "epoch_order_sha256": {},
        "completed_epochs": 0,
        "finished": False,
        "performance": {
            "updates": 0,
            "examples": 0,
            "real_tokens": 0,
            "padded_real_tokens": 0,
            "training_seconds": 0.0,
            "data_wait_seconds_sum": 0.0,
            "validation_seconds": 0.0,
            "validation_passes": 0,
        },
    }
    return raw, optimizer, scheduler, sampler, runtime, contract, state


def _publish_completion(
    raw: KARTransformer,
    state: dict[str, Any],
    *,
    config: ProductionTrainConfig,
    context: DistributedContext,
    contract: dict[str, Any],
    ledger: BudgetLedger | None,
    checkpoint: Path,
    dataset_sha256: str,
    recovered: bool,
) -> dict[str, Any]:
    report = None
    error = None
    if context.primary:
        try:
            assert ledger is not None
            final_sha = model_weights_sha256(raw)
            ledger.settle_elapsed("recover_completed_publication" if recovered else "complete_publication")
            pointer = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
            record = completed_run_record(
                state,
                phase=config.phase,
                planned_epochs=config.planned_epochs,
                runtime=pointer["metadata"]["runtime"],
                contract=contract,
                budget=ledger.state_dict(),
                dataset_sha256=dataset_sha256,
                final_weights_sha256=final_sha,
                checkpoint=pointer,
                recovered=recovered,
            )
            report = publish_run_manifest(checkpoint.parent / "run_manifest.json", record)
        except Exception as caught:
            error = caught
    context.coordinate(error, "recoverable completed run publication")
    result = context.gather(report)[0]
    if not isinstance(result, dict):
        raise ValueError("Primary rank did not publish completion")
    return result


def train_production(  # noqa: PLR0915
    dataset_dir: Path,
    output_dir: Path,
    config: ProductionTrainConfig,
    *,
    authorization: UpdateAuthorization | None = None,
    resume: bool = False,
    ledger_path: Path | None = None,
    main_model_registry: Path | None = None,
) -> dict[str, Any]:
    require_parameter_updates(authorization, operation="train_production")
    require_parameter_updates(
        authorization, operation="train_production", phase=config.phase, recipe_sha256=config.recipe_sha256
    )
    if resume and (not output_dir.is_dir() or not (output_dir / "checkpoint.json").is_file()):
        raise ValueError("Exact resume requires an existing committed checkpoint pointer; never restart silently")
    torch.set_num_threads(config.cpu_threads)
    train = DecisionDataset(dataset_dir, "train")
    validation = DecisionDataset(dataset_dir, "validation")
    require_parameter_updates(
        authorization, operation="train_production", dataset_hash=train.manifest["manifest_sha256"]
    )
    _dataset_contract(train, validation, config)
    registration = None
    if config.phase == "M2":
        if main_model_registry is None:
            raise ValueError("M2 must use its unique registered main model, not an unregistered raw trainer")
        registration = json.loads(main_model_registry.read_text(encoding="utf-8"))
        ledger_path = validate_main_model_registration(
            registration,
            dataset_sha256=train.manifest["manifest_sha256"],
            config=config,
            output_dir=output_dir,
            ledger_path=ledger_path,
        )
    lengths = [int(history) + 65 for _, history in train.indices]
    updates = math.ceil(len(train) / config.effective_batch_size)
    validation_passes = 1 + len(validation_updates(updates)) * config.planned_epochs
    planned_work = len(train) * config.planned_epochs + len(validation) * validation_passes
    if config.budget.forecast.work_units < planned_work or config.budget.forecast.recipe_sha256 != config.recipe_sha256:
        raise ValueError("Cost forecast must cover this exact recipe, all training and periodic full validation")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    context = DistributedContext.open(
        config.device, backend=config.distributed_backend, gloo_bind_host=config.gloo_bind_host
    )
    ledger = None
    run_lease = None
    lease_entered = False
    error = None
    try:
        if config.device == "cuda" and config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Registered BF16 recipe is unsupported; do not silently change precision")
        torch.use_deterministic_algorithms(config.deterministic_algorithms)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
            torch.cuda.reset_peak_memory_stats(context.device)
        if context.primary:
            if output_dir.exists() and not resume:
                raise ValueError("Refuse to overwrite a training run")
            output_dir.mkdir(parents=True, exist_ok=resume)
            run_lease = exclusive_job(output_dir / "training.lock")
            run_lease.__enter__()
            lease_entered = True
            ledger = BudgetLedger(ledger_path or output_dir / "costs.jsonl", config.budget, allow_create=not resume)
            if not resume:
                ledger.assert_room(config.budget.forecast.seconds * config.budget.forecast.resource_usd_per_hour / 3600)
    except Exception as caught:
        error = caught
    try:
        context.coordinate(error, "production startup")
    except Exception:
        if run_lease is not None and lease_entered:
            run_lease.__exit__(None, None, None)
        context.close()
        raise
    attempt_id = 0

    def primary_log(name: str, record: dict[str, Any]) -> None:
        caught = None
        if context.primary:
            try:
                _append_record(output_dir / f"{name}-attempt-{attempt_id:04d}.jsonl", record)
            except Exception as issue:
                caught = issue
        context.coordinate(caught, "append-only run journal")

    try:
        initialized = None
        error = None
        try:
            initialized = _model_components(train, config, context, lengths, updates)
        except Exception as caught:
            error = caught
        context.coordinate(error, "model/optimizer initialization")
        assert initialized is not None
        raw, optimizer, scheduler, sampler, runtime, contract, state = initialized
        identities = context.gather(
            {
                "runtime": runtime,
                "source_sha256": contract["implementation"]["source_sha256"],
                "native_sha256": contract["implementation"]["native_extension_sha256"],
                "data_manifest_sha256": train.manifest["manifest_sha256"],
                "recipe_sha256": config.recipe_sha256,
            }
        )
        if any(value != identities[0] for value in identities):
            raise ValueError("Distributed workers differ in code,data,device or training recipe")
        contract = context.gather(contract)[0]
        if registration is not None:
            contract["main_model_registration_sha256"] = registration["registration_sha256"]
            contract["release_contract_sha256"] = content_hash(
                {key: value for key, value in contract.items() if key != "release_contract_sha256"}
            )
        checkpoint = output_dir / "checkpoint.pt"
        error = None
        try:
            if resume:
                payload = read_production_checkpoint(checkpoint, contract=contract, runtime=runtime)
                if payload["metadata"]["model_contract_hash"] != raw.config.contract_hash:
                    raise ValueError("Resume model config differs")
                raw.load_state_dict(payload["model"], strict=True)
                optimizer.load_state_dict(payload["optimizer"])
                scheduler.load_state_dict(payload["scheduler"])
                state = payload["trainer"]
                sampler.load_state_dict(state["sampler"])
                restore_rank_rng(payload["rank_rng"][context.rank], context.device)
                if context.primary and ledger is not None:
                    ledger.verify_state(state["budget_ledger"])
        except Exception as caught:
            error = caught
        context.coordinate(error, "exact checkpoint restore")
        if state.get("finished"):
            return _publish_completion(
                raw,
                state,
                config=config,
                context=context,
                contract=contract,
                ledger=ledger,
                checkpoint=checkpoint,
                dataset_sha256=train.manifest["manifest_sha256"],
                recovered=True,
            )
        if context.primary:
            attempt_id = len(tuple(output_dir.glob("attempt-*.json")))
        attempt_id = context.gather(attempt_id)[0]
        if context.primary:
            try:
                (output_dir / f"attempt-{attempt_id:04d}.json").write_text(
                    json.dumps(
                        {
                            "resume": resume,
                            "starting_committed_step": state["step"],
                            "recipe_sha256": config.recipe_sha256,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except Exception as caught:
                error = caught
        context.coordinate(error, "run attempt registration")
        context.synchronize_model(raw)
        model = (
            DistributedDataParallel(
                raw,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                broadcast_buffers=False,
                init_sync=False,
            )
            if context.world_size > 1
            else raw
        )

        def budget_boundary(reference: str, reserve_work: int = 0) -> None:
            caught = None
            if context.primary and ledger is not None:
                try:
                    ledger.settle_elapsed(reference)
                    if reserve_work:
                        ledger.reserve_next(
                            reserve_work
                            / config.budget.forecast.measured_units_per_second
                            * config.budget.forecast.safety_multiplier
                        )
                except Exception as issue:
                    caught = issue
            context.coordinate(caught, "phase budget")

        def save() -> None:
            rank_state = None
            caught = None
            try:
                rank_state = rank_rng_state(context.device)
            except Exception as issue:
                caught = issue
            context.coordinate(caught, "per-rank RNG capture")
            rng = context.gather(rank_state)
            caught = None
            if context.primary and ledger is not None:
                try:
                    state["sampler"] = sampler.state_dict()
                    state["budget_ledger"] = ledger.state_dict()
                    save_production_checkpoint(
                        checkpoint,
                        raw,
                        optimizer,
                        scheduler,
                        contract=contract,
                        trainer_state=state,
                        rank_rng=rng,
                        runtime=runtime,
                        step=state["step"],
                    )
                except Exception as issue:
                    caught = issue
            context.coordinate(caught, "checkpoint commit")

        def validation_pass(epoch: int, within: int) -> dict[str, Any]:
            budget_boundary(f"before_validation/{epoch}/{within}", len(validation))
            validation_started = time.perf_counter()
            report = evaluate_complete(
                raw,
                validation,
                context,
                microbatch=config.microbatch_size,
                use_rule_features=config.use_rule_features,
                precision=config.precision,
            )
            validation_seconds = max(context.gather(time.perf_counter() - validation_started))
            state["performance"]["validation_seconds"] += validation_seconds
            state["performance"]["validation_passes"] += 1
            record = {"epoch": epoch, "update_within_epoch": within, "step": state["step"], "report": report}
            state["validation_history"].append(record)
            primary_log("validation", record)
            budget_boundary(f"after_validation/{epoch}/{within}")
            return report

        if not state["validation_history"]:
            validation_pass(0, 0)
            save()
        completed_epochs = state.get("completed_epochs", 0)
        prior_third = state.get("third_epoch_decision")
        while sampler.epoch < config.planned_epochs and not (prior_third is not None and not prior_third["allowed"]):
            epoch = sampler.epoch
            state["epoch_order_sha256"][str(epoch)] = content_hash(sampler.order)
            while not sampler.exhausted:
                group = sampler.peek()
                budget_boundary(f"before_update/{state['step']}", group.global_examples)
                result = production_update(
                    model, raw, optimizer, scheduler, train, group, context, config, authorization
                )
                sampler.commit(group)
                state["step"] += 1
                state["processed_examples"] += group.global_examples
                performance = state["performance"]
                performance["updates"] += 1
                performance["examples"] += group.global_examples
                for key in ("real_tokens", "padded_real_tokens", "data_wait_seconds_sum"):
                    performance[key] += result[key]
                performance["training_seconds"] += result["step_elapsed_seconds"]
                record = {"step": state["step"], "epoch": epoch, "cursor": sampler.cursor, **result}
                primary_log("updates", record)
                budget_boundary(f"update/{state['step']}")
                within = group.update_index + 1
                if within in validation_updates(updates):
                    report = validation_pass(epoch + 1, within)
                    if sampler.exhausted:
                        state["epoch_end_validation"][str(epoch + 1)] = report
                if state["step"] % config.checkpoint_updates == 0 or within in validation_updates(updates):
                    save()
            completed_epochs = epoch + 1
            state["completed_epochs"] = completed_epochs
            if completed_epochs == 2 and config.planned_epochs == 3:
                room = True
                if context.primary and ledger is not None:
                    try:
                        ledger.reserve_next(
                            (len(train) + len(validation) * len(validation_updates(updates)))
                            / config.budget.forecast.measured_units_per_second
                            * config.budget.forecast.safety_multiplier
                        )
                    except Exception:
                        room = False
                room = all(context.gather(room))
                allowed, reason = third_epoch_allowed(
                    state["epoch_end_validation"]["1"],
                    state["epoch_end_validation"]["2"],
                    config.third_epoch_nll_tolerance,
                    budget_has_room=room,
                )
                state["third_epoch_decision"] = {"allowed": allowed, "reason": reason}
                if not allowed:
                    save()
                    break
            sampler.next_epoch()
            save()
        state["finished"] = True
        state["final_weights_sha256"] = model_weights_sha256(raw)
        state["execution_telemetry"] = {
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(context.device)
            if context.device.type == "cuda"
            else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(context.device)
            if context.device.type == "cuda"
            else None,
        }
        budget_boundary("finalize")
        save()
        budget_boundary("final_checkpoint_cost")
        return _publish_completion(
            raw,
            state,
            config=config,
            context=context,
            contract=contract,
            ledger=ledger,
            checkpoint=checkpoint,
            dataset_sha256=train.manifest["manifest_sha256"],
            recovered=False,
        )
    except BaseException as failure:
        if context.primary and ledger is not None:
            try:
                ledger.settle_elapsed("aborted_production_run")
            except Exception:
                pass
            try:
                _append_record(
                    output_dir / "aborted.jsonl",
                    {
                        "status": "BLOCKED",
                        "reason": f"{type(failure).__name__}: {failure}",
                        "budget": ledger.state_dict(),
                    },
                )
            except OSError:
                pass
        raise
    finally:
        if run_lease is not None and lease_entered:
            run_lease.__exit__(None, None, None)
        context.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorization-gated full-epoch KAR trainer; compatible with torchrun")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--main-model-registry", type=Path)
    args = parser.parse_args()
    authorization = load_update_authorization(args.authorization)
    require_parameter_updates(authorization, operation="train_production")
    config = ProductionTrainConfig.from_record(json.loads(args.config.read_text(encoding="utf-8")))
    train_production(
        args.dataset,
        args.output,
        config,
        authorization=authorization,
        resume=args.resume,
        ledger_path=args.ledger,
        main_model_registry=args.main_model_registry,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
