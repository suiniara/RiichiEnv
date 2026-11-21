from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ..authorization import UpdateAuthorization, load_update_authorization, require_parameter_updates
from ..contracts import content_hash
from .batch import feature_mask_draw, rule_feature_view
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig
from .dataset import DecisionDataset, TrainingBatch, collate_examples
from .losses import loss_sums
from .metrics import MetricTotals
from .model import KARTransformer


@dataclass(frozen=True)
class TrainConfig:
    preset: str = "small"
    seed: int = 17
    epochs: int = 1
    max_updates: int = 16
    effective_batch_size: int = 8
    microbatch_size: int = 1
    learning_rate: float = 3e-4
    use_rule_features: bool = True
    feature_mask_probability: float = 0.25
    threads: int = 2
    checkpoint_interval: int = 4
    validation_examples: int = 64
    overfit_examples: int = 0
    diagnostic_cpu_smoke: bool = True

    def __post_init__(self) -> None:
        if self.epochs != 1 or not 1 <= self.max_updates <= 100 or not 1 <= self.threads <= 2:
            raise ValueError("This CPU launcher is bounded to one epoch, 1..100 updates and at most two threads")
        if (
            self.effective_batch_size <= 0
            or self.microbatch_size <= 0
            or self.microbatch_size > self.effective_batch_size
        ):
            raise ValueError("Invalid effective/microbatch sizes")
        if not 0 <= self.feature_mask_probability <= 1 or self.learning_rate <= 0 or self.overfit_examples < 0:
            raise ValueError("Invalid augmentation/learning-rate/overfit configuration")
        if self.checkpoint_interval < 1 or self.validation_examples < 1 or not self.diagnostic_cpu_smoke:
            raise ValueError("A full GPU/epoch probe needs separate resource/budget authorization")


def parameter_groups(model: nn.Module) -> list[dict[str, Any]]:
    groups: dict[float, list[nn.Parameter]] = {0.05: [], 0.01: [], 0.0: []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            decay = 0.05 if name == "value_head.weight" else 0.01 if parameter.ndim >= 2 else 0.0
            groups[decay].append(parameter)
    return [{"params": parameters, "weight_decay": decay} for decay, parameters in groups.items()]


def create_optimizer(model: nn.Module, learning_rate: float = 3e-4) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameter_groups(model), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)


def create_scheduler(optimizer: torch.optim.Optimizer, total_updates: int) -> torch.optim.lr_scheduler.LambdaLR:
    if total_updates < 1:
        raise ValueError("Positive update count required")
    # Warm up proportionally to the actual schedule. A fixed 100-update
    # minimum leaves short M1/CPU diagnostic runs entirely inside warmup.
    warmup = max(1, math.ceil(0.03 * total_updates))

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_updates - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def optimizer_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: list[TrainingBatch],
    *,
    global_examples: int,
    rho_e: float,
    rho_v: float,
    world_size: int = 1,
    scheduler: Any = None,
    authorization: UpdateAuthorization | None = None,
) -> dict[str, Any]:
    require_parameter_updates(authorization, operation="optimizer_update")
    if not microbatches:
        raise ValueError("Use a graph-connected dummy batch rather than skipping a DDP rank")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = MetricTotals()
    objective = 0.0
    for index, batch in enumerate(microbatches):
        synchronization = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index < len(microbatches) - 1
            else nullcontext()
        )
        with synchronization:
            outputs = model(batch.inputs)
            sums = loss_sums(outputs, batch.labels, batch.inputs.legal_mask)
            loss = sums.objective(global_examples=global_examples, world_size=world_size, rho_e=rho_e, rho_v=rho_v)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite objective; update aborted")
            loss.backward()
        objective += float(loss.detach())
        metrics.add(outputs, batch)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite or missing gradients; optimizer/scheduler did not advance")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {"objective": objective, "gradient_norm_before_clip": float(norm), **metrics.report()}


def _epoch_order(dataset: DecisionDataset, seed: int, epoch: int, overfit_examples: int) -> list[int]:
    count = min(len(dataset), overfit_examples) if overfit_examples else len(dataset)
    generator = torch.Generator().manual_seed(seed + 1000003 * epoch)
    order = torch.randperm(count, generator=generator).tolist()
    # Random windows retain the natural sample distribution; sort only to reduce padding.
    for start in range(0, len(order), 128):
        order[start : start + 128] = sorted(order[start : start + 128], key=lambda index: dataset.indices[index][1])
    return order


def _training_view(batch: TrainingBatch, config: TrainConfig, epoch: int) -> TrainingBatch:
    draws = torch.tensor(
        [
            feature_mask_draw(config.seed, epoch, game, row, probability=config.feature_mask_probability)
            for game, row in zip(batch.game_ids, batch.inputs.context_ids)
        ],
        dtype=torch.bool,
    )
    inputs = rule_feature_view(batch.inputs, draws, enabled=config.use_rule_features)
    return TrainingBatch(inputs, batch.labels, batch.game_ids)


@torch.no_grad()
def evaluate(model: KARTransformer, dataset: DecisionDataset, config: TrainConfig) -> dict[str, Any]:
    model.eval()
    metrics = MetricTotals()
    count = min(len(dataset), config.validation_examples)
    for start in range(0, count, config.microbatch_size):
        batch = collate_examples([dataset[index] for index in range(start, min(count, start + config.microbatch_size))])
        # Never stochastic masking in validation, even if the training mode is feature-on.
        inputs = rule_feature_view(
            batch.inputs, torch.zeros(batch.inputs.size, dtype=torch.bool), enabled=config.use_rule_features
        )
        viewed = TrainingBatch(inputs, batch.labels, batch.game_ids)
        metrics.add(model(inputs), viewed)
    return {
        **metrics.report(),
        "available_examples": len(dataset),
        "evaluated_examples": count,
        "full_validation_split_evaluated": count == len(dataset),
        "stochastic_feature_mask_probability": 0.0,
    }


def train_cpu(  # noqa: PLR0915
    dataset_dir: Path,
    output_dir: Path,
    config: TrainConfig | None = None,
    *,
    resume: bool = False,
    authorization: UpdateAuthorization | None = None,
) -> dict[str, Any]:
    require_parameter_updates(authorization, operation="train_cpu")
    config = config or TrainConfig()
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train = DecisionDataset(dataset_dir, "train", include_candidate_branches=True)
    validation = DecisionDataset(dataset_dir, "validation", include_candidate_branches=True)
    if not len(train) or not len(validation):
        raise ValueError("CPU probe needs separate nonempty train and validation games")
    require_parameter_updates(authorization, operation="train_cpu", dataset_hash=train.manifest["manifest_sha256"])
    rho_e, rho_v = train.manifest["rho_E"], train.manifest["rho_V"]
    model = KARTransformer(ModelConfig.from_preset(config.preset)).cpu()
    initial_digest = hashlib.sha256()
    for parameter in model.parameters():
        initial_digest.update(parameter.detach().numpy().tobytes())
    optimizer = create_optimizer(model, config.learning_rate)
    scheduler = create_scheduler(optimizer, math.ceil(len(train) / config.effective_batch_size) * config.epochs)
    run_contract = {
        "dataset_manifest_sha256": train.manifest["manifest_sha256"],
        "initial_weights_sha256": initial_digest.hexdigest(),
        "config": asdict(config),
        "rho_E": rho_e,
        "rho_V": rho_v,
        "lambda_E": 0.25,
        "lambda_V": 0.1,
        "runtime_device": "cpu",
        "budget_cap_usd": 100.0,
        "budget_spent_usd": 0.0,
        "paid_resource_authorized": False,
        "calibration_status": "not_fitted",
        "decision_certificate": "certificate_unavailable",
        "paired_seed_manifest": None,
        "available_M0_N_train_half_games": train.manifest["scope"]["available_split_counts"]["train"],
        "probe_N_train_half_games": train.manifest["N_train_half_games"],
        "weight_decay_groups": {"value_head.weight": 0.05, "other_matrices": 0.01, "bias_norm": 0.0},
    }
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(dataset_dir.resolve()):
        raise ValueError("Training output must stay outside immutable dataset directory")
    checkpoint = output_dir / "checkpoint.pt"
    step, epoch, cursor, history = 0, 0, 0, []
    processed_row_ids: list[str] = []
    order = _epoch_order(train, config.seed, epoch, config.overfit_examples)
    if resume:
        restored = load_checkpoint(checkpoint, model, optimizer, scheduler, run_contract=run_contract)
        step, state = restored["step"], restored["sampler"]
        epoch, cursor, order, history = state["epoch"], state["cursor"], state["order"].tolist(), state["history"]
        processed_row_ids = state["processed_row_ids"]
        log_path = output_dir / "updates.jsonl"
        if log_path.is_file():
            committed = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
            if committed[:step] != history:
                raise ValueError("Committed optimizer log differs from the exact checkpoint")
        temporary_log = output_dir / "updates.jsonl.tmp"
        temporary_log.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history), encoding="utf-8")
        temporary_log.replace(log_path)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    start_step = step
    before = state.get("baseline_validation") if resume else None
    if before is None:
        before = evaluate(model, validation, config)
    start_time, tokens_processed, decisions_processed = time.perf_counter(), 0, 0
    for _ in range(step, config.max_updates):
        if cursor >= len(order):
            if not config.overfit_examples:
                break
            epoch, cursor = epoch + 1, 0
            order = _epoch_order(train, config.seed, epoch, config.overfit_examples)
        group = order[cursor : cursor + config.effective_batch_size]
        processed_row_ids.extend(train.indices[index][0] for index in group)
        microbatches = []
        for start in range(0, len(group), config.microbatch_size):
            batch = collate_examples([train[index] for index in group[start : start + config.microbatch_size]])
            microbatches.append(_training_view(batch, config, epoch))
            tokens_processed += int(batch.inputs.fields["valid"].sum())
        result = optimizer_update(
            model,
            optimizer,
            microbatches,
            global_examples=len(group),
            rho_e=rho_e,
            rho_v=rho_v,
            scheduler=scheduler,
            authorization=authorization,
        )
        cursor += len(group)
        decisions_processed += len(group)
        step += 1
        record = {
            "step": step,
            "epoch": epoch,
            "cursor": cursor,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **result,
        }
        history.append(record)
        with (output_dir / "updates.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, allow_nan=False) + "\n")
        if step % config.checkpoint_interval == 0 or step == config.max_updates or cursor == len(order):
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                run_contract=run_contract,
                sampler_state={
                    "epoch": epoch,
                    "cursor": cursor,
                    "order": torch.tensor(order),
                    "history": history,
                    "baseline_validation": before,
                    "processed_row_ids": processed_row_ids,
                },
                step=step,
            )
    elapsed = time.perf_counter() - start_time
    after = evaluate(model, validation, config)
    report = {
        "status": "PASS",
        "scope": "bounded_cpu_training_pipeline_not_complete_M1_epoch_or_strength_evaluation",
        "model_config": asdict(model.config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "run_contract": run_contract,
        "completed_updates": step,
        "updates_this_invocation": step - start_step,
        "train_history": history,
        "sample_order_sha256": content_hash(processed_row_ids),
        "sample_presentations": len(processed_row_ids),
        "validation_before": before,
        "validation_after": after,
        "elapsed_training_seconds": elapsed,
        "training_tokens_per_second": tokens_processed / elapsed if elapsed else None,
        "training_decisions_per_second": decisions_processed / elapsed if elapsed else None,
        "full_epoch_completed": cursor == len(order) and not config.overfit_examples,
        "gpu_memory_measured": False,
        "gpu_resources_started": False,
        "cloud_spent_usd": 0.0,
        "N_train_half_games": train.manifest["N_train_half_games"],
        "cpu_threads": config.threads,
        "unsupported_value_branches_promoted": False,
        "checkpoint": str(checkpoint.with_suffix(".json")),
        "test_split_used_for_tuning": False,
        "calibration_fitted": False,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded local-CPU KAR M1 training smoke; never launches GPU resources"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("tiny", "small"), default="small")
    parser.add_argument("--max-updates", type=int, default=16)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--feature-off", action="store_true")
    parser.add_argument("--overfit-examples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--update-authorization", type=Path)
    args = parser.parse_args()
    config = TrainConfig(
        preset=args.preset,
        max_updates=args.max_updates,
        effective_batch_size=args.effective_batch,
        microbatch_size=args.microbatch,
        use_rule_features=not args.feature_off,
        overfit_examples=args.overfit_examples,
    )
    report = train_cpu(
        args.dataset,
        args.output,
        config,
        resume=args.resume,
        authorization=load_update_authorization(args.update_authorization),
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "train_history"}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
