from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import torch

from ..calibration import CalibrationArtifact
from ..calibration_roles import CALIBRATION_SPLIT_SALT, calibration_is_fit
from ..policy import support_discards
from ..probability import legal_prior, masked_log_softmax
from ..statistics import FullDiagnostics
from ..support import EventSupport
from .batch import rule_feature_view
from .dataset import DecisionDataset, TrainingBatch, collate_examples
from .distributed import DistributedContext
from .losses import Labels, loss_sums
from .model import KARTransformer, ModelOutputs


def _add_response_diagnostics(
    metrics: FullDiagnostics,
    *,
    game: str,
    logits: list[float],
    allowed: list[bool],
    target: int,
    temperature: float,
    train_supported: tuple[bool, ...] | None,
    value_supported: tuple[bool, ...] | None,
) -> None:
    event_logs = masked_log_softmax(logits, allowed, temperature)
    metrics.add_response_event_calibration(game=game, logs=event_logs, allowed=allowed, target=target)
    if train_supported is not None:
        metrics.add_response_mass(
            tuple(math.exp(value) for value in event_logs),
            train_supported=train_supported, value_supported=value_supported,
        )


@torch.no_grad()
def add_diagnostics(
    metrics: FullDiagnostics,
    outputs: ModelOutputs,
    batch: TrainingBatch,
    *,
    calibration: CalibrationArtifact | None = None,
    support: EventSupport | None = None,
    public_fields: dict[str, torch.Tensor] | None = None,
) -> None:
    labels = batch.labels
    loss_sums(outputs, labels, batch.inputs.legal_mask)
    if calibration is not None and (support is None or calibration.support_sha256 != support.digest):
        raise ValueError("Calibrated diagnostics require the bound support artifact")
    train_supported = support.supported if support is not None else None
    value_supported = (support.value_supported(calibration.value_branch_statuses)
                       if support is not None and calibration is not None else None)
    temperatures = calibration.temperatures if calibration is not None else (1.0, 1.0, 1.0)
    action_log = (
        (outputs.policy_logits.float() / temperatures[0])
        .masked_fill(~batch.inputs.legal_mask, -torch.inf)
        .log_softmax(-1)
    )
    response_allowed = torch.where(labels.response_valid[:, None], labels.response_allowed, True)
    response_log = (
        (outputs.response_logits.float() / temperatures[1]).masked_fill(~response_allowed, -torch.inf).log_softmax(-1)
    )
    safe_event = torch.where(labels.value_valid, labels.response_target, 0)
    rank_log = (
        outputs.rank_logits.float().gather(1, safe_event[:, None, None].expand(-1, 1, 4)).squeeze(1) / temperatures[2]
    ).log_softmax(-1)
    fields = batch.inputs.fields if public_fields is None else public_fields
    opponents = (fields["kind"] == 3) & (fields["actor"] > 1)
    riichi = (opponents & (fields["flags"][..., 1] == 2)).any(-1).cpu().tolist()
    called = ((fields["kind"] == 5) & (fields["actor"] > 1) & (fields["flags"][..., 0] == 2)).any(-1).cpu().tolist()
    candidates = batch.inputs.candidates.cpu().tolist()
    probabilities = {
        head: value.exp().cpu().tolist()
        for head, value in (("action", action_log), ("response", response_log), ("rank", rank_log))
    }
    logs = {
        head: value.cpu().tolist()
        for head, value in (("action", action_log), ("response", response_log), ("rank", rank_log))
    }
    active = {
        "action": labels.action_valid.cpu().tolist(),
        "response": labels.response_valid.cpu().tolist(),
        "rank": labels.value_valid.cpu().tolist(),
    }
    targets = {
        "action": labels.action_target.cpu().tolist(),
        "response": labels.response_target.cpu().tolist(),
        "rank": labels.final_rank_target.cpu().tolist(),
    }
    # Match runtime's scalar normalization for gates, event calibration and mass
    # histograms. Batched FP32 softmax can change threshold/bin membership.
    policy_logits, response_logits = (
        outputs.policy_logits.float().cpu().tolist() if calibration is not None else [],
        outputs.response_logits.float().cpu().tolist(),
    )
    response_masks = labels.response_allowed.cpu().tolist()
    for index, game in enumerate(batch.game_ids):
        ids = tuple(action for action in candidates[index] if action != 546)
        slices = []
        rank_values = fields["stage"][index][fields["kind"][index] == 2]
        rank = int(rank_values.item()) if rank_values.numel() == 1 else 0
        slices.append("current_rank/" + (str(rank) if 1 <= rank <= 4 else "unknown"))
        if len(ids) >= 2:
            slices.append("multiple_legal_actions")
        if sum(action < 148 for action in ids) >= 2:
            slices.append("multiple_complete_discards")
        if riichi[index]:
            slices.append("riichi_threat")
        if called[index]:
            slices.append("open_meld_threat")
        if riichi[index] and called[index]:
            slices.append("mixed_riichi_and_open_meld_threat")
        if not riichi[index] and not called[index]:
            slices.append("no_visible_threat")
        for head in ("action", "response", "rank"):
            if active[head][index]:
                target = targets[head][index]
                metrics.add(
                    game=game,
                    head=head,
                    probabilities=probabilities[head][index],
                    target=target,
                    log_probability=logs[head][index][target],
                    slices=tuple(slices),
                    event=targets["response"][index] if head == "rank" else None,
                )
        if active["response"][index]:
            _add_response_diagnostics(
                metrics, game=game, logits=response_logits[index], allowed=response_masks[index],
                target=targets["response"][index], temperature=temperatures[1],
                train_supported=train_supported, value_supported=value_supported,
            )
        target = targets["action"][index]
        if (
            calibration is not None
            and active["action"][index]
            and target < 148
            and sum(action < 148 for action in ids) >= 2
        ):
            # Gate membership and mass must use calibration/runtime's exact
            # normalization; retain FP32 above for the existing loss diagnostics.
            prior, gate_logs = legal_prior(policy_logits[index], ids, temperatures[0])
            selected = support_discards(ids, gate_logs, calibration.log_threshold)
            metrics.add_gate(
                retained=target in selected,
                support_size=len(selected),
                mass=math.fsum(prior[action] for action in selected),
                slices=tuple(slices),
            )


@torch.no_grad()
def cal_fit_loss_summary(outputs: ModelOutputs, batch: TrainingBatch) -> dict[str, Any]:
    membership = tuple(calibration_is_fit(game) for game in batch.game_ids)
    selected = torch.tensor(membership, dtype=torch.bool, device=batch.labels.action_valid.device)
    labels = batch.labels
    selected_labels = Labels(
        labels.action_target,
        labels.action_valid & selected,
        labels.response_target,
        labels.response_applicable,
        labels.response_observed,
        labels.final_rank_target,
        labels.value_valid & selected,
        labels.response_allowed,
    )
    sums = loss_sums(outputs, selected_labels, batch.inputs.legal_mask)
    return {
        "examples": sum(membership),
        "action": float(sums.action),
        "response": float(sums.response),
        "rank": float(sums.value),
        "action_count": sums.action_count,
        "response_count": sums.response_count,
        "rank_count": sums.value_count,
        "games": sorted({game for game, fit in zip(batch.game_ids, membership) if fit}),
    }


@torch.no_grad()
def evaluate_complete(  # noqa: PLR0915
    model: KARTransformer,
    dataset: DecisionDataset,
    context: DistributedContext,
    *,
    microbatch: int,
    use_rule_features: bool,
    precision: str = "fp32",
    calibration: CalibrationArtifact | None = None,
    support: EventSupport | None = None,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    if dataset.split != "validation":
        raise ValueError("Training selection/epoch gates may inspect validation only, never test")
    if not len(dataset):
        raise ValueError("A whole nonempty validation partition is required")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("Validation precision must be fp32 or bf16")
    if calibration is not None and precision != "fp32":
        raise ValueError("Calibrated validation requires frozen FP32 inference, not BF16")
    before = model.training
    metrics = FullDiagnostics()
    selection: dict[str, Any] = {
        key: 0 for key in ("examples", "action", "response", "rank", "action_count", "response_count", "rank_count")
    }
    selection_games: set[str] = set()
    count = 0
    error = None
    payload = None
    try:
        model.eval()
        if calibration is not None:
            if support is None:
                raise ValueError("Calibrated diagnostics require the bound support artifact")
            from .inference import FrozenModelPredictor  # noqa: PLC0415

            binding = FrozenModelPredictor(
                model, dataset.manifest["manifest_sha256"], use_rule_features=use_rule_features
            ).binding
            calibration.validate(binding, support)
            if calibration.inference_device_type != context.device.type:
                raise ValueError("Calibrated evaluation requires its original inference backend")
        forward_started = perf_counter()
        indices = list(range(context.rank, len(dataset), context.world_size))
        for start in range(0, len(indices), microbatch):
            batch = collate_examples([dataset[index] for index in indices[start : start + microbatch]]).to(
                context.device
            )
            inputs = rule_feature_view(
                batch.inputs,
                torch.zeros(batch.inputs.size, dtype=torch.bool, device=context.device),
                enabled=use_rule_features,
            )
            viewed = TrainingBatch(inputs, batch.labels, batch.game_ids)
            with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                outputs = model(inputs)
            add_diagnostics(
                metrics, outputs, viewed, calibration=calibration, support=support, public_fields=batch.inputs.fields
            )
            partial = cal_fit_loss_summary(outputs, viewed)
            for key in selection:
                selection[key] += partial[key]
            selection_games.update(partial["games"])
            count += batch.inputs.size
        forward_seconds = perf_counter() - forward_started
        aggregation_started = perf_counter()
        payload = {
            "examples": count,
            "metrics": metrics.state_dict(),
            "selection": selection,
            "selection_games": sorted(selection_games),
            "forward_seconds": forward_seconds,
        }
    except Exception as caught:
        error = caught
    finally:
        model.train(before)
    context.coordinate(error, "complete validation forward")
    assert payload is not None
    gathered = context.gather_to_primary(payload)
    report, error = None, None
    if context.primary:
        try:
            assert gathered is not None
            report = _global_validation_report(dataset, gathered, aggregation_started,
                                               bootstrap_replicates, calibrated=calibration is not None)
        except Exception as caught:
            error = caught
    # A primary-only merge/report failure must reach all ranks before any
    # worker attempts to receive a report that will never be broadcast.
    context.coordinate(error, "complete validation global report")
    return context.broadcast_from_primary(report)


def _global_validation_report(
    dataset: DecisionDataset, gathered: list[dict[str, Any]], aggregation_started: float,
    bootstrap_replicates: int, *, calibrated: bool,
) -> dict[str, Any]:
    merged = FullDiagnostics()
    total = 0
    selected_sums = dict.fromkeys(gathered[0]["selection"], 0)
    selected_games: set[str] = set()
    for state in gathered:
        merged.merge(state["metrics"])
        total += state["examples"]
        for key in selected_sums:
            selected_sums[key] += state["selection"][key]
        selected_games.update(state["selection_games"])
    if total != len(dataset):
        raise ValueError("Distributed validation omitted or duplicated decision rows")
    timings = {
        # Max rank wall-clock for the local eval loop (including its CPU diagnostics).
        "forward": max(state["forward_seconds"] for state in gathered),
        "aggregation": perf_counter() - aggregation_started,
    }
    details = merged.report(bootstrap_replicates=bootstrap_replicates, timings=timings)
    heads = merged.heads
    objective = (
        heads["action"].nll
        + 0.25 * heads["response"].nll / dataset.manifest["rho_E"]
        + 0.1 * heads["rank"].nll / dataset.manifest["rho_V"]
    ) / total
    return {
        "partition": "validation",
        "available_examples": len(dataset),
        "evaluated_examples": total,
        "full_validation_split_evaluated": True,
        "stochastic_feature_mask_probability": 0.0,
        "validation_objective": objective,
        **{f"{head}_nll": value.nll / value.count if value.count else None for head, value in heads.items()},
        "model_selection": {
            "partition": "cal_fit",
            "calibration_split_salt": CALIBRATION_SPLIT_SALT,
            "full_cal_fit_evaluated": True,
            "cal_check_used": False,
            "evaluated_examples": selected_sums["examples"],
            "game_ids": sorted(selected_games),
            "validation_objective": (
                selected_sums["action"]
                + 0.25 * selected_sums["response"] / dataset.manifest["rho_E"]
                + 0.1 * selected_sums["rank"] / dataset.manifest["rho_V"]
            )
            / selected_sums["examples"]
            if selected_sums["examples"]
            else None,
            **{
                head + "_nll": selected_sums[head] / selected_sums[head + "_count"]
                if selected_sums[head + "_count"]
                else None
                for head in ("action", "response", "rank")
            },
        },
        "diagnostics": details,
        "timings_seconds": timings,
        "calibration_status": "heldout_artifact" if calibrated else "unfitted",
    }
