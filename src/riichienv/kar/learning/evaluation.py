from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from ..calibration import CalibrationArtifact
from ..calibration_roles import CALIBRATION_SPLIT_SALT, calibration_is_fit
from ..policy import support_discards
from ..statistics import FullDiagnostics
from ..support import EventSupport
from .batch import rule_feature_view
from .dataset import DecisionDataset, TrainingBatch, collate_examples
from .distributed import DistributedContext
from .losses import Labels, loss_sums
from .model import KARTransformer, ModelOutputs


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
        if support is not None and active["response"][index]:
            metrics.add_unsupported(
                sum(value for value, covered in zip(probabilities["response"][index], support.supported) if not covered)
            )
        target = targets["action"][index]
        if (
            calibration is not None
            and active["action"][index]
            and target < 148
            and sum(action < 148 for action in ids) >= 2
        ):
            selected = support_discards(ids, logs["action"][index], calibration.log_threshold)
            metrics.add_gate(
                retained=target in selected,
                support_size=len(selected),
                mass=sum(probabilities["action"][index][action] for action in selected),
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
    before = model.training
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
    metrics = FullDiagnostics()
    selection: dict[str, Any] = {
        key: 0 for key in ("examples", "action", "response", "rank", "action_count", "response_count", "rank_count")
    }
    selection_games: set[str] = set()
    count = 0
    error = None
    model.eval()
    try:
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
            autocast = (
                torch.autocast(device_type=context.device.type, dtype=torch.bfloat16)
                if precision == "bf16"
                else nullcontext()
            )
            with autocast:
                outputs = model(inputs)
            add_diagnostics(
                metrics, outputs, viewed, calibration=calibration, support=support, public_fields=batch.inputs.fields
            )
            partial = cal_fit_loss_summary(outputs, viewed)
            for key in selection:
                selection[key] += partial[key]
            selection_games.update(partial["games"])
            count += batch.inputs.size
    except Exception as caught:
        error = caught
    finally:
        model.train(before)
    context.coordinate(error, "complete validation forward")
    gathered = context.gather(
        {
            "examples": count,
            "metrics": metrics.state_dict(),
            "selection": selection,
            "selection_games": sorted(selection_games),
        }
    )
    merged = FullDiagnostics()
    total = 0
    selected_sums: dict[str, Any] = {key: 0 for key in selection}
    selected_games: set[str] = set()
    for state in gathered:
        merged.merge(state["metrics"])
        total += state["examples"]
        for key in selected_sums:
            selected_sums[key] += state["selection"][key]
        selected_games.update(state["selection_games"])
    if total != len(dataset):
        raise ValueError("Distributed validation omitted or duplicated decision rows")
    details = merged.report(bootstrap_replicates=bootstrap_replicates)
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
        "calibration_status": "heldout_artifact" if calibration is not None else "unfitted",
    }
