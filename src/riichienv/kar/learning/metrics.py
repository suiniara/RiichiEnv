from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .dataset import TrainingBatch
from .losses import loss_sums
from .model import ModelOutputs


@dataclass
class MetricTotals:
    examples: int = 0
    counts: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in ("action", "response", "rank")})
    nll: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in ("action", "response", "rank")})
    brier: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in ("action", "response", "rank")})
    threat: dict[str, MetricTotals] = field(default_factory=dict)

    @torch.no_grad()
    def add(self, outputs: ModelOutputs, batch: TrainingBatch, *, slices: bool = True) -> None:
        labels = batch.labels
        sums = loss_sums(outputs, labels, batch.inputs.legal_mask)
        self.examples += int(batch.source_count or 0)
        for key, value, count in (
            ("action", sums.action, sums.action_count),
            ("response", sums.response, sums.response_count),
            ("rank", sums.value, sums.value_count),
        ):
            self.nll[key] += float(value)
            self.counts[key] += count
        action = outputs.policy_logits.float().masked_fill(~batch.inputs.legal_mask, -torch.inf).softmax(-1)
        valid = labels.response_valid
        allowed = torch.where(valid[:, None], labels.response_allowed, True)
        response = outputs.response_logits.float().masked_fill(~allowed, -torch.inf).softmax(-1)
        event = torch.where(labels.value_valid, labels.response_target, 0)
        rank = outputs.rank_logits.float().gather(1, event[:, None, None].expand(-1, 1, 4)).squeeze(1).softmax(-1)
        for key, probabilities, target, active, weight in (
            ("action", action, labels.action_target, labels.action_valid, None),
            ("response", response, labels.response_target, valid, labels.response_weight),
            ("rank", rank, labels.final_rank_target, labels.value_valid, labels.value_weight),
        ):
            safe = torch.where(active, target, 0)
            errors = (probabilities - F.one_hot(safe, probabilities.shape[-1])).square().sum(-1)
            weighted = errors if weight is None else errors * weight
            self.brier[key] += float(torch.where(active, weighted, 0.0).sum())
        if slices:
            fields = batch.inputs.fields
            opponents = (fields["kind"] == 3) & (fields["actor"] > 1)
            accepted = (opponents & (fields["flags"][..., 1] == 2)).sum(-1)
            for name, mask in (
                ("no_riichi_opponent", accepted == 0),
                ("one_riichi_opponent", accepted == 1),
                ("two_plus_riichi_opponents", accepted >= 2),
            ):
                if mask.any():
                    subset = _subset(batch, outputs, mask)
                    self.threat.setdefault(name, MetricTotals()).add(subset[1], subset[0], slices=False)

    def report(self) -> dict[str, Any]:
        result: dict[str, Any] = {"examples": self.examples}
        for key in self.counts:
            count = self.counts[key]
            result[f"{key}_valid_count"] = count
            result[f"{key}_nll"] = self.nll[key] / count if count else None
            result[f"{key}_brier"] = self.brier[key] / count if count else None
        result["by_threat"] = {key: total.report() for key, total in self.threat.items()}
        return result


def _subset(batch: TrainingBatch, outputs: ModelOutputs, mask: Tensor) -> tuple[TrainingBatch, ModelOutputs]:
    from .batch import ModelBatch  # noqa: PLC0415
    from .losses import Labels  # noqa: PLC0415

    indices = mask.nonzero().flatten().tolist()
    inputs = ModelBatch(
        {key: value[mask] for key, value in batch.inputs.fields.items()},
        batch.inputs.candidates[mask],
        batch.inputs.decide_index[mask],
        batch.inputs.action_index[mask],
        tuple(batch.inputs.context_ids[index] for index in indices),
    )
    labels = Labels(**{key: value[mask] for key, value in batch.labels.__dict__.items()})
    selected = ModelOutputs(
        outputs.policy_logits[mask],
        outputs.response_logits[mask],
        outputs.rank_logits[mask],
        outputs.action_present[mask],
    )
    return TrainingBatch(inputs, labels, tuple(batch.game_ids[index] for index in indices)), selected
