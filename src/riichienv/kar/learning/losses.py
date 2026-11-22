from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import ModelOutputs


@dataclass(frozen=True)
class Labels:
    action_target: Tensor
    action_valid: Tensor
    response_target: Tensor
    response_applicable: Tensor
    response_observed: Tensor
    final_rank_target: Tensor
    value_valid: Tensor
    response_allowed: Tensor
    response_weight: Tensor | None = None
    value_weight: Tensor | None = None

    def __post_init__(self) -> None:
        if self.response_weight is None:
            object.__setattr__(self, "response_weight", torch.ones_like(self.action_target, dtype=torch.float32))
        if self.value_weight is None:
            object.__setattr__(self, "value_weight", torch.ones_like(self.action_target, dtype=torch.float32))

    @property
    def response_valid(self) -> Tensor:
        # Candidate branches have auxiliary supervision without an ACTION-policy label.
        return self.response_applicable & self.response_observed

    def to(self, device: torch.device | str) -> Labels:
        return Labels(
            **{
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in self.__dict__.items()
            }
        )

    def validate(self, batch: int) -> None:
        for key, value in self.__dict__.items():
            if key == "response_allowed":
                if value.shape != (batch, 21) or value.dtype != torch.bool:
                    raise ValueError("Response rules require a [B,21] boolean mask")
            elif value.shape != (batch,):
                raise ValueError(f"{key} must have one entry per sample")
            elif key in {"action_target", "response_target", "final_rank_target"}:
                if value.dtype != torch.int64:
                    raise ValueError("All target indices must be int64")
            elif key in {"response_weight", "value_weight"}:
                if value.shape != (batch,) or not value.is_floating_point():
                    raise ValueError("Auxiliary loss weights must be floating [B] vectors")
                if not torch.isfinite(value).all() or torch.any(value < 0):
                    raise ValueError("Auxiliary loss weights must be finite and nonnegative")
            elif value.dtype != torch.bool:
                raise ValueError("Validity/applicability flags must be boolean")
        if torch.any(self.value_valid & ~self.response_valid):
            raise ValueError("value_valid implies an observed applicable response")
        for value, valid, limit, name in (
            (self.action_target, self.action_valid, 546, "action"),
            (self.response_target, self.response_valid, 21, "response"),
            (self.final_rank_target, self.value_valid, 4, "rank"),
        ):
            if torch.any(valid & ((value < 0) | (value >= limit))):
                raise ValueError(f"Valid {name} target is out of range; never clamp labels")


@dataclass(frozen=True)
class LossSums:
    action: Tensor
    response: Tensor
    value: Tensor
    examples: int
    action_count: int
    response_count: float
    value_count: float

    def objective(self, *, global_examples: int, world_size: int, rho_e: float, rho_v: float) -> Tensor:
        if global_examples <= 0 or world_size <= 0:
            raise ValueError("Expected the full positive global source-decision count")
        if not 0 < rho_v <= rho_e <= 1:
            raise ValueError("Fixed manifest rho_V/rho_E must be positive and consistent")
        return world_size / global_examples * (self.action + 0.25 * self.response / rho_e + 0.1 * self.value / rho_v)

    def metrics(self) -> dict[str, float | int | None]:
        return {
            "examples": self.examples,
            "action_count": self.action_count,
            "response_count": self.response_count,
            "value_count": self.value_count,
            "action_nll": float(self.action.detach()) / self.action_count if self.action_count else None,
            "response_nll": float(self.response.detach()) / self.response_count if self.response_count else None,
            "rank_nll": float(self.value.detach()) / self.value_count if self.value_count else None,
        }


def _masked_ce(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    allowed: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    safe_target = torch.where(valid, target, 0)
    if torch.any(valid & ~allowed.any(-1)):
        raise ValueError("A valid sample has an empty rule/legal mask")
    if torch.any(valid & ~allowed.gather(1, safe_target[:, None]).squeeze(1)):
        raise ValueError("A valid target is excluded by the public/legal mask")
    dummy_allowed = torch.where(valid[:, None], allowed, True)
    dummy_logits = torch.where(valid[:, None], logits.float(), 0.0)
    masked = dummy_logits.masked_fill(~dummy_allowed, -torch.inf)
    nll = F.cross_entropy(masked, safe_target, reduction="none")
    if weights is not None:
        nll = nll * weights
    return torch.where(valid, nll, 0.0).sum()


def loss_sums(outputs: ModelOutputs, labels: Labels, legal_mask: Tensor) -> LossSums:
    batch = outputs.policy_logits.shape[0]
    if (
        outputs.policy_logits.shape != (batch, 546)
        or outputs.response_logits.shape != (batch, 21)
        or outputs.rank_logits.shape != (batch, 21, 4)
    ):
        raise ValueError("Three-head output shapes violate the model contract")
    if legal_mask.shape != (batch, 546) or legal_mask.dtype != torch.bool:
        raise ValueError("Legal action mask must be [B,546] boolean")
    labels.validate(batch)
    if torch.any(labels.response_valid & ~outputs.action_present):
        raise ValueError("Auxiliary labels require a real ACTION suffix")
    if not all(
        torch.isfinite(value).all() for value in (outputs.policy_logits, outputs.response_logits, outputs.rank_logits)
    ):
        raise FloatingPointError("Non-finite raw network logits; do not hide with nan_to_num")
    action = _masked_ce(outputs.policy_logits, labels.action_target, labels.action_valid, legal_mask)
    response_valid = labels.response_valid
    if labels.response_weight is None or labels.value_weight is None:
        raise ValueError("Auxiliary loss weights must be materialized")
    response = _masked_ce(
        outputs.response_logits,
        labels.response_target,
        response_valid,
        labels.response_allowed,
        weights=labels.response_weight,
    )
    safe_event = torch.where(labels.value_valid, labels.response_target, 0)
    safe_rank = torch.where(labels.value_valid, labels.final_rank_target, 0)
    selected = outputs.rank_logits.float().gather(1, safe_event[:, None, None].expand(-1, 1, 4)).squeeze(1)
    selected = torch.where(labels.value_valid[:, None], selected, 0.0)
    rank_nll = F.cross_entropy(selected, safe_rank, reduction="none")
    value = torch.where(labels.value_valid, rank_nll * labels.value_weight, 0.0).sum()
    return LossSums(
        action,
        response,
        value,
        batch,
        int(labels.action_valid.sum()),
        float((labels.response_valid.float() * labels.response_weight).sum()),
        float((labels.value_valid.float() * labels.value_weight).sum()),
    )
