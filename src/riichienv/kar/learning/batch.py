from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..schema import TOKEN_BYTE_SIZE
from ..versioning import RULE_FEATURE_MASK_VERSION

TOKEN_DTYPE = np.dtype(
    [
        ("kind", "u1"),
        ("actor", "u1"),
        ("source", "u1"),
        ("tiles", "u1", (5,)),
        ("stage", "u1"),
        ("wind", "u1"),
        ("slot", "u1"),
        ("flags", "u1", (16,)),
        ("numeric", "<f4", (20,)),
        ("numeric_present", "u1", (20,)),
        ("action_ids", "<u2"),
        ("action_type_ids", "u1"),
        ("action_tile37", "u1"),
        ("action_tsumogiri", "u1"),
        ("action_riichi", "u1"),
        ("block", "u1"),
        ("positions", "<u4"),
        ("valid", "u1"),
    ],
    align=False,
)
assert TOKEN_DTYPE.itemsize == TOKEN_BYTE_SIZE


def canonical_candidate_ids(candidates: Tensor, size: int) -> Tensor:
    """Canonical set order for pooling/cache identity; padding is not a member."""
    if candidates.ndim != 2 or candidates.shape != (size, 32):
        raise ValueError("Expected thirty-two padded complete action candidates")
    if candidates.dtype != torch.int64 or torch.any((candidates < 0) | (candidates > 546)):
        raise ValueError("Candidate IDs must be int64 complete actions or pad=546")
    ordered = candidates.sort(dim=-1).values
    if torch.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != 546)):
        raise ValueError("Legal candidate IDs must be unique")
    if not torch.all((ordered != 546).any(-1)):
        raise ValueError("A decision requires a nonempty legal set")
    return ordered


@dataclass(frozen=True)
class ModelBatch:
    fields: dict[str, Tensor]
    candidates: Tensor
    decide_index: Tensor
    action_index: Tensor
    context_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> ModelBatch:
        return replace(
            self,
            fields={key: value.to(device) for key, value in self.fields.items()},
            candidates=self.candidates.to(device),
            decide_index=self.decide_index.to(device),
            action_index=self.action_index.to(device),
        )

    @property
    def size(self) -> int:
        return self.fields["kind"].shape[0]

    @property
    def legal_mask(self) -> Tensor:
        ordered = canonical_candidate_ids(self.candidates, self.size)
        mask = torch.zeros(self.size, 547, device=ordered.device, dtype=torch.bool)
        mask.scatter_(1, ordered, True)
        mask = mask[:, :546]
        if not torch.all(mask.any(-1)):
            raise ValueError("A decision requires a nonempty legal set")
        return mask


def collate_inputs(examples: list[dict[str, Any]]) -> ModelBatch:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    arrays = [np.frombuffer(example["tokens"], dtype=TOKEN_DTYPE) for example in examples]
    if any(not len(array) for array in arrays):
        raise ValueError("Empty token sequence")
    length = max(len(array) for array in arrays)
    padded = np.zeros((len(arrays), length), dtype=TOKEN_DTYPE)
    padded["action_ids"] = 546
    for index, array in enumerate(arrays):
        padded[index, : len(array)] = array
    fields = {}
    for key in TOKEN_DTYPE.names or ():
        dtype = np.float32 if key == "numeric" else np.bool_ if key in {"numeric_present", "valid"} else np.int64
        fields[key] = torch.from_numpy(padded[key].astype(dtype, copy=True))
    candidates = torch.full((len(arrays), 32), 546, dtype=torch.int64)
    for index, example in enumerate(examples):
        legal = tuple(example["candidates"])
        if (
            not 1 <= len(legal) <= 32
            or legal != tuple(sorted(set(legal)))
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 546 for value in legal)
        ):
            raise ValueError("Candidates must be canonical complete global action IDs")
        candidates[index, : len(legal)] = torch.tensor(legal)
    batch = ModelBatch(
        fields,
        candidates,
        torch.tensor([example["decide_index"] for example in examples], dtype=torch.int64),
        torch.tensor([example.get("action_index", -1) for example in examples], dtype=torch.int64),
        tuple(example["decision_row_id"] for example in examples),
    )
    _ = batch.legal_mask
    return batch


def feature_mask_draw(
    seed: int, epoch: int, game_id: str, decision_id: str, draw: int = 0, probability: float = 0.25
) -> bool:
    if not 0 <= probability <= 1:
        raise ValueError("Mask probability must be in [0,1]")
    import json  # noqa: PLC0415

    identity = json.dumps(
        [RULE_FEATURE_MASK_VERSION, seed, epoch, game_id, decision_id, draw], separators=(",", ":")
    ).encode()
    uniform = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") / 2**64
    return uniform < probability


def rule_feature_view(batch: ModelBatch, mask_samples: Tensor, *, enabled: bool = True) -> ModelBatch:
    if mask_samples.shape != (batch.size,) or mask_samples.dtype != torch.bool:
        raise ValueError("Feature masking must select whole decision samples")
    fields = dict(batch.fields)
    for key in ("flags", "stage", "numeric", "numeric_present"):
        fields[key] = fields[key].clone()
    selected = (
        mask_samples.to(fields["kind"].device)
        if enabled
        else torch.ones_like(mask_samples, device=fields["kind"].device)
    )
    hand = (fields["kind"] == 4) & selected[:, None]
    state = (fields["kind"] == 2) & selected[:, None]
    fields["flags"][..., :8] = torch.where(hand[..., None], 0, fields["flags"][..., :8])
    fields["numeric"][..., 18:20] = torch.where(state[..., None], 0.0, fields["numeric"][..., 18:20])
    fields["numeric_present"][..., 18:20] = torch.where(state[..., None], False, fields["numeric_present"][..., 18:20])
    if not enabled:
        fields["stage"] = torch.where(state, 0, fields["stage"])
    return replace(batch, fields=fields)
