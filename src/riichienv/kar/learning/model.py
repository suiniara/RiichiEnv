from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..schema import ACTION_FACTOR_FIELDS, action_factor_ids
from ..versioning import INPUT_PIPELINE_VERSION
from .batch import ModelBatch, canonical_candidate_ids
from .config import ModelConfig

SCALAR_EMBEDDINGS = {
    "kind": 48,
    "actor": 5,
    "source": 5,
    "stage": 16,
    "wind": 5,
    "slot": 17,
    "action_ids": 547,
    "action_type_ids": 13,
    "action_tile37": 39,
    "action_tsumogiri": 3,
    "action_riichi": 3,
}


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(value.dtype) * self.weight


def block_attention_mask(block: Tensor, valid: Tensor) -> Tensor:
    if block.ndim != 2 or valid.shape != block.shape or valid.dtype != torch.bool:
        raise ValueError("Expected [B,L] blocks and boolean validity")
    if not torch.all(valid[:, 0]) or not torch.all(block[:, 0] == 0):
        raise ValueError("Every sequence requires a visible BOS key")
    length = block.shape[1]
    q, k = block[:, :, None], block[:, None, :]
    indices = torch.arange(length, device=block.device)
    causal = indices[None, :] <= indices[:, None]
    self_only = indices[None, :] == indices[:, None]
    allowed = (
        ((q == 0) & (k == 0) & causal)
        | ((q == 1) & (k <= 1))
        | ((q == 2) & (k <= 2))
        | ((q == 3) & ((k < 3) | self_only))
    )
    allowed &= valid[:, None, :]
    bos = torch.zeros_like(allowed)
    bos[:, :, 0] = True
    return torch.where(valid[:, :, None], allowed, bos)[:, None]


class InputEmbedding(nn.Module):
    action_factors: Tensor

    def __init__(self, width: int):
        super().__init__()
        self.scalar = nn.ModuleDict({key: nn.Embedding(count, width) for key, count in SCALAR_EMBEDDINGS.items()})
        self.tiles = nn.ModuleList([nn.Embedding(39, width) for _ in range(5)])
        self.flags = nn.ModuleList([nn.Embedding(3, width) for _ in range(16)])
        self.numeric = nn.Linear(40, width, bias=False)
        factors = [action_factor_ids(action) for action in range(546)] + [(546, 0, 0, 0, 0)]
        self.register_buffer("action_factors", torch.tensor(factors, dtype=torch.int64), persistent=False)

    def forward(self, fields: dict[str, Tensor], candidates: Tensor | None = None) -> Tensor:
        value = self.numeric(torch.cat((fields["numeric"].float(), fields["numeric_present"].float()), dim=-1))
        for key, table in self.scalar.items():
            value = value + table(fields[key])
        for column, table in enumerate(self.tiles):
            value = value + table(fields["tiles"][..., column])
        for column, table in enumerate(self.flags):
            value = value + table(fields["flags"][..., column])
        legal_rows = (fields["kind"] == 8) & fields["valid"]
        if torch.any(legal_rows):
            if candidates is None or not torch.all(legal_rows.sum(-1) == 1):
                raise ValueError("A prefix requires one LEGAL_SET row and its complete candidates")
            ordered = canonical_candidate_ids(candidates, fields["kind"].shape[0])
            factors = self.action_factors[ordered]
            combined = self.scalar[ACTION_FACTOR_FIELDS[0]](factors[..., 0])
            for column, name in enumerate(ACTION_FACTOR_FIELDS[1:], 1):
                combined = combined + self.scalar[name](factors[..., column])
            members = ordered != 546
            pooled = torch.where(members[..., None], combined, 0).sum(1, dtype=torch.float32)
            pooled = pooled / members.sum(-1, keepdim=True)
            value = value + legal_rows[..., None] * pooled[:, None, :].to(value.dtype)
        return value * fields["valid"][..., None]


class TransformerBlock(nn.Module):
    frequency: Tensor

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads, self.head_dim = config.heads, config.width // config.heads
        self.attention_norm = RMSNorm(config.width, config.norm_eps)
        self.ffn_norm = RMSNorm(config.width, config.norm_eps)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        self.gate = nn.Linear(config.width, config.feedforward, bias=False)
        self.up = nn.Linear(config.width, config.feedforward, bias=False)
        self.down = nn.Linear(config.feedforward, config.width, bias=False)
        frequency = config.rope_base ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        self.register_buffer("frequency", frequency, persistent=False)

    def _rope(self, value: Tensor, positions: Tensor) -> Tensor:
        angle = positions[:, None, :, None].float() * self.frequency[None, None, None, :]
        even, odd = value[..., 0::2], value[..., 1::2]
        rotated = torch.stack((even * angle.cos() - odd * angle.sin(), even * angle.sin() + odd * angle.cos()), dim=-1)
        return rotated.flatten(-2).to(value.dtype)

    def projections(self, value: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        parts = self.qkv(self.attention_norm(value)).chunk(3, dim=-1)
        query, key, content = [
            part.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim).transpose(1, 2) for part in parts
        ]
        return self._rope(query, positions), self._rope(key, positions), content

    def finish(self, value: Tensor, attention: Tensor, valid: Tensor) -> Tensor:
        merged = attention.transpose(1, 2).contiguous().reshape_as(value)
        value = value + self.output(merged)
        normalized = self.ffn_norm(value)
        value = value + self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        return value * valid[..., None]

    def forward(self, value: Tensor, positions: Tensor, mask: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        query, key, content = self.projections(value, positions)
        attention = F.scaled_dot_product_attention(query, key, content, attn_mask=mask, dropout_p=0.0)
        return self.finish(value, attention, valid), key, content


@dataclass(frozen=True)
class ModelOutputs:
    policy_logits: Tensor
    response_logits: Tensor
    rank_logits: Tensor
    action_present: Tensor


@dataclass(frozen=True)
class PrefixCache:
    owner: int
    parameter_identity: tuple[tuple[int, int], ...]
    context_ids: tuple[str, ...]
    key_values: tuple[tuple[Tensor, Tensor], ...]
    tensor_versions: tuple[int, ...]
    valid: Tensor
    action_position: Tensor
    policy_logits: Tensor
    device: torch.device
    dtype: torch.dtype
    candidates: Tensor
    input_pipeline_version: str
    model_contract_sha256: str


class KARTransformer(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.embedding = InputEmbedding(self.config.width)
        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.layers)])
        self.final_norm = RMSNorm(self.config.width, self.config.norm_eps)
        self.policy_head = nn.Linear(self.config.width, 546)
        self.response_head = nn.Linear(self.config.width, 21)
        self.value_head = nn.Linear(self.config.width, 84)
        self.apply(self._initialize)
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != self.config.parameter_count:
            raise RuntimeError(f"KAR parameter contract mismatch: {actual} != {self.config.parameter_count}")

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _identity(self) -> tuple[tuple[int, int], ...]:
        return tuple((tensor.data_ptr(), tensor._version) for tensor in (*self.parameters(), *self.buffers()))

    def _hidden(self, batch: ModelBatch, *, capture: bool = False) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        valid, block, position = batch.fields["valid"], batch.fields["block"], batch.fields["positions"]
        if valid.shape != block.shape or block.shape != position.shape or valid.ndim != 2:
            raise ValueError("Model token metadata must share [B,L] shape")
        if torch.any((block < 0) | (block > 3)) or torch.any(position < 0):
            raise ValueError("Invalid block or logical position")
        if not torch.all(((batch.fields["kind"] == 8) & valid).sum(-1) == 1):
            raise ValueError("Dense/prefix computation requires one LEGAL_SET row")
        mask = block_attention_mask(block, valid)
        value = self.embedding(batch.fields, batch.candidates)
        cache = []
        for layer in self.layers:
            value, key, content = layer(value, position, mask, valid)
            if capture:
                cache.append((key.detach().clone(), content.detach().clone()))
        return self.final_norm(value) * valid[..., None], tuple(cache)

    @staticmethod
    def _gather(hidden: Tensor, index: Tensor, kinds: Tensor, expected: int) -> Tensor:
        if (
            index.shape != (hidden.shape[0],)
            or index.dtype != torch.int64
            or torch.any((index < 0) | (index >= hidden.shape[1]))
        ):
            raise ValueError("Invalid readout token index")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        if not torch.all(kinds[rows, index] == expected):
            raise ValueError("Readout position has a different token kind")
        return hidden[rows, index]

    def _readouts(self, decide: Tensor, action: Tensor, present: Tensor) -> ModelOutputs:
        with torch.autocast(device_type=decide.device.type, enabled=False):
            policy = self.policy_head(decide.float())
            response = self.response_head(action.float()) * present[:, None]
            ranks = self.value_head(action.float()).reshape(-1, 21, 4) * present[:, None, None]
        return ModelOutputs(policy, response, ranks, present)

    def forward(self, batch: ModelBatch) -> ModelOutputs:
        hidden, _ = self._hidden(batch)
        decide = self._gather(hidden, batch.decide_index, batch.fields["kind"], 9)
        present = batch.action_index >= 0
        safe = torch.where(present, batch.action_index, batch.decide_index)
        if torch.any(safe >= hidden.shape[1]):
            raise ValueError("ACTION readout index is out of range")
        rows = torch.arange(batch.size, device=hidden.device)
        if torch.any(present & (batch.fields["kind"][rows, safe] != 10)):
            raise ValueError("ACTION readout has no ACTION token")
        return self._readouts(decide, hidden[rows, safe], present)

    @torch.no_grad()
    def encode_prefix(self, batch: ModelBatch) -> PrefixCache:
        if self.training or torch.any(batch.fields["valid"] & (batch.fields["block"] == 3)):
            raise ValueError("Prefix K/V is eval-only and must exclude teacher ACTION")
        if len(batch.context_ids) != batch.size or any(not identity for identity in batch.context_ids):
            raise ValueError("Prefix cache requires explicit decision identities")
        hidden, keys = self._hidden(batch, capture=True)
        decide = self._gather(hidden, batch.decide_index, batch.fields["kind"], 9)
        rows = torch.arange(batch.size, device=hidden.device)
        valid = batch.fields["valid"].clone()
        positions = batch.fields["positions"][rows, batch.decide_index] + 1
        with torch.autocast(device_type=decide.device.type, enabled=False):
            policy = self.policy_head(decide.float())
        candidates = canonical_candidate_ids(batch.candidates, batch.size).clone()
        tensors = [tensor for pair in keys for tensor in pair] + [valid, positions, policy, candidates]
        return PrefixCache(
            id(self),
            self._identity(),
            batch.context_ids,
            keys,
            tuple(tensor._version for tensor in tensors),
            valid,
            positions,
            policy,
            hidden.device,
            hidden.dtype,
            candidates,
            INPUT_PIPELINE_VERSION,
            self.config.contract_hash,
        )

    @torch.no_grad()
    def score_actions(
        self, cache: PrefixCache, actions: ModelBatch, owners: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if (
            getattr(cache, "input_pipeline_version", None) != INPUT_PIPELINE_VERSION
            or getattr(cache, "model_contract_sha256", None) != self.config.contract_hash
        ):
            raise ValueError("Prefix cache input/model contract differs")
        if self.training or cache.owner != id(self) or cache.parameter_identity != self._identity():
            raise ValueError("Stale or foreign model prefix cache")
        tensors = [tensor for pair in cache.key_values for tensor in pair] + [
            cache.valid,
            cache.action_position,
            cache.policy_logits,
            cache.candidates,
        ]
        if tuple(tensor._version for tensor in tensors) != cache.tensor_versions:
            raise ValueError("Prefix cache tensors were modified")
        parameter = next(self.parameters())
        if parameter.device != cache.device or parameter.dtype != cache.dtype:
            raise ValueError("Model device or dtype changed after prefix encoding")
        if (
            actions.fields["kind"].shape != (actions.size, 1)
            or not torch.all(actions.fields["kind"] == 10)
            or not torch.all(actions.fields["valid"])
        ):
            raise ValueError("Each candidate branch must contain exactly one valid ACTION")
        if owners is None:
            if len(cache.context_ids) != 1:
                raise ValueError("Batched prefixes require explicit candidate owner indices")
            owners = torch.zeros(actions.size, dtype=torch.int64, device=cache.device)
        if (
            owners.dtype != torch.int64
            or owners.shape != (actions.size,)
            or torch.any((owners < 0) | (owners >= len(cache.context_ids)))
        ):
            raise ValueError("Invalid candidate owner index")
        expected_ids = tuple(cache.context_ids[index] for index in owners.cpu().tolist())
        if actions.context_ids != expected_ids:
            raise ValueError("Candidate belongs to another observer/decision")
        if not torch.equal(canonical_candidate_ids(actions.candidates, actions.size), cache.candidates[owners]):
            raise ValueError("Candidate set differs from the cached prefix input")
        if not torch.equal(actions.fields["positions"][:, 0], cache.action_position[owners]):
            raise ValueError("Candidate logical position differs from the cached prefix")
        value = self.embedding(actions.fields)
        for layer, (prefix_key, prefix_value) in zip(self.layers, cache.key_values):
            query, key, content = layer.projections(value, actions.fields["positions"])
            keys = torch.cat((prefix_key[owners], key), dim=2)
            values = torch.cat((prefix_value[owners], content), dim=2)
            allowed = torch.cat(
                (cache.valid[owners], torch.ones(actions.size, 1, device=cache.device, dtype=torch.bool)), dim=-1
            )
            attention = F.scaled_dot_product_attention(
                query, keys, values, attn_mask=allowed[:, None, None, :], dropout_p=0.0
            )
            value = layer.finish(value, attention, actions.fields["valid"])
        value = self.final_norm(value)[:, 0]
        with torch.autocast(device_type=value.device.type, enabled=False):
            return self.response_head(value.float()), self.value_head(value.float()).reshape(-1, 21, 4)
