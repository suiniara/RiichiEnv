from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import torch

from ..observation import PublicSnapshot
from ..runtime_contracts import RuntimeBinding
from ..tokenize import TokenSequence, append_action, encode_prefix
from .batch import collate_inputs, rule_feature_view
from .config import frozen_inference_recipe_sha256
from .model import KARTransformer, PrefixCache


def model_weights_sha256(model: KARTransformer) -> str:
    digest = hashlib.sha256()
    tensors = dict(model.state_dict())
    for name, tensor in model.named_buffers():
        tensors.setdefault("non_parameter_buffer/" + name, tensor)
    for name, tensor in tensors.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenModelPredictor:
    def __init__(self, model: KARTransformer, train_manifest_sha256: str, *, use_rule_features: bool = True):
        recipe = frozen_inference_recipe_sha256(model.config, use_rule_features=use_rule_features)
        if any(value.is_floating_point() and value.dtype != torch.float32
               for value in (*model.parameters(), *model.buffers())):
            raise ValueError("Frozen FP32 inference requires FP32 model parameters and floating buffers")
        self.model = model.eval()
        self.use_rule_features = use_rule_features
        self.binding = RuntimeBinding.current(
            model_weights_sha256(model), train_manifest_sha256, inference_recipe_sha256=recipe
        )
        self._identity = self._frozen_identity()
        self._cache: PrefixCache | None = None
        self._prefix: TokenSequence | None = None
        self._context: tuple[str, str, tuple[int, ...]] | None = None
        self.prefix_calls = self.candidate_calls = self.candidates_scored = 0

    def _frozen_identity(self):
        return (
            self.model._identity(),
            tuple((tensor.data_ptr(), tensor._version) for tensor in self.model.buffers()),
            self.model.config.contract_hash,
            self.use_rule_features,
        )

    def _unchanged(self) -> None:
        if self._frozen_identity() != self._identity or self.model.training:
            self._cache = self._prefix = None
            raise ValueError("Frozen inference model was updated, reloaded or switched to training")

    @torch.no_grad()
    def policy(self, snapshot: PublicSnapshot, candidates: tuple[int, ...], context: str) -> Sequence[float]:
        self._unchanged()
        self._prefix = encode_prefix(snapshot, candidates)
        prefix = self._prefix
        row = {
            "tokens": prefix.to_bytes()[108:],
            "candidates": candidates,
            "decide_index": prefix.decide_index,
            "action_index": -1,
            "decision_row_id": context,
        }
        device = next(self.model.parameters()).device
        batch = collate_inputs([row]).to(device)
        batch = rule_feature_view(
            batch, torch.zeros(1, dtype=torch.bool, device=device), enabled=self.use_rule_features
        )
        with torch.autocast(device_type=device.type, enabled=False):
            self._cache = self.model.encode_prefix(batch)
        self._context = (context, hashlib.sha256(snapshot.to_bytes()).hexdigest(), candidates)
        self.prefix_calls += 1
        return self._cache.policy_logits[0].cpu().tolist()

    @torch.no_grad()
    def candidates(
        self, snapshot: PublicSnapshot, candidates: tuple[int, ...], supported_discards: tuple[int, ...], context: str
    ) -> Mapping[int, tuple[Sequence[float], Sequence[Sequence[float]]]]:
        self._unchanged()
        expected = (context, hashlib.sha256(snapshot.to_bytes()).hexdigest(), candidates)
        if self._context != expected or self._cache is None or self._prefix is None:
            raise ValueError("Candidate scores cannot reuse another observer/decision prefix")
        if (
            not supported_discards
            or len(set(supported_discards)) != len(supported_discards)
            or any(action not in candidates or not 0 <= action < 148 for action in supported_discards)
        ):
            raise ValueError("Expected the full distinct gated discard set")
        rows = []
        for action in supported_discards:
            conditional = append_action(self._prefix, action)
            rows.append(
                {
                    "tokens": conditional.to_bytes()[-139:],
                    "candidates": candidates,
                    "decide_index": 0,
                    "action_index": 0,
                    "decision_row_id": context,
                }
            )
        device = next(self.model.parameters()).device
        batch = collate_inputs(rows).to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            responses, ranks = self.model.score_actions(self._cache, batch)
        self.candidate_calls += 1
        self.candidates_scored += len(rows)
        return {
            action: (responses[index].cpu().tolist(), ranks[index].cpu().tolist())
            for index, action in enumerate(supported_discards)
        }
