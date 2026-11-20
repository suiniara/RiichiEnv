from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from ..observation import PublicSnapshot
from ..runtime_contracts import RuntimeBinding
from ..schema import TOKEN_BYTE_SIZE
from ..tokenize import TokenHistoryCache, TokenSequence, append_action, encode_prefix
from .batch import collate_inputs, rule_feature_view
from .config import frozen_inference_recipe_sha256
from .model import KARTransformer, PrefixCache
from .precision import capture_frozen_backend_precision, frozen_fp32, require_frozen_backend_precision


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
        self.backend_precision = capture_frozen_backend_precision(next(model.parameters()).device.type)
        recipe = frozen_inference_recipe_sha256(
            model.config, use_rule_features=use_rule_features, backend_precision=self.backend_precision
        )
        if any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in (*model.parameters(), *model.buffers())
        ):
            raise ValueError("Frozen FP32 inference requires FP32 model parameters and floating buffers")
        self.model = model.eval()
        self.use_rule_features = use_rule_features
        self.binding = RuntimeBinding.current(
            model_weights_sha256(model), train_manifest_sha256, inference_recipe_sha256=recipe
        )
        self._identity = self._frozen_identity()
        # Bounded by the four seats, with at most one active stream per seat.
        self._history_caches: dict[int, tuple[object, TokenHistoryCache]] = {}
        self._cache: PrefixCache | None = None
        self._prefix: TokenSequence | None = None
        self._context: tuple[str, object | None, str, tuple[int, ...]] | None = None
        self.prefix_calls = self.candidate_calls = self.candidates_scored = 0

    def _frozen_identity(self):
        return (
            self.model._identity(),
            tuple((tensor.data_ptr(), tensor._version) for tensor in self.model.buffers()),
            self.model.config.contract_hash,
            self.use_rule_features,
            self.backend_precision,
        )

    def _unchanged(self) -> None:
        require_frozen_backend_precision(self.backend_precision)
        if self._frozen_identity() != self._identity or self.model.training:
            self._cache = self._prefix = None
            raise ValueError("Frozen inference model was updated, reloaded or switched to training")

    @contextmanager
    def frozen_forward(self) -> Iterator[None]:
        """Guard the whole model forward, including batched auxiliary heads."""
        self._unchanged()
        with frozen_fp32(self.backend_precision):
            yield
        self._unchanged()

    def _encode_policy_row(self, row: Mapping[str, Any]) -> PrefixCache:
        """The same unpadded, single-prefix computation for calibration and play."""
        decide_index = row["decide_index"]
        if type(decide_index) is not int or decide_index < 0:
            raise ValueError("Frozen policy needs an explicit DECIDE prefix boundary")
        length = (decide_index + 1) * TOKEN_BYTE_SIZE
        if len(row["tokens"]) < length:
            raise ValueError("Frozen policy prefix is truncated")
        prefix = {**row, "tokens": row["tokens"][:length], "action_index": -1}
        device = next(self.model.parameters()).device
        batch = collate_inputs([prefix]).to(device)
        batch = rule_feature_view(
            batch, torch.zeros(1, dtype=torch.bool, device=device), enabled=self.use_rule_features
        )
        with self.frozen_forward():
            return self.model.encode_prefix(batch)

    @torch.no_grad()
    def policy_logits_from_rows(self, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        """Archive policy logits without changing the deployed batch/token shape.

        Teacher ACTION and batch padding are excluded before each forward.
        Response/value logits may still be collected in batches separately.
        """
        self._unchanged()
        self._cache = self._prefix = self._context = None
        if not rows:
            raise ValueError("Frozen policy collection requires decision rows")
        logits = [self._encode_policy_row(row).policy_logits[0] for row in rows]
        self.prefix_calls += len(rows)
        return torch.stack(logits)

    @torch.no_grad()
    def policy(self, snapshot: PublicSnapshot, candidates: tuple[int, ...], context: str) -> Sequence[float]:
        self._unchanged()
        self._cache = self._prefix = self._context = None
        seat = snapshot.observer_player_id
        if type(seat) is not int or not 0 <= seat < 4:
            raise ValueError("Invalid observer for frozen inference")
        # Detached snapshots without a ledger lifetime do not populate the cache.
        session = snapshot._history_session
        history_cache = None
        if session is not None:
            slot = self._history_caches.get(seat)
            if slot is None or slot[0] is not session:
                slot = (session, TokenHistoryCache())
                self._history_caches[seat] = slot
            history_cache = slot[1]
        prefix = encode_prefix(snapshot, candidates, history_cache=history_cache)
        row = {
            "tokens": prefix.to_bytes()[108:],
            "candidates": candidates,
            "decide_index": prefix.decide_index,
            "action_index": -1,
            "decision_row_id": context,
        }
        self._cache = self._encode_policy_row(row)
        self._prefix = prefix
        snapshot_sha256 = hashlib.sha256(snapshot.to_bytes()).hexdigest()
        self._context = (context, snapshot._history_session, snapshot_sha256, candidates)
        self.prefix_calls += 1
        return self._cache.policy_logits[0].cpu().tolist()

    @torch.no_grad()
    def candidates(
        self, snapshot: PublicSnapshot, candidates: tuple[int, ...], supported_discards: tuple[int, ...], context: str
    ) -> Mapping[int, tuple[Sequence[float], Sequence[Sequence[float]]]]:
        self._unchanged()
        expected = (context, snapshot._history_session, hashlib.sha256(snapshot.to_bytes()).hexdigest(), candidates)
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
        with self.frozen_forward():
            responses, ranks = self.model.score_actions(self._cache, batch)
        self.candidate_calls += 1
        self.candidates_scored += len(rows)
        return {
            action: (responses[index].cpu().tolist(), ranks[index].cpu().tolist())
            for index, action in enumerate(supported_discards)
        }
