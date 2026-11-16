from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from ..contracts import content_hash

SAMPLER_VERSION = "global_exact_epoch_bucket_sampler_v1"


@dataclass(frozen=True)
class UpdateGroup:
    epoch: int
    update_index: int
    global_indices: tuple[int, ...]
    local_indices: tuple[int, ...]
    microbatch_rounds: int

    @property
    def global_examples(self) -> int:
        return len(self.global_indices)


class GlobalEpochSampler:
    """No padding/duplication of real samples; empty ranks receive graph-only dummies."""

    def __init__(
        self,
        lengths: list[int],
        *,
        seed: int,
        effective_batch: int,
        microbatch: int,
        rank: int = 0,
        world_size: int = 1,
        bucket_window: int = 128,
    ):
        if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
            raise ValueError("Every indexed training decision needs its actual positive token length")
        if min(effective_batch, microbatch, world_size, bucket_window) < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid global sampler configuration")
        self.lengths = tuple(lengths)
        self.seed, self.effective_batch, self.microbatch = seed, effective_batch, microbatch
        self.rank, self.world_size, self.bucket_window = rank, world_size, max(bucket_window, effective_batch)
        self.epoch, self.cursor = 0, 0
        self.order = self._order(0)

    def _order(self, epoch: int) -> tuple[int, ...]:
        order = list(range(len(self.lengths)))
        random.Random(self.seed + 1000003 * epoch).shuffle(order)
        for start in range(0, len(order), self.bucket_window):
            order[start : start + self.bucket_window] = sorted(
                order[start : start + self.bucket_window], key=lambda index: self.lengths[index]
            )
        return tuple(order)

    @property
    def updates_per_epoch(self) -> int:
        return math.ceil(len(self.lengths) / self.effective_batch)

    @property
    def exhausted(self) -> bool:
        return self.cursor == len(self.order)

    def peek(self) -> UpdateGroup:
        if self.exhausted:
            raise StopIteration
        indices = self.order[self.cursor : self.cursor + self.effective_batch]
        local = indices[self.rank :: self.world_size]
        rounds = math.ceil(math.ceil(len(indices) / self.world_size) / self.microbatch)
        return UpdateGroup(self.epoch, self.cursor // self.effective_batch, indices, local, max(1, rounds))

    def commit(self, group: UpdateGroup) -> None:
        if group != self.peek():
            raise ValueError("Only the successfully completed current accumulation group may advance")
        self.cursor += len(group.global_indices)

    def next_epoch(self) -> None:
        if not self.exhausted:
            raise ValueError("Cannot silently abandon an unfinished epoch")
        self.epoch += 1
        self.cursor, self.order = 0, self._order(self.epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": SAMPLER_VERSION,
            "seed": self.seed,
            "effective_batch": self.effective_batch,
            "microbatch": self.microbatch,
            "world_size": self.world_size,
            "bucket_window": self.bucket_window,
            "lengths_sha256": content_hash(self.lengths),
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order_sha256": content_hash(self.order),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        if any(
            state.get(key) != value for key, value in expected.items() if key not in {"epoch", "cursor", "order_sha256"}
        ):
            raise ValueError("Exact sampler resume configuration/data differs")
        epoch, cursor = state["epoch"], state["cursor"]
        if type(epoch) is not int or epoch < 0 or type(cursor) is not int or not 0 <= cursor <= len(self.lengths):
            raise ValueError("Corrupt sampler cursor")
        order = self._order(epoch)
        if content_hash(order) != state["order_sha256"] or (cursor < len(order) and cursor % self.effective_batch):
            raise ValueError("Resume cursor is not a committed optimizer boundary")
        self.epoch, self.cursor, self.order = epoch, cursor, order


def validation_updates(updates_per_epoch: int) -> tuple[int, ...]:
    if updates_per_epoch <= 0:
        raise ValueError("Positive epoch length required")
    return tuple(sorted({math.ceil(updates_per_epoch * fraction / 4) for fraction in range(1, 5)}))
