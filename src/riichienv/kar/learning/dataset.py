from __future__ import annotations

import json
import os
import sqlite3
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..contracts import content_hash
from ..mjsoul import validate_protocol_identity
from ..schema import TOKEN_BYTE_SIZE
from ..versioning import game_count_contract
from .batch import ModelBatch, collate_inputs
from .export import _file_sha, learning_contract
from .losses import Labels


@dataclass(frozen=True)
class TrainingBatch:
    inputs: ModelBatch
    labels: Labels
    game_ids: tuple[str, ...]
    source_count: int | None = None

    def __post_init__(self) -> None:
        if self.source_count is None:
            object.__setattr__(self, "source_count", len(set(self.inputs.context_ids)))
        if self.source_count < 1:
            raise ValueError("A training batch needs at least one source decision")

    def to(self, device: torch.device | str) -> TrainingBatch:
        return TrainingBatch(self.inputs.to(device), self.labels.to(device), self.game_ids, self.source_count)


def collate_examples(examples: list[dict[str, Any]]) -> TrainingBatch:
    inputs = collate_inputs(examples)

    def vector(name: str, *, boolean: bool = False) -> torch.Tensor:
        return torch.tensor([example[name] for example in examples], dtype=torch.bool if boolean else torch.int64)

    labels = Labels(
        vector("action_target"),
        vector("action_valid", boolean=True),
        vector("response_target"),
        vector("response_applicable", boolean=True),
        vector("response_observed", boolean=True),
        vector("final_rank"),
        vector("value_valid", boolean=True),
        torch.tensor([example["response_allowed"] for example in examples], dtype=torch.bool),
    )
    labels.validate(inputs.size)
    return TrainingBatch(inputs, labels, tuple(example["game_id"] for example in examples))


class DecisionDataset(Dataset[dict[str, Any]]):
    def __init__(self, directory: Path | str, split: str, *, verify_hash: bool = True, history_cache_items: int = 32):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown dataset split")
        self.directory, self.split = Path(directory).resolve(), split
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in self.manifest.items() if key != "manifest_sha256"}
        if (
            self.manifest.get("manifest_sha256") != content_hash(unsigned)
            or self.manifest.get("status") != "PASS"
            or not self.manifest.get("complete")
        ):
            raise ValueError("Dataset is incomplete or its manifest changed")
        current = learning_contract()
        if any(self.manifest["scope"].get(key) != value for key, value in current.items()):
            raise ValueError("Dataset schema/rules/feature/loss contract differs")
        validate_protocol_identity(self.manifest["scope"], context="Training dataset")
        counts = self.manifest.get("counts", {})
        scale = game_count_contract(
            self.manifest["scope"].get("available_split_counts", {}),
            {split: counts.get(split, {}).get("games") for split in ("train", "validation", "test")},
        )
        if any(self.manifest.get(key) != value for key, value in scale.items()):
            raise ValueError("Dataset eligible/train/split count evidence differs")
        self.path = self.directory / "decisions.sqlite3"
        if verify_hash and _file_sha(self.path) != self.manifest["database_sha256"]:
            raise ValueError("Dataset database SHA-256 mismatch")
        self._connection: sqlite3.Connection | None = None
        self._pid: int | None = None
        self._histories: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self.history_cache_items = history_cache_items
        db = self._db()
        if db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone() != ("true",):
            raise ValueError("Dataset transaction is not marked complete")
        stored_scope = db.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()
        if stored_scope is None or json.loads(stored_scope[0]) != self.manifest["scope"]:
            raise ValueError("Dataset SQLite/manifest scope contract differs")
        stored_games = dict(db.execute("SELECT split,COUNT(*) FROM games GROUP BY split").fetchall())
        if any(stored_games.get(split, 0) != count for split, count in scale["exported_split_counts"].items()):
            raise ValueError("Dataset complete-game counts differ from SQLite")
        self.indices = db.execute(
            "SELECT s.row_id,s.history_rows FROM samples s JOIN games g ON s.game_id=g.game_id "
            "WHERE g.split=? ORDER BY s.row_id",
            (split,),
        ).fetchall()
        if len(self.indices) != self.manifest["counts"][split]["examples"]:
            raise ValueError("Dataset row count differs from manifest")

    def _db(self) -> sqlite3.Connection:
        if self._connection is None or self._pid != os.getpid():
            self._connection = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True)
            self._pid = os.getpid()
            self._histories = OrderedDict()
        return self._connection

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.update(_connection=None, _pid=None, _histories=OrderedDict())
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_id, _ = self.indices[index]
        cursor = self._db().execute("SELECT * FROM samples WHERE row_id=?", (row_id,))
        record = cursor.fetchone()
        if record is None:
            raise ValueError("Indexed decision disappeared")
        row = dict(zip((item[0] for item in cursor.description), record))
        key = row["game_id"], row["observer"]
        if key not in self._histories:
            found = self._db().execute("SELECT tokens FROM histories WHERE game_id=? AND observer=?", key).fetchone()
            if found is None:
                raise ValueError("Decision history is missing")
            self._histories[key] = found[0]
            if len(self._histories) > self.history_cache_items:
                self._histories.popitem(last=False)
        self._histories.move_to_end(key)
        count = (row["history_rows"] + 1) * TOKEN_BYTE_SIZE
        history = self._histories[key]
        if len(history) < count or len(row["state"]) != 63 * TOKEN_BYTE_SIZE or len(row["action"]) != TOKEN_BYTE_SIZE:
            raise ValueError("Shared history/state/ACTION size mismatch")
        legal = tuple(value for value in struct.unpack("<32H", row["candidates"]) if value != 546)
        return {
            "decision_row_id": row_id,
            "game_id": row["game_id"],
            "tokens": history[:count] + row["state"] + row["action"],
            "candidates": legal,
            "decide_index": row["history_rows"] + 63,
            "action_index": row["history_rows"] + 64,
            "action_target": row["action_target"],
            "action_valid": bool(row["action_valid"]),
            "response_target": row["response_target"],
            "response_applicable": bool(row["response_applicable"]),
            "response_observed": bool(row["response_observed"]),
            "final_rank": row["final_rank"],
            "value_valid": bool(row["value_valid"]),
            "response_allowed": [bool(value) for value in row["response_allowed"]],
            "action_provenance": row["provenance"],
        }
