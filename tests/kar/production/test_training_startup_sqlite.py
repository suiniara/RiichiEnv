"""Real manifest/SQLite startup validation with all model execution blocked."""

# ruff: noqa: E402 -- optional Torch must be checked before ML imports
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from test_full_trainer import FullSyntheticDataset
from test_training_startup import _assert_metered, _metered_config, _stop_before_model
from test_training_startup import startup_driver as startup_driver  # noqa: PLC0414 -- re-export the pytest fixture

from riichienv.kar.budget import BudgetLedger
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning import dataset as dataset_module
from riichienv.kar.learning import trainer
from riichienv.kar.learning.dataset import DecisionDataset
from riichienv.kar.learning.distributed import DistributedStageError
from riichienv.kar.learning.export import SQL_SCHEMA, learning_contract
from riichienv.kar.mjsoul import protocol_identity


def _write_dataset(directory):
    directory.mkdir()
    manifest = FullSyntheticDataset(directory, "train").manifest
    manifest["scope"].update(protocol_identity("c" * 64, "d" * 64))
    database = directory / "decisions.sqlite3"
    with sqlite3.connect(database) as db:
        db.executescript(SQL_SCHEMA)
        db.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [("complete", "true"), ("scope", json.dumps(manifest["scope"]))],
        )
        for split in ("train", "validation"):
            for index in range(3):
                game = f"startup-{split}-{index}"
                db.execute("INSERT INTO games VALUES (?,?,?,?,?)", (game, split, "a" * 64, "[1,2,3,4]", "b" * 64))
                # Real schema/index/count checks; rows are metadata-only and must never reach a model.
                row = (
                    f"{game}/row",
                    game,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    bytes(63 * 139),
                    bytes(139),
                    bytes(64),
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    bytes([1] * 21),
                    "synthetic_metadata",
                    "{}",
                )
                db.execute("INSERT INTO samples VALUES (" + ",".join("?" for _ in row) + ")", row)
    manifest.pop("manifest_sha256")
    manifest["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    manifest["manifest_sha256"] = content_hash(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize(
    ("damage", "message", "hash_calls"),
    [
        (None, "before model allocation", 2),
        ("database", "database SHA-256 mismatch", 1),
        ("manifest", "manifest changed", 0),
        ("contract", "schema/rules/feature/loss contract differs", 0),
        ("authorization", "changed after startup authorization", 1),
    ],
)
def test_manifest_database_and_contract_validation_remain_metered(
    startup_driver,
    tmp_path,
    monkeypatch,
    damage,
    message,
    hash_calls,
):
    _, output, base = startup_driver
    settings, clock = _metered_config(base, monkeypatch)
    directory = tmp_path / "data"
    manifest = _write_dataset(directory)
    authorized_sha = manifest["manifest_sha256"]
    if damage == "database":
        with (directory / "decisions.sqlite3").open("ab") as stream:
            stream.write(b"modified after export")
    elif damage == "manifest":
        manifest["unexpected_change"] = True
    elif damage == "contract":
        manifest["scope"][next(iter(learning_contract()))] = "another-contract"
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = content_hash(manifest)
        authorized_sha = manifest["manifest_sha256"]
    elif damage == "authorization":
        authorized_sha = "e" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest))
    authorization = SimpleNamespace(document={"dataset_manifest_sha256": authorized_sha})
    original_hash, original_read = dataset_module._file_sha, Path.read_text
    hashes = []

    def metered_hash(path):
        _assert_metered(output, settings)
        hashes.append(path)
        clock.now += 60.0
        return original_hash(path)

    def metered_read(path, *args, **kwargs):
        if path == directory / "manifest.json":
            _assert_metered(output, settings)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(dataset_module, "_file_sha", metered_hash)
    monkeypatch.setattr(Path, "read_text", metered_read)
    monkeypatch.setattr(trainer, "DecisionDataset", DecisionDataset)
    monkeypatch.setattr(trainer, "_model_components", _stop_before_model)
    with pytest.raises(DistributedStageError, match=message):
        trainer.train_production(directory, output, settings, authorization=authorization)
    assert len(hashes) == hash_calls
    ledger = BudgetLedger(output / "costs.jsonl", settings.budget, allow_create=False)
    assert ledger.spent_usd == pytest.approx(hash_calls * 0.6) and not ledger.pending_compute
    assert not (output / "checkpoint.json").exists()
    identity = json.loads((output / "startup.json").read_text())["identity"]
    assert identity["dataset_manifest_sha256"] == authorized_sha
