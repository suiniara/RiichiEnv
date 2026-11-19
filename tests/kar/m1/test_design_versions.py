"""CPU-only synthetic serialization tests; no training job or old artifact migration."""

import hashlib
import json
import sqlite3
import struct

import pytest
import torch

from riichienv.kar.contracts import MJSOUL_RULESET, content_hash
from riichienv.kar.learning.checkpoint import load_checkpoint, save_checkpoint
from riichienv.kar.learning.dataset import DecisionDataset, collate_examples
from riichienv.kar.learning.deployment import FrozenRelease, save_frozen_release
from riichienv.kar.learning.export import SQL_SCHEMA, learning_contract
from riichienv.kar.learning.production_checkpoint import (
    rank_rng_state,
    read_production_checkpoint,
    save_production_checkpoint,
)
from riichienv.kar.learning.train import create_optimizer, create_scheduler
from riichienv.kar.mjsoul import protocol_identity
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.response_mask import public_rule_response_mask
from riichienv.kar.versioning import artifact_contract, game_count_contract

CORE_FIELDS = tuple(artifact_contract())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(directory, manifest):
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    record = {**body, "manifest_sha256": content_hash(body)}
    (directory / "manifest.json").write_text(json.dumps(record), encoding="utf-8")
    return record


@pytest.fixture
def synthetic_dataset(tmp_path, make_sample):
    directory = tmp_path / "synthetic-serialization-only"
    directory.mkdir()
    available = {"train": 2, "validation": 1, "test": 1}
    exported = {"train": 1, "validation": 1, "test": 0}
    scope = {
        **learning_contract(),
        **protocol_identity("a" * 64, "c" * 64),
        "available_split_counts": available,
        "split_salt": "synthetic-only",
    }
    database = directory / "decisions.sqlite3"
    db = sqlite3.connect(database)
    db.executescript(SQL_SCHEMA)
    db.execute("INSERT INTO metadata VALUES ('scope',?)", (json.dumps(scope),))
    db.execute("INSERT INTO metadata VALUES ('complete','true')")
    examples = []
    for split, seed in (("train", 42), ("validation", 0)):
        sample = make_sample(seed)
        examples.append(sample)
        prefix = sample["prefix"]
        blob = prefix.to_bytes()[108:]
        end = (prefix.history_rows + 1) * 139
        identity = "synthetic-" + split
        db.execute("INSERT INTO games VALUES (?,?,?,?,?)", (identity, split, "a" * 64, "[1,2,3,4]", "b" * 64))
        db.execute("INSERT INTO histories VALUES (?,?,?)", (identity, 0, blob[:end]))
        ledger = ObserverLedger(0, (0, 1, 2, 3))
        for event in MJSOUL_RULESET.create_env(seed=seed).mjai_log:
            ledger.apply_event(event)
        allowed = public_rule_response_mask(ledger.snapshot(), sample["action_target"])
        candidates = prefix.candidates + (546,) * (32 - len(prefix.candidates))
        values = (
            identity + "/row0",
            identity,
            0,
            0,
            0,
            1,
            2,
            prefix.history_rows,
            blob[end:],
            sample["tokens"][-139:],
            struct.pack("<32H", *candidates),
            sample["action_target"],
            1,
            1,
            1,
            0,
            0,
            1,
            bytes(allowed),
            "synthetic-unit",
            "{}",
        )
        db.execute("INSERT INTO samples VALUES (" + ",".join("?" for _ in values) + ")", values)
    db.commit()
    db.close()
    counts = {
        split: {
            "games": exported[split],
            "examples": exported[split],
            "response_valid": exported[split],
            "value_valid": exported[split],
        }
        for split in exported
    }
    manifest = {
        **game_count_contract(available, exported),
        "scope": scope,
        "complete": True,
        "status": "PASS",
        "counts": counts,
        "rho_E": 1.0,
        "rho_V": 1.0,
        "is_probe_subset": True,
        "database_sha256": _sha(database),
        "fixture_origin": "synthetic-unit-not-a-real-corpus",
    }
    return directory, _write_manifest(directory, manifest), examples


def test_current_sqlite_contract_loads_without_changing_model_inputs(synthetic_dataset):
    directory, manifest, examples = synthetic_dataset
    train = DecisionDataset(directory, "train")
    assert len(train) == 1 and train[0]["tokens"] == examples[0]["tokens"]
    assert train.manifest["N_eligible_half_games"] == 4 and train.manifest["N_train_half_games"] == 1
    original = collate_examples([train[0]])
    poisoned = {
        **train[0],
        "response_target": 255,
        "final_rank": 255,
        "response_allowed": [False] * 21,
        "response_applicable": False,
        "response_observed": False,
        "value_valid": False,
    }
    changed = collate_examples([poisoned])
    for key in original.inputs.fields:
        torch.testing.assert_close(original.inputs.fields[key], changed.inputs.fields[key], rtol=0, atol=0)
    assert manifest["scope"]["response_mask_version"] == "public_rule_response_mask_v3_abort_last_draw"


@pytest.mark.parametrize("field", ["protocol_sha256", "protocol_descriptor_sha256"])
def test_training_dataset_requires_source_and_complete_schema_digests(synthetic_dataset, field):
    directory, manifest, _ = synthetic_dataset
    manifest["scope"].pop(field)
    _write_manifest(directory, manifest)
    with pytest.raises(ValueError, match="Protocol identity requires"):
        DecisionDataset(directory, "train")


@pytest.mark.parametrize("field", CORE_FIELDS)
@pytest.mark.parametrize("missing", [True, False])
def test_old_or_foreign_dataset_contract_is_rejected_before_sqlite(synthetic_dataset, field, missing):
    directory, manifest, _ = synthetic_dataset
    if missing:
        manifest["scope"].pop(field)
    else:
        manifest["scope"][field] = "old-or-foreign"
    _write_manifest(directory, manifest)
    with pytest.raises(ValueError, match="contract differs"):
        DecisionDataset(directory, "train")


def test_sqlite_scope_cannot_be_relabelled_by_only_changing_the_manifest(synthetic_dataset):
    directory, manifest, _ = synthetic_dataset
    database = directory / "decisions.sqlite3"
    with sqlite3.connect(database) as db:
        scope = dict(manifest["scope"], response_mask_version="public_rule_response_mask_v1")
        db.execute("UPDATE metadata SET value=? WHERE key='scope'", (json.dumps(scope),))
    manifest["database_sha256"] = _sha(database)
    _write_manifest(directory, manifest)
    with pytest.raises(ValueError, match="SQLite/manifest scope"):
        DecisionDataset(directory, "train")


def test_dataset_rejects_previous_ryanmen_semantics_even_with_consistent_storage(synthetic_dataset):
    directory, manifest, _ = synthetic_dataset
    manifest["scope"]["feature_contract_version"] = "visible_rule_features_v1_2_2"
    manifest["scope"]["input_pipeline_version"] = "semantic_public_clock_frozen_decisions_v5_response_pass"
    database = directory / "decisions.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("UPDATE metadata SET value=? WHERE key='scope'", (json.dumps(manifest["scope"]),))
    manifest["database_sha256"] = _sha(database)
    _write_manifest(directory, manifest)
    with pytest.raises(ValueError, match="feature/loss contract differs"):
        DecisionDataset(directory, "train")


def _tamper_checkpoint(path, field, *, production):
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    generation = path.parent / sidecar["checkpoint_file"]
    payload = torch.load(generation, weights_only=True)
    metadata = payload["metadata"]
    target = metadata["release_contract"] if production else metadata
    target[field] = "old-or-foreign"
    torch.save(payload, generation)
    sidecar.update(metadata=metadata, metadata_sha256=content_hash(metadata), checkpoint_sha256=_sha(generation))
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return target


@pytest.mark.parametrize("field", CORE_FIELDS)
def test_legacy_checkpoint_rejects_checksum_valid_old_contract(tmp_path, miniature, field):
    optimizer = create_optimizer(miniature)
    scheduler = create_scheduler(optimizer, 10)
    path = tmp_path / "synthetic.pt"
    save_checkpoint(
        path,
        miniature,
        optimizer,
        scheduler,
        run_contract={"dataset_sha256": "c" * 64},
        sampler_state={"cursor": 0},
        step=0,
    )
    load_checkpoint(path, miniature, run_contract={"dataset_sha256": "c" * 64}, restore_rng=False)
    _tamper_checkpoint(path, field, production=False)
    with pytest.raises(ValueError, match="exact resume refused"):
        load_checkpoint(path, miniature, run_contract={"dataset_sha256": "c" * 64}, restore_rng=False)


@pytest.mark.parametrize("field", CORE_FIELDS)
def test_production_checkpoint_rejects_even_matching_checksum_valid_old_contract(tmp_path, miniature, field):
    optimizer = create_optimizer(miniature)
    scheduler = create_scheduler(optimizer, 10)
    path = tmp_path / "synthetic.pt"
    contract, runtime = {**artifact_contract(), "synthetic_unit": True}, {"world_size": 1, "device": "cpu"}
    save_production_checkpoint(
        path,
        miniature,
        optimizer,
        scheduler,
        contract=contract,
        trainer_state={"sampler": {"cursor": 0}},
        rank_rng=[rank_rng_state(torch.device("cpu"))],
        runtime=runtime,
        step=0,
    )
    assert read_production_checkpoint(path, contract=contract, runtime=runtime)["metadata"]["step"] == 0
    old = _tamper_checkpoint(path, field, production=True)
    for requested in (contract, old):
        with pytest.raises(ValueError, match="version contract differs"):
            read_production_checkpoint(path, contract=requested, runtime=runtime)


@pytest.mark.parametrize("field", CORE_FIELDS)
def test_frozen_release_rejects_checksum_valid_old_versions_before_weight_load(tmp_path, miniature, field):
    directory = tmp_path / "synthetic-release"
    record = save_frozen_release(
        directory,
        miniature,
        train_manifest_sha256="c" * 64,
        use_rule_features=True,
        training_contract={**artifact_contract(), "synthetic_unit": True},
    )
    loaded = FrozenRelease(directory, device="cpu")
    assert loaded.record["binding"] == record["binding"]
    # Only synthetic temporary metadata is changed. This tests rejection, not migration.
    record[field] = "old-or-foreign"
    record["training_contract"][field] = "old-or-foreign"
    record["release_sha256"] = content_hash({key: value for key, value in record.items() if key != "release_sha256"})
    (directory / "release.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="version contract differs"):
        FrozenRelease(directory, device="cpu")
