import copy
import hashlib
import inspect
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from riichienv import convert
from riichienv.kar.actions import action_spec, tile136_to_tile37
from riichienv.kar.contracts import MJSOUL_RULESET, content_hash
from riichienv.kar.decision import ReplayDecision
from riichienv.kar.learning.batch import TOKEN_DTYPE
from riichienv.kar.learning.dataset import DecisionDataset, collate_examples
from riichienv.kar.learning.export import SQL_SCHEMA, GameWriter, export_dataset, learning_contract
from riichienv.kar.mjsoul import protocol_identity
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.schema import decode_token


def test_candidate_branch_generation_is_enabled_by_default():
    assert inspect.signature(export_dataset).parameters["branch_probe_decisions"].default == 8
    assert inspect.signature(GameWriter).parameters["branch_probe_decisions"].default == 8
    assert inspect.signature(GameWriter).parameters["split"].default == "train"


def _private_dataset_path(workspace, explicit=None):
    data_root = workspace / "kar-m1-data"
    candidates = (
        (Path(explicit).expanduser(),)
        if explicit is not None
        else (data_root / "cpu-current-contract", data_root / "cpu-probe-v2")
    )
    for path in candidates:
        if explicit is not None or path.exists():
            if not (path / "manifest.json").is_file():
                raise ValueError(f"Selected private M1 fixture is missing or incomplete: {path}")
            # Selection is not compatibility: DecisionDataset still checks every
            # manifest/database contract, including explicitly selected legacy data.
            return path
    return None


def _current_private_fixture(path, *, explicit=False):
    """Never relabel legacy private data or hide corrupt/current/explicit selections.

    Positive integration assertions remain unchanged and require a real current
    export. The default archived fixture may only be marked NOT EXECUTED after
    proving strict rejection and intact manifest/database bytes.
    """
    try:
        probe = DecisionDataset(path, "train")
    except ValueError as error:
        if str(error) != "Dataset schema/rules/feature/loss contract differs" or explicit:
            raise
        document = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        scope = document["scope"]
        if scope.get("dataset_version") != learning_contract()["dataset_version"] or scope.get(
            "input_pipeline_version"
        ) not in {None, "legal_set_pool_private_self_pass_v1"}:
            raise
        database = path / "decisions.sqlite3"
        if not database.is_file() or hashlib.sha256(database.read_bytes()).hexdigest() != document["database_sha256"]:
            raise ValueError("Selected private fixture database is missing or corrupt") from error
        pytest.skip("Existing private export uses a legacy input/rule contract; current V4 rebuild is not authorized")
    else:
        probe._db().close()
    return path


@pytest.fixture(scope="module")
def real_dataset_dir():
    explicit = os.environ.get("KAR_M1_TEST_DATASET_DIR")
    path = _private_dataset_path(Path(__file__).parents[4], explicit)
    if path is None:
        pytest.skip("Private M1 CPU data has not been exported")
    return _current_private_fixture(path, explicit=explicit is not None)


def test_private_fixture_selection_prefers_current_without_bypassing_validation(tmp_path):
    root = tmp_path / "kar-m1-data"
    legacy = root / "cpu-probe-v2"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")
    assert _private_dataset_path(tmp_path) == legacy
    with pytest.raises(ValueError, match="incomplete|changed"):
        DecisionDataset(legacy, "train")
    current = root / "cpu-current-contract"
    current.mkdir()
    (current / "manifest.json").write_text("{}", encoding="utf-8")
    assert _private_dataset_path(tmp_path) == current
    assert _private_dataset_path(tmp_path, explicit=str(legacy)) == legacy
    with pytest.raises(ValueError, match="incomplete|changed"):
        DecisionDataset(current, "train")


@pytest.mark.parametrize("selection", ["explicit-missing", "incomplete-current", "incomplete-legacy"])
def test_missing_or_incomplete_selected_private_fixture_is_not_skipped(tmp_path, selection):
    explicit = str(tmp_path / "explicit") if selection == "explicit-missing" else None
    if explicit is None:
        name = "cpu-current-contract" if selection == "incomplete-current" else "cpu-probe-v2"
        (tmp_path / "kar-m1-data" / name).mkdir(parents=True)
    with pytest.raises(ValueError, match="missing or incomplete"):
        _private_dataset_path(tmp_path, explicit=explicit)


def test_private_fixture_selection_reports_absence_only_when_no_assets_exist(tmp_path):
    assert _private_dataset_path(tmp_path) is None


@pytest.mark.parametrize(
    "case", ["legacy-default", "legacy-explicit", "corrupt-db", "corrupt-manifest", "current-corrupt"]
)
def test_only_intact_default_legacy_fixture_can_report_rebuild_not_executed(tmp_path, case):
    database = tmp_path / "decisions.sqlite3"
    database.write_bytes(b"synthetic legacy payload; never accepted as a current SQLite export")
    scope = {
        **learning_contract(),
        **protocol_identity("a" * 64, "b" * 64),
        "input_pipeline_version": "legal_set_pool_private_self_pass_v1",
    }
    if case == "current-corrupt":
        scope["input_pipeline_version"] = learning_contract()["input_pipeline_version"]
        scope["response_mask_version"] = "corrupt-current-mask"
    unsigned = {
        "status": "PASS",
        "complete": True,
        "scope": scope,
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
    }
    document = {**unsigned, "manifest_sha256": content_hash(unsigned)}
    if case == "corrupt-manifest":
        document["unexpected"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    if case == "corrupt-db":
        database.write_bytes(b"corrupted bytes")
    if case == "legacy-default":
        with pytest.raises(pytest.skip.Exception, match="legacy input/rule contract"):
            _current_private_fixture(tmp_path)
    else:
        with pytest.raises(ValueError):
            _current_private_fixture(tmp_path, explicit=case == "legacy-explicit")


def test_binary_vectorized_dtype_matches_frozen_schema(make_sample):
    sample = make_sample()
    rows = np.frombuffer(sample["tokens"], dtype=TOKEN_DTYPE)
    assert TOKEN_DTYPE.itemsize == 139
    for index, row in enumerate(rows):
        decoded = decode_token(sample["tokens"][139 * index : 139 * (index + 1)])
        for key in TOKEN_DTYPE.names:
            actual = row[key].tolist() if isinstance(row[key], np.ndarray) else row[key].item()
            assert actual == decoded[key]


def test_exported_real_splits_and_proportions_are_disjoint(real_dataset_dir):
    train = DecisionDataset(real_dataset_dir, "train")
    validation = DecisionDataset(real_dataset_dir, "validation")
    train_ids = {row["game_id"] for row in train.manifest["games"] if row["split"] == "train"}
    validation_ids = {row["game_id"] for row in validation.manifest["games"] if row["split"] == "validation"}
    assert len(train_ids) == 6 and len(validation_ids) == 2 and not train_ids.intersection(validation_ids)
    assert train.manifest["counts"]["test"]["games"] == 0
    totals = train.manifest["counts"]["train"]
    assert train.manifest["rho_E"] == totals["response_valid"] / totals["examples"]
    assert train.manifest["rho_V"] == totals["value_valid"] / totals["examples"]
    assert len(train) == totals["examples"] and len(validation) > 0


def test_real_response_labels_are_in_masks_and_passes_are_separate(real_dataset_dir):
    data = DecisionDataset(real_dataset_dir, "train")
    responses = values = passes = 0
    for index in range(len(data)):
        row = data[index]
        assert row["action_target"] in row["candidates"]
        active = row["response_applicable"] and row["response_observed"]
        if active:
            responses += 1
            assert 0 <= row["response_target"] < 21 and row["response_allowed"][row["response_target"]]
            assert row["value_valid"] and 0 <= row["final_rank"] < 4
        else:
            assert row["response_target"] == row["final_rank"] == -1 and row["value_valid"] is False
        values += row["value_valid"]
        if row["action_provenance"] == "inferred_unique_next_draw_pass":
            passes += 1
            assert row["action_target"] == 221 and row["response_applicable"] is False
    assert passes > 0
    assert responses == data.manifest["counts"]["train"]["response_valid"]
    assert values == data.manifest["counts"]["train"]["value_valid"]


def test_target_changes_never_change_model_tensors(real_dataset_dir):
    data = DecisionDataset(real_dataset_dir, "train")
    original = data[0]
    changed = copy.deepcopy(original)
    changed["response_target"] = 20
    changed["final_rank"] = 3
    left, right = collate_examples([original]), collate_examples([changed])
    assert set(left.inputs.fields) == set(TOKEN_DTYPE.names)
    for key in left.inputs.fields:
        torch.testing.assert_close(left.inputs.fields[key], right.inputs.fields[key], rtol=0, atol=0)


def test_resume_rejects_unfinished_dataset(real_dataset_dir, tmp_path):
    manifest = json.loads((real_dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["complete"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete|changed"):
        DecisionDataset(tmp_path, "train")


def test_discard_export_uses_the_actual_uppercase_action_protocol(make_sample):
    env = MJSOUL_RULESET.create_env(seed=42)
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    for event in env.mjai_log:
        ledger.apply_event(event)
    sample = make_sample()
    action = sample["action_target"]
    spec = action_spec(action)
    tile = convert.tid_to_mjai(next(tile for tile in ledger.snapshot().hand if tile136_to_tile37(tile) == spec.tile37))
    events = list(env.mjai_log) + [
        {"type": "dahai", "actor": 0, "pai": tile, "tsumogiri": spec.tsumogiri},
        {"type": "tsumo", "actor": 1, "pai": "1m"},
    ]
    game = SimpleNamespace(events=events, game_id="synthetic-export-contract")
    db = sqlite3.connect(":memory:")
    db.executescript(SQL_SCHEMA)
    writer = GameWriter(db, game)
    writer(ReplayDecision(ledger.snapshot(), sample["prefix"], sample["sequence"], action, len(events) - 2))
    row = db.execute("SELECT response_applicable,response_observed,response_target FROM samples").fetchone()
    assert row == (1, 1, 0)
