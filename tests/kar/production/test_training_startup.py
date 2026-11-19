"""Startup fault injection only: no forward, backward, or optimizer update is allowed."""

# ruff: noqa: E402 -- Torch is optional outside the ML test environment.
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from test_full_trainer import FullSyntheticDataset, _full_config

from riichienv.kar import budget as budget_module
from riichienv.kar.budget import BudgetLedger
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning import orchestration, trainer
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.distributed import DistributedStageError
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.learning.production_checkpoint import (
    advance_training_startup,
    create_training_directory,
    read_training_startup,
)


@pytest.fixture
def startup_driver(tmp_path, monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Startup diagnostics must not execute any training update")

    # Isolate filesystem/control flow from the separately tested authorization boundary.
    # No authorization document is manufactured; the update entry is unconditionally blocked.
    monkeypatch.setattr(trainer, "production_update", forbidden)
    monkeypatch.setattr(trainer, "require_parameter_updates", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "DecisionDataset", FullSyntheticDataset)
    tiny = ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="startup-diagnostic")
    monkeypatch.setattr(trainer, "KARTransformer", lambda config: KARTransformer(tiny))
    settings = _full_config("synthetic startup fault injection, no measured training evidence")
    output = tmp_path / "run"

    def run(*, resume=False, config=settings):
        return trainer.train_production(tmp_path / "data", output, config, resume=resume)

    return run, output, settings


def _fail_validation(*args, **kwargs):
    raise RuntimeError("synthetic initial validation interruption")


def test_initialization_failure_can_retry_same_directory_and_preserves_costs(startup_driver, monkeypatch):
    run, output, settings = startup_driver
    initializations = []

    def fail_initialization(*args, **kwargs):
        initializations.append(True)
        raise RuntimeError("synthetic model allocation failure")

    monkeypatch.setattr(trainer, "_model_components", fail_initialization)
    with pytest.raises(DistributedStageError, match="model/optimizer initialization"):
        run()
    assert not (output / "checkpoint.json").exists()
    ledger = BudgetLedger(output / "costs.jsonl", settings.budget, allow_create=False)
    ledger.charge(2.0, category="compute", reference="synthetic previous attempt cost")
    previous = ledger.path.read_bytes()
    with pytest.raises(DistributedStageError, match="model/optimizer initialization"):
        run(resume=True)
    assert len(initializations) == 2
    assert ledger.path.read_bytes().startswith(previous)
    assert BudgetLedger(ledger.path, settings.budget, allow_create=False).spent_usd == 2.0
    with pytest.raises(DistributedStageError, match="production startup"):
        run()
    assert len(initializations) == 2


def test_initialization_retry_readmits_full_work_against_existing_costs(startup_driver, monkeypatch):
    run, output, settings = startup_driver
    forecast = replace(settings.budget.forecast, resource_usd_per_hour=3600.0)
    settings = replace(settings, budget=replace(settings.budget, forecast=forecast))
    monkeypatch.setattr(budget_module, "time", SimpleNamespace(monotonic=lambda: 0.0))
    initializations = []

    def fail_initialization(*args, **kwargs):
        initializations.append(True)
        raise RuntimeError("synthetic allocation failure before any planned work")

    monkeypatch.setattr(trainer, "_model_components", fail_initialization)
    with pytest.raises(DistributedStageError, match="model/optimizer initialization"):
        run(config=settings)
    ledger = BudgetLedger(output / "costs.jsonl", settings.budget, allow_create=False)
    ledger.charge(99.95, category="compute", reference="synthetic prior initialization costs")
    previous = ledger.path.read_bytes()
    assert settings.budget.forecast.usd > settings.budget.budget_cap_usd - ledger.spent_usd
    with pytest.raises(DistributedStageError, match="production startup"):
        run(resume=True, config=settings)
    assert initializations == [True] and not (output / "checkpoint.json").exists()
    assert ledger.path.read_bytes().startswith(previous)


def test_step_zero_checkpoint_precedes_validation_and_is_used_on_retry(startup_driver, monkeypatch):
    run, output, settings = startup_driver
    validations = []

    def fail_validation(*args, **kwargs):
        pointer = json.loads((output / "checkpoint.json").read_text())
        state = json.loads((output / "startup.json").read_text())
        assert pointer["metadata"]["step"] == 0 and state["phase"] == "checkpoint_committed"
        validations.append(pointer["checkpoint_sha256"])
        _fail_validation()

    monkeypatch.setattr(trainer, "evaluate_complete", fail_validation)
    with pytest.raises(RuntimeError, match="initial validation interruption"):
        run()
    pointer = json.loads((output / "checkpoint.json").read_text())
    payload = torch.load(output / pointer["checkpoint_file"], weights_only=True)
    assert payload["trainer"]["step"] == 0 and payload["trainer"]["processed_examples"] == 0
    assert payload["trainer"]["validation_history"] == [] and payload["optimizer"]["state"] == {}
    original_read = trainer.read_production_checkpoint
    restored = []

    def read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        restored.append(result["metadata"]["step"])
        return result

    monkeypatch.setattr(trainer, "read_production_checkpoint", read)
    with pytest.raises(RuntimeError, match="initial validation interruption"):
        run(resume=True)
    assert restored == [0] and validations == [pointer["checkpoint_sha256"]] * 2
    assert (
        not BudgetLedger(output / "costs.jsonl", settings.budget, allow_create=False)
        .state_dict()
        .get("pending_compute")
    )


def test_first_checkpoint_write_failure_is_still_retriable(startup_driver, monkeypatch):
    run, output, _ = startup_driver
    original_save = trainer.save_production_checkpoint

    def interrupted_write(*args, **kwargs):
        raise OSError("synthetic first checkpoint write failure")

    monkeypatch.setattr(trainer, "save_production_checkpoint", interrupted_write)
    monkeypatch.setattr(trainer, "evaluate_complete", _fail_validation)
    with pytest.raises(DistributedStageError, match="checkpoint commit"):
        run()
    assert not (output / "checkpoint.json").exists()
    assert json.loads((output / "startup.json").read_text())["phase"] == "initializing"
    previous = (output / "costs.jsonl").read_bytes()
    monkeypatch.setattr(trainer, "save_production_checkpoint", original_save)
    with pytest.raises(RuntimeError, match="initial validation interruption"):
        run(resume=True)
    assert (output / "checkpoint.json").is_file()
    assert (output / "costs.jsonl").read_bytes().startswith(previous)


def test_startup_marker_failure_after_pointer_commit_recovers_without_reinitializing(startup_driver, monkeypatch):
    run, output, _ = startup_driver
    original_advance = trainer.advance_training_startup
    validation_calls = []

    def interrupted_marker(*args, **kwargs):
        if kwargs["phase"] == "checkpoint_committed":
            raise OSError("synthetic startup marker write failure")
        return original_advance(*args, **kwargs)

    def validation(*args, **kwargs):
        validation_calls.append(True)
        _fail_validation()

    monkeypatch.setattr(trainer, "advance_training_startup", interrupted_marker)
    monkeypatch.setattr(trainer, "evaluate_complete", validation)
    with pytest.raises(DistributedStageError, match="checkpoint commit"):
        run()
    assert not validation_calls and (output / "checkpoint.json").is_file()
    original_pointer = (output / "checkpoint.json").read_bytes()
    monkeypatch.setattr(trainer, "advance_training_startup", original_advance)
    with pytest.raises(RuntimeError, match="initial validation interruption"):
        run(resume=True)
    assert validation_calls == [True]
    assert (output / "checkpoint.json").read_bytes() == original_pointer
    assert json.loads((output / "startup.json").read_text())["phase"] == "checkpoint_committed"


def test_missing_committed_checkpoint_never_restarts_from_initial_weights(startup_driver, monkeypatch):
    run, output, _ = startup_driver
    monkeypatch.setattr(trainer, "evaluate_complete", _fail_validation)
    with pytest.raises(RuntimeError, match="initial validation interruption"):
        run()
    (output / "checkpoint.json").unlink()

    def forbidden_initialization(*args, **kwargs):
        raise AssertionError("Lost committed weights cannot trigger a fresh initialization")

    monkeypatch.setattr(trainer, "_model_components", forbidden_initialization)
    with pytest.raises(DistributedStageError, match="production startup"):
        run(resume=True)


def test_atomic_directory_publication_never_leaves_an_unidentified_output(tmp_path, monkeypatch):
    from riichienv.kar.learning import production_checkpoint as module  # noqa: PLC0415

    output = tmp_path / "run"

    def fail_record(*args, **kwargs):
        raise OSError("synthetic startup identity write failure")

    original = module._write_startup
    monkeypatch.setattr(module, "_write_startup", fail_record)
    with pytest.raises(OSError, match="identity write failure"):
        create_training_directory(output, identity={"recipe": "same"}, ledger_path=output / "costs.jsonl")
    assert not output.exists()
    monkeypatch.setattr(module, "_write_startup", original)
    create_training_directory(output, identity={"recipe": "same"}, ledger_path=output / "costs.jsonl")
    assert (output / "startup.json").is_file()


@pytest.mark.parametrize("damage", ["missing", "truncated", "modified"])
def test_preparing_run_cannot_replace_an_existing_shared_phase_ledger(tmp_path, damage):
    output, ledger_path = tmp_path / "feature-off", tmp_path / "M1-costs.jsonl"
    original = b'{"existing phase receipt": 1}\n'
    ledger_path.write_bytes(original)
    create_training_directory(output, identity={"recipe": "same"}, ledger_path=ledger_path)
    if damage == "missing":
        ledger_path.unlink()
    else:
        ledger_path.write_bytes(original[:-2] if damage == "truncated" else b"x" * len(original))
    with pytest.raises((ValueError, FileNotFoundError)):
        read_training_startup(output, identity={"recipe": "same"}, ledger_path=ledger_path, checkpoint_exists=False)
    assert json.loads((output / "startup.json").read_text())["phase"] == "preparing"


@pytest.mark.parametrize("evidence", ["updates-attempt-0000.jsonl", "checkpoint.g000000000001-aaaaaaaaaaaa.pt"])
def test_update_evidence_blocks_pre_checkpoint_restart(tmp_path, evidence):
    output = tmp_path / "run"
    ledger_path = output / "costs.jsonl"
    create_training_directory(output, identity={"recipe": "same"}, ledger_path=ledger_path)
    (output / evidence).write_text("synthetic update evidence")
    with pytest.raises(ValueError, match="forbids"):
        read_training_startup(output, identity={"recipe": "same"}, ledger_path=ledger_path, checkpoint_exists=False)


def test_startup_identity_changes_and_missing_legacy_pointer_are_rejected(tmp_path):
    output = tmp_path / "run"
    ledger_path = output / "costs.jsonl"
    create_training_directory(output, identity={"recipe": "original"}, ledger_path=ledger_path)
    with pytest.raises(ValueError, match="identity/phase differs"):
        read_training_startup(output, identity={"recipe": "changed"}, ledger_path=ledger_path, checkpoint_exists=False)
    (output / "startup.json").unlink()
    assert read_training_startup(output, identity={}, ledger_path=ledger_path, checkpoint_exists=True) is None
    with pytest.raises(ValueError, match="never restart silently"):
        read_training_startup(output, identity={}, ledger_path=ledger_path, checkpoint_exists=False)


def test_m1_resume_forwards_existing_initialization_and_starts_untouched_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestration, "require_parameter_updates", lambda *args, **kwargs: None)
    settings = _full_config("synthetic orchestration routing, no training")
    off = replace(settings, use_rule_features=False)
    data, output = tmp_path / "data", tmp_path / "m1"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"manifest_sha256": "f" * 64}))
    (output / "feature-on").mkdir(parents=True)
    requests = []

    def no_training(dataset, directory, config, **kwargs):
        requests.append((directory.name, kwargs["resume"]))
        return {"initial_weights_sha256": "a" * 64, "epoch_order_sha256": {}, "processed_examples": 0, "step": 0}

    monkeypatch.setattr(orchestration, "train_production", no_training)
    orchestration.run_m1_ablation(data, output, settings, off, resume=True)
    assert requests == [("feature-on", True), ("feature-off", False)]


def test_startup_phase_cannot_rewind_after_checkpoint_commit(tmp_path):
    output = tmp_path / "run"
    create_training_directory(output, identity={}, ledger_path=output / "costs.jsonl")
    record = json.loads((output / "startup.json").read_text())
    ready = advance_training_startup(output, record, phase="initializing", budget_ledger={"ledger_head": "receipt"})
    (output / "checkpoint.json").write_text("{}")
    committed = advance_training_startup(output, ready, phase="checkpoint_committed", budget_ledger={})
    with pytest.raises(ValueError, match="cannot be rewound"):
        advance_training_startup(output, committed, phase="initializing", budget_ledger={})
    unsigned = {key: value for key, value in committed.items() if key != "startup_sha256"}
    assert committed["startup_sha256"] == content_hash(unsigned)
