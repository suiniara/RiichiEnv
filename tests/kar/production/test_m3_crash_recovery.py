"""Exercise recovery using metadata only; never load models or play a match."""

import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from riichienv.kar.authorization import update_implementation_sha256
from riichienv.kar.budget import BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning import simulation


@pytest.fixture
def config(tmp_path):
    digest = "a" * 64
    return SimpleNamespace(
        output=tmp_path / "run",
        test_registry=tmp_path / "test.json",
        budget=BudgetPlan(
            "M3",
            250.0,
            0.0,
            CostForecast("M3", 20000, 1000.0, 1.0, 0.0, "synthetic recovery metadata", digest),
        ),
        paired_seed_manifest={
            "paired_seed_manifest_sha256": digest,
            "subject_release_sha256": "b" * 64,
            "partition": "test",
            "blocks": [
                {"pair_index": index, "block_sha256": f"block-{index}", "subject_seat": index} for index in range(2)
            ],
        },
        # This fixture tests durable ordering after production admission. It
        # never represents a validated full 10,000-pair evaluation plan.
        validate=lambda: None,
    )


def _authorization(config):
    record = {
        "version": "explicit_frozen_evaluation_authorization_v1",
        "explicit_user_approval": True,
        "approval_reference": "synthetic metadata-only crash-window test",
        "allowed_operations": ["run_m3"],
        "plan_sha256": config.paired_seed_manifest["paired_seed_manifest_sha256"],
        "source_sha256": update_implementation_sha256(),
    }
    return {**record, "authorization_sha256": content_hash(record)}


def _forbid_computation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Recovery rejection must happen before model/agent/match work")

    monkeypatch.setattr(simulation, "FrozenRelease", forbidden)
    monkeypatch.setattr(simulation, "_agents", forbidden)
    monkeypatch.setattr(simulation, "run_match", forbidden)
    monkeypatch.setattr(simulation.torch, "set_num_threads", forbidden)


@pytest.mark.parametrize("after_publication", [False, True])
def test_registration_crash_always_leaves_its_original_ledger_recoverable(config, monkeypatch, after_publication):
    publish = simulation._publish_record
    original_ledger = []

    def crash(path, record):
        assert path == config.test_registry
        ledger = BudgetLedger(config.output / "costs.jsonl", config.budget, allow_create=False)
        assert not ledger.pending_compute and ledger.spent_usd == 0.0
        original_ledger.append(ledger.path.read_bytes())
        if after_publication:
            publish(path, record)
        raise InterruptedError("process stopped at registration publication")

    with monkeypatch.context() as scoped:
        scoped.setattr(simulation, "_publish_record", crash)
        with pytest.raises(InterruptedError, match="registration publication"):
            simulation.register_m3_run(config)
    assert config.test_registry.exists() == after_publication
    assert simulation.register_m3_run(config) == after_publication
    assert simulation.register_m3_run(config) is True
    restored = simulation.open_m3_ledger(config, update_implementation_sha256(), allow_create=False)
    assert restored.path.read_bytes() == original_ledger[0]
    assert restored.spent_usd == 0.0 and not restored.pending_compute


def test_interrupt_during_ledger_initialization_does_not_publish_registration(config, monkeypatch):
    append = BudgetLedger._append

    def interrupt_after_identity(ledger, kind, **fields):
        result = append(ledger, kind, **fields)
        if kind == "identity":
            raise InterruptedError("process stopped after durable ledger identity")
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(BudgetLedger, "_append", interrupt_after_identity)
        with pytest.raises(InterruptedError, match="durable ledger identity"):
            simulation.register_m3_run(config)
    assert not config.test_registry.exists()
    original_identity = (config.output / "costs.jsonl").read_bytes()
    assert simulation.register_m3_run(config) is False
    restored = simulation.open_m3_ledger(config, update_implementation_sha256(), allow_create=False)
    assert restored.path.read_bytes().startswith(original_identity)
    assert [entry["kind"] for entry in restored.entries] == ["identity", "fixed_reservation"]


@pytest.mark.parametrize("charged", [False, True])
def test_registered_run_still_rejects_missing_history_even_without_receipts(config, monkeypatch, charged):
    _forbid_computation(monkeypatch)
    assert simulation.register_m3_run(config) is False
    path = config.output / "costs.jsonl"
    if charged:
        BudgetLedger(path, config.budget, allow_create=False).charge(
            5.0, category="compute", reference="synthetic prior work, not a real charge"
        )
    registration = config.test_registry.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="required phase ledger is missing"):
        simulation.run_m3(config, authorization=_authorization(config))
    assert not path.exists() and config.test_registry.read_bytes() == registration


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_later_orphan_replay_rejects_every_retry_before_any_computation(config, monkeypatch, arm):
    _forbid_computation(monkeypatch)
    assert simulation.register_m3_run(config) is False
    path = config.output / "costs.jsonl"
    ledger = BudgetLedger(path, config.budget, allow_create=False)
    ledger.charge(5.0, category="compute", reference="synthetic prior completed work")
    original_ledger = path.read_bytes()
    replays = config.output / "replays"
    replays.mkdir()
    # Pair 0 is missing too: checking only immediately before its run_match
    # would still do needless work before discovering this later pair's orphan.
    orphan = replays / f"pair-00001-{arm}.json.gz"
    original_replay = b"synthetic retained orphan, not a real match"
    orphan.write_bytes(original_replay)
    for _ in range(2):
        with pytest.raises(ValueError, match="Orphan replay without receipt"):
            simulation.run_m3(config, authorization=_authorization(config))
    assert path.read_bytes() == original_ledger
    assert orphan.read_bytes() == original_replay
    assert not tuple((config.output / "receipts").glob("pair-*.json"))
    restored = BudgetLedger(path, config.budget, allow_create=False)
    assert restored.spent_usd == 5.0 and not restored.pending_compute


@pytest.mark.parametrize("defect", ["missing", "changed", "wrong_block"])
def test_cached_replay_is_checked_before_model_load(config, monkeypatch, defect):
    _forbid_computation(monkeypatch)
    assert simulation.register_m3_run(config) is False
    source = update_implementation_sha256()
    ledger = simulation.open_m3_ledger(config, source, allow_create=False)
    receipts, replays = config.output / "receipts", config.output / "replays"
    receipts.mkdir()
    replays.mkdir()
    replay = replays / "pair-00000-baseline.json.gz"
    data = b"synthetic replay metadata"
    if defect != "missing":
        replay.write_bytes(data if defect != "changed" else data + b"tampered")
    simulation._atomic_record(
        receipts / "pair-00000-baseline.json",
        {
            "plan_sha256": config.paired_seed_manifest["paired_seed_manifest_sha256"],
            "source_sha256": source,
            "block_sha256": "wrong" if defect == "wrong_block" else "block-0",
            "arm": "baseline",
            "subject_seat": 0,
            "replay_file": replay.name,
            "replay_sha256": hashlib.sha256(data).hexdigest(),
            "budget_ledger_head": ledger.head,
            "budget_ledger_entries": len(ledger.entries),
        },
    )
    message = {"missing": "replay is missing", "changed": "replay changed", "wrong_block": "another seed/seat/arm"}
    with pytest.raises(ValueError, match=message[defect]):
        simulation.run_m3(config, authorization=_authorization(config))
    assert (
        json.loads((receipts / "pair-00000-baseline.json").read_text())["replay_sha256"]
        == hashlib.sha256(data).hexdigest()
    )
