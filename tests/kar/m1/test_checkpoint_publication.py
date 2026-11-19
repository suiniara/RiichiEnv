"""Checkpoint publication faults without forward/backward or parameter updates."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from riichienv.kar.learning.checkpoint import load_checkpoint, save_checkpoint
from riichienv.kar.learning.train import create_optimizer, create_scheduler


@pytest.fixture
def checkpoint_run(tmp_path, miniature):
    path = tmp_path / "checkpoint.pt"
    optimizer = create_optimizer(miniature)
    scheduler = create_scheduler(optimizer, 10)
    contract = {"scope": "checkpoint_publication_fixture"}

    def save(step):
        return save_checkpoint(
            path,
            miniature,
            optimizer,
            scheduler,
            run_contract=contract,
            sampler_state={"cursor": step},
            step=step,
        )

    def restore():
        return load_checkpoint(path, miniature, optimizer, scheduler, run_contract=contract, restore_rng=False)

    yield path, save, restore
    assert not optimizer.state
    assert all(parameter.grad is None for parameter in miniature.parameters())


def test_failed_pointer_publication_preserves_committed_checkpoint(checkpoint_run, monkeypatch):
    path, save, restore = checkpoint_run
    first = save(0)
    previous = path.parent / first.get("checkpoint_file", path.name)
    previous_bytes = previous.read_bytes()
    pointer = path.with_suffix(".json")
    previous_pointer = pointer.read_bytes()
    replace = Path.replace

    def interrupt_pointer(source, target):
        if Path(target) == pointer:
            raise OSError("injected pointer publication failure")
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt_pointer)
        with pytest.raises(OSError, match="pointer publication"):
            save(1)

    assert pointer.read_bytes() == previous_pointer
    assert previous.read_bytes() == previous_bytes
    assert restore()["step"] == 0
    assert restore()["sampler"] == {"cursor": 0}


def test_failed_payload_write_preserves_committed_checkpoint(checkpoint_run, monkeypatch):
    path, save, restore = checkpoint_run
    save(0)
    pointer = path.with_suffix(".json")
    previous_pointer = pointer.read_bytes()

    def interrupt_payload(_payload, target):
        if hasattr(target, "write"):
            target.write(b"incomplete")
        else:
            Path(target).write_bytes(b"incomplete")
        raise OSError("injected payload write failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupt_payload)
        with pytest.raises(OSError, match="payload write"):
            save(1)

    assert pointer.read_bytes() == previous_pointer
    assert restore()["step"] == 0


def test_successful_publication_retains_current_and_previous_generation(checkpoint_run):
    path, save, restore = checkpoint_run
    first, second = save(0), save(1)
    first_path = path.parent / first["checkpoint_file"]
    second_path = path.parent / second["checkpoint_file"]
    assert first_path != second_path and first_path.is_file() and second_path.is_file()
    assert second["previous_generation"] == {
        "file": first_path.name,
        "sha256": hashlib.sha256(first_path.read_bytes()).hexdigest(),
    }
    assert restore()["step"] == 1
    third = save(2)
    assert not first_path.exists()
    assert second_path.is_file() and (path.parent / third["checkpoint_file"]).is_file()
    assert restore()["sampler"] == {"cursor": 2}


def test_checkpoint_pointer_cannot_select_an_unowned_file(checkpoint_run):
    path, save, restore = checkpoint_run
    pointer = path.with_suffix(".json")
    record = save(0)
    record["checkpoint_file"] = "../foreign.pt"
    pointer.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="generation namespace"):
        restore()


def test_overwrite_refuses_a_damaged_committed_generation(checkpoint_run):
    path, save, _restore = checkpoint_run
    first = save(0)
    pointer = path.with_suffix(".json")
    committed_pointer = pointer.read_bytes()
    generation = path.parent / first["checkpoint_file"]
    generation.write_bytes(b"damaged committed generation")
    with pytest.raises(ValueError, match="Previous committed checkpoint"):
        save(1)
    assert pointer.read_bytes() == committed_pointer
