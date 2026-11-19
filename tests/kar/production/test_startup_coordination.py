"""Real process/file handshakes only: no process group, model, or training update."""

# ruff: noqa: E402 -- Torch is optional outside the ML test environment.
import json
import multiprocessing
import os
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("torch")

from riichienv.kar.learning.distributed import StartupCoordinator

STAGE = "production startup accounting"


def _launch_environment(rank, run_id):
    return {
        "RANK": str(rank),
        "WORLD_SIZE": "2",
        "LOCAL_RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29400",  # Identity only; these tests never create a socket.
        "TORCHELASTIC_RUN_ID": run_id,
        "TORCHELASTIC_RESTART_COUNT": "0",
    }


def _worker(output, run_id, rank, ready, reports, start, admit_primary, failing_rank, timeout_seconds):
    os.environ.update(_launch_environment(rank, run_id))
    try:
        coordinator = StartupCoordinator.open(Path(output), timeout_seconds=timeout_seconds)
        ready.put({"rank": rank, "nonce": coordinator.nonce, "directory": str(coordinator.directory)})
        if not start.wait(15):
            raise TimeoutError("Test did not start the worker")
        if rank == 0:
            if not admit_primary.wait(15):
                raise TimeoutError("Test did not release the primary")
            if failing_rank != rank:
                # Model the primary's durable attempt before its accounting-stage report.
                with (Path(output).parent / "synthetic-attempt.json").open("x", encoding="utf-8") as stream:
                    json.dump({"synthetic_control_only": True}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
        error = RuntimeError("synthetic startup admission failure") if failing_rank == rank else None
        coordinator.coordinate(error, STAGE)
        (Path(output).parent / f"expensive-rank-{rank}.marker").write_text("entered", encoding="utf-8")
        reports.put({"rank": rank, "status": "released"})
    except Exception as error:
        reports.put({"rank": rank, "status": "rejected", "error": f"{type(error).__name__}: {error}"})


@pytest.fixture
def process_factory():
    context = multiprocessing.get_context("spawn")
    processes = []
    queues = []
    events = []

    def launch(output, run_id, ranks, *, hold_primary=False, failing_rank=None, timeout_seconds=3.0):
        ready, reports = context.Queue(), context.Queue()
        start, admit_primary = context.Event(), context.Event()
        queues.extend((ready, reports))
        events.extend((start, admit_primary))
        if not hold_primary:
            admit_primary.set()
        active = []
        for rank in ranks:
            process = context.Process(
                target=_worker,
                args=(str(output), run_id, rank, ready, reports, start, admit_primary, failing_rank, timeout_seconds),
            )
            process.start()
            processes.append(process)
            active.append(process)
        participants = {item["rank"]: item for item in (ready.get(timeout=20) for _ in ranks)}
        start.set()
        return participants, reports, admit_primary, active

    yield launch
    for event in events:
        event.set()
    for process in processes:
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        process.close()
    for queue in queues:
        queue.close()
        queue.join_thread()


def _wait_for_file(path, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            pytest.fail(f"Worker did not publish {path.name}")
        time.sleep(0.01)


def _manual_coordinator(output, rank, run_id, monkeypatch):
    for key, value in _launch_environment(rank, run_id).items():
        monkeypatch.setenv(key, value)
    return StartupCoordinator.open(output, timeout_seconds=0.3)


def _collect(reports, processes):
    result = [reports.get(timeout=8) for _ in processes]
    for process in processes:
        process.join(timeout=8)
        assert process.exitcode == 0
    return result


def test_workers_wait_for_primary_accounting_release(tmp_path, process_factory):
    output = tmp_path / "run"
    participants, reports, admit_primary, processes = process_factory(
        output, uuid.uuid4().hex, (0, 1), hold_primary=True
    )
    directory = Path(participants[0]["directory"])
    _wait_for_file(directory / "stage-0-rank-1.json")
    assert not (directory / "stage-0-release.json").exists()
    assert not (tmp_path / "synthetic-attempt.json").exists()
    assert not list(tmp_path.glob("expensive-*.marker"))
    admit_primary.set()
    assert all(result["status"] == "released" for result in _collect(reports, processes))
    release = json.loads((directory / "stage-0-release.json").read_text(encoding="utf-8"))
    assert release["nonces"] == [participants[rank]["nonce"] for rank in (0, 1)]
    assert (tmp_path / "synthetic-attempt.json").is_file()
    assert len(list(tmp_path.glob("expensive-*.marker"))) == 2


@pytest.mark.parametrize("failing_rank", (0, 1))
def test_startup_failure_reaches_both_processes_before_work(tmp_path, process_factory, failing_rank):
    participants, reports, _, processes = process_factory(
        tmp_path / "run", uuid.uuid4().hex, (0, 1), failing_rank=failing_rank
    )
    results = _collect(reports, processes)
    assert all(result["status"] == "rejected" for result in results)
    assert all("synthetic startup admission failure" in result["error"] for result in results)
    assert not (Path(participants[0]["directory"]) / "stage-0-release.json").exists()
    assert not list(tmp_path.glob("expensive-*.marker"))


@pytest.mark.parametrize("stale_release", (False, True))
def test_worker_cannot_use_half_finished_primary_or_old_release(tmp_path, monkeypatch, process_factory, stale_release):
    output, run_id = tmp_path / "run", uuid.uuid4().hex
    primary = _manual_coordinator(output, 0, run_id, monkeypatch)
    primary._publish(
        "stage-0-rank-0.json",
        {"nonce": primary.nonce, "primary_nonce": primary.nonce, "stage": STAGE, "error": None},
    )
    if stale_release:
        primary._publish("stage-0-release.json", {"stage": STAGE, "nonces": [primary.nonce, "old-peer-nonce"]})
    _, reports, _, processes = process_factory(output, run_id, (1,), timeout_seconds=0.3)
    result = _collect(reports, processes)[0]
    assert result["status"] == "rejected"
    assert ("another launch" if stale_release else "Timed out") in result["error"]
    assert not list(tmp_path.glob("expensive-*.marker"))


def test_new_primary_cannot_admit_a_stale_secondary_report(tmp_path, monkeypatch, process_factory):
    output, run_id = tmp_path / "run", uuid.uuid4().hex
    stale_peer = _manual_coordinator(output, 1, run_id, monkeypatch)
    stale_peer._publish(
        "stage-0-rank-1.json",
        {"nonce": stale_peer.nonce, "primary_nonce": "old-primary-nonce", "stage": STAGE, "error": None},
    )
    # Only the secondary's old receipts survive. A fresh primary must not treat
    # those successful reports as participation in its own accounting attempt.
    _, reports, _, processes = process_factory(output, run_id, (0,), timeout_seconds=0.3)
    result = _collect(reports, processes)[0]
    assert result["status"] == "rejected"
    assert not (stale_peer.directory / "stage-0-release.json").exists()
    assert not list(tmp_path.glob("expensive-*.marker"))


def test_duplicate_launch_receipt_is_rejected_without_replacement(tmp_path, monkeypatch):
    output, run_id = tmp_path / "run", uuid.uuid4().hex
    first = _manual_coordinator(output, 0, run_id, monkeypatch)
    receipt = first.directory / "rank-0.json"
    original = receipt.read_bytes()
    with pytest.raises(ValueError, match="fresh launch ID"):
        StartupCoordinator.open(output, timeout_seconds=0.3)
    assert receipt.read_bytes() == original


def test_unset_torchrun_id_uses_explicit_unique_launcher_id(tmp_path, monkeypatch):
    for key, value in _launch_environment(0, "none").items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("KAR_DISTRIBUTED_RUN_ID", raising=False)
    with pytest.raises(ValueError, match="unique"):
        StartupCoordinator.open(tmp_path / "run", timeout_seconds=0.3)
    monkeypatch.setenv("KAR_DISTRIBUTED_RUN_ID", uuid.uuid4().hex)
    coordinator = StartupCoordinator.open(tmp_path / "run", timeout_seconds=0.3)
    assert coordinator.directory is not None and (coordinator.directory / "rank-0.json").is_file()
