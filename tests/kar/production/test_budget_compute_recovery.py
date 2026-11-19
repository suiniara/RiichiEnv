"""CPU metadata-only recovery tests: no models, authorization, or parameter updates."""

from __future__ import annotations

import json
import os
from multiprocessing import get_context
from pathlib import Path

import pytest

from riichienv.kar import budget as module
from riichienv.kar.budget import BudgetExceededError, BudgetLedger, BudgetPlan, CostForecast, PendingComputeError
from riichienv.kar.contracts import content_hash


def _plan(*, fixed=0.0, rate=3600.0, cap=100.0):
    forecast = CostForecast(
        "M1", 10, 1.0, rate, fixed, "synthetic clock/receipts, not production measurements", "a" * 64, 1.0
    )
    return BudgetPlan("M1", cap, 0.0, forecast)


@pytest.fixture
def clock(monkeypatch):
    value = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: value[0])
    return value


def _crash_worker(path, seconds, partial_settlement):
    clock = [0.0]
    module.time.monotonic = lambda: clock[0]
    ledger = BudgetLedger(Path(path), _plan())
    ledger.reserve_next(seconds)
    if partial_settlement:
        clock[0] = 3.0
        ledger.settle_elapsed("synthetic completed part")
    Path(path + ".anchor.json").write_text(json.dumps(ledger.state_dict()), encoding="utf-8")
    clock[0] = 7.0
    os._exit(23)  # Cannot run Python exception/finally settlement.


def test_reservations_coalesce_and_settlements_keep_one_live_attempt(tmp_path, clock):
    ledger = BudgetLedger(tmp_path / "costs.jsonl", _plan(fixed=20.0))
    ledger.reserve_next(10.0)
    attempt = ledger.pending_compute[0]["attempt_id"]
    count = len(ledger.entries)
    ledger.reserve_next(5.0)
    ledger.reserve_next(10.0)
    assert len(ledger.entries) == count
    assert ledger.spent_usd == 0.0 and ledger.reserved_compute_usd == 10.0
    assert ledger.reserved_fixed_usd == 20.0 and ledger.committed_usd == 30.0
    clock[0] = 3.0
    ledger.settle_elapsed("part-1")
    assert ledger.spent_usd == 3.0 and ledger.reserved_compute_usd == 7.0
    assert ledger.pending_compute[0]["attempt_id"] == attempt
    assert ledger.pending_compute[0]["settled_usd"] == 3.0
    assert ledger.committed_usd == 30.0
    # Enlarge the required remaining allowance, without summing repeated requests.
    ledger.reserve_next(10.0)
    ledger.reserve_next(10.0)
    assert ledger.reserved_compute_usd == 10.0 and ledger.committed_usd == 33.0
    clock[0] = 8.0
    ledger.finish_compute("complete")
    assert ledger.spent_usd == 8.0 and not ledger.pending_compute and ledger.reserved_compute_usd == 0
    assert ledger.committed_usd == 28.0
    ledger.charge(20.0, category="storage", reference="actual synthetic storage invoice", receipt_id="storage-1")
    assert ledger.reserved_fixed_usd == 0 and ledger.committed_usd == 28.0
    reopened = BudgetLedger(ledger.path, ledger.plan, allow_create=False)
    reopened.verify_state(ledger.state_dict())
    assert reopened.state_dict() == ledger.state_dict()


@pytest.mark.parametrize(("seconds", "partial_settlement"), [(10.0, False), (10.0, True), (0.0, False)])
def test_abrupt_process_exit_requires_external_reconciliation(tmp_path, seconds, partial_settlement):
    path = tmp_path / "crashed.jsonl"
    worker = get_context("spawn").Process(target=_crash_worker, args=(str(path), seconds, partial_settlement))
    worker.start()
    worker.join(15)
    try:
        assert not worker.is_alive() and worker.exitcode == 23
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
    ledger = BudgetLedger(path, _plan(), allow_create=False)
    anchor = json.loads(Path(str(path) + ".anchor.json").read_text(encoding="utf-8"))
    ledger.verify_state(anchor)
    assert ledger.spent_usd == (3.0 if partial_settlement else 0.0)
    assert ledger.reserved_compute_usd == (7.0 if partial_settlement else seconds)
    assert ledger.state_dict()["billing_status"] == "compute_reconciliation_required"
    before = path.read_bytes()
    for blocked in (
        lambda: ledger.reserve_next(0.0),
        lambda: ledger.settle_elapsed("new clock"),
        lambda: ledger.finish_compute("new clock"),
        lambda: ledger.assert_room(0.0),
    ):
        with pytest.raises(PendingComputeError, match="external receipt"):
            blocked()
    with pytest.raises(PendingComputeError):
        BudgetLedger(path, _plan(), job_id="another-job")
    assert path.read_bytes() == before
    attempt = ledger.pending_compute[0]["attempt_id"]
    receipt = ledger.reconcile_compute(
        attempt, usd=4.0, receipt_id="external-1", reference="synthetic provider evidence for unrecorded remainder"
    )
    expected = 7.0 if partial_settlement else 4.0
    assert ledger.spent_usd == expected and ledger.committed_usd == expected and not ledger.pending_compute
    assert (
        ledger.reconcile_compute(
            attempt, usd=4.0, receipt_id="external-1", reference="synthetic provider evidence for unrecorded remainder"
        )
        == receipt
    )
    with pytest.raises(ValueError, match="different content"):
        ledger.reconcile_compute(attempt, usd=0.0, receipt_id="external-1", reference="erase the expense")
    ledger.verify_state(anchor)  # An old checkpoint anchor remains valid after appended reconciliation.
    following = BudgetLedger(path, _plan(), job_id="another-job")
    following.reserve_next(0.0)
    following.finish_compute("next job metadata-only completion")
    assert not following.pending_compute and following.spent_usd >= expected


def test_caught_exception_closes_attempt_with_measured_cost(tmp_path, clock):
    ledger = BudgetLedger(tmp_path / "costs.jsonl", _plan())
    with pytest.raises(RuntimeError, match="original failure"):
        with ledger.record_compute("work"):
            ledger.reserve_next(15.0)
            clock[0] = 4.0
            ledger.settle_elapsed("part")
            assert ledger.pending_compute
            clock[0] = 9.0
            raise RuntimeError("original failure")
    assert ledger.spent_usd == 9.0 and not ledger.pending_compute
    assert ledger.entries[-1]["reference"] == "work/aborted"
    assert not BudgetLedger(ledger.path, _plan()).pending_compute


def test_measured_overrun_is_durable_and_not_charged_again_by_abort(tmp_path, clock):
    ledger = BudgetLedger(tmp_path / "costs.jsonl", _plan(cap=10.0))
    ledger.reserve_next(1.0)
    clock[0] = 12.0
    with pytest.raises(BudgetExceededError):
        ledger.settle_elapsed("slower than estimated")
    assert ledger.spent_usd == 12.0
    with pytest.raises(BudgetExceededError):
        ledger.finish_compute("abort")
    assert ledger.spent_usd == 12.0 and not ledger.pending_compute


def test_recorded_overrun_can_be_reopened_and_reconciled_but_never_resumed(tmp_path, clock):
    ledger = BudgetLedger(tmp_path / "costs.jsonl", _plan(cap=10.0))
    ledger.reserve_next(1.0)
    clock[0] = 12.0
    with pytest.raises(BudgetExceededError):
        ledger.settle_elapsed("measured overrun before hard stop")
    reopened = BudgetLedger(ledger.path, ledger.plan, allow_create=False)
    assert reopened.spent_usd == 12.0 and reopened.pending_compute
    with pytest.raises(BudgetExceededError):
        reopened.reconcile_compute(
            reopened.pending_compute[0]["attempt_id"],
            usd=0.0,
            receipt_id="post-overrun-remainder",
            reference="synthetic evidence: no additional unrecorded charge",
        )
    assert reopened.spent_usd == 12.0 and not reopened.pending_compute
    final = BudgetLedger(ledger.path, ledger.plan, allow_create=False)
    assert final.spent_usd == 12.0 and not final.pending_compute
    with pytest.raises(BudgetExceededError):
        final.reserve_next(0.0)


@pytest.mark.parametrize("operation", ["reserve", "finish"])
def test_uncertain_fsync_outcome_requires_reopen_without_double_charge(tmp_path, clock, monkeypatch, operation):
    ledger = BudgetLedger(tmp_path / "costs.jsonl", _plan())
    if operation == "finish":
        ledger.reserve_next(10.0)
        clock[0] = 2.0
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("synthetic sync failure")))
        with pytest.raises(OSError, match="sync failure"):
            ledger.reserve_next(10.0) if operation == "reserve" else ledger.finish_compute("complete")
    with pytest.raises(ValueError, match="uncertain outcome"):
        ledger.finish_compute("cannot blindly retry")
    # The complete line may already be present despite fsync's uncertain result.
    reopened = BudgetLedger(ledger.path, _plan(), allow_create=False)
    if operation == "reserve":
        assert reopened.spent_usd == 0.0 and reopened.reserved_compute_usd == 10.0
        with pytest.raises(PendingComputeError):
            reopened.reserve_next(0.0)
    else:
        assert reopened.spent_usd == 2.0 and not reopened.pending_compute
        reopened.finish_compute("no interval to charge again")
        assert reopened.spent_usd == 2.0


@pytest.mark.parametrize("tail", [b'{"sequence":', b'{"invalid":true}\n'])
def test_incomplete_or_invalid_journal_tail_is_preserved_and_rejected(tmp_path, clock, tail):
    path = tmp_path / "costs.jsonl"
    ledger = BudgetLedger(path, _plan())
    ledger.reserve_next(10.0)
    with path.open("ab") as stream:
        stream.write(tail)
    before = path.read_bytes()
    with pytest.raises((ValueError, KeyError)):
        BudgetLedger(path, _plan(), allow_create=False)
    assert path.read_bytes() == before


def test_stale_ledger_cannot_append_or_return_from_noop_reservation(tmp_path, clock):
    first = BudgetLedger(tmp_path / "costs.jsonl", _plan())
    first.reserve_next(10.0)
    observer = BudgetLedger(first.path, _plan(), allow_create=False)
    # Independent actual invoices can arrive, but stale writers must reload.
    observer.charge(1.0, category="storage", reference="synthetic invoice", receipt_id="storage-1")
    before = first.path.read_bytes()
    with pytest.raises(ValueError, match="Another writer"):
        first.reserve_next(1.0)
    with pytest.raises(ValueError, match="Another writer"):
        first.finish_compute("stale owner")
    assert first.path.read_bytes() == before


def test_legacy_ledger_is_not_rewritten_or_silently_migrated(tmp_path):
    path = tmp_path / "legacy.jsonl"
    row = {
        "sequence": 0,
        "phase": "M1",
        "kind": "identity",
        "job_id": "legacy",
        "usd": 0.0,
        "previous_sha256": None,
        "identity": {
            "version": "reserved_phase_cost_ledger_v2",
            "phase": "M1",
            "budget_cap_usd": 100.0,
            "budget_spent_usd": 0.0,
        },
        "category": "control",
        "reference": "legacy synthetic history",
    }
    row["entry_sha256"] = content_hash(row)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="legacy ledgers"):
        BudgetLedger(path, _plan(), allow_create=False)
    assert path.read_bytes() == before
