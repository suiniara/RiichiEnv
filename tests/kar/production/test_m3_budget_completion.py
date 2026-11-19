"""Exercise final accounting with cached synthetic receipts, never models or matches."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from riichienv.kar import budget as budget_module
from riichienv.kar.authorization import update_implementation_sha256
from riichienv.kar.budget import BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.contracts import content_hash
from riichienv.kar.statistics import LatencyDistribution


@pytest.mark.parametrize("fail_statistics", [False, True])
def test_m3_cached_receipt_completion_and_failure_close_compute_attempt(tmp_path, monkeypatch, fail_statistics):
    pytest.importorskip("torch")
    from riichienv.kar.learning import simulation  # noqa: PLC0415

    clock = [0.0]
    monkeypatch.setattr(budget_module.time, "monotonic", lambda: clock[0])
    # Only the accounting orchestration is under test. No production admission,
    # 10,000-pair evaluation, model load, or game is represented by this fixture.
    monkeypatch.setattr(simulation, "M3_PAIRS", 1)
    digest, source = "c" * 64, update_implementation_sha256()
    plan = {
        "paired_seed_manifest_sha256": digest,
        "subject_release_sha256": "subject",
        "opponent_release_ids": ["opponent"] * 3,
        "partition": "test",
        "seat_counts": [1, 0, 0, 0],
        "blocks": [{"pair_index": 0, "block_sha256": "block", "subject_seat": 0}],
    }
    budget = BudgetPlan(
        "M3",
        250.0,
        0.0,
        CostForecast("M3", 2, 1.0, 3600.0, 0.0, "synthetic receipt-only accounting", digest, 1.0),
    )
    output = tmp_path / "run"
    output.mkdir()
    config = SimpleNamespace(
        output=output,
        test_registry=tmp_path / "registry.json",
        budget=budget,
        paired_seed_manifest=plan,
        subject_release="subject",
        opponent_releases=("opponent",) * 3,
        device="cpu",
        benchmark_report={"runtime_identity": {"cpu_threads": 1}},
        bootstrap_seed=17,
        bootstrap_replicates=1,
        validate=lambda: None,
    )
    ledger = BudgetLedger(output / "costs.jsonl", budget)
    (output / "receipts").mkdir()
    (output / "replays").mkdir()
    latency = LatencyDistribution()
    latency.add(0.01)
    stats = {
        "rank": 2,
        "utility": 0.5,
        "points": 25000,
        "first": 0,
        "fourth": 0,
        "wins": 0,
        "deal_ins": 0,
        "decisions": 1,
        "decision_latency_sum": 0.01,
        "reordered_decisions": 0,
        "eligible_gate_decisions": 0,
        "decision_latency_distribution": latency.state_dict(),
    }
    for arm in ("baseline", "candidate"):
        replay = output / "replays" / f"pair-00000-{arm}.json.gz"
        replay.write_bytes(b"cached synthetic metadata, not a real match")
        simulation._atomic_record(
            output / "receipts" / f"pair-00000-{arm}.json",
            {
                "plan_sha256": digest,
                "source_sha256": source,
                "block_sha256": "block",
                "arm": arm,
                "subject_seat": 0,
                "statistics": stats,
                "replay_file": replay.name,
                "replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
                "budget_ledger_head": ledger.head,
                "budget_ledger_entries": len(ledger.entries),
            },
        )

    def metadata_release(path, **_kwargs):
        clock[0] += 1.0
        return SimpleNamespace(
            record={"release_sha256": path},
            options=SimpleNamespace(mode="empirical_only", tau=0.5, diagnostic_argmax=False),
        )

    def forbidden_match(*_args, **_kwargs):
        raise AssertionError("cached accounting fixture must never execute a match")

    def statistics(*_args, **_kwargs):
        clock[0] = 10.0
        if fail_statistics:
            raise RuntimeError("synthetic bootstrap failure")
        return {"status": "synthetic accounting fixture only"}

    monkeypatch.setattr(simulation, "FrozenRelease", metadata_release)
    monkeypatch.setattr(simulation, "run_match", forbidden_match)
    monkeypatch.setattr(simulation, "simulation_runtime_identity", lambda _device: {"cpu_threads": 1})
    monkeypatch.setattr(simulation.GameClusters, "interval", statistics)
    authorization = {
        "version": "explicit_frozen_evaluation_authorization_v1",
        "explicit_user_approval": True,
        "approval_reference": "metadata-only cached receipt completion fixture",
        "allowed_operations": ["run_m3"],
        "plan_sha256": digest,
        "source_sha256": source,
    }
    authorization["authorization_sha256"] = content_hash(authorization)
    if fail_statistics:
        with pytest.raises(RuntimeError, match="synthetic bootstrap failure"):
            simulation.run_m3(config, authorization=authorization)
        report = json.loads(next(output.glob("aborted-*.json")).read_text(encoding="utf-8"))
    else:
        report = simulation.run_m3(config, authorization=authorization)
    assert report["budget"]["spent_usd"] == 10.0 and report["budget"]["pending_compute"] == []
    reopened = BudgetLedger(output / "costs.jsonl", budget, allow_create=False)
    assert reopened.spent_usd == 10.0 and not reopened.pending_compute
    assert sum(row["kind"] == "compute_reservation" for row in reopened.entries) == 2
