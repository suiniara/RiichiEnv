import copy

import pytest

from riichienv.kar.budget import CostForecast
from riichienv.kar.contracts import content_hash
from riichienv.kar.experiments import paired_seed_manifest, validate_paired_manifest
from riichienv.kar.simulation_contracts import (
    BENCHMARK_VERSION,
    benchmark_plan_from_records,
    qualify_benchmark_report,
    validate_m3_forecast,
)


def release_fixture(identity, *, tau=0.5):
    # Synthetic metadata for contract tests only; never saved as a real model release.
    body = {
        "fixture": identity,
        "inference_device_type": "cpu",
        "policy": {
            "mode": "empirical_only" if tau != "infinity" else "prior_only",
            "tau": tau,
            "diagnostic_argmax": False,
        },
    }
    return {**body, "release_sha256": content_hash(body)}


def benchmark_fixture():
    subject = release_fixture("subject")
    opponents = tuple(release_fixture(f"opponent-{index}", tau="infinity") for index in range(3))
    plan = benchmark_plan_from_records(subject, opponents, matches=8, seed=17, device="cpu", source_sha256="a" * 64)
    record = {
        "version": BENCHMARK_VERSION,
        "status": "PASS",
        "complete": True,
        "complete_matches": 8,
        "plan": plan,
        "arms": {
            arm: {
                "complete_matches": 4,
                "seat_counts": [1] * 4,
                "candidate_forward_decisions": 0 if arm == "baseline" else 4,
            }
            for arm in ("baseline", "candidate")
        },
        "elapsed_seconds": 16.0,
        "matches_per_second": 0.5,
        "runtime_identity": {"fixture_only": True, "cpu_threads": 2},
        "m3_forecast_eligible": True,
    }
    record["benchmark_sha256"] = content_hash(record)
    return subject, opponents, record


def signed(value, field):
    result = {key: item for key, item in value.items() if key != field}
    return {**result, field: content_hash(result)}


def test_paired_benchmark_registers_both_arms_shared_walls_and_independent_subject_rng():
    _, _, record = benchmark_fixture()
    plan = record["plan"]
    assert plan["matches"] == 8 and len(plan["blocks"]) == 4 and plan["seat_counts_per_arm"] == [1] * 4
    assert plan["arms"] == ["baseline", "candidate"]
    assert all(row["baseline_subject_rng_seed"] != row["candidate_subject_rng_seed"] for row in plan["blocks"])
    assert plan["partition"] == "benchmark_non_test" and not plan["counterfactual_state_equality_claimed"]
    assert qualify_benchmark_report(record)["matches_per_second"] == 0.5


@pytest.mark.parametrize("matches", [0, 4, 6, 10])
def test_paired_workload_is_total_games_and_cannot_omit_half_an_arm(matches):
    subject, opponents, _ = benchmark_fixture()
    with pytest.raises(ValueError):
        benchmark_plan_from_records(subject, opponents, matches=matches, seed=1, device="cpu", source_sha256="a" * 64)


def test_prior_timing_or_unexercised_candidate_cannot_qualify_m3():
    subject, opponents, record = benchmark_fixture()
    prior = benchmark_plan_from_records(
        subject, opponents, matches=4, seed=1, device="cpu", source_sha256="a" * 64, mode="baseline"
    )
    changed = {**record, "plan": prior, "complete_matches": 4}
    with pytest.raises(ValueError):
        qualify_benchmark_report(signed(changed, "benchmark_sha256"))
    changed = copy.deepcopy(record)
    changed["arms"]["candidate"]["candidate_forward_decisions"] = 0
    with pytest.raises(ValueError):
        qualify_benchmark_report(signed(changed, "benchmark_sha256"))
    with pytest.raises(ValueError):
        benchmark_plan_from_records(
            release_fixture("prior", tau="infinity"), opponents, matches=8, seed=1, device="cpu", source_sha256="a" * 64
        )


def test_m3_cost_evidence_is_bound_to_candidate_opponents_backend_and_registered_measurement():
    subject, opponents, report = benchmark_fixture()
    plan = paired_seed_manifest(
        master_seed=29,
        subject_release_sha256=subject["release_sha256"],
        opponent_release_ids=tuple(value["release_sha256"] for value in opponents),
        benchmark_sha256=report["benchmark_sha256"],
    )
    forecast = CostForecast(
        "M3", 20000, 0.5, 0.0, 0.0, "benchmark:" + report["benchmark_sha256"], plan["paired_seed_manifest_sha256"]
    )
    validate_paired_manifest(plan, require_benchmark=True)
    assert validate_m3_forecast(plan, forecast, report, source_sha256="a" * 64)["matches_per_second"] == 0.5
    for key, value in (("subject_release_sha256", "f" * 64), ("opponent_release_ids", ["f" * 64] * 3)):
        wrong = {**plan, key: value}
        with pytest.raises(ValueError):
            validate_m3_forecast(wrong, forecast, report, source_sha256="a" * 64)
    faster = CostForecast("M3", 20000, 1.0, 0.0, 0.0, forecast.measurement_reference, forecast.recipe_sha256)
    with pytest.raises(ValueError, match="faster"):
        validate_m3_forecast(plan, faster, report, source_sha256="a" * 64)
    with pytest.raises(ValueError):
        validate_m3_forecast(plan, forecast, None, source_sha256="a" * 64)


def test_draft_paired_seed_manifest_is_not_execution_ready():
    subject, opponents, _ = benchmark_fixture()
    plan = paired_seed_manifest(
        master_seed=3,
        subject_release_sha256=subject["release_sha256"],
        opponent_release_ids=tuple(value["release_sha256"] for value in opponents),
    )
    validate_paired_manifest(plan)
    assert not plan["execution_ready"]
    with pytest.raises(ValueError):
        validate_paired_manifest(plan, require_benchmark=True)


def test_real_benchmark_dispatches_candidate_instead_of_overriding_every_policy(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from types import SimpleNamespace  # noqa: PLC0415

    from riichienv.kar.budget import BudgetLedger, BudgetPlan  # noqa: PLC0415
    from riichienv.kar.learning import simulation  # noqa: PLC0415

    subject, opponents, fixture = benchmark_fixture()
    plan = fixture["plan"]
    paths = [tmp_path / f"release-{index}" for index in range(4)]
    records = dict(zip(paths, (subject, *opponents)))

    class FakeRelease:
        def __init__(self, path, *, device):
            self.record = records[path]

        def policy(self, *, seed, baseline=False):
            return SimpleNamespace(seed=seed, baseline=baseline)

    calls = []

    def fake_match(agents, *, wall_seed, max_decisions):
        index = len(calls)
        seat = (index // 2) % 4
        baseline = agents[seat].policy.baseline
        calls.append((baseline, wall_seed))
        return {
            "status": "PASS",
            "complete": True,
            "ranks": [1, 2, 3, 4],
            "scores": [25000] * 4,
            "events": [],
            "decision_count": 1,
            "match_sha256": str(index) * 64,
            "policy_rng_final_sha256": {"0": "e" * 64},
            "decisions": [
                {
                    "seat": seat,
                    "decision_latency_seconds": 0.01,
                    "fallback_reason": None,
                    "evidence_status": "unfitted_prior_only" if baseline else "empirical_only",
                    "telemetry": {"candidate_forward_count": 0 if baseline else 2, "supported_discards": [0, 4]},
                }
            ],
        }

    monkeypatch.setattr(simulation, "FrozenRelease", FakeRelease)
    monkeypatch.setattr(simulation, "run_match", fake_match)
    monkeypatch.setattr(simulation, "simulation_runtime_identity", lambda device: {"fixture_only": True})
    budget = BudgetPlan(
        "M1",
        100.0,
        0.0,
        CostForecast("M1", 8, 100.0, 0.0, 0.0, "synthetic fake-dispatch unit test", plan["plan_sha256"]),
    )
    ledger = BudgetLedger(tmp_path / "costs.jsonl", budget)
    record = simulation._benchmark_measure(
        paths[0], tuple(paths[1:]), output=tmp_path, device="cpu", ledger=ledger, spec=plan, digest=plan["plan_sha256"]
    )
    assert [row[0] for row in calls] == [True, False] * 4
    assert all(calls[index][1] == calls[index + 1][1] for index in range(0, 8, 2))
    assert record["arms"]["candidate"]["candidate_actions_scored"] == 8 and record["complete_matches"] == 8
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 8 and record["m3_forecast_eligible"]
