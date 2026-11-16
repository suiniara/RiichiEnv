from __future__ import annotations

import math
from typing import Any

from .budget import CostForecast
from .contracts import MJSOUL_RULESET, content_hash
from .experiments import independent_seed
from .runtime_contracts import sha256_value

BENCHMARK_VERSION = "complete_policy_pair_benchmark_v2"


def benchmark_plan_from_records(
    subject: dict[str, Any],
    opponents: tuple[dict[str, Any], ...],
    *,
    matches: int,
    seed: int,
    device: str,
    source_sha256: str,
    mode: str = "paired",
    max_decisions: int = 10000,
) -> dict[str, Any]:
    """Preregister real matches, not candidate suffixes or a hidden prior override.

    matches is the TOTAL number of half-games. A paired run needs at least eight:
    four subject seats, with a baseline and a candidate half-game at each seat.
    Baseline-only timing remains an explicit diagnostic, never M3 cost evidence.
    """
    arms = ["baseline", "candidate"] if mode == "paired" else ["baseline"]
    if (
        mode not in {"paired", "baseline"}
        or type(matches) is not int
        or matches < 4 * len(arms)
        or matches % (4 * len(arms))
    ):
        raise ValueError("Register balanced complete matches; paired mode requires a multiple of eight")
    if type(seed) is not int or not 0 <= seed < 2**64 or device not in {"cpu", "cuda"}:
        raise ValueError("Register an explicit unsigned seed and inference backend")
    if type(max_decisions) is not int or max_decisions < 1 or len(opponents) != 3:
        raise ValueError("Register three opponents and a complete-match decision cap")
    sha256_value(source_sha256, "benchmark source")
    for release in (subject, *opponents):
        unsigned = {key: value for key, value in release.items() if key != "release_sha256"}
        if release.get("release_sha256") != content_hash(unsigned):
            raise ValueError("Benchmark release metadata integrity mismatch")
        if release.get("inference_device_type") != device:
            raise ValueError("Benchmark and frozen release backends differ")
    policy = subject["policy"]
    if policy.get("diagnostic_argmax", False):
        raise ValueError("Timing a diagnostic argmax cannot qualify stochastic M3")
    if mode == "paired" and (
        policy.get("mode") != "empirical_only" or not math.isfinite(float(policy["tau"])) or float(policy["tau"]) <= 0
    ):
        raise ValueError("Paired timing requires the validation-frozen finite empirical candidate")
    blocks = []
    for index in range(matches // len(arms)):
        blocks.append(
            {
                "pair_index": index,
                "subject_seat": index % 4,
                "wall_seed": independent_seed(seed, "benchmark/non_test/wall", index),
                **{arm + "_subject_rng_seed": independent_seed(seed, "benchmark/subject", arm, index) for arm in arms},
                "opponent_rng_seeds": [independent_seed(seed, "benchmark/opponent", index, seat) for seat in range(3)],
            }
        )
    record = {
        "version": BENCHMARK_VERSION,
        "mode": mode,
        "partition": "benchmark_non_test",
        "subject_release_sha256": subject["release_sha256"],
        "opponent_release_ids": [value["release_sha256"] for value in opponents],
        "candidate_policy": policy,
        "source_sha256": source_sha256,
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "device": device,
        "max_decisions": max_decisions,
        "master_seed": seed,
        "matches": matches,
        "arms": arms,
        "blocks": blocks,
        "seat_counts_per_arm": [len(blocks) // 4] * 4,
        "counterfactual_state_equality_claimed": False,
    }
    return {**record, "plan_sha256": content_hash(record)}


def qualify_benchmark_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate measured timing; this is not a performance or safety certificate."""
    unsigned = {key: value for key, value in report.items() if key != "benchmark_sha256"}
    if report.get("version") != BENCHMARK_VERSION or report.get("benchmark_sha256") != content_hash(unsigned):
        raise ValueError("Benchmark measurement integrity/version mismatch")
    plan = report["plan"]
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("version") != BENCHMARK_VERSION or plan.get("plan_sha256") != content_hash(plan_body):
        raise ValueError("Benchmark plan integrity mismatch")
    if (
        report.get("status") != "PASS"
        or report.get("complete") is not True
        or report.get("complete_matches") != plan["matches"]
    ):
        raise ValueError("Only completed registered games are throughput evidence")
    if plan.get("mode") != "paired" or plan.get("arms") != ["baseline", "candidate"]:
        raise ValueError("Prior-only timing does not cover M3 candidate inference")
    expected = plan["matches"] // 2
    if expected < 4 or expected % 4 or len(plan["blocks"]) != expected:
        raise ValueError("Paired benchmark is not balanced over all four subject seats")
    for arm in plan["arms"]:
        values = report["arms"][arm]
        if values["complete_matches"] != expected or values["seat_counts"] != [expected // 4] * 4:
            raise ValueError("Benchmark omitted an arm or a subject seat")
    if report["arms"]["baseline"]["candidate_forward_decisions"] != 0:
        raise ValueError("Infinity baseline unexpectedly performed candidate forwards")
    if report["arms"]["candidate"]["candidate_forward_decisions"] < 1:
        raise ValueError("Benchmark never exercised candidate response/value inference")
    policy = plan["candidate_policy"]
    if (
        policy.get("mode") != "empirical_only"
        or policy.get("diagnostic_argmax", False)
        or not math.isfinite(float(policy["tau"]))
        or float(policy["tau"]) <= 0
        or report.get("m3_forecast_eligible") is not True
    ):
        raise ValueError("The measured candidate is not an eligible frozen finite stochastic policy")
    threads = report["runtime_identity"].get("cpu_threads")
    if type(threads) is not int or threads < 1:
        raise ValueError("Benchmark must record the actual inference CPU thread setting")
    elapsed, rate = report["elapsed_seconds"], report["matches_per_second"]
    if (
        isinstance(elapsed, bool)
        or isinstance(rate, bool)
        or not math.isfinite(elapsed)
        or not math.isfinite(rate)
        or min(elapsed, rate) <= 0
    ):
        raise ValueError("Benchmark requires actual finite wall-clock timing")
    if not math.isclose(rate, plan["matches"] / elapsed, rel_tol=1e-9):
        raise ValueError("Throughput does not match measured full-job elapsed time")
    return {
        "benchmark_sha256": report["benchmark_sha256"],
        "matches_per_second": rate,
        "plan": plan,
        "runtime_identity": report["runtime_identity"],
    }


def validate_m3_forecast(
    paired: dict[str, Any],
    forecast: CostForecast,
    benchmark: dict[str, Any] | None,
    *,
    source_sha256: str,
) -> dict[str, Any]:
    if benchmark is None:
        raise ValueError("M3 requires completed paired-policy benchmark evidence, not a bare throughput number")
    evidence = qualify_benchmark_report(benchmark)
    expected = paired.get("forecast_evidence", {}).get("benchmark_sha256")
    if expected != evidence["benchmark_sha256"]:
        raise ValueError("M3 forecast evidence changed after plan registration")
    plan = evidence["plan"]
    if (
        plan["source_sha256"] != source_sha256
        or plan["subject_release_sha256"] != paired["subject_release_sha256"]
        or plan["opponent_release_ids"] != paired["opponent_release_ids"]
        or plan["device"] != paired["execution"]["device"]
        or plan["max_decisions"] != paired["execution"]["max_decisions"]
        or plan["ruleset_hash"] != paired["ruleset_hash"]
    ):
        raise ValueError("M3 timing used another policy, opponents, backend, rules, source or truncation cap")
    walls = {block["wall_seed"] for block in plan["blocks"]}
    if any(block["wall_seed"] in walls for block in paired["blocks"]):
        raise ValueError("Timing games must not consume the registered independent M3 walls")
    if forecast.measured_units_per_second > evidence["matches_per_second"] * (1 + 1e-9):
        raise ValueError("M3 forecast is faster than the completed paired-policy measurement")
    if forecast.measurement_reference != "benchmark:" + evidence["benchmark_sha256"]:
        raise ValueError("M3 cost forecast must reference its exact measured artifact")
    return evidence
