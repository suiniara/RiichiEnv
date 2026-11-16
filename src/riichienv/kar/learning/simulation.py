from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..agent import KARAgent, run_match
from ..authorization import update_implementation_sha256
from ..budget import BudgetLedger, BudgetPlan
from ..contracts import content_hash
from ..evaluation_authorization import exclusive_job, require_evaluation_authorization
from ..experiments import M3_MATCHES, M3_PAIRS, validate_paired_manifest
from ..simulation_contracts import BENCHMARK_VERSION, benchmark_plan_from_records, validate_m3_forecast
from ..statistics import GameClusters, LatencyDistribution, percentile
from .deployment import FrozenRelease


def _atomic_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    signed = {**record, "receipt_sha256": content_hash(record)}
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(signed, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return signed


def _read_record(path: Path, plan_sha: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "receipt_sha256"}
    if record.get("receipt_sha256") != content_hash(unsigned) or record.get("plan_sha256") != plan_sha:
        raise ValueError("Evaluation receipt integrity/plan mismatch")
    return record


def _match_statistics(report: dict[str, Any], subject_seat: int) -> dict[str, Any]:
    if report.get("status") != "PASS" or not report.get("complete"):
        raise RuntimeError("Truncated/blocked matches do not count toward an evaluation")
    decisions = [row for row in report["decisions"] if row["seat"] == subject_seat]
    latencies = [row["decision_latency_seconds"] for row in decisions]
    latency_distribution = LatencyDistribution()
    for seconds in latencies:
        latency_distribution.add(seconds)
    wins = [event for event in report["events"] if event["type"] == "hora"]
    return {
        "rank": report["ranks"][subject_seat],
        "utility": 2.5 - report["ranks"][subject_seat],
        "points": report["scores"][subject_seat],
        "first": int(report["ranks"][subject_seat] == 1),
        "fourth": int(report["ranks"][subject_seat] == 4),
        "wins": sum(event["actor"] == subject_seat for event in wins),
        "deal_ins": len(
            {
                event["source_record_index"]
                for event in wins
                if event["target"] == subject_seat and event["actor"] != subject_seat
            }
        ),
        "decisions": len(decisions),
        "decision_latency_sum": math.fsum(latencies),
        "decision_latency_p95": percentile(latencies, 0.95),
        "decision_latency_distribution": latency_distribution.state_dict(),
        "candidate_forward_decisions": sum(row["telemetry"].get("candidate_forward_count", 0) > 0 for row in decisions),
        "candidate_actions_scored": sum(row["telemetry"].get("candidate_forward_count", 0) for row in decisions),
        "reordered_decisions": sum(
            row["telemetry"].get("distribution_changed", False) and row["fallback_reason"] is None for row in decisions
        ),
        "eligible_gate_decisions": sum(len(row["telemetry"].get("supported_discards", [])) >= 2 for row in decisions),
        "evidence_counts": {
            value: sum(row["evidence_status"] == value for row in decisions)
            for value in sorted({row["evidence_status"] for row in decisions})
        },
    }


def _agents(
    subject: FrozenRelease, opponents: tuple[FrozenRelease, ...], block: dict[str, Any], arm: str
) -> dict[int, KARAgent]:
    seat = block["subject_seat"]
    seed = block[arm + "_subject_rng_seed"]
    agents = {seat: KARAgent(seat, subject.policy(seed=seed, baseline=arm == "baseline"))}
    other_seats = [(seat + offset) % 4 for offset in (1, 2, 3)]
    for index, other in enumerate(other_seats):
        agents[other] = KARAgent(other, opponents[index].policy(seed=block["opponent_rng_seeds"][index]))
    return agents


@dataclass(frozen=True)
class M3Config:
    paired_seed_manifest: dict[str, Any]
    subject_release: Path
    opponent_releases: tuple[Path, Path, Path]
    output: Path
    test_registry: Path
    budget: BudgetPlan
    device: str = "cpu"
    max_decisions: int = 10000
    bootstrap_seed: int = 17
    bootstrap_replicates: int = 10000
    benchmark_report: dict[str, Any] | None = None

    def validate(self) -> None:
        validate_paired_manifest(self.paired_seed_manifest, require_benchmark=True)
        if self.paired_seed_manifest["partition"] != "test":
            raise ValueError("Formal M3 is the frozen independent test evaluation, not validation tuning")
        validate_m3_forecast(
            self.paired_seed_manifest,
            self.budget.forecast,
            self.benchmark_report,
            source_sha256=update_implementation_sha256(),
        )
        digest = self.paired_seed_manifest["paired_seed_manifest_sha256"]
        if (
            self.budget.phase != "M3"
            or self.budget.forecast.work_units != M3_MATCHES
            or self.budget.forecast.recipe_sha256 != digest
        ):
            raise ValueError("M3 budget forecast must cover all 20000 matches of this exact frozen plan")
        execution = self.paired_seed_manifest["execution"]
        statistics = self.paired_seed_manifest["statistics"]
        if (
            self.device != execution["device"]
            or self.max_decisions != execution["max_decisions"]
            or self.bootstrap_seed != statistics["seed"]
            or self.bootstrap_replicates != statistics["bootstrap_replicates"]
        ):
            raise ValueError("Do not change backend,truncation limit or confidence-interval recipe after registration")
        if self.max_decisions < 1 or self.bootstrap_replicates < 1:
            raise ValueError("Invalid complete-match/statistics recipe")


def register_m3_run(config: M3Config) -> bool:
    """Bind the one frozen test plan to its output and phase ledger before work."""
    if not config.test_registry.parent.is_dir():
        raise ValueError("An explicit shared test registry directory must already exist")
    record = {
        "version": "unique_frozen_m3_run_v2",
        "paired_seed_manifest_sha256": config.paired_seed_manifest["paired_seed_manifest_sha256"],
        "subject_release_sha256": config.paired_seed_manifest["subject_release_sha256"],
        "partition": config.paired_seed_manifest["partition"],
        "output": str(config.output.resolve()),
        "budget_ledger": str((config.output / "costs.jsonl").resolve()),
        "phase_cap_usd": config.budget.budget_cap_usd,
        "initial_spent_usd": config.budget.budget_spent_usd,
        "forecast_sha256": config.budget.forecast.to_record()["forecast_sha256"],
    }
    signed = {**record, "registration_sha256": content_hash(record)}
    with exclusive_job(config.test_registry.with_suffix(".lock")):
        existed = config.test_registry.exists()
        if existed:
            if json.loads(config.test_registry.read_text(encoding="utf-8")) != signed:
                raise ValueError(
                    "M3 is already registered; do not change policy/output/pricing or reset its phase ledger"
                )
        else:
            with config.test_registry.open("x", encoding="utf-8") as stream:
                json.dump(signed, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
    return existed


def open_m3_ledger(config: M3Config, source_sha256: str, *, allow_create: bool = True) -> BudgetLedger:
    digest = config.paired_seed_manifest["paired_seed_manifest_sha256"]
    receipts = tuple((config.output / "receipts").glob("pair-*.json"))
    failures = tuple(config.output.glob("aborted-*.json"))
    completed = config.output / "report.json"
    ledger = BudgetLedger(
        config.output / "costs.jsonl",
        config.budget,
        allow_create=allow_create and not (receipts or failures or completed.exists()),
    )
    # Verify every previously durable accounting anchor before starting any new
    # model/game work, even when a missing arm sorts before another cached arm.
    for path in receipts:
        receipt = _read_record(path, digest)
        if receipt.get("source_sha256") != source_sha256:
            raise ValueError("Evaluation receipt belongs to another source build")
        ledger.verify_anchor(receipt.get("budget_ledger_head"), entries=receipt.get("budget_ledger_entries"))
    for path in (*failures, *((completed,) if completed.exists() else ())):
        record = _read_record(path, digest)
        if record.get("source_sha256") != source_sha256:
            raise ValueError("Evaluation publication belongs to another source build")
        ledger.verify_state(record["budget"])
    return ledger


def run_m3(config: M3Config, *, authorization: dict[str, Any] | None = None) -> dict[str, Any]:  # noqa: PLR0915
    plan = config.paired_seed_manifest
    digest = plan["paired_seed_manifest_sha256"]
    require_evaluation_authorization(authorization, operation="run_m3", plan_sha256=digest)
    assert authorization is not None
    config.validate()
    config.output.mkdir(parents=True, exist_ok=True)
    with exclusive_job(config.output / "job.lock"):
        registered = register_m3_run(config)
        ledger = open_m3_ledger(config, authorization["source_sha256"], allow_create=not registered)
        completed_path = config.output / "report.json"
        if completed_path.exists():
            completed_record = _read_record(completed_path, digest)
            if not completed_record.get("complete") or completed_record.get("paired_blocks") != M3_PAIRS:
                raise ValueError("Incomplete report cannot be resumed as complete")
            if completed_record.get("source_sha256") != authorization["source_sha256"]:
                raise ValueError("Completed evaluation belongs to another source build")
            return completed_record
        with ledger.record_compute("M3_load_and_validate"):
            assert config.benchmark_report is not None
            torch.set_num_threads(config.benchmark_report["runtime_identity"]["cpu_threads"])
            subject = FrozenRelease(config.subject_release, device=config.device)
            opponents = tuple(FrozenRelease(path, device=config.device) for path in config.opponent_releases)
            if (
                subject.record["release_sha256"] != plan["subject_release_sha256"]
                or [value.record["release_sha256"] for value in opponents] != plan["opponent_release_ids"]
            ):
                raise ValueError("Subject/opponent releases changed after seed-plan registration")
            assert config.benchmark_report is not None
            if simulation_runtime_identity(config.device) != config.benchmark_report["runtime_identity"]:
                raise ValueError("Execution hardware/runtime differs from the completed cost benchmark")
            if (
                subject.options.mode != "empirical_only"
                or math.isinf(subject.options.tau)
                or subject.options.diagnostic_argmax
            ):
                raise ValueError("M3 candidate must be the validation-frozen stochastic finite empirical policy")
        receipts = config.output / "receipts"
        replays = config.output / "replays"
        receipts.mkdir(exist_ok=True)
        replays.mkdir(exist_ok=True)
        clusters = GameClusters()
        arm_totals = {
            arm: {
                "matches": 0,
                "rank": 0.0,
                "points": 0.0,
                "first": 0.0,
                "fourth": 0.0,
                "wins": 0.0,
                "deal_ins": 0.0,
                "decisions": 0,
                "latency": 0.0,
                "reordered": 0,
                "gate": 0,
            }
            for arm in ("baseline", "candidate")
        }
        arm_latencies = {arm: LatencyDistribution() for arm in ("baseline", "candidate")}
        completed = 0
        started = time.perf_counter()
        try:
            for block in plan["blocks"]:
                pair = []
                for arm in ("baseline", "candidate"):
                    path = receipts / f"pair-{block['pair_index']:05d}-{arm}.json"
                    if path.exists():
                        receipt = _read_record(path, digest)
                        ledger.verify_anchor(
                            receipt.get("budget_ledger_head"), entries=receipt.get("budget_ledger_entries")
                        )
                        replay = replays / receipt["replay_file"]
                        if hashlib.sha256(replay.read_bytes()).hexdigest() != receipt["replay_sha256"]:
                            raise ValueError("Archived replay changed")
                    else:
                        ledger.reserve_next(
                            config.budget.forecast.safety_multiplier / config.budget.forecast.measured_units_per_second
                        )
                        report = run_match(
                            _agents(subject, opponents, block, arm),
                            wall_seed=block["wall_seed"],
                            max_decisions=config.max_decisions,
                        )
                        statistics = _match_statistics(report, block["subject_seat"])
                        replay_name = f"pair-{block['pair_index']:05d}-{arm}.json.gz"
                        replay = replays / replay_name
                        if replay.exists():
                            raise ValueError(
                                "Orphan replay without receipt; preserve it for explicit recovery, never overwrite"
                            )
                        replay_sha = _archive_match(replay, report)
                        ledger.settle_elapsed(f"pair/{block['pair_index']}/{arm}")
                        receipt = _atomic_record(
                            path,
                            {
                                "plan_sha256": digest,
                                "source_sha256": authorization["source_sha256"],
                                "block_sha256": block["block_sha256"],
                                "pair_index": block["pair_index"],
                                "arm": arm,
                                "subject_seat": block["subject_seat"],
                                "statistics": statistics,
                                "match_sha256": report["match_sha256"],
                                "replay_file": replay_name,
                                "replay_sha256": replay_sha,
                                "policy_rng_final_sha256": report["policy_rng_final_sha256"],
                                "budget_ledger_head": ledger.head,
                                "budget_ledger_entries": len(ledger.entries),
                            },
                        )
                    if receipt.get("source_sha256") != authorization["source_sha256"]:
                        raise ValueError("Do not mix evaluation source builds across resumed paired blocks")
                    if (
                        receipt["block_sha256"] != block["block_sha256"]
                        or receipt["arm"] != arm
                        or receipt["subject_seat"] != block["subject_seat"]
                    ):
                        raise ValueError("Paired receipt belongs to another seed/seat/arm")
                    stats = receipt["statistics"]
                    arm_latencies[arm].merge(stats["decision_latency_distribution"])
                    total = arm_totals[arm]
                    total["matches"] += 1
                    for key in ("rank", "points", "first", "fourth", "wins", "deal_ins"):
                        total[key] += stats[key]
                    total["decisions"] += stats["decisions"]
                    total["latency"] += stats["decision_latency_sum"]
                    total["reordered"] += stats["reordered_decisions"]
                    total["gate"] += stats["eligible_gate_decisions"]
                    pair.append(stats)
                for key in ("rank", "utility", "points", "first", "fourth", "wins", "deal_ins"):
                    clusters.add(
                        str(block["pair_index"]), key + "_candidate_minus_baseline", pair[1][key] - pair[0][key]
                    )
                completed += 1
            if completed != M3_PAIRS or any(value["matches"] != M3_PAIRS for value in arm_totals.values()):
                raise RuntimeError("Never publish a reduced M3 as complete")
            ledger.settle_elapsed("M3_finalize")
            report = {
                "status": "PASS",
                "plan_sha256": digest,
                "source_sha256": authorization["source_sha256"],
                "complete": True,
                "paired_seed_manifest_sha256": digest,
                "paired_blocks": completed,
                "matches": 2 * completed,
                "seat_counts": plan["seat_counts"],
                "arms": {
                    arm: {
                        "matches": value["matches"],
                        **{
                            key: value[key] / value["matches"]
                            for key in ("rank", "points", "first", "fourth", "wins", "deal_ins")
                        },
                        "mean_decision_latency_seconds": value["latency"] / value["decisions"]
                        if value["decisions"]
                        else None,
                        "decision_latency_distribution": arm_latencies[arm].report(),
                        "rerank_coverage": value["reordered"] / value["decisions"] if value["decisions"] else None,
                        "support_gate_coverage": value["gate"] / value["decisions"] if value["decisions"] else None,
                    }
                    for arm, value in arm_totals.items()
                },
                "paired_intervals": {
                    key: clusters.interval(
                        key + "_candidate_minus_baseline",
                        seed=config.bootstrap_seed,
                        replicates=config.bootstrap_replicates,
                    )
                    for key in ("rank", "utility", "points", "first", "fourth", "wins", "deal_ins")
                },
                "budget": ledger.state_dict(),
                "elapsed_seconds": time.perf_counter() - started,
                "partition": plan["partition"],
                "counterfactual_state_equality_claimed": False,
                "policy_guarantee": "empirical_only_no_uniform_error_certificate",
            }
            ledger.settle_elapsed("M3_statistics_and_finalize")
            report["budget"] = ledger.state_dict()
            report["elapsed_seconds"] = time.perf_counter() - started
            report["report_sha256"] = content_hash(report)
            path = config.output / "report.json"
            if not path.exists():
                _atomic_record(path, report)
            else:
                previous = _read_record(path, digest) if "plan_sha256" in json.loads(path.read_text()) else None
                if previous is None:
                    raise ValueError("Completed report already exists; do not silently replace evaluation history")
            return report
        except BaseException as error:
            try:
                ledger.settle_elapsed("M3_aborted")
            except Exception:
                pass
            failure = {
                "status": "BLOCKED",
                "complete": False,
                "completed_pairs": completed,
                "required_pairs": M3_PAIRS,
                "plan_sha256": digest,
                "source_sha256": authorization["source_sha256"],
                "reason": f"{type(error).__name__}: {error}",
                "budget": ledger.state_dict(),
            }
            _atomic_record(config.output / f"aborted-{uuid.uuid4().hex}.json", failure)
            raise


def simulation_runtime_identity(device: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "device": device,
        "cuda_runtime": torch.version.cuda if device == "cuda" else None,
    }
    if device == "cuda":
        properties = torch.cuda.get_device_properties(torch.device(device))
        record["cuda_device"] = {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
        }
    return record


def benchmark_plan(
    subject_path: Path,
    opponent_paths: tuple[Path, Path, Path],
    *,
    matches: int,
    seed: int,
    device: str = "cpu",
    mode: str = "paired",
    max_decisions: int = 10000,
) -> dict[str, Any]:
    def metadata(path: Path) -> dict[str, Any]:
        source = path / "release.json"
        if not source.is_file() or source.stat().st_size > 32 * 1024**2:
            raise ValueError("Missing/oversized immutable release metadata")
        return json.loads(source.read_text(encoding="utf-8"))

    return benchmark_plan_from_records(
        metadata(subject_path),
        tuple(metadata(path) for path in opponent_paths),
        matches=matches,
        seed=seed,
        device=device,
        source_sha256=update_implementation_sha256(),
        mode=mode,
        max_decisions=max_decisions,
    )


def _archive_match(path: Path, report: dict[str, Any]) -> str:
    with path.open("xb") as target:
        with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as compressed:
            compressed.write(json.dumps(report, sort_keys=True, allow_nan=False).encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_matches(
    subject_path: Path,
    opponent_paths: tuple[Path, Path, Path],
    *,
    matches: int,
    seed: int,
    output: Path,
    budget: BudgetPlan,
    authorization: dict[str, Any] | None = None,
    device: str = "cpu",
    mode: str = "paired",
    max_decisions: int = 10000,
) -> dict[str, Any]:
    if authorization is None:
        require_evaluation_authorization(None, operation="benchmark_matches", plan_sha256="")
    spec = benchmark_plan(
        subject_path, opponent_paths, matches=matches, seed=seed, device=device, mode=mode, max_decisions=max_decisions
    )
    digest = spec["plan_sha256"]
    require_evaluation_authorization(authorization, operation="benchmark_matches", plan_sha256=digest)
    if budget.phase != "M1" or budget.forecast.work_units < matches or budget.forecast.recipe_sha256 != digest:
        raise ValueError("Simulation forecast must cover every registered baseline and candidate match")
    output.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger(output / "costs.jsonl", budget)
    with ledger.record_compute("benchmark_matches"):
        result = _benchmark_measure(
            subject_path, opponent_paths, output=output, device=device, ledger=ledger, spec=spec, digest=digest
        )
    result["budget"] = ledger.state_dict()
    result["benchmark_sha256"] = content_hash(result)
    (output / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _benchmark_measure(  # noqa: PLR0915
    subject_path: Path,
    opponent_paths: tuple[Path, Path, Path],
    *,
    output: Path,
    device: str,
    ledger: BudgetLedger,
    spec: dict[str, Any],
    digest: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    subject = FrozenRelease(subject_path, device=device)
    opponents = tuple(FrozenRelease(path, device=device) for path in opponent_paths)
    if (
        subject.record["release_sha256"] != spec["subject_release_sha256"]
        or [value.record["release_sha256"] for value in opponents] != spec["opponent_release_ids"]
    ):
        raise ValueError("Release files changed between plan registration and model loading")
    runtime = simulation_runtime_identity(device)
    replays = output / "replays"
    receipts = output / "receipts"
    replays.mkdir()
    receipts.mkdir()
    arms: dict[str, dict[str, Any]] = {
        arm: {
            "complete_matches": 0,
            "seat_counts": [0] * 4,
            "decisions": 0,
            "candidate_forward_decisions": 0,
            "candidate_actions_scored": 0,
            "subject_decisions": 0,
            "subject_latency_sum": 0.0,
            "reordered_decisions": 0,
        }
        for arm in spec["arms"]
    }
    latencies: dict[str, list[float]] = {arm: [] for arm in spec["arms"]}
    for block in spec["blocks"]:
        for arm in spec["arms"]:
            ledger.reserve_next(ledger.plan.forecast.safety_multiplier / ledger.plan.forecast.measured_units_per_second)
            report = run_match(
                _agents(subject, opponents, block, arm),
                wall_seed=block["wall_seed"],
                max_decisions=spec["max_decisions"],
            )
            stats = _match_statistics(report, block["subject_seat"])
            if arm == "baseline" and stats["candidate_forward_decisions"]:
                raise ValueError("Infinity baseline must perform zero auxiliary candidate forwards")
            name = f"pair-{block['pair_index']:05d}-{arm}"
            replay_sha = _archive_match(replays / (name + ".json.gz"), report)
            ledger.settle_elapsed("benchmark/" + name)
            _atomic_record(
                receipts / (name + ".json"),
                {
                    "plan_sha256": digest,
                    "pair_index": block["pair_index"],
                    "arm": arm,
                    "wall_seed": block["wall_seed"],
                    "subject_seat": block["subject_seat"],
                    "statistics": stats,
                    "match_sha256": report["match_sha256"],
                    "replay_sha256": replay_sha,
                    "policy_rng_final_sha256": report["policy_rng_final_sha256"],
                    "budget_ledger_head": ledger.head,
                    "budget_ledger_entries": len(ledger.entries),
                },
            )
            total = arms[arm]
            total["complete_matches"] += 1
            total["seat_counts"][block["subject_seat"]] += 1
            total["decisions"] += report["decision_count"]
            total["subject_decisions"] += stats["decisions"]
            total["subject_latency_sum"] += stats["decision_latency_sum"]
            for key in ("candidate_forward_decisions", "candidate_actions_scored", "reordered_decisions"):
                total[key] += stats[key]
            latencies[arm].extend(
                row["decision_latency_seconds"] for row in report["decisions"] if row["seat"] == block["subject_seat"]
            )
    for arm, total in arms.items():
        total["subject_decision_latency_mean_seconds"] = (
            total["subject_latency_sum"] / total["subject_decisions"] if total["subject_decisions"] else None
        )
        total["subject_decision_latency_p95_seconds"] = percentile(latencies[arm], 0.95)
    complete = sum(value["complete_matches"] for value in arms.values())
    if complete != spec["matches"]:
        raise RuntimeError("A reduced or incomplete benchmark cannot produce throughput evidence")
    elapsed = time.perf_counter() - started
    eligible = spec["mode"] == "paired" and arms["candidate"]["candidate_forward_decisions"] > 0
    return {
        "version": BENCHMARK_VERSION,
        "status": "PASS",
        "complete": True,
        "complete_matches": complete,
        "plan": spec,
        "benchmark_plan_sha256": digest,
        "runtime_identity": runtime,
        "arms": arms,
        "elapsed_seconds": elapsed,
        "matches_per_second": complete / elapsed,
        "decisions_per_second": sum(value["decisions"] for value in arms.values()) / elapsed,
        "m3_forecast_eligible": eligible,
        "m3_forecast_exclusion": None if eligible else "prior_only_or_candidate_auxiliary_not_exercised",
        "timing_scope": "model_loading_agents_complete_matches_replays_receipts_and_bookkeeping",
        "model_weights_updated": False,
        "test_walls_used": False,
    }
