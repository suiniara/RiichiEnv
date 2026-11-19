from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..agent import KARAgent, run_match
from ..authorization import update_implementation_sha256
from ..budget import BudgetLedger, BudgetPlan
from ..calibration import calibrated_policy_identity, choose_validation_tau
from ..contracts import content_hash
from ..evaluation_authorization import require_evaluation_authorization
from ..experiments import independent_seed
from ..policy import PolicyOptions, fixed_tau_grid
from ..runtime import RuntimePolicy
from ..selection_quality import TauSelectionPolicy
from .deployment import FrozenRelease
from .inference import FrozenModelPredictor


def validation_tau_plan(
    subject: FrozenRelease,
    opponents: tuple[FrozenRelease, ...],
    *,
    seed: int,
    grid: tuple[float, ...] = (math.inf, 1.0, 0.5, 0.25),
    selection_policy: TauSelectionPolicy | None = None,
) -> dict[str, Any]:
    if subject.calibration is None or subject.support is None:
        raise ValueError("Validation tau selection needs calibrated prior and train support")
    if not subject.calibration.empirical_check_valid:
        raise ValueError("Calibration-quality checks did not pass on independent cal_check")
    values = fixed_tau_grid(grid, max_prior_amplification=subject.options.max_prior_amplification)
    if any(math.isfinite(tau) for tau in values) and not any(
        subject.support.value_supported(subject.calibration.value_branch_statuses)
    ):
        raise ValueError(
            "Finite tau selection requires at least one train-supported, quality-qualified response branch; "
            "keep the calibrated prior"
        )
    split = subject.calibration.split
    identity = calibrated_policy_identity(
        subject.calibration.binding,
        subject.support,
        subject.calibration,
        max_prior_amplification=subject.options.max_prior_amplification,
    )
    record = {
        "version": "registered_validation_tau_matches_v3",
        "selection_policy": (selection_policy or TauSelectionPolicy()).to_record(),
        "policy_identity": identity,
        "source_sha256": update_implementation_sha256(),
        "subject_binding_sha256": content_hash(subject.record["binding"]),
        "opponent_release_ids": [value.record["release_sha256"] for value in opponents],
        "validation_manifest_sha256": split.validation_manifest_sha256,
        "calibration_split_salt": split.salt,
        "scenario_game_ids": list(split.fit_games),
        "grid": ["infinity" if math.isinf(value) else value for value in values],
        "master_seed": seed,
        "inference_device_type": next(subject.model.parameters()).device.type,
        "scenario_definition": "independent simulated walls seeded by cal_fit ID,not replayed source hands",
        "matches": len(split.fit_games) * len(values),
        "partition": "cal_fit",
    }
    return {**record, "plan_sha256": content_hash(record)}


def _tau_outcomes(
    subject: FrozenRelease,
    opponents: tuple[FrozenRelease, ...],
    output: Path,
    *,
    ledger: BudgetLedger,
    seed: int,
    values: tuple[float, ...],
) -> dict[float, dict[str, float]]:
    assert subject.calibration is not None and subject.support is not None
    outcomes: dict[float, dict[str, float]] = {tau: {} for tau in values}
    for index, game in enumerate(subject.calibration.split.fit_games):
        seat, wall = index % 4, independent_seed(seed, "validation_tau_wall", game)
        for tau in values:
            ledger.reserve_next(ledger.plan.forecast.safety_multiplier / ledger.plan.forecast.measured_units_per_second)
            predictor = FrozenModelPredictor(
                subject.model,
                subject.record["binding"]["train_manifest_sha256"],
                use_rule_features=subject.record["use_rule_features"],
            )
            policy = RuntimePolicy(
                predictor,
                options=PolicyOptions(
                    "prior_only" if math.isinf(tau) else "empirical_only", tau, subject.options.max_prior_amplification
                ),
                support=subject.support,
                calibration=subject.calibration,
                seed=independent_seed(seed, "validation_tau_subject", game, str(tau)),
            )
            agents = {seat: KARAgent(seat, policy)}
            for offset, opponent in enumerate(opponents, 1):
                other = (seat + offset) % 4
                agents[other] = KARAgent(
                    other, opponent.policy(seed=independent_seed(seed, "validation_tau_opponent", game, offset))
                )
            report = run_match(agents, wall_seed=wall)
            if report.get("status") != "PASS" or not report.get("complete"):
                raise RuntimeError("Incomplete validation game cannot select tau")
            outcomes[tau][game] = 2.5 - report["ranks"][seat]
            receipt = {
                "game_id": game,
                "tau": "infinity" if math.isinf(tau) else tau,
                "subject_seat": seat,
                "wall_seed": wall,
                "subject_utility": outcomes[tau][game],
                "match_sha256": report["match_sha256"],
                "policy_rng_final_sha256": report["policy_rng_final_sha256"],
            }
            with (output / "validation_matches.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, sort_keys=True) + "\n")
            ledger.settle_elapsed(f"validation_tau/{game}/{tau}")
    return outcomes


def select_validation_tau(
    subject: FrozenRelease,
    opponents: tuple[FrozenRelease, ...],
    output: Path,
    *,
    budget: BudgetPlan,
    seed: int,
    plan_sha256: str,
    authorization: dict[str, Any] | None = None,
    grid: tuple[float, ...] = (math.inf, 1.0, 0.5, 0.25),
    selection_policy: TauSelectionPolicy | None = None,
) -> dict[str, Any]:
    require_evaluation_authorization(authorization, operation="select_validation_tau", plan_sha256=plan_sha256)
    plan = validation_tau_plan(subject, opponents, seed=seed, grid=grid, selection_policy=selection_policy)
    if (
        plan["plan_sha256"] != plan_sha256
        or budget.forecast.recipe_sha256 != plan_sha256
        or budget.forecast.work_units < plan["matches"]
    ):
        raise ValueError("Tau selection must use preregistered validation-only work and cost")
    if budget.phase not in {"M1", "M2"} or len(opponents) != 3:
        raise ValueError("Validation selection cannot consume test/M3 budget")
    assert subject.calibration is not None and subject.support is not None
    split = subject.calibration.split
    if not split.fit_games:
        raise ValueError("No validation-fit scenarios; never substitute cal_check/test")
    output.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger(output / "costs.jsonl", budget)
    with ledger.record_compute("validation_tau_selection"):
        values = fixed_tau_grid(grid, max_prior_amplification=subject.options.max_prior_amplification)
        outcomes = _tau_outcomes(subject, opponents, output, ledger=ledger, seed=seed, values=values)
        choice = choose_validation_tau(
            outcomes,
            split=split,
            binding=subject.calibration.binding,
            grid=values,
            max_prior_amplification=subject.options.max_prior_amplification,
            diagnostic_fixture=False,
            policy_identity=plan["policy_identity"],
            selection_policy=TauSelectionPolicy.from_record(plan["selection_policy"]),
        )
    result = {
        "status": "PASS",
        "plan": plan,
        "tau_selection": choice,
        "budget": ledger.state_dict(),
        "test_games_used": False,
        "cal_check_used_for_tuning": False,
        "note": "Infinity is a valid result; do not force a finite-policy strength claim",
    }
    result["selection_job_sha256"] = content_hash(result)
    (output / "tau-selection.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
