from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .calibration_roles import CALIBRATION_SPLIT_SALT, calibration_is_fit
from .contracts import content_hash, game_split
from .policy import fixed_tau_grid
from .runtime_contracts import sha256_value
from .statistics import GameClusters

TAU_SELECTION_VERSION = "frozen_validation_tau_grid_v4"
TAU_SELECTION_POLICY_VERSION = "preregistered_tau_selection_v2"
TAU_MATCH_PLAN_VERSION = "registered_validation_tau_matches_v4"
TAU_GAIN_BOUND_VERSION = "paired_rank_hoeffding_union_v1"


@dataclass(frozen=True)
class TauSelectionPolicy:
    minimum_game_clusters: int = 2
    minimum_utility_gain: float = 0.0
    confidence: float = 0.95
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 17
    source_reference: str = "fixed conservative paired-validation criterion,preregistered before outcomes"

    def __post_init__(self) -> None:
        if type(self.minimum_game_clusters) is not int or self.minimum_game_clusters < 2:
            raise ValueError("Reproducible validation needs at least two independent scenario clusters")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (self.minimum_utility_gain, self.confidence)
        ):
            raise ValueError("Selection criteria must be finite preregistered values")
        if self.minimum_utility_gain < 0 or not 0 < self.confidence < 1:
            raise ValueError("Finite tau must improve over the infinity baseline")
        if (
            type(self.bootstrap_replicates) is not int
            or self.bootstrap_replicates < 1
            or type(self.bootstrap_seed) is not int
        ):
            raise ValueError("Register the exact paired-cluster resampling recipe")
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            raise ValueError("Selection criteria need a preregistration reference")

    def to_record(self) -> dict[str, Any]:
        value = {"version": TAU_SELECTION_POLICY_VERSION, **asdict(self)}
        return {**value, "selection_policy_sha256": content_hash(value)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TauSelectionPolicy:
        fields = set(cls.__dataclass_fields__)
        if set(record) - fields - {"version", "selection_policy_sha256"}:
            raise ValueError("Unknown tau-selection criterion")
        value = cls(**{key: item for key, item in record.items() if key in fields})
        if ("version" in record or "selection_policy_sha256" in record) and value.to_record() != dict(record):
            raise ValueError("Tau-selection criteria were modified or use a superseded admission rule")
        return value


def cal_fit_scenario_manifest(
    game_ids: Sequence[str],
    *,
    validation_manifest_sha256: str | None,
    calibration_split_salt: str,
) -> dict[str, Any]:
    if any(not isinstance(game, str) or not game for game in game_ids) or len(game_ids) != len(set(game_ids)):
        raise ValueError("Register distinct complete cal_fit scenario IDs")
    record = {
        "version": "complete_cal_fit_scenarios_v1",
        "partition": "cal_fit",
        "validation_manifest_sha256": validation_manifest_sha256,
        "calibration_split_salt": calibration_split_salt,
        "game_ids": sorted(game_ids),
        "independent_games": len(game_ids),
    }
    return {**record, "scenario_manifest_sha256": content_hash(record)}


def _gain_bound(mean: float, count: int, candidates: int, policy: TauSelectionPolicy) -> dict[str, Any]:
    """Simultaneous one-sided bounds for the preregistered finite grid.

    Each independent paired rank-utility difference lies in [-3, 3]. Hoeffding
    gives exp(-n * radius**2 / 18); a union bound allocates (1-confidence)/K
    to each of K finite candidates. Pairing/correlation between candidates is
    allowed. Independence between predeclared scenario blocks is required.
    This bounds expected paired utility, not statewise model error.
    """
    if count < 1 or candidates < 1 or not -3.0 <= mean <= 3.0:
        raise ValueError("A finite gain bound requires independent paired rank utilities and a finite grid")
    radius = math.sqrt(18.0 * (math.log(candidates) - math.log1p(-policy.confidence)) / count)
    return {
        "version": TAU_GAIN_BOUND_VERSION,
        "difference_range": [-3.0, 3.0],
        "independent_games": count,
        "finite_candidates": candidates,
        "confidence": policy.confidence,
        "per_candidate_failure_probability": (1.0 - policy.confidence) / candidates,
        "mean": mean,
        "radius": radius,
        "lower": max(-3.0, mean - radius),
        "assumption": "independent_preregistered_scenario_blocks",
        "scope": "expected_paired_utility_not_statewise_error",
    }


def paired_tau_evidence(
    outcomes: Mapping[float, Mapping[str, float]],
    grid: Sequence[float],
    policy: TauSelectionPolicy,
) -> dict[str, dict[str, Any]]:
    temperatures = fixed_tau_grid(grid)
    if set(outcomes) != set(temperatures):
        raise ValueError("Paired selection requires every registered candidate and the infinity baseline")
    baseline = outcomes[math.inf]
    games = tuple(sorted(baseline))
    if not games:
        raise ValueError("Paired selection requires independent validation scenarios")
    means = {}
    for tau in temperatures:
        if tuple(sorted(outcomes[tau])) != games:
            raise ValueError("Every registered candidate must use the same complete paired scenarios")
        values = tuple(outcomes[tau][game] for game in games)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not -1.5 <= value <= 1.5
            for value in values
        ):
            raise ValueError("Paired outcomes must use bounded average-rank utility")
        means[tau] = math.fsum(values) / len(games)
    candidates = sum(math.isfinite(tau) for tau in temperatures)
    evidence = {}
    for tau in temperatures:
        clusters = GameClusters()
        for game in games:
            clusters.add(game, "candidate_minus_baseline", outcomes[tau][game] - baseline[game])
        interval = clusters.interval(
            "candidate_minus_baseline",
            seed=policy.bootstrap_seed,
            replicates=policy.bootstrap_replicates,
            confidence=policy.confidence,
        )
        bound = None if math.isinf(tau) else _gain_bound(means[tau] - means[math.inf], len(games), candidates, policy)
        passed = (
            bound is not None
            and len(games) >= policy.minimum_game_clusters
            and bound["lower"] > policy.minimum_utility_gain
        )
        evidence["infinity" if math.isinf(tau) else str(tau)] = {
            "version": TAU_GAIN_BOUND_VERSION,
            "paired_utility_interval": interval,
            "bootstrap_role": "descriptive_only_not_admission",
            "gain_bound": bound,
            "reproducible_gain": passed,
            "status": "infinity_baseline" if math.isinf(tau) else "passed" if passed else "no_reproducible_gain",
            "unit": "independent_validation_scenario_game_id",
            "uniform_error_certificate": False,
        }
    return evidence


def validate_finite_tau_selection(
    selection: Mapping[str, Any],
    *,
    expected_game_ids: Sequence[str],
) -> TauSelectionPolicy:
    unsigned = {key: value for key, value in selection.items() if key != "tau_selection_sha256"}
    if (
        selection.get("version") != TAU_SELECTION_VERSION
        or selection.get("diagnostic_fixture") is not False
        or selection.get("tau_selection_sha256") != content_hash(unsigned)
    ):
        raise ValueError("Finite selection must be an intact production validation artifact")
    policy_record = selection.get("selection_policy")
    if not isinstance(policy_record, Mapping):
        raise ValueError("Production selection requires versioned preregistered criteria")
    policy = TauSelectionPolicy.from_record(policy_record)
    if dict(policy_record) != policy.to_record():
        raise ValueError("Production selection requires current versioned admission criteria")
    source, salt = selection.get("validation_manifest_sha256"), selection.get("calibration_split_salt")
    if not isinstance(source, str) or not isinstance(salt, str) or salt != CALIBRATION_SPLIT_SALT:
        raise ValueError("Production gain evidence requires the frozen validation source and role split")
    sha256_value(source, "validation_manifest_sha256")
    games = selection.get("game_ids")
    expected_manifest = cal_fit_scenario_manifest(
        expected_game_ids,
        validation_manifest_sha256=source,
        calibration_split_salt=salt,
    )
    if (
        not games
        or games != expected_manifest["game_ids"]
        or selection.get("scenario_manifest") != expected_manifest
        or selection.get("selection_partition") != "cal_fit"
        or any(
            game_split(game, "kar-riichienv-v1-games") != "validation" or not calibration_is_fit(game) for game in games
        )
    ):
        raise ValueError("Validation gain requires the complete registered cal_fit scenario manifest")
    grid = selection.get("grid")
    if not isinstance(grid, list) or any(not isinstance(item, str) for item in grid):
        raise ValueError("Validation selection lacks the complete registered tau grid")
    temperatures = fixed_tau_grid(
        [float(item) for item in grid],
        max_prior_amplification=selection.get("max_prior_amplification"),
    )
    if grid != ["infinity" if math.isinf(tau) else str(tau) for tau in temperatures]:
        raise ValueError("Validation grid must preserve every canonical preregistered candidate")
    means, evidence = selection.get("mean_utility"), selection.get("paired_validation_evidence")
    if (
        not isinstance(means, Mapping)
        or not isinstance(evidence, Mapping)
        or set(means) != set(grid)
        or set(evidence) != set(grid)
    ):
        raise ValueError("Validation selection must retain outcomes and evidence for the entire registered grid")
    if any(
        isinstance(mean, bool) or not isinstance(mean, (float, int)) or not -1.5 <= mean <= 1.5
        for mean in means.values()
    ):
        raise ValueError("Validation means must use bounded average-rank utility")
    candidates = len(grid) - 1
    eligible = [math.inf]
    for tau, key in zip(temperatures, grid):
        item = evidence[key]
        if not isinstance(item, Mapping):
            raise ValueError("Invalid paired validation evidence")
        expected_bound = (
            None if math.isinf(tau) else _gain_bound(means[key] - means["infinity"], len(games), candidates, policy)
        )
        passed = (
            expected_bound is not None
            and len(games) >= policy.minimum_game_clusters
            and expected_bound["lower"] > policy.minimum_utility_gain
        )
        status = "infinity_baseline" if math.isinf(tau) else "passed" if passed else "no_reproducible_gain"
        if (
            item.get("version") != TAU_GAIN_BOUND_VERSION
            or item.get("bootstrap_role") != "descriptive_only_not_admission"
            or item.get("gain_bound") != expected_bound
            or item.get("reproducible_gain") is not passed
            or item.get("status") != status
            or item.get("unit") != "independent_validation_scenario_game_id"
            or item.get("uniform_error_certificate") is not False
        ):
            raise ValueError("Paired validation gain bound does not match the registered grid, means and sample count")
        interval = item.get("paired_utility_interval")
        if not isinstance(interval, Mapping):
            raise ValueError("Missing descriptive paired validation interval")
        lower, upper, mean = interval.get("lower"), interval.get("upper"), interval.get("mean")
        if (
            interval.get("independent_games") != len(games)
            or any(
                isinstance(value, bool) or not isinstance(value, (float, int)) or not -3.0 <= value <= 3.0
                for value in (lower, upper, mean)
            )
            or lower > upper
            or not math.isclose(mean, means[key] - means["infinity"], rel_tol=1e-12, abs_tol=1e-12)
            or interval.get("seed") != policy.bootstrap_seed
            or interval.get("confidence") != policy.confidence
            or interval.get("bootstrap_replicates") != policy.bootstrap_replicates
        ):
            raise ValueError("Descriptive interval differs from its paired scenarios or registered resampling recipe")
        if passed:
            eligible.append(tau)
    selected = max(eligible, key=lambda tau: (means["infinity" if math.isinf(tau) else str(tau)], tau))
    if (
        math.isinf(selected)
        or selection.get("selected_tau") != str(selected)
        or selection.get("reproducible_gain") is not True
    ):
        raise ValueError("No chosen finite policy passed the preregistered simultaneous validation-gain gate")
    return policy
