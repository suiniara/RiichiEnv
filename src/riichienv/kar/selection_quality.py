from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import content_hash
from .statistics import GameClusters

TAU_SELECTION_VERSION = "frozen_validation_tau_grid_v3"


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
            isinstance(value, bool) or not math.isfinite(value)
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
        if not self.source_reference.strip():
            raise ValueError("Selection criteria need a preregistration reference")

    def to_record(self) -> dict[str, Any]:
        value = {"version": "preregistered_tau_selection_v1", **asdict(self)}
        return {**value, "selection_policy_sha256": content_hash(value)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TauSelectionPolicy:
        fields = set(cls.__dataclass_fields__)
        if set(record) - fields - {"version", "selection_policy_sha256"}:
            raise ValueError("Unknown tau-selection criterion")
        value = cls(**{key: item for key, item in record.items() if key in fields})
        if ("version" in record or "selection_policy_sha256" in record) and value.to_record() != dict(record):
            raise ValueError("Tau-selection criteria were modified")
        return value


def paired_tau_evidence(
    outcomes: Mapping[float, Mapping[str, float]],
    grid: Sequence[float],
    policy: TauSelectionPolicy,
) -> dict[str, dict[str, Any]]:
    if math.inf not in outcomes:
        raise ValueError("Paired selection requires the unchanged checkpoint's infinity baseline")
    baseline = outcomes[math.inf]
    evidence = {}
    for tau in grid:
        clusters = GameClusters()
        for game, value in outcomes[tau].items():
            clusters.add(game, "candidate_minus_baseline", value - baseline[game])
        interval = clusters.interval(
            "candidate_minus_baseline",
            seed=policy.bootstrap_seed,
            replicates=policy.bootstrap_replicates,
            confidence=policy.confidence,
        )
        lower = interval["lower"]
        passed = (
            not math.isinf(tau)
            and interval["independent_games"] >= policy.minimum_game_clusters
            and lower is not None
            and lower > policy.minimum_utility_gain
        )
        evidence["infinity" if math.isinf(tau) else str(tau)] = {
            "paired_utility_interval": interval,
            "reproducible_gain": passed,
            "status": "infinity_baseline" if math.isinf(tau) else "passed" if passed else "no_reproducible_gain",
            "unit": "independent_validation_scenario_game_id",
            "uniform_error_certificate": False,
        }
    return evidence


def validate_finite_tau_selection(selection: Mapping[str, Any]) -> TauSelectionPolicy:
    unsigned = {key: value for key, value in selection.items() if key != "tau_selection_sha256"}
    if (
        selection.get("version") != TAU_SELECTION_VERSION
        or selection.get("diagnostic_fixture") is not False
        or selection.get("tau_selection_sha256") != content_hash(unsigned)
    ):
        raise ValueError("Finite selection must be an intact production validation artifact")
    policy = TauSelectionPolicy.from_record(selection["selection_policy"])
    tau = float(selection["selected_tau"])
    if not math.isfinite(tau) or tau <= 0 or selection.get("reproducible_gain") is not True:
        raise ValueError("No finite policy passed the preregistered validation-gain gate")
    key = str(tau)
    evidence = selection["paired_validation_evidence"][key]
    interval = evidence["paired_utility_interval"]
    games = selection["game_ids"]
    if (
        evidence.get("reproducible_gain") is not True
        or len(games) != len(set(games))
        or interval.get("independent_games") != len(games)
        or len(games) < policy.minimum_game_clusters
    ):
        raise ValueError("Validation-gain evidence lacks the registered independent game clusters")
    lower, upper, mean = interval["lower"], interval["upper"], interval["mean"]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            for value in (lower, upper, mean)
        )
        or lower > upper
        or lower <= policy.minimum_utility_gain
        or not math.isclose(mean, selection["mean_utility"][key] - selection["mean_utility"]["infinity"], abs_tol=1e-10)
    ):
        raise ValueError("Paired validation interval does not support the chosen finite policy")
    if (
        interval.get("seed") != policy.bootstrap_seed
        or interval.get("confidence") != policy.confidence
        or interval.get("bootstrap_replicates") != policy.bootstrap_replicates
    ):
        raise ValueError("Validation confidence recipe changed after preregistration")
    return policy
