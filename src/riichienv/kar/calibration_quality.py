from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

from .contracts import content_hash

QUALITY_VERSION = "preregistered_calibration_quality_v2_branches"
BRANCH_QUALITY_VERSION = "preregistered_response_branch_quality_v1"
HEADS = ("action", "response", "rank")
BRANCH_HEADS = ("response", "rank")
BRANCH_FALLBACK = "keep_masked_probability_unsupported_utility_minus_1_5"


@dataclass(frozen=True)
class BranchQualityCriterion:
    head: str
    event_id: int
    max_nll: float
    max_brier: float
    max_ece: float
    minimum_independent_games: int

    def __post_init__(self) -> None:
        if self.head not in BRANCH_HEADS or type(self.event_id) is not int or not 0 <= self.event_id < 21:
            raise ValueError("Branch criteria require response/rank and a frozen response event ID")
        for value, upper in ((self.max_nll, None), (self.max_brier, 2.0), (self.max_ece, 1.0)):
            if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                    or value < 0 or (upper is not None and value > upper)):
                raise ValueError("Explicit finite branch quality thresholds are required")
        if type(self.minimum_independent_games) is not int or self.minimum_independent_games < 1:
            raise ValueError("Branch evidence requires a positive independent validation-game minimum")


@dataclass(frozen=True)
class BranchQualityPolicy:
    """Explicit preregistration only. There are deliberately no production thresholds."""
    criteria: tuple[BranchQualityCriterion, ...]
    source_reference: str

    def __post_init__(self) -> None:
        if not self.criteria or any(not isinstance(value, BranchQualityCriterion) for value in self.criteria):
            raise ValueError("Register explicit branch criteria before inspecting cal_check")
        keys = [(value.head, value.event_id) for value in self.criteria]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate branch quality criterion")
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            raise ValueError("Branch criteria need their preregistration provenance")
        object.__setattr__(
            self, "criteria", tuple(sorted(self.criteria, key=lambda value: (value.head, value.event_id)))
        )

    def to_record(self) -> dict[str, Any]:
        record = {
            "version": BRANCH_QUALITY_VERSION,
            "criteria": [asdict(value) for value in self.criteria],
            "source_reference": self.source_reference,
            "partition": "cal_check",
            "metric_population": "conditional_on_observed_response_event",
            "ece_definition": "ten_bin_multiclass_top_label_confidence",
            "counting_unit": "distinct_complete_validation_game_id",
            "required_heads_for_value_support": list(BRANCH_HEADS),
            "missing_criterion_or_evidence": "unavailable",
            "fallback": BRANCH_FALLBACK,
            "uniform_error_certificate": False,
        }
        return {**record, "branch_quality_policy_sha256": content_hash(record)}

    @property
    def digest(self) -> str:
        return self.to_record()["branch_quality_policy_sha256"]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> BranchQualityPolicy:
        if not isinstance(record, Mapping) or record.get("version") != BRANCH_QUALITY_VERSION:
            raise ValueError("Branch quality policy version is missing or incompatible")
        result = cls(tuple(BranchQualityCriterion(**value) for value in record["criteria"]), record["source_reference"])
        if result.to_record() != dict(record):
            raise ValueError("Branch quality policy fields, thresholds or digest differ")
        return result


@dataclass(frozen=True)
class CalibrationQualityPolicy:
    max_nll: Mapping[str, float]
    max_brier: Mapping[str, float]
    max_ece: Mapping[str, float]
    minimum_head_observations: int
    minimum_gate_observations: int
    minimum_gate_retention: float
    source_reference: str
    branch_policy: BranchQualityPolicy | None = None

    def __post_init__(self) -> None:
        for name, upper in (("max_nll", None), ("max_brier", 2.0), ("max_ece", 1.0)):
            values = getattr(self, name)
            if set(values) != set(HEADS) or any(
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
                or (upper is not None and value > upper)
                for value in values.values()
            ):
                raise ValueError("Quality thresholds must explicitly cover all three heads")
            object.__setattr__(self, name, MappingProxyType({key: float(value) for key, value in values.items()}))
        for count in (self.minimum_head_observations, self.minimum_gate_observations):
            if type(count) is not int or count < 1:
                raise ValueError("Quality sample-count requirements must be positive integers")
        if (
            isinstance(self.minimum_gate_retention, bool)
            or not math.isfinite(self.minimum_gate_retention)
            or not 0 <= self.minimum_gate_retention <= 1
        ):
            raise ValueError("Invalid preregistered heldout gate retention requirement")
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            raise ValueError("Quality criteria need an explicit preregistration reference")
        if self.branch_policy is not None and not isinstance(self.branch_policy, BranchQualityPolicy):
            raise ValueError("Branch quality must be an explicit preregistered policy or unavailable")

    def to_record(self) -> dict[str, Any]:
        record = {
            "version": QUALITY_VERSION,
            "max_nll": dict(self.max_nll),
            "max_brier": dict(self.max_brier),
            "max_ece": dict(self.max_ece),
            "minimum_head_observations": self.minimum_head_observations,
            "minimum_gate_observations": self.minimum_gate_observations,
            "minimum_gate_retention": self.minimum_gate_retention,
            "source_reference": self.source_reference,
            "branch_policy": self.branch_policy.to_record() if self.branch_policy is not None else None,
        }
        return {**record, "quality_policy_sha256": content_hash(record)}

    @property
    def digest(self) -> str:
        return self.to_record()["quality_policy_sha256"]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CalibrationQualityPolicy:
        fields = set(cls.__dataclass_fields__)
        if "branch_policy" not in record or set(record) - fields - {"version", "quality_policy_sha256"}:
            raise ValueError("Unknown quality-policy field or missing branch contract")
        values = {key: item for key, item in record.items() if key in fields}
        values["branch_policy"] = (
            BranchQualityPolicy.from_record(record["branch_policy"]) if record["branch_policy"] is not None else None
        )
        value = cls(**values)
        if "version" in record or "quality_policy_sha256" in record:
            if value.to_record() != dict(record):
                raise ValueError("Quality-policy version, thresholds or digest changed")
        return value


def evaluate_branch_quality(metrics: Mapping[str, Any], policy: BranchQualityPolicy | None) -> dict[str, Any]:
    criteria = {} if policy is None else {(value.head, value.event_id): value for value in policy.criteria}
    measured = metrics.get("branch_metrics", {})
    branches = {}
    for event in range(21):
        heads = {}
        for head in BRANCH_HEADS:
            criterion = criteria.get((head, event))
            values = measured.get(head, {}).get(str(event), {})
            missing, failed = [], []
            if criterion is None:
                missing.append("preregistered_branch_criterion_missing")
            else:
                games = values.get("independent_games")
                count = values.get("count")
                if type(games) is not int or games < criterion.minimum_independent_games:
                    missing.append("independent_validation_games")
                if type(count) is not int or count < 1 or (type(games) is int and games > count):
                    missing.append("observations")
                for metric in ("nll", "brier", "ece"):
                    value = values.get(metric)
                    if value is None:
                        missing.append(metric)
                    elif (isinstance(value, bool) or not isinstance(value, (float, int))
                          or not math.isfinite(value) or value < 0 or value > getattr(criterion, "max_" + metric)):
                        failed.append(metric)
            heads[head] = {
                "status": "unavailable" if missing else "failed" if failed else "passed",
                "reasons": sorted(set(missing + failed)),
                "independent_games": values.get("independent_games"),
                "observations": values.get("count"),
            }
        states = [value["status"] for value in heads.values()]
        branches[str(event)] = {
            "status": "failed" if "failed" in states else "unavailable" if "unavailable" in states else "passed",
            "heads": heads,
        }
    return {"version": BRANCH_QUALITY_VERSION, "partition": "cal_check", "branches": branches,
            "branch_quality_policy_sha256": None if policy is None else policy.digest,
            "fallback": BRANCH_FALLBACK, "uniform_error_certificate": False}


def evaluate_quality(metrics: Mapping[str, Any], policy: CalibrationQualityPolicy | None) -> dict[str, Any]:
    if policy is None:
        return {
            "status": "unavailable",
            "reasons": ["preregistered_quality_policy_missing"],
            "partition": "cal_check",
            "uniform_error_certificate": False,
        }
    missing, failed = [], []
    for head in HEADS:
        values = metrics.get(head, {})
        count = values.get("count")
        if type(count) is not int or count < policy.minimum_head_observations:
            missing.append(head + "_observations")
        for metric, limits in (("nll", policy.max_nll), ("brier", policy.max_brier), ("ece", policy.max_ece)):
            value = values.get(metric)
            if value is None:
                missing.append(head + "_" + metric)
            elif (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
                or value > limits[head]
            ):
                failed.append(head + "_" + metric)
    count = metrics.get("gate_multi_discard_count")
    if type(count) is not int or count < policy.minimum_gate_observations:
        missing.append("multi_discard_observations")
    retention = metrics.get("gate_retention")
    if retention is None:
        missing.append("gate_retention")
    elif (
        isinstance(retention, bool)
        or not isinstance(retention, (float, int))
        or not math.isfinite(retention)
        or not policy.minimum_gate_retention <= retention <= 1
    ):
        failed.append("gate_retention")
    return {
        "status": "failed" if failed else "unavailable" if missing else "passed",
        "reasons": sorted(set(failed + missing)),
        "partition": "cal_check",
        "quality_policy_sha256": policy.digest,
        "uniform_error_certificate": False,
    }
