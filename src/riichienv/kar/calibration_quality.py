from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .contracts import content_hash

QUALITY_VERSION = "preregistered_calibration_quality_v1"
HEADS = ("action", "response", "rank")


@dataclass(frozen=True)
class CalibrationQualityPolicy:
    max_nll: Mapping[str, float]
    max_brier: Mapping[str, float]
    max_ece: Mapping[str, float]
    minimum_head_observations: int
    minimum_gate_observations: int
    minimum_gate_retention: float
    source_reference: str

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
        }
        return {**record, "quality_policy_sha256": content_hash(record)}

    @property
    def digest(self) -> str:
        return self.to_record()["quality_policy_sha256"]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CalibrationQualityPolicy:
        fields = set(cls.__dataclass_fields__)
        if set(record) - fields - {"version", "quality_policy_sha256"}:
            raise ValueError("Unknown quality-policy field")
        value = cls(**{key: item for key, item in record.items() if key in fields})
        if "version" in record or "quality_policy_sha256" in record:
            if value.to_record() != dict(record):
                raise ValueError("Quality-policy version, thresholds or digest changed")
        return value


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
