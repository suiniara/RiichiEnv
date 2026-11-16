from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import content_hash

READINESS_VERSION = "kar_feature_readiness_v1"
CORE_REQUIREMENTS = tuple(f"P{index:02d}" for index in range(1, 13)) + ("U01",)
FULL_REMAINING_REQUIREMENTS = tuple(f"T{index:02d}" for index in range(1, 11))


def feature_readiness(
    checks: Mapping[str, bool],
    *,
    evidence: Mapping[str, str],
    source_hash: str,
    data_eligible: bool = False,
    resource_ready: bool = False,
) -> dict[str, Any]:
    if not source_hash or any(type(value) is not bool for value in checks.values()):
        raise ValueError("Readiness needs an explicit code identity and boolean evidence results")
    required = CORE_REQUIREMENTS + FULL_REMAINING_REQUIREMENTS
    missing = [key for key in required if checks.get(key) is not True or not evidence.get(key)]
    report = {
        "version": READINESS_VERSION,
        "source_hash": source_hash,
        "checks": dict(checks),
        "evidence": dict(evidence),
        "missing": missing,
        "core_feature_ready": not any(key in missing for key in CORE_REQUIREMENTS),
        "full_feature_ready": not missing,
        "data_eligible": data_eligible,
        "resource_ready": resource_ready,
        "parameter_updates_authorized": False,
        "note": "Data eligibility and feature readiness never grant user update authorization",
    }
    return {**report, "readiness_sha256": content_hash(report)}


def validate_readiness(document: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in document.items() if key != "readiness_sha256"}
    if document.get("version") != READINESS_VERSION or document.get("readiness_sha256") != content_hash(unsigned):
        raise ValueError("Readiness artifact version or integrity mismatch")
    rebuilt = feature_readiness(
        document["checks"],
        evidence=document["evidence"],
        source_hash=document["source_hash"],
        data_eligible=document["data_eligible"],
        resource_ready=document["resource_ready"],
    )
    if rebuilt != dict(document):
        raise ValueError("Readiness flags do not follow the declared evidence")
