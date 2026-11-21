from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .calibration_roles import CALIBRATION_PARTITION_VERSION, CALIBRATION_SPLIT_SALT
from .contracts import content_hash
from .kan_transactions import KAN_TRANSACTION_VERSION
from .versioning import ARCHITECTURE_ID, artifact_contract

UTILITY_VERSION = "average_rank_utility_v1"
POLICY_VERSION = "restricted_kl_runtime_v3_event_ovr"
SUPPORT_VERSION = "train_event_and_candidate_action_support_v4_min_candidate_sources"
CALIBRATION_VERSION = "heldout_temperatures_log_gate_v6_event_ovr"
CALIBRATED_POLICY_IDENTITY_VERSION = "calibrated_policy_identity_v3_event_ovr"
BOUND_VERSION = "one_sided_v1_2_1"
UTILITIES = (1.5, 0.5, -0.5, -1.5)
STATIC_RECIPE_SHA256 = content_hash({"predictor_kind": "static_logits", "version": 1})


def sha256_value(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be an explicit SHA-256 value")
    return value


def runtime_contract() -> dict[str, Any]:
    return {
        **artifact_contract(),
        "architecture_id": ARCHITECTURE_ID,
        "utility_version": UTILITY_VERSION,
        "utility_values": list(UTILITIES),
        "policy_version": POLICY_VERSION,
        "support_version": SUPPORT_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "calibration_partition_version": CALIBRATION_PARTITION_VERSION,
        "calibration_split_salt": CALIBRATION_SPLIT_SALT,
        "decision_bound_version": BOUND_VERSION,
        "kan_transaction_version": KAN_TRANSACTION_VERSION,
    }


@dataclass(frozen=True)
class RuntimeBinding:
    model_weights_sha256: str
    train_manifest_sha256: str
    schema_hash: str
    ruleset_hash: str
    action_manifest_hash: str
    inference_recipe_sha256: str = STATIC_RECIPE_SHA256

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            sha256_value(value, key)
        current = runtime_contract()
        if any(getattr(self, key) != current[key] for key in ("schema_hash", "ruleset_hash", "action_manifest_hash")):
            raise ValueError("Runtime model/data contracts differ from active frozen contracts")

    def to_record(self) -> dict[str, str]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return content_hash(self.to_record())

    @classmethod
    def current(
        cls,
        model_weights_sha256: str,
        train_manifest_sha256: str,
        *,
        inference_recipe_sha256: str = STATIC_RECIPE_SHA256,
    ) -> RuntimeBinding:
        contract = runtime_contract()
        return cls(
            model_weights_sha256,
            train_manifest_sha256,
            contract["schema_hash"],
            contract["ruleset_hash"],
            contract["action_manifest_hash"],
            inference_recipe_sha256,
        )
