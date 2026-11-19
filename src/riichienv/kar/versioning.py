"""Shared, explicit identities for the RiichiEnv materialization of KAR-v1.2.2.

The design schema ID is a reference, not an alias for the complete host registry.
No legacy bytes, checkpoints, or manifests are upgraded by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from riichienv._riichienv import NATIVE_RULES_VERSION

from .actions import manifest_hash
from .calibration_quality import BRANCH_QUALITY_VERSION
from .contracts import MJSOUL_RULESET, REFERENCE_SCHEMA_ID, RESPONSE_MASK_VERSION, SCHEMA_ID, SCHEMA_OVERLAY_SHA256
from .event_semantics import PUBLIC_CLOCK_VERSION, TERMINAL_PUBLIC_VERSION
from .mjsoul import PROTOCOL_IDENTITY_VERSION
from .native_replay import NATIVE_REPLAY_VERSION
from .schema import schema_status

ARTIFACT_CONTRACT_VERSION = "kar_riichienv_contract_v4"
INPUT_PIPELINE_VERSION = "semantic_public_clock_frozen_decisions_v7_native_rules"
SCHEMA_MAPPING_VERSION = "riichienv_complete_v1_from_kar_v1_2_2_v1"
ARCHITECTURE_ID = "kar_kl_response_value_v1"
FEATURE_CONTRACT_VERSION = "visible_rule_features_v2_ryanmen_pairs"
RULE_FEATURE_MASK_VERSION = "sample_mask_v1"
TRAINING_PLAN_VERSION = "budget_ladder_v1"
LOSS_CONTRACT_VERSION = "sentinel_safe_v1_2_1"
MODEL_TIER_BASIS = "N_train_half_games"
TRAIN_CONFIG_VERSION = "explicit_train_split_scale_v5_decision_time"


def artifact_contract() -> dict[str, Any]:
    if NATIVE_RULES_VERSION != "riichienv_native_rules_v2_scoring":
        raise ValueError("Rebuild the native extension for the active rule contract")
    schema = schema_status()
    if (
        schema["status"] != "PASS"
        or schema["schema_id"] != SCHEMA_ID
        or schema.get("reference_design_schema_id") != REFERENCE_SCHEMA_ID
        or schema.get("reference_overlay_sha256") != SCHEMA_OVERLAY_SHA256
    ):
        raise ValueError("Complete host schema and design-reference identity are required")
    return {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "protocol_identity_version": PROTOCOL_IDENTITY_VERSION,
        "input_pipeline_version": INPUT_PIPELINE_VERSION,
        "native_rules_version": NATIVE_RULES_VERSION,
        "native_replay_version": NATIVE_REPLAY_VERSION,
        "public_clock_version": PUBLIC_CLOCK_VERSION,
        "terminal_public_version": TERMINAL_PUBLIC_VERSION,
        "branch_quality_version": BRANCH_QUALITY_VERSION,
        "schema_id": schema["schema_id"],
        "schema_hash": schema["schema_hash"],
        "schema_mapping_version": SCHEMA_MAPPING_VERSION,
        "reference_design_schema_id": REFERENCE_SCHEMA_ID,
        "reference_overlay_sha256": SCHEMA_OVERLAY_SHA256,
        "legacy_schema_compatibility": "not_claimed",
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "action_manifest_hash": manifest_hash(),
        "architecture_id": ARCHITECTURE_ID,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "response_mask_version": RESPONSE_MASK_VERSION,
        "rule_feature_mask_version": RULE_FEATURE_MASK_VERSION,
        "training_plan_version": TRAINING_PLAN_VERSION,
        "loss_contract_version": LOSS_CONTRACT_VERSION,
        "model_tier_basis": MODEL_TIER_BASIS,
    }


def validate_artifact_contract(record: Mapping[str, Any], *, context: str = "artifact") -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{context} requires an explicit version contract")
    mismatches = [key for key, value in artifact_contract().items() if record.get(key) != value]
    if mismatches:
        raise ValueError(f"{context} version contract differs: {', '.join(mismatches)}")


def game_count_contract(
    available_split_counts: Mapping[str, int], exported_split_counts: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Keep the audited pre-split cohort distinct from an actual exported subset."""
    if not isinstance(available_split_counts, Mapping) or (
        exported_split_counts is not None and not isinstance(exported_split_counts, Mapping)
    ):
        raise ValueError("Game counts require explicit split evidence")
    available = dict(available_split_counts)
    exported = dict(available if exported_split_counts is None else exported_split_counts)
    for values in (available, exported):
        if set(values) != {"train", "validation", "test"} or any(
            type(count) is not int or count < 0 for count in values.values()
        ):
            raise ValueError("Game counts require exact nonnegative train/validation/test counts")
    if any(exported[split] > available[split] for split in available):
        raise ValueError("Exported game counts exceed the audited cohort")
    return {
        "N_eligible_half_games": sum(available.values()),
        "N_train_half_games": exported["train"],
        "available_split_counts": available,
        "exported_split_counts": exported,
        "model_tier_basis": MODEL_TIER_BASIS,
    }
