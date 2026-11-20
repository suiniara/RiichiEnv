"""Pure metadata checks against the frozen design and explicit user count ruling."""

from dataclasses import replace

import pytest

from riichienv.kar.budget import BudgetPlan, CostForecast, choose_training_tier
from riichienv.kar.contracts import REFERENCE_SCHEMA_ID, RESPONSE_MASK_VERSION, SCHEMA_ID, model_tier
from riichienv.kar.learning.export import learning_contract
from riichienv.kar.learning.training_config import ProductionTrainConfig, training_recipe_sha256
from riichienv.kar.release import complete_release_contract
from riichienv.kar.runtime_contracts import runtime_contract
from riichienv.kar.schema import SCHEMA_CANONICAL_SHA256, schema_status
from riichienv.kar.versioning import (
    FEATURE_CONTRACT_VERSION,
    MODEL_TIER_BASIS,
    RULE_FEATURE_MASK_VERSION,
    TRAINING_PLAN_VERSION,
    artifact_contract,
    game_count_contract,
    validate_artifact_contract,
)


def test_design_reference_is_not_an_alias_or_an_overlay_hash_checkpoint():
    schema = schema_status()
    contract = artifact_contract()
    assert schema["status"] == "PASS"
    assert contract["schema_id"] == SCHEMA_ID == "kar-riichienv-v1"
    assert contract["reference_design_schema_id"] == REFERENCE_SCHEMA_ID == "kar-v1.2.2"
    assert contract["schema_hash"] == SCHEMA_CANONICAL_SHA256
    assert contract["reference_overlay_sha256"] != contract["schema_hash"]
    assert contract["legacy_schema_compatibility"] == "not_claimed"
    assert "riichienv_complete_v1" in contract["schema_mapping_version"]
    assert schema["embedding_rows"] == 944 and schema["token_bytes"] == 139
    with pytest.raises(ValueError, match="schema_id"):
        validate_artifact_contract({**contract, "schema_id": REFERENCE_SCHEMA_ID})


def test_training_and_inference_use_the_identical_public_mask_producer():
    from riichienv.kar import response_mask, runtime  # noqa: PLC0415
    from riichienv.kar.learning import export  # noqa: PLC0415

    assert export.public_rule_response_mask is response_mask.public_rule_response_mask
    assert runtime.public_rule_response_mask is response_mask.public_rule_response_mask


def test_training_and_inference_share_all_explicit_versions():
    current = artifact_contract()
    for source in (learning_contract(), runtime_contract()):
        assert {key: source[key] for key in current} == current
    assert current["feature_contract_version"] == FEATURE_CONTRACT_VERSION == "visible_rule_features_v2_ryanmen_pairs"
    assert current["rule_feature_mask_version"] == RULE_FEATURE_MASK_VERSION == "sample_mask_v1"
    assert current["training_plan_version"] == TRAINING_PLAN_VERSION == "budget_ladder_v1"
    assert current["response_mask_version"] == RESPONSE_MASK_VERSION == "public_rule_response_mask_v4_meld_capacity"


@pytest.mark.parametrize("field", tuple(artifact_contract()))
@pytest.mark.parametrize("change", ["missing", "wrong"])
def test_every_missing_or_mismatched_artifact_identity_is_rejected(field, change):
    record = artifact_contract()
    if change == "missing":
        record.pop(field)
    else:
        record[field] = "unversioned-or-foreign"
    with pytest.raises(ValueError, match="version contract differs"):
        validate_artifact_contract(record)


def _config(train, eligible, tier="small"):
    arguments = {
        "phase": "M2",
        "model_tier": tier,
        "N_train_half_games": train,
        "N_eligible_half_games": eligible,
        "planned_epochs": 2,
        "device": "cpu",
        "precision": "fp32",
    }
    recipe = training_recipe_sha256(arguments)
    forecast = CostForecast("M2", 100, 100.0, 0.0, 0.0, "synthetic unit metadata,not measured production", recipe)
    return ProductionTrainConfig(**arguments, budget=BudgetPlan("M2", 650.0, 0.0, forecast))


@pytest.mark.parametrize(
    "train,eligible,tier",
    [(29999, 40000, "small"), (30000, 40000, "medium"), (49999, 70000, "medium"), (50000, 70000, "large")],
)
def test_one_train_split_count_drives_tier_configuration_forecast_and_publication(train, eligible, tier):
    # The user explicitly chose design section 17's train basis over 7.5/14.2.
    assert MODEL_TIER_BASIS == "N_train_half_games" and model_tier(train) == tier
    config = _config(train, eligible, tier)
    assert choose_training_tier(train, {tier: config.budget.forecast}, budget_spent_usd=0.0) == tier
    assert ProductionTrainConfig.from_record(config.to_record()) == config
    manifest = {
        "scope": {**learning_contract(), "split_salt": "synthetic-split"},
        "manifest_sha256": "d" * 64,
        "rho_E": 1.0,
        "rho_V": 1.0,
        "N_train_half_games": train,
        "N_eligible_half_games": eligible,
    }
    release = complete_release_contract(
        training=config.to_record(), data_manifest=manifest, budget=config.budget.to_record()
    )
    assert release["response_mask_version"] == RESPONSE_MASK_VERSION
    assert release["training"]["model_tier_basis"] == MODEL_TIER_BASIS
    assert release["training"]["N_train_half_games"] == train
    assert release["training"]["N_eligible_half_games"] == eligible
    assert release["rule_feature_mask_probability"] == 0.25
    with pytest.raises(ValueError, match="scale or recipe"):
        complete_release_contract(
            training=config.to_record(),
            data_manifest={**manifest, "N_train_half_games": train - 1},
            budget=config.budget.to_record(),
        )


def test_exported_subset_is_not_relabelled_as_the_full_eligible_cohort():
    scale = game_count_contract({"train": 900, "validation": 48, "test": 42}, {"train": 6, "validation": 2, "test": 0})
    assert scale["N_eligible_half_games"] == 990 and scale["N_train_half_games"] == 6
    assert scale["available_split_counts"]["train"] == 900
    assert scale["exported_split_counts"]["train"] == 6
    with pytest.raises(ValueError):
        game_count_contract({"train": 900, "validation": 48, "test": 42}, {"train": 901, "validation": 0, "test": 0})
    with pytest.raises(ValueError):
        game_count_contract({"train": True, "validation": 0, "test": 0})


def test_old_training_configuration_cannot_fill_missing_versions_with_defaults():
    config = _config(900, 990)
    for field in ("training_plan_version", "training_config_version", "model_tier_basis", "rule_feature_mask_version"):
        record = config.to_record()
        record.pop(field)
        with pytest.raises(ValueError, match="version differs"):
            ProductionTrainConfig.from_record(record)
    with pytest.raises(ValueError, match="train-split ladder"):
        replace(config, model_tier="medium", effective_batch_size=0)
    with pytest.raises(ValueError):
        replace(config, N_eligible_half_games=True)
