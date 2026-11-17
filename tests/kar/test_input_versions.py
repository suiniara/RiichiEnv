"""Input-computation migration is explicit despite unchanged tensor/schema shapes."""
import pytest

from riichienv.kar.budget import BudgetPlan, CostForecast
from riichienv.kar.learning.config import MODEL_VERSION
from riichienv.kar.learning.training_config import ProductionTrainConfig, training_recipe_sha256
from riichienv.kar.versioning import INPUT_PIPELINE_VERSION, artifact_contract, validate_artifact_contract


def test_old_artifact_cannot_omit_or_retag_new_input_semantics():
    current = artifact_contract()
    assert current["schema_id"] == "kar-riichienv-v1"
    assert current["response_mask_version"] == "public_rule_response_mask_v2_mjsoul_nagashi"
    assert current["artifact_contract_version"] == "kar_riichienv_contract_v3"
    assert current["input_pipeline_version"] == INPUT_PIPELINE_VERSION == "legal_set_pool_private_self_pass_v1"
    assert MODEL_VERSION == "kar_transformer_v2_legal_set"
    old = dict(current, artifact_contract_version="kar_riichienv_contract_v2")
    old.pop("input_pipeline_version")
    with pytest.raises(ValueError, match="version contract differs"):
        validate_artifact_contract(old)
    with pytest.raises(ValueError, match="input_pipeline_version"):
        validate_artifact_contract({**current, "input_pipeline_version": "legacy-public-only"})


def test_old_forecast_recipe_and_configuration_cannot_silently_cross_input_change():
    arguments = {"phase": "M2", "model_tier": "small", "N_train_half_games": 900,
                 "N_eligible_half_games": 990, "planned_epochs": 2, "device": "cpu", "precision": "fp32"}
    recipe = training_recipe_sha256(arguments)
    forecast = CostForecast("M2", 100, 100.0, 0.0, 0.0, "synthetic metadata only,not a compute job", recipe)
    config = ProductionTrainConfig(**arguments, budget=BudgetPlan("M2", 650.0, 0.0, forecast))
    record = config.to_record()
    assert record["input_pipeline_version"] == INPUT_PIPELINE_VERSION
    assert record["training_config_version"] == "explicit_train_split_scale_v3_inputs"
    assert config.recipe_sha256 == recipe
    record.pop("input_pipeline_version")
    with pytest.raises(ValueError, match="version differs"):
        ProductionTrainConfig.from_record(record)
