"""Precision migration is rejected independently of the public-mask migration."""

# ruff: noqa: E402 -- Torch is optional outside the explicit ML test job.
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from riichienv.kar.calibration import CalibrationArtifact, CalibrationRow, calibrate_static_rows, calibration_split
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning import config as inference_config
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.deployment import FrozenRelease, save_frozen_release
from riichienv.kar.learning.inference import FrozenModelPredictor
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.runtime import RuntimePolicy
from riichienv.kar.support import EventSupport
from riichienv.kar.versioning import artifact_contract


@pytest.fixture
def predictor():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(67)
        model = KARTransformer(ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="precision-migration"))
    return FrozenModelPredictor(model, "b" * 64)


def _previous_recipe(predictor):
    """The exact prior single-prefix recipe, with every other field current."""
    return content_hash(
        {
            "model_config_hash": predictor.model.config.contract_hash,
            "feature_version": inference_config.FEATURE_VERSION,
            "rule_feature_mask_version": inference_config.RULE_FEATURE_MASK_VERSION,
            "use_rule_features": True,
            "inference_feature_dropout": 0,
            "inference_precision": "fp32_no_autocast_single_prefix_v2",
        }
    )


def _calibration(predictor, support):
    split = calibration_split([f"precision-migration-{index}" for index in range(20)])

    def row(game):
        return CalibrationRow(
            game,
            game + "/decision",
            (4, 8),
            (0.0,) * 546,
            4,
            response_logits=(0.0,) * 21,
            response_allowed=(True,) * 21,
            rank_logits=((0.0,) * 4,) * 21,
            response_target=0,
            response_valid=True,
            rank_target=0,
            value_valid=True,
        )

    return calibrate_static_rows(
        [row(split.fit_games[0])],
        [row(split.check_games[0])],
        split=split,
        binding=predictor.binding,
        support=support,
        temperatures=(1.0,),
        source_reference="synthetic static logits; precision-contract rejection only",
        diagnostic_fixture=True,
    )


def test_previous_single_prefix_calibration_is_rejected_with_current_mask_contract(predictor):
    support = EventSupport(predictor.binding.train_manifest_sha256, (500,) * 21, "synthetic migration evidence")
    current = _calibration(predictor, support)
    restored = CalibrationArtifact.from_record(current.to_record(), predictor.binding, support, allow_diagnostic=True)
    assert restored.to_record() == current.to_record()
    RuntimePolicy(predictor, support=support, calibration=restored, allow_diagnostic_calibration=True)

    previous_recipe = _previous_recipe(predictor)
    assert previous_recipe != predictor.binding.inference_recipe_sha256
    previous = replace(current, binding=replace(predictor.binding, inference_recipe_sha256=previous_recipe))
    previous_record = previous.to_record()
    # Both artifacts have the current public-rule contract: a mask-version
    # rejection cannot make this precision-identity regression pass accidentally.
    assert previous_record["contract"] == current.to_record()["contract"]
    previous.validate(previous.binding, support, allow_diagnostic=True)
    with pytest.raises(ValueError, match="provenance mismatch"):
        CalibrationArtifact.from_record(previous_record, predictor.binding, support, allow_diagnostic=True)
    with pytest.raises(ValueError, match="provenance mismatch"):
        RuntimePolicy(predictor, support=support, calibration=previous, allow_diagnostic_calibration=True)


def test_previous_single_prefix_release_is_rejected_with_current_source_and_weights(tmp_path, predictor):
    directory = tmp_path / "synthetic-release"
    current = save_frozen_release(
        directory,
        predictor.model,
        train_manifest_sha256=predictor.binding.train_manifest_sha256,
        use_rule_features=True,
        training_contract={**artifact_contract(), "synthetic_fixture": True},
    )
    assert FrozenRelease(directory).record == current

    previous = json.loads(json.dumps(current))
    previous["binding"]["inference_recipe_sha256"] = _previous_recipe(predictor)
    previous["release_sha256"] = content_hash(
        {key: value for key, value in previous.items() if key != "release_sha256"}
    )
    (directory / "release.json").write_text(json.dumps(previous), encoding="utf-8")
    with pytest.raises(ValueError, match="Loaded weights/configuration/features"):
        FrozenRelease(directory)
