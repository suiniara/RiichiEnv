import pytest

from riichienv.kar.authorization import (
    ParameterUpdatesNotAuthorizedError,
    UpdateAuthorization,
    require_parameter_updates,
)
from riichienv.kar.readiness import CORE_REQUIREMENTS, feature_readiness, validate_readiness


def test_data_and_core_readiness_never_grant_update_permission():
    report = feature_readiness(
        {key: True for key in CORE_REQUIREMENTS},
        evidence={key: "functional-test-artifact" for key in CORE_REQUIREMENTS},
        source_hash="fixture-source",
    )
    validate_readiness(report)
    assert report["core_feature_ready"] and not report["full_feature_ready"]
    assert not report["parameter_updates_authorized"]
    with pytest.raises(ParameterUpdatesNotAuthorizedError):
        require_parameter_updates(None, operation="train_cpu")
    with pytest.raises(ParameterUpdatesNotAuthorizedError):
        require_parameter_updates(UpdateAuthorization({"training_allowed": True}), operation="train_cpu")
