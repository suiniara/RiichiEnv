from dataclasses import replace

import pytest

from riichienv.kar.calibration import calibration_metrics
from riichienv.kar.policy import PolicyOptions
from riichienv.kar.runtime import RuntimePolicy


def test_certified_argmax_combination_is_rejected_before_scoring():
    with pytest.raises(ValueError, match="Argmax"):
        PolicyOptions("certified", 1.0, diagnostic_argmax=True)


@pytest.mark.parametrize("mode", ["empirical_only", "certified"])
def test_quality_failure_falls_back_even_with_nonempty_check_counts(
    mode, fixed_predictor, event_support, calibration_fixture, public_view
):
    artifact, _, _ = calibration_fixture
    check = {key: (dict(value) if hasattr(value, "items") else value) for key, value in artifact.check_metrics.items()}
    check["response"] = {**check["response"], "nll": 500.0, "brier": 1.99, "ece": 0.99}
    failed = replace(artifact, check_metrics=check)
    assert failed.quality_report["status"] == "failed" and not failed.empirical_check_valid
    policy = RuntimePolicy(
        fixed_predictor,
        options=PolicyOptions(mode, 0.5),
        support=event_support,
        calibration=failed,
        allow_diagnostic_calibration=True,
    )
    result = policy.decide(public_view[0], public_view[1], "bad-quality")
    assert result.fallback_reason == "calibration_quality_failed" and fixed_predictor.value_calls == 0
    assert result.evidence_status != "certified"


def test_independent_calibration_report_has_reliability_branches_clusters_and_mass(calibration_fixture, event_support):
    _, fit, check = calibration_fixture
    rows = [replace(fit[0], current_rank=3), replace(check[0], current_rank=1)]
    report = calibration_metrics(rows, (1.0, 1.0, 1.0), -20.0, support=event_support, bootstrap_replicates=20)
    details = report["diagnostics"]
    assert "per_class_reliability" in details["heads"]["response"]
    assert len(details["heads"]["response"]["class_nll"]) == 21
    assert details["heads"]["response"]["class_brier"][0] is not None
    assert len(details["rank_by_response"]) == 21 and details["rank_by_response"]["0"]["count"] == 2
    assert "game_clustered_intervals" in details and "utility_calibration" in details
    assert sum(details["unsupported_mass_histogram"]["counts"]) == 2
    assert report["gate_by_current_rank"]["3"]["count"] == 1
    assert "support_masses" not in report
