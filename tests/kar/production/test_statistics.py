import math

import pytest

from riichienv.kar.statistics import ClassificationMetrics, FullDiagnostics, GameClusters, ReliabilityBins


def test_repeated_decisions_do_not_become_independent_games():
    clusters = GameClusters()
    for _ in range(1000):
        clusters.add("one-half-game", "utility", 1.0)
    report = clusters.interval("utility", replicates=50)
    assert report["independent_games"] == 1 and report["lower"] is None


def test_cluster_resampling_keeps_whole_games_together():
    first, second = GameClusters(), GameClusters()
    for game, value, count in (("a", 1.0, 3), ("b", -1.0, 9), ("c", 0.5, 2)):
        for _ in range(count):
            first.add(game, "value", value)
        for _ in range(count * 2):
            second.add(game, "value", value)
    left = first.interval("value", seed=17, replicates=100)
    right = second.interval("value", seed=17, replicates=100)
    assert left == right and left["independent_games"] == 3


def test_reliability_and_observed_branch_rank_metrics_are_explicit():
    bins = ReliabilityBins()
    bins.add(0.9, 1.0)
    bins.add(0.1, 0.0)
    assert bins.report()["ece"] == pytest.approx(0.1)
    metrics = FullDiagnostics()
    metrics.add(
        game="a",
        head="rank",
        probabilities=[0.7, 0.1, 0.1, 0.1],
        target=0,
        log_probability=math.log(0.7),
        event=8,
        slices=("mixed_riichi_and_open_meld_threat",),
    )
    metrics.add(
        game="b", head="rank", probabilities=[0.1, 0.1, 0.1, 0.7], target=3, log_probability=math.log(0.7), event=4
    )
    metrics.add_response_mass((0.7, 0.3, *(0.0,) * 19), train_supported=(True,) * 21,
                              value_supported=(True,) + (False,) * 20)
    metrics.add_gate(retained=False, support_size=1, mass=0.2)
    result = metrics.report(bootstrap_replicates=50)
    assert result["rank_by_response"]["8"]["count"] == 1 and result["rank_by_response"]["4"]["count"] == 1
    assert result["rank_by_response"]["0"]["nll"] is None
    assert result["gate"]["recorded_action_exclusion_fraction"] == 1.0
    assert result["unsupported_mass_histogram"]["counts"][30] == 1
    assert result["train_only_unsupported_mass_histogram"]["counts"][0] == 1
    assert result["evidence_status"] == "empirical_only_not_uniform_per_state_error_bounds"
    assert result["epsilon_m_empirical_proxy"] is not None


def test_mass_state_merge_rejects_legacy_buckets_and_mixed_value_support():
    left, right = FullDiagnostics(), FullDiagnostics()
    probabilities = (0.7, 0.3) + (0.0,) * 19
    left.add_response_mass(probabilities, train_supported=(True,) * 21, value_supported=(True,) + (False,) * 20)
    right.add_response_mass(probabilities, train_supported=(True,) * 21, value_supported=(True,) * 21)
    with pytest.raises(ValueError, match="different support sets"):
        left.merge(right.state_dict())
    legacy = {**left.state_dict(), "statistics_version": "whole_game_cluster_statistics_v2"}
    with pytest.raises(ValueError, match="statistics version differs"):
        FullDiagnostics().merge(legacy)


def test_metric_merge_preserves_natural_frequency_and_zero_class_is_not_perfect():
    first = ClassificationMetrics(3, per_class_reliability=True)
    second = ClassificationMetrics(3, per_class_reliability=True)
    first.add([0.8, 0.1, 0.1], 0)
    second.add([0.1, 0.8, 0.1], 1)
    first.merge(second.state_dict())
    report = first.report()
    assert report["count"] == 2 and report["class_counts"] == [1, 1, 0]
    assert report["class_recall"] == [1.0, 1.0, None]
    assert report["per_class_reliability"][2]["count"] == 2


def test_probability_underflow_uses_the_stable_log_probability_not_clipping():
    metrics = ClassificationMetrics(2)
    nll, _ = metrics.add([0.0, 1.0], 0, log_probability=-1000.0)
    assert nll == 1000.0
    with pytest.raises(ValueError):
        metrics.add([math.nan, 1.0], 0)
