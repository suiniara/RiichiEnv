import math

import pytest

from riichienv.kar.statistics import FullDiagnostics, LatencyDistribution


def test_support_mass_and_complete_candidate_size_distributions_survive_merge():
    first, second = FullDiagnostics(), FullDiagnostics()
    first.add_gate(retained=False, support_size=0, mass=0.0, slices=("current_rank/1",))
    second.add_gate(retained=True, support_size=32, mass=1.0, slices=("current_rank/1",))
    first.merge(second.state_dict())
    report = first.report(bootstrap_replicates=20)
    gate = report["gate"]
    sliced = report["gate_by_slice"]["current_rank/1"]
    assert gate["support_mass_histogram"][0] == gate["support_mass_histogram"][100] == 1
    assert gate["support_size_histogram"][0] == gate["support_size_histogram"][32] == 1
    assert sum(sliced["support_mass_histogram"]) == 2 and sliced["mean_support_mass"] == 0.5


def test_per_event_intervals_keep_independent_games_not_decision_rows():
    metrics = FullDiagnostics()
    for game, count in (("one-game", 100), ("second-game", 1)):
        for _ in range(count):
            metrics.add(
                game=game,
                head="rank",
                probabilities=[0.7, 0.1, 0.1, 0.1],
                target=0,
                log_probability=math.log(0.7),
                event=8,
            )
    report = metrics.report(bootstrap_replicates=20)
    branch = report["rank_branch_intervals"]["8"]
    assert branch["nll"]["independent_games"] == 2 and branch["utility_error"]["independent_games"] == 2
    assert report["rank_by_response"]["8"]["count"] == 101
    assert report["rank_branch_intervals"]["0"]["nll"]["status"] == "insufficient_independent_games"
    assert report["game_clustered_brier_intervals"]["rank"]["independent_games"] == 2
    assert report["branch_calibration_claimed"] is False


def test_latency_histogram_merge_reports_ranges_not_a_fake_exact_p95():
    first, second = LatencyDistribution(), LatencyDistribution()
    first.add(0.001)
    first.add(0.002)
    second.add(0.04)
    second.add(0.1)
    first.merge(second.state_dict())
    report = first.report()
    assert report["count"] == 4 and report["mean_seconds"] == pytest.approx(0.143 / 4)
    bounds = report["p95_seconds_bounds"]
    assert bounds["lower_seconds"] <= 0.1 <= bounds["upper_seconds"]
    assert "not_exact_point_quantiles" in report["quantile_method"]


def test_feature_off_diagnostics_still_slice_by_original_public_current_rank():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from test_training_contracts import SyntheticDataset  # noqa: PLC0415

    from riichienv.kar.learning.batch import rule_feature_view  # noqa: PLC0415
    from riichienv.kar.learning.dataset import TrainingBatch, collate_examples  # noqa: PLC0415
    from riichienv.kar.learning.evaluation import add_diagnostics  # noqa: PLC0415
    from riichienv.kar.learning.model import ModelOutputs  # noqa: PLC0415

    data = SyntheticDataset()
    batch = collate_examples([data[0]])
    inputs = rule_feature_view(batch.inputs, torch.zeros(1, dtype=torch.bool), enabled=False)
    outputs = ModelOutputs(
        torch.zeros((1, 546)), torch.zeros((1, 21)), torch.zeros((1, 21, 4)), torch.ones(1, dtype=torch.bool)
    )
    metrics = FullDiagnostics()
    add_diagnostics(
        metrics, outputs, TrainingBatch(inputs, batch.labels, batch.game_ids), public_fields=batch.inputs.fields
    )
    report = metrics.report(bootstrap_replicates=20)
    assert report["by_slice"]["current_rank/1"]["action"]["count"] == 1


def test_run_throughput_uses_weighted_totals_not_mean_of_step_rates():
    from riichienv.kar.learning.run_publication import performance_summary  # noqa: PLC0415

    result = performance_summary(
        {
            "updates": 2,
            "examples": 30,
            "real_tokens": 300,
            "padded_real_tokens": 500,
            "training_seconds": 3.0,
            "data_wait_seconds_sum": 0.2,
            "validation_seconds": 1.0,
            "validation_passes": 1,
        }
    )
    assert result["decision_samples_per_second"] == 10.0
    assert result["tokens_per_second"] == 100.0
    assert result["padding_fraction_real_microbatches"] == 0.4
    assert performance_summary(None) is None
