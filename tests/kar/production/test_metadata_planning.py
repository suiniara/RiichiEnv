import json

import pytest
from test_paired_benchmark import benchmark_fixture

from riichienv.kar import planning
from riichienv.kar.budget import BudgetExceededError


def test_metadata_only_planners_create_no_model_or_execution_directory(tmp_path, monkeypatch):
    subject, opponents, benchmark = benchmark_fixture()
    paths = []
    for index, record in enumerate((subject, *opponents)):
        directory = tmp_path / f"release-{index}"
        directory.mkdir()
        (directory / "release.json").write_text(json.dumps(record))
        paths.append(directory)
    report = tmp_path / "benchmark.json"
    report.write_text(json.dumps(benchmark))
    monkeypatch.setattr(planning, "update_implementation_sha256", lambda: "a" * 64)
    source = {
        "subject_release": str(paths[0]),
        "opponent_releases": [str(path) for path in paths[1:]],
        "seed": 17,
        "matches": 8,
        "device": "cpu",
    }
    first = planning.plan_benchmark(source)
    assert first["planning_only"] and not first["model_computation_performed"] and not first["authorization_granted"]
    config = {
        **source,
        "benchmark_report": str(report),
        "master_seed": 29,
        "resource_usd_per_hour": 0.0,
        "fixed_cost_usd": 0.0,
        "budget_cap_usd": 250.0,
        "budget_spent_usd": 0.0,
        "other_phase_spent_usd": 0.0,
        "run_output": str(tmp_path / "not-started"),
        "test_registry": str(tmp_path / "test-registry.json"),
    }
    result = planning.plan_m3(config)
    assert result["paired_seed_manifest"]["execution_ready"] and result["budget"]["forecast"]["work_units"] == 20000
    assert result["budget"]["forecast"]["measurement_reference"] == "benchmark:" + benchmark["benchmark_sha256"]
    assert not (tmp_path / "not-started").exists() and not list(tmp_path.rglob("weights.pt"))
    with pytest.raises(BudgetExceededError):
        planning.plan_m3({**config, "resource_usd_per_hour": 3600.0})
