from types import SimpleNamespace

import pytest

from riichienv.kar.budget import BudgetPlan, CostForecast


def test_registered_m3_cannot_restart_in_a_fresh_output_or_lower_its_price(tmp_path):
    pytest.importorskip("torch")
    from riichienv.kar.learning.simulation import register_m3_run  # noqa: PLC0415

    plan = {"paired_seed_manifest_sha256": "a" * 64, "subject_release_sha256": "b" * 64, "partition": "test"}
    budget = BudgetPlan(
        "M3", 250.0, 0.0, CostForecast("M3", 20000, 100.0, 1.0, 0.0, "synthetic registration-only test", "a" * 64)
    )
    config = SimpleNamespace(
        output=tmp_path / "A", test_registry=tmp_path / "test.json", budget=budget, paired_seed_manifest=plan
    )
    assert register_m3_run(config) is False
    assert register_m3_run(config) is True
    other = SimpleNamespace(**{**config.__dict__, "output": tmp_path / "B"})
    with pytest.raises(ValueError, match="reset"):
        register_m3_run(other)
    free = BudgetPlan(
        "M3", 250.0, 0.0, CostForecast("M3", 20000, 100.0, 0.0, 0.0, "synthetic registration-only test", "a" * 64)
    )
    changed = SimpleNamespace(**{**config.__dict__, "budget": free})
    with pytest.raises(ValueError, match="pricing"):
        register_m3_run(changed)
