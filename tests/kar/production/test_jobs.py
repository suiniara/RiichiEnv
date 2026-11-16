import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from riichienv.kar.contracts import content_hash
from riichienv.kar.evaluation_authorization import require_evaluation_authorization
from riichienv.kar.experiments import M3_MATCHES, M3_PAIRS, paired_seed_manifest, validate_paired_manifest


def test_m3_manifest_freezes_all_twenty_thousand_matches_and_balanced_seats():
    plan = paired_seed_manifest(
        master_seed=17, subject_release_sha256="a" * 64, opponent_release_ids=("b" * 64, "c" * 64, "d" * 64)
    )
    assert plan["pair_count"] == M3_PAIRS == 10000 and plan["match_count"] == M3_MATCHES == 20000
    assert plan["seat_counts"] == [2500] * 4
    assert all(row["baseline_subject_rng_seed"] != row["candidate_subject_rng_seed"] for row in plan["blocks"])
    validate_paired_manifest(json.loads(json.dumps(plan)))
    plan["blocks"].pop()
    plan["paired_seed_manifest_sha256"] = content_hash(
        {key: value for key, value in plan.items() if key != "paired_seed_manifest_sha256"}
    )
    with pytest.raises(ValueError):
        validate_paired_manifest(plan)


def test_no_evaluation_approval_is_not_a_model_or_match_launch():
    with pytest.raises(PermissionError):
        require_evaluation_authorization(None, operation="run_m3", plan_sha256="a" * 64)


def test_new_training_and_evaluation_entries_default_deny_before_compute(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from riichienv.kar.authorization import ParameterUpdatesNotAuthorizedError  # noqa: PLC0415
    from riichienv.kar.learning import orchestration, simulation, trainer  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        raise AssertionError("Model/data execution reached without authorization")

    monkeypatch.setattr(trainer, "DecisionDataset", forbidden)
    monkeypatch.setattr(simulation, "FrozenRelease", forbidden)
    for call in (
        lambda: trainer.train_production(Path("missing"), tmp_path / "run", None),
        lambda: trainer.production_update(None, None, None, None, None, None, None, None, None),
        lambda: orchestration.run_m1_ablation(Path("missing"), tmp_path / "m1", None, None),
        lambda: orchestration.run_m2(
            Path("missing"), tmp_path / "m2", None, {}, main_model_registry=tmp_path / "registry"
        ),
    ):
        with pytest.raises(ParameterUpdatesNotAuthorizedError):
            call()
    with pytest.raises(PermissionError):
        simulation.run_m3(SimpleNamespace(paired_seed_manifest={"paired_seed_manifest_sha256": "a" * 64}))
    with pytest.raises(PermissionError):
        simulation.benchmark_matches(
            Path("missing"),
            (Path("missing"),) * 3,
            matches=4,
            seed=1,
            output=tmp_path / "bench",
            budget=SimpleNamespace(phase="M1"),
        )
    assert not list(tmp_path.iterdir())
