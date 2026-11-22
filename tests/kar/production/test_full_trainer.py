# ruff: noqa: E402 -- optional Torch must be checked before ML imports
import pytest

torch = pytest.importorskip("torch")
from test_training_contracts import SyntheticDataset, _authorization, components, config

from riichienv.kar.authorization import UpdateAuthorization
from riichienv.kar.budget import BudgetPlan, CostForecast
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning.distributed import DistributedContext, DistributedStageError
from riichienv.kar.learning.export import learning_contract
from riichienv.kar.learning.inference import model_weights_sha256
from riichienv.kar.learning.model import ModelOutputs
from riichienv.kar.learning.sampling import GlobalEpochSampler
from riichienv.kar.learning.trainer import _dataset_contract, production_update, train_production
from riichienv.kar.learning.training_config import ProductionTrainConfig, training_recipe_sha256
from riichienv.kar.release import complete_release_contract
from riichienv.kar.versioning import game_count_contract


class FullSyntheticDataset(SyntheticDataset):
    def __init__(self, directory, split, *, include_candidate_branches=False, **_kwargs):
        super().__init__()
        self.split = split
        self.has_candidate_branches = bool(include_candidate_branches)
        self.indices = [(f"synthetic-row-{index}", 9) for index in range(3)]
        available = {"train": 3, "validation": 3, "test": 0}
        self.manifest.update(game_count_contract(available))
        self.manifest.update(
            status="PASS",
            complete=True,
            database_sha256="d" * 64,
            is_probe_subset=True,
            counts={role: {"games": count, "examples": count} for role, count in available.items()},
            scope={
                **learning_contract(),
                "split_salt": "synthetic-production-contract-test",
                "available_split_counts": available,
            },
        )
        self.manifest.pop("manifest_sha256")
        self.manifest["manifest_sha256"] = content_hash(self.manifest)


def _full_config(reference):
    values = {
        "phase": "M1",
        "model_tier": "small",
        "N_train_half_games": 3,
        "N_eligible_half_games": 6,
        "planned_epochs": 1,
        "device": "cpu",
        "precision": "fp32",
        "cpu_threads": 1,
        "microbatch_size": 1,
        "checkpoint_updates": 1,
    }
    recipe = training_recipe_sha256(values)
    forecast = CostForecast("M1", 9, 100.0, 0.0, 0.0, reference, recipe)
    return ProductionTrainConfig(**values, budget=BudgetPlan("M1", 100.0, 0.0, forecast))


def test_full_synthetic_fixture_satisfies_current_production_contract(tmp_path):
    """Catch manifest drift even when every parameter-update test is skipped."""
    train = FullSyntheticDataset(tmp_path, "train")
    validation = FullSyntheticDataset(tmp_path, "validation")
    settings = _full_config("synthetic metadata regression, not a measured forecast")
    assert train.manifest == validation.manifest
    manifest = train.manifest
    sizes = {"train": len(train), "validation": len(validation), "test": 0}
    assert manifest["available_split_counts"] == manifest["exported_split_counts"] == sizes
    assert manifest["scope"]["available_split_counts"] == sizes
    assert {role: values["games"] for role, values in manifest["counts"].items()} == sizes
    assert all(manifest["scope"].get(key) == value for key, value in learning_contract().items())
    _dataset_contract(train, validation, settings)
    release = complete_release_contract(
        training=settings.to_record(),
        data_manifest=manifest,
        budget=settings.budget.to_record(),
    )
    assert release["N_eligible_half_games"] == 6 and release["N_train_half_games"] == 3


@pytest.mark.parameter_updates
def test_full_epoch_runs_all_validation_and_has_no_diagnostic_update_cap(tmp_path, monkeypatch):
    from riichienv.kar.learning import trainer  # noqa: PLC0415

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(trainer, "DecisionDataset", FullSyntheticDataset)
    settings = _full_config("synthetic full-epoch fixture, not a measured production forecast")
    recipe = settings.recipe_sha256
    doc = dict(_authorization().document)
    doc["allowed_operations"] = ["train_production", "production_update"]
    doc["allowed_recipe_sha256"] = [recipe]
    doc["dataset_manifest_sha256"] = FullSyntheticDataset(tmp_path / "data", "train").manifest["manifest_sha256"]
    doc.pop("authorization_sha256")
    doc["authorization_sha256"] = content_hash(doc)
    result = train_production(tmp_path / "data", tmp_path / "run", settings, authorization=UpdateAuthorization(doc))
    assert result["status"] == "PASS" and result["completed_epochs"] == 1 and result["processed_examples"] == 3
    assert result["final_validation"]["report"]["evaluated_examples"] == 3
    assert result["final_validation"]["report"]["full_validation_split_evaluated"]
    assert result["initial_weights_sha256"] != result["final_weights_sha256"]
    assert (tmp_path / "run" / "checkpoint.json").exists()


@pytest.mark.parameter_updates
def test_nonfinite_forward_never_advances_optimizer_scheduler_or_weights(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    data = SyntheticDataset()
    raw, optimizer, scheduler = components()
    context = DistributedContext.open("cpu")
    before = model_weights_sha256(raw)
    step = scheduler.last_epoch
    original = raw.forward

    def nonfinite(batch):
        outputs = original(batch)
        return ModelOutputs(
            outputs.policy_logits * torch.nan, outputs.response_logits, outputs.rank_logits, outputs.action_present
        )

    monkeypatch.setattr(raw, "forward", nonfinite)
    group = GlobalEpochSampler([65] * 3, seed=1, effective_batch=128, microbatch=1).peek()
    try:
        with pytest.raises(DistributedStageError):
            production_update(raw, raw, optimizer, scheduler, data, group, context, config(), _authorization())
        assert model_weights_sha256(raw) == before and scheduler.last_epoch == step and not optimizer.state
    finally:
        context.close()


@pytest.mark.parameter_updates
def test_completed_checkpoint_recovers_publication_without_another_optimizer_update(tmp_path, monkeypatch):
    from riichienv.kar.learning import trainer  # noqa: PLC0415
    from riichienv.kar.learning.run_publication import publish_run_manifest  # noqa: PLC0415

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(trainer, "DecisionDataset", FullSyntheticDataset)
    settings = _full_config("synthetic publication interruption fixture")
    recipe = settings.recipe_sha256
    doc = dict(_authorization().document)
    doc.pop("authorization_sha256")
    doc["allowed_operations"] = ["train_production", "production_update"]
    doc["allowed_recipe_sha256"] = [recipe]
    doc["dataset_manifest_sha256"] = FullSyntheticDataset(tmp_path / "data", "train").manifest["manifest_sha256"]
    auth = UpdateAuthorization({**doc, "authorization_sha256": content_hash(doc)})

    def fail_publish(*args, **kwargs):
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(trainer, "publish_run_manifest", fail_publish)
    output = tmp_path / "run"
    with pytest.raises(DistributedStageError):
        train_production(tmp_path / "data", output, settings, authorization=auth)
    assert (output / "checkpoint.json").exists() and not (output / "run_manifest.json").exists()

    def forbidden_update(*args, **kwargs):
        raise AssertionError("completed recovery must not repeat training")

    monkeypatch.setattr(trainer, "production_update", forbidden_update)
    monkeypatch.setattr(trainer, "publish_run_manifest", publish_run_manifest)
    recovered = train_production(tmp_path / "data", output, settings, authorization=auth, resume=True)
    assert (
        recovered["recovered_completed_checkpoint"] and recovered["processed_examples"] == 3 and recovered["step"] == 1
    )
