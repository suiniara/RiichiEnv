"""Bounded CPU regressions for the legacy BC/CQL evaluation transition."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def legacy_trainer(monkeypatch):
    """Load the real trainer without optional experiment-tracking dependencies."""
    package = Path(__file__).resolve().parents[2] / "riichienv-ml" / "src" / "riichienv_ml"
    monkeypatch.syspath_prepend(str(package.parent))
    wandb = ModuleType("wandb")
    wandb.run = None
    wandb.log = lambda *args, **kwargs: None
    wandb.finish = lambda: None
    wandb.sdk = SimpleNamespace(wandb_run=SimpleNamespace(Run=object))
    loguru = ModuleType("loguru")
    loguru.logger = SimpleNamespace(info=lambda *args: None, error=lambda *args: None)
    config = ModuleType("riichienv_ml.config")
    factories = {}
    config.import_class = factories.__getitem__
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setitem(sys.modules, "loguru", loguru)
    monkeypatch.setitem(sys.modules, "riichienv_ml.config", config)
    utils = _load_source(package / "utils.py", "_bc_evaluation_test_utils")
    monkeypatch.setitem(sys.modules, "riichienv_ml.utils", utils)
    module = _load_source(package / "trainers" / "bc_logs.py", "_bc_evaluation_test_trainer")
    return module, factories


@pytest.mark.parameter_updates
@pytest.mark.parametrize("failure", [None, "periodic", "final"])
def test_evaluation_always_restores_training_mode(tmp_path, monkeypatch, legacy_trainer, failure):
    module, factories = legacy_trainer
    models = []
    forward_modes = []
    evaluation_modes = []

    class TinyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2)
            self.head = torch.nn.Linear(2, 2)
            models.append(self)

        def forward(self, features):
            forward_modes.append(self.training)
            return self.head(self.bn(features))

    class Evaluator:
        def evaluate(self, weights, *, num_episodes):
            evaluation_modes.append(models[0].training)
            call = len(evaluation_modes)
            if (failure == "periodic" and call == 1) or (failure == "final" and call == 3):
                raise RuntimeError("synthetic evaluator failure")
            return {"synthetic_metric": 0.0}

        def metrics_to_logline(self, metrics):
            return "synthetic evaluation completed"

    factories.update(
        model=TinyNetwork,
        encoder=lambda **kwargs: object(),
        dataset=lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(module, "_create_evaluator", lambda **kwargs: Evaluator())
    monkeypatch.setattr(module, "RewardPredictor", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.glob, "glob", lambda *args, **kwargs: ["synthetic-replay"])
    batch = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([0, 1]),
        torch.tensor([1.0, -1.0]),
        torch.ones(2, 2),
    )
    monkeypatch.setattr(module, "DataLoader", lambda *args, **kwargs: [batch, batch, batch])
    evaluator_config = SimpleNamespace(
        model_path=None,
        evaluator_name=None,
        eval_device="cpu",
        opponents=[],
        eval_interval=1,
        eval_episodes=1,
    )
    trainer = module.Trainer(
        grp_model_path="unused",
        pts_weight=[1.0, 0.0, -1.0, -2.0],
        data_glob="synthetic-replay",
        device_str="cpu",
        batch_size=2,
        limit=3,
        num_epochs=1,
        num_workers=1,
        model_class="model",
        dataset_class="dataset",
        encoder_class="encoder",
        evaluator_config=evaluator_config,
    )
    trainer.train(str(tmp_path / "model.pth"))

    # Both periodic evaluations and the final evaluation run in eval mode.
    assert evaluation_modes == [False, False, False]
    # A failed periodic evaluation must not freeze BatchNorm in the next update.
    assert forward_modes == [True, True, True]
    assert int(models[0].bn.num_batches_tracked) == 3
    assert models[0].training is True


@pytest.mark.parametrize("previous_training", [False, True])
@pytest.mark.parametrize("fail_evaluation", [False, True])
def test_online_bc_evaluation_restores_previous_mode(
    tmp_path, monkeypatch, legacy_trainer, previous_training, fail_evaluation
):
    offline, _ = legacy_trainer
    package = Path(offline.__file__).parents[1]
    monkeypatch.setitem(sys.modules, "ray", ModuleType("ray"))
    teacher = ModuleType("riichienv_ml.teacher")

    def forbidden_teacher(*args, **kwargs):
        pytest.fail("The evaluation-mode regression must not launch teacher workers")

    teacher.load_teacher_worker_class = forbidden_teacher
    monkeypatch.setitem(sys.modules, "riichienv_ml.teacher", teacher)
    monkeypatch.setitem(sys.modules, "riichienv_ml.trainers.bc_logs", offline)
    module = _load_source(package / "trainers" / "bc_model.py", "_bc_model_evaluation_test_trainer")
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
    model.train(previous_training)
    seen = []

    class Evaluator:
        def evaluate(self, weights, *, num_episodes):
            seen.append(model.training)
            if fail_evaluation:
                raise RuntimeError("synthetic evaluator failure")
            return {"synthetic_metric": 0.0}

        def metrics_to_logline(self, metrics):
            return "synthetic evaluation completed"

    trainer = object.__new__(module.BCModelTrainer)
    trainer.tp_evaluator = Evaluator()
    trainer.evaluator_config = SimpleNamespace(eval_episodes=1)
    trainer._run_eval(model, str(tmp_path / "model.pth"), 1)

    assert seen == [False]
    assert all(layer.training is previous_training for layer in model.modules())
