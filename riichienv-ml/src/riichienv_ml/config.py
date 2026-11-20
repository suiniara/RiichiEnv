from __future__ import annotations

import importlib
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, computed_field


GAME_PARAMS = {
    4: {
        "tile_dim": 34,
        "num_actions": 82,
        "game_mode": "4p-red-half",
        "replay_rule": "mjsoul",
        "starting_scores": [25000, 25000, 25000, 25000],
    },
    3: {
        "tile_dim": 27,
        "num_actions": 60,
        "game_mode": "3p-red-half",
        "replay_rule": "mjsoul",
        "starting_scores": [35000, 35000, 35000],
    },
}


def import_class(dotted_path: str):
    """Dynamically import a class from a dotted path.

    e.g. "riichienv_ml.models.q_network.QNetwork" -> QNetwork class
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class GameConfig(BaseModel):
    n_players: Literal[3, 4] = 4

    @computed_field
    @property
    def tile_dim(self) -> int:
        return GAME_PARAMS[self.n_players]["tile_dim"]

    @computed_field
    @property
    def num_actions(self) -> int:
        return GAME_PARAMS[self.n_players]["num_actions"]

    @computed_field
    @property
    def game_mode(self) -> str:
        return GAME_PARAMS[self.n_players]["game_mode"]

    @computed_field
    @property
    def replay_rule(self) -> str:
        return GAME_PARAMS[self.n_players]["replay_rule"]

    @computed_field
    @property
    def starting_scores(self) -> list[int]:
        return GAME_PARAMS[self.n_players]["starting_scores"]

    @computed_field
    @property
    def grp_input_dim(self) -> int:
        return self.n_players * 4 + 4


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    in_channels: int = 74
    num_blocks: int = 8
    conv_channels: int = 128
    fc_dim: int = 256
    num_actions: int = 82
    tile_dim: int = 34
    aux_dims: int | None = None


class WandbConfig(BaseModel):
    wandb_entity: str = "smly"
    wandb_project: str = "riichienv"
    wandb_tags: list[str] = []
    wandb_group: str | None = None


class OpponentConfig(BaseModel):
    config_path: str
    model_path: str


class EvaluatorConfig(BaseModel):
    model_path: str | None = None
    evaluator_name: str | None = None
    eval_episodes: int = 48
    eval_interval: int = 50000
    eval_device: str = "cpu"
    opponents: list[OpponentConfig] = []


class GrpConfig(WandbConfig):
    game: GameConfig = GameConfig()
    data_glob: str = "/data/mjsoul/mjsoul-4p/2024/**/*.jsonl.gz"
    val_data_glob: str = "/data/mjsoul/mjsoul-4p/2024/01/**/*.jsonl.gz"
    output: str = "grp_model.pth"
    device: str = "cuda"
    batch_size: int = 128
    num_workers: int = 12
    num_epochs: int = 10
    lr: float = 5e-4
    lr_eta_min: float = 1e-7
    samples_per_file: int = 32


class PpoConfig(WandbConfig):
    model_config = ConfigDict(extra="forbid")

    game: GameConfig = GameConfig()
    algorithm: Literal["ppo"] = "ppo"
    load_model: str | None = None
    device: str = "cuda"
    num_workers: int = 12
    num_steps: int = 5000000
    batch_size: int = 128
    lr: float = 1e-4
    lr_min: float = 1e-6
    max_grad_norm: float = 1.0
    # Optional KL regularization against the initial policy.
    alpha_kl: float = 0.0
    alpha_kl_warmup_steps: int = 0
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    eval_interval: int = 2000
    eval_episodes: int = 100
    weight_sync_freq: int = 10
    worker_device: Literal["cpu", "cuda"] = "cpu"
    gpu_per_worker: float = 0.1
    num_envs_per_worker: int = 16
    gamma: float = 0.99
    weight_decay: float = 0.0
    aux_weight: float = 0.0
    checkpoint_dir: str = "checkpoints"
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.actor_critic.ActorCriticNetwork"
    encoder_class: str = "riichienv_ml.features.feat_v1.ObservationEncoder"
    # Data collection
    collect_hero_only: bool = False
    # GRP reward shaping (per-kyoku reward)
    grp_model: str | None = None
    pts_weight: list[float] = [10.0, 4.0, -4.0, -10.0]
    async_rollout: bool = False
    # Self-play baseline update interval (0 = never update, keep frozen)
    baseline_update_interval: int = 0
    # Fixed opponent model path (if set, baseline is loaded from this path instead of hero weights)
    baseline_model: str | None = None
    # Third-party evaluator
    evaluator: EvaluatorConfig = EvaluatorConfig()


class HandPredConfig(WandbConfig):
    """Config for hand prediction (opponent hand estimation)."""
    game: GameConfig = GameConfig()
    data_glob: str = "/data/mjsoul/mjsoul-4p/2024/**/*.jsonl.gz"
    val_data_glob: str = "/data/mjsoul/mjsoul-4p/2024/01/**/*.jsonl.gz"
    output: str = "hand_pred_model.pth"
    device: str = "cuda"
    batch_size: int = 128
    num_workers: int = 12
    num_epochs: int = 10
    lr: float = 5e-4
    lr_eta_min: float = 1e-7
    weight_decay: float = 0.01
    max_grad_norm: float = 10.0
    samples_per_file: int = 128
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.hand_pred.HandPredCNN"
    encoder_class: str = "riichienv_ml.features.feat_v1.ObservationEncoder"
    dataset_class: str = "riichienv_ml.datasets.hand_pred.HandPredDataset"
    loss_type: str = "cross_entropy"  # "cross_entropy", "smooth_l1", or "mse"
    label_smoothing: float = 0.1
    sum_constraint_weight: float = 0.1  # auxiliary loss weight (regression only)
    val_step_interval: int = 20000  # run validation & save checkpoint every N steps


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grp: GrpConfig = GrpConfig()
    ppo: PpoConfig = PpoConfig()
    hand_pred: HandPredConfig = HandPredConfig()


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
