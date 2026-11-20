# riichienv-ml

PPO training, Global Reward Predictor (GRP) training, and shared model inference
for RiichiEnv. The current KAR learning implementation is a separate module at
[`src/riichienv/kar/learning`](../src/riichienv/kar/learning/README.md); see the
[KAR production guide](../docs/kar-production-code.md) for its training workflow.

Legacy BC/CQL trainers, their command-line entry points and configurations, the
teacher adapter, and the action-level Monte Carlo replay dataset have been
removed. Top-level `bc` / `cql` configuration sections, `algorithm: dqn`, and
retired DQN/CQL settings under `ppo` now raise configuration errors.

## Setup and training

Run these commands from the repository root:

```sh
uv sync --locked --package riichienv-ml

# Train the reward predictor, then a PPO policy.
uv run --locked --package riichienv-ml python riichienv-ml/scripts/train_grp.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/grp.yml
uv run --locked --package riichienv-ml python riichienv-ml/scripts/train_ppo.py \
  -c riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml
```

Replace the example data, checkpoint, evaluator, and experiment-tracking
settings before running. The PPO examples use a trained GRP checkpoint for
reward shaping; set `ppo.grp_model` to the matching GRP output. Three-player
examples are available under `configs/3p/`. The four-player `ppo_v2.yml` example
uses the discard-history/shanten encoder and asynchronous PPO rollout.

PPO examples start with `load_model: null` and `alpha_kl: 0.0`, so they no longer
require a checkpoint produced by a removed trainer. To initialize from an
existing compatible checkpoint, set `ppo.load_model` explicitly. KL
regularization is optional and should only be enabled with an intended
reference policy.

## Inference

`riichienv_ml.agents.Agent` accepts a PPO configuration and loads its checkpoint
from `ppo.load_model` or an explicit `model_path` argument:

```python
from riichienv_ml.agents import Agent

agent = Agent(
    "riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml",
    model_path="checkpoints/model.pth",
    device="cpu",
)
```

Shared model architectures, encoders, evaluator integration, and compatible
checkpoint weight conversion remain available. Public native replay APIs also
remain available; deleting the old training dataset does not change their
action-reconstruction semantics. GRP uses round-level features and final ranks,
not the deleted dataset's action labels.

## Regression tests

The GRP dataset sorts a private copy of input paths, assigns disjoint worker
slices, then shuffles within each worker. Different worker RNG states therefore
preserve full, exactly-once file coverage. Validation and iteration do not
mutate the original file list.

From the repository root:

```sh
uv run --locked --all-extras --dev --group kar-ml pytest tests/ml
```

These tests cover GRP sharding, real spawned DataLoader workers, rank labels,
rejection of retired configuration fields, and loading the retained PPO/GRP
examples. They require no training corpus, parameter updates, evaluator service,
or experiment tracker.
