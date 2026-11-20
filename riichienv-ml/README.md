# riichienv-ml

Mahjong RL training pipeline for RiichiEnv.

## Setup

```sh
uv sync
uv run python scripts/train_grp.py -c src/riichienv_ml/configs/4p/grp.yml

# CQL+PPO
uv run python scripts/train_cql.py -c src/riichienv_ml/configs/4p/cql.yml
uv run python scripts/train_ppo.py -c src/riichienv_ml/configs/4p/ppo.yml

# BC+PPO (requires online teacher, not included in repo)
uv run python scripts/train_bc.py -c src/riichienv_ml/configs/4p/bc_model.yml
uv run python scripts/train_ppo.py -c src/riichienv_ml/configs/4p/bc_ppo.yml
```

## Data and evaluation regressions

File-backed datasets sort a private copy of the input paths, assign disjoint
worker slices, then shuffle within each worker. Different worker RNG states
therefore preserve full, exactly-once file coverage. Validation and iteration
do not mutate a caller's file list.

Both BC trainers restore the model's previous training mode in `finally` after
evaluation, including evaluator failures. This keeps a failed periodic
evaluation from freezing BatchNorm statistics in later training updates.

From the repository root, run the regressions with the shared ML dependencies:

```sh
uv run --locked --all-extras --dev --group kar-ml pytest tests/ml
uv run --locked --all-extras --dev --group kar-ml pytest \
  --allow-parameter-updates -m parameter_updates tests/ml/test_bc_evaluation_mode.py
```

The second command runs only bounded synthetic CPU updates to exercise the real
training/evaluation transition. It does not require an external evaluator,
experiment tracker, or a training corpus.
