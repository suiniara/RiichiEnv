import pytest

pytest.importorskip("torch", reason="Install the optional kar-ml dependency group for M1 tests")

import torch

from riichienv.kar.actions import legal_candidates
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.learning.batch import collate_inputs
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.tokenize import append_action, encode_prefix


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def make_sample():
    def make(seed=42, action_choice=0, *, prefix_only=False):
        env = MJSOUL_RULESET.create_env(seed=seed)
        ledger = ObserverLedger(0, (0, 1, 2, 3))
        for event in env.mjai_log:
            ledger.apply_event(event)
        observation = env.get_observations([0])[0]
        candidates = [action.global_id for action in legal_candidates(observation, drawn_tile=env.drawn_tile)]
        prefix = encode_prefix(ledger.snapshot(), candidates)
        full = prefix if prefix_only else append_action(prefix, candidates[action_choice % len(candidates)])
        return {
            "tokens": full.to_bytes()[108:],
            "candidates": candidates,
            "decide_index": prefix.decide_index,
            "action_index": -1 if prefix_only else len(full.rows) - 1,
            "decision_row_id": f"sample-{seed}",
            "prefix": prefix,
            "sequence": full,
            "action_target": candidates[action_choice % len(candidates)],
            "game_id": f"fixture-{seed}",
        }

    return make


@pytest.fixture
def miniature():
    torch.manual_seed(123)
    return KARTransformer(ModelConfig(layers=2, width=32, heads=4, feedforward=64, preset="test_fixture"))


@pytest.fixture
def model_batch(make_sample):
    return collate_inputs([make_sample(42), make_sample(0)])
