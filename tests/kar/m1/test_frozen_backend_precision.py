"""The actual frozen predictor must reject changed FP32 arithmetic backends."""

import json
import subprocess
import sys

import pytest
import torch

from riichienv.kar.actions import legal_candidates
from riichienv.kar.calibration import CalibrationRow, gate_log_threshold
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.learning.inference import FrozenModelPredictor, model_weights_sha256
from riichienv.kar.learning.precision import FrozenBackendPrecision, capture_frozen_backend_precision
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.policy import support_discards
from riichienv.kar.probability import legal_prior


@pytest.fixture
def ieee_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "ieee")


@pytest.fixture
def decision():
    env = MJSOUL_RULESET.create_env(seed=0)
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    for event in env.mjai_log:
        ledger.apply_event(event)
    candidates = tuple(
        action.global_id for action in legal_candidates(env.get_observations([0])[0], drawn_tile=env.drawn_tile)
    )
    return ledger.snapshot(), candidates


@pytest.mark.parametrize("mode", ["bf16", "tf32"])
def test_frozen_predictor_refuses_non_ieee_cpu_without_changing_host(ieee_cpu, miniature, monkeypatch, mode):
    monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", mode)
    with pytest.raises(ValueError, match="IEEE FP32"):
        FrozenModelPredictor(miniature, "b" * 64)
    assert torch.backends.mkldnn.matmul.fp32_precision == mode


def test_backend_change_cannot_silently_change_a_fitted_gate(ieee_cpu, miniature, decision, monkeypatch):
    predictor = FrozenModelPredictor(miniature, "b" * 64)
    snapshot, candidates = decision
    weights = model_weights_sha256(miniature)
    logits = tuple(predictor.policy(snapshot, candidates, "gate-boundary"))
    row = CalibrationRow("fixture", "gate-boundary", candidates, logits, action_target=56)
    threshold, count = gate_log_threshold([row], 1.0)
    assert count == 1 and threshold is not None
    assert 56 in support_discards(candidates, legal_prior(logits, candidates, 1.0)[1], threshold)

    monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "bf16")
    with pytest.raises(ValueError, match="IEEE FP32"):
        predictor.policy(snapshot, candidates, "gate-boundary")
    assert torch.backends.mkldnn.matmul.fp32_precision == "bf16"
    assert model_weights_sha256(miniature) == weights
    assert all(parameter.grad is None for parameter in miniature.parameters())


def test_backend_change_between_prefix_and_suffix_is_rejected(ieee_cpu, miniature, decision, monkeypatch):
    predictor = FrozenModelPredictor(miniature, "b" * 64)
    snapshot, candidates = decision
    predictor.policy(snapshot, candidates, "prefix")
    monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "bf16")
    with pytest.raises(ValueError, match="IEEE FP32"):
        predictor.candidates(snapshot, candidates, candidates[:2], "prefix")
    assert torch.backends.mkldnn.matmul.fp32_precision == "bf16"


def test_backend_change_during_forward_never_returns_logits(ieee_cpu, miniature, decision, monkeypatch):
    predictor = FrozenModelPredictor(miniature, "b" * 64)
    snapshot, candidates = decision
    original = miniature.encode_prefix

    def change_after_computing(batch):
        result = original(batch)
        monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "bf16")
        return result

    monkeypatch.setattr(miniature, "encode_prefix", change_after_computing)
    with pytest.raises(ValueError, match="IEEE FP32"):
        predictor.policy(snapshot, candidates, "changed-during-forward")
    assert predictor._cache is None


def test_valid_backend_change_is_bound_to_a_different_recipe(ieee_cpu, miniature, decision, monkeypatch):
    predictor = FrozenModelPredictor(miniature, "b" * 64)
    previous = predictor.backend_precision
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", not previous.mkldnn_enabled)
    replacement = FrozenModelPredictor(miniature, "b" * 64)
    assert predictor.binding != replacement.binding
    with pytest.raises(ValueError, match="backend settings changed"):
        predictor.policy(*decision, "another-mkldnn-backend")


def test_cpu_precision_getter_inherits_global_non_ieee_mode(ieee_cpu, monkeypatch):
    monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "none")
    monkeypatch.setattr(torch.backends, "fp32_precision", "bf16")
    with pytest.raises(ValueError, match="IEEE FP32"):
        capture_frozen_backend_precision("cpu")


def test_untouched_none_and_explicit_ieee_have_one_canonical_contract(ieee_cpu, monkeypatch):
    explicit = capture_frozen_backend_precision("cpu")
    monkeypatch.setattr(torch.backends.mkldnn.matmul, "fp32_precision", "none")
    assert capture_frozen_backend_precision("cpu") == explicit
    assert FrozenBackendPrecision.from_record(explicit.to_record()) == explicit


def test_cuda_tf32_configuration_guard_needs_no_cuda_execution(monkeypatch):
    initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32")
    with pytest.raises(ValueError, match="IEEE FP32.*cuda.matmul"):
        capture_frozen_backend_precision("cuda")
    assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    assert torch.cuda.is_initialized() == initialized


def test_backend_metadata_and_recipe_can_be_read_without_importing_torch(ieee_cpu):
    record = json.dumps(capture_frozen_backend_precision("cpu").to_record())
    program = """
import importlib.abc
import json
import sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Metadata parsing imported Torch')

sys.meta_path.insert(0, NoTorch())
from riichienv.kar.learning.config import ModelConfig, frozen_inference_recipe_sha256
from riichienv.kar.learning.precision import FrozenBackendPrecision
backend = FrozenBackendPrecision.from_record(json.loads(sys.argv[1]))
digest = frozen_inference_recipe_sha256(ModelConfig(), use_rule_features=True, backend_precision=backend)
assert len(digest) == 64 and 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program, record], check=True, capture_output=True, text=True)
