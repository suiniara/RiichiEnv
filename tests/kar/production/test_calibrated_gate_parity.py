"""Calibrated gate boundaries must agree across validation and deployment."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from riichienv.kar.calibration import CalibrationRow, calibrate_static_rows, calibration_split
from riichienv.kar.event_semantics import PUBLIC_CLOCK_VERSION
from riichienv.kar.learning.batch import ModelBatch
from riichienv.kar.learning.dataset import TrainingBatch
from riichienv.kar.learning.evaluation import add_diagnostics
from riichienv.kar.learning.losses import Labels
from riichienv.kar.learning.model import ModelOutputs
from riichienv.kar.policy import PolicyOptions
from riichienv.kar.runtime import RuntimePolicy
from riichienv.kar.runtime_contracts import RuntimeBinding
from riichienv.kar.statistics import FullDiagnostics
from riichienv.kar.support import EventSupport

# ruff: noqa: E402 -- optional torch dependency is checked before importing evaluation


class _FixedLogits:
    def __init__(self, binding, logits):
        self.binding, self.logits = binding, logits

    def policy(self, snapshot, candidates, context):
        return self.logits

    def candidates(self, snapshot, candidates, supported_discards, context):
        raise AssertionError("Gate telemetry needs no model forward or value scoring")


def _gate_reports(ids, legal_logits, *, temperature, fit_target, check_target):
    # Large finite logits on illegal actions must not influence the prior.
    policy_logits = torch.full((1, 546), 10000.0, dtype=torch.float32)
    policy_logits[0, list(ids)] = torch.tensor(legal_logits, dtype=torch.float32)
    logits = tuple(policy_logits[0].tolist())
    split = calibration_split([f"synthetic-gate-parity-{index}" for index in range(40)])
    binding = RuntimeBinding.current("a" * 64, "b" * 64)
    support = EventSupport(binding.train_manifest_sha256, (500,) * 21, "synthetic gate-only evidence")

    def row(game, target):
        return CalibrationRow(
            game,
            game + "/decision",
            ids,
            logits,
            target,
            response_logits=(0.0,) * 21,
            rank_logits=((0.0,) * 4,) * 21,
            response_allowed=(True,) * 21,
            response_target=0,
            response_valid=True,
            rank_target=0,
            value_valid=True,
        )

    artifact = calibrate_static_rows(
        [row(split.fit_games[0], fit_target)],
        [row(split.check_games[0], check_target)],
        split=split,
        binding=binding,
        support=support,
        temperatures=(temperature,),
        source_reference="synthetic logits, no forward/backward or parameter updates",
        diagnostic_fixture=True,
        inference_device_type="fixture",
    )
    candidates = torch.full((1, 32), 546, dtype=torch.int64)
    candidates[0, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
    zero = torch.tensor([0])
    valid = torch.tensor([True])
    inputs = ModelBatch(
        {
            "kind": torch.tensor([[2]]),
            "actor": torch.tensor([[1]]),
            "flags": torch.zeros(1, 1, 16, dtype=torch.int64),
            "stage": torch.tensor([[1]]),
        },
        candidates,
        zero,
        zero,
        ("synthetic-gate-row",),
    )
    labels = Labels(
        torch.tensor([check_target]),
        valid,
        zero,
        valid,
        valid,
        zero,
        valid,
        torch.ones(1, 21, dtype=torch.bool),
    )
    outputs = ModelOutputs(policy_logits, torch.zeros(1, 21), torch.zeros(1, 21, 4), valid)
    metrics = FullDiagnostics()
    add_diagnostics(
        metrics,
        outputs,
        TrainingBatch(inputs, labels, (split.check_games[0],)),
        calibration=artifact,
        support=support,
    )
    view = SimpleNamespace(
        history_complete=True,
        visibility_complete=True,
        public_clock_version=PUBLIC_CLOCK_VERSION,
    )
    runtime = RuntimePolicy(
        _FixedLogits(binding, logits),
        options=PolicyOptions("empirical_only", math.inf),
        support=support,
        calibration=artifact,
        allow_diagnostic_calibration=True,
    ).decide(view, ids, "synthetic-gate-decision")
    return artifact, metrics.report(bootstrap_replicates=5)["gate"], runtime.telemetry


def test_equal_logits_at_fitted_threshold_keep_both_actions_in_all_paths():
    artifact, gate, runtime = _gate_reports(
        (0, 4),
        (0.0, 0.0),
        temperature=1.0,
        fit_target=0,
        check_target=0,
    )
    assert artifact.log_threshold == -math.log(2.0)
    assert runtime["supported_discards"] == [0, 4]
    assert gate == artifact.to_record()["check_metrics"]["diagnostics"]["gate"]
    assert gate["recorded_action_exclusion_fraction"] == 0.0
    assert gate["mean_support_mass"] == runtime["support_mass"] == 1.0
    assert gate["support_size_histogram"][2] == 1


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("check_target", [4, 8], ids=["on-threshold", "below-threshold"])
def test_nonuniform_gate_keeps_full_legal_prior_and_matches_runtime(temperature, check_target):
    # The legal ankan (148) retains its prior mass but cannot enter discard S.
    artifact, gate, runtime = _gate_reports(
        (0, 4, 8, 148),
        (1.25, 0.25, -0.5, 3.75),
        temperature=temperature,
        fit_target=4,
        check_target=check_target,
    )
    assert runtime["supported_discards"] == [0, 4]
    assert gate == artifact.to_record()["check_metrics"]["diagnostics"]["gate"]
    assert gate["recorded_action_exclusion_fraction"] == float(check_target == 8)
    assert gate["mean_support_mass"] == runtime["support_mass"]
    assert 0.0 < gate["mean_support_mass"] < 1.0
    assert gate["support_size_histogram"][2] == 1
