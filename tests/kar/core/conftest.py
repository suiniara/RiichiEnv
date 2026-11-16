import hashlib
import sys

import pytest

from riichienv.kar.actions import legal_candidates
from riichienv.kar.calibration import CalibrationRow, calibrate_static_rows, calibration_split
from riichienv.kar.calibration_quality import CalibrationQualityPolicy
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.runtime_contracts import RuntimeBinding
from riichienv.kar.support import EventSupport


@pytest.fixture(autouse=True)
def forbid_backward_and_updates(monkeypatch):
    torch = sys.modules.get("torch")
    if torch is None:
        yield
        return
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Functional acceptance forbids backward/optimizer updates")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    yield
    assert not torch.cuda.is_initialized(), "Core functionality acceptance must not initialize GPU"


@pytest.fixture
def public_view():
    env = MJSOUL_RULESET.create_env(seed=17)
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    for event in env.mjai_log:
        ledger.apply_event(event)
    ids = tuple(
        action.global_id for action in legal_candidates(env.get_observations([0])[0], drawn_tile=env.drawn_tile)
    )
    return ledger.snapshot(), ids


@pytest.fixture
def binding():
    return RuntimeBinding.current(
        hashlib.sha256(b"fixed-logits-functional-fixture").hexdigest(),
        hashlib.sha256(b"fixed-training-manifest-fixture").hexdigest(),
    )


@pytest.fixture
def event_support(binding):
    return EventSupport(
        binding.train_manifest_sha256, (500,) * 21, "synthetic independent train-game fixture,not measured data"
    )


@pytest.fixture
def calibration_fixture(binding, event_support, public_view):
    _, ids = public_view
    split = calibration_split([f"validation-{index}" for index in range(20)])

    def row(game):
        return CalibrationRow(
            game,
            game + "/row0",
            ids,
            (0.0,) * 546,
            ids[0],
            response_logits=(0.0,) * 21,
            rank_logits=((0.0,) * 4,) * 21,
            response_allowed=(True,) * 21,
            response_target=0,
            response_valid=True,
            rank_target=0,
            value_valid=True,
        )

    fit = [row(split.fit_games[0])]
    check = [row(split.check_games[0])]
    artifact = calibrate_static_rows(
        fit,
        check,
        split=split,
        binding=binding,
        support=event_support,
        temperatures=(1.0,),
        source_reference="synthetic static logits only; no model fitting",
        diagnostic_fixture=True,
        quality_policy=CalibrationQualityPolicy(
            {head: 20.0 for head in ("action", "response", "rank")},
            {head: 2.0 for head in ("action", "response", "rank")},
            {head: 1.0 for head in ("action", "response", "rank")},
            1,
            1,
            0.0,
            "synthetic, explicitly preregistered fixture criteria",
        ),
        inference_device_type="fixture",
    )
    return artifact, fit, check


class FixedPredictor:
    def __init__(self, binding):
        self.binding = binding
        self.policy_calls = 0
        self.value_calls = 0

    def policy(self, snapshot, candidates, context):
        self.policy_calls += 1
        return (0.0,) * 546

    def candidates(self, snapshot, candidates, supported_discards, context):
        self.value_calls += 1
        first = min(supported_discards)
        return {
            action: (
                (0.0,) * 21,
                tuple((8.0, 0.0, 0.0, 0.0) if action == first else (0.0, 0.0, 0.0, 8.0) for _ in range(21)),
            )
            for action in supported_discards
        }


@pytest.fixture
def fixed_predictor(binding):
    return FixedPredictor(binding)
