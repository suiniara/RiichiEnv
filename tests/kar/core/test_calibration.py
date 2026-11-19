import copy
import math
from dataclasses import replace

import pytest

from riichienv.kar.calibration import (
    CalibrationArtifact,
    calibrate_static_rows,
    calibration_split,
    gate_log_threshold,
    select_static_temperature,
)
from riichienv.kar.policy import support_discards
from riichienv.kar.probability import legal_prior


def test_validation_split_is_game_fixed_disjoint_and_rejects_leakage():
    first = calibration_split([f"game-{i}" for i in range(30)])
    assert first == calibration_split(list(reversed(first.validation_games)))
    assert not set(first.fit_games) & set(first.check_games)
    with pytest.raises(ValueError):
        calibration_split(["shared"], train_games=["shared"])
    with pytest.raises(ValueError):
        calibration_split(["shared"], test_games=["shared"])
    with pytest.raises(ValueError):
        calibration_split(["same", "same"])
    with pytest.raises(ValueError):
        replace(first, check_games=first.fit_games).validate()


@pytest.mark.parametrize("count", [1, 3, 50, 99, 100, 101, 200])
def test_log_gate_uses_zero_based_n_div_100_and_retains_ties(count, calibration_fixture):
    _, fit, _ = calibration_fixture
    base = fit[0]
    rows = []
    expected = []
    for index in range(count):
        logits = [0.0] * 546
        logits[base.action_target] = -index * 0.01
        row = replace(base, row_id=str(index), policy_logits=tuple(logits))
        rows.append(row)
        expected.append(legal_prior(logits, base.candidates)[1][base.action_target])
    threshold, n = gate_log_threshold(rows, 1.0)
    assert n == count and threshold == sorted(expected)[count // 100]
    assert sum(value >= threshold for value in expected) >= math.ceil(0.99 * count)
    same = [replace(base, row_id=str(index)) for index in range(count)]
    threshold, n = gate_log_threshold(same, 1.0)
    assert all(
        base.action_target
        in support_discards(row.candidates, legal_prior(row.policy_logits, row.candidates)[1], threshold)
        for row in same
    )


def test_gate_is_not_conditioned_on_discard_subset(calibration_fixture):
    _, fit, _ = calibration_fixture
    base = fit[0]
    row = replace(base, candidates=(4, 8, 219), action_target=4, policy_logits=(0.0,) * 546)
    threshold, n = gate_log_threshold([row], 1.0)
    assert n == 1 and threshold == pytest.approx(-math.log(3))
    assert threshold != pytest.approx(-math.log(2))


def test_gate_empty_single_discard_invalid_action_and_nondiscard(calibration_fixture):
    _, fit, _ = calibration_fixture
    base = fit[0]
    rows = [
        replace(base, action_valid=False, response_valid=False, value_valid=False),
        replace(base, candidates=(base.action_target,), response_valid=False, value_valid=False),
        replace(base, candidates=(4, 8, 219), action_target=219, response_valid=False, value_valid=False),
    ]
    assert gate_log_threshold(rows, 1.0) == (None, 0)
    assert support_discards(base.candidates, legal_prior(base.policy_logits, base.candidates)[1], None) == ()


def test_log_threshold_remains_authoritative_under_probability_underflow(calibration_fixture):
    _, fit, _ = calibration_fixture
    base = fit[0]
    logits = [0.0] * 546
    logits[4] = -1000.0
    logits[8] = -1001.0
    row = replace(base, candidates=(4, 8, 219), action_target=4, policy_logits=tuple(logits))
    threshold, n = gate_log_threshold([row], 1.0)
    prior, logs = legal_prior(logits, row.candidates)
    assert prior[4] == 0 and threshold == -1000 and n == 1
    assert support_discards(row.candidates, logs, threshold) == (4,)


def test_static_temperature_grid_really_selects_distinct_heads_without_network_updates(calibration_fixture):
    _, fit, _ = calibration_fixture
    base = fit[0]
    policy = [0.0] * 546
    policy[base.action_target] = 5.0
    response = [0.0] * 21
    response[1] = 5.0
    row = replace(
        base, policy_logits=tuple(policy), response_logits=tuple(response), rank_logits=((0.0, 5.0, 0.0, 0.0),) * 21
    )
    chosen = tuple(
        select_static_temperature([row], head, (0.5, 1.0, 2.0))[0] for head in ("action", "response", "rank")
    )
    assert chosen == (0.5, 2.0, 2.0)


def test_cal_check_does_not_adjust_fit_temperatures_or_gate(binding, event_support, calibration_fixture):
    artifact, fit, check = calibration_fixture
    modified = [replace(check[0], policy_logits=tuple(float(index % 3) for index in range(546)), rank_target=3)]
    other = calibrate_static_rows(
        fit,
        modified,
        split=artifact.split,
        binding=binding,
        support=event_support,
        temperatures=(1.0,),
        source_reference="synthetic fixture",
        diagnostic_fixture=True,
    )
    assert other.temperatures == artifact.temperatures and other.log_threshold == artifact.log_threshold
    assert other.check_rows_sha256 != artifact.check_rows_sha256
    with pytest.raises(ValueError):
        calibrate_static_rows(
            check, fit, split=artifact.split, binding=binding, support=event_support, source_reference="invalid"
        )


@pytest.mark.parametrize("side", ["fit", "check"])
def test_static_calibration_rejects_foreign_games_and_duplicate_decisions(
    side, binding, event_support, calibration_fixture,
):
    artifact, fit, check = calibration_fixture
    kwargs = {"split": artifact.split, "binding": binding, "support": event_support,
              "source_reference": "synthetic membership check", "diagnostic_fixture": True}
    rows = {"fit": fit, "check": check}
    other = "check" if side == "fit" else "fit"
    rows[side] = [replace(rows[side][0], game_id="outside-the-frozen-validation-split")]
    with pytest.raises(ValueError, match="foreign games are forbidden"):
        calibrate_static_rows(rows["fit"], rows["check"], **kwargs)
    original = fit if side == "fit" else check
    rows[side] = [replace(original[0], row_id=rows[other][0].row_id)]
    with pytest.raises(ValueError, match="Duplicate calibration decision identities"):
        calibrate_static_rows(rows["fit"], rows["check"], **kwargs)


def test_calibration_artifact_binding_integrity_and_fixture_boundary(binding, event_support, calibration_fixture):
    artifact, _, _ = calibration_fixture
    record = artifact.to_record()
    assert CalibrationArtifact.from_record(record, binding, event_support, allow_diagnostic=True).to_record() == record
    with pytest.raises(ValueError):
        CalibrationArtifact.from_record(record, binding, event_support)
    tampered = copy.deepcopy(record)
    tampered["temperatures"][0] = 2.0
    with pytest.raises(ValueError):
        CalibrationArtifact.from_record(tampered, binding, event_support, allow_diagnostic=True)
