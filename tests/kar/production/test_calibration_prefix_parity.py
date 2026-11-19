"""Real miniature CPU forwards: archived strict gates equal deployed policy gates."""

# ruff: noqa: E402 -- Torch is optional outside the explicit ML test job.
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from riichienv.kar.actions import legal_candidates
from riichienv.kar.budget import BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.calibration import CalibrationRow, calibrate_static_rows, calibration_split, gate_log_threshold
from riichienv.kar.calibration_quality import CalibrationQualityPolicy
from riichienv.kar.contracts import MJSOUL_RULESET, game_split
from riichienv.kar.learning import calibration_job, evaluation
from riichienv.kar.learning.batch import collate_inputs
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.distributed import DistributedContext
from riichienv.kar.learning.inference import FrozenModelPredictor, model_weights_sha256
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.policy import support_discards
from riichienv.kar.probability import legal_prior
from riichienv.kar.response_mask import public_rule_response_mask
from riichienv.kar.schema import TOKEN_BYTE_SIZE
from riichienv.kar.support import EventSupport
from riichienv.kar.tokenize import append_action, encode_prefix


@pytest.fixture
def prefix_fixture():
    torch.manual_seed(123)
    model = KARTransformer(ModelConfig(layers=2, width=32, heads=4, feedforward=64, preset="test_fixture")).eval()
    predictor = FrozenModelPredictor(model, "b" * 64)
    rows, views = [], []
    for seed, target in ((42, 128), (0, 100), (17, 128)):
        env = MJSOUL_RULESET.create_env(seed=seed)
        ledger = ObserverLedger(0, (0, 1, 2, 3))
        for event in env.mjai_log:
            ledger.apply_event(event)
        ids = tuple(
            action.global_id for action in legal_candidates(env.get_observations([0])[0], drawn_tile=env.drawn_tile)
        )
        snapshot = ledger.snapshot()
        prefix = encode_prefix(snapshot, ids)
        full = append_action(prefix, target)
        rows.append(
            {
                "tokens": full.to_bytes()[108:],
                "candidates": ids,
                "decide_index": prefix.decide_index,
                "action_index": len(full.rows) - 1,
                "decision_row_id": f"prefix-parity-{seed}",
                "game_id": f"synthetic-prefix-game-{seed}",
                "action_target": target,
                "action_valid": True,
                "response_target": 0,
                "response_applicable": True,
                "response_observed": True,
                "final_rank": 0,
                "value_valid": True,
                "response_allowed": public_rule_response_mask(snapshot, target),
            }
        )
        views.append(snapshot)
    return predictor, rows, views


@pytest.mark.parametrize("microbatch", [1, 3])
def test_archived_policy_has_exact_deployed_logits_and_threshold_support(
    prefix_fixture, microbatch, tmp_path, monkeypatch, record_property
):
    predictor, rows, views = prefix_fixture
    before = model_weights_sha256(predictor.model)
    support = EventSupport("b" * 64, (500,) * 21, "synthetic support; no training evidence")
    monkeypatch.setattr(calibration_job, "dataset_event_support", lambda _data: support)
    monkeypatch.setattr(
        calibration_job,
        "calibration_split_from_manifest",
        lambda _manifest: SimpleNamespace(
            fit_games=tuple(row["game_id"] for row in rows),
        ),
    )
    collected = []

    def capture(fit, check, **kwargs):
        collected.extend(fit)
        assert list(check) == []
        return "synthetic-archive-only"

    monkeypatch.setattr(calibration_job, "calibrate_static_rows", capture)
    ledger = BudgetLedger(
        tmp_path / "costs.jsonl",
        BudgetPlan(
            "M2",
            0.0,
            0.0,
            CostForecast("M2", 3, 1.0, 0.0, 0.0, "unpriced miniature forward-only fixture", "a" * 64),
        ),
    )
    with torch.no_grad(), ledger.record_compute("synthetic-prefix-parity"):
        calibration_job._archive_and_fit(
            predictor,
            rows,
            {},
            tmp_path,
            ledger=ledger,
            quality_policy=None,
            temperatures=(1.0,),
            microbatch=microbatch,
            plan_sha256="a" * 64,
        )
    archived = {row.row_id: row for row in collected}
    deployed = []
    for row, view in zip(rows, views):
        stored = archived[row["decision_row_id"]]
        logits = tuple(predictor.policy(view, row["candidates"], row["decision_row_id"]))
        assert stored.policy_logits == logits
        deployed.append(replace(stored, policy_logits=logits))
    # A strict comparison at the fitted boundary is the regression: small FP32
    # differences can pass allclose while excluding the action used to fit it.
    threshold, count = gate_log_threshold(collected, 1.0)
    assert count == 3 and threshold is not None
    assert gate_log_threshold(deployed, 1.0) == (threshold, count)
    for stored, live in zip((archived[row["decision_row_id"]] for row in rows), deployed):
        expected = support_discards(
            stored.candidates, legal_prior(stored.policy_logits, stored.candidates, 1.0)[1], threshold
        )
        actual = support_discards(live.candidates, legal_prior(live.policy_logits, live.candidates, 1.0)[1], threshold)
        assert actual == expected
    boundary = deployed[1]
    boundary_logs = legal_prior(boundary.policy_logits, boundary.candidates, 1.0)[1]
    assert boundary_logs[100] == threshold
    assert support_discards(boundary.candidates, boundary_logs, threshold) == (92, 100)
    assert model_weights_sha256(predictor.model) == before
    assert all(parameter.grad is None for parameter in predictor.model.parameters())
    record_property("gate_support_on_fitted_boundary", [92, 100])
    record_property("microbatch", microbatch)
    record_property("backward_or_update_executed", False)


def test_policy_archive_trims_teacher_action_padding_and_retains_input_validation(
    prefix_fixture, monkeypatch, record_property
):
    predictor, rows, views = prefix_fixture
    seen = []
    original = predictor.model.encode_prefix

    def inspect(batch):
        seen.append((batch.size, int(batch.fields["kind"].shape[1]), bool(batch.fields["valid"][:, -1].all())))
        assert not (batch.fields["block"] == 3).any()
        return original(batch)

    monkeypatch.setattr(predictor.model, "encode_prefix", inspect)
    padded = [{**row, "tokens": row["tokens"] + bytes(7 * TOKEN_BYTE_SIZE)} for row in rows]
    actual = predictor.policy_logits_from_rows(padded).tolist()
    assert seen == [(1, row["decide_index"] + 1, True) for row in rows]
    for row, view, logits in zip(rows, views, actual):
        assert logits == predictor.policy(view, row["candidates"], row["decision_row_id"])
    with pytest.raises(ValueError, match="canonical complete"):
        predictor.policy_logits_from_rows([{**rows[0], "candidates": [128, 128]}])
    with pytest.raises(ValueError, match="prefix is truncated"):
        predictor.policy_logits_from_rows([{**rows[0], "tokens": b""}])
    # The old path uses different GEMM token/batch shapes. Its observed rounding
    # error is diagnostic only; some hardware/kernels may happen to agree exactly.
    with torch.no_grad():
        dense = predictor.model(collate_inputs(rows)).policy_logits
    record_property("dense_policy_max_abs_difference", float((torch.tensor(actual) - dense).abs().max()))


def test_calibrated_complete_validation_uses_the_same_strict_policy_gate(prefix_fixture, monkeypatch):
    predictor, rows, views = prefix_fixture
    salt = "kar-riichienv-v1-games"
    games = [f"synthetic-prefix-eval-{index}" for index in range(1000)]
    split = replace(
        calibration_split([game for game in games if game_split(game, salt) == "validation"]),
        validation_manifest_sha256="d" * 64,
        main_split_salt=salt,
    )
    live = [tuple(predictor.policy(view, row["candidates"], row["decision_row_id"])) for row, view in zip(rows, views)]
    fit = [
        CalibrationRow(
            split.fit_games[index],
            row["decision_row_id"],
            row["candidates"],
            logits,
            row["action_target"],
            response_logits=(0.0,) * 21,
            rank_logits=((0.0,) * 4,) * 21,
            response_allowed=row["response_allowed"],
            response_target=0,
            response_valid=True,
            rank_target=0,
            value_valid=True,
        )
        for index, (row, logits) in enumerate(zip(rows, live))
    ]
    check = [
        replace(row, game_id=split.check_games[index], row_id=row.row_id + "/check") for index, row in enumerate(fit)
    ]
    support = EventSupport("b" * 64, (500,) * 21, "synthetic support; no training evidence")
    heads = ("action", "response", "rank")
    artifact = calibrate_static_rows(
        fit,
        check,
        split=split,
        binding=predictor.binding,
        support=support,
        temperatures=(1.0,),
        quality_policy=CalibrationQualityPolicy(
            dict.fromkeys(heads, 10.0),
            dict.fromkeys(heads, 2.0),
            dict.fromkeys(heads, 1.0),
            1,
            1,
            0.0,
            "synthetic miniature policy gate test; auxiliary logits are fixed",
        ),
        inference_device_type="cpu",
        source_reference="synthetic fixture, not production calibration",
    )

    class ValidationRows(list):
        split = "validation"
        manifest = {"manifest_sha256": "b" * 64, "rho_E": 1.0, "rho_V": 1.0}

    data = ValidationRows([{**row, "game_id": split.fit_games[index]} for index, row in enumerate(rows)])
    original = evaluation.add_diagnostics
    observed = []

    def inspect(metrics, outputs, batch, **kwargs):
        for ids, logits in zip((row["candidates"] for row in data), outputs.policy_logits.tolist()):
            observed.append(support_discards(ids, legal_prior(logits, ids, 1.0)[1], artifact.log_threshold))
        assert outputs.policy_logits.tolist() == [list(logits) for logits in live]
        original(metrics, outputs, batch, **kwargs)

    monkeypatch.setattr(evaluation, "add_diagnostics", inspect)
    predictor.model.train()
    context = DistributedContext.open("cpu")
    try:
        result = evaluation.evaluate_complete(
            predictor.model,
            data,
            context,
            microbatch=3,
            use_rule_features=True,
            calibration=artifact,
            support=support,
            bootstrap_replicates=5,
        )
    finally:
        context.close()
    expected = [
        support_discards(row.candidates, legal_prior(row.policy_logits, row.candidates, 1.0)[1], artifact.log_threshold)
        for row in fit
    ]
    assert observed == expected and observed[1] == (92, 100)
    assert result["diagnostics"]["gate"]["recorded_action_exclusion_fraction"] == 0.0
    assert predictor.model.training
