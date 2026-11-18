"""Synthetic logits and preregistration only; no trained models or real calibration."""
import copy
import json
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from riichienv.kar.actions import legal_candidates
from riichienv.kar.calibration import CalibrationArtifact, CalibrationRow, calibrate_static_rows, calibration_split
from riichienv.kar.calibration_quality import BranchQualityCriterion, BranchQualityPolicy, CalibrationQualityPolicy
from riichienv.kar.calibration_roles import calibration_is_fit
from riichienv.kar.contracts import MJSOUL_RULESET, Response, content_hash
from riichienv.kar.learning.dataset import collate_examples
from riichienv.kar.learning.evaluation import add_diagnostics
from riichienv.kar.learning.model import ModelOutputs
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.policy import PolicyOptions
from riichienv.kar.response_mask import public_rule_response_mask
from riichienv.kar.runtime import RuntimePolicy
from riichienv.kar.runtime_contracts import RuntimeBinding
from riichienv.kar.statistics import FullDiagnostics
from riichienv.kar.support import EventSupport
from riichienv.kar.tokenize import append_action, encode_prefix

# ruff: noqa: E402 -- optional torch dependency is checked before importing evaluation

COMMON, RARE = int(Response.NEXT_DRAW), int(Response.RON_MASK_1)
MASKED = int(Response.ABORT_TRIPLE_RON)
EFFECTIVE = "unsupported_mass_histogram"
TRAIN_ONLY = "train_only_unsupported_mass_histogram"


def _counts(report):
    return {index: count for index, count in enumerate(report["counts"]) if count}


@pytest.fixture(scope="module")
def case():
    # Fixture thresholds are explicit and fixed before building/inspecting any rows.
    policy = CalibrationQualityPolicy(
        {"action": 20.0, "response": 0.1, "rank": 0.2},
        {"action": 2.0, "response": 0.1, "rank": 0.1},
        {"action": 1.0, "response": 0.1, "rank": 0.1},
        1, 1, 0.0, "synthetic overall criteria",
        BranchQualityPolicy(tuple(
            BranchQualityCriterion(head, event, 0.5, 2.0, 1.0, 10)
            for head in ("response", "rank") for event in (COMMON, RARE)
        ), "synthetic branch criteria, never production constants"),
    )
    env = MJSOUL_RULESET.create_env(seed=42)
    observer = ObserverLedger(0, (0, 1, 2, 3))
    for event in env.mjai_log:
        observer.apply_event(event)
    view = observer.snapshot()
    ids = tuple(item.global_id for item in legal_candidates(env.get_observation(0), drawn_tile=env.drawn_tile))
    target = next(action for action in ids if action < 148)
    allowed = public_rule_response_mask(view, target)
    assert allowed[COMMON] and allowed[RARE] and not allowed[MASKED]
    prefix = encode_prefix(view, ids)
    sequence = append_action(prefix, target)
    basis = {"tokens": sequence.to_bytes()[108:], "decide_index": prefix.decide_index,
             "action_index": len(sequence.rows) - 1, "candidates": ids}
    names = (f"synthetic-rho-1000-{index}" for index in range(6000))
    fit_game = next(game for game in names if calibration_is_fit(game))
    check_games = [game for game in names if not calibration_is_fit(game)][:1000]
    split = calibration_split([fit_game, *check_games])
    assert len(split.check_games) == len(set(split.check_games)) == 1000

    def row(game, event):
        response = [0.0] * 21
        response[event], response[MASKED] = 16.0, 1000.0
        ranks = [(16.0, 0.0, 0.0, 0.0)] * 21
        ranks[RARE] = (0.0, 16.0, 0.0, 0.0)
        return CalibrationRow(game, game + "/decision", ids, (0.0,) * 546, target,
                              response_logits=tuple(response), rank_logits=tuple(ranks), response_allowed=allowed,
                              response_target=event, response_valid=True, rank_target=0, value_valid=True)

    fit = [row(fit_game, COMMON)]
    check = [row(game, COMMON if index < 990 else RARE) for index, game in enumerate(check_games)]
    binding = RuntimeBinding.current("a" * 64, "b" * 64)
    support = EventSupport(binding.train_manifest_sha256, (500,) * 21, "synthetic, no measured support")
    artifact = calibrate_static_rows(
        fit, check, split=split, binding=binding, support=support, temperatures=(2.0,),
        source_reference="1000 independent synthetic cal_check games", diagnostic_fixture=True, quality_policy=policy,
    )
    return artifact, support, view, basis, check


class _StaticCandidatePredictor:
    def __init__(self, binding, row):
        self.binding, self.row = binding, row

    def policy(self, *_args):
        return self.row.policy_logits

    def candidates(self, _snapshot, _candidates, selected, _context):
        return {action: (self.row.response_logits, self.row.rank_logits) for action in selected}


def _runtime_buckets(artifact, support, view, rows, record_property):
    histogram = {}
    for event in (COMMON, RARE):
        group = [row for row in rows if row.response_target == event]
        first = group[0]
        assert all((row.response_logits, row.response_allowed, row.rank_logits) ==
                   (first.response_logits, first.response_allowed, first.rank_logits) for row in group)
        runtime = RuntimePolicy(
            _StaticCandidatePredictor(artifact.binding, first), options=PolicyOptions("empirical_only", 0.5),
            calibration=artifact, support=support, allow_diagnostic_calibration=True,
        )
        result = runtime.decide(view, first.candidates, "synthetic-mass-parity")
        details = result.telemetry["candidates"][str(first.action_target)]
        assert tuple(details["rule_allowed"]) == first.response_allowed
        assert details["response_probabilities"][MASKED] == 0
        assert tuple(details["value_supported"]) == support.value_supported(artifact.value_branch_statuses)
        assert result.telemetry["temperatures"] == list(artifact.temperatures)
        mass = details["unsupported_mass"]
        record_property(f"runtime_mass_event_{event}", mass)
        index = min(100, int(mass * 100))
        histogram[index] = histogram.get(index, 0) + len(group)
    return histogram


def test_thousand_independent_games_report_runtime_effective_support_not_train_only(case, record_property):
    artifact, support, view, _, rows = case
    rank = artifact.check_metrics["rank"]
    rare = artifact.check_metrics["branch_metrics"]["rank"][str(RARE)]
    assert rank["nll"] == pytest.approx(0.081005881814928)
    assert rank["brier"] == pytest.approx(0.0199745377)
    assert rank["ece"] == pytest.approx(0.0089946239)
    assert rare["nll"] == pytest.approx(8.001005881814928) and rare["independent_games"] == 10
    assert artifact.quality_report["status"] == "passed"
    assert artifact.value_branch_statuses[COMMON] == "passed" and artifact.value_branch_statuses[RARE] == "failed"
    assert artifact.value_branch_statuses[1] == "unavailable" and all(support.supported)
    runtime = _runtime_buckets(artifact, support, view, rows, record_property)
    diagnostics = artifact.check_metrics["diagnostics"]
    effective = _counts(diagnostics[EFFECTIVE])
    for key in ("nll", "brier", "ece"):
        record_property("overall_rank_" + key, rank[key])
    record_property("rare_rank_nll", rare["nll"])
    record_property("effective_report_histogram", json.dumps(effective))
    record_property("runtime_histogram", json.dumps(runtime))
    assert effective == runtime == {0: 990, 99: 10}
    train_only = _counts(diagnostics[TRAIN_ONLY])
    record_property("train_only_histogram", json.dumps(train_only))
    assert train_only == {0: 1000}
    assert diagnostics[EFFECTIVE]["support_basis"] != diagnostics[TRAIN_ONLY]["support_basis"]
    assert _counts(artifact.fit_metrics["diagnostics"][EFFECTIVE]) == {0: 1}


def _evaluation_batch(rows, basis):
    samples = [{**basis, "decision_row_id": row.row_id, "game_id": row.game_id,
                "action_target": row.action_target, "action_valid": row.action_valid,
                "response_target": row.response_target, "response_applicable": True,
                "response_observed": row.response_valid, "final_rank": row.rank_target, "value_valid": row.value_valid,
                "response_allowed": row.response_allowed} for row in rows]
    outputs = ModelOutputs(*(torch.tensor([getattr(row, head) for row in rows], dtype=torch.float32) for head in (
        "policy_logits", "response_logits", "rank_logits"
    )), action_present=torch.ones(len(rows), dtype=torch.bool))
    return outputs, collate_examples(samples)


def test_frozen_roundtrip_evaluation_and_runtime_share_histograms(case, record_property):
    artifact, support, view, basis, rows = case
    saved = json.loads(json.dumps(artifact.to_record()))
    loaded = CalibrationArtifact.from_record(saved, artifact.binding, support, allow_diagnostic=True)
    assert loaded.to_record() == saved and loaded.digest == artifact.digest
    measured, merged = FullDiagnostics(), FullDiagnostics()
    # Feed the same decision IDs, game IDs, labels, masks and logits through the
    # actual evaluation helper, not a stand-in histogram calculation.
    for start in range(0, len(rows), 128):
        add_diagnostics(measured, *_evaluation_batch(rows[start:start + 128], basis),
                        calibration=loaded, support=support)
    # No response even with valid action, so the dummy/all-false mask contributes nothing.
    invalid = replace(rows[-1], row_id="synthetic-no-observed-response", response_valid=False, value_valid=False,
                      response_target=-1, rank_target=-1, response_allowed=(False,) * 21)
    add_diagnostics(measured, *_evaluation_batch([invalid], basis), calibration=loaded, support=support)
    merged.merge(measured.state_dict())  # the same wire as distributed evaluation
    details = merged.report(bootstrap_replicates=4)
    runtime = _runtime_buckets(loaded, support, view, rows, record_property)
    effective = _counts(details[EFFECTIVE])
    record_property("evaluation_histogram", json.dumps(effective))
    assert effective == _counts(saved["check_metrics"]["diagnostics"][EFFECTIVE]) == runtime == {0: 990, 99: 10}
    assert details[EFFECTIVE] == saved["check_metrics"]["diagnostics"][EFFECTIVE]
    assert details[TRAIN_ONLY] == saved["check_metrics"]["diagnostics"][TRAIN_ONLY]
    assert details["heads"]["action"]["count"] == 1001 and details["heads"]["response"]["count"] == 1000


def test_evaluation_without_frozen_quality_reports_train_only(case):
    _, support, _, basis, rows = case
    measured = FullDiagnostics()
    add_diagnostics(measured, *_evaluation_batch([rows[-1]], basis), support=support)
    details = measured.report(bootstrap_replicates=4)
    assert details[EFFECTIVE]["status"] == "unavailable" and not _counts(details[EFFECTIVE])
    assert _counts(details[TRAIN_ONLY]) == {0: 1}


def test_evaluation_counts_an_observed_response_even_when_rank_is_unavailable(case):
    artifact, support, _, basis, rows = case
    row = replace(rows[-1], rank_target=-1, value_valid=False)
    measured = FullDiagnostics()
    add_diagnostics(measured, *_evaluation_batch([row], basis), calibration=artifact, support=support)
    details = measured.report(bootstrap_replicates=4)
    assert details["heads"]["rank"]["count"] == 0
    assert _counts(details[EFFECTIVE]) == {99: 1}


def test_event_ece_requalifies_the_calibrated_chi_branch_in_all_three_consumers(case, record_property):
    original, support, view, basis, rows = case
    chi = int(Response.CHI_P1)
    # Preregister the discriminator before evaluating the 900/100 mixture.
    # These are synthetic limits, never defaults for a real fitted model.
    branches = BranchQualityPolicy(tuple(
        BranchQualityCriterion(head, event, 5.0 if head == "response" else 1.0,
                               2.0 if head == "response" else 0.2, 0.2, 10)
        for head in ("response", "rank") for event in (COMMON, chi, RARE)
    ), "synthetic 0.2 ECE on response event calibration and conditional rank")
    overall = CalibrationQualityPolicy(
        {"action": 20.0, "response": 1.0, "rank": 1.0}, {head: 2.0 for head in ("action", "response", "rank")},
        {head: 1.0 for head in ("action", "response", "rank")}, 1, 1, 0.0, "synthetic overall", branches,
    )
    response = [(-1000.0 if event != MASKED else 1000.0) for event in range(21)]
    response[COMMON], response[chi] = math.log(0.9), math.log(0.1)
    common_logits = {"response_logits": tuple(response), "rank_logits": ((12.0, 0.0, 0.0, 0.0),) * 21}
    check = [replace(row, response_target=COMMON if index < 900 else chi, **common_logits)
             for index, row in enumerate(rows)]
    assert all(row.response_allowed[chi] and row.response_allowed[RARE] for row in check)
    fit = [replace(check[0], game_id=original.split.fit_games[0], row_id="synthetic-fit-uniform-p")]
    artifact = calibrate_static_rows(fit, check, split=original.split, binding=original.binding, support=support,
                                     temperatures=(1.0,), source_reference="immutable synthetic identical predictions",
                                     diagnostic_fixture=True, quality_policy=overall)
    measured = artifact.check_metrics
    original_style = measured["response_classification_by_true_event"]
    old_next, old_chi = (original_style[str(event)]["top_label_ece"] for event in (COMMON, chi))
    assert (old_next, old_chi) == pytest.approx((0.1, 0.9))
    limit = 0.2
    assert old_next < limit < old_chi
    assert measured["response"]["ece"] == pytest.approx(0.0, abs=1e-12)
    for event, positives in ((COMMON, 900), (chi, 100)):
        metric = measured["branch_metrics"]["response"][str(event)]
        counts = (metric["count"], metric["positive_count"], metric["positive_independent_games"])
        assert counts == (1000, positives, positives)
        assert metric["ece"] == pytest.approx(0.0, abs=1e-12)
        assert metric["nll"] == pytest.approx(-0.9 * math.log(0.9) - 0.1 * math.log(0.1))
        assert metric["brier"] == pytest.approx(0.09)
        assert measured["branch_metrics"]["rank"][str(event)]["count"] == positives
        assert artifact.value_branch_statuses[event] == "passed"
        record_property(f"event_{event}_ovr_ece", metric["ece"])
    record_property("legacy_next_top_label_ece", old_next)
    record_property("legacy_chi_top_label_ece", old_chi)
    assert artifact.value_branch_statuses[RARE] == "unavailable"  # no positives despite 1000 eligible rows
    report = artifact.to_record()
    loaded = CalibrationArtifact.from_record(report, artifact.binding, support, allow_diagnostic=True)
    evaluated = FullDiagnostics()
    for start in range(0, len(check), 128):
        shard = FullDiagnostics()
        add_diagnostics(
            shard, *_evaluation_batch(check[start:start + 128], basis), calibration=loaded, support=support,
        )
        evaluated.merge(shard.state_dict())
    eval_report = evaluated.report(bootstrap_replicates=4)
    observed = eval_report["response_event_calibration"][str(chi)]
    assert (observed["count"], observed["positive_count"], observed["positive_independent_games"]) == (1000, 100, 100)
    assert observed["ece"] == pytest.approx(0.0, abs=1e-8)
    runtime = RuntimePolicy(
        _StaticCandidatePredictor(artifact.binding, check[0]), options=PolicyOptions("empirical_only", 0.5),
        calibration=loaded, support=support, allow_diagnostic_calibration=True,
    ).decide(view, check[0].candidates, "synthetic-ovr-parity")
    candidate = runtime.telemetry["candidates"][str(check[0].action_target)]
    effective = tuple(event in (COMMON, chi) for event in range(21))
    for event_set in (candidate["value_supported"], eval_report[EFFECTIVE]["event_supported"],
                      measured["diagnostics"][EFFECTIVE]["event_supported"]):
        assert tuple(event_set) == effective
    assert candidate["response_probabilities"][chi] == pytest.approx(0.1)
    assert candidate["unsupported_mass"] == 0.0 and _counts(eval_report[EFFECTIVE]) == {0: 1000}


@pytest.mark.parametrize(
    "stale", ["old-calibration-version", "old-statistics-version", "legacy-classification", "missing-event-stats",
              "missing-effective-mass", "wrong-support"]
)
def test_calibration_loader_does_not_relabel_old_unsupported_mass(case, stale):
    artifact, support, _, _, _ = case
    document = copy.deepcopy(artifact.to_record())
    if stale == "old-calibration-version":
        document["version"] = "heldout_temperatures_log_gate_v5_value_mass"
    elif stale == "old-statistics-version":
        document["check_metrics"]["diagnostics"]["statistics_version"] = "whole_game_cluster_statistics_v3_value_mass"
    elif stale == "legacy-classification":
        metrics = document["check_metrics"]
        metrics["branch_metrics"]["response"] = metrics["response_classification_by_true_event"]
    elif stale == "missing-event-stats":
        del document["check_metrics"]["diagnostics"]["response_event_calibration"]
    elif stale == "missing-effective-mass":
        del document["check_metrics"]["diagnostics"][EFFECTIVE]
    else:
        document["check_metrics"]["diagnostics"][EFFECTIVE]["event_supported"] = [True] * 21
    unsigned = {key: value for key, value in document.items() if key != "calibration_sha256"}
    document["calibration_sha256"] = content_hash(unsigned)  # synthetic negative control only
    with pytest.raises(ValueError, match="Calibration.*(version|mass|support|response)"):
        CalibrationArtifact.from_record(document, artifact.binding, support, allow_diagnostic=True)
