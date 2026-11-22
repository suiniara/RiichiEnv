"""ML regression tests for KAR audit fixes (scheduler, benchmark non-discard, P0-4 calibration_job)."""
# ruff: noqa: E402, PLR0915

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from riichienv.kar.actions import legal_candidates
from riichienv.kar.budget import BudgetLedger, BudgetPlan, CostForecast
from riichienv.kar.calibration_quality import BranchQualityCriterion, BranchQualityPolicy, CalibrationQualityPolicy
from riichienv.kar.calibration_roles import calibration_is_fit
from riichienv.kar.contracts import MJSOUL_RULESET, content_hash, game_split
from riichienv.kar.learning import benchmark as benchmark_module
from riichienv.kar.learning import calibration_job
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.dataset import DecisionDataset
from riichienv.kar.learning.export import SQL_SCHEMA, learning_contract
from riichienv.kar.learning.inference import FrozenModelPredictor
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.learning.train import create_optimizer, create_scheduler
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.policy import PolicyOptions
from riichienv.kar.response_mask import public_rule_response_mask
from riichienv.kar.runtime import RuntimePolicy
from riichienv.kar.support import EventSupport
from riichienv.kar.tokenize import append_action, encode_prefix
from riichienv.kar.versioning import game_count_contract


def test_p2_4_scheduler_hits_peak_once_and_decays_to_ten_percent():
    model = KARTransformer(ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="mini-sched"))
    opt = create_optimizer(model, learning_rate=1.0)
    sched = create_scheduler(opt, total_updates=100)  # warmup = 3
    fn = sched.lr_lambdas[0]
    assert fn(0) == pytest.approx(1 / 3)
    assert fn(1) == pytest.approx(2 / 3)
    assert fn(2) == pytest.approx(1.0)
    assert fn(3) < 1.0
    assert fn(99) == pytest.approx(0.1)


def test_p1_4_benchmark_candidates_filters_non_discard_actions(monkeypatch):
    env = MJSOUL_RULESET.create_env(seed=17)
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    for event in env.mjai_log:
        ledger.apply_event(event)
    obs = env.get_observations([0])[0]
    ids = tuple(action.global_id for action in legal_candidates(obs, drawn_tile=env.drawn_tile))
    snapshot = ledger.snapshot()
    # Include a non-discard candidate (PASS=221) alongside discards; benchmark must filter < 148.
    mixed_ids = tuple(sorted((*ids, 221)))
    prefix = encode_prefix(snapshot, mixed_ids)
    full = append_action(prefix, ids[0])
    row = {
        "decision_row_id": "bench-mixed-0",
        "tokens": full.to_bytes()[108:],
        "candidates": mixed_ids,
        "decide_index": prefix.decide_index,
        "action_index": len(full.rows) - 1,
    }

    class _FakeDataset:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return row

    monkeypatch.setattr(benchmark_module, "DecisionDataset", _FakeDataset)
    report = benchmark_module.benchmark_candidates(Path("unused"), preset="small", repetitions=1)
    assert report["candidates"] == len([a for a in mixed_ids if a < 148])


def test_p0_4_end_to_end_calibration_job_collects_heldout_candidate_branches(tmp_path):
    salt = "kar-riichienv-v1-games"
    names = [f"e2e-cal-branch-{i}" for i in range(1000)]
    train_game = next(g for g in names if game_split(g, salt) == "train")
    val_games = [g for g in names if game_split(g, salt) == "validation"]
    fit_game = next(g for g in val_games if calibration_is_fit(g))
    check_game = next(g for g in val_games if not calibration_is_fit(g))

    env = MJSOUL_RULESET.create_env(seed=17)
    ledger_obs = ObserverLedger(0, (0, 1, 2, 3))
    for event in env.mjai_log:
        ledger_obs.apply_event(event)
    obs = env.get_observations([0])[0]
    ids = tuple(action.global_id for action in legal_candidates(obs, drawn_tile=env.drawn_tile))
    discards = [action for action in ids if action < 148]
    snapshot = ledger_obs.snapshot()
    prefix = encode_prefix(snapshot, ids)
    target = discards[0]
    full = append_action(prefix, target)
    full_bytes = full.to_bytes()[108:]
    history_bytes = full_bytes[: (prefix.history_rows + 1) * 139]
    state_bytes = full_bytes[(prefix.history_rows + 1) * 139 : (prefix.history_rows + 64) * 139]
    action_bytes = full_bytes[-139:]
    padded = list(ids) + [546] * (32 - len(ids))
    candidate_blob = struct.pack("<32H", *padded)
    allowed = public_rule_response_mask(snapshot, target)

    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    db_path = data_dir / "decisions.sqlite3"
    available = {"train": 1, "validation": 2, "test": 0}
    scope = {
        **learning_contract(),
        "candidate_branch_generation": {"enabled": True},
        "protocol_sha256": "d" * 64,
        "protocol_descriptor_sha256": "e" * 64,
        "available_split_counts": available,
        "split_salt": salt,
    }
    games_meta = []
    with sqlite3.connect(db_path) as db:
        db.executescript(SQL_SCHEMA)
        db.executemany("INSERT INTO metadata VALUES (?,?)", [("complete", "true"), ("scope", json.dumps(scope))])
        for game_id, split_name in ((train_game, "train"), (fit_game, "validation"), (check_game, "validation")):
            games_meta.append({"game_id": game_id, "split": split_name})
            db.execute("INSERT INTO games VALUES (?,?,?,?,?)", (game_id, split_name, "a" * 64, "[1,2,3,4]", "b" * 64))
            db.execute("INSERT INTO histories VALUES (?,?,?)", (game_id, 0, history_bytes))
            row_id = f"{game_id}/0"
            db.execute(
                "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row_id,
                    game_id,
                    0,
                    0,
                    0,
                    0,
                    0,
                    prefix.history_rows,
                    state_bytes,
                    action_bytes,
                    candidate_blob,
                    target,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    bytes(allowed),
                    "explicit",
                    "{}",
                ),
            )
            if split_name in {"train", "validation"} and game_id != fit_game:
                for candidate_id in discards:
                    cond = append_action(prefix, candidate_id)
                    cond_bytes = cond.to_bytes()[108:]
                    cand_allowed = [bool(v) for v in public_rule_response_mask(snapshot, candidate_id)]
                    payload = {
                        "version": "kar_candidate_action_branch_v2",
                        "split": split_name,
                        "source_decision_row_id": row_id,
                        "source_game_id": game_id,
                        "source_cutoff_event_id": 0,
                        "observer": 0,
                        "source_state_identity": f"state-{game_id}",
                        "candidates": list(ids),
                        "expected_branch_action_ids": discards,
                        "candidate_action_id": candidate_id,
                        "replicate": 0,
                        "replicates": 1,
                        "decide_index": prefix.decide_index,
                        "action_index": prefix.decide_index + 1,
                        "response_target": 0,
                        "response_allowed": cand_allowed,
                        "final_rank": 0,
                        "status": "PASS",
                        "complete": True,
                        "tokens_hex": cond_bytes.hex(),
                    }
                    payload["branch_sha256"] = content_hash(payload)
                    stored = {k: v for k, v in payload.items() if k != "tokens_hex"}
                    db.execute(
                        "INSERT INTO branch_samples VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            f"{row_id}/a{candidate_id}/r0",
                            game_id,
                            row_id,
                            split_name,
                            candidate_id,
                            0,
                            "PASS",
                            cond_bytes,
                            json.dumps(stored),
                        ),
                    )
    manifest = {
        "status": "PASS",
        "complete": True,
        "scope": scope,
        **game_count_contract(available),
        "games": games_meta,
        "rho_E": 1.0,
        "rho_V": 1.0,
        "database_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
        "counts": {
            "train": {"games": 1, "examples": 1},
            "validation": {"games": 2, "examples": 2},
            "test": {"games": 0, "examples": 0},
        },
        "candidate_branch_counts": {
            "complete_source_decisions_by_split": {"train": 1, "validation": 1, "test": 0},
            "candidate_action_source_counts_by_split": {
                "train": {str(action): 1 for action in discards},
                "validation": {str(action): 1 for action in discards},
                "test": {},
            },
        },
    }
    manifest["manifest_sha256"] = content_hash(manifest)
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    selected_val = [{"game_id": g, "source_role": "main_cohort_candidate"} for g in (fit_game, check_game)]
    val_manifest = {"games": selected_val, "scope": scope, "manifest_sha256": content_hash(selected_val)}

    torch.manual_seed(11)
    model = KARTransformer(ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="e2e-cal")).eval()
    predictor = FrozenModelPredictor(model, manifest["manifest_sha256"], use_rule_features=True)
    quality = CalibrationQualityPolicy(
        {h: 20.0 for h in ("action", "response", "rank")},
        {h: 2.0 for h in ("action", "response", "rank")},
        {h: 1.0 for h in ("action", "response", "rank")},
        1,
        1,
        0.0,
        "synthetic e2e calibration_job test",
        BranchQualityPolicy(
            tuple(BranchQualityCriterion(h, 0, 20.0, 2.0, 1.0, 1) for h in ("response", "rank")),
            "synthetic e2e branch policy",
        ),
    )
    validation = DecisionDataset(data_dir, "validation", include_candidate_branches=False)
    out = tmp_path / "cal_out"
    out.mkdir()
    ledger = BudgetLedger(
        out / "costs.jsonl",
        BudgetPlan(
            "M2",
            100.0,
            0.0,
            CostForecast("M2", 2, 1.0, 0.0, 0.0, "e2e synthetic", "a" * 64),
        ),
    )
    with torch.no_grad(), ledger.record_compute("e2e-cal"):
        artifact, support = calibration_job._archive_and_fit(
            predictor,
            validation,
            val_manifest,
            out,
            ledger=ledger,
            quality_policy=quality,
            temperatures=(1.0,),
            microbatch=1,
            plan_sha256="a" * 64,
        )
    cond_val = artifact.branch_quality_report["candidate_action_conditional_validation"]
    assert cond_val["status"] == "passed"
    assert cond_val["distinct_source_decision_and_candidate_action"] == len(discards)
    assert cond_val["independent_games"] == 1
    assert artifact.empirical_check_valid
    assert support.candidate_branch_source_decisions == 1
    assert sum(support.candidate_action_source_counts) == len(discards)
    with sqlite3.connect(out / "logits.sqlite3") as ldb:
        archived_count = ldb.execute("SELECT COUNT(*) FROM candidate_branch_logits").fetchone()[0]
    assert archived_count == len(discards)

    # Now re-run with 500-game train support to verify production RuntimePolicy executes finite-tau KL reranking.
    rich_support = EventSupport(
        manifest["manifest_sha256"],
        (500,) + (0,) * 20,
        "e2e synthetic rich support",
        100,
        tuple(50 if a in discards else 0 for a in range(148)),
    )
    out2 = tmp_path / "cal_out_rich"
    out2.mkdir()
    ledger2 = BudgetLedger(
        out2 / "costs.jsonl",
        BudgetPlan(
            "M2",
            100.0,
            0.0,
            CostForecast("M2", 2, 1.0, 0.0, 0.0, "e2e synthetic rich", "a" * 64),
        ),
    )
    orig_support = calibration_job.dataset_event_support
    calibration_job.dataset_event_support = lambda _v: rich_support
    try:
        with torch.no_grad(), ledger2.record_compute("e2e-cal-rich"):
            artifact_rich, _ = calibration_job._archive_and_fit(
                predictor,
                validation,
                val_manifest,
                out2,
                ledger=ledger2,
                quality_policy=quality,
                temperatures=(1.0,),
                microbatch=1,
                plan_sha256="a" * 64,
            )
    finally:
        calibration_job.dataset_event_support = orig_support

    policy = RuntimePolicy(
        predictor,
        options=PolicyOptions("empirical_only", 0.5),
        support=rich_support,
        calibration=artifact_rich,
        seed=7,
    )
    decision = policy.decide(snapshot, ids, "e2e-prod-finite-tau")
    assert decision.evidence_status == "empirical_only"
    assert decision.fallback_reason is None
    assert decision.telemetry["candidate_forward_count"] >= 2
    assert "branch_quality_summary" in decision.telemetry
    assert "branch_quality_report" not in decision.telemetry
