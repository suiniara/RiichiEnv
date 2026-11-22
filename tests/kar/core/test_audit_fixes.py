"""Regression tests for KAR audit fixes (P0, P1, P2)."""
# ruff: noqa: PLC0415

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from riichienv import Phase
from riichienv.kar.actions import legal_candidates
from riichienv.kar.agent import functional_match
from riichienv.kar.calibration import CalibrationRow, select_static_temperature
from riichienv.kar.candidate_branches import (
    _choose_own_candidate,
    _discard_efficiency_score,
    _dora_base,
    simulate_candidate_branches,
)
from riichienv.kar.contracts import MJSOUL_RULESET, Response
from riichienv.kar.decision import ReplayOpportunity
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.response import resolve_discard_response
from riichienv.kar.selection_quality import TauSelectionPolicy, _gain_bound, empirical_bernstein_gain_bound
from riichienv.kar.statistics import quantile_ece
from riichienv.kar.support import EventSupport, candidate_value
from riichienv.kar.tokenize import append_action, encode_action_token, encode_prefix


def test_p0_2_mjai_replay_candidate_branches_succeed_with_duplicates_and_dora_overlap():
    env = MJSOUL_RULESET.create_env(seed=17)
    events = [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "1m", "2m", "3m", "4m", "5mr", "5m", "6m", "7m", "8m", "9m", "E", "E"],
                ["1p", "1p", "2p", "3p", "4p", "5pr", "5p", "6p", "7p", "8p", "9p", "P", "P"],
                ["1s", "1s", "2s", "3s", "4s", "5sr", "5s", "6s", "7s", "8s", "9s", "F", "F"],
                ["2m", "2m", "3m", "3m", "4m", "4m", "5m", "6m", "6m", "7m", "8m", "C", "C"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "1m"},
    ]
    ledger = ObserverLedger(0, (0, 1, 2, 3))
    for event in events:
        env.apply_event(event)
        ledger.apply_event(event)
    assert env.phase == Phase.WaitAct
    obs = env.get_observations([0])[0]
    mapped = legal_candidates(obs, drawn_tile=env.drawn_tile)
    assert not any(item.native_execution_ambiguous for item in mapped)
    opportunity = ReplayOpportunity.observe(ledger, obs)
    rows = simulate_candidate_branches(
        env,
        opportunity,
        source_game_id="mjai-replay-branch-test",
        cutoff_event_id=2,
        master_seed=42,
        replicates=1,
        max_decisions=2000,
    )
    assert rows
    assert all(row["status"] == "PASS" for row in rows), [
        (r["candidate_action_id"], r["status"], r.get("reason")) for r in rows
    ]


def test_p0_3_resolve_discard_response_accepts_raw_rust_draw_reasons():
    cases = [
        ("nagashimangan", Response.NAGASHI_MANGAN, MJSOUL_RULESET),
        ("sufuurenta", Response.ABORT_FOUR_WINDS, MJSOUL_RULESET),
        ("suukansansen", Response.ABORT_FOUR_KANS, MJSOUL_RULESET),
        ("sanchaho", Response.ABORT_TRIPLE_RON, replace(MJSOUL_RULESET, sanchaho_is_draw=True)),
    ]
    for raw_reason, expected, ruleset in cases:
        events = [
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
            {"type": "ryukyoku", "reason": raw_reason},
        ]
        result = resolve_discard_response(events, 0, ruleset=ruleset)
        assert result.response_observed and result.response_target == int(expected)


def test_p1_1_baseline_dora_wrap_isolated_bonus_and_riichi_preference():
    # Wind wrap: N(30) -> E(27); Dragon wrap: C(33) -> P(31)
    assert _dora_base(30 * 4) == 27
    assert _dora_base(27 * 4) == 28
    assert _dora_base(33 * 4) == 31
    assert _dora_base(31 * 4) == 32

    # Discarding isolated honor (E=108) must strictly beat discarding 4th copy of 2m in 2m2m2m2m3m.
    hand = [4, 5, 6, 7, 8, 36, 40, 44, 72, 76, 80, 108, 124, 128]
    assert _discard_efficiency_score(hand, 108, set()) > _discard_efficiency_score(hand, 4, set())

    # Suited isolated pair (1p=36, 37) must NOT get isolated bonus compared to singleton 9s (104).
    pair_hand = [0, 4, 8, 12, 16, 20, 36, 37, 72, 76, 80, 104, 124, 125]
    assert _discard_efficiency_score(pair_hand, 104, set()) > _discard_efficiency_score(pair_hand, 36, set())

    # Riichi preference works even when 14-tile observation.is_tenpai is False.
    env = MJSOUL_RULESET.create_env(seed=17)
    events = [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "9s",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "2p", "3p", "4p", "6s", "7s", "E", "E"],
                ["1p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "P", "P", "W"],
                ["1s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "F", "F", "N"],
                ["2m", "2m", "3m", "3m", "4m", "4m", "5m", "6m", "7m", "8m", "9m", "C", "C"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "S"},
    ]
    for event in events:
        env.apply_event(event)
    obs = env.get_observations([0])[0]
    assert obs.is_tenpai is False  # 14-tile WaitAct hand
    mapped = [item for item in legal_candidates(obs, drawn_tile=env.drawn_tile) if not item.native_execution_ambiguous]
    chosen = _choose_own_candidate(mapped, obs, random.Random(0))
    assert chosen.spec.riichi is True


def test_p1_2_functional_match_hash_is_deterministic():
    first = functional_match(seed=17, max_decisions=10000)
    second = functional_match(seed=17, max_decisions=10000)
    assert first["status"] == "PASS"
    assert first["match_sha256"] == second["match_sha256"]
    assert [row["seat"] for row in first["decisions"]] == [row["seat"] for row in second["decisions"]]


def test_p2_neutral_fallback_continuous_temp_and_action_token(public_view):
    # P2-12: neutral_draws fallback prevents -1.5 sabotage of Nagashi Mangan.
    support = EventSupport("a" * 64, (500,) + (0,) * 20, "test")
    resp_logits = [-100.0] * 21
    resp_logits[int(Response.NAGASHI_MANGAN)] = 10.0
    ranks = ((0.0, 0.0, 0.0, 0.0),) * 21
    allowed = tuple(i == int(Response.NAGASHI_MANGAN) for i in range(21))
    worst = candidate_value(0, resp_logits, ranks, allowed, support, unsupported_fallback="worst_case")
    neutral = candidate_value(0, resp_logits, ranks, allowed, support, unsupported_fallback="neutral_draws")
    assert worst.conservative_value == pytest.approx(-1.5)
    assert neutral.conservative_value == pytest.approx(0.0)

    # P2-13: continuous temperature optimization beats coarse (0.5, 1.0, 2.0) grid.
    _, ids = public_view
    rows = [
        CalibrationRow(
            "g0",
            f"g0/{i}",
            (0, 4),
            (3.0, 0.0) + (0.0,) * 544,
            0 if i < 7 else 4,
            action_valid=True,
        )
        for i in range(10)
    ]
    best_t, best_stat = select_static_temperature(rows, "action", "continuous")
    coarse_t, coarse_stat = select_static_temperature(rows, "action", (0.5, 1.0, 2.0))
    assert float(best_stat["nll"]) < float(coarse_stat["nll"])
    assert math.isfinite(best_t) and best_t != coarse_t

    # P2-13: adaptive quantile ECE catches inverted rare-event predictor.
    probs = [0.001] * 90 + [0.02] * 10
    outcomes = [True] * 10 + [False] * 90
    assert quantile_ece(probs, outcomes, bins=10) > 0.01

    # P2-17: encode_action_token matches append_action tail byte-for-byte.
    view, ids = public_view
    prefix = encode_prefix(view, ids)
    for action in ids[:5]:
        assert encode_action_token(prefix, action) == append_action(prefix, action).to_bytes()[-139:]


def test_p2_10_and_p2_11_observer_hand_resampling_and_empirical_bernstein():
    # P2-10: prepare_observer_continuation_wall preserves observer hand while resampling opponent hands.
    env = MJSOUL_RULESET.create_env(seed=17)
    env.reset()
    observer = int(env.current_player)
    orig_hands = [list(h) for h in env.hands]
    clone1 = env.clone()
    clone2 = env.clone()
    clone1.prepare_observer_continuation_wall(observer, 12345)
    clone2.prepare_observer_continuation_wall(observer, 67890)
    assert list(clone1.hands[observer]) == orig_hands[observer]
    assert list(clone2.hands[observer]) == orig_hands[observer]
    assert [list(clone1.hands[s]) for s in range(4) if s != observer] != [
        list(clone2.hands[s]) for s in range(4) if s != observer
    ]
    assert all(len(clone1.hands[s]) == len(orig_hands[s]) for s in range(4))

    # P2-11: empirical Bernstein is strictly tighter than worst-case Hoeffding when variance is low.
    policy = TauSelectionPolicy()
    diffs = [0.25 if i % 2 == 0 else 0.15 for i in range(1000)]
    eb = empirical_bernstein_gain_bound(diffs, candidates=3, policy=policy)
    hoeff = _gain_bound(0.20, 1000, 3, policy)
    assert eb["radius"] < hoeff["radius"]
    assert eb["lower"] > 0.0 and hoeff["lower"] < 0.0
