"""Stage-1 Automated Contract Verification Suite for KAR V2.5 (Rust + PyTorch)."""

from __future__ import annotations

import io
import json
import math
import random

import numpy as np
import pytest
import torch

import riichienv
from riichienv import _riichienv as native
from riichienv.kar.v2 import (
    CriticModelConfig,
    CriticWarmupGate,
    FrozenGAERolloutBatch,
    FrozenGRP,
    HanchanTrajectoryInput,
    KARV2Actor,
    OracleCriticV2,
    PFSPOpponentPool,
    PureStateGRP,
    RolloutTruncation,
    RolloutVersionTag,
    RoundBoundary,
    V2ModelConfig,
    belief_auxiliary_loss,
    compute_explained_variance,
    compute_hanchan_gae,
    compute_pbrs_round_rewards,
    evaluate_grp_calibration,
    masked_entropy_and_kl_fp32,
    oracle_critic_loss,
    ppo_actor_loss,
    train_pure_state_grp_step,
)
from riichienv.kar.v2.model import (
    ACTION_COUNT,
    CAND_DYN_DIM,
    CAND_PAD_ID,
    EVENT_BYTES,
    GLOBAL_FEATS_DIM,
    MAX_CANDIDATES,
    ORACLE_PLAYER_DIM,
    ORACLE_TILE_DIM,
    PLAYER_FEATS_DIM,
    TILE_DIM,
    TILE_FEATS_DIM,
    build_static_manifest_buffers,
)


def _dummy_batch(
    batch_size: int = 4,
    seq_len: int = 16,
    *,
    force_k1: bool = False,
    include_null_slot_actions: bool = False,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(2026)
    tile_feats = torch.randn(batch_size, TILE_DIM, TILE_FEATS_DIM, generator=g)
    oracle_tile = torch.randn(batch_size, TILE_DIM, ORACLE_TILE_DIM, generator=g)
    global_feats = torch.randn(batch_size, GLOBAL_FEATS_DIM, generator=g)
    player_feats = torch.randn(batch_size, 4, PLAYER_FEATS_DIM, generator=g)
    oracle_player = torch.randn(batch_size, 3, ORACLE_PLAYER_DIM, generator=g)

    events = torch.zeros((batch_size, seq_len, EVENT_BYTES), dtype=torch.uint8)
    events[:, 0, 0] = 1  # Mandatory initial DORA at index 0
    events[:, 0, 1] = 4  # ACTOR_TABLE
    events[:, 0, 2] = 4  # SOURCE_NONE
    events[:, 0, 3] = 5  # 5m indicator
    events[:, 0, 4] = 37
    events[:, 0, 5] = 37
    for t in range(1, seq_len):
        events[:, t, 0] = 3  # DAHAI
        events[:, t, 1] = t % 4
        events[:, t, 2] = 4
        events[:, t, 3] = t % 37
        events[:, t, 4] = 37
        events[:, t, 5] = 37
        events[:, t, 8] = t
    event_len = torch.full((batch_size,), seq_len, dtype=torch.long)

    cand_ids = torch.full((batch_size, MAX_CANDIDATES), CAND_PAD_ID, dtype=torch.long)
    cand_mask = torch.zeros((batch_size, MAX_CANDIDATES), dtype=torch.bool)
    cand_mode = torch.zeros((batch_size, MAX_CANDIDATES), dtype=torch.long)
    cand_dyn = torch.randn(batch_size, MAX_CANDIDATES, CAND_DYN_DIM, generator=g)

    if force_k1:
        cand_ids[:, 0] = 10  # Single forced discard
        cand_mask[:, 0] = True
        cand_mode[:, 0] = 1
    elif include_null_slot_actions:
        # Include TSUMO(219), KYUSHU(220), PASS(221), CHI(222), PON(312), DAIMINKAN(432), RON(543)
        ids = [0, 148, 182, 219, 220, 221, 222, 312, 432, 543]
        modes = [1, 3, 3, 5, 6, 4, 2, 2, 3, 5]
        for idx, (gid, mode) in enumerate(zip(ids, modes, strict=True)):
            cand_ids[:, idx] = gid
            cand_mask[:, idx] = True
            cand_mode[:, idx] = mode
    else:
        for k in range(12):
            cand_ids[:, k] = k * 4
            cand_mask[:, k] = True
            cand_mode[:, k] = 1

    return {
        "tile_feats": tile_feats,
        "oracle_tile": oracle_tile,
        "events": events,
        "event_len": event_len,
        "global_feats": global_feats,
        "player_feats": player_feats,
        "oracle_player": oracle_player,
        "cand_ids": cand_ids,
        "cand_mask": cand_mask,
        "cand_mode": cand_mode,
        "cand_dyn": cand_dyn,
    }


# =========================================================================
# 1. Cross-Language Rust <-> Python 546-Action Manifest Parity
# =========================================================================
def test_rust_and_python_546_manifest_tables_match_byte_for_byte() -> None:
    rust_tbl = native.v2_manifest_table()
    py_tbl = build_static_manifest_buffers()

    assert rust_tbl["num_rows"] == ACTION_COUNT + 1
    r_type = torch.from_numpy(np.frombuffer(rust_tbl["action_type"], dtype=np.uint8).copy()).long()
    r_tgt = torch.from_numpy(np.frombuffer(rust_tbl["target_base34"], dtype=np.uint8).copy()).long()
    r_c0 = torch.from_numpy(np.frombuffer(rust_tbl["consumed0_base34"], dtype=np.uint8).copy()).long()
    r_c1 = torch.from_numpy(np.frombuffer(rust_tbl["consumed1_base34"], dtype=np.uint8).copy()).long()
    r_src = torch.from_numpy(np.frombuffer(rust_tbl["source_rel"], dtype=np.uint8).copy()).long()
    r_has = torch.from_numpy(np.frombuffer(rust_tbl["has_slot_flags"], dtype=np.float32).copy()).view(-1, 4)
    r_flags = torch.from_numpy(np.frombuffer(rust_tbl["static_flags"], dtype=np.float32).copy()).view(-1, 4)

    assert torch.equal(r_type, py_tbl["action_type"])
    assert torch.equal(r_tgt, py_tbl["target_base34"])
    assert torch.equal(r_c0, py_tbl["c0_base34"])
    assert torch.equal(r_c1, py_tbl["c1_base34"])
    assert torch.equal(r_src, py_tbl["source_rel"])
    assert torch.equal(r_has, py_tbl["has_slot_flags"])
    assert torch.equal(r_flags, py_tbl["static_flags"])


# =========================================================================
# 2. Rule-Aware HandEvaluator::calc_with_rule via PyO3
# =========================================================================
def test_pyo3_calc_with_rule_tenhou_vs_mjsoul() -> None:
    kokushi_13 = [0, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132]
    cond = native.Conditions(tsumo=False, player_wind=native.Wind.South, round_wind=native.Wind.East)
    res_tenhou = native.v2_calc_with_rule(kokushi_13, [], 1, [4], [], cond, riichienv.GameRule.default_tenhou())
    res_mjs = native.v2_calc_with_rule(kokushi_13, [], 1, [4], [], cond, riichienv.GameRule.default_mjsoul())
    assert res_tenhou.han == 13 and res_tenhou.ron_agari == 32_000
    assert res_mjs.han == 26 and res_mjs.ron_agari == 64_000


# =========================================================================
# 3. Pure-State GRP Permutation Equivariance, Freeze & Telescoping PBRS
# =========================================================================
def test_pure_state_grp_equivariance_freeze_and_telescoping() -> None:
    torch.manual_seed(7)
    grp = PureStateGRP(hidden_dim=64)
    payoffs = [0.90, 0.45, 0.0, -1.35]
    frozen = FrozenGRP(
        grp,
        grp_version="grp_v2_test",
        ruleset_id="tenhou_houou_4p_hanchan_v1",
        ruleset_sha256="abc123",
        rank_payoffs=payoffs,
    )
    assert all(not p.requires_grad for p in frozen.model.parameters())

    # Mismatched checksum MUST fail
    with pytest.raises(ValueError, match="checksum mismatch"):
        FrozenGRP(
            grp,
            grp_version="grp_v2_test",
            ruleset_id="tenhou_houou_4p_hanchan_v1",
            ruleset_sha256="abc123",
            rank_payoffs=payoffs,
            expected_grp_sha256="deadbeef",
        )

    # Exact 4-seat cyclic permutation equivariance (< 1e-6)
    base_scores = [38_200, 24_100, 21_700, 16_000]
    base_probs, base_phi = frozen.predict_all_seats(base_scores, round_wind=1, oya=2, honba=1, kyoutaku=2)
    for shift in (1, 2, 3):
        rot_scores = [base_scores[(i - shift) % 4] for i in range(4)]
        rot_oya = (2 + shift) % 4
        rot_probs, rot_phi = frozen.predict_all_seats(rot_scores, round_wind=1, oya=rot_oya, honba=1, kyoutaku=2)
        for orig_seat in range(4):
            new_seat = (orig_seat + shift) % 4
            assert torch.allclose(rot_probs[new_seat], base_probs[orig_seat], atol=1e-6)
            assert abs(float(rot_phi[new_seat] - base_phi[orig_seat])) < 1e-6

    # Exact PBRS telescoping across an 8-round Hanchan
    phis = [0.02, 0.35, 0.18, -0.10, -0.42, 0.05, 0.55, 0.72]
    r_term = 0.90
    rewards = compute_pbrs_round_rewards(phis, r_term)
    assert abs(sum(rewards) - (r_term - phis[0])) < 1e-6


# =========================================================================
# 4. Full-Hanchan Segmented GAE (Zero-Decision Skipped Rounds & Truncation)
# =========================================================================
def test_hanchan_gae_handles_skipped_rounds_and_truncation_bootstrap() -> None:
    # 5-round Hanchan where seat had 0 decisions in round 1 (e.g. opponent Tenhou)
    # and 0 decisions in round 4 (last round ended before seat acted)
    phis = [0.0, 0.25, -0.15, 0.40, 0.60]
    steps = [
        RoundBoundary(round_index=0, phi_k=phis[0], v_raw=0.05),
        RoundBoundary(round_index=0, phi_k=phis[0], v_raw=0.10),
        # round 1 skipped!
        RoundBoundary(round_index=2, phi_k=phis[2], v_raw=-0.10),
        RoundBoundary(round_index=3, phi_k=phis[3], v_raw=0.50),
        # round 4 skipped before Hanchan termination!
    ]
    r_term = 0.90
    step_rewards, advs, rets = compute_hanchan_gae(
        steps, phis, terminated=True, terminal_reward=r_term, gamma=1.0, gae_lambda=0.95
    )
    # Sum of step rewards MUST strictly equal R_terminal - Phi_0!
    assert abs(sum(step_rewards) - (r_term - phis[0])) < 1e-6
    assert len(advs) == len(steps) and len(rets) == len(steps)

    # Mid-Hanchan truncation MUST reject missing bootstrap value
    with pytest.raises(ValueError, match="truncated_bootstrap_v_shaped"):
        compute_hanchan_gae(steps, phis, terminated=False)

    # Mid-Hanchan intra-round truncation (still inside round 3) bootstraps cleanly with non-zero V_shaped
    _, adv_trunc, ret_trunc = compute_hanchan_gae(
        steps, phis[:4], terminated=False, truncated_bootstrap_v_shaped=0.33, gamma=1.0, gae_lambda=0.95
    )
    assert abs(ret_trunc[-1] - 0.33) < 1e-6
    assert abs(adv_trunc[-1] - (0.33 - steps[-1].v_shaped)) < 1e-6


# =========================================================================
# 5. Contract v2.4 P0-2: Dual-Encoder Total Independence (Actor <-> Critic)
# =========================================================================
def test_oracle_critic_has_zero_dependency_on_actor_weights_or_gradients() -> None:
    torch.manual_seed(11)
    actor = KARV2Actor(V2ModelConfig.v2_base())
    critic = OracleCriticV2(CriticModelConfig())
    batch = _dummy_batch(batch_size=3, seq_len=12, include_null_slot_actions=True)
    grp_probs = torch.full((3, 4), 0.25)
    payoffs = torch.tensor([0.90, 0.45, 0.0, -1.35])

    critic_out_before = critic(
        tile_feats=batch["tile_feats"],
        oracle_tile=batch["oracle_tile"],
        events=batch["events"],
        event_len=batch["event_len"],
        global_feats=batch["global_feats"],
        player_feats=batch["player_feats"],
        oracle_player=batch["oracle_player"],
        grp_prior_probs=grp_probs,
        rank_payoffs=payoffs,
    )

    # Randomly scramble ALL Actor parameters -> Critic output MUST be bit-for-bit identical!
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(torch.randn_like(p) * 5.0)

    critic_out_after = critic(
        tile_feats=batch["tile_feats"],
        oracle_tile=batch["oracle_tile"],
        events=batch["events"],
        event_len=batch["event_len"],
        global_feats=batch["global_feats"],
        player_feats=batch["player_feats"],
        oracle_player=batch["oracle_player"],
        grp_prior_probs=grp_probs,
        rank_payoffs=payoffs,
    )
    assert torch.equal(critic_out_before["rank_logits"], critic_out_after["rank_logits"])
    assert torch.equal(critic_out_before["v_shaped"], critic_out_after["v_shaped"])

    # Backprop BOTH gae_mse and mc_mse Critic losses -> zero gradient touches Actor!
    for mode in ("gae_mse", "mc_mse"):
        critic.zero_grad(set_to_none=True)
        actor.zero_grad(set_to_none=True)
        out = critic(
            tile_feats=batch["tile_feats"],
            oracle_tile=batch["oracle_tile"],
            events=batch["events"],
            event_len=batch["event_len"],
            global_feats=batch["global_feats"],
            player_feats=batch["player_feats"],
            oracle_player=batch["oracle_player"],
            grp_prior_probs=grp_probs,
            rank_payoffs=payoffs,
        )
        loss_dict = oracle_critic_loss(
            rank_logits=out["rank_logits"],
            v_shaped=out["v_shaped"],
            final_rank_idx=torch.tensor([0, 2, 3]),
            shaped_target=torch.tensor([0.5, -0.1, -0.8]),
            td_mode=mode,
        )
        loss_dict["loss"].backward()
        assert all(p.grad is None for p in actor.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())


# =========================================================================
# 6. All 4 Model Presets, NULL Slots, K=1 Forced Action & PFSP Seat Masking
# =========================================================================
@pytest.mark.parametrize(
    "cfg",
    [
        V2ModelConfig.spatial_only(),
        V2ModelConfig.v2_base(),
        V2ModelConfig.v2_base_flat(),
        V2ModelConfig.v2_large(),
    ],
)
def test_all_v2_presets_handle_null_slot_actions_without_nan(cfg: V2ModelConfig) -> None:
    model = KARV2Actor(cfg)
    batch = _dummy_batch(batch_size=2, seq_len=10, include_null_slot_actions=True)
    out = model(
        tile_feats=batch["tile_feats"],
        events=batch["events"],
        event_len=batch["event_len"],
        global_feats=batch["global_feats"],
        player_feats=batch["player_feats"],
        cand_ids=batch["cand_ids"],
        cand_mask=batch["cand_mask"],
        cand_mode=batch["cand_mode"],
        cand_dyn=batch["cand_dyn"],
        return_belief=True,
    )
    assert out["logits"].shape == (2, MAX_CANDIDATES)
    assert torch.isfinite(out["logits"]).all()
    assert out["tenpai_logits"].shape == (2, 3)
    assert out["wait_logits"].shape == (2, 3, TILE_DIM)
    assert out["dealin_logits"].shape == (2, TILE_DIM)
    assert out["hand_cnt_logits"].shape == (2, 3, TILE_DIM, 5)


def test_ppo_masking_k1_forced_steps_and_opponent_seats_zero_actor_grad() -> None:
    actor = KARV2Actor(V2ModelConfig.v2_base())
    # Case 1: K=1 forced action (e.g., riichi tsumogiri) -> zero NaN & strictly zero Actor grad!
    b_k1 = _dummy_batch(batch_size=3, seq_len=8, force_k1=True)
    out = actor(
        tile_feats=b_k1["tile_feats"],
        events=b_k1["events"],
        event_len=b_k1["event_len"],
        global_feats=b_k1["global_feats"],
        player_feats=b_k1["player_feats"],
        cand_ids=b_k1["cand_ids"],
        cand_mask=b_k1["cand_mask"],
        cand_mode=b_k1["cand_mode"],
        cand_dyn=b_k1["cand_dyn"],
    )
    ent, kl = masked_entropy_and_kl_fp32(out["logits"], b_k1["cand_mask"], out["logits"].detach())
    assert torch.isfinite(ent).all() and torch.allclose(ent, torch.zeros_like(ent))
    assert kl is not None and torch.isfinite(kl).all() and torch.allclose(kl, torch.zeros_like(kl))

    ppo_res = ppo_actor_loss(
        logits=out["logits"],
        cand_ids=b_k1["cand_ids"],
        cand_mask=b_k1["cand_mask"],
        chosen_cand_idx=torch.zeros(3, dtype=torch.long),
        chosen_global_id=b_k1["cand_ids"][:, 0],
        old_log_prob=torch.zeros(3),
        advantages=torch.tensor([1.0, -1.0, 0.5]),
        is_learner_seat=torch.ones(3, dtype=torch.bool),
        ref_bc_logits=out["logits"].detach(),
    )
    assert ppo_res["active_count"].item() == 0.0
    ppo_res["loss"].backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)
    assert grad_norm == 0.0

    # Case 2: Non-learner PFSP opponent seats (is_learner_seat=False) with K > 1 -> strictly zero Actor grad!
    b_multi = _dummy_batch(batch_size=3, seq_len=8, force_k1=False)
    actor.zero_grad(set_to_none=True)
    out_m = actor(
        tile_feats=b_multi["tile_feats"],
        events=b_multi["events"],
        event_len=b_multi["event_len"],
        global_feats=b_multi["global_feats"],
        player_feats=b_multi["player_feats"],
        cand_ids=b_multi["cand_ids"],
        cand_mask=b_multi["cand_mask"],
        cand_mode=b_multi["cand_mode"],
        cand_dyn=b_multi["cand_dyn"],
    )
    opp_res = ppo_actor_loss(
        logits=out_m["logits"],
        cand_ids=b_multi["cand_ids"],
        cand_mask=b_multi["cand_mask"],
        chosen_cand_idx=torch.zeros(3, dtype=torch.long),
        chosen_global_id=b_multi["cand_ids"][:, 0],
        old_log_prob=torch.full((3,), -2.0),
        advantages=torch.tensor([1.0, -1.0, 0.5]),
        is_learner_seat=torch.zeros(3, dtype=torch.bool),
        ref_bc_logits=out_m["logits"].detach(),
    )
    assert opp_res["active_count"].item() == 0.0
    opp_res["loss"].backward()
    opp_grad_norm = sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)
    assert opp_grad_norm == 0.0

    # Case 3: BF16 logits MUST upcast Softmax, Entropy, KL, and PPO loss to fp32 without NaN!
    bf16_logits = out_m["logits"].detach().to(torch.bfloat16).requires_grad_(True)
    bf16_ref = out_m["logits"].detach().to(torch.bfloat16)
    ent_fp32, kl_fp32 = masked_entropy_and_kl_fp32(bf16_logits, b_multi["cand_mask"], bf16_ref)
    assert ent_fp32.dtype == torch.float32 and torch.isfinite(ent_fp32).all()
    assert kl_fp32 is not None and kl_fp32.dtype == torch.float32 and torch.isfinite(kl_fp32).all()
    bf16_ppo = ppo_actor_loss(
        logits=bf16_logits,
        cand_ids=b_multi["cand_ids"],
        cand_mask=b_multi["cand_mask"],
        chosen_cand_idx=torch.zeros(3, dtype=torch.long),
        chosen_global_id=b_multi["cand_ids"][:, 0],
        old_log_prob=torch.full((3,), -2.0),
        advantages=torch.tensor([1.0, -1.0, 0.5]),
        is_learner_seat=torch.ones(3, dtype=torch.bool),
        ref_bc_logits=bf16_ref,
    )
    assert bf16_ppo["loss"].dtype == torch.float32 and torch.isfinite(bf16_ppo["loss"]).all()

    # Case 4: Mismatched chosen_global_id MUST fail fast
    with pytest.raises(ValueError, match="chosen_global_id"):
        ppo_actor_loss(
            logits=out["logits"].detach(),
            cand_ids=b_k1["cand_ids"],
            cand_mask=b_k1["cand_mask"],
            chosen_cand_idx=torch.zeros(3, dtype=torch.long),
            chosen_global_id=torch.full((3,), 99, dtype=torch.long),
            old_log_prob=torch.zeros(3),
            advantages=torch.zeros(3),
            is_learner_seat=torch.ones(3, dtype=torch.bool),
        )


# =========================================================================
# 7. Live PyO3 End-to-End Integration: RiichiEnv -> Rust V2 -> PyTorch V2
# =========================================================================
def test_live_riichienv_to_rust_v2_to_pytorch_actor_and_critic() -> None:
    env = riichienv.RiichiEnv(seed=2026)
    obs_map = env.reset()
    observer = next(iter(obs_map.keys()))

    step_data = native.v2_encode_env_step(env, observer, "tenhou", False)
    tile_feats = torch.from_numpy(np.frombuffer(step_data["tile_feats"], dtype=np.float32).copy()).view(
        1, TILE_DIM, TILE_FEATS_DIM
    )
    global_feats = torch.from_numpy(np.frombuffer(step_data["global_feats"], dtype=np.float32).copy()).view(
        1, GLOBAL_FEATS_DIM
    )
    player_feats = torch.from_numpy(np.frombuffer(step_data["player_feats"], dtype=np.float32).copy()).view(
        1, 4, PLAYER_FEATS_DIM
    )
    cand_ids = torch.from_numpy(np.frombuffer(step_data["cand_ids"], dtype=np.int16).copy()).view(1, MAX_CANDIDATES)
    cand_mask = (
        torch.from_numpy(np.frombuffer(step_data["cand_mask"], dtype=np.uint8).copy()).view(1, MAX_CANDIDATES).bool()
    )
    cand_mode = torch.from_numpy(np.frombuffer(step_data["cand_mode"], dtype=np.uint8).copy()).view(1, MAX_CANDIDATES)
    cand_dyn = torch.from_numpy(np.frombuffer(step_data["cand_dyn"], dtype=np.float32).copy()).view(
        1, MAX_CANDIDATES, CAND_DYN_DIM
    )
    oracle_tile = torch.from_numpy(np.frombuffer(step_data["oracle_tile"], dtype=np.float32).copy()).view(
        1, TILE_DIM, ORACLE_TILE_DIM
    )
    oracle_player = torch.from_numpy(np.frombuffer(step_data["oracle_player"], dtype=np.float32).copy()).view(
        1, 3, ORACLE_PLAYER_DIM
    )

    assert step_data["num_valid"] >= 1
    assert step_data["num_valid"] <= MAX_CANDIDATES

    # Decode first legal global_id back to native Action(s)
    first_gid = int(cand_ids[0, 0].item())
    recovered = native.v2_decode_global_id(first_gid, obs_map[observer], None)
    assert len(recovered) >= 1

    # Forward through Actor + independent Oracle Critic using REAL Rust-encoded events!
    ev_len = int(step_data["event_len"])
    assert ev_len >= 2  # initial DORA at idx 0 + dealer's opening TSUMO_SELF
    events = torch.from_numpy(np.frombuffer(step_data["events"], dtype=np.uint8).copy()).view(1, ev_len, EVENT_BYTES)
    assert int(events[0, 0, 0].item()) == 1  # EventKind::Dora at index 0
    event_len = torch.tensor([ev_len], dtype=torch.long)

    actor = KARV2Actor(V2ModelConfig.v2_base())
    critic = OracleCriticV2(CriticModelConfig())
    a_out = actor(
        tile_feats=tile_feats,
        events=events,
        event_len=event_len,
        global_feats=global_feats,
        player_feats=player_feats,
        cand_ids=cand_ids,
        cand_mask=cand_mask,
        cand_mode=cand_mode,
        cand_dyn=cand_dyn,
    )
    c_out = critic(
        tile_feats=tile_feats,
        oracle_tile=oracle_tile,
        events=events,
        event_len=event_len,
        global_feats=global_feats,
        player_feats=player_feats,
        oracle_player=oracle_player,
        grp_prior_probs=torch.full((1, 4), 0.25),
        rank_payoffs=torch.tensor([0.90, 0.45, 0.0, -1.35]),
    )
    assert torch.isfinite(a_out["logits"]).all()
    assert torch.isfinite(c_out["v_shaped"]).all()


# =========================================================================
# 8. Contract v2.4.1: GAE Target stopgrad, RolloutVersionTag, & obs.furiten
# =========================================================================
def test_oracle_critic_loss_enforces_stopgrad_on_gae_target_and_version_tags() -> None:
    # Construct a target tensor that DEPENDS on v_shaped with requires_grad=True!
    # Without stopgrad/detach, d/dv (v - (0.5*v + 1))^2 = (0.5*v - 1) instead of 2*(v - target).
    v_shaped = torch.tensor([0.4, -0.2, 0.8], requires_grad=True)
    leaky_gae_target = 0.5 * v_shaped + 1.0  # intentionally attached to v_shaped!
    rank_logits = torch.zeros((3, 4))
    final_ranks = torch.tensor([0, 1, 3])
    alpha_td = 0.5

    loss_dict = oracle_critic_loss(
        rank_logits=rank_logits,
        v_shaped=v_shaped,
        final_rank_idx=final_ranks,
        shaped_target=leaky_gae_target,
        alpha_td=alpha_td,
        td_mode="gae_mse",
    )
    loss_dict["loss"].backward()

    # Exact gradient under stopgrad: d/dv [alpha_td * mean((v - c)^2)] = alpha_td * 2 * (v - c) / N
    expected_grad = alpha_td * 2.0 * (v_shaped.detach() - leaky_gae_target.detach()) / 3.0
    assert v_shaped.grad is not None
    assert torch.allclose(v_shaped.grad, expected_grad, atol=1e-6)

    # RolloutVersionTag checks both policy_version AND critic_version
    tag = RolloutVersionTag(
        policy_version=10,
        critic_version=4,
        grp_sha256="grp_ok",
        ruleset_sha256="rule_ok",
    )
    tag.validate_compatible(
        current_policy_version=12,
        expected_critic_version=4,
        expected_grp_sha256="grp_ok",
        expected_ruleset_sha256="rule_ok",
        max_staleness=2,
    )
    with pytest.raises(ValueError, match="critic_version"):
        tag.validate_compatible(
            current_policy_version=11,
            expected_critic_version=5,
            expected_grp_sha256="grp_ok",
            expected_ruleset_sha256="rule_ok",
        )
    with pytest.raises(ValueError, match="Stale rollout"):
        tag.validate_compatible(
            current_policy_version=15,
            expected_critic_version=4,
            expected_grp_sha256="grp_ok",
            expected_ruleset_sha256="rule_ok",
            max_staleness=2,
        )


def test_observation_preserves_furiten_state_and_241_byte_visible_counts() -> None:
    fur = riichienv.FuritenState(permanent=False, temporary=True, riichi=False)
    assert fur.any() and fur.non_river_active()
    obs = riichienv.Observation(
        player_id=0,
        hands=[[0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 112, 113], [], [], []],
        melds=[[], [], [], []],
        discards=[[], [], [], []],
        dora_indicators=[1],
        scores=[25_000] * 4,
        riichi_declared=[False] * 4,
        legal_actions=[],
        events=[],
        honba=0,
        riichi_sticks=0,
        round_wind=0,
        oya=0,
        kyoku_index=0,
        waits=[],
        is_tenpai=False,
        riichi_sutehais=[None] * 4,
        last_tedashis=[None] * 4,
        last_discard=None,
        furiten=fur,
    )
    assert obs.furiten == fur
    assert list(obs.encode_furiten_state()) == [0, 1, 0]
    assert len(obs.encode_visible_tile_counts()) == 241

    # v2_encode_actor_obs automatically reads obs.furiten without manual flags!
    encoded = native.v2_encode_actor_obs(obs)
    global_feats = np.frombuffer(encoded["global_feats"], dtype=np.float32)
    assert global_feats[19] == 1.0  # non-river furiten automatically propagated!


# =========================================================================
# 9. Contract v2.5: Live Events + HanchanTracker + Belief Targets + Stage-4 RL
# =========================================================================
def test_v2_encode_env_step_emits_real_events_hanchan_tracker_and_belief_targets() -> None:
    env = riichienv.RiichiEnv(game_mode="4p-red-half", seed=202611)
    obs_map = env.reset()

    saw_completed_round = False
    for _ in range(350):
        if env.done():
            break
        actions = {}
        for pid, obs in obs_map.items():
            # Actor-only + Belief mode (Stage-3 BC pipeline, zero Oracle compute)
            step_bc = native.v2_encode_env_step(env, pid, "tenhou", False, False, True)
            assert "oracle_tile" not in step_bc
            assert "opp_tenpai" in step_bc
            assert step_bc["event_len"] >= 1
            assert len(step_bc["events"]) == step_bc["event_len"] * EVENT_BYTES
            ev = np.frombuffer(step_bc["events"], dtype=np.uint8).reshape(-1, EVENT_BYTES)
            assert ev[0, 0] == 1  # Mandatory initial DORA at index 0

            # Standalone belief targets API matches inline include_belief=True bit-for-bit
            bt = native.v2_encode_belief_targets_env(env, pid)
            assert bt["opp_tenpai"] == step_bc["opp_tenpai"]
            assert bt["opp_waits"] == step_bc["opp_waits"]
            assert bt["exact_dealin_now"] == step_bc["exact_dealin_now"]
            assert bt["opp_hand_counts"] == step_bc["opp_hand_counts"]

            if step_bc["rounds_completed"] >= 1:
                pf = np.frombuffer(step_bc["player_feats"], dtype=np.float32).reshape(4, PLAYER_FEATS_DIM)
                # p[20] = rounds_completed / 12.0 MUST be non-zero after round 1!
                assert pf[0, 20] > 0.0
                saw_completed_round = True

            legal = obs.legal_actions()
            chosen = next(
                (a for a in legal if a.action_type in (riichienv.ActionType.TSUMO, riichienv.ActionType.RON)),
                None,
            )
            if chosen is None:
                chosen = next((a for a in legal if a.action_type == riichienv.ActionType.PASS), None)
            if chosen is None:
                chosen = next((a for a in legal if a.action_type == riichienv.ActionType.DISCARD), legal[0])
            actions[pid] = chosen
        obs_map = env.step(actions)
        if saw_completed_round:
            break

    assert saw_completed_round, "expected >= 1 completed round with live HanchanTracker propagation"


def test_belief_auxiliary_loss_masks_non_tenpai_waits_and_backprops_cleanly() -> None:
    actor = KARV2Actor(V2ModelConfig.v2_base())
    batch = _dummy_batch(batch_size=3, seq_len=12)
    out = actor(
        tile_feats=batch["tile_feats"],
        events=batch["events"],
        event_len=batch["event_len"],
        global_feats=batch["global_feats"],
        player_feats=batch["player_feats"],
        cand_ids=batch["cand_ids"],
        cand_mask=batch["cand_mask"],
        cand_mode=batch["cand_mode"],
        cand_dyn=batch["cand_dyn"],
        return_belief=True,
    )

    # Case 1: Zero tenpai opponents -> wait_loss == 0.0, finite total loss, clean backward
    loss_no_tenpai = belief_auxiliary_loss(
        belief_out=out,
        opp_tenpai=torch.zeros(3, 3),
        opp_waits=torch.ones(3, 3, TILE_DIM),  # deliberately non-zero; MUST be masked out!
        exact_dealin_now=torch.zeros(3, TILE_DIM),
        opp_hand_counts=torch.zeros(3, 3, TILE_DIM, dtype=torch.long),
    )
    assert loss_no_tenpai["tenpai_active_count"].item() == 0.0
    assert loss_no_tenpai["wait_loss"].item() == 0.0
    assert torch.isfinite(loss_no_tenpai["loss"])

    # Case 2: Active tenpai opponent -> wait_loss > 0 and gradients flow through Actor backbone
    opp_tenpai = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    opp_waits = torch.zeros(3, 3, TILE_DIM)
    opp_waits[0, 0, 12] = 1.0
    opp_waits[1, 1, 4] = 1.0
    dealin = torch.zeros(3, TILE_DIM)
    dealin[0, 12] = 1.0
    hand_counts = torch.randint(0, 5, (3, 3, TILE_DIM))

    actor.zero_grad(set_to_none=True)
    loss_active = belief_auxiliary_loss(
        belief_out=out,
        opp_tenpai=opp_tenpai,
        opp_waits=opp_waits,
        exact_dealin_now=dealin,
        opp_hand_counts=hand_counts,
    )
    assert loss_active["tenpai_active_count"].item() == 2.0
    assert loss_active["wait_loss"].item() > 0.0
    loss_active["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.spatial.parameters())


def test_pure_state_grp_training_and_calibration_report() -> None:
    torch.manual_seed(7)
    model = PureStateGRP(hidden_dim=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Synthetic separable score-to-rank dataset (seat 0 score strongly drives rank)
    n = 256
    feats = torch.zeros(n, 16)
    ranks = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        cls = i % 4
        ranks[i] = cls
        # Score advantage monotone with placement (0=1st highest, 3=4th lowest)
        feats[i, 0] = (1.5 - cls) * 0.6
        feats[i, 12:16] = torch.tensor([0.90, 0.45, 0.0, -1.35])

    init_probs = model.predict_probs(feats)
    rep_before = evaluate_grp_calibration(init_probs, ranks, n_bins=10)
    assert not rep_before.passes_gate(max_ece=0.03, max_classwise_ece=0.03, max_brier=0.68)

    for _ in range(150):
        stats = train_pure_state_grp_step(model, opt, feats, ranks)
        assert stats["loss"] > 0.0

    final_probs = model.predict_probs(feats)
    rep_after = evaluate_grp_calibration(final_probs, ranks, n_bins=10)
    assert rep_after.nll < rep_before.nll
    assert rep_after.brier_score < rep_before.brier_score
    assert rep_after.passes_gate(max_ece=0.03, max_classwise_ece=0.03, max_brier=0.05)


def test_critic_warmup_gate_blocks_actor_until_ev_streak() -> None:
    gate = CriticWarmupGate(ev_threshold=0.20, required_consecutive=2, min_critic_steps=10)
    assert not gate.actor_updates_allowed
    with pytest.raises(RuntimeError, match="CriticWarmupGate CLOSED"):
        gate.assert_actor_allowed()

    y = torch.linspace(-1.0, 1.0, 64)
    # High EV but before min_critic_steps=10 -> stays closed
    ev0 = gate.record_eval(y, y, step=5)
    assert abs(ev0 - 1.0) < 1e-6
    assert not gate.actor_updates_allowed

    # Low EV at step 10 -> resets streak
    ev1 = gate.record_eval(torch.zeros_like(y), y, step=10)
    assert abs(ev1 - 0.0) < 1e-6
    assert gate.consecutive_passes == 0

    # First pass at step 20 (streak=1 < 2 -> still blocked)
    ev2 = gate.record_eval(0.9 * y, y, step=20)
    assert ev2 >= 0.20
    assert not gate.actor_updates_allowed

    # Second consecutive pass at step 30 -> UNLOCKED!
    ev3 = gate.record_eval(0.95 * y, y, step=30)
    assert ev3 >= 0.20
    assert gate.actor_updates_allowed
    gate.assert_actor_allowed()  # Must not raise

    # Round-trip checkpoint serialization
    restored = CriticWarmupGate(ev_threshold=0.20, required_consecutive=2, min_critic_steps=10)
    restored.load_state_dict(gate.state_dict())
    assert restored.actor_updates_allowed
    assert compute_explained_variance(y, y) == pytest.approx(1.0)


def test_pfsp_opponent_pool_enforces_bc_anchor_floor_and_hardness_priority() -> None:
    with pytest.raises(ValueError, match="bc_anchor_prob"):
        PFSPOpponentPool(bc_anchor_prob=0.0)

    pool = PFSPOpponentPool(
        bc_anchor_id="bc_stage3",
        latest_self_prob=0.25,
        bc_anchor_prob=0.20,
        max_snapshots=3,
        pfsp_power=2.0,
    )
    # Before any snapshots, BC anchor gets >= 0.20 (actually 0.75)
    d0 = pool.sampling_probabilities()
    assert d0["bc_stage3"] == pytest.approx(0.75)
    assert d0[PFSPOpponentPool.LATEST_SELF_ID] == pytest.approx(0.25)

    # Add 5 snapshots -> FIFO eviction caps at 3, NEVER evicts bc_anchor
    for v in range(1, 6):
        pool.add_snapshot(f"rl_v{v}", policy_version=v)
    assert [s.opponent_id for s in pool.snapshots] == ["rl_v3", "rl_v4", "rl_v5"]

    # Record easy opponent (rl_v3: learner scores 0.90) vs hard opponent (rl_v5: learner scores 0.15)
    for _ in range(15):
        pool.record_match_result("rl_v3", 0.90)
        pool.record_match_result("rl_v5", 0.15)

    dist = pool.sampling_probabilities()
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["bc_stage3"] == pytest.approx(0.20)
    assert dist[PFSPOpponentPool.LATEST_SELF_ID] == pytest.approx(0.25)
    assert dist["rl_v5"] > dist["rl_v3"] * 5.0  # hard opponent strongly prioritized!

    rng = random.Random(1234)
    opps = pool.sample_table_opponents(rng=rng)
    assert len(opps) == 3
    assert all(o in dist for o in opps)


def test_frozen_gae_rollout_batch_immutability_and_version_binding() -> None:
    tag = RolloutVersionTag(
        policy_version=7,
        critic_version=3,
        grp_sha256="grp_sha_v1",
        ruleset_sha256="rule_sha_v1",
    )
    traj = HanchanTrajectoryInput(
        steps=[
            RoundBoundary(round_index=0, phi_k=0.05, v_raw=0.10),
            RoundBoundary(round_index=0, phi_k=0.05, v_raw=0.20),
            RoundBoundary(round_index=1, phi_k=0.30, v_raw=0.50),
        ],
        round_start_potentials=[0.05, 0.30],
        terminated=True,
        terminal_reward=0.90,
    )
    batch = FrozenGAERolloutBatch.from_hanchan_trajectories([traj], version_tag=tag)
    assert batch.num_steps == 3
    assert not batch.advantages.requires_grad
    assert not batch.shaped_returns.requires_grad

    # Valid across multiple PPO epochs while policy/critic update within max_staleness
    for _epoch in range(4):
        batch.assert_ready_for_ppo_epoch(
            current_policy_version=8,
            expected_grp_sha256="grp_sha_v1",
            expected_ruleset_sha256="rule_sha_v1",
            max_staleness=2,
        )

    # Reject GRP mismatch
    with pytest.raises(ValueError, match="GRP mismatch"):
        batch.assert_ready_for_ppo_epoch(
            current_policy_version=7,
            expected_grp_sha256="grp_sha_WRONG",
            expected_ruleset_sha256="rule_sha_v1",
        )

    # Cryptographic guard detects accidental in-place mutation of frozen GAE targets!
    batch.shaped_returns[0] += 0.123
    with pytest.raises(RuntimeError, match="mutated across PPO epochs"):
        batch.assert_ready_for_ppo_epoch(
            current_policy_version=7,
            expected_grp_sha256="grp_sha_v1",
            expected_ruleset_sha256="rule_sha_v1",
        )

    # Cryptographic guard MUST also cover step_rewards!
    batch2 = FrozenGAERolloutBatch.from_hanchan_trajectories([traj], version_tag=tag)
    batch2.step_rewards[-1] -= 0.05
    with pytest.raises(RuntimeError, match="mutated across PPO epochs"):
        batch2.assert_ready_for_ppo_epoch(
            current_policy_version=7,
            expected_grp_sha256="grp_sha_v1",
            expected_ruleset_sha256="rule_sha_v1",
        )


def test_ppo_multi_epoch_critic_updates_keep_frozen_gae_targets_bit_identical() -> None:
    torch.manual_seed(42)
    critic = OracleCriticV2(CriticModelConfig())
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
    dummy = _dummy_batch(batch_size=3, seq_len=8)
    grp_probs = torch.full((3, 4), 0.25)
    payoffs = torch.tensor([0.90, 0.45, 0.0, -1.35])
    final_ranks = torch.tensor([0, 1, 3], dtype=torch.long)

    # Rollout-time frozen Critic predictions used ONCE to construct FrozenGAERolloutBatch
    with torch.no_grad():
        rollout_out = critic(
            tile_feats=dummy["tile_feats"],
            oracle_tile=dummy["oracle_tile"],
            events=dummy["events"],
            event_len=dummy["event_len"],
            global_feats=dummy["global_feats"],
            player_feats=dummy["player_feats"],
            oracle_player=dummy["oracle_player"],
            grp_prior_probs=grp_probs,
            rank_payoffs=payoffs,
        )
    v_raw_rollout = rollout_out["v_raw"].tolist()
    phis = [0.05, 0.20]
    traj = HanchanTrajectoryInput(
        steps=[
            RoundBoundary(round_index=0, phi_k=phis[0], v_raw=v_raw_rollout[0]),
            RoundBoundary(round_index=0, phi_k=phis[0], v_raw=v_raw_rollout[1]),
            RoundBoundary(round_index=1, phi_k=phis[1], v_raw=v_raw_rollout[2]),
        ],
        round_start_potentials=phis,
        terminated=True,
        terminal_reward=0.90,
    )
    tag = RolloutVersionTag(
        policy_version=12,
        critic_version=4,
        grp_sha256="grp_v2",
        ruleset_sha256="rule_v2",
    )
    frozen_batch = FrozenGAERolloutBatch.from_hanchan_trajectories([traj], version_tag=tag)
    orig_sha = frozen_batch.target_sha256
    orig_adv = frozen_batch.advantages.clone()
    orig_ret = frozen_batch.shaped_returns.clone()
    orig_rew = frozen_batch.step_rewards.clone()

    # Run 4 real PPO epochs updating Critic parameters -> frozen targets MUST NEVER move!
    w_before = next(critic.parameters()).detach().clone()
    for _epoch in range(4):
        frozen_batch.assert_ready_for_ppo_epoch(
            current_policy_version=12,
            expected_grp_sha256="grp_v2",
            expected_ruleset_sha256="rule_v2",
            max_staleness=2,
        )
        optimizer.zero_grad(set_to_none=True)
        out = critic(
            tile_feats=dummy["tile_feats"],
            oracle_tile=dummy["oracle_tile"],
            events=dummy["events"],
            event_len=dummy["event_len"],
            global_feats=dummy["global_feats"],
            player_feats=dummy["player_feats"],
            oracle_player=dummy["oracle_player"],
            grp_prior_probs=grp_probs,
            rank_payoffs=payoffs,
        )
        loss_dict = oracle_critic_loss(
            rank_logits=out["rank_logits"],
            v_shaped=out["v_shaped"],
            final_rank_idx=final_ranks,
            shaped_target=frozen_batch.shaped_returns,
            td_mode="gae_mse",
        )
        loss_dict["loss"].backward()
        optimizer.step()

        assert frozen_batch.target_sha256 == orig_sha
        assert torch.equal(frozen_batch.advantages, orig_adv)
        assert torch.equal(frozen_batch.shaped_returns, orig_ret)
        assert torch.equal(frozen_batch.step_rewards, orig_rew)

    assert not torch.equal(w_before, next(critic.parameters()).detach())


def test_hanchan_gae_cross_round_truncation_skipped_round_truncation_and_chunk_telescoping() -> None:
    # Potentials across 6 started rounds: Phi_0..Phi_5
    phis = [0.05, 0.25, -0.10, 0.40, 0.15, 0.60]

    # 1. Intra-round normal truncation: reward == 0.0, exact bootstrap (next_value != 0.0)
    steps_r0 = [
        RoundBoundary(round_index=0, phi_k=phis[0], v_raw=0.08),
        RoundBoundary(round_index=0, phi_k=phis[0], v_raw=0.14),
    ]
    rew_intra, adv_intra, ret_intra = compute_hanchan_gae(
        steps_r0,
        phis[:1],
        terminated=False,
        truncation=RolloutTruncation(kind="intra_round", bootstrap_v_shaped=0.12),
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert rew_intra == [0.0, 0.0]
    assert adv_intra[-1] == pytest.approx(0.12 - steps_r0[-1].v_shaped)
    assert ret_intra[-1] == pytest.approx(0.12)

    # 2. Exact round-end truncation: round 0 settled, round 1 started (no step in round 1 yet!)
    v_raw_trunc_r1 = 0.31
    v_shaped_trunc_r1 = v_raw_trunc_r1 - phis[1]
    rew_end, adv_end, ret_end = compute_hanchan_gae(
        steps_r0,
        phis[:2],
        terminated=False,
        truncation=RolloutTruncation(
            kind="cross_round",
            bootstrap_v_shaped=v_shaped_trunc_r1,
            next_round_index=1,
            next_round_phi=phis[1],
        ),
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert rew_end[0] == pytest.approx(0.0)
    assert rew_end[-1] == pytest.approx(phis[1] - phis[0])
    # Exact Bellman cancellation: delta[-1] == v_raw_trunc_r1 - v_raw[-1]
    assert adv_end[-1] == pytest.approx(v_raw_trunc_r1 - steps_r0[-1].v_raw)
    assert ret_end[-1] == pytest.approx((phis[1] - phis[0]) + v_shaped_trunc_r1)

    # 3. Cross-round continuation with normal next decision: verify bootstrap to next valid step
    #    and zero duplicate potential reward across chunks!
    step_r1_first = RoundBoundary(round_index=1, phi_k=phis[1], v_raw=v_raw_trunc_r1)
    step_r1_second = RoundBoundary(round_index=1, phi_k=phis[1], v_raw=0.42)
    r_term_2round = 0.90
    rew_full, adv_full, _ = compute_hanchan_gae(
        [*steps_r0, step_r1_first, step_r1_second],
        phis[:2],
        terminated=True,
        terminal_reward=r_term_2round,
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert rew_full == pytest.approx([0.0, phis[1] - phis[0], 0.0, r_term_2round - phis[1]])
    assert sum(rew_full) == pytest.approx(r_term_2round - phis[0])
    # Split into Chunk 1 (cross-round truncated bootstrapping step_r1_first.v_shaped) + Chunk 2 (from step_r1_first)
    rew_c1, adv_c1, _ = compute_hanchan_gae(
        steps_r0,
        phis[:2],
        terminated=False,
        truncation=RolloutTruncation(
            kind="cross_round",
            bootstrap_v_shaped=step_r1_first.v_shaped,
            next_round_index=1,
            next_round_phi=phis[1],
        ),
        gamma=1.0,
        gae_lambda=0.95,
    )
    rew_c2, _, _ = compute_hanchan_gae(
        [step_r1_first, step_r1_second],
        phis[:2],
        terminated=True,
        terminal_reward=r_term_2round,
        gamma=1.0,
        gae_lambda=0.95,
    )
    # Potential jump Phi_1 - Phi_0 is issued EXACTLY ONCE across Chunk 1 + Chunk 2!
    assert sum(rew_c1) + sum(rew_c2) == pytest.approx(r_term_2round - phis[0])
    assert rew_c1[-1] == pytest.approx(phis[1] - phis[0])
    assert rew_c2[0] == pytest.approx(0.0)
    assert adv_c1[-1] == pytest.approx(step_r1_first.v_raw - steps_r0[-1].v_raw)
    assert adv_full[1] == pytest.approx(
        (phis[1] - phis[0]) + step_r1_first.v_shaped - steps_r0[-1].v_shaped + 0.95 * adv_full[2]
    )

    # 4. Single skipped-round truncation: round 0 -> skip round 1 -> truncated at round 2 start
    v_raw_trunc_r2 = -0.05
    v_shaped_trunc_r2 = v_raw_trunc_r2 - phis[2]
    rew_skip1, adv_skip1, _ = compute_hanchan_gae(
        steps_r0,
        phis[:3],
        terminated=False,
        truncation=RolloutTruncation(
            kind="cross_round",
            bootstrap_v_shaped=v_shaped_trunc_r2,
            next_round_index=2,
            next_round_phi=phis[2],
        ),
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert rew_skip1[-1] == pytest.approx(phis[2] - phis[0])
    assert adv_skip1[-1] == pytest.approx(v_raw_trunc_r2 - steps_r0[-1].v_raw)

    # 5. Multiple consecutive skipped rounds (skip rounds 1, 2, 3 -> round 4):
    #    (a) cross-round truncation merges Phi_4 - Phi_0 without loss or duplication
    v_raw_trunc_r4 = 0.22
    v_shaped_trunc_r4 = v_raw_trunc_r4 - phis[4]
    rew_skip_multi_trunc, adv_skip_multi_trunc, _ = compute_hanchan_gae(
        steps_r0,
        phis[:5],
        terminated=False,
        truncation=RolloutTruncation(
            kind="cross_round",
            bootstrap_v_shaped=v_shaped_trunc_r4,
            next_round_index=4,
            next_round_phi=phis[4],
        ),
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert rew_skip_multi_trunc == pytest.approx([0.0, phis[4] - phis[0]])
    assert adv_skip_multi_trunc[-1] == pytest.approx(v_raw_trunc_r4 - steps_r0[-1].v_raw)

    #    (b) full Hanchan skipping rounds 1, 2, 3 inside trajectory + skipping round 5 before termination!
    steps_multi_skip_full = [
        *steps_r0,
        RoundBoundary(round_index=4, phi_k=phis[4], v_raw=0.22),
    ]
    r_terminal = 0.90
    rew_multi_full, adv_multi_full, ret_multi_full = compute_hanchan_gae(
        steps_multi_skip_full,
        phis,
        terminated=True,
        terminal_reward=r_terminal,
        gamma=1.0,
        gae_lambda=0.95,
    )
    # 6. True Hanchan termination: last step uses R_terminal - Phi_last with next_value == 0.0
    assert rew_multi_full == pytest.approx([0.0, phis[4] - phis[0], r_terminal - phis[4]])
    assert sum(rew_multi_full) == pytest.approx(r_terminal - phis[0])
    assert adv_multi_full[-1] == pytest.approx((r_terminal - phis[4]) - steps_multi_skip_full[-1].v_shaped)
    assert ret_multi_full[-1] == pytest.approx(r_terminal - phis[4])

    #    Exact 2-chunk telescoping equivalence across multi-skip boundary
    rew_chunk2_r4, _, _ = compute_hanchan_gae(
        [RoundBoundary(round_index=4, phi_k=phis[4], v_raw=0.22)],
        phis,
        terminated=True,
        terminal_reward=r_terminal,
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert sum(rew_skip_multi_trunc) + sum(rew_chunk2_r4) == pytest.approx(r_terminal - phis[0])


def test_hanchan_gae_rejects_contradictory_and_missing_cross_round_boundaries() -> None:
    phis = [0.05, 0.25, -0.10]
    steps_r0 = [
        RoundBoundary(round_index=0, phi_k=phis[0], v_raw=0.08),
        RoundBoundary(round_index=0, phi_k=phis[0], v_raw=0.14),
    ]
    v_shaped_trunc_r1 = 0.31 - phis[1]

    # 7. Contradictory terminal + truncation inputs MUST be explicitly rejected!
    with pytest.raises(ValueError, match="must not be supplied when terminated=True"):
        compute_hanchan_gae(
            steps_r0,
            phis[:2],
            terminated=True,
            terminal_reward=0.90,
            truncation=RolloutTruncation(kind="intra_round", bootstrap_v_shaped=0.1),
        )
    with pytest.raises(ValueError, match="must not be supplied when terminated=True"):
        compute_hanchan_gae(
            steps_r0,
            phis[:1],
            terminated=True,
            terminal_reward=0.90,
            truncated_bootstrap_v_shaped=0.1,
        )
    with pytest.raises(ValueError, match="terminal_reward must not be supplied when terminated=False"):
        compute_hanchan_gae(
            steps_r0,
            phis[:1],
            terminated=False,
            terminal_reward=0.90,
            truncation=RolloutTruncation(kind="intra_round", bootstrap_v_shaped=0.1),
        )

    # 8. Missing cross-round potential MUST raise ValueError (even if round_start_potentials has Phi_1)!
    for pot_slice in (phis[:1], phis[:2]):
        with pytest.raises(ValueError, match="next_round_phi"):
            compute_hanchan_gae(
                steps_r0,
                pot_slice,
                terminated=False,
                truncation=RolloutTruncation(
                    kind="cross_round",
                    bootstrap_v_shaped=0.1,
                    next_round_index=1,
                    next_round_phi=None,
                ),
            )
    with pytest.raises(ValueError, match="next_round_index > last_k"):
        compute_hanchan_gae(
            steps_r0,
            phis[:2],
            terminated=False,
            truncation=RolloutTruncation(
                kind="cross_round",
                bootstrap_v_shaped=0.1,
                next_round_index=None,
                next_round_phi=phis[1],
            ),
        )
    # Fail-closed guard when round_start_potentials already has next round but caller omits RolloutTruncation
    with pytest.raises(ValueError, match="RolloutTruncation"):
        compute_hanchan_gae(
            steps_r0,
            phis[:2],
            terminated=False,
            truncated_bootstrap_v_shaped=0.1,
        )

    # 9. FrozenGAERolloutBatch wires cross-round RolloutTruncation seamlessly
    tag = RolloutVersionTag(
        policy_version=1,
        critic_version=1,
        grp_sha256="grp_sha",
        ruleset_sha256="rule_sha",
    )
    batch = FrozenGAERolloutBatch.from_hanchan_trajectories(
        [
            HanchanTrajectoryInput(
                steps=steps_r0,
                round_start_potentials=phis[:2],
                terminated=False,
                truncation=RolloutTruncation(
                    kind="cross_round",
                    bootstrap_v_shaped=v_shaped_trunc_r1,
                    next_round_index=1,
                    next_round_phi=phis[1],
                    bootstrap_round_index=1,
                    bootstrap_v_raw=0.31,
                ),
            )
        ],
        version_tag=tag,
    )
    assert float(batch.step_rewards[-1].item()) == pytest.approx(phis[1] - phis[0])


def test_hanchan_gae_pre_vs_post_cross_round_rollout_and_early_vs_full_hanchan() -> None:
    # Full 8-round Hanchan potentials (East 1 .. South 4)
    phis_8 = [0.00, 0.12, -0.08, 0.20, 0.35, 0.18, 0.50, 0.72]

    # Scenario 6: Rollout ends BEFORE round boundary (still inside round 0);
    # Chunk 2 finishes round 0 and crosses to round 1.
    step0 = RoundBoundary(round_index=0, phi_k=phis_8[0], v_raw=0.04)
    step1 = RoundBoundary(round_index=0, phi_k=phis_8[0], v_raw=0.09)
    step2 = RoundBoundary(round_index=1, phi_k=phis_8[1], v_raw=0.22)
    r_early_tobi = -1.35  # Scenario 8: Early Hanchan termination (Tobi at East 2, K=2)

    rew_pre_c1, adv_pre_c1, ret_pre_c1 = compute_hanchan_gae(
        [step0],
        phis_8[:1],
        terminated=False,
        truncation=RolloutTruncation(
            kind="intra_round",
            bootstrap_v_shaped=step1.v_shaped,
            next_round_index=0,
            next_round_phi=phis_8[0],
            bootstrap_round_index=0,
            bootstrap_v_raw=step1.v_raw,
        ),
        gamma=1.0,
        gae_lambda=0.95,
    )
    rew_pre_c2, adv_pre_c2, ret_pre_c2 = compute_hanchan_gae(
        [step1, step2],
        phis_8[:2],
        terminated=True,
        terminal_reward=r_early_tobi,
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert rew_pre_c1 == pytest.approx([0.0])
    assert adv_pre_c1[0] == pytest.approx(step1.v_shaped - step0.v_shaped)
    assert ret_pre_c1[0] == pytest.approx(step1.v_shaped)
    assert rew_pre_c2 == pytest.approx([phis_8[1] - phis_8[0], r_early_tobi - phis_8[1]])
    assert adv_pre_c2[-1] == pytest.approx((r_early_tobi - phis_8[1]) - step2.v_shaped)
    assert ret_pre_c2[-1] == pytest.approx(r_early_tobi - phis_8[1])
    assert sum(rew_pre_c1) + sum(rew_pre_c2) == pytest.approx(r_early_tobi - phis_8[0])

    # Scenario 7: Rollout ends AFTER round boundary (round 0 settled, round 1 started);
    # potential jump belongs in Chunk 1 ONLY.
    rew_post_c1, adv_post_c1, ret_post_c1 = compute_hanchan_gae(
        [step0, step1],
        phis_8[:2],
        terminated=False,
        truncation=RolloutTruncation(
            kind="cross_round",
            bootstrap_v_shaped=step2.v_shaped,
            next_round_index=1,
            next_round_phi=phis_8[1],
            bootstrap_round_index=1,
            bootstrap_v_raw=step2.v_raw,
        ),
        gamma=1.0,
        gae_lambda=0.95,
    )
    rew_post_c2, adv_post_c2, ret_post_c2 = compute_hanchan_gae(
        [step2],
        phis_8[:2],
        terminated=True,
        terminal_reward=r_early_tobi,
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert rew_post_c1 == pytest.approx([0.0, phis_8[1] - phis_8[0]])
    assert adv_post_c1[-1] == pytest.approx(step2.v_raw - step1.v_raw)
    assert ret_post_c1[-1] == pytest.approx((phis_8[1] - phis_8[0]) + step2.v_shaped)
    assert rew_post_c2 == pytest.approx([r_early_tobi - phis_8[1]])
    assert adv_post_c2[-1] == pytest.approx((r_early_tobi - phis_8[1]) - step2.v_shaped)
    assert ret_post_c2[-1] == pytest.approx(r_early_tobi - phis_8[1])
    assert sum(rew_post_c1) + sum(rew_post_c2) == pytest.approx(r_early_tobi - phis_8[0])

    # Scenario 9: Normal full 8-round Hanchan termination (East 1 .. South 4, K=8)
    steps_8 = [RoundBoundary(round_index=k, phi_k=phis_8[k], v_raw=phis_8[k] + 0.03 * (k - 3)) for k in range(8)]
    r_full = 0.90
    rew_8, adv_8, ret_8 = compute_hanchan_gae(
        steps_8,
        phis_8,
        terminated=True,
        terminal_reward=r_full,
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert sum(rew_8) == pytest.approx(r_full - phis_8[0])
    assert rew_8[-1] == pytest.approx(r_full - phis_8[7])
    assert adv_8[-1] == pytest.approx((r_full - phis_8[7]) - steps_8[-1].v_shaped)
    assert ret_8[-1] == pytest.approx(r_full - phis_8[7])
    # Verify exact backward GAE recursion at every step t
    for t in range(len(steps_8) - 1):
        delta_t = rew_8[t] + 1.0 * steps_8[t + 1].v_shaped - steps_8[t].v_shaped
        assert adv_8[t] == pytest.approx(delta_t + 0.95 * adv_8[t + 1])
        assert ret_8[t] == pytest.approx(adv_8[t] + steps_8[t].v_shaped)


def test_hanchan_gae_rejects_non_finite_regressive_and_mismatched_bootstrap() -> None:
    phis = [0.05, 0.25]
    steps = [RoundBoundary(round_index=0, phi_k=0.05, v_raw=0.10)]

    # Non-finite floats in bootstrap, next_round_phi, terminal_reward, step, potentials, gamma, gae_lambda
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            compute_hanchan_gae(
                steps,
                phis[:1],
                terminated=False,
                truncation=RolloutTruncation(kind="intra_round", bootstrap_v_shaped=bad),
            )
        with pytest.raises(ValueError, match="finite"):
            compute_hanchan_gae(
                steps,
                phis[:2],
                terminated=False,
                truncation=RolloutTruncation(
                    kind="cross_round",
                    bootstrap_v_shaped=0.1,
                    next_round_index=1,
                    next_round_phi=bad,
                ),
            )
        with pytest.raises(ValueError, match="finite"):
            compute_hanchan_gae(steps, phis[:1], terminated=True, terminal_reward=bad)
        with pytest.raises(ValueError, match="finite"):
            compute_hanchan_gae(
                [RoundBoundary(round_index=0, phi_k=0.05, v_raw=bad)],
                phis[:1],
                terminated=True,
                terminal_reward=0.9,
            )
        with pytest.raises(ValueError, match="finite"):
            compute_hanchan_gae(steps, [bad], terminated=True, terminal_reward=0.9)
        with pytest.raises(ValueError, match="finite"):
            compute_hanchan_gae(steps, phis[:1], terminated=True, terminal_reward=0.9, gamma=bad)
        with pytest.raises(ValueError, match="finite"):
            compute_hanchan_gae(steps, phis[:1], terminated=True, terminal_reward=0.9, gae_lambda=bad)

    # Illegal / regressive round indices
    with pytest.raises(ValueError, match="out of bounds"):
        compute_hanchan_gae(
            [RoundBoundary(round_index=-1, phi_k=0.05, v_raw=0.10)],
            phis,
            terminated=True,
            terminal_reward=0.9,
        )
    with pytest.raises(ValueError, match="Non-monotone"):
        compute_hanchan_gae(
            [
                RoundBoundary(round_index=1, phi_k=0.25, v_raw=0.30),
                RoundBoundary(round_index=0, phi_k=0.05, v_raw=0.10),
            ],
            phis,
            terminated=True,
            terminal_reward=0.9,
        )

    # Bootstrap round mismatch & stale-Phi bootstrap_v_shaped mismatch (Fail-Closed Rule #8)
    with pytest.raises(ValueError, match="bootstrap_round_index"):
        compute_hanchan_gae(
            steps,
            phis,
            terminated=False,
            truncation=RolloutTruncation(
                kind="cross_round",
                bootstrap_v_shaped=0.31 - phis[1],
                next_round_index=1,
                next_round_phi=phis[1],
                bootstrap_round_index=0,  # WRONG round!
            ),
        )
    with pytest.raises(ValueError, match="bootstrap_v_shaped"):
        compute_hanchan_gae(
            steps,
            phis,
            terminated=False,
            truncation=RolloutTruncation(
                kind="cross_round",
                bootstrap_v_shaped=0.31 - phis[0],  # BUG: used stale Phi_0 instead of Phi_1!
                next_round_index=1,
                next_round_phi=phis[1],
                bootstrap_round_index=1,
                bootstrap_v_raw=0.31,
            ),
        )


def test_rollout_boundary_serialization_checkpoint_resume_and_ppo_consistency() -> None:
    phis = [0.05, 0.25, -0.10]
    traj_intra = HanchanTrajectoryInput(
        steps=[RoundBoundary(round_index=0, phi_k=phis[0], v_raw=0.08)],
        round_start_potentials=phis[:1],
        terminated=False,
        truncation=RolloutTruncation(
            kind="intra_round",
            bootstrap_v_shaped=0.07,
            next_round_index=0,
            next_round_phi=phis[0],
            bootstrap_round_index=0,
            bootstrap_v_raw=0.12,
        ),
    )
    traj_cross = HanchanTrajectoryInput(
        steps=[
            RoundBoundary(round_index=0, phi_k=phis[0], v_raw=0.09),
            RoundBoundary(round_index=0, phi_k=phis[0], v_raw=0.14),
        ],
        round_start_potentials=phis,
        terminated=False,
        truncation=RolloutTruncation(
            kind="cross_round",
            bootstrap_v_shaped=-0.05 - phis[2],
            next_round_index=2,
            next_round_phi=phis[2],
            bootstrap_round_index=2,
            bootstrap_v_raw=-0.05,
        ),
    )
    traj_term = HanchanTrajectoryInput(
        steps=[RoundBoundary(round_index=1, phi_k=phis[1], v_raw=0.40)],
        round_start_potentials=phis[:2],
        terminated=True,
        terminal_reward=0.90,
    )

    # 1. JSON round-trip of HanchanTrajectoryInput & RolloutTruncation preserves every field verbatim
    trajs = [traj_intra, traj_cross, traj_term]
    encoded_json = json.dumps([t.to_dict() for t in trajs])
    decoded_trajs = [HanchanTrajectoryInput.from_dict(item) for item in json.loads(encoded_json)]
    assert decoded_trajs == trajs

    tag = RolloutVersionTag(
        policy_version=9,
        critic_version=4,
        grp_sha256="grp_sha_resume",
        ruleset_sha256="rule_sha_resume",
    )
    batch = FrozenGAERolloutBatch.from_hanchan_trajectories(decoded_trajs, version_tag=tag)

    # 2. PyTorch checkpoint save / resume preserves tensors, boundary metadata, version tag, and SHA-256
    buf = io.BytesIO()
    torch.save(batch.state_dict(), buf)
    buf.seek(0)
    restored = FrozenGAERolloutBatch.from_state_dict(torch.load(buf, weights_only=False))

    assert restored.target_sha256 == batch.target_sha256
    assert restored.version_tag == batch.version_tag
    assert restored.trajectory_boundaries == batch.trajectory_boundaries
    assert [b["boundary_type"] for b in restored.trajectory_boundaries] == ["intra_round", "cross_round", "terminal"]
    assert restored.trajectory_boundaries[1]["truncation"]["next_round_index"] == 2
    assert restored.trajectory_boundaries[1]["truncation"]["next_round_phi"] == pytest.approx(phis[2])
    assert torch.equal(restored.step_rewards, batch.step_rewards)
    assert torch.equal(restored.advantages, batch.advantages)
    assert torch.equal(restored.shaped_returns, batch.shaped_returns)
    assert torch.equal(restored.rollout_v_shaped, batch.rollout_v_shaped)

    # Verify exact rewards across all 3 boundary types after resume
    assert restored.step_rewards.tolist() == pytest.approx([0.0, 0.0, phis[2] - phis[0], 0.90 - phis[1]])

    # 3. Resumed batch passes multi-epoch PPO readiness guard (including critic_version check)
    for _ in range(3):
        restored.assert_ready_for_ppo_epoch(
            current_policy_version=10,
            expected_grp_sha256="grp_sha_resume",
            expected_ruleset_sha256="rule_sha_resume",
            expected_critic_version=4,
            max_staleness=2,
        )
    with pytest.raises(ValueError, match="critic_version"):
        restored.assert_ready_for_ppo_epoch(
            current_policy_version=10,
            expected_grp_sha256="grp_sha_resume",
            expected_ruleset_sha256="rule_sha_resume",
            expected_critic_version=5,
            max_staleness=2,
        )

    # 4. Tampering serialized boundary metadata is caught by SHA-256 on restore!
    tampered_state = batch.state_dict()
    tampered_state["trajectory_boundaries"][1]["truncation"]["kind"] = "intra_round"
    with pytest.raises((RuntimeError, ValueError)):
        FrozenGAERolloutBatch.from_state_dict(tampered_state)
    tampered_phi = batch.state_dict()
    tampered_phi["trajectory_boundaries"][1]["truncation"]["next_round_phi"] = 0.999
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        FrozenGAERolloutBatch.from_state_dict(tampered_phi)

    # 5. Missing required fields MUST reject restore (no silent defaults!)
    for missing_key in ("trajectory_boundaries", "target_sha256", "version_tag", "step_rewards"):
        bad_state = batch.state_dict()
        del bad_state[missing_key]
        with pytest.raises(ValueError, match="missing required field"):
            FrozenGAERolloutBatch.from_state_dict(bad_state)
    bad_traj_dict = traj_cross.to_dict()
    del bad_traj_dict["truncation"]["next_round_phi"]
    with pytest.raises(ValueError, match="next_round_phi"):
        HanchanTrajectoryInput.from_dict(bad_traj_dict)


def test_hanchan_gae_leading_zero_decisions_duplicate_potential_and_full_6boundary_identity() -> None:
    phis = [0.00, 0.14, -0.06, 0.22, 0.08, 0.35, 0.52, 0.68]

    # Scenario 8: Consecutive zero-decision rounds starting from East 1
    # (rounds 0 & 1 skipped, first decision in round 2).
    steps_lead_skip = [
        RoundBoundary(round_index=2, phi_k=phis[2], v_raw=0.01),
        RoundBoundary(round_index=2, phi_k=phis[2], v_raw=0.05),
        RoundBoundary(round_index=3, phi_k=phis[3], v_raw=0.28),
    ]
    r_term = 0.90
    rew_lead, adv_lead, ret_lead = compute_hanchan_gae(
        steps_lead_skip,
        phis[:4],
        terminated=True,
        terminal_reward=r_term,
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert rew_lead == pytest.approx([0.0, phis[3] - phis[2], r_term - phis[3]])
    # Origin potential shifts cleanly to first active round Phi_2: sum(r) == R_terminal - Phi_2!
    assert sum(rew_lead) == pytest.approx(r_term - phis[2])
    assert adv_lead[-1] == pytest.approx((r_term - phis[3]) - steps_lead_skip[-1].v_shaped)
    assert ret_lead[-1] == pytest.approx(r_term - phis[3])

    # Leading zero-decision rounds followed by cross-round truncation (round 2 -> round 3 start)
    rew_lead_trunc, adv_lead_trunc, ret_lead_trunc = compute_hanchan_gae(
        steps_lead_skip[:2],
        phis[:4],
        terminated=False,
        truncation=RolloutTruncation(
            kind="cross_round",
            bootstrap_v_shaped=steps_lead_skip[2].v_shaped,
            next_round_index=3,
            next_round_phi=phis[3],
            bootstrap_round_index=3,
            bootstrap_v_raw=steps_lead_skip[2].v_raw,
        ),
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert rew_lead_trunc == pytest.approx([0.0, phis[3] - phis[2]])
    assert adv_lead_trunc[-1] == pytest.approx(steps_lead_skip[2].v_raw - steps_lead_skip[1].v_raw)
    assert ret_lead_trunc[-1] == pytest.approx((phis[3] - phis[2]) + steps_lead_skip[2].v_shaped)

    # Scenario 16: Reject duplicate cross-round potential issuance when round already entered in steps!
    with pytest.raises(ValueError, match="duplicate or regressive potential"):
        compute_hanchan_gae(
            steps_lead_skip,  # already entered round 3!
            phis[:4],
            terminated=False,
            truncation=RolloutTruncation(
                kind="cross_round",
                bootstrap_v_shaped=0.05,
                next_round_index=3,  # tries to re-issue round 3!
                next_round_phi=phis[3],
            ),
        )

    # Scenario 20: Combined 6-boundary-type multi-chunk Hanchan full mathematical identity check:
    # Chunk A: leading zero-decision round 0 skipped; steps in round 1 -> intra-round truncation
    # Chunk B: resumes in round 1 -> normal cross-round to round 2 -> skips rounds 3,4 -> cross-round trunc into round 5
    # Chunk C: resumes in round 5 -> skips round 6 -> normal step in round 7 -> Hanchan termination
    s1a = RoundBoundary(round_index=1, phi_k=phis[1], v_raw=0.18)
    s1b = RoundBoundary(round_index=1, phi_k=phis[1], v_raw=0.21)
    s2a = RoundBoundary(round_index=2, phi_k=phis[2], v_raw=-0.02)
    s5a = RoundBoundary(round_index=5, phi_k=phis[5], v_raw=0.41)
    s7a = RoundBoundary(round_index=7, phi_k=phis[7], v_raw=0.75)

    rew_a, adv_a, ret_a = compute_hanchan_gae(
        [s1a],
        phis[:2],
        terminated=False,
        truncation=RolloutTruncation(
            kind="intra_round",
            bootstrap_v_shaped=s1b.v_shaped,
            next_round_index=1,
            next_round_phi=phis[1],
            bootstrap_round_index=1,
            bootstrap_v_raw=s1b.v_raw,
        ),
        gamma=1.0,
        gae_lambda=0.95,
    )
    rew_b, adv_b, ret_b = compute_hanchan_gae(
        [s1b, s2a],
        phis[:6],
        terminated=False,
        truncation=RolloutTruncation(
            kind="cross_round",
            bootstrap_v_shaped=s5a.v_shaped,
            next_round_index=5,
            next_round_phi=phis[5],
            bootstrap_round_index=5,
            bootstrap_v_raw=s5a.v_raw,
        ),
        gamma=1.0,
        gae_lambda=0.95,
    )
    rew_c, adv_c, ret_c = compute_hanchan_gae(
        [s5a, s7a],
        phis,
        terminated=True,
        terminal_reward=r_term,
        gamma=1.0,
        gae_lambda=0.95,
    )

    # Exact multi-chunk telescoping identity: sum(rew_A + rew_B + rew_C) == R_terminal - Phi_1
    assert sum(rew_a) + sum(rew_b) + sum(rew_c) == pytest.approx(r_term - phis[1])
    # Verify exact single-step Bellman & GAE equations across every chunk
    assert adv_a[0] == pytest.approx(s1b.v_shaped - s1a.v_shaped)
    assert ret_a[0] == pytest.approx(s1b.v_shaped)
    delta_b1 = (phis[5] - phis[2]) + s5a.v_shaped - s2a.v_shaped
    delta_b0 = (phis[2] - phis[1]) + s2a.v_shaped - s1b.v_shaped
    assert adv_b[1] == pytest.approx(delta_b1)
    assert adv_b[0] == pytest.approx(delta_b0 + 0.95 * delta_b1)
    assert ret_b[0] == pytest.approx(adv_b[0] + s1b.v_shaped)
    assert ret_b[1] == pytest.approx(adv_b[1] + s2a.v_shaped)
    delta_c1 = (r_term - phis[7]) - s7a.v_shaped
    delta_c0 = (phis[7] - phis[5]) + s7a.v_shaped - s5a.v_shaped
    assert adv_c[1] == pytest.approx(delta_c1)
    assert adv_c[0] == pytest.approx(delta_c0 + 0.95 * delta_c1)
    assert ret_c[0] == pytest.approx(adv_c[0] + s5a.v_shaped)
    assert ret_c[1] == pytest.approx(r_term - phis[7])
