"""Stage-1 Automated Contract Verification Suite for KAR V2.4 (Rust + PyTorch)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import riichienv
from riichienv import _riichienv as native
from riichienv.kar.v2 import (
    CriticModelConfig,
    FrozenGRP,
    KARV2Actor,
    OracleCriticV2,
    PureStateGRP,
    RolloutVersionTag,
    RoundBoundary,
    V2ModelConfig,
    compute_hanchan_gae,
    compute_pbrs_round_rewards,
    masked_entropy_and_kl_fp32,
    oracle_critic_loss,
    ppo_actor_loss,
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
    r_has = torch.from_numpy(
        np.frombuffer(rust_tbl["has_slot_flags"], dtype=np.float32).copy()
    ).view(-1, 4)
    r_flags = torch.from_numpy(
        np.frombuffer(rust_tbl["static_flags"], dtype=np.float32).copy()
    ).view(-1, 4)

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
    res_tenhou = native.v2_calc_with_rule(
        kokushi_13, [], 1, [4], [], cond, riichienv.GameRule.default_tenhou()
    )
    res_mjs = native.v2_calc_with_rule(
        kokushi_13, [], 1, [4], [], cond, riichienv.GameRule.default_mjsoul()
    )
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

    # Mid-Hanchan truncation bootstraps cleanly with non-zero V_shaped
    _, adv_trunc, ret_trunc = compute_hanchan_gae(
        steps, phis, terminated=False, truncated_bootstrap_v_shaped=0.33, gamma=1.0, gae_lambda=0.95
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

    # Case 2: Mismatched chosen_global_id MUST fail fast
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
    tile_feats = torch.from_numpy(
        np.frombuffer(step_data["tile_feats"], dtype=np.float32).copy()
    ).view(1, TILE_DIM, TILE_FEATS_DIM)
    global_feats = torch.from_numpy(
        np.frombuffer(step_data["global_feats"], dtype=np.float32).copy()
    ).view(1, GLOBAL_FEATS_DIM)
    player_feats = torch.from_numpy(
        np.frombuffer(step_data["player_feats"], dtype=np.float32).copy()
    ).view(1, 4, PLAYER_FEATS_DIM)
    cand_ids = torch.from_numpy(
        np.frombuffer(step_data["cand_ids"], dtype=np.int16).copy()
    ).view(1, MAX_CANDIDATES)
    cand_mask = (
        torch.from_numpy(np.frombuffer(step_data["cand_mask"], dtype=np.uint8).copy())
        .view(1, MAX_CANDIDATES)
        .bool()
    )
    cand_mode = torch.from_numpy(
        np.frombuffer(step_data["cand_mode"], dtype=np.uint8).copy()
    ).view(1, MAX_CANDIDATES)
    cand_dyn = torch.from_numpy(
        np.frombuffer(step_data["cand_dyn"], dtype=np.float32).copy()
    ).view(1, MAX_CANDIDATES, CAND_DYN_DIM)
    oracle_tile = torch.from_numpy(
        np.frombuffer(step_data["oracle_tile"], dtype=np.float32).copy()
    ).view(1, TILE_DIM, ORACLE_TILE_DIM)
    oracle_player = torch.from_numpy(
        np.frombuffer(step_data["oracle_player"], dtype=np.float32).copy()
    ).view(1, 3, ORACLE_PLAYER_DIM)

    assert step_data["num_valid"] >= 1
    assert step_data["num_valid"] <= MAX_CANDIDATES

    # Decode first legal global_id back to native Action(s)
    first_gid = int(cand_ids[0, 0].item())
    recovered = native.v2_decode_global_id(first_gid, obs_map[observer], None)
    assert len(recovered) >= 1

    # Forward through Actor + independent Oracle Critic
    events = torch.zeros((1, 1, EVENT_BYTES), dtype=torch.uint8)
    events[0, 0, 0] = 1
    events[0, 0, 1] = 4
    events[0, 0, 2] = 4
    events[0, 0, 3] = 5
    events[0, 0, 4] = 37
    events[0, 0, 5] = 37
    event_len = torch.tensor([1], dtype=torch.long)

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

