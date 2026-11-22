//! PyO3 zero-copy bindings for KAR V2.4 Rust feature engine.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use riichienv_core::action::Action;
use riichienv_core::game_variant::GameStateVariant;
use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::observation::Observation;
use riichienv_core::rule::GameRule;
use riichienv_core::types::{Conditions, Meld, WinResult};
use riichienv_core::v2_features::action_manifest::{decode_global_id_to_native, manifest};
use riichienv_core::v2_features::candidates::{WinEvalContext, encode_candidates};
use riichienv_core::v2_features::furiten::FuritenState;
use riichienv_core::v2_features::global_player::{RoundGlobalContext, encode_global_player};
use riichienv_core::v2_features::oracle::{encode_belief_targets, encode_oracle};
use riichienv_core::v2_features::tile_feats::encode_tile_feats;
use riichienv_core::v2_features::visible::VisibleTileCounts;
use riichienv_core::v2_features::{
    DecisionPhase, HanchanTracker, RuleContract, V2FeatureConfig,
};

use crate::env::RiichiEnv;

fn slice_to_pybytes<'py, T>(py: Python<'py>, slice: &[T]) -> Bound<'py, PyBytes> {
    let byte_len = std::mem::size_of_val(slice);
    let ptr = slice.as_ptr() as *const u8;
    // SAFETY: `T` is a plain numeric primitive (`f32`, `i16`, `u8`) with no padding.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
    PyBytes::new(py, bytes)
}

fn rule_contract_from_profile(profile: &str, rule: GameRule) -> RuleContract {
    let mut rc = match profile {
        "mjsoul" | "mjsoul_throne" => RuleContract::default_mjsoul_throne(),
        _ => RuleContract::default_tenhou_houou(),
    };
    rc.rule = rule;
    rc
}

fn phase_from_u8(v: u8) -> PyResult<DecisionPhase> {
    match v {
        0 => Ok(DecisionPhase::OwnAct),
        1 => Ok(DecisionPhase::DiscardResp),
        2 => Ok(DecisionPhase::KakanResp),
        3 => Ok(DecisionPhase::AnkanResp),
        _ => Err(PyValueError::new_err(format!("invalid DecisionPhase {v}"))),
    }
}

/// Export the 547-row static `ACTION_MANIFEST` lookup table for PyTorch `register_buffer`.
#[pyfunction]
fn v2_manifest_table<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let table = manifest();
    let n = table.len();
    let mut action_type = Vec::with_capacity(n);
    let mut target_base34 = Vec::with_capacity(n);
    let mut consumed0_base34 = Vec::with_capacity(n);
    let mut consumed1_base34 = Vec::with_capacity(n);
    let mut source_rel = Vec::with_capacity(n);
    let mut has_slot_flags = Vec::with_capacity(n * 4);
    let mut static_flags = Vec::with_capacity(n * 4);

    for e in table {
        action_type.push(e.kind as u8);
        target_base34.push(e.target_base34);
        consumed0_base34.push(e.consumed0_base34);
        consumed1_base34.push(e.consumed1_base34);
        source_rel.push(e.source_rel);
        for f in e.has_slot_flags {
            has_slot_flags.push(if f { 1.0f32 } else { 0.0f32 });
        }
        for f in e.static_flags {
            static_flags.push(if f { 1.0f32 } else { 0.0f32 });
        }
    }

    let d = PyDict::new(py);
    d.set_item("num_rows", n)?;
    d.set_item("action_type", slice_to_pybytes(py, &action_type))?;
    d.set_item("target_base34", slice_to_pybytes(py, &target_base34))?;
    d.set_item("consumed0_base34", slice_to_pybytes(py, &consumed0_base34))?;
    d.set_item("consumed1_base34", slice_to_pybytes(py, &consumed1_base34))?;
    d.set_item("source_rel", slice_to_pybytes(py, &source_rel))?;
    d.set_item("has_slot_flags", slice_to_pybytes(py, &has_slot_flags))?;
    d.set_item("static_flags", slice_to_pybytes(py, &static_flags))?;
    Ok(d)
}

/// Rule-aware hand evaluation (`HandEvaluator::calc_with_rule`) exposed to Python.
#[pyfunction]
#[pyo3(signature = (hand_136, melds, win_tile, dora_indicators, ura_indicators, conditions=None, rule=None))]
fn v2_calc_with_rule(
    hand_136: Vec<u8>,
    melds: Vec<Meld>,
    win_tile: u8,
    dora_indicators: Vec<u8>,
    ura_indicators: Vec<u8>,
    conditions: Option<Conditions>,
    rule: Option<GameRule>,
) -> WinResult {
    let eval = HandEvaluator::new(hand_136, melds);
    let r = rule.unwrap_or_default();
    eval.calc_with_rule(win_tile, &dora_indicators, &ura_indicators, conditions, &r)
}

/// Encode Actor-only tensors strictly from `&Observation` (compile-time zero leakage).
///
/// Uses `obs.furiten` captured from the Rust state machine by default; optional
/// `temp_furiten`/`riichi_furiten` flags may override only for synthetic fixtures.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    obs,
    phase=0,
    trigger_actor_rel=None,
    live_wall_left=70,
    is_first_turn=false,
    temp_furiten=None,
    riichi_furiten=None,
    rule_profile="tenhou",
    enable_early_shape_hint=false
))]
fn v2_encode_actor_obs<'py>(
    py: Python<'py>,
    obs: &Observation,
    phase: u8,
    trigger_actor_rel: Option<u8>,
    live_wall_left: u8,
    is_first_turn: bool,
    temp_furiten: Option<bool>,
    riichi_furiten: Option<bool>,
    rule_profile: &str,
    enable_early_shape_hint: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let dec_phase = phase_from_u8(phase)?;
    let vis = VisibleTileCounts::from_observation(obs).map_err(PyValueError::new_err)?;
    let fur = match (temp_furiten, riichi_furiten) {
        (None, None) => obs.furiten,
        (t, r) => FuritenState::from_observation(
            obs,
            t.unwrap_or(obs.furiten.temporary),
            r.unwrap_or(obs.furiten.riichi),
            &obs.rule,
        ),
    };
    let rc = rule_contract_from_profile(rule_profile, obs.rule);
    let tracker = HanchanTracker::default();
    let ctx = RoundGlobalContext {
        phase: dec_phase,
        trigger_actor_rel,
        live_wall_left,
        is_first_uninterrupted_turn: is_first_turn,
        ..Default::default()
    };
    let cfg = V2FeatureConfig {
        enable_early_shape_hint,
    };

    let tile_feats = encode_tile_feats(obs, &vis, fur, &obs.rule, dec_phase, &cfg)
        .map_err(PyValueError::new_err)?;
    let (global_feats, player_feats) =
        encode_global_player(obs, fur, &tracker, &rc, &ctx).map_err(PyValueError::new_err)?;
    let cands = encode_candidates(
        obs,
        &vis,
        fur,
        &tracker,
        &rc,
        &ctx,
        &WinEvalContext::default(),
    )
    .map_err(PyValueError::new_err)?;

    let player_flat: Vec<f32> = player_feats.iter().flatten().copied().collect();
    let cand_mask_u8: [u8; 32] = cands.cand_mask.map(|m| if m { 1 } else { 0 });
    let cand_dyn_flat: Vec<f32> = cands.cand_dyn.iter().flatten().copied().collect();

    let d = PyDict::new(py);
    d.set_item("tile_feats", slice_to_pybytes(py, &tile_feats))?;
    d.set_item("global_feats", slice_to_pybytes(py, &global_feats))?;
    d.set_item("player_feats", slice_to_pybytes(py, &player_flat))?;
    d.set_item("cand_ids", slice_to_pybytes(py, &cands.cand_ids))?;
    d.set_item("cand_mask", slice_to_pybytes(py, &cand_mask_u8))?;
    d.set_item("cand_mode", slice_to_pybytes(py, &cands.cand_mode))?;
    d.set_item("cand_dyn", slice_to_pybytes(py, &cand_dyn_flat))?;
    d.set_item("num_valid", cands.num_valid)?;
    d.set_item("ruleset_sha256", rc.ruleset_sha256())?;
    Ok(d)
}

/// Encode training-only Oracle Critic features strictly from a live 4-player [`RiichiEnv`].
#[pyfunction]
fn v2_encode_oracle_env<'py>(
    py: Python<'py>,
    env: &RiichiEnv,
    observer: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let GameStateVariant::FourPlayer(ref state) = env.variant else {
        return Err(PyValueError::new_err(
            "KAR V2 Oracle encoder only supports 4-player RiichiEnv",
        ));
    };
    let (oracle_tile, oracle_player) =
        encode_oracle(state, observer).map_err(PyValueError::new_err)?;
    let op_flat: Vec<f32> = oracle_player.iter().flatten().copied().collect();
    let d = PyDict::new(py);
    d.set_item("oracle_tile", slice_to_pybytes(py, &oracle_tile))?;
    d.set_item("oracle_player", slice_to_pybytes(py, &op_flat))?;
    Ok(d)
}

/// Encode exact Stage-3 Opponent Belief supervision targets strictly from a
/// live 4-player [`RiichiEnv`] (`&GameState`-isolated; NEVER exposed to Actor inputs).
#[pyfunction]
fn v2_encode_belief_targets_env<'py>(
    py: Python<'py>,
    env: &RiichiEnv,
    observer: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let GameStateVariant::FourPlayer(ref state) = env.variant else {
        return Err(PyValueError::new_err(
            "KAR V2 Belief target encoder only supports 4-player RiichiEnv",
        ));
    };
    let targets = encode_belief_targets(state, observer).map_err(PyValueError::new_err)?;
    let waits_flat: Vec<u8> = targets.opp_waits.iter().flatten().copied().collect();
    let counts_flat: Vec<u8> = targets.opp_hand_counts.iter().flatten().copied().collect();
    let d = PyDict::new(py);
    d.set_item("opp_tenpai", slice_to_pybytes(py, &targets.opp_tenpai))?;
    d.set_item("opp_waits", slice_to_pybytes(py, &waits_flat))?;
    d.set_item(
        "exact_dealin_now",
        slice_to_pybytes(py, &targets.exact_dealin_now),
    )?;
    d.set_item("opp_hand_counts", slice_to_pybytes(py, &counts_flat))?;
    Ok(d)
}

/// Encode full Actor (+ live lossless `events` + real `HanchanTracker`) and optional
/// Oracle / Belief supervision tensors directly from a live or replayed 4-player [`RiichiEnv`].
#[pyfunction]
#[pyo3(signature = (
    env,
    observer,
    rule_profile="tenhou",
    enable_early_shape_hint=false,
    include_oracle=true,
    include_belief=false
))]
fn v2_encode_env_step<'py>(
    py: Python<'py>,
    env: &mut RiichiEnv,
    observer: u8,
    rule_profile: &str,
    enable_early_shape_hint: bool,
    include_oracle: bool,
    include_belief: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let GameStateVariant::FourPlayer(ref mut state) = env.variant else {
        return Err(PyValueError::new_err(
            "KAR V2 encoder only supports 4-player RiichiEnv",
        ));
    };

    let obs = state.get_observation(observer);
    let vis = VisibleTileCounts::from_observation(&obs).map_err(PyValueError::new_err)?;
    let fur = FuritenState::from_game_state(state, observer);
    let rc = rule_contract_from_profile(rule_profile, state.rule);
    let ctx = state.v2_round_global_context(observer);
    let phase = ctx.phase;
    let cfg = V2FeatureConfig {
        enable_early_shape_hint,
    };

    let tile_feats = encode_tile_feats(&obs, &vis, fur, &state.rule, phase, &cfg)
        .map_err(PyValueError::new_err)?;
    let (global_feats, player_feats) =
        encode_global_player(&obs, fur, &state.v2_hanchan_tracker, &rc, &ctx)
            .map_err(PyValueError::new_err)?;
    let cands = encode_candidates(
        &obs,
        &vis,
        fur,
        &state.v2_hanchan_tracker,
        &rc,
        &ctx,
        &WinEvalContext {
            is_rinshan: state.is_rinshan_flag,
            is_chankan: matches!(phase, DecisionPhase::KakanResp | DecisionPhase::AnkanResp),
            is_haitei_or_houtei: state.wall.drawable_count == 0,
            is_tsumo_first_turn: state.is_first_turn,
            pao_payer: state.players[observer as usize].pao.values().next().copied(),
        },
    )
    .map_err(PyValueError::new_err)?;
    let events = state
        .v2_encode_round_events(observer)
        .map_err(PyValueError::new_err)?;
    let events_flat: Vec<u8> = events.iter().flatten().copied().collect();

    let player_flat: Vec<f32> = player_feats.iter().flatten().copied().collect();
    let cand_mask_u8: [u8; 32] = cands.cand_mask.map(|m| if m { 1 } else { 0 });
    let cand_dyn_flat: Vec<f32> = cands.cand_dyn.iter().flatten().copied().collect();

    let d = PyDict::new(py);
    d.set_item("tile_feats", slice_to_pybytes(py, &tile_feats))?;
    d.set_item("global_feats", slice_to_pybytes(py, &global_feats))?;
    d.set_item("player_feats", slice_to_pybytes(py, &player_flat))?;
    d.set_item("cand_ids", slice_to_pybytes(py, &cands.cand_ids))?;
    d.set_item("cand_mask", slice_to_pybytes(py, &cand_mask_u8))?;
    d.set_item("cand_mode", slice_to_pybytes(py, &cands.cand_mode))?;
    d.set_item("cand_dyn", slice_to_pybytes(py, &cand_dyn_flat))?;
    d.set_item("num_valid", cands.num_valid)?;
    d.set_item("events", slice_to_pybytes(py, &events_flat))?;
    d.set_item("event_len", events.len())?;
    d.set_item("rounds_completed", state.v2_hanchan_tracker.rounds_completed)?;
    d.set_item("ruleset_sha256", rc.ruleset_sha256())?;

    if include_oracle {
        let (oracle_tile, oracle_player) =
            encode_oracle(state, observer).map_err(PyValueError::new_err)?;
        let op_flat: Vec<f32> = oracle_player.iter().flatten().copied().collect();
        d.set_item("oracle_tile", slice_to_pybytes(py, &oracle_tile))?;
        d.set_item("oracle_player", slice_to_pybytes(py, &op_flat))?;
    }
    if include_belief {
        let targets = encode_belief_targets(state, observer).map_err(PyValueError::new_err)?;
        let waits_flat: Vec<u8> = targets.opp_waits.iter().flatten().copied().collect();
        let counts_flat: Vec<u8> = targets.opp_hand_counts.iter().flatten().copied().collect();
        d.set_item("opp_tenpai", slice_to_pybytes(py, &targets.opp_tenpai))?;
        d.set_item("opp_waits", slice_to_pybytes(py, &waits_flat))?;
        d.set_item(
            "exact_dealin_now",
            slice_to_pybytes(py, &targets.exact_dealin_now),
        )?;
        d.set_item("opp_hand_counts", slice_to_pybytes(py, &counts_flat))?;
    }
    Ok(d)
}

/// Recover native execution `Action`(s) for a chosen 546 `global_id`.
#[pyfunction]
#[pyo3(signature = (global_id, obs, source_rel_hint=None))]
fn v2_decode_global_id(
    global_id: u16,
    obs: &Observation,
    source_rel_hint: Option<u8>,
) -> PyResult<Vec<Action>> {
    decode_global_id_to_native(global_id, obs, source_rel_hint).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(v2_manifest_table, m)?)?;
    m.add_function(wrap_pyfunction!(v2_calc_with_rule, m)?)?;
    m.add_function(wrap_pyfunction!(v2_encode_actor_obs, m)?)?;
    m.add_function(wrap_pyfunction!(v2_encode_oracle_env, m)?)?;
    m.add_function(wrap_pyfunction!(v2_encode_belief_targets_env, m)?)?;
    m.add_function(wrap_pyfunction!(v2_encode_env_step, m)?)?;
    m.add_function(wrap_pyfunction!(v2_decode_global_id, m)?)?;
    Ok(())
}
