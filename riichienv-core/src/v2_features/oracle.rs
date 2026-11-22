//! Compile-time isolated Asymmetric Oracle Critic features (Contract v2.4 Section 3.2).
//!
//! Accepts ONLY `&GameState` (never callable on `&Observation`).
//! Live wall features are strictly permutation-invariant over draw order.

use crate::hand_evaluator::HandEvaluator;
use crate::shanten::calc_shanten_from_counts;
use crate::state::GameState;
use crate::types::{Conditions, TILE_MAX, Wind};

use super::furiten::{FuritenState, own_discard_mask_u8};
use super::{
    ORACLE_PLAYER_DIM, ORACLE_TILE_DIM, TILE_DIM, dora_indicator_to_dora_base34, is_red5_tile136,
};

const LN_61: f32 = 4.110_873_8;
const LN_64001: f32 = 11.066_654;

#[inline]
fn set_ot(out: &mut [f32; TILE_DIM * ORACLE_TILE_DIM], tile: usize, ch: usize, val: f32) {
    out[tile * ORACLE_TILE_DIM + ch] = val;
}

/// Extract the exact dead-wall uradora indicators matching currently revealed dora indicators.
///
/// In `WallState`: rinshan draws call `wall.tiles.remove(0)`, shifting all
/// dead-wall indices down by `rinshan_draw_count` (matching `GameState::_get_ura_indicators`).
fn revealed_uradora_indicators(state: &GameState) -> Vec<u8> {
    let shift = state.wall.rinshan_draw_count as usize;
    let mut ura = Vec::with_capacity(state.wall.dora_indicators.len());
    for i in 0..state.wall.dora_indicators.len() {
        let ura_idx = (5 + i * 2).saturating_sub(shift);
        if ura_idx < state.wall.tiles.len() {
            ura.push(state.wall.tiles[ura_idx]);
        }
    }
    ura
}

/// Extract remaining unseen live-wall tiles (`wall.tiles[(14 - rinshan_draw_count)..]`,
/// permutation-invariant histogram) and remaining rinshan tiles (`0..(4 - rinshan_draw_count)`).
fn live_wall_and_rinshan_histograms(state: &GameState) -> ([u8; TILE_MAX], bool, [u8; TILE_MAX]) {
    let shift = (state.wall.rinshan_draw_count as usize).min(4);
    let dead_wall_end = 14usize.saturating_sub(shift);
    let mut live_counts = [0u8; TILE_MAX];
    let mut live_has_red5 = false;
    if state.wall.tiles.len() > dead_wall_end {
        for &t in &state.wall.tiles[dead_wall_end..] {
            let b = (t / 4) as usize;
            if b < TILE_MAX {
                live_counts[b] = live_counts[b].saturating_add(1);
            }
            if is_red5_tile136(t) {
                live_has_red5 = true;
            }
        }
    }

    let mut rinshan_counts = [0u8; TILE_MAX];
    let remaining_rinshan = (4usize.saturating_sub(shift)).min(state.wall.tiles.len());
    for &t in &state.wall.tiles[..remaining_rinshan] {
        let b = (t / 4) as usize;
        if b < TILE_MAX {
            rinshan_counts[b] = rinshan_counts[b].saturating_add(1);
        }
    }

    (live_counts, live_has_red5, rinshan_counts)
}

/// Exact Stage-3 Opponent Belief ground-truth supervision targets (compile-time
/// isolated to `&GameState`; NEVER visible to Actor inputs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefTargets {
    /// Whether each relative opponent `[1=Shimocha, 2=Toimen, 3=Kamicha]` is tenpai (`0` or `1`).
    pub opp_tenpai: [u8; 3],
    /// Exact rule-valid wait mask `[3, 34]` (`0` or `1`) per relative opponent.
    pub opp_waits: [[u8; TILE_MAX]; 3],
    /// Exact 1-step deal-in mask `[34]` (`0` or `1`): discarding tile `b` right now deals into >=1
    /// non-furiten opponent with valid yaku.
    pub exact_dealin_now: [u8; TILE_MAX],
    /// Exact closed-hand copy counts `[3, 34]` (`0..=4`) per relative opponent.
    pub opp_hand_counts: [[u8; TILE_MAX]; 3],
}

/// Extract exact Stage-3 Opponent Belief supervision targets from `&GameState`.
pub fn encode_belief_targets(state: &GameState, observer: u8) -> Result<BeliefTargets, String> {
    let pid = observer as usize;
    if pid >= 4 {
        return Err(format!("invalid observer {observer}"));
    }

    let mut opp_tenpai = [0u8; 3];
    let mut opp_waits = [[0u8; TILE_MAX]; 3];
    let mut exact_dealin_now = [0u8; TILE_MAX];
    let mut opp_hand_counts = [[0u8; TILE_MAX]; 3];

    for rel in 1..=3usize {
        let opp = (pid + rel) % 4;
        let opp_state = &state.players[opp];

        for &t in &opp_state.hand {
            let b = (t / 4) as usize;
            if b < TILE_MAX {
                opp_hand_counts[rel - 1][b] = opp_hand_counts[rel - 1][b].saturating_add(1).min(4);
            }
        }

        if opp_state.hand.len() % 3 != 1 {
            continue;
        }
        let eval = HandEvaluator::new(opp_state.hand.clone(), opp_state.melds.clone());
        let waits = eval.get_waits_u8_with_rule(&state.rule);
        if waits.is_empty() {
            continue;
        }
        opp_tenpai[rel - 1] = 1;
        let opp_furiten = FuritenState::from_game_state(state, opp as u8);
        let opp_own_disc = own_discard_mask_u8(&opp_state.discards);
        let opp_is_furiten = opp_furiten.any()
            || waits
                .iter()
                .any(|&w| (w as usize) < TILE_MAX && opp_own_disc[w as usize]);
        let opp_wind = Wind::from((opp as u8 + 4 - state.oya) % 4);

        for &w in &waits {
            let wb = w as usize;
            if wb >= TILE_MAX {
                continue;
            }
            opp_waits[rel - 1][wb] = 1;
            if !opp_is_furiten {
                let cond = Conditions {
                    tsumo: false,
                    riichi: opp_state.riichi_declared,
                    double_riichi: opp_state.double_riichi_declared,
                    ippatsu: opp_state.ippatsu_cycle,
                    houtei: state.wall.drawable_count == 0,
                    player_wind: opp_wind,
                    round_wind: Wind::from(state.round_wind),
                    honba: u32::from(state.honba),
                    ..Default::default()
                };
                let res = eval.calc_with_rule(
                    w * 4 + 1,
                    &state.wall.dora_indicators,
                    &[],
                    Some(cond),
                    &state.rule,
                );
                if res.is_win {
                    exact_dealin_now[wb] = 1;
                }
            }
        }
    }

    Ok(BeliefTargets {
        opp_tenpai,
        opp_waits,
        exact_dealin_now,
        opp_hand_counts,
    })
}

/// Encode training-only Oracle features:
/// - `oracle_tile: [34 * 32]` (row-major `[34, 32]`)
/// - `oracle_player: [[f32; 12]; 3]` (relative opponents `1=下家, 2=对家, 3=上家`)
pub fn encode_oracle(
    state: &GameState,
    observer: u8,
) -> Result<
    (
        [f32; TILE_DIM * ORACLE_TILE_DIM],
        [[f32; ORACLE_PLAYER_DIM]; 3],
    ),
    String,
> {
    let pid = observer as usize;
    if pid >= 4 {
        return Err(format!("invalid observer {observer}"));
    }

    let mut oracle_tile = [0.0f32; TILE_DIM * ORACLE_TILE_DIM];
    let mut oracle_player = [[0.0f32; ORACLE_PLAYER_DIM]; 3];

    let (live_counts, live_has_red5, rinshan_counts) = live_wall_and_rinshan_histograms(state);
    let ura_indicators = revealed_uradora_indicators(state);

    let mut dora_counts = [0u8; TILE_MAX];
    for &ind in &state.wall.dora_indicators {
        let b = dora_indicator_to_dora_base34(ind / 4) as usize;
        if b < TILE_MAX {
            dora_counts[b] = dora_counts[b].saturating_add(1);
        }
    }

    let mut uradora_counts = [0u8; TILE_MAX];
    for &ind in &ura_indicators {
        let b = dora_indicator_to_dora_base34(ind / 4) as usize;
        if b < TILE_MAX {
            uradora_counts[b] = uradora_counts[b].saturating_add(1);
        }
    }

    // [15..18]: live wall count thermometer (permutation-invariant!)
    // [19]: live_wall_has_red5 (per red-5 base34 slot 4/13/22)
    // [20]: true_uradora_value_norm
    // [21]: dead_wall_rinshan_cnt_norm
    let dead_wall_end = 14usize.saturating_sub((state.wall.rinshan_draw_count as usize).min(4));
    let mut live_red5_per_suit = [false; 3];
    if live_has_red5 && state.wall.tiles.len() > dead_wall_end {
        for &t in &state.wall.tiles[dead_wall_end..] {
            if t == 16 {
                live_red5_per_suit[0] = true;
            } else if t == 52 {
                live_red5_per_suit[1] = true;
            } else if t == 88 {
                live_red5_per_suit[2] = true;
            }
        }
    }
    for b in 0..TILE_MAX {
        let lc = live_counts[b];
        for k in 0..4 {
            if lc > k as u8 {
                set_ot(&mut oracle_tile, b, 15 + k, 1.0);
            }
        }
        set_ot(
            &mut oracle_tile,
            b,
            20,
            (f32::from(uradora_counts[b]) / 4.0).min(1.0),
        );
        set_ot(
            &mut oracle_tile,
            b,
            21,
            (f32::from(rinshan_counts[b]) / 4.0).min(1.0),
        );
    }
    for (suit, &has_r5) in live_red5_per_suit.iter().enumerate() {
        if has_r5 {
            set_ot(&mut oracle_tile, suit * 9 + 4, 19, 1.0);
        }
    }

    // Per-opponent hidden state [rel = 1..=3]
    let mut deal_in_now = [false; TILE_MAX];

    for rel in 1..=3usize {
        let opp = (pid + rel) % 4;
        let opp_state = &state.players[opp];

        let mut opp_counts = [0u8; TILE_MAX];
        let mut opp_red5 = [false; TILE_MAX];
        let mut closed_dora_total = 0u32;
        let mut closed_aka_total = 0u32;
        let mut uradora_hits = 0u32;

        for &t in &opp_state.hand {
            let b = (t / 4) as usize;
            if b < TILE_MAX {
                opp_counts[b] = opp_counts[b].saturating_add(1);
                closed_dora_total += u32::from(dora_counts[b]);
                uradora_hits += u32::from(uradora_counts[b]);
            }
            if is_red5_tile136(t) {
                opp_red5[b] = true;
                closed_aka_total += 1;
            }
        }
        for m in &opp_state.melds {
            for &t in &m.tiles {
                let b = (t / 4) as usize;
                if b < TILE_MAX {
                    uradora_hits += u32::from(uradora_counts[b]);
                }
            }
        }

        // [0..11]: opp_closed_cnt_ge{1..4}, [12..14]: opp_closed_has_red5, [29..31]: opp_closed_holds_dora_copy
        let cnt_off = (rel - 1) * 4;
        for b in 0..TILE_MAX {
            let c = opp_counts[b];
            for k in 0..4 {
                if c > k as u8 {
                    set_ot(&mut oracle_tile, b, cnt_off + k, 1.0);
                }
            }
            if opp_red5[b] {
                set_ot(&mut oracle_tile, b, 12 + (rel - 1), 1.0);
            }
            if c > 0 && dora_counts[b] > 0 {
                set_ot(
                    &mut oracle_tile,
                    b,
                    29 + (rel - 1),
                    (f32::from(c) * f32::from(dora_counts[b]) / 4.0).min(1.0),
                );
            }
        }

        // Opponent true shanten, tenpai, waits, effective tiles, furiten, max wait value
        let opp_furiten = FuritenState::from_game_state(state, opp as u8);
        let opp_own_disc = own_discard_mask_u8(&opp_state.discards);
        let num_tiles: u8 = opp_counts.iter().sum();
        let len_div3 = num_tiles / 3;
        let true_shanten = calc_shanten_from_counts(&opp_counts, len_div3);

        let mut true_ukeire = 0u32;
        if num_tiles % 3 == 1 {
            let mut next = opp_counts;
            for b in 0..TILE_MAX {
                if next[b] >= 4 {
                    continue;
                }
                next[b] += 1;
                let s = calc_shanten_from_counts(&next, len_div3);
                next[b] -= 1;
                if s < true_shanten {
                    set_ot(&mut oracle_tile, b, 25 + (rel - 1), 1.0);
                    true_ukeire += u32::from(live_counts[b]);
                }
            }
        }

        let eval = HandEvaluator::new(opp_state.hand.clone(), opp_state.melds.clone());
        let waits = if opp_state.hand.len() % 3 == 1 {
            eval.get_waits_u8_with_rule(&state.rule)
        } else {
            Vec::new()
        };
        let is_tenpai = !waits.is_empty();
        let mut live_waits_remaining = 0u32;
        let mut max_wait_han = 0u32;
        let mut max_wait_pts = 0u32;
        let mut has_valid_ron_yaku_now = false;

        if is_tenpai {
            let opp_wind = Wind::from((opp as u8 + 4 - state.oya) % 4);
            let opp_is_furiten = opp_furiten.any()
                || waits
                    .iter()
                    .any(|&w| (w as usize) < TILE_MAX && opp_own_disc[w as usize]);

            for &w in &waits {
                let wb = w as usize;
                if wb >= TILE_MAX {
                    continue;
                }
                set_ot(&mut oracle_tile, wb, 22 + (rel - 1), 1.0);
                live_waits_remaining += u32::from(live_counts[wb]);

                let cond = Conditions {
                    tsumo: false,
                    riichi: opp_state.riichi_declared,
                    double_riichi: opp_state.double_riichi_declared,
                    ippatsu: opp_state.ippatsu_cycle,
                    houtei: state.wall.drawable_count == 0,
                    player_wind: opp_wind,
                    round_wind: Wind::from(state.round_wind),
                    honba: u32::from(state.honba),
                    ..Default::default()
                };
                // Pick canonical non-red tile136 for wait evaluation
                let win_t136 = w * 4 + 1;
                let ura_for_opp: &[u8] = if opp_state.riichi_declared {
                    &ura_indicators
                } else {
                    &[]
                };
                let res = eval.calc_with_rule(
                    win_t136,
                    &state.wall.dora_indicators,
                    ura_for_opp,
                    Some(cond),
                    &state.rule,
                );
                if res.is_win {
                    has_valid_ron_yaku_now = true;
                    if res.han > max_wait_han {
                        max_wait_han = res.han;
                    }
                    if res.ron_agari > max_wait_pts {
                        max_wait_pts = res.ron_agari;
                    }
                    if !opp_is_furiten {
                        deal_in_now[wb] = true;
                    }
                }
            }
        }

        let op = &mut oracle_player[rel - 1];
        op[0] = (f32::from(true_shanten).max(0.0) / 8.0).min(1.0);
        op[1] = if is_tenpai { 1.0 } else { 0.0 };
        op[2] = ((1.0 + true_ukeire as f32).ln() / LN_61).min(1.0);
        op[3] = if opp_furiten.any() { 1.0 } else { 0.0 };
        op[4] = (closed_dora_total as f32 / 8.0).min(2.0);
        op[5] = (closed_aka_total as f32 / 3.0).min(1.0);
        op[6] = if opp_state.riichi_declared {
            (uradora_hits as f32 / 4.0).min(2.0)
        } else {
            0.0
        };
        op[7] = (max_wait_han as f32 / 13.0).min(2.0);
        op[8] = ((1.0 + max_wait_pts as f32).ln() / LN_64001).min(1.5);
        op[9] = if opp_state.melds.iter().all(|m| !m.opened) {
            1.0
        } else {
            0.0
        };
        op[10] = if has_valid_ron_yaku_now { 1.0 } else { 0.0 };
        op[11] = (live_waits_remaining as f32 / 16.0).min(1.0);
    }

    // [28]: exact_deal_in_now
    for (b, &danger) in deal_in_now.iter().enumerate() {
        if danger {
            set_ot(&mut oracle_tile, b, 28, 1.0);
        }
    }

    Ok((oracle_tile, oracle_player))
}
