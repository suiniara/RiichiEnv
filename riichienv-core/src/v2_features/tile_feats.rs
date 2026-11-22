//! 64-channel 34-tile spatial feature encoder (Contract v2.4 Section 1.2).
//!
//! Strictly consumes `&Observation`, `&VisibleTileCounts`, `&FuritenState`, and
//! `&V2FeatureConfig`. Zero opponent private inference; zero double-counting.

use crate::action::ActionType;
use crate::hand_evaluator::HandEvaluator;
use crate::observation::Observation;
use crate::rule::GameRule;
use crate::types::TILE_MAX;

use super::furiten::{FuritenState, own_discard_mask_u32};
use super::visible::VisibleTileCounts;
use super::{
    DecisionPhase, TILE_DIM, TILE_FEATS_DIM, V2FeatureConfig, dora_indicator_to_dora_base34,
    is_red5_tile136,
};

const LN_61: f32 = 4.110_873_8; // ln(61.0)

fn ryanmen_pairs(rank1: usize) -> &'static [(usize, usize)] {
    match rank1 {
        1 => &[(2, 3)],
        2 => &[(3, 4)],
        3 => &[(4, 5)],
        4 => &[(2, 3), (5, 6)],
        5 => &[(3, 4), (6, 7)],
        6 => &[(4, 5), (7, 8)],
        7 => &[(5, 6)],
        8 => &[(6, 7)],
        9 => &[(7, 8)],
        _ => &[],
    }
}

fn suji_partners(rank1: usize) -> &'static [usize] {
    match rank1 {
        1 => &[4],
        2 => &[5],
        3 => &[6],
        4 => &[1, 7],
        5 => &[2, 8],
        6 => &[3, 9],
        7 => &[4],
        8 => &[5],
        9 => &[6],
        _ => &[],
    }
}

fn yakuhai_weight(base34: usize, round_wind: u8, seat_wind: u8) -> f32 {
    if (31..=33).contains(&base34) {
        return 0.5;
    }
    if (27..=30).contains(&base34) {
        let w = (base34 - 27) as u8;
        let mut count = 0u8;
        if w == round_wind % 4 {
            count += 1;
        }
        if w == seat_wind % 4 {
            count += 1;
        }
        return match count {
            0 => 0.0,
            1 => 0.5,
            _ => 1.0,
        };
    }
    0.0
}

#[inline]
fn set_ch(out: &mut [f32; TILE_DIM * TILE_FEATS_DIM], tile: usize, ch: usize, val: f32) {
    out[tile * TILE_FEATS_DIM + ch] = val;
}

/// Encode `[34, 64]` row-major spatial tile features strictly matching KAR V2.4.
pub fn encode_tile_feats(
    obs: &Observation,
    visible: &VisibleTileCounts,
    furiten: FuritenState,
    rule: &GameRule,
    phase: DecisionPhase,
    config: &V2FeatureConfig,
) -> Result<[f32; TILE_DIM * TILE_FEATS_DIM], String> {
    let pid = obs.player_id as usize;
    if pid >= 4 {
        return Err(format!("invalid observer player_id {}", obs.player_id));
    }

    let mut out = [0.0f32; TILE_DIM * TILE_FEATS_DIM];

    // Precompute legal discard base34 mask in OwnAct.
    let legal = obs.legal_actions_method();
    let mut legal_discard_base34 = [false; TILE_MAX];
    if phase == DecisionPhase::OwnAct {
        for act in &legal {
            if act.action_type == ActionType::Discard
                && let Some(t) = act.tile
            {
                let b = (t / 4) as usize;
                if b < TILE_MAX {
                    legal_discard_base34[b] = true;
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Group A [0..18]: Own hand, melds, tsumo, kuikae, dora, unseen, own river
    // ---------------------------------------------------------------------
    let mut self_closed_red5 = [false; TILE_MAX];
    for &t in &obs.hands[pid] {
        let t8 = t as u8;
        if is_red5_tile136(t8) {
            self_closed_red5[(t8 / 4) as usize] = true;
        }
    }

    let mut self_meld_counts = [0u8; TILE_MAX];
    for meld in &obs.melds[pid] {
        for &t in &meld.tiles {
            let b = (t / 4) as usize;
            if b < TILE_MAX {
                self_meld_counts[b] = self_meld_counts[b].saturating_add(1);
            }
        }
    }

    let mut dora_counts = [0u8; TILE_MAX];
    for &ind in &obs.dora_indicators {
        let dora_b = dora_indicator_to_dora_base34((ind / 4) as u8) as usize;
        if dora_b < TILE_MAX {
            dora_counts[dora_b] = dora_counts[dora_b].saturating_add(1);
        }
    }

    let self_discarded_ever = own_discard_mask_u32(&obs.discards[pid]);

    // Immediately after CHI/PON in OwnAct (drawn_tile is None, not riichi),
    // any tile present in closed hand that is excluded from legal discards is kuikae-forbidden.
    let is_post_call_discard =
        phase == DecisionPhase::OwnAct && obs.drawn_tile.is_none() && !obs.riichi_declared[pid];

    for b in 0..TILE_MAX {
        let h = visible.self_hand_base34[b];
        for k in 0..4 {
            if h > k as u8 {
                set_ch(&mut out, b, k, 1.0);
            }
        }
        if self_closed_red5[b] {
            set_ch(&mut out, b, 4, 1.0);
        }
        let m = self_meld_counts[b];
        for k in 0..4 {
            if m > k as u8 {
                set_ch(&mut out, b, 5 + k, 1.0);
            }
        }
        if is_post_call_discard && h > 0 && !legal_discard_base34[b] {
            set_ch(&mut out, b, 11, 1.0);
        }
        set_ch(&mut out, b, 12, (f32::from(dora_counts[b]) / 4.0).min(1.0));
        let u = visible.unseen_base34[b];
        for k in 0..4 {
            if u > k as u8 {
                set_ch(&mut out, b, 14 + k, 1.0);
            }
        }
        if self_discarded_ever[b] {
            set_ch(&mut out, b, 18, 1.0);
        }
    }

    // Channels 9..10: self_last_tsumo & self_last_tsumo_is_red5 (OwnAct post-draw only)
    if phase == DecisionPhase::OwnAct
        && let Some(dt) = obs.drawn_tile
    {
        let b = (dt / 4) as usize;
        if b < TILE_MAX {
            set_ch(&mut out, b, 9, 1.0);
            if is_red5_tile136(dt) {
                set_ch(&mut out, b, 10, 1.0);
            }
        }
    }

    // Channel 13: red5_unseen at 5m(4), 5p(13), 5s(22)
    for (suit, &unseen_r5) in visible.red5_unseen.iter().enumerate() {
        if unseen_r5 {
            set_ch(&mut out, suit * 9 + 4, 13, 1.0);
        }
    }

    // ---------------------------------------------------------------------
    // Group B1 [19..45]: 3 opponents x 9 strictly public own-river/meld channels
    // ---------------------------------------------------------------------
    for rel in 1..=3usize {
        let opp = (pid + rel) % 4;
        let off = 19 + (rel - 1) * 9;
        let river = &obs.discards[opp];
        let tsumogiri_flags = &obs.tsumogiri_flags[opp];

        let mut opp_own_river = [false; TILE_MAX];
        let mut opp_last_pos = [None::<usize>; TILE_MAX];
        let mut opp_last_tedashi = [false; TILE_MAX];

        for (idx, &t) in river.iter().enumerate() {
            let b = (t / 4) as usize;
            if b < TILE_MAX {
                opp_own_river[b] = true;
                opp_last_pos[b] = Some(idx);
                // If tsumogiri_flags is populated, tedashi == !tsumogiri.
                if let Some(&is_tsumogiri) = tsumogiri_flags.get(idx) {
                    opp_last_tedashi[b] = !is_tsumogiri;
                }
            }
        }

        // +0: opp_own_river_genbutsu
        for (b, &seen) in opp_own_river.iter().enumerate() {
            if seen {
                set_ch(&mut out, b, off, 1.0);
            }
        }

        // +1: opp_riichi_declare_tile (public sideways tile in opp's river)
        if let Some(rt) = obs.riichi_sutehais[opp] {
            let b = (rt / 4) as usize;
            if b < TILE_MAX {
                set_ch(&mut out, b, off + 1, 1.0);
            }
        }

        // +2: opp_last_discard_tile (latest tile in opp's river)
        if let Some(&last_t) = river.last() {
            let b = (last_t / 4) as usize;
            if b < TILE_MAX {
                set_ch(&mut out, b, off + 2, 1.0);
            }
        }

        // +3, +4: opp_suji_one_side, opp_suji_full (STRICTLY from opp_own_river)
        for suit in 0..3usize {
            for rank1 in 1..=9usize {
                let b = suit * 9 + (rank1 - 1);
                let partners = suji_partners(rank1);
                let any_hit = partners.iter().any(|&p| opp_own_river[suit * 9 + (p - 1)]);
                let all_hit = partners.iter().all(|&p| opp_own_river[suit * 9 + (p - 1)]);
                if any_hit {
                    set_ch(&mut out, b, off + 3, 1.0);
                }
                if all_hit {
                    set_ch(&mut out, b, off + 4, 1.0);
                }
            }
        }

        // +5: opp_early_discard_shape_hint (Contract v2.4 P1-2: default 0.0 when disabled)
        if config.enable_early_shape_hint {
            let mut hint = [false; TILE_MAX];
            for &t in river.iter().take(6) {
                let b = (t / 4) as usize;
                if b < 27 {
                    let suit = b / 9;
                    let r_d = (b % 9) + 1;
                    if (2..=4).contains(&r_d) {
                        for r in 1..r_d {
                            hint[suit * 9 + (r - 1)] = true;
                        }
                    } else if (6..=8).contains(&r_d) {
                        for r in (r_d + 1)..=9 {
                            hint[suit * 9 + (r - 1)] = true;
                        }
                    }
                }
            }
            for (b, &h) in hint.iter().enumerate() {
                if h {
                    set_ch(&mut out, b, off + 5, 1.0);
                }
            }
        }

        // +6: opp_discard_recency, +7: opp_last_discard_tedashi
        let river_len = river.len();
        for b in 0..TILE_MAX {
            if let Some(pos) = opp_last_pos[b] {
                let delta = (river_len - 1 - pos) as f32;
                set_ch(&mut out, b, off + 6, (-delta / 6.0).exp());
                if opp_last_tedashi[b] {
                    set_ch(&mut out, b, off + 7, 1.0);
                }
            }
        }

        // +8: opp_meld_cnt_norm
        let mut opp_meld_cnt = [0u8; TILE_MAX];
        for meld in &obs.melds[opp] {
            for &t in &meld.tiles {
                let b = (t / 4) as usize;
                if b < TILE_MAX {
                    opp_meld_cnt[b] = opp_meld_cnt[b].saturating_add(1);
                }
            }
        }
        for (b, &cnt) in opp_meld_cnt.iter().enumerate() {
            if cnt > 0 {
                set_ch(&mut out, b, off + 8, (f32::from(cnt) / 4.0).min(1.0));
            }
        }
    }

    // ---------------------------------------------------------------------
    // Group B2 [46..53]: Kabe (from VisibleTileCounts), dora adj, yakuhai
    // ---------------------------------------------------------------------
    for suit in 0..3usize {
        for rank1 in 1..=9usize {
            let b = suit * 9 + (rank1 - 1);
            let pairs = ryanmen_pairs(rank1);
            let no_chance = pairs.iter().all(|&(l, r)| {
                visible.total_known(suit * 9 + (l - 1)) == 4
                    || visible.total_known(suit * 9 + (r - 1)) == 4
            });
            if no_chance {
                set_ch(&mut out, b, 46, 1.0);
            } else {
                let one_chance = pairs.iter().all(|&(l, r)| {
                    visible.total_known(suit * 9 + (l - 1)) >= 3
                        || visible.total_known(suit * 9 + (r - 1)) >= 3
                });
                if one_chance {
                    set_ch(&mut out, b, 47, 1.0);
                }
            }

            // Dora adjacency dist1 / dist2 in same suit
            let mut adj1 = false;
            let mut adj2 = false;
            for d_rank1 in 1..=9usize {
                if dora_counts[suit * 9 + (d_rank1 - 1)] > 0 {
                    let dist = rank1.abs_diff(d_rank1);
                    if dist == 1 {
                        adj1 = true;
                    } else if dist == 2 {
                        adj2 = true;
                    }
                }
            }
            if adj1 {
                set_ch(&mut out, b, 48, 1.0);
            }
            if adj2 {
                set_ch(&mut out, b, 49, 1.0);
            }
        }
    }

    // Yakuhai weights [50..53]
    for b in 27..TILE_MAX {
        let self_wind = (obs.player_id + 4 - obs.oya) % 4;
        set_ch(&mut out, b, 50, yakuhai_weight(b, obs.round_wind, self_wind));
        for rel in 1..=3usize {
            let opp = ((obs.player_id as usize + rel) % 4) as u8;
            let opp_wind = (opp + 4 - obs.oya) % 4;
            set_ch(
                &mut out,
                b,
                50 + rel,
                yakuhai_weight(b, obs.round_wind, opp_wind),
            );
        }
    }

    // ---------------------------------------------------------------------
    // Group C [54..63]: Rust exact shanten/ukeire (strictly phase-scoped)
    // ---------------------------------------------------------------------
    if phase == DecisionPhase::OwnAct && obs.hands[pid].len() % 3 == 2 {
        let hand_u8: Vec<u8> = obs.hands[pid].iter().map(|&t| t as u8).collect();
        let mut post_shanten = [i8::MAX; TILE_MAX];
        let mut post_ukeire = [0u32; TILE_MAX];
        let mut post_tenpai = [false; TILE_MAX];
        let mut post_furiten = [false; TILE_MAX];

        let mut min_shanten = i8::MAX;
        let mut best_ukeire = 0u32;

        for b in 0..TILE_MAX {
            if !legal_discard_base34[b] || visible.self_hand_base34[b] == 0 {
                continue;
            }
            let mut next_counts = visible.self_hand_base34;
            next_counts[b] -= 1;
            let (s, u, _) = visible.shanten_and_visible_ukeire(&next_counts);
            post_shanten[b] = s;
            post_ukeire[b] = u;
            if s < min_shanten {
                min_shanten = s;
            }
            if u > best_ukeire {
                best_ukeire = u;
            }

            if s == 0 {
                // Rule-accurate tenpai & waits (respecting allow_fifth_copy_tenpai)
                let mut rem_hand = hand_u8.clone();
                if let Some(pos) = rem_hand.iter().position(|&t| (t / 4) as usize == b) {
                    rem_hand.remove(pos);
                }
                let eval = HandEvaluator::new(rem_hand, obs.melds[pid].clone());
                let waits = eval.get_waits_u8_with_rule(rule);
                if !waits.is_empty() {
                    post_tenpai[b] = true;
                    // Recompute exact visible ukeire over rule-valid waits
                    let wait_ukeire: u32 = waits
                        .iter()
                        .map(|&w| u32::from(visible.unseen_base34[w as usize]))
                        .sum();
                    post_ukeire[b] = wait_ukeire;
                    if wait_ukeire > best_ukeire {
                        best_ukeire = wait_ukeire;
                    }
                    post_furiten[b] =
                        furiten.is_post_discard_furiten(&waits, &self_discarded_ever, Some(b as u8));
                }
            }
        }

        let best_u_denom = best_ukeire.max(1) as f32;
        for b in 0..TILE_MAX {
            if !legal_discard_base34[b] || post_shanten[b] == i8::MAX {
                continue;
            }
            set_ch(&mut out, b, 54, 1.0);
            let gap = post_shanten[b] - min_shanten;
            if gap == 0 {
                set_ch(&mut out, b, 55, 1.0);
            } else if gap == 1 {
                set_ch(&mut out, b, 56, 1.0);
            }
            set_ch(&mut out, b, 57, (f32::from(post_shanten[b]).max(0.0) / 8.0).min(1.0));
            set_ch(&mut out, b, 58, ((1.0 + post_ukeire[b] as f32).ln() / LN_61).min(1.0));
            set_ch(&mut out, b, 59, (post_ukeire[b] as f32 / best_u_denom).min(1.0));
            if post_tenpai[b] {
                set_ch(&mut out, b, 60, 1.0);
            }
            if post_furiten[b] {
                set_ch(&mut out, b, 61, 1.0);
            }
        }
    } else if phase != DecisionPhase::OwnAct && obs.hands[pid].len() % 3 == 1 {
        let hand_u8: Vec<u8> = obs.hands[pid].iter().map(|&t| t as u8).collect();
        let eval = HandEvaluator::new(hand_u8, obs.melds[pid].clone());
        let waits = eval.get_waits_u8_with_rule(rule);
        for &w in &waits {
            if (w as usize) < TILE_MAX {
                set_ch(&mut out, w as usize, 62, 1.0);
            }
        }
        let (_, _, effective) = visible.shanten_and_visible_ukeire(&visible.self_hand_base34);
        for (b, &is_eff) in effective.iter().enumerate() {
            if is_eff {
                set_ch(&mut out, b, 63, 1.0);
            }
        }
    }

    Ok(out)
}
