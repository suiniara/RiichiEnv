//! Structured 546-action candidate encoder with 7 mutually-exclusive `cand_mode`s
//! (Contract v2.4 Section 1.5).

use crate::hand_evaluator::HandEvaluator;
use crate::observation::Observation;
use crate::types::{Conditions, Meld, MeldType, TILE_MAX, Wind};

use super::action_manifest::{ManifestActionKind, ResolvedCandidate, resolve_legal_candidates};
use super::furiten::{FuritenState, own_discard_mask_u32};
use super::global_player::RoundGlobalContext;
use super::visible::VisibleTileCounts;
use super::{
    CAND_DYN_DIM, CAND_PAD_ID, DecisionPhase, HanchanTracker, MAX_CANDIDATES, RuleContract,
    dora_indicator_to_dora_base34, is_red5_tile136, seat_rank_and_gaps,
};

const LN_61: f32 = 4.110_873_8;
const LN_64001: f32 = 11.066_654;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandidateMode {
    Pad = 0,
    PostDiscardExact = 1,
    PreDiscardMeldEnvelope = 2,
    PendingRinshanKan = 3,
    PassCurrentState = 4,
    WinEval = 5,
    KyushuDrawEval = 6,
}

/// Fixed-size encoded candidate tensors for one decision step.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedCandidates {
    pub cand_ids: [i16; MAX_CANDIDATES],
    pub cand_mask: [bool; MAX_CANDIDATES],
    pub cand_mode: [u8; MAX_CANDIDATES],
    pub cand_dyn: [[f32; CAND_DYN_DIM]; MAX_CANDIDATES],
    pub num_valid: usize,
}

/// Optional win context for `WIN_EVAL` (unilateral single-win assumption, zero leakage).
#[derive(Debug, Clone, Copy, Default)]
pub struct WinEvalContext {
    pub is_rinshan: bool,
    pub is_chankan: bool,
    pub is_haitei_or_houtei: bool,
    pub is_tsumo_first_turn: bool,
    /// Publicly determined Pao liable seat (absolute `0..3`), if any.
    pub pao_payer: Option<u8>,
}

fn dora_counts_from_indicators(indicators: &[u32]) -> [u8; TILE_MAX] {
    let mut counts = [0u8; TILE_MAX];
    for &ind in indicators {
        let dora_b = dora_indicator_to_dora_base34((ind / 4) as u8) as usize;
        if dora_b < TILE_MAX {
            counts[dora_b] = counts[dora_b].saturating_add(1);
        }
    }
    counts
}

fn known_dora_count(hand_136: &[u8], melds: &[Meld], dora_counts: &[u8; TILE_MAX]) -> u32 {
    let mut total = 0u32;
    for &t in hand_136 {
        let b = (t / 4) as usize;
        if b < TILE_MAX {
            total += u32::from(dora_counts[b]);
        }
        if is_red5_tile136(t) {
            total += 1;
        }
    }
    for m in melds {
        for &t in &m.tiles {
            let b = (t / 4) as usize;
            if b < TILE_MAX {
                total += u32::from(dora_counts[b]);
            }
            if is_red5_tile136(t) {
                total += 1;
            }
        }
    }
    total
}

/// Exact kuikae-forbidden `base34` mask after executing a `CHI` or `PON`.
fn kuikae_forbidden_mask(
    kind: ManifestActionKind,
    claimed_base34: u8,
    c0_base34: u8,
    c1_base34: u8,
    kuikae_forbidden: bool,
) -> [bool; TILE_MAX] {
    let mut mask = [false; TILE_MAX];
    if !kuikae_forbidden {
        return mask;
    }
    if (claimed_base34 as usize) < TILE_MAX {
        mask[claimed_base34 as usize] = true;
    }
    if kind == ManifestActionKind::Chi && claimed_base34 < 27 {
        let suit = claimed_base34 / 9;
        let rc = (claimed_base34 % 9) + 1;
        let r0 = (c0_base34 % 9) + 1;
        let r1 = (c1_base34 % 9) + 1;
        let (lo, hi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
        // Two-sided sequence: consumed tiles are adjacent (hi - lo == 1)
        if hi == lo + 1 {
            if rc + 1 == lo && hi < 9 {
                mask[(suit * 9 + hi) as usize] = true;
            } else if rc == hi + 1 && lo > 1 {
                mask[(suit * 9 + (lo - 2)) as usize] = true;
            }
        }
    }
    mask
}

/// Encode up to 32 strictly sorted legal candidates with exact 7-mode `cand_dyn`.
#[allow(clippy::too_many_arguments)]
pub fn encode_candidates(
    obs: &Observation,
    visible: &VisibleTileCounts,
    furiten: FuritenState,
    tracker: &HanchanTracker,
    rule_contract: &RuleContract,
    round_ctx: &RoundGlobalContext,
    win_ctx: &WinEvalContext,
) -> Result<EncodedCandidates, String> {
    let pid = obs.player_id as usize;
    if pid >= 4 {
        return Err(format!("invalid observer player_id {}", obs.player_id));
    }

    let resolved: Vec<ResolvedCandidate> =
        resolve_legal_candidates(obs, round_ctx.trigger_actor_rel)?;

    let mut cand_ids = [CAND_PAD_ID; MAX_CANDIDATES];
    let mut cand_mask = [false; MAX_CANDIDATES];
    let mut cand_mode = [CandidateMode::Pad as u8; MAX_CANDIDATES];
    let mut cand_dyn = [[0.0f32; CAND_DYN_DIM]; MAX_CANDIDATES];

    let hand_u8: Vec<u8> = obs.hands[pid].iter().map(|&t| t as u8).collect();
    let dora_counts = dora_counts_from_indicators(&obs.dora_indicators);
    let self_discarded_ever = own_discard_mask_u32(&obs.discards[pid]);
    let cur_known_dora = known_dora_count(&hand_u8, &obs.melds[pid], &dora_counts);

    let total_kans: usize = obs
        .melds
        .iter()
        .map(|ms| {
            ms.iter()
                .filter(|m| {
                    matches!(
                        m.meld_type,
                        MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan
                    )
                })
                .count()
        })
        .sum();

    // Current hand baseline for 3k+1 (response phases) or 3k+2 (OwnAct).
    let (cur_shanten_3k1, cur_ukeire_3k1, cur_tenpai_3k1, cur_furiten_3k1) =
        if hand_u8.len() % 3 == 1 {
            let (s, u, _) = visible.shanten_and_visible_ukeire(&visible.self_hand_base34);
            let eval = HandEvaluator::new(hand_u8.clone(), obs.melds[pid].clone());
            let waits = eval.get_waits_u8_with_rule(&rule_contract.rule);
            let tenpai = !waits.is_empty();
            let u_exact = if tenpai {
                waits
                    .iter()
                    .map(|&w| u32::from(visible.unseen_base34[w as usize]))
                    .sum()
            } else {
                u
            };
            let is_fur = tenpai && furiten.is_post_discard_furiten(&waits, &self_discarded_ever, None);
            (s, u_exact, tenpai, is_fur)
        } else {
            let s = crate::shanten::calculate_shanten(&obs.hands[pid]) as i8;
            (s, 0, s == 0, furiten.any())
        };

    // Precompute best discard shanten & ukeire across all legal DISCARD candidates in OwnAct.
    let mut discard_stats: [Option<(i8, u32, bool, bool)>; TILE_MAX] = [None; TILE_MAX];
    let mut min_discard_shanten = i8::MAX;
    let mut best_discard_ukeire = 0u32;

    if round_ctx.phase == DecisionPhase::OwnAct && hand_u8.len() % 3 == 2 {
        for cand in &resolved {
            if cand.entry.kind == ManifestActionKind::Discard {
                let b = cand.entry.target_base34 as usize;
                if b < TILE_MAX && discard_stats[b].is_none() && visible.self_hand_base34[b] > 0 {
                    let mut next_counts = visible.self_hand_base34;
                    next_counts[b] -= 1;
                    let (s, mut u, _) = visible.shanten_and_visible_ukeire(&next_counts);
                    let mut tenpai = false;
                    let mut is_fur = false;
                    if s == 0 {
                        let mut rem_hand = hand_u8.clone();
                        if let Some(pos) = rem_hand.iter().position(|&t| (t / 4) as usize == b) {
                            rem_hand.remove(pos);
                        }
                        let eval = HandEvaluator::new(rem_hand, obs.melds[pid].clone());
                        let waits = eval.get_waits_u8_with_rule(&rule_contract.rule);
                        if !waits.is_empty() {
                            tenpai = true;
                            u = waits
                                .iter()
                                .map(|&w| u32::from(visible.unseen_base34[w as usize]))
                                .sum();
                            is_fur = furiten.is_post_discard_furiten(
                                &waits,
                                &self_discarded_ever,
                                Some(b as u8),
                            );
                        }
                    }
                    if s < min_discard_shanten {
                        min_discard_shanten = s;
                    }
                    if u > best_discard_ukeire {
                        best_discard_ukeire = u;
                    }
                    discard_stats[b] = Some((s, u, tenpai, is_fur));
                }
            }
        }
    }

    for (slot, cand) in resolved.iter().enumerate() {
        cand_ids[slot] = cand.global_id as i16;
        cand_mask[slot] = true;
        let dyn_row = &mut cand_dyn[slot];

        match cand.entry.kind {
            ManifestActionKind::Discard => {
                cand_mode[slot] = CandidateMode::PostDiscardExact as u8;
                let b = cand.entry.target_base34 as usize;
                let discard_t136 = cand
                    .native_actions
                    .last()
                    .and_then(|a| a.tile)
                    .unwrap_or((b * 4) as u8);
                let dropped_dora = u32::from(dora_counts[b])
                    + u32::from(is_red5_tile136(discard_t136));
                let post_dora = cur_known_dora.saturating_sub(dropped_dora);

                if let Some((s, u, tenpai, is_fur)) = discard_stats[b] {
                    dyn_row[0] = (f32::from(s).max(0.0) / 8.0).min(1.0);
                    dyn_row[1] = (f32::from(s - min_discard_shanten) / 4.0).clamp(-1.0, 2.0);
                    dyn_row[2] = ((1.0 + u as f32).ln() / LN_61).min(1.0);
                    dyn_row[3] = (u as f32 / best_discard_ukeire.max(1) as f32).min(1.0);
                    dyn_row[4] = if tenpai { 1.0 } else { 0.0 };
                    dyn_row[5] = if is_fur { 1.0 } else { 0.0 };
                    dyn_row[6] = (post_dora as f32 / 8.0).min(2.0);
                    dyn_row[7] = 0.0;
                }
            }

            ManifestActionKind::Chi | ManifestActionKind::Pon => {
                cand_mode[slot] = CandidateMode::PreDiscardMeldEnvelope as u8;
                let native = &cand.native_actions[0];
                let claimed_t136 = native.tile.unwrap_or(0);
                let mut post_hand = hand_u8.clone();
                for &ct in &native.consume_tiles {
                    if let Some(pos) = post_hand.iter().position(|&x| x == ct) {
                        post_hand.remove(pos);
                    }
                }
                let mut post_melds = obs.melds[pid].clone();
                let mut meld_tiles = native.consume_tiles.clone();
                meld_tiles.push(claimed_t136);
                meld_tiles.sort();
                post_melds.push(Meld {
                    meld_type: if cand.entry.kind == ManifestActionKind::Chi {
                        MeldType::Chi
                    } else {
                        MeldType::Pon
                    },
                    tiles: meld_tiles,
                    opened: true,
                    from_who: ((pid + cand.entry.source_rel as usize) % 4) as i8,
                    called_tile: Some(claimed_t136),
                });

                let kuikae = kuikae_forbidden_mask(
                    cand.entry.kind,
                    cand.entry.target_base34,
                    cand.entry.consumed0_base34,
                    cand.entry.consumed1_base34,
                    rule_contract.rule.kuikae_forbidden,
                );

                let mut post_counts = [0u8; TILE_MAX];
                let mut blocked_tiles = 0usize;
                for &t in &post_hand {
                    let b = (t / 4) as usize;
                    post_counts[b] += 1;
                    if kuikae[b] {
                        blocked_tiles += 1;
                    }
                }

                // Evaluate envelope over ALL legal post-meld discards D_legal(h')
                let mut best_s = i8::MAX;
                let mut per_discard: Vec<(usize, i8, u32, bool)> = Vec::new();
                for b in 0..TILE_MAX {
                    if post_counts[b] == 0 || kuikae[b] {
                        continue;
                    }
                    let mut after_d = post_counts;
                    after_d[b] -= 1;
                    let (s, mut u, _) = visible.shanten_and_visible_ukeire(&after_d);
                    let mut is_fur = false;
                    if s == 0 {
                        let mut rem = post_hand.clone();
                        if let Some(pos) = rem.iter().position(|&t| (t / 4) as usize == b) {
                            rem.remove(pos);
                        }
                        let eval = HandEvaluator::new(rem, post_melds.clone());
                        let waits = eval.get_waits_u8_with_rule(&rule_contract.rule);
                        if !waits.is_empty() {
                            u = waits
                                .iter()
                                .map(|&w| u32::from(visible.unseen_base34[w as usize]))
                                .sum();
                            // Temporary furiten carries forward across CHI/PON!
                            is_fur = furiten.is_post_discard_furiten(
                                &waits,
                                &self_discarded_ever,
                                Some(b as u8),
                            );
                        } else {
                            // Formal-tenpai rejected (e.g. MJS 5th-copy exclusion)
                            continue;
                        }
                    }
                    if s < best_s {
                        best_s = s;
                    }
                    per_discard.push((b, s, u, is_fur));
                }

                let post_dora = known_dora_count(&post_hand, &post_melds, &dora_counts);
                if best_s < i8::MAX {
                    let opt: Vec<_> = per_discard.iter().filter(|x| x.1 == best_s).collect();
                    let max_u = opt.iter().map(|x| x.2).max().unwrap_or(0);
                    let can_tenpai = best_s == 0;
                    let all_tenpai_furiten = can_tenpai && opt.iter().all(|x| x.3);

                    dyn_row[0] = (f32::from(best_s).max(0.0) / 8.0).min(1.0);
                    dyn_row[1] = (f32::from(best_s - cur_shanten_3k1) / 4.0).clamp(-2.0, 2.0);
                    dyn_row[2] = ((1.0 + max_u as f32).ln() / LN_61).min(1.0);
                    dyn_row[3] = (opt.len() as f32 / 14.0).min(1.0);
                    dyn_row[4] = if can_tenpai { 1.0 } else { 0.0 };
                    dyn_row[5] = if all_tenpai_furiten { 1.0 } else { 0.0 };
                    dyn_row[6] = (post_dora as f32 / 8.0).min(2.0);
                    dyn_row[7] = blocked_tiles as f32 / post_hand.len().max(1) as f32;
                }
            }

            ManifestActionKind::Ankan
            | ManifestActionKind::Kakan
            | ManifestActionKind::Daiminkan => {
                cand_mode[slot] = CandidateMode::PendingRinshanKan as u8;
                // Contract v2.4 P1-3: never fabricate post-rinshan ukeire; explicitly flag Chankan risk.
                let chankan_possible = match cand.entry.kind {
                    ManifestActionKind::Kakan => true,
                    ManifestActionKind::Ankan => rule_contract.rule.allows_ron_on_ankan_for_kokushi_musou,
                    ManifestActionKind::Daiminkan => false,
                    _ => false,
                };
                let extra_dora = if cand.entry.kind == ManifestActionKind::Daiminkan {
                    let t = cand.native_actions[0].tile.unwrap_or(0);
                    u32::from(dora_counts[(t / 4) as usize]) + u32::from(is_red5_tile136(t))
                } else {
                    0
                };
                dyn_row[0] = (f32::from(cur_shanten_3k1).max(0.0) / 8.0).min(1.0);
                dyn_row[1] = 0.0;
                dyn_row[2] = 0.0;
                dyn_row[3] = 0.0;
                dyn_row[4] = if cur_tenpai_3k1 { 1.0 } else { 0.0 };
                dyn_row[5] = if chankan_possible { 1.0 } else { 0.0 };
                dyn_row[6] = ((cur_known_dora + extra_dora) as f32 / 8.0).min(2.0);
                dyn_row[7] = ((total_kans + 1) as f32 / 4.0).min(1.0);
            }

            ManifestActionKind::Pass => {
                cand_mode[slot] = CandidateMode::PassCurrentState as u8;
                dyn_row[0] = (f32::from(cur_shanten_3k1).max(0.0) / 8.0).min(1.0);
                dyn_row[1] = 0.0;
                dyn_row[2] = ((1.0 + cur_ukeire_3k1 as f32).ln() / LN_61).min(1.0);
                dyn_row[3] = 1.0;
                dyn_row[4] = if cur_tenpai_3k1 { 1.0 } else { 0.0 };
                dyn_row[5] = if cur_furiten_3k1 { 1.0 } else { 0.0 };
                dyn_row[6] = (cur_known_dora as f32 / 8.0).min(2.0);
                dyn_row[7] = 0.0;
            }

            ManifestActionKind::Tsumo | ManifestActionKind::Ron => {
                cand_mode[slot] = CandidateMode::WinEval as u8;
                let is_tsumo = cand.entry.kind == ManifestActionKind::Tsumo;
                let win_tile = if is_tsumo {
                    obs.drawn_tile
                        .or_else(|| hand_u8.last().copied())
                        .unwrap_or(0)
                } else {
                    obs.last_discard.map(|x| x as u8).unwrap_or(0)
                };
                let seat_wind = Wind::from((obs.player_id + 4 - obs.oya) % 4);
                let cond = Conditions {
                    tsumo: is_tsumo,
                    riichi: obs.riichi_declared[pid],
                    double_riichi: round_ctx.double_riichi_declared[pid],
                    ippatsu: round_ctx.ippatsu_active[pid],
                    haitei: is_tsumo && win_ctx.is_haitei_or_houtei,
                    houtei: !is_tsumo && win_ctx.is_haitei_or_houtei,
                    rinshan: win_ctx.is_rinshan,
                    chankan: win_ctx.is_chankan,
                    tsumo_first_turn: win_ctx.is_tsumo_first_turn,
                    player_wind: seat_wind,
                    round_wind: Wind::from(obs.round_wind),
                    honba: u32::from(obs.honba),
                    ..Default::default()
                };
                let dora_u8: Vec<u8> = obs.dora_indicators.iter().map(|&x| x as u8).collect();
                let eval = HandEvaluator::new(hand_u8.clone(), obs.melds[pid].clone());
                // Strictly empty uradora (zero leakage) + rule-aware evaluation!
                let win_res = eval.calc_with_rule(
                    win_tile,
                    &dora_u8,
                    &[],
                    Some(cond),
                    &rule_contract.rule,
                );

                // Contract v2.4 P1-4: Unilateral single-win settlement (honba + full kyoutaku + public Pao)
                let mut post_scores = obs.scores;
                let kyoutaku_pts = (obs.riichi_sticks * 1000) as i32;
                let honba_total = i32::from(obs.honba) * 300;
                let is_oya = pid == obs.oya as usize;

                let gain: i32 = if is_tsumo {
                    let tsumo_oya = win_res.tsumo_agari_oya as i32;
                    let tsumo_ko = win_res.tsumo_agari_ko as i32;
                    if let Some(pao_seat) = win_ctx.pao_payer {
                        // Public Pao tsumo: liable payer pays full amount + honba
                        let base_pay = if is_oya {
                            tsumo_ko * 3
                        } else {
                            tsumo_oya + tsumo_ko * 2
                        };
                        let total_pay = base_pay + honba_total;
                        post_scores[pao_seat as usize] -= total_pay;
                        post_scores[pid] += total_pay + kyoutaku_pts;
                        total_pay + kyoutaku_pts
                    } else {
                        let mut sum = 0i32;
                        for (seat, score_slot) in post_scores.iter_mut().enumerate() {
                            if seat == pid {
                                continue;
                            }
                            let pay = if is_oya {
                                tsumo_ko
                            } else if seat == obs.oya as usize {
                                tsumo_oya
                            } else {
                                tsumo_ko
                            } + i32::from(obs.honba) * 100;
                            *score_slot -= pay;
                            sum += pay;
                        }
                        post_scores[pid] += sum + kyoutaku_pts;
                        sum + kyoutaku_pts
                    }
                } else {
                    let loser = (pid + cand.entry.source_rel as usize) % 4;
                    let base_pay = win_res.ron_agari as i32;
                    if let Some(pao_seat) = win_ctx.pao_payer {
                        if pao_seat as usize == loser {
                            let total = base_pay + honba_total;
                            post_scores[loser] -= total;
                            post_scores[pid] += total + kyoutaku_pts;
                            total + kyoutaku_pts
                        } else {
                            let half = base_pay / 2;
                            post_scores[pao_seat as usize] -= half;
                            post_scores[loser] -= (base_pay - half) + honba_total;
                            post_scores[pid] += base_pay + honba_total + kyoutaku_pts;
                            base_pay + honba_total + kyoutaku_pts
                        }
                    } else {
                        let total = base_pay + honba_total;
                        post_scores[loser] -= total;
                        post_scores[pid] += total + kyoutaku_pts;
                        total + kyoutaku_pts
                    }
                };

                let (post_rank, _, _) =
                    seat_rank_and_gaps(&post_scores, &tracker.initial_seat_order, obs.player_id);
                let any_tobi = rule_contract.has_tobi_end && post_scores.iter().any(|&s| s < 0);
                let last_wind = if rule_contract.is_tonpuusen { 0u8 } else { 1u8 };
                let is_all_last = obs.round_wind == last_wind && (obs.kyoku_index % 4) == 3;
                let top_ge_30k = post_scores.iter().any(|&s| s >= 30_000);
                let ends_by_all_last = (is_all_last || obs.round_wind > last_wind)
                    && top_ge_30k
                    && (!is_oya || (rule_contract.has_agari_yame && post_rank == 1));
                let would_end_hanchan = any_tobi || ends_by_all_last;

                dyn_row[0] = (win_res.han as f32 / 13.0).min(2.0);
                dyn_row[1] = (win_res.fu as f32 / 110.0).min(1.5);
                dyn_row[2] = ((1.0 + gain.max(0) as f32).ln() / LN_64001).min(1.5);
                dyn_row[3] = if win_res.yakuman { 1.0 } else { 0.0 };
                dyn_row[4] = f32::from(post_rank - 1) / 3.0;
                dyn_row[5] = if would_end_hanchan { 1.0 } else { 0.0 };
                dyn_row[6] = if obs.riichi_declared[pid] { 1.0 } else { 0.0 };
                dyn_row[7] = 0.0;
            }

            ManifestActionKind::KyushuKyuhai => {
                cand_mode[slot] = CandidateMode::KyushuDrawEval as u8;
                let distinct_yaochuu = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
                    .iter()
                    .filter(|&&b| visible.self_hand_base34[b] > 0)
                    .count();
                let (cur_rank, _, _) =
                    seat_rank_and_gaps(&obs.scores, &tracker.initial_seat_order, obs.player_id);
                dyn_row[0] = (f32::from(cur_shanten_3k1).max(0.0) / 8.0).min(1.0);
                dyn_row[1] = distinct_yaochuu as f32 / 13.0;
                dyn_row[2] = (cur_known_dora as f32 / 8.0).min(2.0);
                dyn_row[3] = if pid == obs.oya as usize { 1.0 } else { 0.0 };
                dyn_row[4] = f32::from(cur_rank - 1) / 3.0;
                dyn_row[5] = 0.0;
                dyn_row[6] = 0.0;
                dyn_row[7] = 0.0;
            }

            ManifestActionKind::Pad => {}
        }
    }

    Ok(EncodedCandidates {
        cand_ids,
        cand_mask,
        cand_mode,
        cand_dyn,
        num_valid: resolved.len(),
    })
}
