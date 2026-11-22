//! KAR V2 Hybrid Architecture Rust Feature Engine (Frozen Contract v2.4).
//!
//! All actor-visible features strictly consume `&Observation` and obey physical
//! tile deduplication (`VisibleTileCounts`), full furiten tracking
//! (`FuritenState`), rule-aware evaluation (`HandEvaluator::calc_with_rule`),
//! lossless observer-visible round events (`1 <= L_ev <= 256`), and exact
//! 546-action bijective candidate mapping (`MAX_CANDIDATES = 32`).
//!
//! Oracle critic features live in [`oracle`] and accept ONLY `&GameState`
//! (compile-time isolation from `&Observation`).

pub mod action_manifest;
pub mod candidates;
pub mod events;
pub mod furiten;
pub mod global_player;
pub mod oracle;
pub mod tile_feats;
pub mod visible;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::rule::GameRule;

/// Contract tensor dimensions (KAR V2.4 Frozen).
pub const TILE_DIM: usize = 34;
pub const TILE_FEATS_DIM: usize = 64;
pub const EVENT_BYTES: usize = 10;
pub const MAX_EVENT_LEN: usize = 256;
pub const GLOBAL_FEATS_DIM: usize = 48;
pub const PLAYER_FEATS_DIM: usize = 32;
pub const ACTION_COUNT: usize = 546;
pub const MAX_CANDIDATES: usize = 32;
pub const CAND_PAD_ID: i16 = 546;
pub const CAND_MODE_COUNT: usize = 7;
pub const CAND_DYN_DIM: usize = 8;
pub const ORACLE_TILE_DIM: usize = 32;
pub const ORACLE_PLAYER_DIM: usize = 12;
pub const NULL_TILE_IDX: u8 = 34;
pub const NULL_SOURCE_IDX: u8 = 4;

/// Configuration knobs for V2 feature extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct V2FeatureConfig {
    /// Contract v2.4 P1-2: `opp_early_discard_shape_hint` defaults to `false`
    /// (channel filled with `0.0`) until validated by ablation.
    pub enable_early_shape_hint: bool,
}

/// Decision phase for the current observer step.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum DecisionPhase {
    #[default]
    OwnAct = 0,
    DiscardResp = 1,
    KakanResp = 2,
    AnkanResp = 3,
}

/// Full ruleset & placement payoff contract bound to every shard/checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleContract {
    pub ruleset_id: String,
    pub rule: GameRule,
    pub is_tonpuusen: bool,
    pub has_aka_dora: bool,
    pub has_kuitan: bool,
    pub has_tobi_end: bool,
    pub has_agari_yame: bool,
    pub has_abortive_draws: bool,
    pub has_nagashi_mangan: bool,
    /// Normalized 4-placement payoff vector w = (w_1, w_2, w_3, w_4).
    pub rank_payoffs: [f32; 4],
}

impl RuleContract {
    pub fn default_tenhou_houou() -> Self {
        Self {
            ruleset_id: "tenhou_houou_4p_hanchan_v1".to_string(),
            rule: GameRule::default_tenhou(),
            is_tonpuusen: false,
            has_aka_dora: true,
            has_kuitan: true,
            has_tobi_end: true,
            has_agari_yame: true,
            has_abortive_draws: true,
            has_nagashi_mangan: true,
            rank_payoffs: [0.90, 0.45, 0.0, -1.35],
        }
    }

    pub fn default_mjsoul_throne() -> Self {
        Self {
            ruleset_id: "mjsoul_throne_4p_hanchan_v1".to_string(),
            rule: GameRule::default_mjsoul(),
            is_tonpuusen: false,
            has_aka_dora: true,
            has_kuitan: true,
            has_tobi_end: true,
            has_agari_yame: true,
            has_abortive_draws: true,
            has_nagashi_mangan: true,
            rank_payoffs: [1.25, 0.60, -0.05, -1.80],
        }
    }

    /// Deterministic SHA-256 hash of the entire rule contract.
    pub fn ruleset_sha256(&self) -> String {
        let payload = serde_json::to_vec(self).expect("RuleContract serialization is infallible");
        let mut hasher = Sha256::new();
        hasher.update(payload);
        format!("{:x}", hasher.finalize())
    }
}

impl Default for RuleContract {
    fn default() -> Self {
        Self::default_tenhou_houou()
    }
}

/// Summary of one completed round, used to update [`HanchanTracker`] in O(1).
#[derive(Debug, Clone, Copy, Default)]
pub struct RoundOutcomeSummary {
    pub score_deltas: [i32; 4],
    pub won: [bool; 4],
    pub dealt_in: [bool; 4],
    pub declared_riichi: [bool; 4],
    pub opened_meld: [bool; 4],
}

/// O(1) cross-round Hanchan behavioral statistics per absolute seat.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HanchanTracker {
    pub initial_seat_order: [u8; 4],
    pub rounds_completed: u32,
    pub riichi_rounds: [u32; 4],
    pub open_rounds: [u32; 4],
    pub win_rounds: [u32; 4],
    pub dealin_rounds: [u32; 4],
    pub win_pts_sum: [i64; 4],
    pub dealin_pts_sum: [i64; 4],
    pub prev_round_delta: [i32; 4],
    pub prev_round_won: [bool; 4],
    pub prev_round_dealin: [bool; 4],
    pub prev_round_riichi: [bool; 4],
    pub prev_round_opened: [bool; 4],
}

impl Default for HanchanTracker {
    fn default() -> Self {
        Self::new([0, 1, 2, 3])
    }
}

impl HanchanTracker {
    pub fn new(initial_seat_order: [u8; 4]) -> Self {
        Self {
            initial_seat_order,
            rounds_completed: 0,
            riichi_rounds: [0; 4],
            open_rounds: [0; 4],
            win_rounds: [0; 4],
            dealin_rounds: [0; 4],
            win_pts_sum: [0; 4],
            dealin_pts_sum: [0; 4],
            prev_round_delta: [0; 4],
            prev_round_won: [false; 4],
            prev_round_dealin: [false; 4],
            prev_round_riichi: [false; 4],
            prev_round_opened: [false; 4],
        }
    }

    pub fn record_round_end(&mut self, summary: &RoundOutcomeSummary) {
        self.rounds_completed += 1;
        for seat in 0..4 {
            if summary.declared_riichi[seat] {
                self.riichi_rounds[seat] += 1;
            }
            if summary.opened_meld[seat] {
                self.open_rounds[seat] += 1;
            }
            if summary.won[seat] {
                self.win_rounds[seat] += 1;
                if summary.score_deltas[seat] > 0 {
                    self.win_pts_sum[seat] += summary.score_deltas[seat] as i64;
                }
            }
            if summary.dealt_in[seat] {
                self.dealin_rounds[seat] += 1;
                if summary.score_deltas[seat] < 0 {
                    self.dealin_pts_sum[seat] += (-summary.score_deltas[seat]) as i64;
                }
            }
        }
        self.prev_round_delta = summary.score_deltas;
        self.prev_round_won = summary.won;
        self.prev_round_dealin = summary.dealt_in;
        self.prev_round_riichi = summary.declared_riichi;
        self.prev_round_opened = summary.opened_meld;
    }
}

/// Convert physical `tile136` (0..135) to canonical `tile37` (0..36).
/// Red 5m/5p/5s (`16/52/88`) map to `0/10/20`.
#[inline]
pub fn tile136_to_tile37(tile136: u8) -> u8 {
    debug_assert!(tile136 < 136);
    if tile136 < 108 {
        let suit = tile136 / 36;
        let offset = tile136 % 36;
        if offset == 16 {
            suit * 10
        } else {
            suit * 10 + offset / 4 + 1
        }
    } else {
        30 + (tile136 - 108) / 4
    }
}

/// Convert `tile37` (0..36) to `base34` (0..33).
#[inline]
pub fn tile37_to_base34(tile37: u8) -> u8 {
    debug_assert!(tile37 < 37);
    if tile37 >= 30 {
        tile37 - 3
    } else {
        let suit = tile37 / 10;
        let rank = tile37 % 10;
        suit * 9 + if rank == 0 { 4 } else { rank - 1 }
    }
}

#[inline]
pub fn is_red5_tile136(tile136: u8) -> bool {
    tile136 == 16 || tile136 == 52 || tile136 == 88
}

#[inline]
pub fn is_red5_tile37(tile37: u8) -> bool {
    tile37 == 0 || tile37 == 10 || tile37 == 20
}

/// Map a dora indicator `base34` to its actual dora `base34`.
#[inline]
pub fn dora_indicator_to_dora_base34(indicator_base34: u8) -> u8 {
    if indicator_base34 < 27 {
        let suit_start = (indicator_base34 / 9) * 9;
        suit_start + ((indicator_base34 % 9) + 1) % 9
    } else if indicator_base34 < 31 {
        27 + ((indicator_base34 - 27) + 1) % 4
    } else {
        31 + ((indicator_base34 - 31) + 1) % 3
    }
}

/// Rank tie-breaking matching `rule_features.current_rank_features`.
/// Returns `(rank_1_to_4, gap_to_higher_pts, gap_to_lower_pts)` for `seat`.
pub fn seat_rank_and_gaps(
    scores: &[i32; 4],
    initial_seat_order: &[u8; 4],
    seat: u8,
) -> (u8, i32, i32) {
    let mut ranking = *initial_seat_order;
    ranking.sort_by(|&a, &b| {
        let ord = scores[b as usize].cmp(&scores[a as usize]);
        if ord != std::cmp::Ordering::Equal {
            ord
        } else {
            let ia = initial_seat_order.iter().position(|&x| x == a).unwrap_or(0);
            let ib = initial_seat_order.iter().position(|&x| x == b).unwrap_or(0);
            ia.cmp(&ib)
        }
    });
    let pos = ranking.iter().position(|&x| x == seat).unwrap_or(0);
    let own = scores[seat as usize];
    let gap_higher = if pos > 0 {
        scores[ranking[pos - 1] as usize] - own
    } else {
        0
    };
    let gap_lower = if pos < 3 {
        own - scores[ranking[pos + 1] as usize]
    } else {
        0
    };
    ((pos + 1) as u8, gap_higher, gap_lower)
}
