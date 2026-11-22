//! Full furiten state contract (Contract v2.4 P1-2 & Stage-1 Step 2).
//!
//! Captures all three Japanese Riichi Mahjong furiten conditions and is carried
//! directly on frozen [`Observation`] from the Rust [`GameState`] machine:
//! - `permanent`: own river contains any current winning wait (including called discards)
//! - `temporary`: declined a legal ron and have not yet drawn a new tile (persists across chi/pon!)
//! - `riichi`: declined a legal ron after declaring riichi (permanent for the round)

#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::hand_evaluator::HandEvaluator;
use crate::observation::Observation;
use crate::rule::GameRule;
use crate::state::GameState;
use crate::types::TILE_MAX;

/// Complete furiten state for the observer, preserved on frozen [`Observation`].
#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", get_all, eq, from_py_object)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FuritenState {
    pub permanent: bool,
    pub temporary: bool,
    pub riichi: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl FuritenState {
    #[new]
    #[pyo3(signature = (permanent=false, temporary=false, riichi=false))]
    pub fn py_new(permanent: bool, temporary: bool, riichi: bool) -> Self {
        Self {
            permanent,
            temporary,
            riichi,
        }
    }

    #[pyo3(name = "any")]
    pub fn py_any(&self) -> bool {
        self.any()
    }

    #[pyo3(name = "non_river_active")]
    pub fn py_non_river_active(&self) -> bool {
        self.non_river_active()
    }

    fn __repr__(&self) -> String {
        format!(
            "FuritenState(permanent={}, temporary={}, riichi={})",
            self.permanent, self.temporary, self.riichi
        )
    }
}

impl FuritenState {
    #[inline]
    pub fn any(self) -> bool {
        self.permanent || self.temporary || self.riichi
    }

    #[inline]
    pub fn non_river_active(self) -> bool {
        self.temporary || self.riichi
    }

    #[inline]
    pub fn to_bytes(self) -> [u8; 3] {
        [
            u8::from(self.permanent),
            u8::from(self.temporary),
            u8::from(self.riichi),
        ]
    }

    /// Extract exact [`FuritenState`] from live [`GameState`] for `player_id`.
    pub fn from_game_state(state: &GameState, player_id: u8) -> Self {
        let pid = player_id as usize;
        let player = &state.players[pid];
        let own_discards = own_discard_mask_u8(&player.discards);

        let permanent = if player.hand.len() % 3 == 1 {
            let eval = HandEvaluator::new(player.hand.clone(), player.melds.clone());
            let waits = eval.get_waits_u8_with_rule(&state.rule);
            waits
                .iter()
                .any(|&w| (w as usize) < TILE_MAX && own_discards[w as usize])
        } else {
            false
        };

        Self {
            permanent,
            temporary: player.missed_agari_doujun,
            riichi: player.missed_agari_riichi,
        }
    }

    /// Derive river-based `permanent` furiten from an [`Observation`] combined
    /// with explicit non-river `temporary`/`riichi` flags.
    pub fn from_observation(
        obs: &Observation,
        temporary: bool,
        riichi: bool,
        rule: &GameRule,
    ) -> Self {
        let pid = obs.player_id as usize;
        let own_discards = own_discard_mask_u32(&obs.discards[pid]);
        let permanent = if obs.hands[pid].len() % 3 == 1 {
            let hand_u8: Vec<u8> = obs.hands[pid].iter().map(|&t| t as u8).collect();
            let eval = HandEvaluator::new(hand_u8, obs.melds[pid].clone());
            let waits = eval.get_waits_u8_with_rule(rule);
            waits
                .iter()
                .any(|&w| (w as usize) < TILE_MAX && own_discards[w as usize])
        } else {
            false
        };
        Self {
            permanent,
            temporary,
            riichi,
        }
    }

    /// Evaluate whether a post-discard (or post-meld-discard) tenpai wait set is
    /// in furiten, carrying forward active `temporary` and `riichi` furiten.
    #[inline]
    pub fn is_post_discard_furiten(
        self,
        waits_base34: &[u8],
        own_discards_base34: &[bool; TILE_MAX],
        discarded_base34: Option<u8>,
    ) -> bool {
        if waits_base34.is_empty() {
            return false;
        }
        if self.temporary || self.riichi {
            return true;
        }
        waits_base34.iter().any(|&w| {
            let idx = w as usize;
            idx < TILE_MAX && (own_discards_base34[idx] || discarded_base34 == Some(w))
        })
    }
}

pub fn own_discard_mask_u8(discards: &[u8]) -> [bool; TILE_MAX] {
    let mut mask = [false; TILE_MAX];
    for &t in discards {
        let base = (t / 4) as usize;
        if base < TILE_MAX {
            mask[base] = true;
        }
    }
    mask
}

pub fn own_discard_mask_u32(discards: &[u32]) -> [bool; TILE_MAX] {
    let mut mask = [false; TILE_MAX];
    for &t in discards {
        let base = (t / 4) as usize;
        if base < TILE_MAX {
            mask[base] = true;
        }
    }
    mask
}
