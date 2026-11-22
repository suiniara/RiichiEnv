//! Single source of truth for physical visible/unseen tile counts (Contract P0-2).
//!
//! Harmonized with remote commit `77ca96c` / `5de9ff3`:
//! - Exact 136-tile physical deduplication (`seen_136`)
//! - Strict source-discard validation for claimed meld tiles
//! - Replay-safe `Kakan` claimed-tile recovery via unique river intersection
//! - Algebraic verification `V_t = R_t + M_t - C_t + D_t` and `U_t = 4 - H_t - V_t`
//! - Exact visible-deducted `shanten_and_visible_ukeire` helper

use crate::observation::Observation;
use crate::shanten::calc_shanten_from_counts;
use crate::types::{MeldType, TILE_MAX};

/// Unified physical tile visibility state for an observer decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleTileCounts {
    /// Physical tiles (`0..135`) seen by observer (own hand + public table).
    pub seen_136: [bool; 136],
    /// Public visible counts per `base34` excluding own closed hand (`V_t = R_t + M_t - C_t + D_t`).
    pub visible_base34: [u8; TILE_MAX],
    /// Own closed hand counts per `base34` (`H_t`).
    pub self_hand_base34: [u8; TILE_MAX],
    /// Remaining unseen copies per `base34` (`U_t = 4 - H_t - V_t`).
    pub unseen_base34: [u8; TILE_MAX],
    /// Whether physical red-5 (`16=0m, 52=0p, 88=0s`) is still unseen anywhere.
    pub red5_unseen: [bool; 3],
}

impl VisibleTileCounts {
    /// Build and verify exact physical tile counts from an [`Observation`].
    pub fn from_observation(observation: &Observation) -> Result<Self, String> {
        let observer = usize::from(observation.player_id);
        if observer >= 4 {
            return Err(format!(
                "invalid observer player id: {}",
                observation.player_id
            ));
        }

        let mut seen_136 = [false; 136];
        let mut hand_seen = [false; 136];
        let mut discard_seen = [false; 136];
        let mut meld_seen = [false; 136];
        let mut dora_seen = [false; 136];
        let mut called_from_river = [false; 136];

        // 1. Own closed hand H_t (opponent concealed hands are strictly ignored)
        for &raw in &observation.hands[observer] {
            let tile = usize::try_from(raw).map_err(|_| "invalid concealed tile id".to_string())?;
            if tile >= 136 {
                return Err(format!("concealed tile id out of range: {tile}"));
            }
            if hand_seen[tile] {
                return Err(format!("duplicate physical tile in observer hand: {tile}"));
            }
            hand_seen[tile] = true;
            seen_136[tile] = true;
        }

        // 2. Public rivers R_t
        let mut river_counts = [0u8; TILE_MAX];
        for discards in &observation.discards {
            for &raw in discards {
                let tile =
                    usize::try_from(raw).map_err(|_| "invalid discard tile id".to_string())?;
                if tile >= 136 {
                    return Err(format!("discard tile id out of range: {tile}"));
                }
                if discard_seen[tile] {
                    return Err(format!(
                        "duplicate physical tile in discard records: {tile}"
                    ));
                }
                if hand_seen[tile] {
                    return Err(format!(
                        "physical tile appears in both hand and discard: {tile}"
                    ));
                }
                discard_seen[tile] = true;
                seen_136[tile] = true;
                river_counts[tile / 4] += 1;
            }
        }

        // 3. Identify claimed tiles transferred from rivers into melds (C_t)
        let mut called_counts = [0u8; TILE_MAX];
        for melds in &observation.melds {
            for meld in melds {
                let called = if let Some(called) = meld.called_tile {
                    Some(usize::from(called))
                } else if meld.meld_type == MeldType::Kakan {
                    // Replay formats may omit the upgraded Pon's called_tile;
                    // recover it from the unique physical intersection with rivers.
                    let matches: Vec<usize> = meld
                        .tiles
                        .iter()
                        .map(|&t| usize::from(t))
                        .filter(|&t| t < 136 && discard_seen[t])
                        .collect();
                    match matches.as_slice() {
                        [tile] => Some(*tile),
                        [] => {
                            return Err(
                                "Kakan has no recoverable claimed physical tile".to_string()
                            );
                        }
                        _ => {
                            return Err("Kakan has multiple claimed physical tiles".to_string());
                        }
                    }
                } else {
                    None
                };

                if let Some(called) = called {
                    if called >= 136 {
                        return Err(format!("called tile id out of range: {called}"));
                    }
                    if !meld.tiles.iter().any(|&t| usize::from(t) == called) {
                        return Err(format!("called tile {called} is absent from its meld"));
                    }
                    if !discard_seen[called] {
                        return Err(format!("called tile {called} has no public source discard"));
                    }
                    if called_from_river[called] {
                        return Err(format!("physical tile called more than once: {called}"));
                    }
                    called_from_river[called] = true;
                    called_counts[called / 4] += 1;
                }
            }
        }

        // 4. All melds M_t
        let mut meld_counts = [0u8; TILE_MAX];
        for melds in &observation.melds {
            for meld in melds {
                for &raw in &meld.tiles {
                    let tile = usize::from(raw);
                    if tile >= 136 {
                        return Err(format!("meld tile id out of range: {tile}"));
                    }
                    if hand_seen[tile] {
                        return Err(format!("physical tile appears in hand and meld: {tile}"));
                    }
                    if meld_seen[tile] {
                        return Err(format!("duplicate physical tile across melds: {tile}"));
                    }
                    if discard_seen[tile] && !called_from_river[tile] {
                        return Err(format!(
                            "meld/discard overlap is not a claimed tile: {tile}"
                        ));
                    }
                    meld_seen[tile] = true;
                    seen_136[tile] = true;
                    meld_counts[tile / 4] += 1;
                }
            }
        }

        // 5. Revealed dora indicators D_t
        let mut dora_counts = [0u8; TILE_MAX];
        for &raw in &observation.dora_indicators {
            let tile =
                usize::try_from(raw).map_err(|_| "invalid dora indicator tile id".to_string())?;
            if tile >= 136 {
                return Err(format!("dora indicator tile id out of range: {tile}"));
            }
            if hand_seen[tile] || discard_seen[tile] || meld_seen[tile] || dora_seen[tile] {
                return Err(format!(
                    "physical dora indicator {tile} duplicates another visible/hand tile"
                ));
            }
            dora_seen[tile] = true;
            seen_136[tile] = true;
            dora_counts[tile / 4] += 1;
        }

        // 6. Aggregate physical bitmap & cross-check V_t = R_t + M_t - C_t + D_t
        let mut self_hand_base34 = [0u8; TILE_MAX];
        let mut visible_base34 = [0u8; TILE_MAX];
        let mut unseen_base34 = [0u8; TILE_MAX];

        for tile in 0..136 {
            let base = tile / 4;
            if hand_seen[tile] {
                self_hand_base34[base] += 1;
            } else if seen_136[tile] {
                visible_base34[base] += 1;
            }
        }

        for base in 0..TILE_MAX {
            let alg_v = i16::from(river_counts[base]) + i16::from(meld_counts[base])
                - i16::from(called_counts[base])
                + i16::from(dora_counts[base]);
            if alg_v != i16::from(visible_base34[base]) {
                return Err(format!(
                    "visible count algebra mismatch on base34 {base}: R+M-C+D={alg_v} vs bitmap={}",
                    visible_base34[base]
                ));
            }
            let total = self_hand_base34[base] + visible_base34[base];
            if total > 4 {
                return Err(format!(
                    "physical conservation violated on base34 {base}: H={}+V={} > 4",
                    self_hand_base34[base], visible_base34[base]
                ));
            }
            unseen_base34[base] = 4 - total;
        }

        let red5_unseen = [!seen_136[16], !seen_136[52], !seen_136[88]];

        Ok(Self {
            seen_136,
            visible_base34,
            self_hand_base34,
            unseen_base34,
            red5_unseen,
        })
    }

    /// Encode into the canonical 241-byte binary layout (commit `5de9ff3`):
    /// `seen_136[136] || visible_base34[34] || self_hand_base34[34] || unseen_base34[34] || red5_unseen[3]`.
    pub fn to_bytes_241(&self) -> [u8; 241] {
        let mut out = [0u8; 241];
        for (i, &v) in self.seen_136.iter().enumerate() {
            out[i] = u8::from(v);
        }
        out[136..170].copy_from_slice(&self.visible_base34);
        out[170..204].copy_from_slice(&self.self_hand_base34);
        out[204..238].copy_from_slice(&self.unseen_base34);
        for (i, &v) in self.red5_unseen.iter().enumerate() {
            out[238 + i] = u8::from(v);
        }
        out
    }

    /// Public visible count for `base34` (returns `None` if out of bounds).
    #[inline]
    pub fn visible_count(&self, base34: usize) -> Option<u8> {
        self.visible_base34.get(base34).copied()
    }

    /// Remaining unseen count for `base34` (returns `None` if out of bounds).
    #[inline]
    pub fn unseen_count(&self, base34: usize) -> Option<u8> {
        self.unseen_base34.get(base34).copied()
    }

    /// Total known copies of `base34` (own closed hand + public table: `H_t + V_t`).
    #[inline]
    pub fn total_known(&self, base34: usize) -> u8 {
        self.self_hand_base34[base34] + self.visible_base34[base34]
    }

    /// Compute `(shanten, visible_ukeire, effective_mask)` for a `3k+1` concealed
    /// hand count array using exact `unseen_base34` counts.
    pub fn shanten_and_visible_ukeire(
        &self,
        hand_counts_3k1: &[u8; TILE_MAX],
    ) -> (i8, u32, [bool; TILE_MAX]) {
        let num_tiles: u8 = hand_counts_3k1.iter().sum();
        debug_assert_eq!(num_tiles % 3, 1, "expected 3k+1 concealed hand");
        let len_div3 = num_tiles / 3;
        let base_shanten = calc_shanten_from_counts(hand_counts_3k1, len_div3);

        let mut next = *hand_counts_3k1;
        let mut effective = [false; TILE_MAX];
        let mut ukeire = 0u32;

        for tile in 0..TILE_MAX {
            if next[tile] >= 4 {
                continue;
            }
            next[tile] += 1;
            let s = calc_shanten_from_counts(&next, len_div3);
            next[tile] -= 1;
            if s < base_shanten {
                effective[tile] = true;
                ukeire += u32::from(self.unseen_base34[tile]);
            }
        }

        (base_shanten, ukeire, effective)
    }
}

#[cfg(test)]
mod remote_parity_tests {
    use super::*;
    use crate::types::{Meld, MeldType};

    fn observation() -> Observation {
        Observation::new(
            0,
            [vec![0, 1, 16], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25_000; 4],
            [false; 4],
            vec![],
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
        )
    }

    #[test]
    fn counts_called_tile_once_across_river_and_meld() {
        let mut obs = observation();
        obs.discards[1].push(4);
        obs.melds[0].push(Meld::new(MeldType::Chi, vec![4, 8, 12], true, -1, Some(4)));
        let counts = VisibleTileCounts::from_observation(&obs).unwrap();
        assert_eq!(counts.visible_count(1), Some(1));
        assert_eq!(counts.visible_count(2), Some(1));
        assert_eq!(counts.visible_count(3), Some(1));
        assert_eq!(counts.self_hand_base34[0], 2);
        assert_eq!(counts.red5_unseen, [false, true, true]);
        assert_eq!(counts.unseen_count(0), Some(2));
        for base in 0..34 {
            assert_eq!(
                counts.self_hand_base34[base]
                    + counts.visible_base34[base]
                    + counts.unseen_base34[base],
                4
            );
        }
    }

    #[test]
    fn rejects_unexplained_duplicate_physical_tiles() {
        let mut obs = observation();
        obs.discards[1].push(20);
        obs.discards[2].push(20);
        assert!(VisibleTileCounts::from_observation(&obs).is_err());
    }

    #[test]
    fn ignores_opponent_concealed_hands() {
        let mut obs = observation();
        obs.hands[1].push(40);
        let counts = VisibleTileCounts::from_observation(&obs).unwrap();
        assert!(!counts.seen_136[40]);
        assert_eq!(counts.unseen_base34[10], 4);
    }

    #[test]
    fn rejects_called_tile_without_matching_source_discard() {
        let mut obs = observation();
        obs.melds[0].push(Meld::new(
            MeldType::Pon,
            vec![40, 41, 42],
            true,
            -1,
            Some(40),
        ));
        assert!(VisibleTileCounts::from_observation(&obs).is_err());
    }

    #[test]
    fn recovers_kakan_claim_from_physical_river_intersection() {
        let mut obs = observation();
        obs.discards[2].push(40);
        obs.melds[0].push(Meld::new(
            MeldType::Kakan,
            vec![40, 41, 42, 43],
            true,
            -1,
            None,
        ));
        let counts = VisibleTileCounts::from_observation(&obs).unwrap();
        assert_eq!(counts.visible_base34[10], 4);
    }
}
