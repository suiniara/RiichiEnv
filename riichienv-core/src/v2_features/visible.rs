use crate::observation::Observation;
use crate::types::MeldType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleTileCounts {
    pub seen_136: [bool; 136],
    pub visible_base34: [u8; 34],
    pub self_hand_base34: [u8; 34],
    pub unseen_base34: [u8; 34],
    pub red5_unseen: [bool; 3],
}

impl VisibleTileCounts {
    pub fn from_observation(observation: &Observation) -> Result<Self, String> {
        let observer = usize::from(observation.player_id);
        if observer >= 4 {
            return Err(format!("invalid observer player id: {}", observation.player_id));
        }

        let mut seen_136 = [false; 136];
        let mut hand_seen = [false; 136];
        let mut discard_seen = [false; 136];
        let mut meld_seen = [false; 136];
        let mut dora_seen = [false; 136];
        let mut called_from_river = [false; 136];

        for &raw in &observation.hands[observer] {
            let tile = usize::try_from(raw).map_err(|_| "invalid concealed tile id")?;
            if tile >= 136 { return Err(format!("concealed tile id out of range: {tile}")); }
            if hand_seen[tile] { return Err(format!("duplicate physical tile in observer hand: {tile}")); }
            hand_seen[tile] = true;
            seen_136[tile] = true;
        }

        for discards in &observation.discards {
            for &raw in discards {
                let tile = usize::try_from(raw).map_err(|_| "invalid discard tile id")?;
                if tile >= 136 { return Err(format!("discard tile id out of range: {tile}")); }
                if discard_seen[tile] { return Err(format!("duplicate physical tile in discard records: {tile}")); }
                if hand_seen[tile] { return Err(format!("physical tile appears in both hand and discard: {tile}")); }
                discard_seen[tile] = true;
                seen_136[tile] = true;
            }
        }

        for melds in &observation.melds {
            for meld in melds {
                let called = if let Some(called) = meld.called_tile {
                    Some(usize::from(called))
                } else if meld.meld_type == MeldType::Kakan {
                    // Replay formats may omit the upgraded Pon's called_tile.
                    // Recover it from the unique physical intersection with rivers.
                    let matches = meld.tiles.iter().map(|&tile| usize::from(tile))
                        .filter(|&tile| tile < 136 && discard_seen[tile])
                        .collect::<Vec<_>>();
                    match matches.as_slice() {
                        [tile] => Some(*tile),
                        [] => return Err("Kakan has no recoverable claimed physical tile".into()),
                        _ => return Err("Kakan has multiple claimed physical tiles".into()),
                    }
                } else { None };

                if let Some(called) = called {
                    if called >= 136 { return Err(format!("called tile id out of range: {called}")); }
                    if !meld.tiles.iter().any(|&tile| usize::from(tile) == called) {
                        return Err(format!("called tile {called} is absent from its meld"));
                    }
                    if !discard_seen[called] { return Err(format!("called tile {called} has no public source discard")); }
                    if called_from_river[called] { return Err(format!("physical tile called more than once: {called}")); }
                    called_from_river[called] = true;
                }
            }
        }

        for melds in &observation.melds {
            for meld in melds {
                for &raw in &meld.tiles {
                    let tile = usize::from(raw);
                    if tile >= 136 { return Err(format!("meld tile id out of range: {tile}")); }
                    if hand_seen[tile] { return Err(format!("physical tile appears in hand and meld: {tile}")); }
                    if meld_seen[tile] { return Err(format!("duplicate physical tile across melds: {tile}")); }
                    if discard_seen[tile] && !called_from_river[tile] {
                        return Err(format!("meld/discard overlap is not a claimed tile: {tile}"));
                    }
                    meld_seen[tile] = true;
                    seen_136[tile] = true;
                }
            }
        }

        for tile in 0..136 {
            if called_from_river[tile] && !meld_seen[tile] {
                return Err(format!("claimed river tile is absent from melds: {tile}"));
            }
        }

        for &raw in &observation.dora_indicators {
            let tile = usize::try_from(raw).map_err(|_| "invalid dora indicator id")?;
            if tile >= 136 { return Err(format!("dora indicator tile id out of range: {tile}")); }
            if hand_seen[tile] || discard_seen[tile] || meld_seen[tile] || dora_seen[tile] {
                return Err(format!("dora indicator duplicates another known physical tile: {tile}"));
            }
            dora_seen[tile] = true;
            seen_136[tile] = true;
        }

        let mut visible_base34 = [0u8; 34];
        let mut self_hand_base34 = [0u8; 34];
        for tile in 0..136 {
            let base = tile / 4;
            if hand_seen[tile] { self_hand_base34[base] += 1; }
            else if seen_136[tile] { visible_base34[base] += 1; }
        }

        let mut unseen_base34 = [0u8; 34];
        for base in 0..34 {
            let known = self_hand_base34[base] + visible_base34[base];
            if known > 4 { return Err(format!("more than four physical copies known for base34 {base}: {known}")); }
            unseen_base34[base] = 4 - known;
        }

        let red5_unseen = [16usize, 52, 88].map(|tile| !seen_136[tile]);
        Ok(Self { seen_136, visible_base34, self_hand_base34, unseen_base34, red5_unseen })
    }

    #[inline]
    pub fn visible_count(&self, base34: usize) -> Option<u8> {
        self.visible_base34.get(base34).copied()
    }

    #[inline]
    pub fn unseen_count(&self, base34: usize) -> Option<u8> {
        self.unseen_base34.get(base34).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Meld, MeldType};

    fn observation() -> Observation {
        Observation::new(
            0,
            [vec![0, 1, 16], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25_000; 4], [false; 4], vec![], vec![],
            0, 0, 0, 0, 0, vec![], false,
            [None; 4], [None; 4], None, None,
        )
    }

    #[test]
    fn counts_called_tile_once_across_river_and_meld() {
        let mut obs = observation();
        obs.discards[1].push(4);
        obs.melds[0].push(Meld::new(MeldType::Chi, vec![4, 8, 12], true, -1, Some(4)));
        let counts = VisibleTileCounts::from_observation(&obs).unwrap();
        assert_eq!(counts.visible_base34[1], 1);
        assert_eq!(counts.visible_base34[2], 1);
        assert_eq!(counts.visible_base34[3], 1);
        assert_eq!(counts.self_hand_base34[0], 2);
        assert_eq!(counts.red5_unseen, [false, true, true]);
        assert_eq!(counts.unseen_base34[0], 2);
        for base in 0..34 {
            assert_eq!(counts.self_hand_base34[base] + counts.visible_base34[base] + counts.unseen_base34[base], 4);
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
        obs.melds[0].push(Meld::new(MeldType::Pon, vec![40, 41, 42], true, -1, Some(40)));
        assert!(VisibleTileCounts::from_observation(&obs).is_err());
    }

    #[test]
    fn recovers_kakan_claim_from_physical_river_intersection() {
        let mut obs = observation();
        obs.discards[2].push(40);
        obs.melds[0].push(Meld::new(MeldType::Kakan, vec![40, 41, 42, 43], true, -1, None));
        let counts = VisibleTileCounts::from_observation(&obs).unwrap();
        assert_eq!(counts.visible_base34[10], 4);
    }
}
