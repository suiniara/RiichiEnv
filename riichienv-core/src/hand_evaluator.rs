#![allow(clippy::useless_conversion)]
use crate::agari;
use crate::errors::RiichiResult;
use crate::rule::GameRule;
use crate::score;
use crate::types::{Conditions, Hand, Meld, MeldType, WinResult, Wind};
use crate::yaku;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass)]
pub struct HandEvaluator {
    pub hand: Hand,      // Normalised for agari detection
    pub full_hand: Hand, // Full counts for dora/yaku
    pub melds: Vec<Meld>,
    pub aka_dora_count: u8,
}

impl HandEvaluator {
    pub fn hand_from_text(text: &str) -> RiichiResult<Self> {
        let (tiles, melds) = crate::parser::parse_hand_internal(text)?;
        Ok(Self::new(tiles, melds))
    }

    pub fn new(tiles_136: Vec<u8>, melds: Vec<Meld>) -> Self {
        let mut aka_dora_count = 0;
        let mut tiles_34 = Vec::with_capacity(tiles_136.len());

        for &t in &tiles_136 {
            if t == 16 || t == 52 || t == 88 {
                aka_dora_count += 1;
            }
            tiles_34.push(t / 4);
        }

        let mut full_hand = Hand::new(Some(tiles_34));
        let mut hand = full_hand.clone();

        let mut internal_melds = Vec::with_capacity(melds.len());

        for meld in &melds {
            let mut new_meld = meld.clone();

            if new_meld.meld_type == MeldType::Daiminkan
                || new_meld.meld_type == MeldType::Ankan
                || new_meld.meld_type == MeldType::Kakan
            {
                let t_34 = new_meld.tiles[0] / 4;
                if hand.counts[t_34 as usize] == 4 {
                    hand.counts[t_34 as usize] = 3;
                }
            }

            let mut meld_tiles_34 = Vec::with_capacity(new_meld.tiles.len());
            for &t in &new_meld.tiles {
                if t == 16 || t == 52 || t == 88 {
                    aka_dora_count += 1;
                }
                let t_34 = t / 4;
                meld_tiles_34.push(t_34);
                full_hand.add(t_34);
            }
            new_meld.tiles = meld_tiles_34;
            if new_meld.meld_type == MeldType::Chi {
                new_meld.tiles.sort();
            }
            internal_melds.push(new_meld);
        }

        Self {
            hand,
            full_hand,
            melds: internal_melds,
            aka_dora_count,
        }
    }

    pub fn calc(
        &self,
        win_tile: u8,
        dora_indicators: Vec<u8>,
        ura_indicators: Vec<u8>,
        conditions: Option<Conditions>,
    ) -> WinResult {
        let win_tile_136 = win_tile;
        let conditions = conditions.unwrap_or_default();
        let win_tile_34 = win_tile_136 / 4;

        let mut hand_14 = self.hand.clone();
        let mut full_hand_14 = self.full_hand.clone();

        let current_total: u8 = hand_14.counts.iter().sum::<u8>() + (self.melds.len() as u8 * 3);

        if current_total == 13 {
            hand_14.add(win_tile_34);
            full_hand_14.add(win_tile_34);
        }

        let is_agari = agari::is_agari(&mut hand_14);

        if !is_agari {
            return WinResult::new(false, false, 0, 0, 0, vec![], 0, 0, None, false);
        }

        let mut dora_count = 0;
        for &indicator_136 in &dora_indicators {
            let next_tile_34 = get_next_tile(indicator_136 / 4);
            dora_count += full_hand_14.counts[next_tile_34 as usize];
        }

        let mut ura_dora_count = 0;
        for &indicator_136 in &ura_indicators {
            let next_tile_34 = get_next_tile(indicator_136 / 4);
            ura_dora_count += full_hand_14.counts[next_tile_34 as usize];
        }

        let mut aka_dora = self.aka_dora_count;
        if current_total == 13 && (win_tile_136 == 16 || win_tile_136 == 52 || win_tile_136 == 88) {
            aka_dora += 1;
        }

        let ctx = yaku::YakuContext {
            is_tsumo: conditions.tsumo,
            is_reach: conditions.riichi,
            is_daburu_reach: conditions.double_riichi,
            is_ippatsu: conditions.ippatsu,
            is_haitei: conditions.haitei,
            is_houtei: conditions.houtei,
            is_rinshan: conditions.rinshan,
            is_chankan: conditions.chankan,
            is_tsumo_first_turn: conditions.tsumo_first_turn,
            dora_count,
            aka_dora,
            ura_dora_count,
            round_wind: 27 + conditions.round_wind as u8,
            seat_wind: 27 + conditions.player_wind as u8,
            is_menzen: self.melds.iter().all(|m| !m.opened),
        };

        let _divisions = agari::find_divisions(&hand_14);
        let yaku_res = yaku::calculate_yaku(&hand_14, &self.melds, &ctx, win_tile_34);

        let is_oya = conditions.player_wind == Wind::East;
        // Kazoe yakuman: cap han at 13 for scoring (single yakuman)
        let scoring_han = if yaku_res.yakuman_count == 0 && yaku_res.han >= 13 {
            13
        } else {
            yaku_res.han
        };
        let score_res = score::calculate_score(
            scoring_han,
            yaku_res.fu,
            is_oya,
            conditions.tsumo,
            conditions.honba,
            4, // Always 4 players
        );
        let has_yaku = yaku_res
            .yaku_ids
            .iter()
            .any(|&id| id != yaku::ID_DORA && id != yaku::ID_AKADORA && id != yaku::ID_URADORA);

        let official_yaku: Vec<u32> = yaku_res.yaku_ids.into_iter().collect();

        WinResult {
            is_win: (has_yaku || yaku_res.yakuman_count > 0) && yaku_res.han >= 1,
            yakuman: yaku_res.yakuman_count > 0,
            ron_agari: score_res.pay_ron,
            tsumo_agari_oya: score_res.pay_tsumo_oya,
            tsumo_agari_ko: score_res.pay_tsumo_ko,
            yaku: official_yaku,
            han: yaku_res.han as u32,
            fu: yaku_res.fu as u32,
            pao_payer: None,
            has_win_shape: true,
        }
    }

    pub fn is_tenpai(&self) -> bool {
        self.wait_tiles(true).next().is_some()
    }

    /// Formal tenpai depends on the ruleset when all four copies of a wait
    /// are already in this player's concealed hand and established melds.
    pub fn is_tenpai_with_rule(&self, rule: &GameRule) -> bool {
        self.wait_tiles(rule.allow_fifth_copy_tenpai)
            .next()
            .is_some()
    }

    pub fn get_waits_u8(&self) -> Vec<u8> {
        self.wait_tiles(true).collect()
    }

    pub fn get_waits_u8_with_rule(&self, rule: &GameRule) -> Vec<u8> {
        self.wait_tiles(rule.allow_fifth_copy_tenpai).collect()
    }

    fn wait_tiles(&self, allow_fifth_copy_tenpai: bool) -> impl Iterator<Item = u8> + '_ {
        let current_total: usize =
            self.hand.counts.iter().map(|&n| n as usize).sum::<usize>() + self.melds.len() * 3;
        let mut hand_14 = self.hand.clone();
        (0..crate::types::TILE_MAX).filter_map(move |i| {
            // The legacy interface and Tenhou use concealed counts for formal
            // tenpai. Mahjong Soul also excludes copies in the player's melds.
            // Neither rule counts visible tiles belonging to other players.
            let owned_count = if allow_fifth_copy_tenpai {
                self.hand.counts[i]
            } else {
                self.full_hand.counts[i]
            };
            if current_total != 13 || owned_count >= 4 {
                return None;
            }
            hand_14.add(i as u8);
            let is_agari = agari::is_agari(&mut hand_14);
            hand_14.remove(i as u8);
            is_agari.then_some(i as u8)
        })
    }

    pub fn get_waits(&self) -> Vec<u32> {
        self.wait_tiles(true).map(u32::from).collect()
    }

    pub fn get_waits_with_rule(&self, rule: &GameRule) -> Vec<u32> {
        self.wait_tiles(rule.allow_fifth_copy_tenpai)
            .map(u32::from)
            .collect()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl HandEvaluator {
    #[staticmethod]
    #[pyo3(name = "hand_from_text")]
    pub fn hand_from_text_py(text: &str) -> PyResult<Self> {
        Self::hand_from_text(text).map_err(Into::into)
    }

    #[new]
    #[pyo3(signature = (tiles_136, melds=vec![]))]
    pub fn py_new(tiles_136: Vec<u8>, melds: Vec<Meld>) -> Self {
        Self::new(tiles_136, melds)
    }

    #[pyo3(signature = (win_tile, dora_indicators=vec![], ura_indicators=vec![], conditions=None))]
    #[pyo3(name = "calc")]
    pub fn calc_py(
        &self,
        win_tile: u8,
        dora_indicators: Vec<u8>,
        ura_indicators: Vec<u8>,
        conditions: Option<Conditions>,
    ) -> WinResult {
        self.calc(win_tile, dora_indicators, ura_indicators, conditions)
    }

    #[pyo3(name = "is_tenpai")]
    pub fn is_tenpai_py(&self) -> bool {
        self.is_tenpai()
    }

    #[pyo3(name = "is_tenpai_with_rule")]
    pub fn is_tenpai_with_rule_py(&self, rule: &GameRule) -> bool {
        self.is_tenpai_with_rule(rule)
    }

    #[pyo3(name = "get_waits_u8")]
    pub fn get_waits_u8_py(&self) -> Vec<u8> {
        self.get_waits_u8()
    }

    #[pyo3(name = "get_waits_u8_with_rule")]
    pub fn get_waits_u8_with_rule_py(&self, rule: &GameRule) -> Vec<u8> {
        self.get_waits_u8_with_rule(rule)
    }

    #[pyo3(name = "get_waits")]
    pub fn get_waits_py(&self) -> Vec<u32> {
        self.get_waits()
    }

    #[pyo3(name = "get_waits_with_rule")]
    pub fn get_waits_with_rule_py(&self, rule: &GameRule) -> Vec<u32> {
        self.get_waits_with_rule(rule)
    }
}

pub fn check_riichi_candidates(tiles_136: Vec<u8>) -> Vec<u32> {
    let mut candidates = Vec::new();
    // Convert to 34-tile hand
    let mut tiles_34 = Vec::with_capacity(tiles_136.len());
    for t in &tiles_136 {
        tiles_34.push(t / 4);
    }

    for (i, &t_discard) in tiles_136.iter().enumerate() {
        let mut hand = crate::types::Hand::default();
        for (j, &t) in tiles_34.iter().enumerate() {
            if i != j {
                hand.add(t);
            }
        }

        if agari::is_tenpai(&mut hand) {
            candidates.push(t_discard as u32);
        }
    }
    candidates
}

fn get_next_tile(tile: u8) -> u8 {
    if tile < 9 {
        if tile == 8 { 0 } else { tile + 1 }
    } else if tile < 18 {
        if tile == 17 { 9 } else { tile + 1 }
    } else if tile < 27 {
        if tile == 26 { 18 } else { tile + 1 }
    } else if tile < 31 {
        if tile == 30 { 27 } else { tile + 1 }
    } else if tile == 33 {
        31
    } else {
        tile + 1
    }
}

#[cfg(test)]
mod physical_wait_tests {
    use super::*;

    #[test]
    fn own_pon_and_fourth_tile_formal_wait_depends_on_rule() {
        // Concealed 1m 234567p 789s, with an established pon of 1m.
        let hand = vec![3, 40, 44, 48, 53, 56, 60, 96, 100, 104];
        let pon = Meld {
            meld_type: MeldType::Pon,
            tiles: vec![0, 1, 2],
            opened: true,
            from_who: 0,
            called_tile: Some(0),
        };
        let evaluator = HandEvaluator::new(hand, vec![pon]);
        assert!(evaluator.is_tenpai());
        assert_eq!(evaluator.get_waits_u8(), vec![0]);
        assert_eq!(evaluator.get_waits(), vec![0]);
        assert!(evaluator.is_tenpai_with_rule(&GameRule::default_tenhou()));
        assert_eq!(
            evaluator.get_waits_u8_with_rule(&GameRule::default_tenhou()),
            vec![0]
        );
        assert!(!evaluator.is_tenpai_with_rule(&GameRule::default_mjsoul()));
        assert!(
            evaluator
                .get_waits_u8_with_rule(&GameRule::default_mjsoul())
                .is_empty()
        );
        assert!(
            evaluator
                .get_waits_with_rule(&GameRule::default_mjsoul())
                .is_empty()
        );
    }

    #[test]
    fn an_available_fourth_copy_remains_a_valid_wait() {
        // Concealed 23m 234p 789s EE, with a pon containing three 1m.
        // The fourth 1m can complete 123m; publicly exhausted tiles elsewhere
        // are not part of this player's physical-copy constraint.
        let hand = vec![4, 8, 40, 44, 48, 96, 100, 104, 108, 109];
        let pon = Meld {
            meld_type: MeldType::Pon,
            tiles: vec![0, 1, 2],
            opened: true,
            from_who: 0,
            called_tile: Some(0),
        };
        let evaluator = HandEvaluator::new(hand, vec![pon]);
        assert!(evaluator.is_tenpai());
        assert_eq!(evaluator.get_waits_u8(), vec![0, 3]);
        for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
            assert!(evaluator.is_tenpai_with_rule(&rule));
            assert_eq!(evaluator.get_waits_u8_with_rule(&rule), vec![0, 3]);
        }
    }

    #[test]
    fn concealed_four_copies_cannot_be_a_formal_wait_under_either_rule() {
        // 1111m 234567p 789s: only a fifth 1m could produce the missing pair.
        let hand = vec![0, 1, 2, 3, 40, 44, 48, 53, 56, 60, 96, 100, 104];
        let evaluator = HandEvaluator::new(hand, vec![]);
        assert!(!evaluator.is_tenpai());
        assert!(evaluator.get_waits_u8().is_empty());
        for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
            assert!(!evaluator.is_tenpai_with_rule(&rule));
            assert!(evaluator.get_waits_u8_with_rule(&rule).is_empty());
        }
    }
}
