use riichienv_core::types::{Conditions, WinResult, Wind};
use riichienv_core::yaku::{
    ID_CHANTA, ID_CHIHO, ID_CHUUREN, ID_IPEIKO, ID_JUNSEI_CHUUREN, ID_RIICHI, ID_SUANKO,
    ID_SUANKO_TANKI, ID_TENHO,
};

fn assert_yakuman(result: &WinResult, expected: &[u32]) {
    assert!(result.is_win && result.yakuman);
    let mut actual = result.yaku.clone();
    let mut expected = expected.to_vec();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
}

macro_rules! scoring_selection_tests {
    ($name:ident, $evaluator:path, $players:expr) => {
        mod $name {
            use super::*;
            type Evaluator = $evaluator;

            fn conditions(tsumo: bool, first_turn: bool, wind: Wind) -> Conditions {
                Conditions {
                    tsumo,
                    tsumo_first_turn: first_turn,
                    player_wind: wind,
                    is_sanma: $players == 3,
                    num_players: $players,
                    ..Default::default()
                }
            }

            #[test]
            fn counted_yakuman_cannot_replace_a_true_double_yakuman() {
                // Both 111/222/333/444/55 and 123/123/123/444/55 are possible.
                // The second interpretation has 34 ordinary han with these
                // physically distinct dora/ura tiles, but pays only one yakuman.
                let evaluator = Evaluator::hand_from_text("1112223334440p").unwrap();
                let mut context = conditions(true, false, Wind::South);
                context.riichi = true;
                context.double_riichi = true;
                for (dora, ura) in [
                    (vec![], vec![]),
                    (vec![68, 69, 70, 71], vec![39, 43, 47, 51]),
                ] {
                    let result = evaluator.calc(53, dora, ura, Some(context.clone()));
                    assert_yakuman(&result, &[ID_SUANKO_TANKI]);
                    assert_eq!(result.han, 26);
                    assert_eq!(result.tsumo_agari_oya, 32000);
                    assert_eq!(result.tsumo_agari_ko, 16000);
                }
            }

            #[test]
            fn ordinary_ambiguous_hand_keeps_its_highest_payment() {
                // Interpreting 111222333p as 123/123/123 earns Iipeikou and
                // Chanta; the triplet interpretation cannot replace it.
                let evaluator = Evaluator::hand_from_text("111222333p789s11z").unwrap();
                let mut context = conditions(false, false, Wind::South);
                context.riichi = true;
                let result = evaluator.calc(36, vec![], vec![], Some(context));
                assert!(result.is_win && !result.yakuman);
                let mut actual_yaku = result.yaku.clone();
                actual_yaku.sort();
                assert_eq!(actual_yaku, [ID_RIICHI, ID_IPEIKO, ID_CHANTA]);
                assert_eq!((result.han, result.fu, result.ron_agari), (4, 40, 8000));
            }

            #[test]
            fn tenhou_selects_the_best_winning_tile_for_every_initial_deal_order() {
                // Mahjong Soul ranked rules: at Tenhou, treat the tile yielding
                // the highest payment as the winning tile (news/46).
                for (text, expected_id) in [
                    ("111m222p333s111z55p", ID_SUANKO_TANKI),
                    ("11123455678999p", ID_JUNSEI_CHUUREN),
                ] {
                    let evaluator = Evaluator::hand_from_text(text).unwrap();
                    for (tile, count) in evaluator.hand.counts.iter().enumerate() {
                        if *count == 0 {
                            continue;
                        }
                        let result = evaluator.calc(
                            tile as u8 * 4,
                            vec![],
                            vec![],
                            Some(conditions(true, true, Wind::East)),
                        );
                        assert_yakuman(&result, &[ID_TENHO, expected_id]);
                        assert_eq!(result.han, 39, "{text}, last tile {tile}");
                        assert_eq!(result.tsumo_agari_ko, 48000);
                    }
                }
            }

            #[test]
            fn ordinary_tsumo_and_chiihou_keep_the_actual_winning_tile() {
                for (text, ordinary_id, double_id, ordinary_tile, double_tile) in [
                    ("111m222p333s111z55p", ID_SUANKO, ID_SUANKO_TANKI, 0, 53),
                    ("11123455678999p", ID_CHUUREN, ID_JUNSEI_CHUUREN, 36, 53),
                ] {
                    let evaluator = Evaluator::hand_from_text(text).unwrap();
                    for first_turn in [false, true] {
                        for (tile, id, units) in
                            [(ordinary_tile, ordinary_id, 1), (double_tile, double_id, 2)]
                        {
                            let result = evaluator.calc(
                                tile,
                                vec![],
                                vec![],
                                Some(conditions(true, first_turn, Wind::South)),
                            );
                            let mut expected = vec![id];
                            if first_turn {
                                expected.push(ID_CHIHO);
                            }
                            assert_yakuman(&result, &expected);
                            assert_eq!(result.han, 13 * (units + u32::from(first_turn)));
                        }
                    }
                }
            }

            #[test]
            fn ron_does_not_gain_tenhou_or_an_invented_pair_wait() {
                let result = Evaluator::hand_from_text("111m222p333s111z55p")
                    .unwrap()
                    .calc(0, vec![], vec![], Some(conditions(false, true, Wind::East)));
                assert!(result.is_win);
                assert!(!result.yakuman);
                assert!(!result.yaku.contains(&ID_TENHO));
                assert!(!result.yaku.contains(&ID_SUANKO_TANKI));
            }
        }
    };
}

scoring_selection_tests!(
    four_players,
    riichienv_core::hand_evaluator::HandEvaluator,
    4
);
scoring_selection_tests!(
    three_players,
    riichienv_core::hand_evaluator_3p::HandEvaluator3P,
    3
);
