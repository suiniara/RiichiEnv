//! Four-player settlement and match progression, independent of a GameState.
//!
//! Inputs are evaluated hands and explicit round facts.  Recorded server
//! payments are comparison targets, never inputs to these calculations.
//! Three-player play deliberately retains its separate rule implementation.

use crate::errors::{RiichiError, RiichiResult};
use crate::rule::GameRule;
use crate::score::{Score, calculate_score};
use crate::types::WinResult;

pub const NUM_PLAYERS: usize = 4;
pub type Deltas = [i32; NUM_PLAYERS];

#[derive(Debug, Clone)]
pub struct EvaluatedWin {
    pub player_id: u8,
    pub result: WinResult,
    pub liable_player: Option<u8>,
    pub liable_yakuman_units: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WinnerPayment {
    pub player_id: u8,
    pub deltas: Deltas,
    pub honba: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WinPayments {
    pub deltas: Deltas,
    /// Clockwise from the discarder; a self-draw has one entry.
    pub winners: Vec<WinnerPayment>,
}

fn invalid(message: &str) -> RiichiError {
    RiichiError::InvalidState {
        message: message.into(),
    }
}

fn seat(player: u8) -> RiichiResult<()> {
    if usize::from(player) >= NUM_PLAYERS {
        return Err(invalid("A four-player seat must be in 0..4"));
    }
    Ok(())
}

fn narrow(values: [i64; NUM_PLAYERS]) -> RiichiResult<Deltas> {
    let mut result = [0; NUM_PLAYERS];
    for (out, value) in result.iter_mut().zip(values) {
        *out =
            i32::try_from(value).map_err(|_| invalid("Settlement exceeds the i32 point range"))?;
    }
    Ok(result)
}

fn base_score(win: &EvaluatedWin, oya: u8, tsumo: bool, rule: &GameRule) -> RiichiResult<Score> {
    seat(win.player_id)?;
    let result = &win.result;
    if !result.is_win || !(1..=255).contains(&result.han) || result.fu > 255 {
        return Err(invalid(
            "A winner requires a valid independently evaluated winning result",
        ));
    }
    if result.yakuman && !result.han.is_multiple_of(13) {
        return Err(invalid("Yakuman han must identify whole yakuman units"));
    }
    let scoring_han = if result.yakuman {
        result.han
    } else {
        result.han.min(13)
    };
    if scoring_han < 5 && result.fu > 250 {
        return Err(invalid(
            "Fu cannot be rounded within the supported score range",
        ));
    }
    let base = calculate_score(
        scoring_han as u8,
        result.fu as u8,
        win.player_id == oya,
        tsumo,
        0,
        4,
    );
    if base.total == 0 {
        return Err(invalid("A winning result must have a nonzero base payment"));
    }
    match win.liable_player {
        None if win.liable_yakuman_units != 0 => {
            return Err(invalid("Responsible yakuman units require a liable player"));
        }
        Some(liable) => {
            seat(liable)?;
            if liable == win.player_id
                || !result.yakuman
                || win.liable_yakuman_units == 0
                || win.liable_yakuman_units > result.han / 13
            {
                return Err(invalid(
                    "Liability requires supported units within a true yakuman result and a distinct payer",
                ));
            }
            let mut possible = vec![0];
            for (yaku, units) in [(37, 1), (50, if rule.is_daisuushii_double { 2 } else { 1 })] {
                if result.yaku.contains(&yaku) {
                    let extra: Vec<u32> = possible.iter().map(|count| count + units).collect();
                    possible.extend(extra);
                }
            }
            if !possible.contains(&win.liable_yakuman_units) {
                return Err(invalid("Liable units disagree with responsibility yakuman"));
            }
        }
        None => {}
    }
    Ok(base)
}

/// Audit adapters require zero-honba base fields.  Online WinResult fields may
/// already include honba; settlement itself always recalculates the base.
pub fn validate_base_win(
    win: &EvaluatedWin,
    oya: u8,
    tsumo: bool,
    rule: &GameRule,
) -> RiichiResult<()> {
    seat(oya)?;
    let base = base_score(win, oya, tsumo, rule)?;
    let result = &win.result;
    if (
        result.ron_agari,
        result.tsumo_agari_oya,
        result.tsumo_agari_ko,
    ) != (base.pay_ron, base.pay_tsumo_oya, base.pay_tsumo_ko)
    {
        return Err(invalid(
            "Result must be a zero-honba base evaluation for this win role",
        ));
    }
    Ok(())
}

pub fn win_payments(
    wins: &[EvaluatedWin],
    oya: u8,
    target: Option<u8>,
    honba: u32,
    riichi_sticks: u32,
    rule: &GameRule,
) -> RiichiResult<WinPayments> {
    seat(oya)?;
    if wins.is_empty() || wins.len() > 3 || (target.is_none() && wins.len() != 1) {
        return Err(invalid(
            "A self-draw needs one winner; ron needs one to three winners",
        ));
    }
    if let Some(discarder) = target {
        seat(discarder)?;
    }
    let mut seen = [false; NUM_PLAYERS];
    for win in wins {
        seat(win.player_id)?;
        if seen[usize::from(win.player_id)] || target == Some(win.player_id) {
            return Err(invalid(
                "Winners must be distinct and cannot include the discarder",
            ));
        }
        seen[usize::from(win.player_id)] = true;
    }
    let mut ordered: Vec<_> = wins.iter().collect();
    if let Some(discarder) = target {
        ordered.sort_by_key(|win| (win.player_id + 4 - discarder) % 4);
    }
    let mut total = [0i64; NUM_PLAYERS];
    let mut winners = Vec::with_capacity(wins.len());
    for (index, win) in ordered.into_iter().enumerate() {
        let base = base_score(win, oya, target.is_none(), rule)?;
        let winner = usize::from(win.player_id);
        let mut delta = [0i64; NUM_PLAYERS];
        let bonus = if index == 0 {
            i64::from(honba) * 300
        } else {
            0
        };
        let total_units = if win.result.yakuman {
            win.result.han / 13
        } else {
            0
        };
        let liable_units = if rule.yakuman_pao_is_liability_only {
            win.liable_yakuman_units
        } else {
            total_units
        };
        let unit: i64 = if win.player_id == oya { 48000 } else { 32000 };
        if let Some(discarder) = target {
            let payment = i64::from(base.pay_ron) + bonus;
            let liability = if let Some(liable) = win.liable_player {
                let value = i64::from(liable_units) * unit / 2 + bonus;
                delta[usize::from(liable)] -= value;
                value
            } else {
                0
            };
            delta[usize::from(discarder)] -= payment - liability;
            delta[winner] += payment;
        } else if let Some(liable) = win.liable_player {
            let payment = i64::from(liable_units) * unit + bonus;
            delta[usize::from(liable)] -= payment;
            delta[winner] += payment;
            let remaining = total_units - liable_units;
            for payer in 0..NUM_PLAYERS {
                if payer != winner {
                    let value = i64::from(remaining)
                        * if win.player_id == oya || payer == usize::from(oya) {
                            16000
                        } else {
                            8000
                        };
                    delta[payer] -= value;
                    delta[winner] += value;
                }
            }
        } else {
            for payer in 0..NUM_PLAYERS {
                if payer != winner {
                    let value = i64::from(if payer == usize::from(oya) {
                        base.pay_tsumo_oya
                    } else {
                        base.pay_tsumo_ko
                    }) + i64::from(honba) * 100;
                    delta[payer] -= value;
                    delta[winner] += value;
                }
            }
        }
        if index == 0 {
            delta[winner] += i64::from(riichi_sticks) * 1000;
        }
        for (sum, value) in total.iter_mut().zip(delta) {
            *sum += value;
        }
        winners.push(WinnerPayment {
            player_id: win.player_id,
            deltas: narrow(delta)?,
            honba: if index == 0 { honba } else { 0 },
        });
    }
    Ok(WinPayments {
        deltas: narrow(total)?,
        winners,
    })
}

pub fn draw_payments(
    tenpai: [bool; NUM_PLAYERS],
    oya: u8,
    nagashi_winners: &[u8],
    abortive: bool,
) -> RiichiResult<Deltas> {
    seat(oya)?;
    let mut seen = [false; NUM_PLAYERS];
    for &winner in nagashi_winners {
        seat(winner)?;
        if seen[usize::from(winner)] {
            return Err(invalid("Duplicate nagashi winner"));
        }
        seen[usize::from(winner)] = true;
    }
    if abortive && !nagashi_winners.is_empty() {
        return Err(invalid("An abortive draw cannot settle nagashi mangan"));
    }
    let mut delta = [0i64; NUM_PLAYERS];
    if abortive {
        return Ok([0; NUM_PLAYERS]);
    }
    if !nagashi_winners.is_empty() {
        for &winner in nagashi_winners {
            let base = calculate_score(5, 30, winner == oya, true, 0, 4);
            for payer in 0..NUM_PLAYERS {
                if payer != usize::from(winner) {
                    let payment = i64::from(if payer == usize::from(oya) {
                        base.pay_tsumo_oya
                    } else {
                        base.pay_tsumo_ko
                    });
                    delta[payer] -= payment;
                    delta[usize::from(winner)] += payment;
                }
            }
        }
    } else {
        let count = tenpai.iter().filter(|&&ready| ready).count() as i64;
        if count > 0 && count < NUM_PLAYERS as i64 {
            for (value, ready) in delta.iter_mut().zip(tenpai) {
                *value = if ready {
                    3000 / count
                } else {
                    -3000 / (NUM_PLAYERS as i64 - count)
                };
            }
        }
    }
    narrow(delta)
}

pub fn ranked_seats(
    scores: Deltas,
    initial_seat_order: [u8; NUM_PLAYERS],
) -> RiichiResult<[u8; NUM_PLAYERS]> {
    let mut seen = [false; NUM_PLAYERS];
    for &player in &initial_seat_order {
        seat(player)?;
        if seen[usize::from(player)] {
            return Err(invalid(
                "Initial seat order must contain four distinct seats",
            ));
        }
        seen[usize::from(player)] = true;
    }
    let mut order = initial_seat_order;
    // Stable sorting preserves initial seat priority for equal scores.
    order.sort_by_key(|&player| std::cmp::Reverse(scores[usize::from(player)]));
    Ok(order)
}

pub fn end_game_pot(
    scores: Deltas,
    riichi_sticks: u32,
    initial_seat_order: [u8; NUM_PLAYERS],
) -> RiichiResult<Deltas> {
    let order = ranked_seats(scores, initial_seat_order)?;
    let mut delta = [0i64; NUM_PLAYERS];
    delta[usize::from(order[0])] = i64::from(riichi_sticks) * 1000;
    narrow(delta)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundEnd {
    Win { dealer_won: bool, winner_count: u8 },
    ExhaustiveDraw { dealer_tenpai: bool },
    AbortiveDraw,
}

#[derive(Debug, Clone, Copy)]
pub struct RoundContext {
    pub scores: Deltas,
    pub dealer: u8,
    pub round_wind: u8,
    pub honba: u32,
    pub game_mode: u8,
    pub initial_seat_order: [u8; NUM_PLAYERS],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundTransition {
    pub dealer: u8,
    pub round_wind: u8,
    pub honba: u32,
    pub renchan: bool,
    pub match_end: bool,
    pub reason: Option<&'static str>,
}

pub fn round_transition(
    context: RoundContext,
    end: RoundEnd,
    rule: &GameRule,
) -> RiichiResult<RoundTransition> {
    seat(context.dealer)?;
    let ranked = ranked_seats(context.scores, context.initial_seat_order)?;
    let (renchan, draw, abortive, dealer_multi_ron) = match end {
        RoundEnd::Win {
            dealer_won,
            winner_count,
        } => {
            if !(1..=3).contains(&winner_count) {
                return Err(invalid("A win needs one to three winners"));
            }
            (dealer_won, false, false, dealer_won && winner_count > 1)
        }
        RoundEnd::ExhaustiveDraw { dealer_tenpai } => (dealer_tenpai, true, false, false),
        RoundEnd::AbortiveDraw => (true, true, true, false),
    };
    let dealer = if renchan {
        context.dealer
    } else {
        (context.dealer + 1) % 4
    };
    let wind = context
        .round_wind
        .checked_add(u8::from(!renchan && dealer == 0))
        .ok_or_else(|| invalid("Round wind exceeds the supported range"))?;
    let honba = if renchan || draw {
        context
            .honba
            .checked_add(1)
            .ok_or_else(|| invalid("Honba exceeds the supported range"))?
    } else {
        0
    };
    let extension_wind = match context.game_mode {
        0 | 3 => None,
        1 | 4 => Some(1),
        2 | 5 => Some(2),
        _ => return Err(invalid("Unknown four-player game mode")),
    };
    let all_last = extension_wind.is_some_and(|extension| {
        context.round_wind >= extension
            || (context.round_wind.saturating_add(1) == extension && context.dealer == 3)
    });
    let top = ranked[0];
    let max_score = context.scores[usize::from(top)];
    let repeat_has_priority =
        abortive || dealer_multi_ron || (renchan && rule.extension_dealer_repeat_priority);
    let reason = if context.scores.iter().any(|&score| score < 0) {
        Some("bankruptcy")
    } else if extension_wind.is_none() {
        Some("single_round")
    } else if !abortive && renchan && all_last && top == context.dealer && max_score >= 30000 {
        Some("dealer_yame")
    } else if !repeat_has_priority
        && extension_wind.is_some_and(|extension| {
            // Entering overtime requires a dealer change.  Once in overtime,
            // MjSoul ordinary draws reaching 30000 end even with a ready dealer;
            // the separate rule flag above preserves Tenhou's renchan priority.
            (context.round_wind >= extension || (!renchan && wind >= extension))
                && (max_score >= 30000 || wind > extension)
        })
    {
        Some("extension_limit")
    } else {
        None
    };
    Ok(RoundTransition {
        dealer,
        round_wind: wind,
        honba,
        renchan,
        match_end: reason.is_some(),
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(
        player_id: u8,
        oya: u8,
        tsumo: bool,
        han: u8,
        yakuman: bool,
        yaku: Vec<u32>,
    ) -> EvaluatedWin {
        let base = calculate_score(han, 40, player_id == oya, tsumo, 0, 4);
        EvaluatedWin {
            player_id,
            result: WinResult::new(
                true,
                yakuman,
                base.pay_ron,
                base.pay_tsumo_oya,
                base.pay_tsumo_ko,
                yaku,
                han as u32,
                40,
                None,
                true,
            ),
            liable_player: None,
            liable_yakuman_units: 0,
        }
    }

    fn mjsoul_transition(context: RoundContext, end: RoundEnd) -> RiichiResult<RoundTransition> {
        round_transition(context, end, &GameRule::default_mjsoul())
    }

    #[test]
    fn extension_dealer_priority_is_an_explicit_ruleset_choice() {
        let context = west([26500, 31000, 21000, 21500]);
        let draw = RoundEnd::ExhaustiveDraw {
            dealer_tenpai: true,
        };
        assert!(
            round_transition(context, draw, &GameRule::default_mjsoul())
                .unwrap()
                .match_end
        );
        assert!(
            !round_transition(context, draw, &GameRule::default_tenhou())
                .unwrap()
                .match_end
        );
        let win = RoundEnd::Win {
            dealer_won: true,
            winner_count: 1,
        };
        assert!(
            round_transition(context, win, &GameRule::default_mjsoul())
                .unwrap()
                .match_end
        );
        assert!(
            !round_transition(context, win, &GameRule::default_tenhou())
                .unwrap()
                .match_end
        );
        let multiple = RoundEnd::Win {
            dealer_won: true,
            winner_count: 2,
        };
        for rule in [GameRule::default_mjsoul(), GameRule::default_tenhou()] {
            assert!(
                !round_transition(context, multiple, &rule)
                    .unwrap()
                    .match_end
            );
            assert!(
                !round_transition(context, RoundEnd::AbortiveDraw, &rule)
                    .unwrap()
                    .match_end
            );
        }
    }

    fn west(scores: Deltas) -> RoundContext {
        RoundContext {
            scores,
            dealer: 0,
            round_wind: 2,
            honba: 0,
            game_mode: 2,
            initial_seat_order: [0, 1, 2, 3],
        }
    }

    #[test]
    fn tsumo_and_multi_ron_payments_use_one_bonus_allocation() {
        let rule = GameRule::default_mjsoul();
        assert_eq!(
            win_payments(&[win(1, 0, true, 3, false, vec![1])], 0, None, 2, 3, &rule)
                .unwrap()
                .deltas,
            [-2800, 8800, -1500, -1500]
        );
        // Input order does not decide which winner receives honba and deposits.
        let wins = [
            win(0, 0, false, 3, false, vec![1]),
            win(2, 0, false, 3, false, vec![1]),
        ];
        let paid = win_payments(&wins, 0, Some(1), 2, 1, &rule).unwrap();
        assert_eq!(paid.deltas, [7700, -13500, 6800, 0]);
        assert_eq!(
            paid.winners
                .iter()
                .map(|row| (row.player_id, row.honba))
                .collect::<Vec<_>>(),
            vec![(2, 2), (0, 0)]
        );
        assert_eq!(
            paid.winners
                .iter()
                .map(|row| row.deltas.iter().sum::<i32>())
                .sum::<i32>(),
            1000
        );
    }

    #[test]
    fn liability_preserves_rule_specific_share_of_other_yakuman() {
        let mut tsumo = win(1, 0, true, 26, true, vec![37, 38]);
        tsumo.liable_player = Some(3);
        tsumo.liable_yakuman_units = 1;
        assert_eq!(
            win_payments(&[tsumo.clone()], 0, None, 2, 0, &GameRule::default_mjsoul())
                .unwrap()
                .deltas,
            [-16000, 64600, -8000, -40600]
        );
        assert_eq!(
            win_payments(&[tsumo], 0, None, 2, 0, &GameRule::default_tenhou())
                .unwrap()
                .deltas,
            [0, 64600, 0, -64600]
        );
        let mut ron = win(1, 0, false, 26, true, vec![37, 38]);
        ron.liable_player = Some(3);
        ron.liable_yakuman_units = 1;
        assert_eq!(
            win_payments(&[ron], 0, Some(2), 2, 0, &GameRule::default_mjsoul())
                .unwrap()
                .deltas,
            [0, 64600, -48000, -16600]
        );
    }

    #[test]
    fn draw_pool_nagashi_and_final_pot_have_distinct_sources() {
        assert_eq!(
            draw_payments([true, true, false, false], 0, &[], false).unwrap(),
            [1500, 1500, -1500, -1500]
        );
        assert_eq!(
            draw_payments([true, true, false, false], 0, &[0, 1], false).unwrap(),
            [8000, 4000, -6000, -6000]
        );
        assert_eq!(
            draw_payments([true, true, false, false], 0, &[], true).unwrap(),
            [0; 4]
        );
        assert_eq!(
            end_game_pot([25000; 4], 2, [2, 1, 0, 3]).unwrap(),
            [0, 0, 2000, 0]
        );
        assert_eq!(
            ranked_seats([i32::MIN, 0, i32::MAX, 0], [3, 1, 0, 2]).unwrap(),
            [2, 3, 1, 0]
        );
    }

    #[test]
    fn mjsoul_west_draw_crossing_threshold_ends_despite_dealer_tenpai() {
        let context = west([26500, 31000, 21000, 21500]);
        let transition = mjsoul_transition(
            context,
            RoundEnd::ExhaustiveDraw {
                dealer_tenpai: true,
            },
        )
        .unwrap();
        assert!(transition.renchan);
        assert!(transition.match_end);
        assert_eq!(transition.reason, Some("extension_limit"));
        assert!(
            !mjsoul_transition(
                context,
                RoundEnd::Win {
                    dealer_won: true,
                    winner_count: 2
                }
            )
            .unwrap()
            .match_end
        );
        assert!(
            !mjsoul_transition(context, RoundEnd::AbortiveDraw)
                .unwrap()
                .match_end
        );
        let south = RoundContext {
            dealer: 3,
            round_wind: 1,
            ..context
        };
        assert!(
            !mjsoul_transition(
                south,
                RoundEnd::ExhaustiveDraw {
                    dealer_tenpai: true
                }
            )
            .unwrap()
            .match_end
        );
    }

    #[test]
    fn extension_limit_bankruptcy_and_dealer_yame_are_preserved() {
        let mut context = west([25000; 4]);
        context.dealer = 3;
        assert!(
            mjsoul_transition(
                context,
                RoundEnd::ExhaustiveDraw {
                    dealer_tenpai: false
                }
            )
            .unwrap()
            .match_end
        );
        assert!(
            !mjsoul_transition(
                context,
                RoundEnd::ExhaustiveDraw {
                    dealer_tenpai: true
                }
            )
            .unwrap()
            .match_end
        );
        assert!(
            mjsoul_transition(
                west([31000, 25000, 22000, 22000]),
                RoundEnd::Win {
                    dealer_won: true,
                    winner_count: 1
                }
            )
            .unwrap()
            .match_end
        );
        assert!(
            mjsoul_transition(west([-1, 33333, 33334, 33334]), RoundEnd::AbortiveDraw)
                .unwrap()
                .match_end
        );
    }

    #[test]
    fn malformed_or_overflowing_payment_contexts_are_rejected() {
        let rule = GameRule::default_mjsoul();
        let good = win(1, 0, false, 3, false, vec![1]);
        assert!(win_payments(std::slice::from_ref(&good), 0, Some(1), 0, 0, &rule).is_err());
        assert!(win_payments(&[good.clone(), good.clone()], 0, Some(0), 0, 0, &rule).is_err());
        assert!(win_payments(&[good], 0, Some(0), u32::MAX, 0, &rule).is_err());
        assert!(end_game_pot([25000; 4], u32::MAX, [0, 1, 2, 3]).is_err());
        assert!(draw_payments([false; 4], 0, &[4], false).is_err());
        assert!(ranked_seats([25000; 4], [0, 1, 1, 3]).is_err());
    }
}
