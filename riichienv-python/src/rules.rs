//! Python adapters for the same pure settlement rules used by online play.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use riichienv_core::rule::GameRule;
use riichienv_core::rules::settlement::{self, EvaluatedWin, RoundContext, RoundEnd};
use riichienv_core::types::WinResult;

type RoundTransitionTuple = (u8, u8, u32, bool, bool, Option<&'static str>);

type PythonWin = (u8, WinResult, Option<u8>, u32);

#[pyfunction]
#[pyo3(signature = (wins, oya, target, honba, riichi_sticks))]
fn native_win_payments(
    wins: Vec<PythonWin>,
    oya: u8,
    target: Option<u8>,
    honba: u32,
    riichi_sticks: u32,
) -> PyResult<[i32; 4]> {
    let rule = GameRule::default_mjsoul();
    let wins: Vec<_> = wins
        .into_iter()
        .map(
            |(player_id, result, liable_player, liable_yakuman_units)| EvaluatedWin {
                player_id,
                result,
                liable_player,
                liable_yakuman_units,
            },
        )
        .collect();
    for win in &wins {
        settlement::validate_base_win(win, oya, target.is_none(), &rule)?;
    }
    Ok(settlement::win_payments(&wins, oya, target, honba, riichi_sticks, &rule)?.deltas)
}

#[pyfunction]
fn native_draw_payments(
    tenpai: [bool; 4],
    oya: u8,
    nagashi_winners: Vec<u8>,
    abortive: bool,
) -> PyResult<[i32; 4]> {
    Ok(settlement::draw_payments(
        tenpai,
        oya,
        &nagashi_winners,
        abortive,
    )?)
}

#[pyfunction]
fn native_end_game_pot(
    scores: [i32; 4],
    riichi_sticks: u32,
    initial_seat_order: [u8; 4],
) -> PyResult<[i32; 4]> {
    Ok(settlement::end_game_pot(
        scores,
        riichi_sticks,
        initial_seat_order,
    )?)
}

/// Compute four-player progression for an explicit rule profile.
/// `rule=None` selects Mahjong Soul for the KAR audit adapter.
#[pyfunction]
#[pyo3(signature = (scores, oya, round_wind, honba, game_mode, kind, dealer_continues, winner_count=1, initial_seat_order=[0, 1, 2, 3], rule=None))]
#[allow(clippy::too_many_arguments)]
fn native_round_transition(
    scores: [i32; 4],
    oya: u8,
    round_wind: u8,
    honba: u32,
    game_mode: u8,
    kind: &str,
    dealer_continues: bool,
    winner_count: u8,
    initial_seat_order: [u8; 4],
    rule: Option<GameRule>,
) -> PyResult<RoundTransitionTuple> {
    // This adapter is used by KAR's frozen Mahjong Soul audit.  Other rule
    // profiles must be supplied explicitly rather than inferred from scores.
    let rule = rule.unwrap_or_else(GameRule::default_mjsoul);
    let end = match kind {
        "win" => RoundEnd::Win {
            dealer_won: dealer_continues,
            winner_count,
        },
        "exhaustive_draw" => RoundEnd::ExhaustiveDraw {
            dealer_tenpai: dealer_continues,
        },
        "abortive_draw" => {
            if !dealer_continues {
                return Err(PyValueError::new_err(
                    "An abortive draw must repeat the dealer",
                ));
            }
            RoundEnd::AbortiveDraw
        }
        _ => return Err(PyValueError::new_err("Unknown round resolution kind")),
    };
    let transition = settlement::round_transition(
        RoundContext {
            scores,
            dealer: oya,
            round_wind,
            honba,
            game_mode,
            initial_seat_order,
        },
        end,
        &rule,
    )?;
    Ok((
        transition.round_wind,
        transition.dealer,
        transition.honba,
        transition.renchan,
        transition.match_end,
        transition.reason,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_win_payments, m)?)?;
    m.add_function(wrap_pyfunction!(native_draw_payments, m)?)?;
    m.add_function(wrap_pyfunction!(native_end_game_pot, m)?)?;
    m.add_function(wrap_pyfunction!(native_round_transition, m)?)?;
    Ok(())
}
