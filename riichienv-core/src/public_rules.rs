//! Conservative response predicates over an explicit public-information boundary.
//!
//! No GameState, hands, wall, private furiten, or actual response list is accepted.
//! A true bit means public facts have not ruled an event out, not that a hidden
//! opponent can legally choose it. Missing history remains unknown.

use crate::errors::{RiichiError, RiichiResult};
use crate::types::is_terminal_tile;

pub const RESPONSE_COUNT: usize = 21;
pub const RESPONSE_NAMES: [&str; RESPONSE_COUNT] = [
    "NEXT_DRAW",
    "CHI_P1",
    "PON_P1",
    "PON_P2",
    "PON_P3",
    "DAIMINKAN_P1",
    "DAIMINKAN_P2",
    "DAIMINKAN_P3",
    "RON_MASK_1",
    "RON_MASK_2",
    "RON_MASK_3",
    "RON_MASK_4",
    "RON_MASK_5",
    "RON_MASK_6",
    "RON_MASK_7",
    "EXHAUSTIVE_DRAW",
    "ABORT_FOUR_WINDS",
    "ABORT_FOUR_RIICHI",
    "ABORT_FOUR_KANS",
    "ABORT_TRIPLE_RON",
    "NAGASHI_MANGAN",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicDiscard {
    /// Red and ordinary fives share one base-34 type.
    pub tile_type: u8,
    pub declares_riichi: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicRules {
    pub chi_allowed: bool,
    pub sanchaho_is_draw: bool,
    pub four_winds_abort: bool,
    pub four_riichi_abort: bool,
    pub four_kans_abort: bool,
    pub nagashi_mangan: bool,
}

impl Default for PublicRules {
    fn default() -> Self {
        Self {
            chi_allowed: true,
            sanchaho_is_draw: false,
            four_winds_abort: true,
            four_riichi_abort: true,
            four_kans_abort: true,
            nagashi_mangan: true,
        }
    }
}

/// The entire query whitelist. Histories/rivers contain public physical tile
/// IDs (0..136); called-away tiles remain only in discard_history.
#[derive(Debug, Clone, Copy)]
pub struct PublicRuleContext<'a> {
    pub actor: usize,
    pub candidate: PublicDiscard,
    pub discard_history: [&'a [u8]; 4],
    pub current_rivers: [&'a [u8]; 4],
    pub meld_counts: [usize; 4],
    pub kan_counts: [usize; 4],
    pub riichi_accepted: [bool; 4],
    pub remaining_draws: Option<usize>,
    pub history_complete: bool,
    pub rules: PublicRules,
}

fn invalid(message: &str) -> RiichiError {
    RiichiError::InvalidState {
        message: message.to_owned(),
    }
}

fn validate_public_discards(
    actor: usize,
    candidate_tile: u8,
    history: &[&[u8]; 4],
    rivers: &[&[u8]; 4],
) -> RiichiResult<()> {
    if actor >= 4 || candidate_tile >= 34 {
        return Err(invalid(
            "A public discard requires a four-player actor and base-34 tile",
        ));
    }
    if history
        .iter()
        .chain(rivers.iter())
        .flat_map(|tiles| tiles.iter())
        .any(|&tile| tile >= 136)
    {
        return Err(invalid(
            "Public discard history contains an invalid physical tile",
        ));
    }
    Ok(())
}

/// Shared online/offline predicate; arguments describe the state after discard.
pub fn four_winds_abort(
    discard_counts: [usize; 4],
    first_discards: [Option<u8>; 4],
    meld_counts: [usize; 4],
) -> bool {
    discard_counts == [1; 4]
        && meld_counts == [0; 4]
        && first_discards[0].is_some_and(|tile| (27..=30).contains(&(tile / 4)))
        && first_discards
            .iter()
            .all(|tile| tile.map(|value| value / 4) == first_discards[0].map(|value| value / 4))
}

/// Four kans by one owner do not cause the multi-owner abort.
pub fn four_kans_abort(kan_counts: [usize; 4]) -> bool {
    kan_counts.iter().sum::<usize>() >= 4
        && kan_counts.iter().filter(|&&count| count > 0).count() > 1
}

pub fn four_riichi_abort(accepted: [bool; 4]) -> bool {
    accepted.into_iter().all(|value| value)
}

/// Own melds do not disqualify Mahjong Soul nagashi. Called-away evidence must
/// be known, not inferred merely from the absence of incomplete history.
pub fn nagashi_eligible(only_terminal_discards: bool, any_discard_called: bool) -> bool {
    only_terminal_discards && !any_discard_called
}

pub fn nagashi_player_possible(
    history: &[u8],
    river: &[u8],
    candidate_tile: Option<u8>,
    history_complete: bool,
) -> bool {
    let terminals = history
        .iter()
        .chain(river.iter())
        .all(|&tile| is_terminal_tile(tile))
        && candidate_tile.is_none_or(|tile| is_terminal_tile(tile * 4));
    nagashi_eligible(terminals, history_complete && history.len() != river.len())
}

pub fn nagashi_possibilities(
    actor: usize,
    candidate_tile: u8,
    history: [&[u8]; 4],
    rivers: [&[u8]; 4],
    history_complete: bool,
) -> RiichiResult<[bool; 4]> {
    validate_public_discards(actor, candidate_tile, &history, &rivers)?;
    Ok(std::array::from_fn(|seat| {
        nagashi_player_possible(
            history[seat],
            rivers[seat],
            (seat == actor).then_some(candidate_tile),
            history_complete,
        )
    }))
}

/// Native priority: winds, kans, riichi. Possible-but-unproved aborts must not
/// suppress ordinary continuation; Ron keeps priority over all three.
fn abortive_after_discard(context: &PublicRuleContext<'_>) -> ([bool; 3], [bool; 3]) {
    let actor = context.actor;
    let tile = context.candidate.tile_type;
    let history = context.discard_history;
    let wind_negative = context.meld_counts.iter().any(|&count| count > 0)
        || !(27..=30).contains(&tile)
        || !history[actor].is_empty()
        || (0..4).filter(|&seat| seat != actor).any(|seat| {
            history[seat].len() > 1 || history[seat].first().is_some_and(|first| first / 4 != tile)
        });
    let counts = std::array::from_fn(|seat| history[seat].len() + usize::from(seat == actor));
    let first = std::array::from_fn(|seat| {
        history[seat]
            .first()
            .copied()
            .or_else(|| (seat == actor).then_some(tile * 4))
    });
    let wind_certain = context.rules.four_winds_abort
        && context.history_complete
        && four_winds_abort(counts, first, context.meld_counts);
    let winds = context.rules.four_winds_abort
        && !wind_negative
        && (wind_certain || !context.history_complete);
    let kans_certain = context.rules.four_kans_abort && four_kans_abort(context.kan_counts);
    let kans = context.rules.four_kans_abort
        && (kans_certain
            || (!context.history_complete && context.kan_counts.iter().sum::<usize>() < 4));
    let mut accepted = context.riichi_accepted;
    accepted[actor] |= context.candidate.declares_riichi;
    let riichi_certain = context.rules.four_riichi_abort && four_riichi_abort(accepted);
    let riichi = context.rules.four_riichi_abort && (riichi_certain || !context.history_complete);
    (
        [winds, kans, riichi],
        [wind_certain, kans_certain, riichi_certain],
    )
}

pub fn public_response_mask(
    context: &PublicRuleContext<'_>,
) -> RiichiResult<[bool; RESPONSE_COUNT]> {
    validate_public_discards(
        context.actor,
        context.candidate.tile_type,
        &context.discard_history,
        &context.current_rivers,
    )?;
    if (0..4).any(|seat| {
        context.kan_counts[seat] > context.meld_counts[seat] || context.meld_counts[seat] > 4
    }) {
        return Err(invalid("Public meld and kan counts are inconsistent"));
    }
    let mut allowed = [true; RESPONSE_COUNT];
    let tile = context.candidate.tile_type;
    allowed[1] = context.rules.chi_allowed && tile < 27;
    allowed[19] = context.rules.sanchaho_is_draw;
    allowed[14] = !context.rules.sanchaho_is_draw;
    for relative in 1..4 {
        let player = (context.actor + relative) % 4;
        if context.riichi_accepted[player] {
            allowed[1 + relative] = false;
            allowed[4 + relative] = false;
            if relative == 1 {
                allowed[1] = false;
            }
        }
        if context.history_complete
            && context.discard_history[player]
                .iter()
                .any(|&value| value / 4 == tile)
        {
            for ron_mask in 1..8 {
                if ron_mask & (1 << (relative - 1)) != 0 {
                    allowed[7 + ron_mask] = false;
                }
            }
        }
    }
    if context.kan_counts.iter().sum::<usize>() >= 4 {
        allowed[5..8].fill(false);
    }
    if let Some(remaining) = context.remaining_draws {
        allowed[0] = remaining > 0;
        allowed[15] = remaining == 0;
        if remaining == 0 {
            allowed[1..8].fill(false);
        }
    }
    let (possible, certain) = abortive_after_discard(context);
    for (index, value) in [16, 18, 17].into_iter().zip(possible) {
        allowed[index] = value;
    }
    if certain[0] {
        allowed[18] = false;
        allowed[17] = false;
    } else if certain[1] {
        allowed[17] = false;
    }
    let abort_certain = certain.into_iter().any(|value| value);
    if abort_certain {
        allowed[..8].fill(false);
        allowed[15] = false;
    }
    allowed[20] = !abort_certain
        && context.rules.nagashi_mangan
        && context
            .remaining_draws
            .is_none_or(|remaining| remaining == 0)
        && (0..4).any(|seat| {
            nagashi_player_possible(
                context.discard_history[seat],
                context.current_rivers[seat],
                (seat == context.actor).then_some(tile),
                context.history_complete,
            )
        });
    if !allowed.iter().any(|&value| value) {
        return Err(invalid("Public response mask excludes every category"));
    }
    Ok(allowed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> PublicRuleContext<'static> {
        PublicRuleContext {
            actor: 0,
            candidate: PublicDiscard {
                tile_type: 0,
                declares_riichi: false,
            },
            discard_history: [&[]; 4],
            current_rivers: [&[]; 4],
            meld_counts: [0; 4],
            kan_counts: [0; 4],
            riichi_accepted: [false; 4],
            remaining_draws: Some(70),
            history_complete: true,
            rules: PublicRules::default(),
        }
    }

    #[test]
    fn missing_history_is_not_proof_of_an_abort_or_furiten() {
        let mut value = context();
        value.history_complete = false;
        value.candidate.tile_type = 27;
        value.discard_history = [&[], &[108], &[], &[]];
        let mask = public_response_mask(&value).unwrap();
        assert!(mask[0] && mask[8] && mask[16] && mask[17] && mask[18]);
        value.history_complete = true;
        let complete = public_response_mask(&value).unwrap();
        assert!(complete[0]);
        assert!(!complete[8] && !complete[16] && !complete[17] && !complete[18]);
    }

    #[test]
    fn proven_abort_preserves_ron_priority_even_with_partial_history() {
        let mut value = context();
        value.history_complete = false;
        value.meld_counts = [2, 2, 0, 0];
        value.kan_counts = value.meld_counts;
        let mask = public_response_mask(&value).unwrap();
        assert!(mask[18] && mask[8] && mask[14]);
        assert!(!mask[..8].iter().any(|&bit| bit) && !mask[15] && !mask[17] && !mask[20]);
        value.meld_counts = [4, 0, 0, 0];
        value.kan_counts = value.meld_counts;
        let one_owner = public_response_mask(&value).unwrap();
        assert!(one_owner[0] && !one_owner[18]);
    }

    #[test]
    fn final_discard_keeps_ron_but_disables_every_open_call() {
        let mut value = context();
        value.remaining_draws = Some(0);
        let mask = public_response_mask(&value).unwrap();
        assert!(!mask[..8].iter().any(|&bit| bit));
        assert!(mask[8] && mask[15] && mask[20]);
    }

    #[test]
    fn partial_nagashi_still_rejects_observed_middle_tiles() {
        let history: [&[u8]; 4] = [&[0], &[32], &[108], &[4]];
        let rivers: [&[u8]; 4] = [&[], &[32], &[108], &[4]];
        assert_eq!(
            nagashi_possibilities(1, 1, history, rivers, false).unwrap(),
            [true, false, true, false]
        );
        assert_eq!(
            nagashi_possibilities(1, 1, history, rivers, true).unwrap(),
            [false, false, true, false]
        );
    }

    #[test]
    fn four_winds_checks_tile_types_and_interruption() {
        assert!(four_winds_abort(
            [1; 4],
            [Some(108), Some(109), Some(110), Some(111)],
            [0; 4]
        ));
        assert!(!four_winds_abort(
            [1; 4],
            [Some(108), Some(112), Some(110), Some(111)],
            [0; 4]
        ));
        assert!(!four_winds_abort([1; 4], [Some(108); 4], [0, 1, 0, 0]));
    }

    #[test]
    fn malformed_context_is_rejected_instead_of_reading_other_state() {
        let mut value = context();
        value.actor = 4;
        assert!(public_response_mask(&value).is_err());
        value.actor = 0;
        value.kan_counts = [1, 0, 0, 0];
        assert!(public_response_mask(&value).is_err());
        value.kan_counts = [0; 4];
        value.discard_history[1] = &[136];
        assert!(public_response_mask(&value).is_err());
    }
}
