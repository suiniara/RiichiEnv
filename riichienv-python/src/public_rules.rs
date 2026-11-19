//! Native boundary accepts primitive public facts, never an environment,
//! observation object, hidden hand, or actual response list.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use riichienv_core::public_rules::{
    PublicDiscard, PublicRuleContext, PublicRules, RESPONSE_NAMES, nagashi_possibilities,
    public_response_mask,
};

#[pyfunction]
#[pyo3(name = "kar_public_response_mask", signature = (*,
    actor, candidate_tile, declares_riichi, discard_history, current_rivers, meld_kinds,
    riichi_accepted, remaining_draws, history_complete, chi_allowed, sanchaho_is_draw,
    four_winds_abort, four_riichi_abort, four_kans_abort, nagashi_mangan
))]
#[allow(clippy::too_many_arguments)]
fn public_response_mask_py(
    actor: usize,
    candidate_tile: u8,
    declares_riichi: bool,
    discard_history: [Vec<u8>; 4],
    current_rivers: [Vec<u8>; 4],
    meld_kinds: [Vec<String>; 4],
    riichi_accepted: [bool; 4],
    remaining_draws: Option<usize>,
    history_complete: bool,
    chi_allowed: bool,
    sanchaho_is_draw: bool,
    four_winds_abort: bool,
    four_riichi_abort: bool,
    four_kans_abort: bool,
    nagashi_mangan: bool,
) -> PyResult<Vec<bool>> {
    let mut kan_counts = [0; 4];
    for (seat, kinds) in meld_kinds.iter().enumerate() {
        for kind in kinds {
            match kind.as_str() {
                "ankan" | "kakan" | "daiminkan" => kan_counts[seat] += 1,
                "chi" | "pon" => {}
                _ => {
                    return Err(PyValueError::new_err(
                        "Expected a canonical committed public meld kind",
                    ));
                }
            }
        }
    }
    let context = PublicRuleContext {
        actor,
        candidate: PublicDiscard {
            tile_type: candidate_tile,
            declares_riichi,
        },
        discard_history: std::array::from_fn(|seat| discard_history[seat].as_slice()),
        current_rivers: std::array::from_fn(|seat| current_rivers[seat].as_slice()),
        meld_counts: std::array::from_fn(|seat| meld_kinds[seat].len()),
        kan_counts,
        riichi_accepted,
        remaining_draws,
        history_complete,
        rules: PublicRules {
            chi_allowed,
            sanchaho_is_draw,
            four_winds_abort,
            four_riichi_abort,
            four_kans_abort,
            nagashi_mangan,
        },
    };
    Ok(public_response_mask(&context)?.to_vec())
}

#[pyfunction]
#[pyo3(name = "kar_public_nagashi_possibilities", signature = (*,
    actor, candidate_tile, discard_history, current_rivers, history_complete
))]
fn nagashi_possibilities_py(
    actor: usize,
    candidate_tile: u8,
    discard_history: [Vec<u8>; 4],
    current_rivers: [Vec<u8>; 4],
    history_complete: bool,
) -> PyResult<Vec<bool>> {
    Ok(nagashi_possibilities(
        actor,
        candidate_tile,
        std::array::from_fn(|seat| discard_history[seat].as_slice()),
        std::array::from_fn(|seat| current_rivers[seat].as_slice()),
        history_complete,
    )?
    .to_vec())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(public_response_mask_py, module)?)?;
    module.add_function(wrap_pyfunction!(nagashi_possibilities_py, module)?)?;
    module.add("KAR_PUBLIC_RESPONSE_NAMES", RESPONSE_NAMES.to_vec())?;
    Ok(())
}
