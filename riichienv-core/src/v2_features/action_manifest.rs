//! Exact 546-action manifest mirror of `src/riichienv/kar/actions.py`.
//!
//! Provides:
//! - Exact static lookup table for all 546 global action IDs (+ slot 546 = PAD)
//! - Bijective mapping between native [`Action`]s + [`Observation`] and sorted
//!   `global_id ∈ [0, 545]`
//! - Hard runtime enforcement of `MAX_CANDIDATES = 32`
//! - Explicit `NULL_TILE` (34) / `NULL_SOURCE` (4) + `has_slot_flags` contract

use std::sync::OnceLock;

use crate::action::{Action, ActionType};
use crate::observation::Observation;
use crate::types::MeldType;

use super::{
    ACTION_COUNT, MAX_CANDIDATES, NULL_SOURCE_IDX, NULL_TILE_IDX, is_red5_tile37, tile37_to_base34,
    tile136_to_tile37,
};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManifestActionKind {
    Discard = 0,
    Ankan = 1,
    Kakan = 2,
    Tsumo = 3,
    KyushuKyuhai = 4,
    Pass = 5,
    Chi = 6,
    Pon = 7,
    Daiminkan = 8,
    Ron = 9,
    Pad = 10,
}

/// Static structured specification for one entry of the 546-action manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionManifestEntry {
    pub global_id: u16,
    pub kind: ManifestActionKind,
    /// `0..36` if present, `37` if NONE.
    pub tile37: u8,
    /// `0..33` if present, `34` (`NULL_TILE_IDX`) if NONE.
    pub target_base34: u8,
    /// `0..36` if present, `37` if NONE.
    pub consumed0_tile37: u8,
    /// `0..36` if present, `37` if NONE.
    pub consumed1_tile37: u8,
    /// `0..33` if present, `34` (`NULL_TILE_IDX`) if NONE.
    pub consumed0_base34: u8,
    /// `0..33` if present, `34` (`NULL_TILE_IDX`) if NONE.
    pub consumed1_base34: u8,
    /// `0..3` relative source if present, `4` (`NULL_SOURCE_IDX`) if NONE.
    pub source_rel: u8,
    pub tsumogiri: bool,
    pub riichi: bool,
    /// `[has_target, has_consumed0, has_consumed1, has_source]`
    pub has_slot_flags: [bool; 4],
    /// `[target_is_red5, consumed_has_red5, is_tsumogiri, is_riichi]`
    pub static_flags: [bool; 4],
}

pub const TILE37_NONE: u8 = 37;

const CHI_PATTERNS: [(u8, u8, u8); 30] = [
    (2, 3, 1),
    (1, 3, 2),
    (3, 4, 2),
    (1, 2, 3),
    (2, 4, 3),
    (4, 5, 3),
    (4, 0, 3),
    (2, 3, 4),
    (3, 5, 4),
    (3, 0, 4),
    (5, 6, 4),
    (0, 6, 4),
    (3, 4, 5),
    (3, 4, 0),
    (4, 6, 5),
    (4, 6, 0),
    (6, 7, 5),
    (6, 7, 0),
    (4, 5, 6),
    (4, 0, 6),
    (5, 7, 6),
    (0, 7, 6),
    (7, 8, 6),
    (5, 6, 7),
    (0, 6, 7),
    (6, 8, 7),
    (8, 9, 7),
    (6, 7, 8),
    (7, 9, 8),
    (7, 8, 9),
];

fn pon_patterns() -> Vec<(u8, u8, u8)> {
    let mut patterns = Vec::with_capacity(40);
    for suit in 0..3u8 {
        for rank in 1..=9u8 {
            let normal = suit * 10 + rank;
            patterns.push((normal, normal, normal));
            if rank == 5 {
                let red = suit * 10;
                patterns.push((red, normal, normal));
                patterns.push((normal, normal, red));
            }
        }
    }
    for tile in 30..37u8 {
        patterns.push((tile, tile, tile));
    }
    patterns
}

fn build_manifest() -> Vec<ActionManifestEntry> {
    let mut entries = Vec::with_capacity(ACTION_COUNT + 1);

    // 0..147: DISCARD (37 * 2 * 2 = 148)
    for tile37 in 0..37u8 {
        let base34 = tile37_to_base34(tile37);
        let red = is_red5_tile37(tile37);
        for tsumogiri in [false, true] {
            for riichi in [false, true] {
                entries.push(ActionManifestEntry {
                    global_id: entries.len() as u16,
                    kind: ManifestActionKind::Discard,
                    tile37,
                    target_base34: base34,
                    consumed0_tile37: TILE37_NONE,
                    consumed1_tile37: TILE37_NONE,
                    consumed0_base34: NULL_TILE_IDX,
                    consumed1_base34: NULL_TILE_IDX,
                    source_rel: NULL_SOURCE_IDX,
                    tsumogiri,
                    riichi,
                    has_slot_flags: [true, false, false, false],
                    static_flags: [red, false, tsumogiri, riichi],
                });
            }
        }
    }
    debug_assert_eq!(entries.len(), 148);

    // 148..181: ANKAN (34)
    for base34 in 0..34u8 {
        entries.push(ActionManifestEntry {
            global_id: entries.len() as u16,
            kind: ManifestActionKind::Ankan,
            tile37: TILE37_NONE,
            target_base34: base34,
            consumed0_tile37: TILE37_NONE,
            consumed1_tile37: TILE37_NONE,
            consumed0_base34: NULL_TILE_IDX,
            consumed1_base34: NULL_TILE_IDX,
            source_rel: NULL_SOURCE_IDX,
            tsumogiri: false,
            riichi: false,
            has_slot_flags: [true, false, false, false],
            static_flags: [false, false, false, false],
        });
    }
    debug_assert_eq!(entries.len(), 182);

    // 182..218: KAKAN (37)
    for tile37 in 0..37u8 {
        let base34 = tile37_to_base34(tile37);
        let red = is_red5_tile37(tile37);
        entries.push(ActionManifestEntry {
            global_id: entries.len() as u16,
            kind: ManifestActionKind::Kakan,
            tile37,
            target_base34: base34,
            consumed0_tile37: TILE37_NONE,
            consumed1_tile37: TILE37_NONE,
            consumed0_base34: NULL_TILE_IDX,
            consumed1_base34: NULL_TILE_IDX,
            source_rel: NULL_SOURCE_IDX,
            tsumogiri: false,
            riichi: false,
            has_slot_flags: [true, false, false, false],
            static_flags: [red, false, false, false],
        });
    }
    debug_assert_eq!(entries.len(), 219);

    // 219..221: TSUMO, KYUSHU_KYUHAI, PASS
    for kind in [
        ManifestActionKind::Tsumo,
        ManifestActionKind::KyushuKyuhai,
        ManifestActionKind::Pass,
    ] {
        entries.push(ActionManifestEntry {
            global_id: entries.len() as u16,
            kind,
            tile37: TILE37_NONE,
            target_base34: NULL_TILE_IDX,
            consumed0_tile37: TILE37_NONE,
            consumed1_tile37: TILE37_NONE,
            consumed0_base34: NULL_TILE_IDX,
            consumed1_base34: NULL_TILE_IDX,
            source_rel: NULL_SOURCE_IDX,
            tsumogiri: false,
            riichi: false,
            has_slot_flags: [false, false, false, false],
            static_flags: [false, false, false, false],
        });
    }
    debug_assert_eq!(entries.len(), 222);

    // 222..311: CHI (3 * 30 = 90)
    for suit in 0..3u8 {
        for &(a, b, claimed) in &CHI_PATTERNS {
            let tile37 = suit * 10 + claimed;
            let mut c0 = suit * 10 + a;
            let mut c1 = suit * 10 + b;
            if c0 > c1 {
                std::mem::swap(&mut c0, &mut c1);
            }
            let red_t = is_red5_tile37(tile37);
            let red_c = is_red5_tile37(c0) || is_red5_tile37(c1);
            entries.push(ActionManifestEntry {
                global_id: entries.len() as u16,
                kind: ManifestActionKind::Chi,
                tile37,
                target_base34: tile37_to_base34(tile37),
                consumed0_tile37: c0,
                consumed1_tile37: c1,
                consumed0_base34: tile37_to_base34(c0),
                consumed1_base34: tile37_to_base34(c1),
                source_rel: 3,
                tsumogiri: false,
                riichi: false,
                has_slot_flags: [true, true, true, true],
                static_flags: [red_t, red_c, false, false],
            });
        }
    }
    debug_assert_eq!(entries.len(), 312);

    // 312..431: PON (3 * 40 = 120)
    let pons = pon_patterns();
    for source in 1..=3u8 {
        for &(a, b, claimed) in &pons {
            let mut c0 = a;
            let mut c1 = b;
            if c0 > c1 {
                std::mem::swap(&mut c0, &mut c1);
            }
            let red_t = is_red5_tile37(claimed);
            let red_c = is_red5_tile37(c0) || is_red5_tile37(c1);
            entries.push(ActionManifestEntry {
                global_id: entries.len() as u16,
                kind: ManifestActionKind::Pon,
                tile37: claimed,
                target_base34: tile37_to_base34(claimed),
                consumed0_tile37: c0,
                consumed1_tile37: c1,
                consumed0_base34: tile37_to_base34(c0),
                consumed1_base34: tile37_to_base34(c1),
                source_rel: source,
                tsumogiri: false,
                riichi: false,
                has_slot_flags: [true, true, true, true],
                static_flags: [red_t, red_c, false, false],
            });
        }
    }
    debug_assert_eq!(entries.len(), 432);

    // 432..542: DAIMINKAN (3 * 37 = 111)
    for source in 1..=3u8 {
        for tile37 in 0..37u8 {
            let red_t = is_red5_tile37(tile37);
            entries.push(ActionManifestEntry {
                global_id: entries.len() as u16,
                kind: ManifestActionKind::Daiminkan,
                tile37,
                target_base34: tile37_to_base34(tile37),
                consumed0_tile37: TILE37_NONE,
                consumed1_tile37: TILE37_NONE,
                consumed0_base34: NULL_TILE_IDX,
                consumed1_base34: NULL_TILE_IDX,
                source_rel: source,
                tsumogiri: false,
                riichi: false,
                has_slot_flags: [true, false, false, true],
                static_flags: [red_t, false, false, false],
            });
        }
    }
    debug_assert_eq!(entries.len(), 543);

    // 543..545: RON (3)
    for source in 1..=3u8 {
        entries.push(ActionManifestEntry {
            global_id: entries.len() as u16,
            kind: ManifestActionKind::Ron,
            tile37: TILE37_NONE,
            target_base34: NULL_TILE_IDX,
            consumed0_tile37: TILE37_NONE,
            consumed1_tile37: TILE37_NONE,
            consumed0_base34: NULL_TILE_IDX,
            consumed1_base34: NULL_TILE_IDX,
            source_rel: source,
            tsumogiri: false,
            riichi: false,
            has_slot_flags: [false, false, false, true],
            static_flags: [false, false, false, false],
        });
    }
    debug_assert_eq!(entries.len(), ACTION_COUNT);

    // 546: PAD
    entries.push(ActionManifestEntry {
        global_id: ACTION_COUNT as u16,
        kind: ManifestActionKind::Pad,
        tile37: TILE37_NONE,
        target_base34: NULL_TILE_IDX,
        consumed0_tile37: TILE37_NONE,
        consumed1_tile37: TILE37_NONE,
        consumed0_base34: NULL_TILE_IDX,
        consumed1_base34: NULL_TILE_IDX,
        source_rel: NULL_SOURCE_IDX,
        tsumogiri: false,
        riichi: false,
        has_slot_flags: [false, false, false, false],
        static_flags: [false, false, false, false],
    });

    entries
}

static MANIFEST: OnceLock<Vec<ActionManifestEntry>> = OnceLock::new();

pub fn manifest() -> &'static [ActionManifestEntry] {
    MANIFEST.get_or_init(build_manifest)
}

pub fn entry(global_id: u16) -> Result<&'static ActionManifestEntry, String> {
    let table = manifest();
    if usize::from(global_id) >= ACTION_COUNT {
        return Err(format!(
            "global_id {global_id} out of range [0, {ACTION_COUNT})"
        ));
    }
    Ok(&table[usize::from(global_id)])
}

pub fn lookup_discard_id(tile37: u8, tsumogiri: bool, riichi: bool) -> Result<u16, String> {
    if tile37 >= 37 {
        return Err(format!("invalid tile37 {tile37} for DISCARD"));
    }
    Ok(u16::from(tile37) * 4 + u16::from(tsumogiri) * 2 + u16::from(riichi))
}

pub fn lookup_ankan_id(base34: u8) -> Result<u16, String> {
    if base34 >= 34 {
        return Err(format!("invalid base34 {base34} for ANKAN"));
    }
    Ok(148 + u16::from(base34))
}

pub fn lookup_kakan_id(tile37: u8) -> Result<u16, String> {
    if tile37 >= 37 {
        return Err(format!("invalid tile37 {tile37} for KAKAN"));
    }
    Ok(182 + u16::from(tile37))
}

pub fn lookup_chi_id(claimed_tile37: u8, c0_tile37: u8, c1_tile37: u8) -> Result<u16, String> {
    let (lo, hi) = if c0_tile37 <= c1_tile37 {
        (c0_tile37, c1_tile37)
    } else {
        (c1_tile37, c0_tile37)
    };
    manifest()[222..312]
        .iter()
        .find(|e| {
            e.tile37 == claimed_tile37 && e.consumed0_tile37 == lo && e.consumed1_tile37 == hi
        })
        .map(|e| e.global_id)
        .ok_or_else(|| format!("unknown CHI ({claimed_tile37}, [{lo}, {hi}])"))
}

pub fn lookup_pon_id(
    source_rel: u8,
    claimed_tile37: u8,
    c0_tile37: u8,
    c1_tile37: u8,
) -> Result<u16, String> {
    if !(1..=3).contains(&source_rel) {
        return Err(format!("invalid PON source_rel {source_rel}"));
    }
    let (lo, hi) = if c0_tile37 <= c1_tile37 {
        (c0_tile37, c1_tile37)
    } else {
        (c1_tile37, c0_tile37)
    };
    manifest()[312..432]
        .iter()
        .find(|e| {
            e.source_rel == source_rel
                && e.tile37 == claimed_tile37
                && e.consumed0_tile37 == lo
                && e.consumed1_tile37 == hi
        })
        .map(|e| e.global_id)
        .ok_or_else(|| format!("unknown PON (src={source_rel}, {claimed_tile37}, [{lo}, {hi}])"))
}

pub fn lookup_daiminkan_id(source_rel: u8, claimed_tile37: u8) -> Result<u16, String> {
    if !(1..=3).contains(&source_rel) {
        return Err(format!("invalid DAIMINKAN source_rel {source_rel}"));
    }
    if claimed_tile37 >= 37 {
        return Err(format!("invalid tile37 {claimed_tile37} for DAIMINKAN"));
    }
    Ok(432 + u16::from(source_rel - 1) * 37 + u16::from(claimed_tile37))
}

pub fn lookup_ron_id(source_rel: u8) -> Result<u16, String> {
    if !(1..=3).contains(&source_rel) {
        return Err(format!("invalid RON source_rel {source_rel}"));
    }
    Ok(542 + u16::from(source_rel))
}

/// One resolved candidate action bridging a canonical 546 `global_id` to the
/// concrete native execution step(s) in `riichienv-core`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCandidate {
    pub global_id: u16,
    pub entry: &'static ActionManifestEntry,
    /// Native execution sequence: 1 action normally, or `[Riichi, Discard]` for
    /// a riichi declaration discard.
    pub native_actions: Vec<Action>,
}

/// Project `obs._legal_actions` into a strictly ascending list of canonical
/// 546-action [`ResolvedCandidate`]s, enforcing `len <= MAX_CANDIDATES (32)`.
///
/// `source_rel_hint` provides the relative seat (`1..=3`) of the trigger
/// discard/kan player in response phases (`None` in `OwnAct`).
pub fn resolve_legal_candidates(
    obs: &Observation,
    source_rel_hint: Option<u8>,
) -> Result<Vec<ResolvedCandidate>, String> {
    let pid = obs.player_id as usize;
    if pid >= 4 {
        return Err(format!("invalid observer player_id {}", obs.player_id));
    }

    let hand = &obs.hands[pid];
    let drawn_tile = obs.drawn_tile;
    let legal = obs.legal_actions_method();

    // Count copies of each tile37 in own hand to distinguish tsumogiri vs tedashi
    // when holding both the drawn copy and another copy of the same tile37.
    let mut hand_tile37_counts = [0u8; 37];
    for &t in hand {
        if t >= 136 {
            return Err(format!("invalid hand tile136 {t}"));
        }
        hand_tile37_counts[tile136_to_tile37(t as u8) as usize] += 1;
    }

    // Collect riichi discard tile136 candidates if Riichi is offered.
    let riichi_action = legal
        .iter()
        .find(|a| a.action_type == ActionType::Riichi)
        .cloned();
    let riichi_discards: Vec<Action> = if riichi_action.is_some() {
        obs.legal_riichi_discard_actions()?
    } else {
        Vec::new()
    };

    let mut resolved: Vec<ResolvedCandidate> = Vec::with_capacity(MAX_CANDIDATES);
    let mut seen_ids = [false; ACTION_COUNT];

    let mut push_unique = |global_id: u16, native_actions: Vec<Action>| -> Result<(), String> {
        let idx = usize::from(global_id);
        if idx >= ACTION_COUNT {
            return Err(format!("global_id {global_id} out of range"));
        }
        if !seen_ids[idx] {
            seen_ids[idx] = true;
            resolved.push(ResolvedCandidate {
                global_id,
                entry: &manifest()[idx],
                native_actions,
            });
        }
        Ok(())
    };

    // Helper: classify whether a native Discard action represents tsumogiri and/or tedashi.
    let discard_tsumogiri_modes = |tile136: u8| -> (bool, bool) {
        if obs.forced_tedashi {
            return (false, true);
        }
        match drawn_tile {
            Some(dt) if dt == tile136 => {
                // Physical drawn tile itself is tsumogiri. If another physical copy
                // with identical tile37 exists in hand, a separate Discard action
                // with that copy (or same tile37) represents tedashi.
                (true, false)
            }
            Some(dt) if tile136_to_tile37(dt) == tile136_to_tile37(tile136) => (false, true),
            _ => (false, true),
        }
    };

    // Note: in `riichienv-core`, legal_actions deduplicates discards by base34 or
    // tile37; if drawn_tile is present and hand has >= 2 copies of drawn tile37,
    // check whether both tedashi and tsumogiri physical tiles appear in `legal`.
    let has_physical_drawn_discard = drawn_tile.is_some_and(|dt| {
        legal
            .iter()
            .any(|a| a.action_type == ActionType::Discard && a.tile == Some(dt))
    });
    let has_other_same_t37_discard = drawn_tile.is_some_and(|dt| {
        let dt37 = tile136_to_tile37(dt);
        legal.iter().any(|a| {
            a.action_type == ActionType::Discard
                && a.tile
                    .is_some_and(|t| t != dt && tile136_to_tile37(t) == dt37)
        })
    });

    for action in &legal {
        match action.action_type {
            ActionType::Discard => {
                let tile136 = action
                    .tile
                    .ok_or_else(|| "DISCARD action missing tile".to_string())?;
                let tile37 = tile136_to_tile37(tile136);
                let (is_tsumo, is_ted) = discard_tsumogiri_modes(tile136);
                if is_tsumo {
                    let gid = lookup_discard_id(tile37, true, false)?;
                    push_unique(gid, vec![action.clone()])?;
                    // If engine collapsed duplicate tile37 discards into the drawn
                    // physical tile while hand holds >= 2 copies of this tile37,
                    // also expose the tedashi variant mapped to the non-drawn physical copy.
                    if !obs.forced_tedashi
                        && !has_other_same_t37_discard
                        && hand_tile37_counts[tile37 as usize] >= 2
                        && let Some(other_t136) = hand
                            .iter()
                            .copied()
                            .map(|x| x as u8)
                            .find(|&t| t != tile136 && tile136_to_tile37(t) == tile37)
                    {
                        let ted_action = Action::new(
                            ActionType::Discard,
                            Some(other_t136),
                            vec![],
                            action.actor,
                        );
                        let ted_gid = lookup_discard_id(tile37, false, false)?;
                        push_unique(ted_gid, vec![ted_action])?;
                    }
                }
                if is_ted {
                    let gid = lookup_discard_id(tile37, false, false)?;
                    push_unique(gid, vec![action.clone()])?;
                    // Symmetrically, if engine emitted only the non-drawn physical copy
                    // while drawn_tile has identical tile37, expose tsumogiri mapped to drawn_tile.
                    if !obs.forced_tedashi
                        && !has_physical_drawn_discard
                        && let Some(dt) = drawn_tile
                        && tile136_to_tile37(dt) == tile37
                    {
                        let tsumo_action =
                            Action::new(ActionType::Discard, Some(dt), vec![], action.actor);
                        let tsumo_gid = lookup_discard_id(tile37, true, false)?;
                        push_unique(tsumo_gid, vec![tsumo_action])?;
                    }
                }
            }
            ActionType::Riichi => {
                // Expanded below via `riichi_discards`.
            }
            ActionType::Ankan => {
                let base34 = if let Some(t) = action.tile {
                    t / 4
                } else if let Some(&c) = action.consume_tiles.first() {
                    c / 4
                } else {
                    return Err("ANKAN action missing tile and consume_tiles".to_string());
                };
                let gid = lookup_ankan_id(base34)?;
                push_unique(gid, vec![action.clone()])?;
            }
            ActionType::Kakan => {
                let tile136 = action
                    .tile
                    .ok_or_else(|| "KAKAN action missing tile".to_string())?;
                let tile37 = tile136_to_tile37(tile136);
                let base34 = tile37_to_base34(tile37);
                // Contract v2.4: verify unique upgradeable PON slot in observer's melds.
                let pon_count = obs.melds[pid]
                    .iter()
                    .filter(|m| {
                        m.meld_type == MeldType::Pon
                            && m.tiles.first().is_some_and(|&t| t / 4 == base34)
                    })
                    .count();
                if pon_count != 1 {
                    return Err(format!(
                        "KAKAN on base34 {base34} expected exactly 1 existing PON, found {pon_count}"
                    ));
                }
                let gid = lookup_kakan_id(tile37)?;
                push_unique(gid, vec![action.clone()])?;
            }
            ActionType::Tsumo => {
                push_unique(219, vec![action.clone()])?;
            }
            ActionType::KyushuKyuhai => {
                push_unique(220, vec![action.clone()])?;
            }
            ActionType::Pass => {
                push_unique(221, vec![action.clone()])?;
            }
            ActionType::Chi => {
                let claimed = action
                    .tile
                    .ok_or_else(|| "CHI action missing claimed tile".to_string())?;
                if action.consume_tiles.len() != 2 {
                    return Err(format!(
                        "CHI action expected 2 consume_tiles, got {}",
                        action.consume_tiles.len()
                    ));
                }
                let gid = lookup_chi_id(
                    tile136_to_tile37(claimed),
                    tile136_to_tile37(action.consume_tiles[0]),
                    tile136_to_tile37(action.consume_tiles[1]),
                )?;
                push_unique(gid, vec![action.clone()])?;
            }
            ActionType::Pon => {
                let claimed = action
                    .tile
                    .ok_or_else(|| "PON action missing claimed tile".to_string())?;
                if action.consume_tiles.len() != 2 {
                    return Err(format!(
                        "PON action expected 2 consume_tiles, got {}",
                        action.consume_tiles.len()
                    ));
                }
                let src = source_rel_hint
                    .ok_or_else(|| "PON action requires response source_rel_hint".to_string())?;
                let gid = lookup_pon_id(
                    src,
                    tile136_to_tile37(claimed),
                    tile136_to_tile37(action.consume_tiles[0]),
                    tile136_to_tile37(action.consume_tiles[1]),
                )?;
                push_unique(gid, vec![action.clone()])?;
            }
            ActionType::Daiminkan => {
                let claimed = action
                    .tile
                    .ok_or_else(|| "DAIMINKAN action missing claimed tile".to_string())?;
                let src = source_rel_hint.ok_or_else(|| {
                    "DAIMINKAN action requires response source_rel_hint".to_string()
                })?;
                let gid = lookup_daiminkan_id(src, tile136_to_tile37(claimed))?;
                push_unique(gid, vec![action.clone()])?;
            }
            ActionType::Ron => {
                let src = source_rel_hint
                    .ok_or_else(|| "RON action requires response source_rel_hint".to_string())?;
                let gid = lookup_ron_id(src)?;
                push_unique(gid, vec![action.clone()])?;
            }
            ActionType::Kita => {
                return Err("Kita action is 3-player only and invalid in 4p KAR V2".to_string());
            }
        }
    }

    // Expand Riichi + Discard pairs.
    if let Some(ra) = riichi_action {
        let has_riichi_drawn =
            drawn_tile.is_some_and(|dt| riichi_discards.iter().any(|a| a.tile == Some(dt)));
        let has_riichi_other = drawn_tile.is_some_and(|dt| {
            let dt37 = tile136_to_tile37(dt);
            riichi_discards.iter().any(|a| {
                a.tile
                    .is_some_and(|t| t != dt && tile136_to_tile37(t) == dt37)
            })
        });

        for d_act in &riichi_discards {
            let tile136 = d_act
                .tile
                .ok_or_else(|| "Riichi discard candidate missing tile".to_string())?;
            let tile37 = tile136_to_tile37(tile136);
            let (is_tsumo, is_ted) = discard_tsumogiri_modes(tile136);
            if is_tsumo {
                let gid = lookup_discard_id(tile37, true, true)?;
                push_unique(gid, vec![ra.clone(), d_act.clone()])?;
                if !obs.forced_tedashi
                    && !has_riichi_other
                    && hand_tile37_counts[tile37 as usize] >= 2
                    && let Some(other_t136) = hand
                        .iter()
                        .copied()
                        .map(|x| x as u8)
                        .find(|&t| t != tile136 && tile136_to_tile37(t) == tile37)
                {
                    let ted_act =
                        Action::new(ActionType::Discard, Some(other_t136), vec![], d_act.actor);
                    let ted_gid = lookup_discard_id(tile37, false, true)?;
                    push_unique(ted_gid, vec![ra.clone(), ted_act])?;
                }
            }
            if is_ted {
                let gid = lookup_discard_id(tile37, false, true)?;
                push_unique(gid, vec![ra.clone(), d_act.clone()])?;
                if !obs.forced_tedashi
                    && !has_riichi_drawn
                    && let Some(dt) = drawn_tile
                    && tile136_to_tile37(dt) == tile37
                {
                    let tsumo_act = Action::new(ActionType::Discard, Some(dt), vec![], d_act.actor);
                    let tsumo_gid = lookup_discard_id(tile37, true, true)?;
                    push_unique(tsumo_gid, vec![ra.clone(), tsumo_act])?;
                }
            }
        }
    }

    // Contract v2.4 Invariant #2 & P1-1: strict ascending order & hard MAX_CANDIDATES guard.
    resolved.sort_by_key(|c| c.global_id);
    assert!(
        resolved.len() <= MAX_CANDIDATES,
        "KAR V2 contract violation: legal candidate count {} exceeds MAX_CANDIDATES ({})",
        resolved.len(),
        MAX_CANDIDATES
    );

    Ok(resolved)
}

/// Recover the exact native execution sequence for a chosen `global_id` from the
/// current observer's legal candidates (bijective verification).
pub fn decode_global_id_to_native(
    global_id: u16,
    obs: &Observation,
    source_rel_hint: Option<u8>,
) -> Result<Vec<Action>, String> {
    let candidates = resolve_legal_candidates(obs, source_rel_hint)?;
    candidates
        .into_iter()
        .find(|c| c.global_id == global_id)
        .map(|c| c.native_actions)
        .ok_or_else(|| format!("global_id {global_id} is not legal in current observation"))
}
