//! Lossless 10-byte observer-visible round event stream (Contract v2.4 Section 1.3).
//!
//! Strictly append-only, mandatory initial `DORA` at index 0 (`L_ev >= 1`),
//! hard error on `L_ev > MAX_EVENT_LEN (256)` (never silently truncates),
//! with provable round-trip bijective decode for all 10 observer-visible event types.

use serde::{Deserialize, Serialize};

use super::{EVENT_BYTES, MAX_EVENT_LEN, is_red5_tile37};

pub const ACTOR_TABLE: u8 = 4;
pub const SOURCE_NONE: u8 = 4;
pub const TILE37_NONE: u8 = 37;

pub const FLAG_TSUMOGIRI: u8 = 1 << 0;
pub const FLAG_RIICHI_DECLARE: u8 = 1 << 1;
pub const FLAG_DOUBLE_RIICHI: u8 = 1 << 2;
pub const FLAG_RINSHAN: u8 = 1 << 3;
pub const FLAG_INVOLVES_RED5: u8 = 1 << 4;

pub const PASS_CHI: u8 = 1 << 0;
pub const PASS_PON: u8 = 1 << 1;
pub const PASS_KAN: u8 = 1 << 2;
pub const PASS_RON: u8 = 1 << 3;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventKind {
    Pad = 0,
    Dora = 1,
    TsumoSelf = 2,
    Dahai = 3,
    ReachAccepted = 4,
    Chi = 5,
    Pon = 6,
    Daiminkan = 7,
    Ankan = 8,
    Kakan = 9,
    SelfPass = 10,
}

impl EventKind {
    pub fn from_u8(v: u8) -> Result<Self, String> {
        match v {
            0 => Ok(Self::Pad),
            1 => Ok(Self::Dora),
            2 => Ok(Self::TsumoSelf),
            3 => Ok(Self::Dahai),
            4 => Ok(Self::ReachAccepted),
            5 => Ok(Self::Chi),
            6 => Ok(Self::Pon),
            7 => Ok(Self::Daiminkan),
            8 => Ok(Self::Ankan),
            9 => Ok(Self::Kakan),
            10 => Ok(Self::SelfPass),
            _ => Err(format!("invalid V2 EventKind byte {v}")),
        }
    }
}

/// Canonical observer-relative visible event (exact round-trip domain).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalVisibleEvent {
    pub kind: EventKind,
    /// Relative seat `0..3`, or `ACTOR_TABLE (4)` for `Dora`.
    pub actor_rel: u8,
    /// Relative source `0..3` for `Chi/Pon/Daiminkan/SelfPass`, else `SOURCE_NONE (4)`.
    pub source_rel: u8,
    pub tile37: u8,
    pub consumed0_37: u8,
    pub consumed1_37: u8,
    pub flags: u8,
    pub sub_mask: u8,
    pub turn_idx: u8,
    pub actor_seq_idx: u8,
}

impl CanonicalVisibleEvent {
    pub fn to_bytes(self) -> Result<[u8; EVENT_BYTES], String> {
        if self.kind == EventKind::Pad {
            return Err("cannot encode Pad as a live canonical visible event".to_string());
        }
        if self.actor_rel > ACTOR_TABLE {
            return Err(format!("actor_rel {} out of range", self.actor_rel));
        }
        if self.source_rel > SOURCE_NONE {
            return Err(format!("source_rel {} out of range", self.source_rel));
        }
        if self.tile37 > TILE37_NONE
            || self.consumed0_37 > TILE37_NONE
            || self.consumed1_37 > TILE37_NONE
        {
            return Err("tile37 field out of range [0, 37]".to_string());
        }
        Ok([
            self.kind as u8,
            self.actor_rel,
            self.source_rel,
            self.tile37,
            self.consumed0_37,
            self.consumed1_37,
            self.flags,
            self.sub_mask,
            self.turn_idx,
            self.actor_seq_idx,
        ])
    }

    pub fn from_bytes(raw: [u8; EVENT_BYTES]) -> Result<Self, String> {
        let kind = EventKind::from_u8(raw[0])?;
        if kind == EventKind::Pad {
            return Err("unexpected Pad event in non-padded event sequence".to_string());
        }
        let ev = Self {
            kind,
            actor_rel: raw[1],
            source_rel: raw[2],
            tile37: raw[3],
            consumed0_37: raw[4],
            consumed1_37: raw[5],
            flags: raw[6],
            sub_mask: raw[7],
            turn_idx: raw[8],
            actor_seq_idx: raw[9],
        };
        // Re-validate bounds.
        let _ = ev.to_bytes()?;
        Ok(ev)
    }
}

/// Absolute-seat observer-visible round event entry stored inside [`RoundEventLog`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundEvent {
    Dora {
        indicator_tile37: u8,
        dora_idx: u8,
        turn_idx: u8,
    },
    TsumoPrivate {
        actor: u8,
        tile37: u8,
        is_rinshan: bool,
        turn_idx: u8,
    },
    Dahai {
        actor: u8,
        tile37: u8,
        tsumogiri: bool,
        is_riichi_declare: bool,
        is_double_riichi: bool,
        turn_idx: u8,
        river_idx: u8,
    },
    ReachAccepted {
        actor: u8,
        turn_idx: u8,
    },
    Chi {
        actor: u8,
        source: u8,
        claimed_tile37: u8,
        consumed0_37: u8,
        consumed1_37: u8,
        turn_idx: u8,
        meld_idx: u8,
    },
    Pon {
        actor: u8,
        source: u8,
        claimed_tile37: u8,
        consumed0_37: u8,
        consumed1_37: u8,
        turn_idx: u8,
        meld_idx: u8,
    },
    Daiminkan {
        actor: u8,
        source: u8,
        claimed_tile37: u8,
        turn_idx: u8,
        meld_idx: u8,
    },
    Ankan {
        actor: u8,
        tile37: u8,
        involves_red5: bool,
        turn_idx: u8,
        meld_idx: u8,
    },
    Kakan {
        actor: u8,
        tile37: u8,
        turn_idx: u8,
        meld_idx: u8,
    },
    SelfPassPrivate {
        actor: u8,
        source: u8,
        tile37: u8,
        pass_mask: u8,
        turn_idx: u8,
    },
}

/// Append-only single-round event log. Index 0 is ALWAYS the initial `Dora`.
#[derive(Debug, Clone, Default)]
pub struct RoundEventLog {
    events: Vec<RoundEvent>,
}

impl RoundEventLog {
    pub fn new(initial_dora_indicator_tile37: u8) -> Result<Self, String> {
        if initial_dora_indicator_tile37 >= 37 {
            return Err(format!(
                "invalid initial dora tile37 {initial_dora_indicator_tile37}"
            ));
        }
        Ok(Self {
            events: vec![RoundEvent::Dora {
                indicator_tile37: initial_dora_indicator_tile37,
                dora_idx: 0,
                turn_idx: 0,
            }],
        })
    }

    pub fn push(&mut self, event: RoundEvent) {
        self.events.push(event);
    }

    pub fn raw_events(&self) -> &[RoundEvent] {
        &self.events
    }

    /// Project the round log to observer `observer ∈ 0..3`, filtering out other
    /// players' private draws and private pass decisions (Contract v2.4 P1-5).
    pub fn project_visible(&self, observer: u8) -> Result<Vec<CanonicalVisibleEvent>, String> {
        if observer >= 4 {
            return Err(format!("invalid observer {observer}"));
        }
        if self.events.is_empty() {
            return Err("RoundEventLog missing mandatory initial DORA at index 0".to_string());
        }
        if !matches!(self.events[0], RoundEvent::Dora { dora_idx: 0, .. }) {
            return Err("RoundEventLog index 0 must be initial DORA (dora_idx=0)".to_string());
        }

        let rel = |abs: u8| -> u8 { (abs + 4 - observer) % 4 };
        let mut out = Vec::with_capacity(self.events.len());

        for &ev in &self.events {
            let mapped = match ev {
                RoundEvent::Dora {
                    indicator_tile37,
                    dora_idx,
                    turn_idx,
                } => Some(CanonicalVisibleEvent {
                    kind: EventKind::Dora,
                    actor_rel: ACTOR_TABLE,
                    source_rel: SOURCE_NONE,
                    tile37: indicator_tile37,
                    consumed0_37: TILE37_NONE,
                    consumed1_37: TILE37_NONE,
                    flags: if is_red5_tile37(indicator_tile37) {
                        FLAG_INVOLVES_RED5
                    } else {
                        0
                    },
                    sub_mask: dora_idx,
                    turn_idx,
                    actor_seq_idx: 0,
                }),
                RoundEvent::TsumoPrivate {
                    actor,
                    tile37,
                    is_rinshan,
                    turn_idx,
                } => {
                    if actor != observer {
                        None
                    } else {
                        let mut flags = 0u8;
                        if is_rinshan {
                            flags |= FLAG_RINSHAN;
                        }
                        if is_red5_tile37(tile37) {
                            flags |= FLAG_INVOLVES_RED5;
                        }
                        Some(CanonicalVisibleEvent {
                            kind: EventKind::TsumoSelf,
                            actor_rel: 0,
                            source_rel: SOURCE_NONE,
                            tile37,
                            consumed0_37: TILE37_NONE,
                            consumed1_37: TILE37_NONE,
                            flags,
                            sub_mask: 0,
                            turn_idx,
                            actor_seq_idx: 0,
                        })
                    }
                }
                RoundEvent::Dahai {
                    actor,
                    tile37,
                    tsumogiri,
                    is_riichi_declare,
                    is_double_riichi,
                    turn_idx,
                    river_idx,
                } => {
                    let mut flags = 0u8;
                    if tsumogiri {
                        flags |= FLAG_TSUMOGIRI;
                    }
                    if is_riichi_declare {
                        flags |= FLAG_RIICHI_DECLARE;
                    }
                    if is_double_riichi {
                        flags |= FLAG_DOUBLE_RIICHI;
                    }
                    if is_red5_tile37(tile37) {
                        flags |= FLAG_INVOLVES_RED5;
                    }
                    Some(CanonicalVisibleEvent {
                        kind: EventKind::Dahai,
                        actor_rel: rel(actor),
                        source_rel: SOURCE_NONE,
                        tile37,
                        consumed0_37: TILE37_NONE,
                        consumed1_37: TILE37_NONE,
                        flags,
                        sub_mask: 0,
                        turn_idx,
                        actor_seq_idx: river_idx,
                    })
                }
                RoundEvent::ReachAccepted { actor, turn_idx } => Some(CanonicalVisibleEvent {
                    kind: EventKind::ReachAccepted,
                    actor_rel: rel(actor),
                    source_rel: SOURCE_NONE,
                    tile37: TILE37_NONE,
                    consumed0_37: TILE37_NONE,
                    consumed1_37: TILE37_NONE,
                    flags: 0,
                    sub_mask: 0,
                    turn_idx,
                    actor_seq_idx: 0,
                }),
                RoundEvent::Chi {
                    actor,
                    source,
                    claimed_tile37,
                    consumed0_37,
                    consumed1_37,
                    turn_idx,
                    meld_idx,
                } => {
                    let (c0, c1) = if consumed0_37 <= consumed1_37 {
                        (consumed0_37, consumed1_37)
                    } else {
                        (consumed1_37, consumed0_37)
                    };
                    let red =
                        is_red5_tile37(claimed_tile37) || is_red5_tile37(c0) || is_red5_tile37(c1);
                    Some(CanonicalVisibleEvent {
                        kind: EventKind::Chi,
                        actor_rel: rel(actor),
                        source_rel: rel(source),
                        tile37: claimed_tile37,
                        consumed0_37: c0,
                        consumed1_37: c1,
                        flags: if red { FLAG_INVOLVES_RED5 } else { 0 },
                        sub_mask: 0,
                        turn_idx,
                        actor_seq_idx: meld_idx,
                    })
                }
                RoundEvent::Pon {
                    actor,
                    source,
                    claimed_tile37,
                    consumed0_37,
                    consumed1_37,
                    turn_idx,
                    meld_idx,
                } => {
                    let (c0, c1) = if consumed0_37 <= consumed1_37 {
                        (consumed0_37, consumed1_37)
                    } else {
                        (consumed1_37, consumed0_37)
                    };
                    let red =
                        is_red5_tile37(claimed_tile37) || is_red5_tile37(c0) || is_red5_tile37(c1);
                    Some(CanonicalVisibleEvent {
                        kind: EventKind::Pon,
                        actor_rel: rel(actor),
                        source_rel: rel(source),
                        tile37: claimed_tile37,
                        consumed0_37: c0,
                        consumed1_37: c1,
                        flags: if red { FLAG_INVOLVES_RED5 } else { 0 },
                        sub_mask: 0,
                        turn_idx,
                        actor_seq_idx: meld_idx,
                    })
                }
                RoundEvent::Daiminkan {
                    actor,
                    source,
                    claimed_tile37,
                    turn_idx,
                    meld_idx,
                } => Some(CanonicalVisibleEvent {
                    kind: EventKind::Daiminkan,
                    actor_rel: rel(actor),
                    source_rel: rel(source),
                    tile37: claimed_tile37,
                    consumed0_37: TILE37_NONE,
                    consumed1_37: TILE37_NONE,
                    flags: if is_red5_tile37(claimed_tile37) {
                        FLAG_INVOLVES_RED5
                    } else {
                        0
                    },
                    sub_mask: 0,
                    turn_idx,
                    actor_seq_idx: meld_idx,
                }),
                RoundEvent::Ankan {
                    actor,
                    tile37,
                    involves_red5,
                    turn_idx,
                    meld_idx,
                } => Some(CanonicalVisibleEvent {
                    kind: EventKind::Ankan,
                    actor_rel: rel(actor),
                    source_rel: SOURCE_NONE,
                    tile37,
                    consumed0_37: TILE37_NONE,
                    consumed1_37: TILE37_NONE,
                    flags: if involves_red5 || is_red5_tile37(tile37) {
                        FLAG_INVOLVES_RED5
                    } else {
                        0
                    },
                    sub_mask: 0,
                    turn_idx,
                    actor_seq_idx: meld_idx,
                }),
                RoundEvent::Kakan {
                    actor,
                    tile37,
                    turn_idx,
                    meld_idx,
                } => Some(CanonicalVisibleEvent {
                    kind: EventKind::Kakan,
                    actor_rel: rel(actor),
                    source_rel: SOURCE_NONE,
                    tile37,
                    consumed0_37: TILE37_NONE,
                    consumed1_37: TILE37_NONE,
                    flags: if is_red5_tile37(tile37) {
                        FLAG_INVOLVES_RED5
                    } else {
                        0
                    },
                    sub_mask: 0,
                    turn_idx,
                    actor_seq_idx: meld_idx,
                }),
                RoundEvent::SelfPassPrivate {
                    actor,
                    source,
                    tile37,
                    pass_mask,
                    turn_idx,
                } => {
                    if actor != observer {
                        None
                    } else {
                        Some(CanonicalVisibleEvent {
                            kind: EventKind::SelfPass,
                            actor_rel: 0,
                            source_rel: rel(source),
                            tile37,
                            consumed0_37: TILE37_NONE,
                            consumed1_37: TILE37_NONE,
                            flags: if is_red5_tile37(tile37) {
                                FLAG_INVOLVES_RED5
                            } else {
                                0
                            },
                            sub_mask: pass_mask,
                            turn_idx,
                            actor_seq_idx: 0,
                        })
                    }
                }
            };

            if let Some(c) = mapped {
                out.push(c);
            }
        }

        if out.len() > MAX_EVENT_LEN {
            return Err(format!(
                "KAR V2 contract violation: observer-visible event length {} exceeds MAX_EVENT_LEN ({})",
                out.len(),
                MAX_EVENT_LEN
            ));
        }

        Ok(out)
    }
}

/// Encode observer-visible round events into compact 10-byte records.
/// Never truncates silently; returns `Err` if `len == 0` or `len > 256`.
pub fn encode_events(log: &RoundEventLog, observer: u8) -> Result<Vec<[u8; EVENT_BYTES]>, String> {
    let visible = log.project_visible(observer)?;
    if visible.is_empty() {
        return Err("encoded event stream cannot be empty (L_ev >= 1 required)".to_string());
    }
    if visible.len() > MAX_EVENT_LEN {
        return Err(format!(
            "encoded event stream length {} exceeds MAX_EVENT_LEN ({MAX_EVENT_LEN})",
            visible.len()
        ));
    }
    visible
        .into_iter()
        .map(CanonicalVisibleEvent::to_bytes)
        .collect()
}

/// Decode 10-byte event records back into canonical observer-visible events.
pub fn decode_events(encoded: &[[u8; EVENT_BYTES]]) -> Result<Vec<CanonicalVisibleEvent>, String> {
    if encoded.is_empty() {
        return Err("cannot decode empty event slice".to_string());
    }
    if encoded.len() > MAX_EVENT_LEN {
        return Err(format!(
            "cannot decode {} events exceeding MAX_EVENT_LEN ({MAX_EVENT_LEN})",
            encoded.len()
        ));
    }
    encoded
        .iter()
        .copied()
        .map(CanonicalVisibleEvent::from_bytes)
        .collect()
}
