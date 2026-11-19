//! Player-visible event projection shared by live play and recorded-game replay.
//!
//! This reducer is deliberately independent of `GameState`: it consumes only
//! observed events, retains only the selected player's private tiles, and never
//! queries an opponent's real hand, wall, or legal actions. Python owns training
//! provenance/history encoding; all tile, phase, riichi and kan transitions here
//! are engine responsibilities.

use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize)]
pub struct PublicMeld {
    pub kind: String,
    pub tiles: Vec<u8>,
    pub source: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ObserverState {
    pub observer: usize,
    pub have_start: bool,
    pub have_round: bool,
    pub own_hand_complete: bool,
    pub hand: Vec<u8>,
    pub scores: [i32; 4],
    pub history: [Vec<u8>; 4],
    pub rivers: [Vec<u8>; 4],
    pub melds: [Vec<PublicMeld>; 4],
    pub dora: Vec<u8>,
    pub declared: [bool; 4],
    pub accepted: [bool; 4],
    pub remaining: Option<i32>,
    pub last_actor: Option<usize>,
    pub last_tile: Option<u8>,
    pub oya: usize,
    pub honba: u32,
    pub riichi_sticks: u32,
    pub pending: Option<(String, usize, Vec<u8>)>,
    pub round_wind: u8,
    pub public_hand_sizes: [i32; 4],
    pub draw_counts: [u32; 4],
    pub riichi_ordinals: [Option<usize>; 4],
    pub current_draw_actor: Option<usize>,
    pub drawn_tile: Option<u8>,
    pub decision_phase: String,
    pub post_call_actor: Option<usize>,
    pub interrupted: bool,
    pub visibility_complete: bool,
}

fn text<'a>(event: &'a Value, key: &str) -> Result<&'a str, String> {
    event
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Missing/invalid {key}"))
}

fn seat(event: &Value, key: &str) -> Result<usize, String> {
    event
        .get(key)
        .and_then(Value::as_u64)
        .filter(|&x| x < 4)
        .map(|x| x as usize)
        .ok_or_else(|| format!("{key} must be a four-player seat"))
}

fn counter(event: &Value, key: &str, default: Option<u32>) -> Result<u32, String> {
    match event.get(key) {
        None => default.ok_or_else(|| format!("Missing {key}")),
        Some(value) => value
            .as_u64()
            .and_then(|x| u32::try_from(x).ok())
            .ok_or_else(|| format!("{key} must be a nonnegative integer")),
    }
}

fn tiles(value: &Value) -> Result<Vec<u8>, String> {
    value
        .as_array()
        .ok_or("Expected tile list")?
        .iter()
        .map(|tile| canonical_tile(tile.as_str().ok_or("Expected tile name")?))
        .collect()
}

fn points(value: &Value) -> Result<[i32; 4], String> {
    let values = value
        .as_array()
        .filter(|x| x.len() == 4)
        .ok_or("Scores must have four entries")?;
    let mut result = [0; 4];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_i64()
            .and_then(|x| i32::try_from(x).ok())
            .ok_or("Scores must contain integer points")?;
    }
    Ok(result)
}

/// Canonical 136-ID with red fives kept distinct; unseen tiles are not accepted.
pub fn canonical_tile(value: &str) -> Result<u8, String> {
    if let Some(index) = ["E", "S", "W", "N", "P", "F", "C"]
        .iter()
        .position(|&x| x == value)
    {
        return Ok(108 + index as u8 * 4);
    }
    let bytes = value.as_bytes();
    if bytes.len() != 2 && bytes.len() != 3 {
        return Err(format!("Invalid tile {value}"));
    }
    let offset = match bytes[1] {
        b'm' => 0,
        b'p' => 36,
        b's' => 72,
        b'z' => 108,
        _ => return Err(format!("Invalid tile {value}")),
    };
    let rank = bytes[0].wrapping_sub(b'0');
    if offset == 108 {
        if bytes.len() != 2 || !(1..=7).contains(&rank) {
            return Err(format!("Invalid tile {value}"));
        }
        return Ok(offset + (rank - 1) * 4);
    }
    if rank == 0 && bytes.len() == 2 {
        return Ok(offset + 16);
    }
    if !(1..=9).contains(&rank) || (bytes.len() == 3 && (bytes[2] != b'r' || rank != 5)) {
        return Err(format!("Invalid tile {value}"));
    }
    Ok(offset + (rank - 1) * 4 + u8::from(rank == 5 && bytes.len() == 2))
}

fn public_tile(value: &str) -> Result<String, String> {
    Ok(crate::parser::tid_to_mjai(canonical_tile(value)?))
}

/// A delayed indicator for an older kan cannot commit a newer declaration.
/// Both the observer projection and recorded-game adapter use this predicate.
pub fn pending_kan_commits(
    event: &Value,
    actor: usize,
    committed_kans: i64,
    indicators: i64,
) -> Result<bool, String> {
    match text(event, "type")? {
        "tsumo" | "kan_commit" => {
            if seat(event, "actor")? != actor {
                return Err("Kan commitment/replacement draw belongs to another actor".into());
            }
            Ok(true)
        }
        "dora" => {
            if committed_kans < 0 || indicators < 1 || indicators > committed_kans + 1 {
                return Err("Public kan/indicator accounting is inconsistent".into());
            }
            Ok(indicators == committed_kans + 1)
        }
        _ => Ok(false),
    }
}

impl ObserverState {
    pub fn new(observer: usize) -> Result<Self, String> {
        if observer >= 4 {
            return Err("Observer must be a four-player seat".into());
        }
        Ok(Self {
            observer,
            have_start: false,
            have_round: false,
            own_hand_complete: false,
            hand: Vec::new(),
            scores: [25000; 4],
            history: std::array::from_fn(|_| Vec::new()),
            rivers: std::array::from_fn(|_| Vec::new()),
            melds: std::array::from_fn(|_| Vec::new()),
            dora: Vec::new(),
            declared: [false; 4],
            accepted: [false; 4],
            remaining: None,
            last_actor: None,
            last_tile: None,
            oya: 0,
            honba: 0,
            riichi_sticks: 0,
            pending: None,
            round_wind: 0,
            public_hand_sizes: [13; 4],
            draw_counts: [0; 4],
            riichi_ordinals: [None; 4],
            current_draw_actor: None,
            drawn_tile: None,
            decision_phase: "unknown".into(),
            post_call_actor: None,
            interrupted: false,
            visibility_complete: true,
        })
    }

    fn remove_hand(&mut self, tile: u8) -> Result<(), String> {
        if let Some(index) = self.hand.iter().position(|&x| x == tile) {
            self.hand.remove(index);
        } else if self.own_hand_complete {
            return Err("Public event consumes a tile absent from the observer's hand".into());
        }
        Ok(())
    }

    fn commit_kan(&mut self) -> Result<(), String> {
        let Some((kind, actor, tiles)) = self.pending.clone() else {
            return Ok(());
        };
        if kind == "kakan" {
            let slots: Vec<_> = self.melds[actor]
                .iter()
                .enumerate()
                .filter(|(_, m)| {
                    m.kind == "pon"
                        && m.tiles.first().map(|x| x / 4) == tiles.first().map(|x| x / 4)
                })
                .map(|(index, _)| index)
                .collect();
            if slots.len() != 1 || tiles.len() != 1 {
                return Err("Kakan must update exactly one committed pon".into());
            }
            let meld = &mut self.melds[actor][slots[0]];
            meld.kind = kind;
            meld.tiles.push(tiles[0]);
        } else {
            self.melds[actor].push(PublicMeld {
                kind,
                tiles: tiles.clone(),
                source: None,
            });
        }
        if actor == self.observer {
            for &tile in &tiles {
                self.remove_hand(tile)?;
            }
        }
        self.public_hand_sizes[actor] -= tiles.len() as i32;
        self.interrupted = true;
        self.decision_phase = "await_draw".into();
        self.pending = None;
        Ok(())
    }

    /// Reject malformed events without partially advancing the observer state.
    pub fn apply_event(&mut self, event: &Value) -> Result<Value, String> {
        let mut staged = self.clone();
        let public = staged.apply_inner(event)?;
        *self = staged;
        Ok(public)
    }

    fn apply_inner(&mut self, event: &Value) -> Result<Value, String> {
        let typ = text(event, "type")?;
        if let Some((kind, actor, _)) = &self.pending {
            let count = self
                .melds
                .iter()
                .flatten()
                .filter(|m| matches!(m.kind.as_str(), "ankan" | "kakan" | "daiminkan"))
                .count();
            if pending_kan_commits(event, *actor, count as i64, self.dora.len() as i64)? {
                if typ == "kan_commit" && text(event, "kan_type")? != kind {
                    return Err("Native acknowledgement belongs to another kan type".into());
                }
                self.commit_kan()?;
            } else if typ == "kan_commit" {
                return Err("Kan commitment lacks a pending declaration".into());
            }
        } else if typ == "kan_commit" {
            return Err("Kan commitment lacks a pending declaration".into());
        }
        let mut public = json!({"type": typ});
        match typ {
            "start_game" => self.have_start = true,
            "start_kyoku" => {
                let hands = event
                    .get("tehais")
                    .and_then(Value::as_array)
                    .filter(|x| x.len() == 4)
                    .ok_or("KAR requires four-player rounds")?;
                let own = hands[self.observer]
                    .as_array()
                    .ok_or("Invalid owner hand")?;
                self.own_hand_complete =
                    own.len() == 13 && own.iter().all(|x| x.as_str().is_some_and(|x| x != "?"));
                self.hand = own
                    .iter()
                    .filter(|x| x.as_str() != Some("?"))
                    .map(|x| canonical_tile(x.as_str().ok_or("Invalid owner tile")?))
                    .collect::<Result<_, _>>()?;
                self.scores = points(&event["scores"])?;
                self.history = std::array::from_fn(|_| Vec::new());
                self.rivers = std::array::from_fn(|_| Vec::new());
                self.melds = std::array::from_fn(|_| Vec::new());
                self.declared = [false; 4];
                self.accepted = [false; 4];
                self.dora = vec![canonical_tile(text(event, "dora_marker")?)?];
                self.remaining = Some(70);
                self.last_actor = None;
                self.last_tile = None;
                self.pending = None;
                self.have_round = true;
                self.oya = seat(event, "oya")?;
                self.round_wind = match text(event, "bakaze")? {
                    "E" => 0,
                    "S" => 1,
                    "W" => 2,
                    "N" => 3,
                    _ => return Err("Invalid round wind".into()),
                };
                self.public_hand_sizes = [13; 4];
                self.draw_counts = [0; 4];
                self.riichi_ordinals = [None; 4];
                self.current_draw_actor = None;
                self.drawn_tile = None;
                self.decision_phase = "await_draw".into();
                self.post_call_actor = None;
                self.interrupted = false;
                self.honba = counter(event, "honba", None)?;
                self.riichi_sticks =
                    counter(event, "kyotaku", Some(counter(event, "kyoutaku", Some(0))?))?;
                for key in ["bakaze", "kyoku", "honba", "oya", "scores", "dora_marker"] {
                    public[key] = event
                        .get(key)
                        .ok_or_else(|| format!("Missing {key}"))?
                        .clone();
                }
                public["tehais"] = Value::Array(
                    (0..4)
                        .map(|seat| {
                            if seat == self.observer {
                                hands[seat].clone()
                            } else {
                                json!(vec!["?"; 13])
                            }
                        })
                        .collect(),
                );
                public["kyotaku"] = json!(self.riichi_sticks);
            }
            "kan_commit" => {
                public["actor"] = json!(seat(event, "actor")?);
                public["kan_type"] = json!(text(event, "kan_type")?);
            }
            "tsumo" => {
                let actor = seat(event, "actor")?;
                let value = text(event, "pai")?;
                self.public_hand_sizes[actor] += 1;
                self.draw_counts[actor] += 1;
                self.current_draw_actor = Some(actor);
                self.drawn_tile = if actor == self.observer && value != "?" {
                    Some(canonical_tile(value)?)
                } else {
                    None
                };
                self.decision_phase = "own_act".into();
                self.post_call_actor = None;
                public["actor"] = json!(actor);
                public["pai"] = json!(if actor == self.observer && value != "?" {
                    public_tile(value)?
                } else {
                    "?".into()
                });
                if actor == self.observer {
                    if value == "?" {
                        self.own_hand_complete = false;
                    } else {
                        self.hand.push(canonical_tile(value)?);
                    }
                }
                if let Some(remaining) = &mut self.remaining {
                    *remaining -= 1;
                }
            }
            "dahai" => {
                let actor = seat(event, "actor")?;
                let tile = canonical_tile(text(event, "pai")?)?;
                self.public_hand_sizes[actor] -= 1;
                self.current_draw_actor = None;
                self.drawn_tile = None;
                self.decision_phase = "discard_response".into();
                self.post_call_actor = None;
                if actor == self.observer {
                    self.remove_hand(tile)?;
                }
                self.history[actor].push(tile);
                self.rivers[actor].push(tile);
                self.last_actor = Some(actor);
                self.last_tile = Some(tile);
                public["actor"] = json!(actor);
                public["pai"] = json!(crate::parser::tid_to_mjai(tile));
                public["tsumogiri"] = json!(
                    event
                        .get("tsumogiri")
                        .and_then(Value::as_bool)
                        .ok_or("Missing/invalid tsumogiri")?
                );
            }
            "chi" | "pon" | "kan" | "daiminkan" => {
                let actor = seat(event, "actor")?;
                let source = seat(event, "target")?;
                let called = canonical_tile(text(event, "pai")?)?;
                let consumed = tiles(&event["consumed"])?;
                if actor == source || self.rivers[source].last() != Some(&called) {
                    return Err("A call must consume the triggering public river tile".into());
                }
                self.public_hand_sizes[actor] -= consumed.len() as i32;
                self.current_draw_actor = None;
                self.drawn_tile = None;
                self.interrupted = true;
                let kan = matches!(typ, "kan" | "daiminkan");
                self.decision_phase = if kan { "await_draw" } else { "own_act" }.into();
                self.post_call_actor = if kan { None } else { Some(actor) };
                self.rivers[source].pop();
                if actor == self.observer {
                    for &tile in &consumed {
                        self.remove_hand(tile)?;
                    }
                }
                self.melds[actor].push(PublicMeld {
                    kind: if kan { "daiminkan" } else { typ }.into(),
                    tiles: std::iter::once(called)
                        .chain(consumed.iter().copied())
                        .collect(),
                    source: Some(source),
                });
                public["actor"] = json!(actor);
                public["target"] = json!(source);
                public["pai"] = json!(crate::parser::tid_to_mjai(called));
                public["consumed"] = json!(
                    consumed
                        .iter()
                        .map(|&t| crate::parser::tid_to_mjai(t))
                        .collect::<Vec<_>>()
                );
            }
            "kakan" | "ankan" => {
                if self.pending.is_some() {
                    return Err("Unresolved kan declaration".into());
                }
                let actor = seat(event, "actor")?;
                let mut consumed = if typ == "kakan" {
                    vec![canonical_tile(text(event, "pai")?)?]
                } else {
                    tiles(&event["consumed"])?
                };
                public["actor"] = json!(actor);
                if typ == "kakan" {
                    public["pai"] = json!(crate::parser::tid_to_mjai(consumed[0]));
                } else {
                    if consumed.len() != 4 || consumed.iter().any(|t| t / 4 != consumed[0] / 4) {
                        return Err("Ankan declaration must contain four tiles of one kind".into());
                    }
                    if actor != self.observer {
                        for tile in &mut consumed {
                            let base = *tile / 4;
                            *tile = base * 4 + u8::from(matches!(base, 4 | 13 | 22));
                        }
                    }
                    public["consumed"] = json!(
                        consumed
                            .iter()
                            .map(|&t| crate::parser::tid_to_mjai(t))
                            .collect::<Vec<_>>()
                    );
                }
                self.pending = Some((typ.into(), actor, consumed));
                self.current_draw_actor = None;
                self.drawn_tile = None;
                self.decision_phase = "kan_response".into();
                self.post_call_actor = None;
            }
            "reach" => {
                let actor = seat(event, "actor")?;
                self.declared[actor] = true;
                self.riichi_ordinals[actor] = Some(self.history[actor].len() + 1);
                public["actor"] = json!(actor);
            }
            "reach_accepted" => {
                let actor = seat(event, "actor")?;
                if !self.declared[actor] || self.accepted[actor] {
                    return Err("Riichi acceptance must follow one unaccepted declaration".into());
                }
                self.accepted[actor] = true;
                self.scores[actor] = self.scores[actor]
                    .checked_sub(1000)
                    .ok_or("Riichi score overflow")?;
                self.riichi_sticks = self
                    .riichi_sticks
                    .checked_add(1)
                    .ok_or("Riichi pot overflow")?;
                public["actor"] = json!(actor);
            }
            "dora" => {
                let tile = canonical_tile(text(event, "dora_marker")?)?;
                self.dora.push(tile);
                public["dora_marker"] = json!(crate::parser::tid_to_mjai(tile));
            }
            "hora" | "ryukyoku" => {
                for key in [
                    "actor",
                    "target",
                    "pai",
                    "reason",
                    "scores",
                    "deltas",
                    "delta",
                    "public_hand",
                    "public_hands",
                    "visibility_complete",
                    "visibility_gap",
                    "public_reveal_provenance",
                    "settlement_provenance",
                    "declaring_actor_provenance",
                ] {
                    if let Some(value) = event.get(key) {
                        public[key] = value.clone();
                    }
                }
                if event.get("visibility_complete") == Some(&Value::Bool(false)) {
                    self.visibility_complete = false;
                }
                if typ == "hora"
                    && self.accepted[seat(event, "actor")?]
                    && event.get("ura_markers").is_some()
                {
                    public["public_ura_markers"] = json!(
                        tiles(&event["ura_markers"])?
                            .iter()
                            .map(|&t| crate::parser::tid_to_mjai(t))
                            .collect::<Vec<_>>()
                    );
                }
                if let Some(value) = event.get("scores") {
                    self.scores = points(value)?;
                }
                if typ == "hora" {
                    self.riichi_sticks = 0;
                }
                self.pending = None;
                self.current_draw_actor = None;
                self.drawn_tile = None;
                self.decision_phase = "closed".into();
                self.post_call_actor = None;
            }
            "end_kyoku" => {}
            "end_game" => {
                if let Some(value) = event.get("scores") {
                    self.scores = points(value)?;
                }
                self.riichi_sticks = 0;
            }
            _ => return Err(format!("Unsupported public event: {typ}")),
        }
        if let Some(value) = event.get("remaining_draws") {
            let remaining = value
                .as_i64()
                .filter(|&x| (0..=70).contains(&x))
                .ok_or("Invalid public live-wall count")?;
            self.remaining = Some(remaining as i32);
            public["remaining_draws"] = json!(remaining);
        }
        Ok(public)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start() -> Value {
        json!({"type":"start_kyoku", "bakaze":"E", "kyoku":1, "oya":0, "honba":0,
            "scores":vec![25000;4], "dora_marker":"9s", "tehais":[
                ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
                vec!["?";13], vec!["?";13], vec!["?";13]]})
    }

    #[test]
    fn opponent_private_tiles_do_not_enter_state_or_events() {
        let mut a = ObserverState::new(0).unwrap();
        let mut b = ObserverState::new(0).unwrap();
        let first = start();
        let mut changed = start();
        changed["tehais"][1] = json!(vec!["C"; 13]);
        assert_eq!(
            a.apply_event(&first).unwrap(),
            b.apply_event(&changed).unwrap()
        );
        assert_eq!(
            a.apply_event(&json!({"type":"tsumo","actor":1,"pai":"5pr"}))
                .unwrap(),
            b.apply_event(&json!({"type":"tsumo","actor":1,"pai":"1m"}))
                .unwrap()
        );
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
    }

    #[test]
    fn rejected_event_does_not_partly_charge_riichi_or_move_tiles() {
        let mut state = ObserverState::new(0).unwrap();
        state.apply_event(&start()).unwrap();
        let before = serde_json::to_value(&state).unwrap();
        assert!(
            state
                .apply_event(&json!({"type":"reach_accepted","actor":0}))
                .is_err()
        );
        assert!(
            state
                .apply_event(&json!({"type":"dahai","actor":0,"pai":"C","tsumogiri":false}))
                .is_err()
        );
        assert_eq!(serde_json::to_value(state).unwrap(), before);
    }
}
