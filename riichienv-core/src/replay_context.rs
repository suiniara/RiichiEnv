//! Prefix-only context for MJAI replay.
//!
//! Game flags live in `GameState`/`PlayerState`; this reducer only retains the
//! unresolved declarations needed to apply the next public event. In particular,
//! a committed kan proves that Ron offers went unused, not that an opponent chose
//! a particular private Pass action.

use std::collections::HashMap;

use crate::action::{Action, ActionType, Phase};
use crate::observer::pending_kan_commits;
use crate::parser::{mjai_to_tid, tid_to_mjai};
use crate::replay::MjaiEvent;
use crate::state::GameState;
use crate::state::legal_actions::GameStateLegalActions;
use crate::types::{Meld, MeldType};

pub const REPLAY_CONTEXT_VERSION: &str = "native_replay_context_v4";

#[derive(Clone, Debug)]
pub struct KanReplayWindow {
    pub actor: u8,
    pub declaration: Option<Action>,
    pub responses: HashMap<u8, Vec<Action>>,
    pub committed_kans_before: usize,
    pub applied: bool,
    pub ended_by_win: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ReplayContext {
    pub pending_double: [Option<bool>; 4],
    pub replacement_actor: Option<u8>,
    pub kan_window: Option<KanReplayWindow>,
    // A public-only MJAI client may use '?' for an opponent's concealed tiles.
    // Such a prefix cannot certify that opponent's private Ron opportunities.
    pub private_hands_complete: [bool; 4],
}

fn canonical_name(value: &str) -> Result<String, String> {
    mjai_to_tid(value)
        .map(tid_to_mjai)
        .ok_or_else(|| format!("Invalid kan tile {value}"))
}

fn matches_declaration(event: &MjaiEvent, actor: u8, action: &Action) -> Result<bool, String> {
    match event {
        MjaiEvent::Ankan {
            actor: seat,
            consumed,
        } => {
            if *seat != actor as usize || action.action_type != ActionType::Ankan {
                return Ok(false);
            }
            let mut actual = consumed
                .iter()
                .map(|tile| canonical_name(tile))
                .collect::<Result<Vec<_>, _>>()?;
            let mut expected = action
                .consume_tiles
                .iter()
                .map(|&tile| tid_to_mjai(tile))
                .collect::<Vec<_>>();
            actual.sort();
            expected.sort();
            Ok(actual == expected)
        }
        MjaiEvent::Kakan { actor: seat, pai } => Ok(*seat == actor as usize
            && action.action_type == ActionType::Kakan
            && action.tile.map(tid_to_mjai).as_ref() == Some(&canonical_name(pai)?)),
        _ => Ok(false),
    }
}

impl GameState {
    /// Preview through the same native transition as live play, without mutating
    /// state, event-delta counters, RNG, or any observer's private history.
    pub fn preview_kan_response_window(
        &self,
        actor: u8,
        declaration: &Action,
    ) -> Result<HashMap<u8, Vec<Action>>, String> {
        if actor >= 4 {
            return Err("actor must be a four-player seat".into());
        }
        if self.is_done
            || self.needs_initialize_next_round
            || self.needs_tsumo
            || self.phase != Phase::WaitAct
            || self.current_player != actor
        {
            return Err("kan response preview requires the actor's pre-declaration turn".into());
        }
        if !matches!(
            declaration.action_type,
            ActionType::Ankan | ActionType::Kakan
        ) || declaration.actor.is_some_and(|seat| seat != actor)
        {
            return Err(
                "declaration must be a complete ankan or kakan belonging to the actor".into(),
            );
        }
        let mut declaration = declaration.clone();
        declaration.actor = Some(actor);
        if !self
            ._get_legal_actions_internal(actor)
            .contains(&declaration)
        {
            return Err("declaration is not a complete legal kan action".into());
        }
        let mut preview = self.clone();
        // MJAI WaitAct can retain claims from the preceding closed window.
        preview.current_claims.clear();
        preview.step(&HashMap::from([(actor, declaration)]));
        if preview.last_error.is_some() {
            return Err("native kan preview rejected a previously validated declaration".into());
        }
        let mut responses = HashMap::new();
        for player in 0..4 {
            if player == actor {
                continue;
            }
            let actions = if preview.phase == Phase::WaitResponse
                && preview.active_players.contains(&player)
            {
                preview._get_legal_actions_internal(player)
            } else {
                Vec::new()
            };
            if actions.iter().any(|action| {
                !matches!(action.action_type, ActionType::Ron | ActionType::Pass)
                    || action.actor != Some(player)
            }) {
                return Err("native kan preview exposed a non-response action".into());
            }
            responses.insert(player, actions);
        }
        Ok(responses)
    }

    pub fn prepare_replay_kan(&mut self, actor: u8, declaration: Action) -> Result<(), String> {
        if self
            .replay_context
            .kan_window
            .as_ref()
            .is_some_and(|window| window.applied)
        {
            return Err("The preceding kan response window has not resolved".into());
        }
        let responses = self.preview_kan_response_window(actor, &declaration)?;
        self.replay_context.kan_window = Some(KanReplayWindow {
            actor,
            declaration: Some(declaration),
            responses,
            committed_kans_before: self
                .players
                .iter()
                .flat_map(|player| &player.melds)
                .filter(|meld| {
                    matches!(
                        meld.meld_type,
                        MeldType::Ankan | MeldType::Kakan | MeldType::Daiminkan
                    )
                })
                .count(),
            applied: false,
            ended_by_win: false,
        });
        Ok(())
    }

    /// The KAR bridge uses this check before applying any externally supplied
    /// snapshots. The snapshots are not inputs to native rule transitions.
    pub fn validate_replay_kan_event(&self, event: &MjaiEvent) -> Result<(), String> {
        if !matches!(event, MjaiEvent::Ankan { .. } | MjaiEvent::Kakan { .. }) {
            return Ok(());
        }
        let window = self
            .replay_context
            .kan_window
            .as_ref()
            .filter(|window| !window.applied)
            .ok_or("Validated kan preview is required before applying a declaration")?;
        let declaration = window
            .declaration
            .as_ref()
            .ok_or("Validated kan preview is required before applying a declaration")?;
        if !matches_declaration(event, window.actor, declaration)? {
            return Err("Kan preview tile differs from the observed declaration".into());
        }
        Ok(())
    }

    /// Check an observed draw against the native prefix without applying it.
    /// A replacement draw can itself close a pending robbery window, even
    /// though live state correctly still has needs_tsumo=false while waiting
    /// for those responses. No draw or private Pass is inferred by this query.
    pub fn validate_replay_draw(&self, actor: u8) -> Result<(), String> {
        if actor >= 4 {
            return Err("Expected a four-player draw actor".into());
        }
        let unavailable =
            "Draw is not allowed before a mandatory discard or after round termination";
        if self.is_done || self.needs_initialize_next_round {
            return Err(unavailable.into());
        }
        if let Some(window) = self
            .replay_context
            .kan_window
            .as_ref()
            .filter(|window| window.applied)
        {
            if window.ended_by_win {
                return Err(unavailable.into());
            }
            if actor != window.actor {
                return Err("Replacement draw belongs to another player".into());
            }
            return Ok(());
        }
        if !self.needs_tsumo {
            return Err(unavailable.into());
        }
        if self
            .replay_context
            .replacement_actor
            .is_some_and(|seat| seat != actor)
        {
            return Err("Replacement draw belongs to another player".into());
        }
        Ok(())
    }

    fn interrupt_replay_calls(&mut self) {
        self.is_first_turn = false;
        for player in &mut self.players {
            player.ippatsu_cycle = false;
        }
    }

    /// The same physical kakan commit is used by live play and MJAI replay.
    /// Until this point, a robbed declaration leaves the original Pon/hand.
    pub(crate) fn commit_kakan_tiles(&mut self, actor: u8, tile: u8) {
        let player = &mut self.players[actor as usize];
        if let Some(index) = player.hand.iter().position(|&held| held == tile) {
            player.hand.remove(index);
        }
        if let Some(meld) = player.melds.iter_mut().find(|meld| {
            meld.meld_type == MeldType::Pon
                && meld.tiles.first().is_some_and(|&held| held / 4 == tile / 4)
        }) {
            meld.meld_type = MeldType::Kakan;
            meld.tiles.push(tile);
            meld.tiles.sort();
        }
    }

    fn commit_replay_kan(&mut self) {
        let window = self
            .replay_context
            .kan_window
            .take()
            .expect("pending replay kan");
        if let Some((actor, action)) = self.pending_kan.take() {
            if action.action_type == ActionType::Kakan {
                self.commit_kakan_tiles(actor, action.tile.expect("kakan tile"));
            } else {
                for tile in &action.consume_tiles {
                    if let Some(index) = self.players[actor as usize]
                        .hand
                        .iter()
                        .position(|held| held == tile)
                    {
                        self.players[actor as usize].hand.remove(index);
                    }
                }
                self.players[actor as usize].melds.push(Meld {
                    meld_type: MeldType::Ankan,
                    tiles: action.consume_tiles,
                    opened: false,
                    from_who: -1,
                    called_tile: None,
                });
            }
        }
        for (seat, actions) in window.responses {
            if actions
                .iter()
                .any(|action| action.action_type == ActionType::Ron)
            {
                self.players[seat as usize].record_declined_ron();
            }
        }
        self.interrupt_replay_calls();
        self.current_claims.clear();
        self.phase = Phase::WaitAct;
        self.current_player = window.actor;
        self.active_players.clear();
        self.needs_tsumo = true;
    }

    pub(crate) fn open_replay_kan_window(&mut self, actor: u8) {
        self.current_player = actor;
        self.current_claims.clear();
        self.active_players.clear();
        let window = self
            .replay_context
            .kan_window
            .as_ref()
            .expect("declared replay kan");
        for seat in 0..4 {
            let offers = window
                .responses
                .get(&seat)
                .into_iter()
                .flatten()
                .filter(|action| action.action_type == ActionType::Ron)
                .cloned()
                .collect::<Vec<_>>();
            if !offers.is_empty() {
                self.active_players.push(seat);
                self.current_claims.insert(seat, offers);
            }
        }
        self.phase = if self.active_players.is_empty() {
            Phase::WaitAct
        } else {
            Phase::WaitResponse
        };
        self.needs_tsumo = self.active_players.is_empty();
        if let Some((_, action)) = &self.pending_kan {
            self.last_discard = action.tile.map(|tile| (actor, tile));
        }
    }

    /// Called by `apply_mjai_event` itself, also when Python has no KAR bridge.
    pub(crate) fn advance_mjai_context(&mut self, event: &MjaiEvent) -> Result<(), String> {
        if !matches!(event, MjaiEvent::Ankan { .. } | MjaiEvent::Kakan { .. })
            && self
                .replay_context
                .kan_window
                .as_ref()
                .is_some_and(|window| !window.applied)
        {
            // A preview belongs to the exact pre-declaration prefix. It must
            // never survive an intervening event and authorize a later state.
            self.replay_context.kan_window = None;
        }
        match event {
            MjaiEvent::StartGame { .. } | MjaiEvent::StartKyoku { .. } => {
                self.replay_context = ReplayContext::default();
                if let MjaiEvent::StartKyoku { tehais, .. } = event {
                    for (seat, hand) in tehais.iter().enumerate().take(4) {
                        self.replay_context.private_hands_complete[seat] =
                            hand.len() == 13 && hand.iter().all(|tile| mjai_to_tid(tile).is_some());
                    }
                }
            }
            MjaiEvent::Reach { actor } => {
                if *actor >= 4 {
                    return Err("Expected a four-player actor".into());
                }
                if self.replay_context.pending_double[*actor].is_some()
                    || self.players[*actor].riichi_declared
                {
                    return Err("Duplicate riichi declaration".into());
                }
                self.replay_context.pending_double[*actor] = Some(
                    self.is_first_turn
                        && self.players[*actor].discards.is_empty()
                        && self.players.iter().all(|player| player.melds.is_empty()),
                );
            }
            MjaiEvent::ReachAccepted { actor } => {
                if *actor >= 4 {
                    return Err("Expected a four-player actor".into());
                }
                let double = self.replay_context.pending_double[*actor]
                    .take()
                    .ok_or("Riichi acceptance lacks its prefix declaration")?;
                self.players[*actor].double_riichi_declared = double;
                self.players[*actor].ippatsu_cycle = true;
            }
            MjaiEvent::Pon { .. } | MjaiEvent::Chi { .. } | MjaiEvent::Kan { .. } => {
                if self
                    .replay_context
                    .kan_window
                    .as_ref()
                    .is_some_and(|window| window.applied)
                {
                    return Err("A kan robbery window cannot resolve as an ordinary call".into());
                }
                self.interrupt_replay_calls();
                self.replay_context.replacement_actor = if let MjaiEvent::Kan { actor, .. } = event
                {
                    Some(*actor as u8)
                } else {
                    None
                };
                self.is_rinshan_flag = false;
            }
            MjaiEvent::Ankan { actor, .. } | MjaiEvent::Kakan { actor, .. } => {
                if *actor >= 4 {
                    return Err("Expected a four-player actor".into());
                }
                if self.replay_context.kan_window.is_none() {
                    // Full recorded games have enough prefix evidence to obtain
                    // every native response. Public-only clients cannot infer
                    // hidden Ron offers; no unknown private intent is invented.
                    if self
                        .replay_context
                        .private_hands_complete
                        .iter()
                        .all(|&known| known)
                    {
                        let mut declaration = None;
                        for action in self._get_legal_actions_internal(*actor as u8) {
                            if matches_declaration(event, *actor as u8, &action)? {
                                declaration = Some(action);
                                break;
                            }
                        }
                        self.prepare_replay_kan(
                            *actor as u8,
                            declaration
                                .ok_or("Observed kan is not a complete legal declaration")?,
                        )?;
                    } else {
                        self.replay_context.kan_window = Some(KanReplayWindow {
                            actor: *actor as u8,
                            declaration: None,
                            responses: HashMap::new(),
                            committed_kans_before: self
                                .players
                                .iter()
                                .flat_map(|player| &player.melds)
                                .filter(|meld| {
                                    matches!(
                                        meld.meld_type,
                                        MeldType::Ankan | MeldType::Kakan | MeldType::Daiminkan
                                    )
                                })
                                .count(),
                            applied: false,
                            ended_by_win: false,
                        });
                    }
                }
                if self
                    .replay_context
                    .kan_window
                    .as_ref()
                    .unwrap()
                    .declaration
                    .is_some()
                {
                    self.validate_replay_kan_event(event)?;
                }
                self.replay_context.kan_window.as_mut().unwrap().applied = true;
                self.replay_context.replacement_actor = Some(*actor as u8);
                let action = match event {
                    MjaiEvent::Ankan { consumed, .. } => {
                        let tiles = consumed
                            .iter()
                            .map(|tile| mjai_to_tid(tile).ok_or("Invalid kan tile"))
                            .collect::<Result<Vec<_>, _>>()?;
                        Action::new(
                            ActionType::Ankan,
                            tiles.first().copied(),
                            tiles,
                            Some(*actor as u8),
                        )
                    }
                    MjaiEvent::Kakan { pai, .. } => Action::new(
                        ActionType::Kakan,
                        mjai_to_tid(pai),
                        vec![],
                        Some(*actor as u8),
                    ),
                    _ => unreachable!(),
                };
                self.pending_kan = Some((*actor as u8, action));
            }
            MjaiEvent::Tsumo { actor, pai } => {
                if *actor >= 4 {
                    return Err("Expected a four-player actor".into());
                }
                if let Some(window) = self
                    .replay_context
                    .kan_window
                    .as_ref()
                    .filter(|window| window.applied)
                {
                    if window.ended_by_win {
                        return Err("A robbed kan cannot receive a replacement draw".into());
                    }
                    pending_kan_commits(
                        &serde_json::json!({"type":"tsumo", "actor":actor}),
                        window.actor as usize,
                        window.committed_kans_before as i64,
                        self.wall.dora_indicators.len() as i64,
                    )?;
                    self.commit_replay_kan();
                }
                if self
                    .replay_context
                    .replacement_actor
                    .is_some_and(|seat| seat as usize != *actor)
                {
                    return Err("Replacement draw belongs to another player".into());
                }
                self.is_rinshan_flag =
                    self.replay_context.replacement_actor.take() == Some(*actor as u8);
                if self.is_rinshan_flag {
                    self.wall.rinshan_draw_count += 1;
                }
                if mjai_to_tid(pai).is_none() {
                    self.replay_context.private_hands_complete[*actor] = false;
                }
            }
            MjaiEvent::Dora { .. } => {
                if let Some(window) = self
                    .replay_context
                    .kan_window
                    .as_ref()
                    .filter(|window| window.applied && !window.ended_by_win)
                    && pending_kan_commits(
                        &serde_json::json!({"type":"dora"}),
                        window.actor as usize,
                        window.committed_kans_before as i64,
                        self.wall.dora_indicators.len() as i64,
                    )?
                {
                    self.commit_replay_kan();
                }
            }
            MjaiEvent::KanCommit { actor } => {
                let window = self
                    .replay_context
                    .kan_window
                    .as_ref()
                    .filter(|window| window.applied && !window.ended_by_win)
                    .ok_or("Kan commitment lacks its pending declaration")?;
                pending_kan_commits(
                    &serde_json::json!({"type":"kan_commit", "actor":actor}),
                    window.actor as usize,
                    window.committed_kans_before as i64,
                    self.wall.dora_indicators.len() as i64,
                )?;
                self.commit_replay_kan();
            }
            MjaiEvent::Hora { .. } | MjaiEvent::Ryukyoku { .. } => {
                if let Some(window) = &mut self.replay_context.kan_window {
                    window.ended_by_win = true;
                }
            }
            MjaiEvent::EndKyoku | MjaiEvent::EndGame => {
                self.replay_context.kan_window = None;
                self.replay_context.replacement_actor = None;
                self.pending_kan = None;
            }
            _ => {}
        }
        Ok(())
    }
}
