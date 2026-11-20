use std::collections::HashMap;
#[cfg(feature = "python")]
use std::sync::Arc;

use serde_json::Value;

use crate::action::{Action, ActionType, Phase};
use crate::errors::{RiichiError, RiichiResult};
use crate::observation::Observation;
use crate::parser::tid_to_mjai;
use crate::replay::Action as LogAction;
use crate::replay::MjaiEvent;
use crate::rule::GameRule;
use crate::rules::settlement::{self, EvaluatedWin, RoundContext, RoundEnd};
use crate::types::{Conditions, INITIAL_HAND_SIZE, Meld, MeldType, WinResult, Wind};

pub mod event_handler;
pub mod game_mode;
pub mod legal_actions;
pub mod player;
pub mod wall;
use event_handler::GameStateEventHandler;
use game_mode::GameModeConfig;
use legal_actions::GameStateLegalActions;
use player::PlayerState;
use wall::WallState;

const NP: usize = 4;

#[cfg_attr(feature = "python", pyo3::pyclass(from_py_object))]
#[derive(Debug, Clone)]
pub struct GameState {
    pub replay_context: crate::replay_context::ReplayContext,
    pub wall: WallState,
    pub players: [PlayerState; 4],

    pub current_player: u8,
    pub turn_count: u32,
    pub is_done: bool,
    pub needs_tsumo: bool,
    pub needs_initialize_next_round: bool,
    pub pending_oya_won: bool,
    pub pending_is_draw: bool,

    pub riichi_sticks: u32,
    pub phase: Phase,
    pub active_players: Vec<u8>,
    pub last_discard: Option<(u8, u8)>,
    pub current_claims: HashMap<u8, Vec<Action>>,
    pub pending_kan: Option<(u8, Action)>,

    pub oya: u8,
    pub honba: u8,
    pub kyoku_idx: u8,
    pub round_wind: u8,

    pub is_rinshan_flag: bool,
    pub is_first_turn: bool,
    pub riichi_pending_acceptance: Option<u8>,
    pub drawn_tile: Option<u8>,

    pub win_results: HashMap<u8, WinResult>,
    pub last_win_results: HashMap<u8, WinResult>,
    pub round_end_scores: Option<Vec<i32>>,

    pub mjai_log: Vec<String>,
    /// Canonical engine events for decision/replay consumers. Unlike legacy
    /// MJAI, a robbable ankan declaration is emitted before its response window
    /// and its later commitment is explicit. This aggregate log contains owner
    /// draws/deals: model consumers must use the native observer projection.
    pub rule_log: Vec<String>,
    rule_pending_ankan: Option<u8>,
    pub player_event_counts: [usize; NP],
    pub kyoku_start_event_counts: [usize; NP],
    pub mjai_log_per_player: [Vec<String>; NP],

    /// Pre-computed progression tuples for the current round (shared by all players).
    /// Incrementally updated in _push_mjai_event to avoid O(N²) JSON re-parsing.
    /// Wrapped in Arc for O(1) snapshot when creating observations.
    #[cfg(feature = "python")]
    pub round_seq_progression: Arc<Vec<[u16; 5]>>,
    #[cfg(feature = "python")]
    pub round_seq_prog_pending_reach: Option<u8>,
    /// When true, incrementally track progression tuples for sequence feature encoding.
    /// Disable when sequence features are not needed to avoid overhead.
    #[cfg(feature = "python")]
    pub enable_seq_caching: bool,

    pub mode: GameModeConfig,
    pub game_mode: u8,
    pub skip_mjai_logging: bool,
    pub seed: Option<u64>,
    pub rule: GameRule,
    pub last_error: Option<String>,
    pub is_after_kan: bool,

    pub riichi_sutehais: [Option<u8>; NP], // Tile discarded when declaring riichi
    pub last_tedashis: [Option<u8>; NP],   // Last hand discard (not tsumogiri)
}

impl GameState {
    /// Replace the replay placeholder wall with a seeded, physically consistent
    /// continuation wall. All already observed/held physical tiles remain fixed.
    pub fn prepare_replay_continuation_wall(&mut self, seed: u64) -> RiichiResult<()> {
        use rand::prelude::SliceRandom;
        use rand::SeedableRng;

        if self.is_done || self.phase != Phase::WaitAct || self.pending_kan.is_some() {
            return Err(RiichiError::InvalidState {
                message: "Continuation wall can only be prepared at a stable own-action cutoff".into(),
            });
        }
        let mut visible = [false; 136];
        for player in &self.players {
            for &tile in &player.hand {
                visible[tile as usize] = true;
            }
            for &tile in &player.discards {
                visible[tile as usize] = true;
            }
            for meld in &player.melds {
                for &tile in &meld.tiles {
                    visible[tile as usize] = true;
                }
            }
        }
        let mut pool: Vec<u8> = (0..136u8).filter(|&tile| !visible[tile as usize]).collect();
        if pool.len() != self.wall.tiles.len() {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "Replay continuation wall has {} unknown tiles but native wall needs {}",
                    pool.len(),
                    self.wall.tiles.len()
                ),
            });
        }

        let mut fixed = Vec::with_capacity(self.wall.dora_indicators.len());
        for (index, &indicator) in self.wall.dora_indicators.iter().enumerate() {
            if visible[indicator as usize] {
                return Err(RiichiError::InvalidState {
                    message: "A public dora indicator is also present in a visible player zone".into(),
                });
            }
            let position = (4 + 2 * index).saturating_sub(self.wall.rinshan_draw_count as usize);
            if position >= self.wall.tiles.len() {
                return Err(RiichiError::InvalidState {
                    message: "A revealed dora indicator lies outside the remaining wall".into(),
                });
            }
            let Some(found) = pool.iter().position(|&tile| tile == indicator) else {
                return Err(RiichiError::InvalidState {
                    message: "A revealed dora indicator is missing from the unknown tile pool".into(),
                });
            };
            pool.swap_remove(found);
            fixed.push((position, indicator));
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        pool.shuffle(&mut rng);
        let mut wall = vec![0u8; self.wall.tiles.len()];
        for (position, tile) in fixed {
            wall[position] = tile;
        }
        let mut next = pool.into_iter();
        let fixed_positions: std::collections::HashSet<usize> = self
            .wall
            .dora_indicators
            .iter()
            .enumerate()
            .map(|(index, _)| (4 + 2 * index).saturating_sub(self.wall.rinshan_draw_count as usize))
            .collect();
        for (index, tile) in wall.iter_mut().enumerate() {
            if !fixed_positions.contains(&index) {
                *tile = next.next().expect("wall pool length was checked");
            }
        }
        if next.next().is_some() {
            return Err(RiichiError::InvalidState {
                message: "Continuation wall left unused physical tiles".into(),
            });
        }
        self.wall.tiles = wall;
        self.wall.seed = Some(seed);
        self.wall.hand_index = 1;
        self.wall.salt = format!("replay-continuation-{seed:016x}");
        self.wall.wall_digest.clear();
        Ok(())
    }

    pub fn np(&self) -> usize {
        NP
    }

    fn _evaluated_win(&self, player_id: u8, result: &WinResult) -> EvaluatedWin {
        let mut liable_player = None;
        let mut liable_yakuman_units = 0;
        if result.yakuman {
            for &yaku in &result.yaku {
                let units = match yaku {
                    37 => 1,
                    50 => {
                        if self.rule.is_daisuushii_double {
                            2
                        } else {
                            1
                        }
                    }
                    _ => continue,
                };
                if let Some(&payer) = self.players[player_id as usize].pao.get(&(yaku as u8)) {
                    liable_player = Some(payer);
                    liable_yakuman_units += units;
                }
            }
        }
        EvaluatedWin {
            player_id,
            result: result.clone(),
            liable_player,
            liable_yakuman_units,
        }
    }

    pub fn new(
        game_mode: u8,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: u8,
        rule: GameRule,
    ) -> Self {
        let mode = GameModeConfig::from_game_mode(game_mode, rule);
        let players = [(); 4].map(|_| PlayerState::new(mode.starting_score()));

        let wall = WallState::new(seed);

        let mut state = Self {
            replay_context: Default::default(),
            wall,
            players,
            current_player: 0,
            turn_count: 0,
            is_done: false,
            needs_tsumo: false,
            needs_initialize_next_round: false,
            pending_oya_won: false,
            pending_is_draw: false,
            riichi_sticks: 0,
            phase: Phase::WaitAct,
            active_players: Vec::new(),
            last_discard: None,
            current_claims: HashMap::new(),
            pending_kan: None,
            oya: 0,
            honba: 0,
            kyoku_idx: 0,
            round_wind,
            is_rinshan_flag: false,
            is_first_turn: true,
            riichi_pending_acceptance: None,
            drawn_tile: None,
            win_results: HashMap::new(),
            last_win_results: HashMap::new(),
            round_end_scores: None,
            mjai_log: Vec::new(),
            rule_log: Vec::new(),
            rule_pending_ankan: None,
            player_event_counts: [0; NP],
            kyoku_start_event_counts: [0; NP],
            mjai_log_per_player: Default::default(),
            #[cfg(feature = "python")]
            round_seq_progression: Arc::new(Vec::new()),
            #[cfg(feature = "python")]
            round_seq_prog_pending_reach: None,
            #[cfg(feature = "python")]
            enable_seq_caching: false,
            mode,
            game_mode,
            skip_mjai_logging,
            seed,
            rule,
            last_error: None,
            is_after_kan: false,
            riichi_sutehais: [None; NP],
            last_tedashis: [None; NP],
        };

        if !state.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_game".to_string()));
            state._push_mjai_event(Value::Object(ev));
        }

        // Initial setup
        state._initialize_round(0, round_wind, 0, 0, None, None);
        state
    }

    /// Clear logs and event counters in preparation for a new game.
    /// Call `_initialize_round()` afterwards to set up the first round.
    pub fn reset(&mut self) {
        self.replay_context = Default::default();
        self.mjai_log = Vec::new();
        self.rule_log.clear();
        self.rule_pending_ankan = None;
        self.mjai_log_per_player = Default::default();
        self.player_event_counts = [0; NP];
        self.kyoku_start_event_counts = [0; NP];
        #[cfg(feature = "python")]
        {
            Arc::make_mut(&mut self.round_seq_progression).clear();
            self.round_seq_prog_pending_reach = None;
        }

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_game".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }
    }

    pub fn get_observation(&mut self, player_id: u8) -> Observation {
        let pid = player_id as usize;

        let masked_hands: [Vec<u8>; 4] = std::array::from_fn(|i| {
            if i == pid {
                self.players[i].hand.clone()
            } else {
                Vec::new()
            }
        });

        let legal_actions = if self.is_done {
            Vec::new()
        } else if (self.phase == Phase::WaitAct && self.current_player == player_id)
            || (self.phase == Phase::WaitResponse && self.active_players.contains(&player_id))
        {
            self._get_legal_actions_internal(player_id)
        } else {
            Vec::new()
        };

        // Return incremental delta events (for new_events property, debugging, etc.)
        let old_count = self.player_event_counts[pid];
        let full_log_len = self.mjai_log_per_player[pid].len();
        let new_events = if old_count < full_log_len {
            self.mjai_log_per_player[pid][old_count..].to_vec()
        } else {
            Vec::new()
        };
        self.player_event_counts[pid] = full_log_len;

        let calc = crate::hand_evaluator::HandEvaluator::new(
            self.players[pid].hand.clone(),
            self.players[pid].melds.clone(),
        );
        let waits = calc.get_waits_u8_with_rule(&self.rule);
        let is_tenpai = !waits.is_empty();

        let melds: [Vec<Meld>; 4] = std::array::from_fn(|i| self.players[i].melds.clone());
        let discards: [Vec<u8>; 4] = std::array::from_fn(|i| self.players[i].discards.clone());
        let scores: [i32; 4] = std::array::from_fn(|i| self.players[i].score);
        let riichi_declared: [bool; 4] = std::array::from_fn(|i| self.players[i].riichi_declared);

        let mut obs = Observation::new(
            player_id,
            masked_hands,
            melds,
            discards,
            self.wall.dora_indicators.clone(),
            scores,
            riichi_declared,
            legal_actions,
            new_events,
            self.honba,
            self.riichi_sticks,
            self.round_wind,
            self.oya,
            self.kyoku_idx,
            waits,
            is_tenpai,
            self.riichi_sutehais,
            self.last_tedashis,
            self.last_discard.map(|(_pid, tile)| tile as u32),
            self.drawn_tile,
        );

        obs.rule = self.rule;
        obs.forced_tedashi = self.is_forced_tedashi(player_id);

        // Attach pre-computed progression snapshot.
        #[cfg(feature = "python")]
        if self.enable_seq_caching {
            obs.cached_progression = Some((*self.round_seq_progression).clone());
        }

        obs
    }

    pub fn get_observation_for_replay(
        &mut self,
        pid: u8,
        env_action: &Action,
        log_action_str: &str,
    ) -> RiichiResult<Observation> {
        let original_phase = self.phase;
        let original_active_players = self.active_players.clone();
        let original_claims = self.current_claims.clone();
        let original_riichi = self.players[pid as usize].riichi_declared;

        match env_action.action_type {
            ActionType::Ron | ActionType::Chi | ActionType::Pon | ActionType::Daiminkan => {
                self.phase = Phase::WaitResponse;
                self.active_players = vec![pid];
                self.current_claims
                    .entry(pid)
                    .or_default()
                    .push(env_action.clone());
            }
            _ => {}
        }

        let mut obs = self.get_observation(pid);

        let mut exists = obs
            ._legal_actions
            .iter()
            .any(|a| a.action_type == env_action.action_type && a.tile == env_action.tile);

        if !exists
            && env_action.action_type == ActionType::Discard
            && self.players[pid as usize].riichi_declared
        {
            self.players[pid as usize].riichi_declared = false;
            let new_obs = self.get_observation(pid);
            let is_legal_retry = new_obs
                ._legal_actions
                .iter()
                .any(|a| a.action_type == ActionType::Discard && a.tile == env_action.tile);

            if is_legal_retry {
                obs = new_obs;
                exists = true;
            } else {
                self.players[pid as usize].riichi_declared = original_riichi;
            }
        }

        self.phase = original_phase;
        self.active_players = original_active_players;
        self.current_claims = original_claims;

        if !exists {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "Replay desync:\n  Env action: {:?}\n  Log action: {}\n  Self state:\n    phase: {:?}\n    drawn: {:?}",
                    env_action, log_action_str, self.phase, self.drawn_tile
                ),
            });
        }

        Ok(obs)
    }

    pub fn step(&mut self, actions: &HashMap<u8, Action>) {
        if self.is_done {
            return;
        }

        if self.needs_initialize_next_round {
            self._initialize_next_round(self.pending_oya_won, self.pending_is_draw);
            return;
        }
        // Validation
        let np = NP;
        let mut normalized_kakan = None;
        for pid in 0..np {
            if let Some(act) = actions.get(&(pid as u8)) {
                let legals = self._get_legal_actions_internal(pid as u8);
                let is_valid = legals.iter().any(|l| {
                    if l.action_type != act.action_type {
                        return false;
                    }

                    if act.action_type == ActionType::Kakan {
                        normalized_kakan = act.resolve_kakan(l);
                        return normalized_kakan.is_some();
                    }

                    let tiles_match = l.tile == act.tile;
                    let consumes_match = l.consume_tiles == act.consume_tiles;

                    if tiles_match {
                        if consumes_match {
                            return true;
                        }
                        // Allow empty consume for Discard, Riichi, Tsumo, Ron, Pass
                        if act.consume_tiles.is_empty()
                            && matches!(
                                l.action_type,
                                ActionType::Discard
                                    | ActionType::Riichi
                                    | ActionType::Tsumo
                                    | ActionType::Ron
                                    | ActionType::Pass
                            )
                        {
                            return true;
                        }
                    }

                    if consumes_match && l.action_type == ActionType::Ankan {
                        return true;
                    }

                    // Allow None from python for context-implied actions
                    if act.tile.is_none() {
                        return matches!(
                            l.action_type,
                            ActionType::Tsumo
                                | ActionType::Ron
                                | ActionType::Riichi
                                | ActionType::KyushuKyuhai
                                | ActionType::Kita
                        );
                    }
                    false
                });

                if !is_valid {
                    let reason = format!("Error: Illegal Action by Player {}", pid);
                    self.last_error = Some(reason.clone());
                    self._trigger_ryukyoku(&reason);
                    return;
                }
            }
        }

        // --- Phase: WaitAct (Discards, Riichi, Tsumo, Kan) ---
        if self.phase == Phase::WaitAct {
            let pid = self.current_player;
            if let Some(act) = actions.get(&pid) {
                let act = normalized_kakan.as_ref().unwrap_or(act);
                match act.action_type {
                    ActionType::Discard => {
                        if let Some(tile) = act.tile {
                            let mut tsumogiri = false;
                            let mut valid = false;
                            if let Some(dt) = self.drawn_tile
                                && dt == tile
                            {
                                tsumogiri = true;
                                valid = true;
                            }
                            if let Some(idx) = self.players[pid as usize]
                                .hand
                                .iter()
                                .position(|&t| t == tile)
                            {
                                self.players[pid as usize].hand.remove(idx);
                                self.players[pid as usize].hand.sort();
                                valid = true;
                                if let Some(dt) = self.drawn_tile
                                    && dt == tile
                                {
                                    tsumogiri = true;
                                }
                            }
                            if valid {
                                self._resolve_discard(pid, tile, tsumogiri);
                            }
                        }
                    }
                    ActionType::KyushuKyuhai => {
                        self._trigger_ryukyoku("kyushu_kyuhai");
                    }
                    ActionType::Riichi => {
                        // Declare Riichi. `!riichi_stage` guards against a
                        // second Riichi action arriving while the previous
                        // one is still mid-declaration (between the reach
                        // event and its committing discard).
                        if self.players[pid as usize].score >= 1000
                            && self.wall.drawable_count >= 4
                            && !self.players[pid as usize].riichi_declared
                            && !self.players[pid as usize].riichi_stage
                        {
                            self.players[pid as usize].riichi_stage = true;
                            if !self.skip_mjai_logging {
                                let mut ev = serde_json::Map::new();
                                ev.insert("type".to_string(), Value::String("reach".to_string()));
                                ev.insert("actor".to_string(), Value::Number(pid.into()));
                                self._push_mjai_event(Value::Object(ev));
                            }
                            if let Some(t) = act.tile {
                                let mut tsumogiri = false;
                                if let Some(dt) = self.drawn_tile
                                    && dt == t
                                {
                                    tsumogiri = true;
                                }
                                if let Some(idx) =
                                    self.players[pid as usize].hand.iter().position(|&x| x == t)
                                {
                                    self.players[pid as usize].hand.remove(idx);
                                    self.players[pid as usize].hand.sort();
                                }
                                self._resolve_discard(pid, t, tsumogiri);
                            }
                        }
                    }
                    ActionType::Ankan => {
                        // Ankan Logic
                        let tile = act.tile.or(act.consume_tiles.first().copied()).unwrap_or(0);
                        self._push_rule_event(serde_json::json!({
                            "type": "ankan", "actor": pid,
                            "consumed": act.consume_tiles.iter().map(|&t| tid_to_mjai(t)).collect::<Vec<_>>(),
                        }));
                        self.rule_pending_ankan = Some(pid);
                        let mut chankan_ronners = Vec::new();
                        if self.rule.allows_ron_on_ankan_for_kokushi_musou {
                            for i in 0..np as u8 {
                                if i == pid {
                                    continue;
                                }

                                // Check Kokushi Only
                                let hand = &self.players[i as usize].hand;
                                let melds = &self.players[i as usize].melds;

                                // Furiten check
                                let tile_class = tile / 4;
                                let in_discards = self.players[i as usize]
                                    .discards
                                    .iter()
                                    .any(|&d| d / 4 == tile_class);
                                if in_discards {
                                    continue;
                                }

                                let p_wind = (i + np as u8 - self.oya) % np as u8;
                                let cond = Conditions {
                                    tsumo: false,
                                    riichi: self.players[i as usize].riichi_declared,
                                    chankan: true,
                                    player_wind: Wind::from(p_wind),
                                    round_wind: Wind::from(self.round_wind),
                                    ..Default::default()
                                };
                                let calc = crate::hand_evaluator::HandEvaluator::new(
                                    hand.clone(),
                                    melds.clone(),
                                );
                                let res = calc.calc(
                                    tile,
                                    self.wall.dora_indicators.clone(),
                                    vec![],
                                    Some(cond),
                                );

                                // 42=Kokushi, 49=Kokushi13
                                if res.is_win && (res.yaku.contains(&42) || res.yaku.contains(&49))
                                {
                                    chankan_ronners.push(i);
                                    self.current_claims.entry(i).or_default().push(Action::new(
                                        ActionType::Ron,
                                        Some(tile),
                                        vec![],
                                        Some(i),
                                    ));
                                }
                            }
                        }

                        if !chankan_ronners.is_empty() {
                            self.pending_kan = Some((pid, act.clone()));
                            self.phase = Phase::WaitResponse;
                            self.active_players = chankan_ronners;
                            self.last_discard = Some((pid, tile));
                        } else {
                            self._resolve_kan(pid, act.clone());
                        }
                    }
                    ActionType::Kakan => {
                        let tile = act.tile.or(act.consume_tiles.first().copied()).unwrap_or(0);
                        // Keep the original Pon and added tile during chankan.
                        // _resolve_kan commits the physical mutation only after
                        // every Ron opportunity has gone unused.

                        // Log Kakan immediately (before Chankan check)
                        if !self.skip_mjai_logging {
                            let mut ev = serde_json::Map::new();
                            ev.insert("type".to_string(), Value::String("kakan".to_string()));
                            ev.insert("actor".to_string(), Value::Number(pid.into()));
                            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                            let cons: Vec<String> =
                                act.consume_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
                            ev.insert(
                                "consumed".to_string(),
                                serde_json::to_value(cons).expect("valid JSON"),
                            );
                            self._push_mjai_event(Value::Object(ev));
                        }

                        // Reveal any pending kan doras from previous kans
                        while self.wall.pending_kan_dora_count > 0 {
                            self.wall.pending_kan_dora_count -= 1;
                            self._reveal_kan_dora();
                        }

                        // Kakan Logic
                        // Check Chankan
                        let tile = act.tile.or(act.consume_tiles.first().copied()).unwrap_or(0);
                        let mut chankan_ronners = Vec::new();
                        for i in 0..np as u8 {
                            if i == pid {
                                continue;
                            }
                            // Check WinResult
                            let hand = &self.players[i as usize].hand;
                            let melds = &self.players[i as usize].melds;
                            let p_wind = (i + np as u8 - self.oya) % np as u8;
                            let cond = Conditions {
                                tsumo: false,
                                riichi: self.players[i as usize].riichi_declared,
                                double_riichi: self.players[i as usize].double_riichi_declared,
                                ippatsu: self.players[i as usize].ippatsu_cycle,
                                player_wind: Wind::from(p_wind),
                                round_wind: Wind::from(self.round_wind),
                                chankan: true,
                                haitei: false,
                                houtei: false,
                                rinshan: false,
                                tsumo_first_turn: false,
                                riichi_sticks: self.riichi_sticks,
                                honba: self.honba as u32,
                                ..Default::default()
                            };
                            let calc = crate::hand_evaluator::HandEvaluator::new(
                                hand.clone(),
                                melds.clone(),
                            );

                            // Check Furiten
                            let mut is_furiten = false;
                            let waits = calc.get_waits_u8_with_rule(&self.rule);
                            for &w in &waits {
                                if self.players[i as usize]
                                    .discards
                                    .iter()
                                    .any(|&d| d / 4 == w)
                                {
                                    is_furiten = true;
                                    break;
                                }
                            }
                            if self.players[i as usize].missed_agari_riichi
                                || self.players[i as usize].missed_agari_doujun
                            {
                                is_furiten = true;
                            }

                            // If valid:
                            let res = if !is_furiten {
                                calc.calc(
                                    tile,
                                    self.wall.dora_indicators.clone(),
                                    vec![],
                                    Some(cond),
                                )
                            } else {
                                crate::types::WinResult::new(
                                    false,
                                    false,
                                    0,
                                    0,
                                    0,
                                    vec![],
                                    0,
                                    0,
                                    None,
                                    false,
                                )
                            };

                            if res.is_win && (res.yakuman || res.han >= 1) {
                                // Add Ron action offer
                                chankan_ronners.push(i);
                                self.current_claims.entry(i).or_default().push(Action::new(
                                    ActionType::Ron,
                                    Some(tile),
                                    vec![],
                                    Some(i),
                                ));
                            }
                        }

                        if !chankan_ronners.is_empty() {
                            self.pending_kan = Some((pid, act.clone()));
                            self.phase = Phase::WaitResponse;
                            self.active_players = chankan_ronners;
                            self.last_discard = Some((pid, tile)); // Treat Kakan tile as discard for Ron targeting
                        } else {
                            self._resolve_kan(pid, act.clone());
                        }
                    }
                    ActionType::Tsumo => {
                        let hand = &self.players[pid as usize].hand;
                        let melds = &self.players[pid as usize].melds;
                        let p_wind = (pid + np as u8 - self.oya) % np as u8;
                        let cond = Conditions {
                            tsumo: true,
                            riichi: self.players[pid as usize].riichi_declared,
                            double_riichi: self.players[pid as usize].double_riichi_declared,
                            ippatsu: self.players[pid as usize].ippatsu_cycle,
                            haitei: self.wall.drawable_count == 0 && !self.is_rinshan_flag,
                            rinshan: self.is_rinshan_flag,
                            tsumo_first_turn: self.is_first_turn
                                && self.players.iter().all(|p| p.melds.is_empty()),
                            player_wind: Wind::from(p_wind),
                            round_wind: Wind::from(self.round_wind),
                            riichi_sticks: self.riichi_sticks,
                            honba: self.honba as u32,
                            ..Default::default()
                        };
                        let calc =
                            crate::hand_evaluator::HandEvaluator::new(hand.clone(), melds.clone());
                        let win_tile = self.drawn_tile.unwrap_or(0);
                        let ura_indicators = if self.players[pid as usize].riichi_declared {
                            self._get_ura_indicators()
                        } else {
                            vec![]
                        };
                        let mut res = calc.calc(
                            win_tile,
                            self.wall.dora_indicators.clone(),
                            ura_indicators,
                            Some(cond.clone()),
                        );

                        // Cap double yakuman patterns when not enabled per rule flags
                        if res.yakuman && res.han > 13 {
                            let mut cap = 0u32;
                            for &y in &res.yaku {
                                match y {
                                    47 if !self.rule.is_junsei_chuurenpoutou_double => cap += 13,
                                    48 if !self.rule.is_suuankou_tanki_double => cap += 13,
                                    49 if !self.rule.is_kokushi_musou_13machi_double => cap += 13,
                                    50 if !self.rule.is_daisuushii_double => cap += 13,
                                    _ => {}
                                }
                            }
                            if cap > 0 {
                                res.han = res.han.saturating_sub(cap).max(13);
                                let capped = crate::score::calculate_score(
                                    res.han as u8,
                                    0,
                                    pid == self.oya,
                                    cond.tsumo,
                                    cond.honba,
                                    np as u8,
                                );
                                res.ron_agari = capped.pay_ron;
                                res.tsumo_agari_oya = capped.pay_tsumo_oya;
                                res.tsumo_agari_ko = capped.pay_tsumo_ko;
                            }
                        }

                        if res.is_win {
                            let win = self._evaluated_win(pid, &res);
                            let deltas = settlement::win_payments(
                                &[win],
                                self.oya,
                                None,
                                self.honba as u32,
                                self.riichi_sticks,
                                &self.rule,
                            )
                            .expect("validated online self-draw settlement")
                            .deltas;
                            self.riichi_sticks = 0;

                            for (i, p) in self.players.iter_mut().enumerate() {
                                p.score += deltas[i];
                                p.score_delta = deltas[i];
                            }

                            let mut val = res;
                            for (&yid, &liable) in &self.players[pid as usize].pao {
                                if val.yaku.contains(&(yid as u32)) {
                                    val.pao_payer = Some(liable);
                                    break;
                                }
                            }
                            self.win_results.insert(pid, val);

                            if !self.skip_mjai_logging {
                                let mut ev = serde_json::Map::new();
                                ev.insert("type".to_string(), Value::String("hora".to_string()));
                                ev.insert("actor".to_string(), Value::Number(pid.into()));
                                ev.insert("target".to_string(), Value::Number(pid.into()));
                                ev.insert(
                                    "deltas".to_string(),
                                    serde_json::to_value(deltas).expect("valid JSON"),
                                );
                                ev.insert("tsumo".to_string(), Value::Bool(true));
                                let public_hand: Vec<_> = self.players[pid as usize]
                                    .hand
                                    .iter()
                                    .map(|&tile| tid_to_mjai(tile))
                                    .collect();
                                ev.insert(
                                    "public_hand".to_string(),
                                    serde_json::to_value(public_hand).expect("valid JSON"),
                                );
                                ev.insert(
                                    "scores".to_string(),
                                    serde_json::to_value(self.players.each_ref().map(|p| p.score))
                                        .expect("valid JSON"),
                                );
                                ev.insert("visibility_complete".to_string(), Value::Bool(true));
                                ev.insert(
                                    "public_reveal_provenance".to_string(),
                                    Value::String("native_round_resolution_v1".to_string()),
                                );

                                let mut ura_markers = Vec::new();
                                if self.players[pid as usize].riichi_declared {
                                    ura_markers = self._get_ura_markers();
                                }
                                ev.insert(
                                    "ura_markers".to_string(),
                                    serde_json::to_value(&ura_markers).expect("valid JSON"),
                                );

                                self._push_native_resolution(Value::Object(ev));
                            }

                            self._initialize_next_round(pid == self.oya, false);
                        } else {
                            self.current_player = (self.current_player + 1) % np as u8;
                            self._deal_next();
                        }
                    }
                    ActionType::Kita => {
                        // Kita is only valid in 3P; handled by GameState3P
                    }
                    _ => {}
                }
            }
        } else if self.phase == Phase::WaitResponse {
            // Check Missed WinResult for all who could Ron but didn't
            for (&pid, legals) in &self.current_claims {
                if legals.iter().any(|a| a.action_type == ActionType::Ron) {
                    let mut roned = false;
                    if let Some(act) = actions.get(&pid)
                        && act.action_type == ActionType::Ron
                    {
                        roned = true;
                    }
                    if !roned {
                        self.players[pid as usize].record_declined_ron();
                    }
                }
            }

            let mut ron_claims = Vec::new();
            let mut call_claim: Option<(u8, Action)> = None;

            for &pid in &self.active_players {
                if let Some(act) = actions.get(&pid) {
                    if act.action_type == ActionType::Ron {
                        ron_claims.push(pid);
                    } else if act.action_type == ActionType::Pon
                        || act.action_type == ActionType::Daiminkan
                        || act.action_type == ActionType::Chi
                    {
                        if let Some((_old_pid, old_act)) = &call_claim {
                            let old_is_pon = old_act.action_type == ActionType::Pon
                                || old_act.action_type == ActionType::Daiminkan;
                            let new_is_pon = act.action_type == ActionType::Pon
                                || act.action_type == ActionType::Daiminkan;
                            if !old_is_pon && new_is_pon {
                                call_claim = Some((pid, act.clone()));
                            }
                        } else {
                            call_claim = Some((pid, act.clone()));
                        }
                    }
                }
            }

            // Retire this response's offers after recording missed wins and
            // selecting claims, before a call can open a new response window.
            self.current_claims.clear();

            if !ron_claims.is_empty() {
                // Sanchaho: all non-discarders ron → abortive draw
                if ron_claims.len() >= NP - 1 && self.rule.sanchaho_is_draw {
                    self._trigger_ryukyoku("sanchaho");
                    return;
                }

                let (target_pid, win_tile) = self.last_discard.unwrap_or((self.current_player, 0));

                ron_claims.sort_by_key(|&pid| (pid + np as u8 - target_pid) % np as u8);

                let winners = ron_claims;

                let mut evaluated_wins = Vec::new();
                let mut oya_won = false;

                for &w_pid in &winners {
                    let hand = &self.players[w_pid as usize].hand;
                    let melds = &self.players[w_pid as usize].melds;
                    let p_wind = (w_pid + np as u8 - self.oya) % np as u8;
                    let is_chankan = self.pending_kan.is_some();

                    let cond = Conditions {
                        tsumo: false,
                        riichi: self.players[w_pid as usize].riichi_declared,
                        double_riichi: self.players[w_pid as usize].double_riichi_declared,
                        ippatsu: self.players[w_pid as usize].ippatsu_cycle,
                        haitei: false,
                        houtei: self.wall.drawable_count == 0 && !self.is_rinshan_flag,
                        rinshan: false,
                        chankan: is_chankan,
                        tsumo_first_turn: false,
                        player_wind: Wind::from(p_wind),
                        round_wind: Wind::from(self.round_wind),
                        riichi_sticks: self.riichi_sticks,
                        honba: 0,
                        ..Default::default()
                    };

                    let calc =
                        crate::hand_evaluator::HandEvaluator::new(hand.clone(), melds.clone());
                    let ura_indicators = if self.players[w_pid as usize].riichi_declared {
                        self._get_ura_indicators()
                    } else {
                        vec![]
                    };
                    let mut res = calc.calc(
                        win_tile,
                        self.wall.dora_indicators.clone(),
                        ura_indicators,
                        Some(cond),
                    );

                    // Cap double yakuman patterns when not enabled per rule flags
                    if res.yakuman && res.han > 13 {
                        let mut cap = 0u32;
                        for &y in &res.yaku {
                            match y {
                                47 if !self.rule.is_junsei_chuurenpoutou_double => cap += 13,
                                48 if !self.rule.is_suuankou_tanki_double => cap += 13,
                                49 if !self.rule.is_kokushi_musou_13machi_double => cap += 13,
                                50 if !self.rule.is_daisuushii_double => cap += 13,
                                _ => {}
                            }
                        }
                        if cap > 0 {
                            res.han = res.han.saturating_sub(cap).max(13);
                            let capped = crate::score::calculate_score(
                                res.han as u8,
                                0,
                                w_pid == self.oya,
                                false,
                                0,
                                np as u8,
                            );
                            res.ron_agari = capped.pay_ron;
                            res.tsumo_agari_oya = capped.pay_tsumo_oya;
                            res.tsumo_agari_ko = capped.pay_tsumo_ko;
                        }
                    }

                    if res.is_win {
                        evaluated_wins.push(self._evaluated_win(w_pid, &res));

                        let mut val = res;
                        for (&yid, &liable) in &self.players[w_pid as usize].pao {
                            if val.yaku.contains(&(yid as u32)) {
                                val.pao_payer = Some(liable);
                                break;
                            }
                        }
                        self.win_results.insert(w_pid, val);

                        if w_pid == self.oya {
                            oya_won = true;
                        }
                    }
                }

                let payments = settlement::win_payments(
                    &evaluated_wins,
                    self.oya,
                    Some(target_pid),
                    self.honba as u32,
                    self.riichi_sticks,
                    &self.rule,
                )
                .expect("validated online ron settlement");
                self.riichi_sticks = 0;
                for (i, p) in self.players.iter_mut().enumerate() {
                    p.score += payments.deltas[i];
                    p.score_delta = payments.deltas[i];
                }
                for paid in payments.winners {
                    let w_pid = paid.player_id;
                    if let Some(result) = self.win_results.get_mut(&w_pid) {
                        let scoring_han = if result.yakuman {
                            result.han
                        } else {
                            result.han.min(13)
                        };
                        let score = crate::score::calculate_score(
                            scoring_han as u8,
                            result.fu as u8,
                            w_pid == self.oya,
                            false,
                            paid.honba,
                            4,
                        );
                        result.ron_agari = score.pay_ron;
                        result.tsumo_agari_oya = score.pay_tsumo_oya;
                        result.tsumo_agari_ko = score.pay_tsumo_ko;
                    }
                    if !self.skip_mjai_logging {
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), Value::String("hora".to_string()));
                        ev.insert("actor".to_string(), Value::Number(w_pid.into()));
                        ev.insert("target".to_string(), Value::Number(target_pid.into()));
                        ev.insert(
                            "deltas".to_string(),
                            serde_json::to_value(paid.deltas).expect("valid JSON"),
                        );
                        let mut public_hand = self.players[w_pid as usize].hand.clone();
                        public_hand.push(win_tile);
                        ev.insert(
                            "public_hand".to_string(),
                            serde_json::to_value(
                                public_hand.into_iter().map(tid_to_mjai).collect::<Vec<_>>(),
                            )
                            .expect("valid JSON"),
                        );
                        // Every winner sees the same final simultaneous settlement.
                        ev.insert(
                            "scores".to_string(),
                            serde_json::to_value(self.players.each_ref().map(|p| p.score))
                                .expect("valid JSON"),
                        );
                        ev.insert("visibility_complete".to_string(), Value::Bool(true));
                        ev.insert(
                            "public_reveal_provenance".to_string(),
                            Value::String("native_round_resolution_v1".to_string()),
                        );
                        let ura_markers = if self.players[w_pid as usize].riichi_declared {
                            self._get_ura_markers()
                        } else {
                            Vec::new()
                        };
                        ev.insert(
                            "ura_markers".to_string(),
                            serde_json::to_value(&ura_markers).expect("valid JSON"),
                        );
                        self._push_native_resolution(Value::Object(ev));
                    }
                }

                self._initialize_next_round(oya_won, false);
            } else if let Some((claimer, action)) = call_claim {
                self._accept_riichi();
                self.is_rinshan_flag = false;
                self.is_first_turn = false;

                // Discard was called → discarder loses nagashi eligibility
                if let Some((discarder_pid, _)) = self.last_discard {
                    self.players[discarder_pid as usize].nagashi_eligible = false;
                }

                for p in 0..np {
                    self.players[p].ippatsu_cycle = false;
                }

                if action.action_type == ActionType::Daiminkan {
                    self.current_player = claimer;
                    self.active_players = vec![claimer];
                    self.players[claimer as usize].forbidden_discards.clear();
                    // Handled exclusively by _resolve_kan
                    self._resolve_kan(claimer, action.clone());
                    return; // Skip the rest of claim handling (Pon/Chi)
                }

                for &t in &action.consume_tiles {
                    if let Some(idx) = self.players[claimer as usize]
                        .hand
                        .iter()
                        .position(|&x| x == t)
                    {
                        self.players[claimer as usize].hand.remove(idx);
                    }
                }
                let (discarder, tile) = self.last_discard.expect("last_discard must be set");
                let mut tiles = action.consume_tiles.clone();
                tiles.push(tile);
                tiles.sort();
                let meld_type = match action.action_type {
                    ActionType::Pon => MeldType::Pon,
                    ActionType::Chi => MeldType::Chi,
                    _ => MeldType::Chi, // Should not happen for this block anymore
                };
                self.players[claimer as usize].melds.push(Meld {
                    meld_type,
                    tiles: tiles.clone(),
                    opened: true,
                    from_who: discarder as i8,
                    called_tile: Some(tile),
                });

                if !self.skip_mjai_logging {
                    let type_str = match action.action_type {
                        ActionType::Pon => Some("pon"),
                        ActionType::Chi => Some("chi"),
                        ActionType::Daiminkan => Some("daiminkan"),
                        _ => None,
                    };
                    if let Some(s) = type_str {
                        let mut ev = serde_json::Map::new();
                        ev.insert("type".to_string(), serde_json::Value::String(s.to_string()));
                        ev.insert(
                            "actor".to_string(),
                            serde_json::Value::Number(claimer.into()),
                        );
                        ev.insert(
                            "target".to_string(),
                            serde_json::Value::Number(discarder.into()),
                        );
                        ev.insert(
                            "pai".to_string(),
                            serde_json::Value::String(tid_to_mjai(tile)),
                        );
                        let cons_strs: Vec<String> = action
                            .consume_tiles
                            .iter()
                            .map(|&t| tid_to_mjai(t))
                            .collect();
                        ev.insert(
                            "consumed".to_string(),
                            serde_json::to_value(cons_strs).expect("valid JSON"),
                        );
                        self._push_mjai_event(serde_json::Value::Object(ev));
                    }
                }

                // PAO implementation
                if meld_type == MeldType::Pon
                    || meld_type == MeldType::Daiminkan
                    || meld_type == MeldType::Kakan
                {
                    let tile_val = tile / 4;
                    if (31..=33).contains(&tile_val) {
                        let dragon_melds = self.players[claimer as usize]
                            .melds
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (31..=33).contains(&t) && (m.meld_type != MeldType::Chi)
                            })
                            .count();
                        if dragon_melds == 3 {
                            self.players[claimer as usize].pao.insert(37, discarder);
                        }
                    } else if (27..=30).contains(&tile_val) {
                        let wind_melds = self.players[claimer as usize]
                            .melds
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (27..=30).contains(&t) && (m.meld_type != MeldType::Chi)
                            })
                            .count();
                        if wind_melds == 4 {
                            self.players[claimer as usize].pao.insert(50, discarder);
                        }
                    }
                }

                self.current_player = claimer;
                self.phase = Phase::WaitAct;
                self.active_players = vec![claimer];
                self.players[claimer as usize].forbidden_discards.clear();

                if self.rule.kuikae_forbidden && action.action_type == ActionType::Pon {
                    self.players[claimer as usize].forbidden_discards.push(tile);
                } else if self.rule.kuikae_forbidden && action.action_type == ActionType::Chi {
                    self.players[claimer as usize].forbidden_discards.push(tile);
                    let t34 = tile / 4;
                    let mut consumed_34: Vec<u8> =
                        action.consume_tiles.iter().map(|&x| x / 4).collect();
                    consumed_34.sort();
                    if consumed_34[0] == t34 + 1 && consumed_34[1] == t34 + 2 {
                        if t34 % 9 <= 5 {
                            self.players[claimer as usize]
                                .forbidden_discards
                                .push((t34 + 3) * 4);
                        }
                    } else if t34 >= 2
                        && consumed_34[1] == t34 - 1
                        && consumed_34[0] == t34 - 2
                        && t34 % 9 >= 3
                    {
                        self.players[claimer as usize]
                            .forbidden_discards
                            .push((t34 - 3) * 4);
                    }
                }

                if action.action_type == ActionType::Daiminkan {
                    self._resolve_kan(claimer, action.clone());
                } else {
                    self.needs_tsumo = false;
                    self.drawn_tile = None;
                }
            } else {
                // All Pass
                self.active_players.clear();

                if let Some((pk_pid, pk_act)) = self.pending_kan.take() {
                    self._resolve_kan(pk_pid, pk_act);
                } else {
                    self._accept_riichi();
                    if self.check_abortive_draw() {
                        return;
                    }
                    self.turn_count += 1;
                    self.current_player = (self.current_player + 1) % np as u8;
                    self._deal_next();
                    if self.turn_count >= np as u32 {
                        self.is_first_turn = false;
                    }
                }
            }
        }
    }

    fn is_forced_tedashi(&self, pid: u8) -> bool {
        self.rule.dealer_first_discard_is_tedashi
            && pid == self.oya
            && self.is_first_turn
            && self.players[pid as usize].discards.is_empty()
    }

    fn _resolve_discard(&mut self, pid: u8, tile: u8, tsumogiri: bool) {
        let tsumogiri = tsumogiri && !self.is_forced_tedashi(pid);
        // After a discard the rinshan context is over. Clearing here ensures
        // that houtei (last-discard win) is correctly detected even when the
        // discard comes after a kan draw.
        self.is_rinshan_flag = false;

        // Clear ippatsu for the discarding player. When a riichi player discards
        // without tsumo winning, their ippatsu window is over. Note: the riichi
        // declaration discard won't wrongly clear it because _accept_riichi() runs
        // AFTER this and sets ippatsu_cycle = true.
        self.players[pid as usize].ippatsu_cycle = false;
        self.players[pid as usize].discards.push(tile);
        self.last_discard = Some((pid, tile));
        self.drawn_tile = None;
        self.players[pid as usize]
            .discard_from_hand
            .push(!tsumogiri);
        let riichi_stage = self.players[pid as usize].riichi_stage;
        self.players[pid as usize]
            .discard_is_riichi
            .push(riichi_stage);

        // Track last tedashi (hand discard, not tsumogiri)
        if !tsumogiri {
            self.last_tedashis[pid as usize] = Some(tile);
        }

        self.needs_tsumo = true;

        if self.players[pid as usize].riichi_stage {
            self.riichi_sutehais[pid as usize] = Some(tile);
            self.players[pid as usize].riichi_declared = true;
            if self.is_first_turn {
                self.players[pid as usize].double_riichi_declared = true;
            }
            self.players[pid as usize].riichi_declaration_index =
                Some(self.players[pid as usize].discards.len() - 1);
            self.players[pid as usize].riichi_stage = false;
            self.riichi_pending_acceptance = Some(pid);
        }

        // Reveal pending kan doras before dahai event
        while self.wall.pending_kan_dora_count > 0 {
            self.wall.pending_kan_dora_count -= 1;
            self._reveal_kan_dora();
        }

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("dahai".to_string()));
            ev.insert("actor".to_string(), Value::Number(pid.into()));
            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
            ev.insert("tsumogiri".to_string(), Value::Bool(tsumogiri));
            self._push_mjai_event(Value::Object(ev));
        }

        self.players[pid as usize].nagashi_eligible &= crate::types::is_terminal_tile(tile);
        self.populate_discard_claims(pid, tile);
        if !self.active_players.is_empty() {
            self.phase = Phase::WaitResponse;
        } else {
            if let Some(_rp) = self.riichi_pending_acceptance {
                self._accept_riichi();
            }
            if !self.check_abortive_draw() {
                self.turn_count += 1;
                self.current_player = (pid + 1) % NP as u8;
                self._deal_next();
                if self.turn_count >= NP as u32 {
                    self.is_first_turn = false;
                }
            }
        }
    }

    fn populate_discard_claims(&mut self, discarder: u8, tile: u8) {
        // Only the player's own discard ends same-turn furiten, including after a call.
        self.players[discarder as usize].missed_agari_doujun = false;
        self.current_claims.clear();
        self.active_players.clear();
        for seat in (0..NP as u8).filter(|&seat| seat != discarder) {
            let (actions, no_yaku_win_shape) =
                self._get_claim_actions_for_player(seat, discarder, tile);
            if no_yaku_win_shape {
                // This is not a declined Ron: no Ron was offered in this window.
                self.players[seat as usize].missed_agari_doujun = true;
            }
            if !actions.is_empty() {
                self.active_players.push(seat);
                self.current_claims.insert(seat, actions);
            }
        }
    }

    pub fn _resolve_kan(&mut self, pid: u8, action: Action) {
        let p_idx = pid as usize;
        // A caller may resume a real response window opened by apply_event.
        // Its declaration is already in rule_log; the legacy completion log
        // must acknowledge it rather than emit a second declaration.
        if self
            .replay_context
            .kan_window
            .as_ref()
            .is_some_and(|window| window.applied && window.actor == pid)
        {
            if action.action_type == ActionType::Ankan {
                self.rule_pending_ankan = Some(pid);
            }
            self.replay_context.kan_window = None;
            self.replay_context.replacement_actor = None;
        }
        if action.action_type == ActionType::Kakan {
            let tile = action.tile.expect("complete kakan declaration");
            self.commit_kakan_tiles(pid, tile);
            self._push_rule_event(serde_json::json!({
                "type": "kan_commit", "actor": pid, "kan_type": "kakan",
            }));
        } else {
            // Ankan / Daiminkan
            for &t in &action.consume_tiles {
                if let Some(idx) = self.players[p_idx].hand.iter().position(|&x| x == t) {
                    self.players[p_idx].hand.remove(idx);
                }
            }
            let (m_type, tiles, from_who, ct) = if action.action_type == ActionType::Ankan {
                (MeldType::Ankan, action.consume_tiles.clone(), -1i8, None)
            } else {
                let (discarder, tile) = self.last_discard.expect("last_discard must be set");
                let mut t_vec = action.consume_tiles.clone();
                t_vec.push(tile);
                t_vec.sort();
                (MeldType::Daiminkan, t_vec, discarder as i8, Some(tile))
            };
            self.players[p_idx].melds.push(Meld {
                meld_type: m_type,
                tiles,
                opened: m_type == MeldType::Daiminkan,
                from_who,
                called_tile: ct,
            });

            // PAO check for Daiminkan
            if action.action_type == ActionType::Daiminkan {
                let (discarder, tile) = self.last_discard.expect("last_discard must be set");
                let tile_val = tile / 4;
                if (31..=33).contains(&tile_val) {
                    let dragon_melds = self.players[p_idx]
                        .melds
                        .iter()
                        .filter(|m| {
                            let t = m.tiles[0] / 4;
                            (31..=33).contains(&t) && (m.meld_type != MeldType::Chi)
                        })
                        .count();
                    if dragon_melds == 3 {
                        self.players[p_idx].pao.insert(37, discarder);
                    }
                } else if (27..=30).contains(&tile_val) {
                    let wind_melds = self.players[p_idx]
                        .melds
                        .iter()
                        .filter(|m| {
                            let t = m.tiles[0] / 4;
                            (27..=30).contains(&t) && (m.meld_type != MeldType::Chi)
                        })
                        .count();
                    if wind_melds == 4 {
                        self.players[p_idx].pao.insert(50, discarder);
                    }
                }
            }
        }

        self.is_first_turn = false;
        for p in &mut self.players {
            p.ippatsu_cycle = false;
        }

        if self.wall.drawable_count > 0 {
            // Rinshan tiles are at the beginning of the wall vector (0-3)
            let t = self.wall.tiles.remove(0);
            self.wall.drawable_count -= 1;
            self.players[p_idx].hand.push(t);
            self.drawn_tile = Some(t);
            self.needs_tsumo = false;
            self.wall.rinshan_draw_count += 1;
            self.is_rinshan_flag = true;

            if !self.skip_mjai_logging {
                let m_type = match action.action_type {
                    ActionType::Ankan => Some("ankan"),
                    ActionType::Daiminkan => Some("daiminkan"),
                    ActionType::Kakan => None, // Logged in step()
                    _ => None,
                };
                if let Some(s) = m_type {
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String(s.to_string()));
                    ev.insert("actor".to_string(), Value::Number(pid.into()));
                    if action.action_type == ActionType::Ankan {
                        let tile = action.tile.unwrap_or_else(|| action.consume_tiles[0]);
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                    } else if action.action_type == ActionType::Daiminkan
                        && let Some((target, tile)) = self.last_discard
                    {
                        ev.insert("target".to_string(), Value::Number(target.into()));
                        ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
                    }
                    let cons_strs: Vec<String> = action
                        .consume_tiles
                        .iter()
                        .map(|&t| tid_to_mjai(t))
                        .collect();
                    ev.insert(
                        "consumed".to_string(),
                        serde_json::to_value(cons_strs).expect("valid JSON"),
                    );
                    self._push_mjai_event(Value::Object(ev));
                }
            }

            // Reveal any pending doras from previous kans
            while self.wall.pending_kan_dora_count > 0 {
                self.wall.pending_kan_dora_count -= 1;
                self._reveal_kan_dora();
            }

            // Ankan: always reveal dora immediately (before rinshan tsumo)
            // Daiminkan/Kakan: defer dora reveal until discard (dora event before dahai)
            if action.action_type == ActionType::Ankan {
                self._reveal_kan_dora();
            } else {
                self.wall.pending_kan_dora_count += 1;
            }

            if !self.skip_mjai_logging {
                // Rinshan tsumo logging should apply to Kakan as well
                let mut t_ev = serde_json::Map::new();
                t_ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                t_ev.insert("actor".to_string(), Value::Number(pid.into()));
                t_ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(t_ev));
            }
            self.phase = Phase::WaitAct;
            self.active_players = vec![pid];
        }
    }

    fn _accept_riichi(&mut self) {
        if let Some(p) = self.riichi_pending_acceptance {
            self.players[p as usize].score -= 1000;
            self.players[p as usize].score_delta -= 1000;
            self.riichi_sticks += 1;
            self.players[p as usize].riichi_declared = true;
            self.players[p as usize].ippatsu_cycle = true;
            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert(
                    "type".to_string(),
                    Value::String("reach_accepted".to_string()),
                );
                ev.insert("actor".to_string(), Value::Number(p.into()));
                self._push_mjai_event(Value::Object(ev));
            }
            self.riichi_pending_acceptance = None;
        }
    }

    pub fn _deal_next(&mut self) {
        self.is_rinshan_flag = false;
        if self.wall.drawable_count == 0 {
            self._trigger_ryukyoku("exhaustive_draw");
            return;
        }
        if let Some(t) = self.wall.tiles.pop() {
            self.wall.drawable_count -= 1;
            let pid = self.current_player;
            self.players[pid as usize].hand.push(t);
            self.drawn_tile = Some(t);
            self.needs_tsumo = false;
            self.phase = Phase::WaitAct;
            self.active_players = vec![pid];

            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                ev.insert("actor".to_string(), Value::Number(pid.into()));
                ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(ev));
            }
            self.players[pid as usize].forbidden_discards.clear();
        }
    }

    pub fn _initialize_next_round(&mut self, is_renchan: bool, is_draw: bool) {
        self._advance_round(is_renchan, is_draw, false);
    }

    fn _advance_round(&mut self, is_renchan: bool, is_draw: bool, is_abortive_draw: bool) {
        if self.is_done {
            return;
        }

        let end = if is_abortive_draw {
            RoundEnd::AbortiveDraw
        } else if is_draw {
            RoundEnd::ExhaustiveDraw {
                dealer_tenpai: is_renchan,
            }
        } else {
            RoundEnd::Win {
                dealer_won: is_renchan,
                winner_count: self.win_results.len().max(1) as u8,
            }
        };
        let transition = settlement::round_transition(
            RoundContext {
                scores: self.players.each_ref().map(|player| player.score),
                dealer: self.oya,
                round_wind: self.round_wind,
                honba: self.honba as u32,
                game_mode: self.game_mode,
                initial_seat_order: [0, 1, 2, 3],
            },
            end,
            &self.rule,
        )
        .expect("validated online round transition");
        if transition.match_end {
            self._process_end_game();
            return;
        }
        let next_oya = transition.dealer;
        let next_round_wind = transition.round_wind;
        let next_honba = transition.honba.min(u8::MAX as u32) as u8;

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_kyoku".to_string()));
            self._push_mjai_event(Value::Object(ev));
        }

        let next_scores: Vec<i32> = self.players.iter().map(|p| p.score).collect();
        let next_sticks = self.riichi_sticks;
        self._initialize_round(
            next_oya,
            next_round_wind,
            next_honba,
            next_sticks,
            None,
            Some(next_scores),
        );
    }

    /// Initialize (or re-initialize) a single round.
    ///
    /// Used both at the start of a new game and when advancing to the next
    /// round within a game.  When `scores` is `None`, player scores are
    /// carried over from the previous round (normal round progression).
    pub fn _initialize_round(
        &mut self,
        oya: u8,
        round_wind: u8,
        honba: u8,
        kyotaku: u32,
        wall: Option<Vec<u8>>,
        scores: Option<Vec<i32>>,
    ) {
        self.replay_context = Default::default();
        let np = NP;
        self.oya = oya;
        self.kyoku_idx = oya;
        self.current_player = oya;
        self.honba = honba;
        self.riichi_sticks = kyotaku;
        self.round_wind = round_wind;

        for p in &mut self.players {
            p.reset_round();
        }
        self.is_done = false;
        self.current_claims = HashMap::new();
        self.pending_kan = None;
        self.is_rinshan_flag = false;
        self.wall.rinshan_draw_count = 0;
        self.wall.pending_kan_dora_count = 0;
        self.is_first_turn = true;
        self.riichi_pending_acceptance = None;
        self.turn_count = 0;
        self.needs_tsumo = true;
        self.needs_initialize_next_round = false;
        self.pending_oya_won = false;
        self.pending_is_draw = false;
        self.last_discard = None;
        self.win_results.clear();
        self.last_win_results.clear();
        self.round_end_scores = None;
        self.riichi_sutehais = [None; NP];
        self.last_tedashis = [None; NP];

        if let Some(s) = scores {
            for (i, &sc) in s.iter().enumerate() {
                if i < self.players.len() {
                    self.players[i].score = sc;
                }
            }
        }

        if let Some(w) = wall {
            self.wall.load_wall(w);
        } else {
            self.wall.shuffle();
        }

        // Deal logic
        for _ in 0..3 {
            for idx in 0..np {
                let p = (idx + oya as usize) % np;
                for _ in 0..4 {
                    if let Some(t) = self.wall.tiles.pop() {
                        self.players[p].hand.push(t);
                    }
                }
            }
        }
        for idx in 0..np {
            let p = (idx + oya as usize) % np;
            if let Some(t) = self.wall.tiles.pop() {
                self.players[p].hand.push(t);
            }
        }
        for p in &mut self.players {
            p.hand.sort();
        }

        // 14 = dead wall tiles (2 rinshan stacks + 5 dora stacks)
        self.wall.drawable_count = (self.wall.tiles.len() as u8) - 14;

        // Record kyoku start offsets so get_observation() can return
        // the full round history (not just incremental deltas).
        for i in 0..NP {
            self.kyoku_start_event_counts[i] = self.mjai_log_per_player[i].len();
        }
        // Reset pre-computed progression cache for the new round.
        #[cfg(feature = "python")]
        {
            Arc::make_mut(&mut self.round_seq_progression).clear();
            self.round_seq_prog_pending_reach = None;
        }

        if !self.skip_mjai_logging {
            let wind_str = match round_wind % 4 {
                0 => "E",
                1 => "S",
                2 => "W",
                _ => "N",
            };
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("start_kyoku".to_string()));
            ev.insert("bakaze".to_string(), Value::String(wind_str.to_string()));
            ev.insert("kyoku".to_string(), Value::Number((oya + 1).into()));
            ev.insert("honba".to_string(), Value::Number(honba.into()));
            ev.insert("kyotaku".to_string(), Value::Number(kyotaku.into()));
            ev.insert("oya".to_string(), Value::Number(oya.into()));
            let scores_vec: Vec<i32> = self.players.iter().map(|p| p.score).collect();
            ev.insert(
                "scores".to_string(),
                serde_json::to_value(scores_vec).expect("valid JSON"),
            );
            ev.insert(
                "dora_marker".to_string(),
                Value::String(tid_to_mjai(self.wall.dora_indicators[0])),
            );

            let mut tehais = Vec::new();
            for p in &self.players {
                let hand_strs: Vec<String> = p.hand.iter().map(|&t| tid_to_mjai(t)).collect();
                tehais.push(hand_strs);
            }
            ev.insert(
                "tehais".to_string(),
                serde_json::to_value(tehais).expect("valid JSON"),
            );

            self._push_mjai_event(Value::Object(ev));
        }

        self.current_player = self.oya;
        self.phase = Phase::WaitAct;
        self.active_players = vec![self.oya];

        // Draw 14th tile for Oya
        if let Some(t) = self.wall.tiles.pop() {
            self.wall.drawable_count -= 1;
            self.players[self.oya as usize].hand.push(t);
            self.drawn_tile = Some(t);
            self.needs_tsumo = false;

            if !self.skip_mjai_logging {
                let mut ev = serde_json::Map::new();
                ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                ev.insert("actor".to_string(), Value::Number(self.oya.into()));
                ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(ev));
            }
        } else {
            self.needs_tsumo = true;
            self.drawn_tile = None;
        }
    }

    pub fn _trigger_ryukyoku(&mut self, reason: &str) {
        if reason == "sanchaho" {
            // Ron on the declaration discard prevents the riichi deposit,
            // including when three ron claims end in an abortive draw.
            self.riichi_pending_acceptance = None;
        } else {
            self._accept_riichi();
        }

        let np = NP;
        let mut tenpai = vec![false; np];
        let mut final_reason = reason.to_string();
        let mut nagashi_winners = Vec::new();

        if reason == "exhaustive_draw" {
            // Riichi deposits were already recorded by reach_accepted.
            // Track only this draw's noten or nagashi-mangan payments.
            for player in &mut self.players {
                player.score_delta = 0;
            }
            for (i, p) in self.players.iter().enumerate() {
                let calc =
                    crate::hand_evaluator::HandEvaluator::new(p.hand.clone(), p.melds.clone());
                if calc.is_tenpai_with_rule(&self.rule) {
                    tenpai[i] = true;
                }
            }
            for (i, p) in self.players.iter().enumerate() {
                if p.nagashi_eligible {
                    nagashi_winners.push(i as u8);
                }
            }

            if !nagashi_winners.is_empty() {
                final_reason = "nagashimangan".to_string();
            }
            let deltas = settlement::draw_payments(
                [tenpai[0], tenpai[1], tenpai[2], tenpai[3]],
                self.oya,
                &nagashi_winners,
                false,
            )
            .expect("validated online draw settlement");
            for (player, delta) in self.players.iter_mut().zip(deltas) {
                player.score += delta;
                player.score_delta = delta;
            }
        } else if let Some(stripped) = reason.strip_prefix("Error: Illegal Action by Player ")
            && let Ok(pid) = stripped.parse::<usize>()
            && pid < np
        {
            let is_offender_oya = (pid as u8) == self.oya;
            if is_offender_oya {
                let penalty = 4000 * (np as i32 - 1);
                let each_get = penalty / (np as i32 - 1);
                for i in 0..np {
                    if i == pid {
                        self.players[i].score -= penalty;
                        self.players[i].score_delta = -penalty;
                    } else {
                        self.players[i].score += each_get;
                        self.players[i].score_delta = each_get;
                    }
                }
            } else {
                let total_penalty = 4000 + 2000 * (np as i32 - 2);
                for i in 0..np {
                    if i == pid {
                        self.players[i].score -= total_penalty;
                        self.players[i].score_delta = -total_penalty;
                    } else if (i as u8) == self.oya {
                        self.players[i].score += 4000;
                        self.players[i].score_delta = 4000;
                    } else {
                        self.players[i].score += 2000;
                        self.players[i].score_delta = 2000;
                    }
                }
            }
        }

        // Nagashi mangan replaces the payments, but renchan still depends on
        // dealer tenpai as in any other exhaustive draw.
        let is_renchan = if reason == "exhaustive_draw" {
            tenpai[self.oya as usize]
        } else {
            true
        };

        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("ryukyoku".to_string()));
            ev.insert("reason".to_string(), Value::String(final_reason.clone()));
            // Riichi deposits were already recorded by reach_accepted.
            // An abortive draw itself transfers no points.
            let deltas: Vec<i32> = if matches!(
                reason,
                "kyushu_kyuhai" | "sufuurenta" | "suukansansen" | "suucha_riichi" | "sanchaho"
            ) {
                vec![0; NP]
            } else {
                self.players.iter().map(|p| p.score_delta).collect()
            };
            ev.insert(
                "deltas".to_string(),
                serde_json::to_value(deltas).expect("valid JSON"),
            );
            let public_hands: Vec<Option<Vec<String>>> = self
                .players
                .iter()
                .enumerate()
                .map(|(seat, player)| {
                    (tenpai[seat]
                        || (reason == "kyushu_kyuhai" && seat == self.current_player as usize))
                        .then(|| player.hand.iter().map(|&tile| tid_to_mjai(tile)).collect())
                })
                .collect();
            if reason == "kyushu_kyuhai" {
                ev.insert(
                    "actor".to_string(),
                    Value::Number(self.current_player.into()),
                );
            }
            ev.insert(
                "public_hands".to_string(),
                serde_json::to_value(public_hands).expect("valid JSON"),
            );
            ev.insert(
                "scores".to_string(),
                serde_json::to_value(self.players.each_ref().map(|p| p.score)).expect("valid JSON"),
            );
            ev.insert("visibility_complete".to_string(), Value::Bool(true));
            ev.insert(
                "public_reveal_provenance".to_string(),
                Value::String("native_round_resolution_v1".to_string()),
            );
            self._push_native_resolution(Value::Object(ev));
        }

        self._advance_round(is_renchan, true, reason != "exhaustive_draw");
    }

    fn check_abortive_draw(&mut self) -> bool {
        if let Some(reason) = self.abortive_draw_reason() {
            self._trigger_ryukyoku(reason);
            true
        } else {
            false
        }
    }

    fn kan_counts(&self) -> [usize; NP] {
        self.players.each_ref().map(|p| {
            p.melds
                .iter()
                .filter(|m| {
                    matches!(
                        m.meld_type,
                        MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan
                    )
                })
                .count()
        })
    }

    fn abortive_draw_reason(&self) -> Option<&'static str> {
        // Shared with the public-information response query, in rule priority order.
        if crate::public_rules::four_winds_abort(
            self.players.each_ref().map(|p| p.discards.len()),
            self.players.each_ref().map(|p| p.discards.first().copied()),
            self.players.each_ref().map(|p| p.melds.len()),
        ) {
            return Some("sufuurenta");
        }
        if crate::public_rules::four_kans_abort(self.kan_counts()) {
            return Some("suukansansen");
        }
        if crate::public_rules::four_riichi_abort(
            self.players.each_ref().map(|p| p.riichi_declared),
        ) {
            return Some("suucha_riichi");
        }
        None
    }

    pub fn _reveal_kan_dora(&mut self) {
        let count = self.wall.dora_indicators.len();
        if count < 5 {
            // Base indices for Omote Dora are 4, 6, 8, 10, 12 in the wall
            // Since we use remove(0) for Rinshan draws, the indices shift down.
            let base_idx = (4 + 2 * count).saturating_sub(self.wall.rinshan_draw_count as usize);
            if base_idx < self.wall.tiles.len() {
                self.wall.dora_indicators.push(self.wall.tiles[base_idx]);
                if !self.skip_mjai_logging {
                    let mut ev = serde_json::Map::new();
                    ev.insert("type".to_string(), Value::String("dora".to_string()));
                    ev.insert(
                        "dora_marker".to_string(),
                        Value::String(tid_to_mjai(
                            self.wall
                                .dora_indicators
                                .last()
                                .copied()
                                .expect("dora_indicators must not be empty"),
                        )),
                    );
                    self._push_mjai_event(Value::Object(ev));
                }
            }
        }
    }

    fn _get_ura_indicators(&self) -> Vec<u8> {
        let mut indicators = Vec::new();
        for i in 0..self.wall.dora_indicators.len() {
            let idx = (5 + 2 * i).saturating_sub(self.wall.rinshan_draw_count as usize);
            if idx < self.wall.tiles.len() {
                indicators.push(self.wall.tiles[idx]);
            }
        }
        indicators
    }

    pub fn _get_ura_markers(&self) -> Vec<String> {
        let mut markers = Vec::new();
        for i in 0..self.wall.dora_indicators.len() {
            let idx = (5 + 2 * i).saturating_sub(self.wall.rinshan_draw_count as usize); // Ura is next to front.
            // Original: `self.wall[5 + 2*i]`
            if idx < self.wall.tiles.len() {
                markers.push(tid_to_mjai(self.wall.tiles[idx]));
            }
        }
        markers
    }

    pub(crate) fn _process_end_game(&mut self) {
        // Single-round mode exposes hand settlement, including any unclaimed
        // pot. A completed match awards that pot to the top-ranked seat.
        let mut end_game_pot_deltas = [0; NP];
        if !matches!(self.game_mode, 0 | 3) && self.riichi_sticks > 0 {
            end_game_pot_deltas = settlement::end_game_pot(
                self.players.each_ref().map(|player| player.score),
                self.riichi_sticks,
                [0, 1, 2, 3],
            )
            .expect("validated online final pot");
            for (player, delta) in self.players.iter_mut().zip(end_game_pot_deltas) {
                player.score += delta;
                player.score_delta += delta;
            }
            self.riichi_sticks = 0;
        }
        self.is_done = true;
        if !self.skip_mjai_logging {
            let mut ek = serde_json::Map::new();
            ek.insert("type".to_string(), Value::String("end_kyoku".to_string()));
            self._push_mjai_event(Value::Object(ek));

            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("end_game".to_string()));
            ev.insert(
                "scores".to_string(),
                serde_json::to_value(self.players.each_ref().map(|p| p.score)).expect("valid JSON"),
            );
            ev.insert(
                "end_game_pot_deltas".to_string(),
                serde_json::to_value(end_game_pot_deltas).expect("valid JSON"),
            );
            self._push_native_resolution(Value::Object(ev));
        }
    }

    pub fn apply_mjai_event(&mut self, event: MjaiEvent) {
        <Self as GameStateEventHandler>::apply_mjai_event(self, event)
    }

    pub fn apply_log_action(&mut self, action: &LogAction) {
        <Self as GameStateEventHandler>::apply_log_action(self, action)
    }
}

impl GameState {
    fn _push_rule_event(&mut self, event: Value) {
        if !self.skip_mjai_logging {
            self.rule_log
                .push(serde_json::to_string(&event).expect("valid engine event"));
        }
    }

    pub fn _push_mjai_event(&mut self, event: Value) {
        // Replay callers own their event schema, including any supplied
        // terminal scores/reveals.  Never infer origin from payload fields.
        self._push_logged_event(event, false);
    }

    fn _push_native_resolution(&mut self, event: Value) {
        self._push_logged_event(event, true);
    }

    fn _push_logged_event(&mut self, event: Value, native_resolution: bool) {
        if self.skip_mjai_logging {
            return;
        }
        let type_str = event["type"].as_str().unwrap_or("").to_string();
        let actor = event["actor"].as_u64().map(|a| a as usize);

        // Keep legacy MJAI byte/event timing compatible while exposing complete
        // transactions to KAR. There is exactly one declaration and one commit.
        if type_str == "ankan" && self.rule_pending_ankan.map(usize::from) == actor {
            self._push_rule_event(serde_json::json!({
                "type": "kan_commit", "actor": actor, "kan_type": "ankan",
            }));
            self.rule_pending_ankan = None;
        } else {
            if type_str == "start_game" && self.rule_log.len() == 1 {
                self.rule_log.clear(); // reset() inserted the generic header.
            }
            self._push_rule_event(event.clone());
        }
        if matches!(
            type_str.as_str(),
            "start_kyoku" | "hora" | "ryukyoku" | "end_game"
        ) {
            self.rule_pending_ankan = None;
        }

        // Terminal reveals and independently computed endpoint facts extend
        // rule_log only.  Preserve the legacy MJAI event contract, including
        // start_kyoku scores and the existing hora payment/ura fields.
        let mut event = event;
        let canonical_only: &[&str] = match type_str.as_str() {
            "hora" => &[
                "public_hand",
                "scores",
                "visibility_complete",
                "public_reveal_provenance",
            ],
            "ryukyoku" => &[
                "public_hands",
                "scores",
                "visibility_complete",
                "public_reveal_provenance",
            ],
            "end_game" => &["scores", "end_game_pot_deltas"],
            _ => &[],
        };
        if native_resolution && let Some(fields) = event.as_object_mut() {
            for &name in canonical_only {
                fields.remove(name);
            }
            if type_str == "ryukyoku"
                && fields.get("reason").and_then(Value::as_str) == Some("kyushu_kyuhai")
            {
                fields.remove("actor");
            }
        }
        let json_str = serde_json::to_string(&event).expect("valid JSON");
        self.mjai_log.push(json_str.clone());

        let np = NP;
        for pid in 0..np {
            let should_push = true;
            let mut final_json = json_str.clone();

            if type_str == "start_kyoku" {
                if let Some(tehais_val) = event.get("tehais").and_then(|v| v.as_array()) {
                    let mut masked_tehais = Vec::new();
                    for (i, hand_val) in tehais_val.iter().enumerate() {
                        if i == pid {
                            masked_tehais.push(hand_val.clone());
                        } else {
                            let len = hand_val
                                .as_array()
                                .map(|a| a.len())
                                .unwrap_or(INITIAL_HAND_SIZE);
                            let masked = vec!["?".to_string(); len];
                            masked_tehais.push(serde_json::to_value(masked).expect("valid JSON"));
                        }
                    }
                    let mut masked_event = event
                        .as_object()
                        .expect("event must be a JSON object")
                        .clone();
                    masked_event.insert("tehais".to_string(), Value::Array(masked_tehais));
                    final_json =
                        serde_json::to_string(&Value::Object(masked_event)).expect("valid JSON");
                }
            } else if type_str == "tsumo"
                && let Some(act_id) = actor
                && act_id != pid
            {
                let mut masked_event = event
                    .as_object()
                    .expect("event must be a JSON object")
                    .clone();
                masked_event.insert("pai".to_string(), Value::String("?".to_string()));
                final_json =
                    serde_json::to_string(&Value::Object(masked_event)).expect("valid JSON");
            }

            if should_push {
                self.mjai_log_per_player[pid].push(final_json);
            }
        }

        // Incrementally update pre-computed progression cache.
        // Uses the original (unmasked) event Value directly — no JSON parsing.
        #[cfg(feature = "python")]
        if self.enable_seq_caching
            && let Some(entry) =
                crate::observation::sequence_features::process_single_event_progression(
                    &event,
                    &mut self.round_seq_prog_pending_reach,
                )
        {
            Arc::make_mut(&mut self.round_seq_progression).push(entry);
        }
    }
}
