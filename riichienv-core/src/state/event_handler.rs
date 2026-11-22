use crate::action::{ActionType, Phase};
use crate::replay::{Action as LogAction, MjaiEvent};
use crate::replay_context::same_mjai_tile;
use crate::state::GameState;
use crate::types::{INITIAL_HAND_SIZE, Meld, MeldType, TILES_4P, Wind};

fn remove_hand_tile_matching(hand: &mut Vec<u8>, tile: u8) -> Option<u8> {
    if let Some(idx) = hand.iter().position(|&x| x == tile) {
        return Some(hand.remove(idx));
    }
    if let Some(idx) = hand.iter().position(|&x| same_mjai_tile(x, tile)) {
        return Some(hand.remove(idx));
    }
    None
}

pub trait GameStateEventHandler {
    fn apply_mjai_event(&mut self, event: MjaiEvent);
    fn apply_log_action(&mut self, action: &LogAction);
}

impl GameState {
    fn record_mjai_discard_ron_declines(&mut self) {
        // These public successors are only possible once the discard's Ron
        // offers have all gone unused. This does not infer a chosen Pass action.
        // The successor moves out of WaitResponse; retained raw claims are not
        // treated as a second window (and the kan preview clears its clone).
        if self.phase == Phase::WaitResponse {
            for (&seat, actions) in &self.current_claims {
                if actions
                    .iter()
                    .any(|action| action.action_type == ActionType::Ron)
                {
                    self.players[seat as usize].record_declined_ron();
                }
            }
        }
    }
}

impl GameStateEventHandler for GameState {
    fn apply_mjai_event(&mut self, event: MjaiEvent) {
        if let Err(error) = self.advance_mjai_context(&event) {
            self.last_error = Some(error);
            return;
        }
        self.last_error = None;
        if matches!(
            &event,
            MjaiEvent::Tsumo { .. }
                | MjaiEvent::Pon { .. }
                | MjaiEvent::Chi { .. }
                | MjaiEvent::Kan { .. }
        ) {
            self.record_mjai_discard_ron_declines();
            if self.phase == Phase::WaitResponse {
                let winning = match &event {
                    MjaiEvent::Pon { actor, .. } => Some((*actor as u8, ActionType::Pon)),
                    MjaiEvent::Chi { actor, .. } => Some((*actor as u8, ActionType::Chi)),
                    MjaiEvent::Kan { actor, .. } => Some((*actor as u8, ActionType::Daiminkan)),
                    _ => None,
                };
                let retired = std::mem::take(&mut self.current_claims);
                self.v2_record_passes_from_claims(&retired, winning);
            }
        }
        match event {
            MjaiEvent::StartGame { .. } => {
                // Clear stale state from constructor's reset() so that
                // get_observation() does not return stale legal actions.
                self.current_player = u8::MAX;
                self.active_players.clear();
                self.v2_round_events = crate::v2_features::events::RoundEventLog::default();
                self.v2_hanchan_tracker = crate::v2_features::HanchanTracker::default();
                self.v2_round_start_scores = [self.mode.starting_score(); 4];
                self.v2_round_won = [false; 4];
                self.v2_round_dealt_in = [false; 4];
                self.v2_round_settled = true;
                self.v2_discard_seq = 0;
                self.v2_replay_end_scores = None;
                self.v2_replay_accum_deltas = [0; 4];
                self.v2_replay_has_deltas = false;
            }
            MjaiEvent::StartKyoku {
                bakaze,
                kyoku,
                honba,
                kyoutaku,
                scores,
                dora_marker,
                tehais,
                oya,
                ..
            } => {
                let start_scores: [i32; 4] = [
                    scores.first().copied().unwrap_or(25_000),
                    scores.get(1).copied().unwrap_or(25_000),
                    scores.get(2).copied().unwrap_or(25_000),
                    scores.get(3).copied().unwrap_or(25_000),
                ];
                self.v2_record_round_end_if_needed(Some(start_scores));
                // Replay starts before the dealer's opening draw.
                // Keep a placeholder wall sized to that pre-tsumo state.
                self.honba = honba;
                self.riichi_sticks = kyoutaku as u32;
                self.round_wind = match bakaze.as_str() {
                    "E" => Wind::East as u8,
                    "S" => Wind::South as u8,
                    "W" => Wind::West as u8,
                    "N" => Wind::North as u8,
                    _ => Wind::East as u8,
                };
                self.oya = oya;
                self.kyoku_idx = kyoku.saturating_sub(1);
                self.current_player = u8::MAX; // No active player until first tsumo
                self.turn_count = 0;
                self.is_done = false;
                self.needs_tsumo = true;
                self.needs_initialize_next_round = false;
                self.pending_oya_won = false;
                self.pending_is_draw = false;
                self.phase = Phase::WaitAct;
                self.active_players.clear();
                self.last_discard = None;
                self.current_claims.clear();
                self.pending_kan = None;
                self.is_rinshan_flag = false;
                self.is_first_turn = true;
                self.riichi_pending_acceptance = None;
                self.drawn_tile = None;
                self.win_results.clear();
                self.last_win_results.clear();
                self.round_end_scores = None;
                self.last_error = None;
                self.is_after_kan = false;
                self.riichi_sutehais = [None; 4];
                self.last_tedashis = [None; 4];
                self.wall.tiles = vec![0; TILES_4P - INITIAL_HAND_SIZE * 4];
                self.wall.rinshan_draw_count = 0;
                self.wall.pending_kan_dora_count = 0;
                // Match _initialize_round: 14 = dead wall (rinshan + dora stacks).
                // Without this reset, drawable_count carries over from prior rounds
                // and can fall below riichi/kan thresholds (e.g. >= 4) on replay.
                self.wall.drawable_count = (self.wall.tiles.len() as u8) - 14;
                self.wall.wall_digest.clear();
                self.wall.salt.clear();

                for p in &mut self.players {
                    p.reset_round();
                }
                self.players.iter_mut().enumerate().for_each(|(i, p)| {
                    p.score = scores[i];
                });
                for (i, hand_strs) in tehais.iter().enumerate() {
                    let mut hand = Vec::with_capacity(hand_strs.len());
                    for tile_str in hand_strs {
                        hand.push(self.allocate_replay_tile(tile_str));
                    }
                    hand.sort();
                    self.players[i].hand = hand;
                }
                // Allocate the opening dora indicator after tehais so seat 0 keeps
                // canonical base IDs for first copies while the indicator stays disjoint.
                let dora_tid = self.allocate_replay_tile(&dora_marker);
                self.wall.dora_indicators = vec![dora_tid];
                self.v2_init_round_tracking(dora_tid, start_scores);
            }
            MjaiEvent::Tsumo { actor, pai } => {
                let is_known_tile = crate::parser::mjai_to_tid(&pai).is_some();
                let tile = self.allocate_replay_tile(&pai);
                self.current_player = actor as u8;
                self.drawn_tile = Some(tile);
                self.players[actor].hand.push(tile);
                self.players[actor].hand.sort();
                self.players[actor].forbidden_discards.clear();
                if !self.wall.tiles.is_empty() {
                    self.wall.tiles.pop();
                    self.wall.drawable_count = self.wall.drawable_count.saturating_sub(1);
                }
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = false;
                if is_known_tile {
                    self.v2_round_events.push(
                        crate::v2_features::events::RoundEvent::TsumoPrivate {
                            actor: actor as u8,
                            tile37: crate::v2_features::tile136_to_tile37(tile),
                            is_rinshan: self.is_rinshan_flag,
                            turn_idx: self.v2_discard_seq,
                        },
                    );
                }
            }
            MjaiEvent::Dahai {
                actor,
                pai,
                tsumogiri,
            } => {
                let tile = self.take_replay_hand_tile(actor, &pai, tsumogiri);
                let is_riichi = self.players[actor].riichi_stage;
                let is_double_riichi = is_riichi
                    && self.replay_context.pending_double[actor].unwrap_or(false);
                self.current_player = actor as u8;
                self.players[actor].discards.push(tile);
                self.players[actor].discard_from_hand.push(!tsumogiri);
                self.players[actor].discard_is_riichi.push(is_riichi);
                // Match live step / LogAction replay: only own public discards
                // and other players calling them revoke nagashi eligibility.
                self.players[actor].nagashi_eligible &= crate::types::is_terminal_tile(tile);
                self.last_discard = Some((actor as u8, tile));
                if !tsumogiri {
                    self.last_tedashis[actor] = Some(tile);
                }
                self.drawn_tile = None;
                self.turn_count += 1;
                if self.turn_count >= self.players.len() as u32 {
                    self.is_first_turn = false;
                }

                if is_riichi {
                    self.riichi_sutehais[actor] = Some(tile);
                    self.players[actor].riichi_declaration_index =
                        Some(self.players[actor].discards.len() - 1);
                    self.players[actor].riichi_declared = true;
                    self.players[actor].riichi_stage = false;
                }

                let river_idx = (self.players[actor].discards.len().saturating_sub(1)) as u8;
                self.v2_round_events
                    .push(crate::v2_features::events::RoundEvent::Dahai {
                        actor: actor as u8,
                        tile37: crate::v2_features::tile136_to_tile37(tile),
                        tsumogiri,
                        is_riichi_declare: is_riichi,
                        is_double_riichi,
                        turn_idx: self.v2_discard_seq,
                        river_idx,
                    });
                self.v2_discard_seq = self.v2_discard_seq.saturating_add(1);

                self.players[actor].ippatsu_cycle = false;
                self.is_rinshan_flag = false;
                self.populate_discard_claims(actor as u8, tile);
                if !self.active_players.is_empty() {
                    self.phase = Phase::WaitResponse;
                } else {
                    self.phase = Phase::WaitAct;
                    self.current_player = u8::MAX;
                }
                self.needs_tsumo = true;
            }
            MjaiEvent::Pon {
                actor,
                target,
                pai,
                consumed,
            } => {
                let tile = self.resolve_replay_called_tile(target, &pai);
                self.players[target].nagashi_eligible = false;
                self.current_player = actor as u8;
                let c1 = self.take_replay_hand_tile(actor, &consumed[0], false);
                let c2 = self.take_replay_hand_tile(actor, &consumed[1], false);
                let form_tiles = vec![tile, c1, c2];

                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Pon,
                    tiles: form_tiles,
                    opened: true,
                    from_who: target as i8,
                    called_tile: Some(tile),
                });
                let meld_idx = (self.players[actor].melds.len().saturating_sub(1)) as u8;
                self.v2_round_events
                    .push(crate::v2_features::events::RoundEvent::Pon {
                        actor: actor as u8,
                        source: target as u8,
                        claimed_tile37: crate::v2_features::tile136_to_tile37(tile),
                        consumed0_37: crate::v2_features::tile136_to_tile37(c1),
                        consumed1_37: crate::v2_features::tile136_to_tile37(c2),
                        turn_idx: self.v2_discard_seq,
                        meld_idx,
                    });
                self.record_replay_pao(actor, target, tile);
                self.drawn_tile = None;
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = false;
                // Kuikae (swap-calling) forbidden
                self.players[actor].forbidden_discards.clear();
                if self.rule.kuikae_forbidden {
                    self.players[actor].forbidden_discards.push(tile);
                }
            }
            MjaiEvent::Chi {
                actor,
                target,
                pai,
                consumed,
            } => {
                let tile = self.resolve_replay_called_tile(target, &pai);
                self.players[target].nagashi_eligible = false;
                self.current_player = actor as u8;
                let c1 = self.take_replay_hand_tile(actor, &consumed[0], false);
                let c2 = self.take_replay_hand_tile(actor, &consumed[1], false);
                let form_tiles = vec![tile, c1, c2];

                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Chi,
                    tiles: form_tiles,
                    opened: true,
                    from_who: target as i8,
                    called_tile: Some(tile),
                });
                let meld_idx = (self.players[actor].melds.len().saturating_sub(1)) as u8;
                self.v2_round_events
                    .push(crate::v2_features::events::RoundEvent::Chi {
                        actor: actor as u8,
                        source: target as u8,
                        claimed_tile37: crate::v2_features::tile136_to_tile37(tile),
                        consumed0_37: crate::v2_features::tile136_to_tile37(c1),
                        consumed1_37: crate::v2_features::tile136_to_tile37(c2),
                        turn_idx: self.v2_discard_seq,
                        meld_idx,
                    });
                self.drawn_tile = None;
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = false;
                // Kuikae forbidden for chi: mirror claim-resolution logic
                self.players[actor].forbidden_discards.clear();
                if self.rule.kuikae_forbidden {
                    self.players[actor].forbidden_discards.push(tile);
                    let t34 = tile / 4;
                    let mut consumed_34 = [c1 / 4, c2 / 4];
                    consumed_34.sort();
                    if consumed_34[0] == t34 + 1 && consumed_34[1] == t34 + 2 {
                        if t34 % 9 <= 5 {
                            self.players[actor].forbidden_discards.push((t34 + 3) * 4);
                        }
                    } else if t34 >= 2
                        && consumed_34[1] == t34 - 1
                        && consumed_34[0] == t34 - 2
                        && t34 % 9 >= 3
                    {
                        self.players[actor].forbidden_discards.push((t34 - 3) * 4);
                    }
                }
            }
            MjaiEvent::Kan {
                actor,
                target,
                pai,
                consumed,
            } => {
                let tile = self.resolve_replay_called_tile(target, &pai);
                self.players[target].nagashi_eligible = false;
                self.current_player = actor as u8;
                let mut tiles = vec![tile];
                for c in &consumed {
                    tiles.push(self.take_replay_hand_tile(actor, c, false));
                }

                self.players[actor].melds.push(Meld {
                    meld_type: MeldType::Daiminkan,
                    tiles,
                    opened: true,
                    from_who: target as i8,
                    called_tile: Some(tile),
                });
                let meld_idx = (self.players[actor].melds.len().saturating_sub(1)) as u8;
                self.v2_round_events.push(
                    crate::v2_features::events::RoundEvent::Daiminkan {
                        actor: actor as u8,
                        source: target as u8,
                        claimed_tile37: crate::v2_features::tile136_to_tile37(tile),
                        turn_idx: self.v2_discard_seq,
                        meld_idx,
                    },
                );
                self.record_replay_pao(actor, target, tile);
                self.sync_replay_pending_kan_dora();
                self.phase = Phase::WaitAct;
                self.active_players = vec![actor as u8];
                self.needs_tsumo = true;
            }
            MjaiEvent::Ankan { actor, .. } => {
                // The declaration opens the native response window. Concealed
                // tiles stay unchanged until public commit.
                self.open_replay_kan_window(actor as u8);
            }
            MjaiEvent::Kakan { actor, pai } => {
                let kakan_tile = self
                    .pending_kan
                    .as_ref()
                    .and_then(|(_, act)| act.tile)
                    .or_else(|| crate::parser::mjai_to_tid(&pai))
                    .unwrap_or(0);
                let meld_idx = self.players[actor]
                    .melds
                    .iter()
                    .position(|m| m.meld_type == MeldType::Pon && m.tiles[0] / 4 == kakan_tile / 4)
                    .unwrap_or(0) as u8;
                self.v2_round_events
                    .push(crate::v2_features::events::RoundEvent::Kakan {
                        actor: actor as u8,
                        tile37: crate::v2_features::tile136_to_tile37(kakan_tile),
                        turn_idx: self.v2_discard_seq,
                        meld_idx,
                    });
                // The declaration opens the native response window. Original
                // Pon stays unchanged until public commit.
                self.open_replay_kan_window(actor as u8);
            }
            MjaiEvent::Reach { actor } => {
                self.players[actor].riichi_stage = true;
                // Keep current_player and phase so that the agent can
                // select a riichi discard tile via legal_actions.
            }
            MjaiEvent::ReachAccepted { actor } => {
                self.players[actor].riichi_declared = true;
                self.riichi_sticks += 1;
                self.players[actor].score -= 1000;
                self.v2_round_events.push(
                    crate::v2_features::events::RoundEvent::ReachAccepted {
                        actor: actor as u8,
                        turn_idx: self.v2_discard_seq,
                    },
                );
            }
            MjaiEvent::Dora { dora_marker } => {
                let tile = self.allocate_replay_tile(&dora_marker);
                self.wall.dora_indicators.push(tile);
                self.v2_round_events
                    .push(crate::v2_features::events::RoundEvent::Dora {
                        indicator_tile37: crate::v2_features::tile136_to_tile37(tile),
                        dora_idx: (self.wall.dora_indicators.len().saturating_sub(1)) as u8,
                        turn_idx: self.v2_discard_seq,
                    });
                self.sync_replay_pending_kan_dora();
            }
            MjaiEvent::Kita { .. } => {
                // Kita is 3P only; ignored in 4P event handler
            }
            MjaiEvent::Hora {
                actor,
                target,
                scores,
                delta,
                ..
            } => {
                if actor < 4 {
                    self.v2_round_won[actor] = true;
                }
                if target < 4 && target != actor {
                    self.v2_round_dealt_in[target] = true;
                }
                if let Some(sc) = scores
                    && sc.len() == 4
                {
                    self.v2_replay_end_scores = Some([sc[0], sc[1], sc[2], sc[3]]);
                } else if let Some(d) = delta
                    && d.len() == 4
                {
                    for (acc, &val) in self.v2_replay_accum_deltas.iter_mut().zip(d.iter()) {
                        *acc += val;
                    }
                    self.v2_replay_has_deltas = true;
                }
                self.is_done = true;
            }
            MjaiEvent::Ryukyoku { scores, delta, .. } => {
                if let Some(sc) = scores
                    && sc.len() == 4
                {
                    self.v2_replay_end_scores = Some([sc[0], sc[1], sc[2], sc[3]]);
                } else if let Some(d) = delta
                    && d.len() == 4
                {
                    for (acc, &val) in self.v2_replay_accum_deltas.iter_mut().zip(d.iter()) {
                        *acc += val;
                    }
                    self.v2_replay_has_deltas = true;
                }
                self.is_done = true;
            }
            MjaiEvent::EndKyoku | MjaiEvent::EndGame => {
                self.v2_record_round_end_if_needed(None);
                self.is_done = true;
            }
            _ => {}
        }
    }

    fn apply_log_action(&mut self, action: &LogAction) {
        match action {
            LogAction::DiscardTile {
                seat,
                tile,
                is_liqi,
                is_wliqi,
                ..
            } => {
                let s = *seat;
                let t = *tile;
                let is_tsumogiri = self.drawn_tile == Some(t) && !self.is_forced_tedashi(s as u8);

                // Update progression cache (replay mode).
                #[cfg(feature = "python")]
                if self.enable_seq_caching {
                    use crate::observation::sequence_features::process_single_event_progression;
                    use crate::parser::tid_to_mjai;
                    use std::sync::Arc;

                    if *is_liqi || *is_wliqi {
                        let ev = serde_json::json!({"type": "reach", "actor": s});
                        if let Some(entry) = process_single_event_progression(
                            &ev,
                            &mut self.round_seq_prog_pending_reach,
                        ) {
                            Arc::make_mut(&mut self.round_seq_progression).push(entry);
                        }
                    }
                    let pai = tid_to_mjai(t);
                    let ev = serde_json::json!({
                        "type": "dahai",
                        "actor": s,
                        "pai": pai,
                        "tsumogiri": is_tsumogiri,
                    });
                    if let Some(entry) = process_single_event_progression(
                        &ev,
                        &mut self.round_seq_prog_pending_reach,
                    ) {
                        Arc::make_mut(&mut self.round_seq_progression).push(entry);
                    }
                }

                remove_hand_tile_matching(&mut self.players[s].hand, t);
                self.players[s].hand.sort();
                self.players[s].discards.push(t);
                self.players[s].discard_from_hand.push(!is_tsumogiri);
                self.players[s]
                    .discard_is_riichi
                    .push(*is_liqi || *is_wliqi);
                self.last_discard = Some((s as u8, t));
                if !is_tsumogiri {
                    self.last_tedashis[s] = Some(t);
                }
                if *is_liqi || *is_wliqi {
                    self.riichi_sutehais[s] = Some(t);
                }
                self.drawn_tile = None;
                // Reset same-turn furiten after own discard.
                self.players[s].missed_agari_doujun = false;
                // Track nagashi eligibility: discard must be terminal/honor
                self.players[s].nagashi_eligible &= crate::types::is_terminal_tile(t);

                if *is_liqi || *is_wliqi {
                    if !self.players[s].riichi_declared {
                        self.players[s].riichi_declared = true;
                        if *is_wliqi {
                            self.players[s].double_riichi_declared = true;
                        }
                        // Defer the 1000 deposit; it gets voided if this
                        // discard is ronned, otherwise finalized on the next
                        // DealTile / ChiPengGang.
                        self.riichi_pending_acceptance = Some(s as u8);
                    }
                    self.players[s].riichi_declaration_index =
                        Some(self.players[s].discards.len() - 1);
                }
                self.current_player = (s as u8 + 1) % 4;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.needs_tsumo = true;
                self.turn_count += 1;
                if self.turn_count >= self.players.len() as u32 {
                    self.is_first_turn = false;
                }
                self.is_after_kan = false;
                let river_idx = (self.players[s].discards.len().saturating_sub(1)) as u8;
                self.v2_round_events
                    .push(crate::v2_features::events::RoundEvent::Dahai {
                        actor: s as u8,
                        tile37: crate::v2_features::tile136_to_tile37(t),
                        tsumogiri: is_tsumogiri,
                        is_riichi_declare: *is_liqi || *is_wliqi,
                        is_double_riichi: *is_wliqi,
                        turn_idx: self.v2_discard_seq,
                        river_idx,
                    });
                self.v2_discard_seq = self.v2_discard_seq.saturating_add(1);
            }
            LogAction::DealTile { seat, tile, .. } => {
                // Finalize pending riichi deposit (discard was not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                    self.v2_round_events.push(
                        crate::v2_features::events::RoundEvent::ReachAccepted {
                            actor: rp,
                            turn_idx: self.v2_discard_seq,
                        },
                    );
                }
                self.players[*seat].hand.push(*tile);
                self.drawn_tile = Some(*tile);
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.is_rinshan_flag = self.is_after_kan && *seat == self.current_player as usize;
                self.needs_tsumo = false;
                self.is_after_kan = false;
                self.players[*seat].hand.sort();
                if !self.wall.tiles.is_empty() {
                    self.wall.tiles.pop();
                    self.wall.drawable_count = self.wall.drawable_count.saturating_sub(1);
                }
                self.v2_round_events.push(
                    crate::v2_features::events::RoundEvent::TsumoPrivate {
                        actor: *seat as u8,
                        tile37: crate::v2_features::tile136_to_tile37(*tile),
                        is_rinshan: self.is_rinshan_flag,
                        turn_idx: self.v2_discard_seq,
                    },
                );
            }
            LogAction::ChiPengGang {
                seat,
                meld_type,
                tiles,
                froms,
            } => {
                // Update progression cache (replay mode).
                #[cfg(feature = "python")]
                if self.enable_seq_caching {
                    use crate::observation::sequence_features::process_single_event_progression;
                    use crate::parser::tid_to_mjai;
                    use std::sync::Arc;

                    let mtype_str = match meld_type {
                        MeldType::Chi => "chi",
                        MeldType::Pon => "pon",
                        MeldType::Daiminkan => "daiminkan",
                        _ => "",
                    };
                    if !mtype_str.is_empty() {
                        let target = froms.iter().find(|&&f| f != *seat).copied().unwrap_or(0);
                        let called_tile: Option<u8> = tiles
                            .iter()
                            .zip(froms.iter())
                            .find(|&(_, &f)| f != *seat)
                            .map(|(&t, _)| t);
                        let consumed_tiles: Vec<u8> = tiles
                            .iter()
                            .zip(froms.iter())
                            .filter(|&(_, &f)| f == *seat)
                            .map(|(&t, _)| t)
                            .collect();
                        let pai_str = called_tile.map(tid_to_mjai).unwrap_or_default();
                        let consumed_strs: Vec<String> =
                            consumed_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
                        let ev = serde_json::json!({
                            "type": mtype_str,
                            "actor": *seat,
                            "target": target,
                            "pai": pai_str,
                            "consumed": consumed_strs,
                        });
                        if let Some(entry) = process_single_event_progression(
                            &ev,
                            &mut self.round_seq_prog_pending_reach,
                        ) {
                            Arc::make_mut(&mut self.round_seq_progression).push(entry);
                        }
                    }
                }

                // Finalize pending riichi deposit (discard was claimed, not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                    self.v2_round_events.push(
                        crate::v2_features::events::RoundEvent::ReachAccepted {
                            actor: rp,
                            turn_idx: self.v2_discard_seq,
                        },
                    );
                }
                // Discard was called -> discarder loses nagashi eligibility
                if let Some((discarder_pid, _)) = self.last_discard {
                    self.players[discarder_pid as usize].nagashi_eligible = false;
                }
                // Remove tiles from hand
                for (i, t) in tiles.iter().enumerate() {
                    if i < froms.len() && froms[i] == *seat {
                        remove_hand_tile_matching(&mut self.players[*seat].hand, *t);
                    }
                }
                self.players[*seat].hand.sort();

                let from_who = froms
                    .iter()
                    .find(|&&f| f != *seat)
                    .map(|&f| f as i8)
                    .unwrap_or(-1);
                let ct = tiles
                    .iter()
                    .zip(froms.iter())
                    .find(|&(_, &f)| f != *seat)
                    .map(|(&t, _)| t);
                let discarder = from_who.max(0) as u8;
                self.players[*seat].melds.push(Meld {
                    meld_type: *meld_type,
                    tiles: tiles.clone(),
                    opened: true,
                    from_who,
                    called_tile: ct,
                });
                let meld_idx = (self.players[*seat].melds.len().saturating_sub(1)) as u8;
                let claimed_t37 = ct
                    .map(crate::v2_features::tile136_to_tile37)
                    .unwrap_or(crate::v2_features::events::TILE37_NONE);
                let consumed_t37: Vec<u8> = tiles
                    .iter()
                    .zip(froms.iter())
                    .filter(|&(_, &f)| f == *seat)
                    .map(|(&t, _)| crate::v2_features::tile136_to_tile37(t))
                    .collect();
                let c0 = consumed_t37
                    .first()
                    .copied()
                    .unwrap_or(crate::v2_features::events::TILE37_NONE);
                let c1 = consumed_t37
                    .get(1)
                    .copied()
                    .unwrap_or(crate::v2_features::events::TILE37_NONE);
                match meld_type {
                    MeldType::Chi => {
                        self.v2_round_events
                            .push(crate::v2_features::events::RoundEvent::Chi {
                                actor: *seat as u8,
                                source: discarder,
                                claimed_tile37: claimed_t37,
                                consumed0_37: c0,
                                consumed1_37: c1,
                                turn_idx: self.v2_discard_seq,
                                meld_idx,
                            });
                    }
                    MeldType::Pon => {
                        self.v2_round_events
                            .push(crate::v2_features::events::RoundEvent::Pon {
                                actor: *seat as u8,
                                source: discarder,
                                claimed_tile37: claimed_t37,
                                consumed0_37: c0,
                                consumed1_37: c1,
                                turn_idx: self.v2_discard_seq,
                                meld_idx,
                            });
                    }
                    MeldType::Daiminkan => {
                        self.v2_round_events.push(
                            crate::v2_features::events::RoundEvent::Daiminkan {
                                actor: *seat as u8,
                                source: discarder,
                                claimed_tile37: claimed_t37,
                                turn_idx: self.v2_discard_seq,
                                meld_idx,
                            },
                        );
                    }
                    _ => {}
                }

                // PAO detection: daisangen (3 dragon melds) or daisuushii (4 wind melds)
                if (*meld_type == MeldType::Pon || *meld_type == MeldType::Daiminkan)
                    && let Some(&called) = ct.as_ref()
                {
                    let tile_val = called / 4;
                    if (31..=33).contains(&tile_val) {
                        let dragon_melds = self.players[*seat]
                            .melds
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (31..=33).contains(&t) && m.meld_type != MeldType::Chi
                            })
                            .count();
                        if dragon_melds == 3 {
                            self.players[*seat].pao.insert(37, discarder);
                        }
                    } else if (27..=30).contains(&tile_val) {
                        let wind_melds = self.players[*seat]
                            .melds
                            .iter()
                            .filter(|m| {
                                let t = m.tiles[0] / 4;
                                (27..=30).contains(&t) && m.meld_type != MeldType::Chi
                            })
                            .count();
                        if wind_melds == 4 {
                            self.players[*seat].pao.insert(50, discarder);
                        }
                    }
                }

                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                let is_gang = *meld_type == MeldType::Daiminkan;
                self.needs_tsumo = is_gang;
                self.is_first_turn = false;
                self.is_after_kan = is_gang;
            }
            LogAction::AnGangAddGang {
                seat,
                meld_type,
                tiles,
                ..
            } => {
                // Update progression cache (replay mode).
                #[cfg(feature = "python")]
                if self.enable_seq_caching {
                    use crate::observation::sequence_features::process_single_event_progression;
                    use crate::parser::tid_to_mjai;
                    use std::sync::Arc;

                    let ev = if *meld_type == MeldType::Ankan {
                        let t_val = tiles[0] / 4;
                        let consumed_tids =
                            [t_val * 4, t_val * 4 + 1, t_val * 4 + 2, t_val * 4 + 3];
                        let consumed_strs: Vec<String> =
                            consumed_tids.iter().map(|&t| tid_to_mjai(t)).collect();
                        serde_json::json!({
                            "type": "ankan",
                            "actor": *seat,
                            "consumed": consumed_strs,
                        })
                    } else {
                        // Kakan
                        let pai_str = tid_to_mjai(tiles[0]);
                        serde_json::json!({
                            "type": "kakan",
                            "actor": *seat,
                            "pai": pai_str,
                        })
                    };
                    if let Some(entry) = process_single_event_progression(
                        &ev,
                        &mut self.round_seq_prog_pending_reach,
                    ) {
                        Arc::make_mut(&mut self.round_seq_progression).push(entry);
                    }
                }

                if *meld_type == MeldType::Ankan {
                    let t_val = tiles[0] / 4;
                    for _ in 0..4 {
                        if let Some(idx) = self.players[*seat]
                            .hand
                            .iter()
                            .position(|&x| x / 4 == t_val)
                        {
                            self.players[*seat].hand.remove(idx);
                        }
                    }
                    let mut m_tiles = vec![t_val * 4, t_val * 4 + 1, t_val * 4 + 2, t_val * 4 + 3];
                    if t_val == 4 {
                        m_tiles = vec![16, 17, 18, 19];
                    } else if t_val == 13 {
                        m_tiles = vec![52, 53, 54, 55];
                    } else if t_val == 22 {
                        m_tiles = vec![88, 89, 90, 91];
                    }

                    let tile37 = m_tiles
                        .iter()
                        .map(|&x| crate::v2_features::tile136_to_tile37(x))
                        .min()
                        .unwrap_or(0);
                    let involves_red5 = m_tiles
                        .iter()
                        .any(|&x| crate::v2_features::is_red5_tile136(x));
                    self.players[*seat].melds.push(Meld {
                        meld_type: *meld_type,
                        tiles: m_tiles,
                        opened: false,
                        from_who: -1,
                        called_tile: None,
                    });
                    let meld_idx = (self.players[*seat].melds.len().saturating_sub(1)) as u8;
                    self.v2_round_events
                        .push(crate::v2_features::events::RoundEvent::Ankan {
                            actor: *seat as u8,
                            tile37,
                            involves_red5,
                            turn_idx: self.v2_discard_seq,
                            meld_idx,
                        });
                } else {
                    // Kakan
                    let tile = tiles[0];
                    remove_hand_tile_matching(&mut self.players[*seat].hand, tile);
                    let mut kakan_meld_idx = 0u8;
                    for (idx, m) in self.players[*seat].melds.iter_mut().enumerate() {
                        if m.meld_type == MeldType::Pon && m.tiles[0] / 4 == tile / 4 {
                            m.meld_type = MeldType::Kakan;
                            m.tiles.push(tile);
                            m.tiles.sort();
                            kakan_meld_idx = idx as u8;
                            break;
                        }
                    }
                    self.v2_round_events
                        .push(crate::v2_features::events::RoundEvent::Kakan {
                            actor: *seat as u8,
                            tile37: crate::v2_features::tile136_to_tile37(tile),
                            turn_idx: self.v2_discard_seq,
                            meld_idx: kakan_meld_idx,
                        });
                    // Set last_discard so chankan ron targets the kakan player
                    self.last_discard = Some((*seat as u8, tile));
                }
                self.players[*seat].hand.sort();
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.active_players = vec![self.current_player];
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = true;
                // Also record ankan for kokushi chankan (kokushi can ron on closed kan)
                if *meld_type == MeldType::Ankan {
                    self.last_discard = Some((*seat as u8, tiles[0]));
                }
            }
            LogAction::Dora { dora_marker } => {
                self.wall.dora_indicators.push(*dora_marker);
                self.v2_round_events
                    .push(crate::v2_features::events::RoundEvent::Dora {
                        indicator_tile37: crate::v2_features::tile136_to_tile37(*dora_marker),
                        dora_idx: (self.wall.dora_indicators.len().saturating_sub(1)) as u8,
                        turn_idx: self.v2_discard_seq,
                    });
            }
            LogAction::Hule { hules } => {
                // If a riichi deposit is pending and this is a ron, the deposit
                // is voided (MjSoul does not deduct it when the discard is ronned).
                let first_is_ron = hules.first().is_some_and(|h| !h.zimo);
                if first_is_ron {
                    self.riichi_pending_acceptance = None;
                }

                let honba = self.honba;
                let riichi_on_table = self.riichi_sticks;
                let mut honba_taken = false;

                for h in hules {
                    let winner = h.seat;
                    let is_tsumo = h.zimo;
                    if winner < 4 {
                        self.v2_round_won[winner] = true;
                    }
                    if !is_tsumo
                        && let Some((discarder, _)) = self.last_discard
                        && (discarder as usize) < 4
                        && discarder as usize != winner
                    {
                        self.v2_round_dealt_in[discarder as usize] = true;
                    }

                    if is_tsumo {
                        let is_oya = (winner as u8) == self.oya;

                        // Check PAO (sekinin barai): for yakuman tsumo, the PAO
                        // player pays the full amount for all non-winning
                        // players. We detect PAO from the player's pao map
                        // which was populated when dragon/wind melds were claimed.
                        let mut pao_payer = None;
                        let mut pao_yakuman_val: i32 = 0;
                        let mut total_yakuman_val: i32 = 0;

                        if h.yiman {
                            // Daisangen = yaku 37, Daisuushii = yaku 50
                            // Double yakuman IDs: 47, 48, 49, 50
                            for &yid in &h.fans {
                                let val: i32 = if [47, 48, 49, 50].contains(&yid) {
                                    2
                                } else {
                                    1
                                };
                                total_yakuman_val += val;
                                if let Some(&liable) = self.players[winner].pao.get(&(yid as u8)) {
                                    pao_yakuman_val += val;
                                    pao_payer = Some(liable);
                                }
                            }
                        }

                        if pao_yakuman_val > 0 {
                            // PAO: liable player pays the PAO portion entirely
                            let unit: i32 = if is_oya { 48000 } else { 32000 };
                            let pao_amt = pao_yakuman_val * unit;
                            let non_pao_yakuman_val = total_yakuman_val - pao_yakuman_val;
                            let non_pao_amt = non_pao_yakuman_val * unit;

                            if let Some(pp) = pao_payer {
                                self.players[pp as usize].score -= pao_amt;
                                self.players[winner].score += pao_amt;
                            }

                            // Non-PAO part split normally
                            if non_pao_amt > 0 {
                                if is_oya {
                                    let share = non_pao_amt / 3;
                                    for i in 0..4 {
                                        if i != winner {
                                            self.players[i].score -= share;
                                            self.players[winner].score += share;
                                        }
                                    }
                                } else {
                                    for i in 0..4 {
                                        if i != winner {
                                            let share = if (i as u8) == self.oya {
                                                non_pao_amt / 2
                                            } else {
                                                non_pao_amt / 4
                                            };
                                            self.players[i].score -= share;
                                            self.players[winner].score += share;
                                        }
                                    }
                                }
                            }

                            // Add honba bonus (paid by PAO player)
                            if let Some(pp) = pao_payer {
                                let honba_total = honba as i32 * 300;
                                self.players[pp as usize].score -= honba_total;
                                self.players[winner].score += honba_total;
                            }
                        } else {
                            // Standard tsumo distribution
                            for i in 0..4 {
                                if i != winner {
                                    let base_pay = if is_oya {
                                        h.point_zimo_xian
                                    } else if (i as u8) == self.oya {
                                        h.point_zimo_qin
                                    } else {
                                        h.point_zimo_xian
                                    };
                                    let pay = base_pay as i32 + honba as i32 * 100;
                                    self.players[i].score -= pay;
                                    self.players[winner].score += pay;
                                }
                            }
                        }
                    } else if let Some((discarder, _)) = self.last_discard {
                        // Only the first ron winner gets the honba bonus
                        let ron_honba = if !honba_taken {
                            honba_taken = true;
                            honba
                        } else {
                            0
                        };

                        // Check PAO for ron yakuman: target pays half,
                        // PAO player pays the other half.
                        let mut pao_payer_ron: Option<u8> = None;
                        if h.yiman {
                            for &yid in &h.fans {
                                if let Some(&liable) = self.players[winner].pao.get(&(yid as u8)) {
                                    pao_payer_ron = Some(liable);
                                    break;
                                }
                            }
                        }

                        if let Some(pp) = pao_payer_ron {
                            let half = h.point_rong as i32 / 2;
                            let honba_pts = ron_honba as i32 * 300;
                            self.players[pp as usize].score -= half + honba_pts;
                            self.players[discarder as usize].score -= half;
                            self.players[winner].score += h.point_rong as i32 + honba_pts;
                        } else {
                            let pay = h.point_rong as i32 + ron_honba as i32 * 300;
                            self.players[discarder as usize].score -= pay;
                            self.players[winner].score += pay;
                        }
                    }
                }

                // Distribute riichi sticks to first winner
                if !hules.is_empty() {
                    let winner = hules[0].seat;
                    self.players[winner].score += riichi_on_table as i32 * 1000;
                    self.riichi_sticks = 0;
                }

                self.v2_record_round_end_if_needed(None);
                self.is_done = true;
            }
            LogAction::NoTile => {
                // Finalize pending riichi deposit (exhaustive draw, not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                    self.v2_round_events.push(
                        crate::v2_features::events::RoundEvent::ReachAccepted {
                            actor: rp,
                            turn_idx: self.v2_discard_seq,
                        },
                    );
                }

                // Check for nagashi mangan first
                let np = 4usize;
                let mut nagashi_winners = Vec::new();
                for (i, p) in self.players.iter().enumerate() {
                    if p.nagashi_eligible {
                        nagashi_winners.push(i as u8);
                    }
                }

                if !nagashi_winners.is_empty() {
                    // Nagashi mangan: apply mangan tsumo payment (no honba)
                    for &w in &nagashi_winners {
                        let is_oya = w == self.oya;
                        let score_res =
                            crate::score::calculate_score(5, 30, is_oya, true, 0, np as u8);
                        if is_oya {
                            for i in 0..np {
                                if i as u8 != w {
                                    self.players[i].score -= score_res.pay_tsumo_ko as i32;
                                    self.players[w as usize].score += score_res.pay_tsumo_ko as i32;
                                }
                            }
                        } else {
                            for i in 0..np {
                                if i as u8 != w {
                                    let pay = if i as u8 == self.oya {
                                        score_res.pay_tsumo_oya as i32
                                    } else {
                                        score_res.pay_tsumo_ko as i32
                                    };
                                    self.players[i].score -= pay;
                                    self.players[w as usize].score += pay;
                                }
                            }
                        }
                    }
                } else {
                    // Compute tenpai/noten payments
                    let mut tenpai = [false; 4];
                    for (i, p) in self.players.iter().enumerate() {
                        if i < 4 {
                            let calc = crate::hand_evaluator::HandEvaluator::new(
                                p.hand.clone(),
                                p.melds.clone(),
                            );
                            tenpai[i] = calc.is_tenpai_with_rule(&self.rule);
                        }
                    }
                    let num_tp = tenpai.iter().filter(|&&t| t).count();
                    if num_tp > 0 && num_tp < 4 {
                        let pk = 3000 / num_tp as i32;
                        let pn = 3000 / (4 - num_tp) as i32;
                        for (i, tp) in tenpai.iter().enumerate() {
                            let delta = if *tp { pk } else { -pn };
                            self.players[i].score += delta;
                        }
                    }
                }
                self.v2_record_round_end_if_needed(None);
                self.is_done = true;
            }
            LogAction::LiuJu { .. } => {
                // Finalize pending riichi deposit
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                    self.v2_round_events.push(
                        crate::v2_features::events::RoundEvent::ReachAccepted {
                            actor: rp,
                            turn_idx: self.v2_discard_seq,
                        },
                    );
                }
                // Abortive draw - no score changes
                self.v2_record_round_end_if_needed(None);
                self.is_done = true;
            }
            _ => {}
        }
    }
}
