//! Formal fifth-copy waits differ between Tenhou and Mahjong Soul. Preserve
//! that distinction through legal actions, frozen observations, and payments.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::observation::Observation;
use riichienv_core::replay::Action as LogAction;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use riichienv_core::types::{Meld, MeldType};
use serde_json::Value;

fn formal_wait_state(rule: GameRule, has_draw: bool) -> GameState {
    let mut state = GameState::new(0, false, Some(42), 0, rule);
    for player in &mut state.players {
        player.reset_round();
        player.nagashi_eligible = false;
    }
    // 2m4m 456p 789s EE, with an established concealed kan of 3m.
    // The only formal completion is a fifth 3m; the concealed hand has none.
    state.players[0].hand = vec![4, 12, 48, 53, 56, 96, 100, 104, 108, 109];
    state.players[0].melds = vec![Meld::new(
        MeldType::Ankan,
        vec![8, 9, 10, 11],
        false,
        0,
        None,
    )];
    if has_draw {
        state.players[0].hand.push(132);
    }
    let own_tiles = state.players[0]
        .hand
        .iter()
        .copied()
        .chain(state.players[0].melds[0].tiles.iter().copied())
        .collect::<Vec<_>>();
    let mut pool = (0..136)
        .filter(|tile| !own_tiles.contains(tile))
        .collect::<Vec<u8>>();
    // Six distinct honors make these hands unambiguously noten.
    for player in state.players.iter_mut().skip(1) {
        for kind in [0, 4, 6, 8, 9, 17, 18, 28, 29, 30, 31, 32, 33] {
            let slot = pool.iter().position(|tile| tile / 4 == kind).unwrap();
            player.hand.push(pool.remove(slot));
        }
    }
    state.wall.tiles = pool;
    state.wall.drawable_count = (state.wall.tiles.len() - 14) as u8;
    state.wall.dora_indicators = vec![state.wall.tiles[4]];
    state.wall.rinshan_draw_count = 1;
    state.wall.pending_kan_dora_count = 0;
    state.current_player = 0;
    state.active_players = vec![0];
    state.drawn_tile = has_draw.then_some(132);
    state.needs_tsumo = false;
    state.is_first_turn = false;
    state.phase = Phase::WaitAct;
    state.mjai_log.clear();
    state.rule_log.clear();
    state.mjai_log_per_player = Default::default();
    state.player_event_counts = [0; 4];
    let mut inventory = state
        .wall
        .tiles
        .iter()
        .copied()
        .chain(state.players.iter().flat_map(|player| {
            player.hand.iter().copied().chain(
                player
                    .melds
                    .iter()
                    .flat_map(|meld| meld.tiles.iter().copied()),
            )
        }))
        .collect::<Vec<_>>();
    inventory.sort();
    assert_eq!(inventory, (0..136).collect::<Vec<u8>>());
    state
}

fn riichi_tiles(actions: Vec<Action>) -> Vec<u8> {
    actions
        .into_iter()
        .map(|action| action.tile.unwrap())
        .collect()
}

#[test]
fn native_waits_use_the_producing_ruleset() {
    for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
        let mut state = formal_wait_state(rule, false);
        let observation = state.get_observation(0);
        assert_eq!(observation.rule, rule);
        assert_eq!(observation.is_tenpai, rule.allow_fifth_copy_tenpai);
        assert_eq!(
            observation.waits,
            if rule.allow_fifth_copy_tenpai {
                vec![2]
            } else {
                vec![]
            }
        );
    }
}

#[test]
fn complete_and_two_step_riichi_share_formal_tenpai_rule() {
    for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
        let mut state = formal_wait_state(rule, true);
        let expected = if rule.allow_fifth_copy_tenpai {
            vec![132]
        } else {
            vec![]
        };
        assert_eq!(riichi_tiles(state.legal_riichi_discards(0)), expected);
        let observation = state.get_observation(0);
        assert_eq!(
            observation
                .legal_actions_method()
                .iter()
                .any(|action| action.action_type == ActionType::Riichi),
            rule.allow_fifth_copy_tenpai
        );
        assert_eq!(
            riichi_tiles(observation.legal_riichi_discard_actions().unwrap()),
            expected
        );
        // The old two-step declaration stage applies the very same rule.
        state.players[0].riichi_stage = true;
        let staged_discards = state
            .get_observation(0)
            .legal_actions_method()
            .into_iter()
            .filter(|action| action.action_type == ActionType::Discard)
            .filter_map(|action| action.tile)
            .collect::<Vec<_>>();
        assert_eq!(staged_discards, expected);
    }
}

#[test]
fn frozen_observation_retains_rule_and_legacy_payload_defaults() {
    let mut state = formal_wait_state(GameRule::default_tenhou(), true);
    let observation = state.get_observation(0);
    let mut json = serde_json::to_value(&observation).unwrap();
    // Keep an identical frozen candidate list while changing only the rule.
    // This catches an observation helper accidentally using the old default.
    json["rule"] = serde_json::to_value(GameRule::default_mjsoul()).unwrap();
    let strict: Observation = serde_json::from_value(json).unwrap();
    assert!(strict.legal_riichi_discard_actions().unwrap().is_empty());
    let restored =
        Observation::deserialize_from_base64(&strict.serialize_to_base64().unwrap()).unwrap();
    assert_eq!(restored.rule, GameRule::default_mjsoul());
    assert!(restored.legal_riichi_discard_actions().unwrap().is_empty());

    let mut legacy = serde_json::to_value(&observation).unwrap();
    legacy.as_object_mut().unwrap().remove("rule");
    let payload = BASE64.encode(serde_json::to_vec(&legacy).unwrap());
    let restored = Observation::deserialize_from_base64(&payload).unwrap();
    assert_eq!(restored.rule, GameRule::default_tenhou());
    assert_eq!(
        riichi_tiles(restored.legal_riichi_discard_actions().unwrap()),
        vec![132]
    );
}

#[test]
fn exhaustive_draw_and_log_no_tile_apply_rule_specific_noten_payments() {
    for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
        let expected = if rule.allow_fifth_copy_tenpai {
            [28000, 24000, 24000, 24000]
        } else {
            [25000; 4]
        };
        let mut online = formal_wait_state(rule, false);
        online.wall.drawable_count = 0;
        online._trigger_ryukyoku("exhaustive_draw");
        assert_eq!(
            online.players.each_ref().map(|player| player.score),
            expected
        );
        let resolution: Value = online
            .rule_log
            .iter()
            .map(|event| serde_json::from_str::<Value>(event).unwrap())
            .find(|event| event["type"] == "ryukyoku")
            .unwrap();
        assert_eq!(
            resolution["public_hands"][0].is_array(),
            rule.allow_fifth_copy_tenpai
        );

        let mut replay = formal_wait_state(rule, false);
        replay.apply_log_action(&LogAction::NoTile);
        assert_eq!(
            replay.players.each_ref().map(|player| player.score),
            expected
        );
        assert!(replay.is_done);
    }
}
