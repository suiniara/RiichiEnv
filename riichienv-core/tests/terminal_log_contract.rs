//! Rich terminal facts belong to the canonical rule log, while legacy MJAI
//! clients retain the pre-existing event and per-player log schemas.

use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use serde_json::{Value, json};

fn last(log: &[String]) -> Value {
    serde_json::from_str(log.last().expect("one emitted event")).unwrap()
}

fn native_kyushu() -> GameState {
    use riichienv_core::action::ActionType;
    use riichienv_core::state::legal_actions::GameStateLegalActions;
    use std::collections::HashMap;

    let reported: Vec<Value> = include_str!("fixtures/issue_229_south_4.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let mut state = GameState::new(2, false, Some(229), 0, GameRule::default_mjsoul());
    state.reset();
    for event in &reported[..2] {
        state.apply_mjai_event(serde_json::from_value(event.clone()).unwrap());
        assert!(state.last_error.is_none());
        state._push_mjai_event(event.clone());
    }
    let action = state
        ._get_legal_actions_internal(3)
        .into_iter()
        .find(|action| action.action_type == ActionType::KyushuKyuhai)
        .expect("the reported declaration is legal");
    state.step(&HashMap::from([(3, action)]));
    assert!(state.last_error.is_none());
    state
}

#[test]
fn actual_native_resolution_keeps_rich_facts_out_of_legacy_logs() {
    let state = native_kyushu();
    let resolution = |log: &[String]| -> Value {
        log.iter()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .find(|event| event["type"] == "ryukyoku")
            .unwrap()
    };
    let legacy = json!({"type":"ryukyoku", "reason":"kyushu_kyuhai", "deltas":[0,0,0,0]});
    assert_eq!(resolution(&state.mjai_log), legacy);
    for log in &state.mjai_log_per_player {
        assert_eq!(resolution(log), legacy);
    }
    let canonical = resolution(&state.rule_log);
    assert_eq!(canonical["actor"], 3);
    assert_eq!(canonical["scores"], json!([6000, 34500, 14600, 44900]));
    assert_eq!(canonical["public_hands"][3].as_array().unwrap().len(), 14);
    assert_eq!(canonical["public_hands"][0], Value::Null);
    assert_eq!(canonical["visibility_complete"], true);
    assert_eq!(
        canonical["public_reveal_provenance"],
        "native_round_resolution_v1"
    );
}

#[test]
fn replay_supplied_terminal_facts_are_preserved_even_with_native_provenance() {
    let source = native_kyushu();
    let mut replay = GameState::new(2, false, Some(73), 0, GameRule::default_mjsoul());
    replay.reset();
    replay.mjai_log.clear();
    for log in &mut replay.mjai_log_per_player {
        log.clear();
    }
    for line in &source.rule_log {
        let mut event: Value = serde_json::from_str(line).unwrap();
        event["caller_annotation"] = json!("must survive raw replay logging");
        // This is the apply_event / observe_event native adapter sequence.
        replay.apply_mjai_event(serde_json::from_value(event.clone()).unwrap());
        assert!(replay.last_error.is_none(), "{:?}", replay.last_error);
        replay._push_mjai_event(event.clone());
        assert_eq!(last(&replay.mjai_log), event);
        assert_eq!(last(&replay.rule_log), event);
        if event["type"] == "ryukyoku" {
            for log in &replay.mjai_log_per_player {
                assert_eq!(last(log), event);
            }
        }
    }
}

#[test]
fn opening_scores_and_observer_hand_masking_keep_the_existing_schema() {
    let state = GameState::new(0, false, Some(72), 0, GameRule::default_mjsoul());
    let opening = |log: &[String]| -> Value {
        log.iter()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .find(|event| event["type"] == "start_kyoku")
            .unwrap()
    };
    let full = opening(&state.mjai_log);
    assert_eq!(full["scores"], json!(vec![25000; 4]));
    assert_eq!(opening(&state.rule_log), full);
    for (seat, log) in state.mjai_log_per_player.iter().enumerate() {
        let observed = opening(log);
        assert_eq!(observed["scores"], full["scores"]);
        for player in 0..4 {
            if player == seat {
                assert_eq!(observed["tehais"][player], full["tehais"][player]);
            } else {
                assert_eq!(observed["tehais"][player], json!(vec!["?"; 13]));
            }
        }
    }
}
