//! Mahjong Soul FAQ id=47: https://mahjongsoul.com/api/faq/list
//! Own calls/kans preserve nagashi; called-away discards and middle discards do not.
//! These are physically allocated event-prefix regressions, not full-game replays.
//! Full legal-wall rounds and independent payments are checked in Python's
//! tests/kar/test_nagashi_consistency.py; no eligibility flags are injected here.

use riichienv_core::parser::{mjai_to_tid, tid_to_mjai};
use riichienv_core::replay::Action as LogAction;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use riichienv_core::types::MeldType;
use serde_json::{Value, json};

fn take(pool: &mut Vec<u8>, name: &str) -> String {
    let slot = pool
        .iter()
        .position(|&tile| tid_to_mjai(tile) == name)
        .unwrap();
    tid_to_mjai(pool.remove(slot))
}

fn prefix(kind: &str) -> Vec<Value> {
    let mut pool: Vec<u8> = (0..136).collect();
    let dora = take(&mut pool, "C");
    let call = matches!(kind, "chi" | "pon" | "daiminkan" | "kakan");
    let called = if kind == "chi" { "1m" } else { "E" };
    let consumed = match kind {
        "chi" => vec!["2m", "3m"],
        "pon" | "kakan" => vec!["E", "E"],
        "daiminkan" => vec!["E"; 3],
        "ankan" => vec!["E"; 4],
        _ => vec![],
    };
    let owner = if kind == "ankan" { 0 } else { 1 };
    let mut hands = vec![Vec::new(); 4];
    if call {
        hands[0].push(take(&mut pool, called));
    }
    for name in &consumed {
        hands[owner].push(take(&mut pool, name));
    }
    hands[1].push(take(&mut pool, "9p"));
    let mut reserved = vec!["2p"];
    if kind == "kakan" {
        reserved.extend(["3p", "4p", "5p", "E", "9s"]);
    } else if matches!(kind, "ankan" | "daiminkan") {
        reserved.push("9s");
    }
    for name in reserved {
        take(&mut pool, name);
    }
    for hand in &mut hands {
        while hand.len() < 13 {
            hand.push(tid_to_mjai(pool.remove(0)));
        }
    }
    let mut events = vec![
        json!({"type":"start_kyoku", "bakaze":"E", "kyoku":1, "honba":0,
               "kyotaku":0, "oya":0, "scores":[25000,25000,25000,25000], "dora_marker":dora, "tehais":hands}),
        json!({"type":"tsumo", "actor":0, "pai":"2p"}),
    ];
    if call {
        events.push(json!({"type":"dahai", "actor":0, "pai":called, "tsumogiri":false}));
        events.push(json!({"type":if kind == "kakan" {"pon"} else {kind},
                           "actor":1, "target":0, "pai":called, "consumed":consumed}));
        if kind == "daiminkan" {
            events.push(json!({"type":"tsumo", "actor":1, "pai":"9s"}));
        }
        events.push(json!({"type":"dahai", "actor":1, "pai":"9p", "tsumogiri":false}));
        if kind == "kakan" {
            for (actor, pai) in [(2, "3p"), (3, "4p"), (0, "5p")] {
                events.push(json!({"type":"tsumo", "actor":actor, "pai":pai}));
                events.push(json!({"type":"dahai", "actor":actor, "pai":pai, "tsumogiri":true}));
            }
            events.extend([
                json!({"type":"tsumo", "actor":1, "pai":"E"}),
                json!({"type":"kakan", "actor":1, "pai":"E"}),
                json!({"type":"tsumo", "actor":1, "pai":"9s"}),
                json!({"type":"dahai", "actor":1, "pai":"9s", "tsumogiri":true}),
            ]);
        }
    } else if kind == "ankan" {
        events.extend([
            json!({"type":"ankan", "actor":0, "consumed":consumed}),
            json!({"type":"tsumo", "actor":0, "pai":"9s"}),
            json!({"type":"dahai", "actor":0, "pai":"9s", "tsumogiri":true}),
        ]);
    } else {
        events.push(json!({"type":"dahai", "actor":0, "pai":"2p", "tsumogiri":true}));
    }
    events
}

fn tile(event: &Value) -> u8 {
    mjai_to_tid(event["pai"].as_str().unwrap()).unwrap()
}

fn ingest_logs(state: &mut GameState, event: &Value) {
    let seat = event["actor"].as_u64().unwrap_or(0) as usize;
    let action = match event["type"].as_str().unwrap() {
        "tsumo" => LogAction::DealTile {
            seat,
            tile: tile(event),
            doras: None,
            left_tile_count: None,
        },
        "dahai" => LogAction::DiscardTile {
            seat,
            tile: tile(event),
            is_liqi: false,
            is_wliqi: false,
            doras: None,
        },
        "chi" | "pon" | "daiminkan" => {
            let kind = match event["type"].as_str().unwrap() {
                "chi" => MeldType::Chi,
                "pon" => MeldType::Pon,
                _ => MeldType::Daiminkan,
            };
            let mut tiles = vec![tile(event)];
            tiles.extend(
                event["consumed"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| mjai_to_tid(value.as_str().unwrap()).unwrap()),
            );
            let mut froms = vec![event["target"].as_u64().unwrap() as usize];
            froms.extend(vec![seat; tiles.len() - 1]);
            LogAction::ChiPengGang {
                seat,
                meld_type: kind,
                tiles,
                froms,
            }
        }
        "ankan" | "kakan" => {
            let concealed = event["type"] == "ankan";
            let tiles = if concealed {
                event["consumed"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| mjai_to_tid(value.as_str().unwrap()).unwrap())
                    .collect()
            } else {
                vec![tile(event)]
            };
            LogAction::AnGangAddGang {
                seat,
                meld_type: if concealed {
                    MeldType::Ankan
                } else {
                    MeldType::Kakan
                },
                tiles,
                tile_raw_id: 0,
                doras: None,
            }
        }
        other => panic!("Unexpected test event {other}"),
    };
    state.apply_log_action(&action);
}

fn compare(kind: &str) {
    let events = prefix(kind);
    let mut mjai = GameState::new(2, false, Some(214), 0, GameRule::default_mjsoul());
    let mut log = mjai.clone();
    for event in &events {
        mjai.apply_mjai_event(serde_json::from_value(event.clone()).unwrap());
        if event["type"] == "start_kyoku" {
            log.apply_mjai_event(serde_json::from_value(event.clone()).unwrap());
        } else {
            ingest_logs(&mut log, event);
        }
    }
    let expected: Vec<bool> = log
        .players
        .iter()
        .map(|player| player.nagashi_eligible)
        .collect();
    if matches!(kind, "chi" | "pon" | "daiminkan" | "kakan") {
        assert!(!expected[0], "the called-away discarder is disqualified");
        assert!(
            expected[1],
            "the caller keeps eligibility under Mahjong Soul"
        );
    } else if kind == "ankan" {
        assert!(expected[0], "an own concealed kan is not a discard");
    } else {
        assert!(
            !expected[0],
            "the known middle discard disqualifies its owner"
        );
    }
    assert_eq!(
        mjai.players
            .iter()
            .map(|player| player.nagashi_eligible)
            .collect::<Vec<_>>(),
        expected,
        "MJAI and LogAction ingestion disagree for {kind}"
    );
    for state in [&mjai, &log] {
        assert!(
            state
                .players
                .iter()
                .all(|player| player.hand.len() + 3 * player.melds.len() == 13)
        );
    }
}

#[test]
fn mjai_middle_discard_matches_log_replay() {
    compare("none");
}
#[test]
fn mjai_chi_clears_discarder_not_caller() {
    compare("chi");
}
#[test]
fn mjai_pon_clears_discarder_not_caller() {
    compare("pon");
}
#[test]
fn mjai_daiminkan_clears_discarder_not_caller() {
    compare("daiminkan");
}
#[test]
fn own_ankan_preserves_nagashi_on_both_ingestion_paths() {
    compare("ankan");
}
#[test]
fn own_kakan_preserves_nagashi_after_existing_pon() {
    compare("kakan");
}
