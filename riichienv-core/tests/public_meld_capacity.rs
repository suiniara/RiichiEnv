use riichienv_core::public_rules::{
    PublicDiscard, PublicRuleContext, PublicRules, public_response_mask,
};

fn context() -> PublicRuleContext<'static> {
    PublicRuleContext {
        actor: 0,
        candidate: PublicDiscard {
            tile_type: 0,
            declares_riichi: false,
        },
        discard_history: [&[]; 4],
        current_rivers: [&[]; 4],
        meld_counts: [0; 4],
        kan_counts: [0; 4],
        riichi_accepted: [false; 4],
        remaining_draws: Some(20),
        history_complete: true,
        rules: PublicRules::default(),
    }
}

#[test]
fn four_established_melds_exclude_further_calls_but_not_ron() {
    for actor in 0..4 {
        for relative in 1..4 {
            for kans in 0..=4 {
                for history_complete in [false, true] {
                    let mut value = context();
                    value.actor = actor;
                    value.history_complete = history_complete;
                    let player = (actor + relative) % 4;
                    value.meld_counts[player] = 4;
                    value.kan_counts[player] = kans;
                    let mask = public_response_mask(&value).unwrap();
                    assert!(!mask[1 + relative], "fifth meld cannot be a pon");
                    assert!(!mask[4 + relative], "fifth meld cannot be a daiminkan");
                    assert_eq!(mask[1], relative != 1, "only shimocha can chi");
                    assert!(
                        mask[7 + (1 << (relative - 1))],
                        "tanki Ron remains possible"
                    );
                    assert!(mask[0], "one owner's kans do not cause an abort");
                }
            }
        }
    }
}

#[test]
fn three_established_melds_leave_room_for_one_more_call() {
    for actor in 0..4 {
        for relative in 1..4 {
            for kans in 0..=3 {
                let mut value = context();
                value.actor = actor;
                let player = (actor + relative) % 4;
                value.meld_counts[player] = 3;
                value.kan_counts[player] = kans;
                let mask = public_response_mask(&value).unwrap();
                assert!(mask[1 + relative]);
                assert!(mask[4 + relative]);
                assert!(mask[1]);
            }
        }
    }
}
