//! Automated Stage-1 Contract Test Suite for KAR V2.4 Rust Feature Engine.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::action::{Action, ActionType};
use crate::hand_evaluator::HandEvaluator;
use crate::observation::Observation;
use crate::rule::GameRule;
use crate::state::GameState;
use crate::types::{Conditions, Meld, MeldType, TILE_MAX, Wind};

use super::action_manifest::{
    ManifestActionKind, decode_global_id_to_native, entry, lookup_ankan_id, lookup_chi_id,
    lookup_daiminkan_id, lookup_discard_id, lookup_kakan_id, lookup_pon_id, lookup_ron_id,
    manifest, resolve_legal_candidates,
};
use super::candidates::{CandidateMode, WinEvalContext, encode_candidates};
use super::events::{
    EventKind, PASS_CHI, PASS_KAN, PASS_PON, PASS_RON, RoundEvent, RoundEventLog, decode_events,
    encode_events,
};
use super::furiten::FuritenState;
use super::global_player::{RoundGlobalContext, encode_global_player};
use super::oracle::encode_oracle;
use super::tile_feats::encode_tile_feats;
use super::visible::VisibleTileCounts;
use super::{
    ACTION_COUNT, DecisionPhase, HanchanTracker, MAX_CANDIDATES, MAX_EVENT_LEN, NULL_SOURCE_IDX,
    NULL_TILE_IDX, RoundOutcomeSummary, RuleContract, TILE_FEATS_DIM, V2FeatureConfig,
};

fn empty_obs(player_id: u8) -> Observation {
    Observation::new(
        player_id,
        [vec![], vec![], vec![], vec![]],
        [vec![], vec![], vec![], vec![]],
        [vec![], vec![], vec![], vec![]],
        vec![0], // 1m indicator
        [25_000; 4],
        [false; 4],
        vec![],
        vec![],
        0,
        0,
        0,
        0,
        0,
        vec![],
        false,
        [None; 4],
        [None; 4],
        None,
        None,
    )
}

// =========================================================================
// 1. P0-2: Physical VisibleTileCounts deduplication across all meld types
// =========================================================================
#[test]
fn visible_tile_counts_deduplicates_called_tiles_and_tracks_red5() {
    let mut obs = empty_obs(0);
    // Self holds normal 5m (17), red 5p (52), and 1s (72)
    obs.hands[0] = vec![17, 52, 72];
    // Seat 1 discarded red 5m (16) and normal 5m (18); seat 2 PON'd 16 using 19 + 16? Wait:
    // Use distinct physical 5m copies: hand=17, seat1 river=16(red5m), seat2 pon={16, 18, 19} called=16!
    obs.discards[1] = vec![16, 24]; // 0m(16), 7m(24)
    obs.melds[2].push(Meld {
        meld_type: MeldType::Pon,
        tiles: vec![16, 18, 19],
        opened: true,
        from_who: 1,
        called_tile: Some(16),
    });
    // Seat 3 upgrades a PON of 7m (24,25,26 called=24) to KAKAN with 27:
    obs.melds[3].push(Meld {
        meld_type: MeldType::Kakan,
        tiles: vec![24, 25, 26, 27],
        opened: true,
        from_who: 1,
        called_tile: Some(24),
    });
    // Seat 0 has an ANKAN of 9s (104,105,106,107, called_tile=None):
    obs.melds[0].push(Meld {
        meld_type: MeldType::Ankan,
        tiles: vec![104, 105, 106, 107],
        opened: false,
        from_who: 0,
        called_tile: None,
    });
    // Dora indicator is 1m (0)
    obs.dora_indicators = vec![0];

    let vis = VisibleTileCounts::from_observation(&obs).expect("valid physical state");

    // 5m (base34 = 4): H=1 (17), R=1 (16), M=3 (16,18,19), C=1 (16) => V=3, U=0!
    assert_eq!(vis.self_hand_base34[4], 1);
    assert_eq!(vis.visible_base34[4], 3);
    assert_eq!(vis.unseen_base34[4], 0);
    // 7m (base34 = 6): H=0, R=1 (24), M=4 (24..27), C=1 (24) => V=4, U=0!
    assert_eq!(vis.visible_base34[6], 4);
    assert_eq!(vis.unseen_base34[6], 0);
    // 9s (base34 = 26): H=0, M=4, C=0 => V=4, U=0!
    assert_eq!(vis.visible_base34[26], 4);
    assert_eq!(vis.unseen_base34[26], 0);
    // Red 5s: 0m(16) seen in pon, 0p(52) seen in self hand, 0s(88) still unseen!
    assert_eq!(vis.red5_unseen, [false, false, true]);

    // Global conservation across all 34 tile types
    for b in 0..TILE_MAX {
        assert_eq!(
            vis.self_hand_base34[b] + vis.visible_base34[b] + vis.unseen_base34[b],
            4
        );
    }

    // Also verify 241-byte layout, Kakan recovery when called_tile=None, and orphan called_tile rejection
    let raw = vis.to_bytes_241();
    assert_eq!(raw.len(), 241);
    assert_eq!(raw[136 + 4], 3); // visible_base34[4] == 3
    assert_eq!(raw[170 + 4], 1); // self_hand_base34[4] == 1
    assert_eq!(raw[204 + 4], 0); // unseen_base34[4] == 0
    assert_eq!(&raw[238..241], &[0, 0, 1]);

    // Kakan with called_tile=None recovers claimed tile via unique river intersection
    let mut obs_kakan = empty_obs(0);
    obs_kakan.dora_indicators.clear();
    obs_kakan.discards[2].push(40);
    obs_kakan.melds[0].push(Meld::new(
        MeldType::Kakan,
        vec![40, 41, 42, 43],
        true,
        2,
        None,
    ));
    let vis_kakan = VisibleTileCounts::from_observation(&obs_kakan).unwrap();
    assert_eq!(vis_kakan.visible_base34[10], 4);

    // Orphan called_tile absent from rivers MUST be rejected
    let mut obs_orphan = empty_obs(0);
    obs_orphan.dora_indicators.clear();
    obs_orphan.melds[0].push(Meld::new(
        MeldType::Pon,
        vec![40, 41, 42],
        true,
        2,
        Some(40),
    ));
    assert!(VisibleTileCounts::from_observation(&obs_orphan).is_err());

    // Live GameState preserves temporary & riichi furiten directly on frozen Observation.furiten
    let mut state = GameState::new(0, true, Some(99), 0, GameRule::default_tenhou());
    state.players[0].missed_agari_doujun = true;
    state.players[0].missed_agari_riichi = true;
    let frozen_obs = state.get_observation(0);
    assert!(frozen_obs.furiten.temporary);
    assert!(frozen_obs.furiten.riichi);
    assert!(frozen_obs.furiten.any());
}

// =========================================================================
// 2. P1-2: Complete FuritenState (permanent, temporary across chi/pon, riichi)
// =========================================================================
#[test]
fn furiten_state_covers_permanent_temporary_and_riichi() {
    let mut obs = empty_obs(0);
    // Tenpai hand waiting on 1m(0) / 4m(3): 23m 234p 567p 789s EE
    obs.hands[0] = vec![4, 8, 40, 44, 48, 53, 56, 60, 96, 100, 104, 108, 109];
    obs.discards[0] = vec![12]; // discarded 4m -> permanent furiten!
    let rule = GameRule::default_tenhou();

    let f_perm = FuritenState::from_observation(&obs, false, false, &rule);
    assert!(f_perm.permanent);
    assert!(!f_perm.temporary);
    assert!(!f_perm.riichi);
    assert!(f_perm.any());

    // Clear own river, test temporary furiten:
    // - persists when NOT discarding (`PASS`, discarded_base34 == None)
    // - CLEARS after executing own discard (`DISCARD(a)`, discarded_base34 == Some(a)),
    //   matching `GameState::populate_discard_claims` and `tests/temporary_furiten.rs`.
    obs.discards[0].clear();
    let f_temp = FuritenState::from_observation(&obs, true, false, &rule);
    assert!(!f_temp.permanent);
    assert!(f_temp.temporary);
    assert!(f_temp.non_river_active());
    let own_disc = [false; TILE_MAX];
    assert!(f_temp.is_post_discard_furiten(&[0, 3], &own_disc, None));
    assert!(!f_temp.is_post_discard_furiten(&[0, 3], &own_disc, Some(10)));
    // Self-discarding one of the new waits (`3`) makes post-discard permanently river-furiten
    assert!(f_temp.is_post_discard_furiten(&[0, 3], &own_disc, Some(3)));

    // Riichi furiten stays active permanently even across own discard
    let f_riichi = FuritenState::from_observation(&obs, false, true, &rule);
    assert!(f_riichi.riichi);
    assert!(f_riichi.is_post_discard_furiten(&[0], &own_disc, None));
    assert!(f_riichi.is_post_discard_furiten(&[0], &own_disc, Some(10)));

    // Per-candidate permanent furiten recomputation on `tile_feats[b, 61]` and `cand_dyn[slot, 5]`:
    // 14-tile hand: 235m 234p 567p 789s EE with own river = [1m(0)].
    // - Discard 5m(base34=4) -> waits {1m, 4m} hits 1m in river -> post_is_furiten == 1.0
    // - Discard 2m(base34=1) -> waits {4m} misses river [1m]    -> post_is_furiten == 0.0
    let mut obs_switch = empty_obs(0);
    obs_switch.hands[0] = vec![4, 8, 17, 40, 44, 48, 53, 56, 60, 96, 100, 104, 108, 109];
    obs_switch.discards[0] = vec![0]; // 1m in own river
    obs_switch.dora_indicators = vec![1];
    obs_switch._legal_actions = vec![
        Action::new(ActionType::Discard, Some(17), vec![], Some(0)), // 5m (base34=4)
        Action::new(ActionType::Discard, Some(4), vec![], Some(0)),  // 2m (base34=1)
    ];
    let vis_sw = VisibleTileCounts::from_observation(&obs_switch).unwrap();
    let fur_sw = FuritenState::from_observation(&obs_switch, false, false, &rule);
    let tf_sw = encode_tile_feats(
        &obs_switch,
        &vis_sw,
        fur_sw,
        &rule,
        DecisionPhase::OwnAct,
        &V2FeatureConfig::default(),
    )
    .unwrap();
    assert_eq!(tf_sw[4 * TILE_FEATS_DIM + 60], 1.0); // 5m discard is tenpai
    assert_eq!(tf_sw[4 * TILE_FEATS_DIM + 61], 1.0); // 5m discard -> waits {1m,4m} -> FURITEN
    assert_eq!(tf_sw[TILE_FEATS_DIM + 60], 1.0); // 2m discard is tenpai
    assert_eq!(tf_sw[TILE_FEATS_DIM + 61], 0.0); // 2m discard -> waits {4m} -> NOT furiten

    // Verify global_feats[19] reflects temporary || riichi
    let tracker = HanchanTracker::default();
    let rc = RuleContract::default_tenhou_houou();
    let ctx = RoundGlobalContext::default();
    let cands_sw = encode_candidates(
        &obs_switch,
        &vis_sw,
        fur_sw,
        &tracker,
        &rc,
        &ctx,
        &WinEvalContext::default(),
    )
    .unwrap();
    for slot in 0..cands_sw.num_valid {
        let e = entry(cands_sw.cand_ids[slot] as u16).unwrap();
        if e.target_base34 == 4 {
            assert_eq!(cands_sw.cand_dyn[slot][5], 1.0);
        } else if e.target_base34 == 1 {
            assert_eq!(cands_sw.cand_dyn[slot][5], 0.0);
        }
    }
    let (g_temp, _) = encode_global_player(&obs, f_temp, &tracker, &rc, &ctx).unwrap();
    let (g_none, _) =
        encode_global_player(&obs, FuritenState::default(), &tracker, &rc, &ctx).unwrap();
    assert_eq!(g_temp[19], 1.0);
    assert_eq!(g_none[19], 0.0);
}

// =========================================================================
// 3. P1-2 (v2.2): HandEvaluator::calc_with_rule Tenhou vs MJSoul double yakuman
// =========================================================================
#[test]
fn hand_evaluator_calc_with_rule_distinguishes_tenhou_and_mjsoul_double_yakuman() {
    // 13-wait Kokushi Musou: all 13 distinct terminals/honors + win on 1m
    let hand_13 = vec![
        0, 32, 36, 68, 72, 104, // 1m 9m 1p 9p 1s 9s
        108, 112, 116, 120, 124, 128, 132, // E S W N P F C
    ];
    let eval = HandEvaluator::new(hand_13, vec![]);
    let cond = Conditions {
        tsumo: false,
        player_wind: Wind::South,
        round_wind: Wind::East,
        ..Default::default()
    };

    let res_tenhou = eval.calc_with_rule(
        1,
        &[4],
        &[],
        Some(cond.clone()),
        &GameRule::default_tenhou(),
    );
    let res_mjs = eval.calc_with_rule(1, &[4], &[], Some(cond), &GameRule::default_mjsoul());

    assert!(res_tenhou.is_win && res_tenhou.yakuman);
    assert!(res_mjs.is_win && res_mjs.yakuman);
    // Tenhou caps Kokushi 13-machi to single yakuman (13 han, 32,000 ko); MJS awards double (26 han, 64,000 ko)
    assert_eq!(res_tenhou.han, 13);
    assert_eq!(res_tenhou.ron_agari, 32_000);
    assert_eq!(res_mjs.han, 26);
    assert_eq!(res_mjs.ron_agari, 64_000);
}

// =========================================================================
// 4. P0-1 & P1-2 (v2.4): tile_feats strict public observability & zero-leakage
// =========================================================================
#[test]
fn tile_feats_respects_early_shape_hint_flag_and_strict_own_river_suji() {
    let mut obs = empty_obs(0);
    obs.hands[0] = vec![0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 112, 113, 116];
    obs.dora_indicators = vec![1]; // distinct physical 1m copy from hand's 0
    // Seat 1 (rel=1) discards 4m(12) early; seat 2 (rel=2) discards 5m(17) AFTER seat 1 riichi.
    obs.discards[1] = vec![12]; // 4m (base34=3)
    obs.tsumogiri_flags[1] = vec![false]; // tedashi
    obs.riichi_declared[1] = true;
    obs.riichi_sutehais[1] = Some(12);
    obs.discards[2] = vec![17]; // 5m (base34=4)
    obs.tsumogiri_flags[2] = vec![true];

    let vis = VisibleTileCounts::from_observation(&obs).unwrap();
    let fur = FuritenState::default();
    let rule = GameRule::default_tenhou();

    // Default config: enable_early_shape_hint == false
    let f_default = encode_tile_feats(
        &obs,
        &vis,
        fur,
        &rule,
        DecisionPhase::OwnAct,
        &V2FeatureConfig::default(),
    )
    .unwrap();

    // Check seat 1 (rel=1, off=19):
    // +0 own_river_genbutsu: 4m(3) = 1.0, 5m(4) MUST be 0.0 (discarded by seat 2, not seat 1!)
    assert_eq!(f_default[3 * TILE_FEATS_DIM + 19], 1.0);
    assert_eq!(f_default[4 * TILE_FEATS_DIM + 19], 0.0);
    // +1 riichi_declare_tile: 4m(3) = 1.0
    assert_eq!(f_default[3 * TILE_FEATS_DIM + 20], 1.0);
    // +2 last_discard_tile: 4m(3) = 1.0
    assert_eq!(f_default[3 * TILE_FEATS_DIM + 21], 1.0);
    // +3/+4 suji from seat 1's own 4m: 1m(0) and 7m(6) are full suji; 2m(1)/8m(7) from seat 2's 5m MUST be 0 for seat 1!
    assert_eq!(f_default[22], 1.0);
    assert_eq!(f_default[23], 1.0);
    assert_eq!(f_default[6 * TILE_FEATS_DIM + 23], 1.0);
    assert_eq!(f_default[TILE_FEATS_DIM + 22], 0.0);
    assert_eq!(f_default[7 * TILE_FEATS_DIM + 22], 0.0);
    // +5 early_discard_shape_hint MUST be all 0.0 under default config!
    for b in 0..TILE_MAX {
        assert_eq!(f_default[b * TILE_FEATS_DIM + 24], 0.0);
    }

    // Enabled config: early 4m -> 1m(0), 2m(1), 3m(2) get shape hint 1.0, 4m(3)/5m(4) remain 0.0
    let f_hint = encode_tile_feats(
        &obs,
        &vis,
        fur,
        &rule,
        DecisionPhase::OwnAct,
        &V2FeatureConfig {
            enable_early_shape_hint: true,
        },
    )
    .unwrap();
    assert_eq!(f_hint[24], 1.0);
    assert_eq!(f_hint[TILE_FEATS_DIM + 24], 1.0);
    assert_eq!(f_hint[2 * TILE_FEATS_DIM + 24], 1.0);
    assert_eq!(f_hint[3 * TILE_FEATS_DIM + 24], 0.0);
    assert_eq!(f_hint[4 * TILE_FEATS_DIM + 24], 0.0);
}

// =========================================================================
// 5. P1-1 & P1-5 (v2.4): Events 10B round-trip + 10,000 extreme round fuzz
// =========================================================================
#[test]
fn events_10byte_round_trip_all_kinds_and_filters_opponent_private() {
    let mut log = RoundEventLog::new(10).unwrap(); // red 5p initial dora
    log.push(RoundEvent::TsumoPrivate {
        actor: 0,
        tile37: 0, // red 5m
        is_rinshan: false,
        turn_idx: 1,
    });
    // Opponent private draw MUST NOT appear in observer 0's stream
    log.push(RoundEvent::TsumoPrivate {
        actor: 1,
        tile37: 33,
        is_rinshan: false,
        turn_idx: 1,
    });
    log.push(RoundEvent::Dahai {
        actor: 0,
        tile37: 0,
        tsumogiri: true,
        is_riichi_declare: true,
        is_double_riichi: true,
        turn_idx: 1,
        river_idx: 0,
    });
    log.push(RoundEvent::ReachAccepted {
        actor: 0,
        turn_idx: 1,
    });
    log.push(RoundEvent::Chi {
        actor: 1,
        source: 0,
        claimed_tile37: 0,
        consumed0_37: 4,
        consumed1_37: 6,
        turn_idx: 2,
        meld_idx: 0,
    });
    log.push(RoundEvent::Pon {
        actor: 2,
        source: 1,
        claimed_tile37: 15,
        consumed0_37: 10,
        consumed1_37: 15,
        turn_idx: 3,
        meld_idx: 0,
    });
    log.push(RoundEvent::Daiminkan {
        actor: 3,
        source: 2,
        claimed_tile37: 20,
        turn_idx: 4,
        meld_idx: 0,
    });
    log.push(RoundEvent::Dora {
        indicator_tile37: 30,
        dora_idx: 1,
        turn_idx: 4,
    });
    log.push(RoundEvent::Ankan {
        actor: 0,
        tile37: 5,
        involves_red5: true,
        turn_idx: 5,
        meld_idx: 0,
    });
    log.push(RoundEvent::TsumoPrivate {
        actor: 0,
        tile37: 31,
        is_rinshan: true,
        turn_idx: 5,
    });
    log.push(RoundEvent::Kakan {
        actor: 2,
        tile37: 15,
        turn_idx: 6,
        meld_idx: 0,
    });
    log.push(RoundEvent::SelfPassPrivate {
        actor: 0,
        source: 2,
        tile37: 15,
        pass_mask: PASS_RON | PASS_PON,
        turn_idx: 6,
    });
    // Opponent private pass MUST NOT appear in observer 0's stream
    log.push(RoundEvent::SelfPassPrivate {
        actor: 3,
        source: 2,
        tile37: 15,
        pass_mask: PASS_CHI | PASS_KAN,
        turn_idx: 6,
    });

    let expected = log.project_visible(0).unwrap();
    assert_eq!(expected.len(), 12); // 14 raw (initial dora + 13 pushed) minus 2 opponent-private events
    assert_eq!(expected[0].kind, EventKind::Dora);
    let encoded = encode_events(&log, 0).unwrap();
    let decoded = decode_events(&encoded).unwrap();
    assert_eq!(decoded, expected);
}

#[test]
fn events_10000_extreme_rounds_never_exceed_214_and_overflow_errors() {
    let mut rng = StdRng::seed_from_u64(0x2026_1122);
    let mut max_seen_len = 0usize;

    for _ in 0..10_000 {
        let mut log = RoundEventLog::new(1).unwrap();
        let mut live_draws_left = 70u32;
        let mut total_kans = 0u32;
        let mut melds_per_seat = [0u32; 4];
        let mut river_idx = [0u8; 4];
        let mut riichi_done = [false; 4];
        let mut seat = 0u8;
        let mut turn = 0u8;

        // Stress-generate legal maximal-meld / 4-kan / maximal-self-pass rounds
        while live_draws_left > 0 {
            turn = turn.saturating_add(1);
            live_draws_left -= 1;

            log.push(RoundEvent::TsumoPrivate {
                actor: seat,
                tile37: 5,
                is_rinshan: false,
                turn_idx: turn,
            });

            // Opportunistically perform Ankan or Kakan (up to 4 total kans)
            if total_kans < 4 && melds_per_seat[seat as usize] < 4 && turn.is_multiple_of(7) {
                total_kans += 1;
                let meld_slot = melds_per_seat[seat as usize] as u8;
                melds_per_seat[seat as usize] += 1;
                log.push(RoundEvent::Ankan {
                    actor: seat,
                    tile37: 5,
                    involves_red5: false,
                    turn_idx: turn,
                    meld_idx: meld_slot,
                });
                if seat != 0 {
                    // Worst case: observer passes Kokushi chankan on every opp ankan
                    log.push(RoundEvent::SelfPassPrivate {
                        actor: 0,
                        source: seat,
                        tile37: 5,
                        pass_mask: PASS_RON,
                        turn_idx: turn,
                    });
                }
                log.push(RoundEvent::Dora {
                    indicator_tile37: 2,
                    dora_idx: total_kans as u8,
                    turn_idx: turn,
                });
                // Rinshan draw
                log.push(RoundEvent::TsumoPrivate {
                    actor: seat,
                    tile37: 6,
                    is_rinshan: true,
                    turn_idx: turn,
                });
            }

            // Discard
            let do_riichi =
                !riichi_done[seat as usize] && melds_per_seat[seat as usize] == 0 && turn > 4;
            if do_riichi {
                riichi_done[seat as usize] = true;
            }
            log.push(RoundEvent::Dahai {
                actor: seat,
                tile37: 4,
                tsumogiri: false,
                is_riichi_declare: do_riichi,
                is_double_riichi: false,
                turn_idx: turn,
                river_idx: river_idx[seat as usize],
            });
            river_idx[seat as usize] = river_idx[seat as usize].saturating_add(1);

            // Worst case: observer passes on EVERY opponent discard
            if seat != 0 {
                log.push(RoundEvent::SelfPassPrivate {
                    actor: 0,
                    source: seat,
                    tile37: 4,
                    pass_mask: PASS_CHI | PASS_PON,
                    turn_idx: turn,
                });
            }
            if do_riichi {
                log.push(RoundEvent::ReachAccepted {
                    actor: seat,
                    turn_idx: turn,
                });
            }

            // Opportunistically make a non-kan CHI/PON call (maximizing discards up to 70+4+16=90)
            let next_seat = (seat + 1) % 4;
            if melds_per_seat[next_seat as usize] < 4
                && !riichi_done[next_seat as usize]
                && turn.is_multiple_of(3)
            {
                let meld_slot = melds_per_seat[next_seat as usize] as u8;
                melds_per_seat[next_seat as usize] += 1;
                log.push(RoundEvent::Chi {
                    actor: next_seat,
                    source: seat,
                    claimed_tile37: 4,
                    consumed0_37: 2,
                    consumed1_37: 3,
                    turn_idx: turn,
                    meld_idx: meld_slot,
                });
                // Post-call discard without consuming live wall
                log.push(RoundEvent::Dahai {
                    actor: next_seat,
                    tile37: 7,
                    tsumogiri: false,
                    is_riichi_declare: false,
                    is_double_riichi: false,
                    turn_idx: turn,
                    river_idx: river_idx[next_seat as usize],
                });
                river_idx[next_seat as usize] = river_idx[next_seat as usize].saturating_add(1);
                if next_seat != 0 {
                    log.push(RoundEvent::SelfPassPrivate {
                        actor: 0,
                        source: next_seat,
                        tile37: 7,
                        pass_mask: PASS_PON,
                        turn_idx: turn,
                    });
                }
                seat = (next_seat + 1) % 4;
            } else {
                seat = next_seat;
            }
            // Randomize turn progression occasionally to explore diverse seat paths
            if live_draws_left.is_multiple_of(19) {
                let mut seats = [0u8, 1, 2, 3];
                seats.shuffle(&mut rng);
                seat = seats[0];
            }
        }

        for observer in 0..4u8 {
            let enc = encode_events(&log, observer).expect("valid legal round must fit <= 256");
            assert!(!enc.is_empty());
            assert!(enc.len() <= 214, "exceeded proven bound 214: {}", enc.len());
            max_seen_len = max_seen_len.max(enc.len());
        }
    }
    assert!(
        max_seen_len >= 120,
        "fuzz should exercise long rounds, max={max_seen_len}"
    );

    // Hard error on overflow (> 256) — NEVER silently truncate!
    let mut bad_log = RoundEventLog::new(1).unwrap();
    for i in 0..=MAX_EVENT_LEN {
        bad_log.push(RoundEvent::Dahai {
            actor: (i % 4) as u8,
            tile37: 1,
            tsumogiri: true,
            is_riichi_declare: false,
            is_double_riichi: false,
            turn_idx: (i % 255) as u8,
            river_idx: 0,
        });
    }
    assert!(encode_events(&bad_log, 0).is_err());
}

// =========================================================================
// 6. P1-1, P1-3, P1-5 (v2.3/v2.4): 546 manifest, NULL slots, 111 DAIMINKAN
//    bijection, and 4-phase MAX_CANDIDATES = 32 extremes
// =========================================================================
#[test]
fn manifest_546_null_slots_and_all_111_daiminkan_bijection() {
    let table = manifest();
    assert_eq!(table.len(), ACTION_COUNT + 1);
    assert_eq!(table[ACTION_COUNT].kind, ManifestActionKind::Pad);

    // Check NULL_TILE / NULL_SOURCE on TSUMO(219), KYUSHU(220), PASS(221), RON(543..545)
    for id in [219u16, 220, 221] {
        let e = entry(id).unwrap();
        assert_eq!(e.target_base34, NULL_TILE_IDX);
        assert_eq!(e.consumed0_base34, NULL_TILE_IDX);
        assert_eq!(e.consumed1_base34, NULL_TILE_IDX);
        assert_eq!(e.source_rel, NULL_SOURCE_IDX);
        assert_eq!(e.has_slot_flags, [false, false, false, false]);
    }
    for src in 1..=3u8 {
        let e = entry(lookup_ron_id(src).unwrap()).unwrap();
        assert_eq!(e.target_base34, NULL_TILE_IDX);
        assert_eq!(e.source_rel, src);
        assert_eq!(e.has_slot_flags, [false, false, false, true]);
    }

    // Exhaustively test ALL 111 DAIMINKAN IDs (3 sources x 37 tile37 including red 5m/5p/5s)
    for src in 1..=3u8 {
        for t37 in 0..37u8 {
            let gid = lookup_daiminkan_id(src, t37).unwrap();
            let e = entry(gid).unwrap();
            assert_eq!(e.kind, ManifestActionKind::Daiminkan);
            assert_eq!(e.source_rel, src);
            assert_eq!(e.tile37, t37);

            // Construct physical claimed tile136 + remaining 3 copies in hand
            let base34 = e.target_base34;
            let claimed_t136 = if t37 == 0 {
                16
            } else if t37 == 10 {
                52
            } else if t37 == 20 {
                88
            } else {
                // Non-red physical copy (for 5s pick +1 so it is not red)
                if base34 == 4 || base34 == 13 || base34 == 22 {
                    base34 * 4 + 1
                } else {
                    base34 * 4
                }
            };
            let consume_3: Vec<u8> = (base34 * 4..base34 * 4 + 4)
                .filter(|&x| x != claimed_t136)
                .collect();
            assert_eq!(consume_3.len(), 3);

            let mut obs = empty_obs(0);
            obs.hands[0] = consume_3.iter().map(|&x| u32::from(x)).collect();
            obs.dora_indicators = vec![if base34 == 0 { 4 } else { 0 }];
            let native = Action::new(
                ActionType::Daiminkan,
                Some(claimed_t136),
                consume_3.clone(),
                Some(0),
            );
            obs._legal_actions = vec![
                native.clone(),
                Action::new(ActionType::Pass, None, vec![], Some(0)),
            ];

            let cands = resolve_legal_candidates(&obs, Some(src)).unwrap();
            assert_eq!(cands.len(), 2);
            assert_eq!(cands[0].global_id, 221); // PASS < DAIMINKAN (strictly ascending)
            assert_eq!(cands[1].global_id, gid);
            let recovered = decode_global_id_to_native(gid, &obs, Some(src)).unwrap();
            assert_eq!(recovered, vec![native]);
        }
    }

    // Spot check ANKAN(34), KAKAN(37), CHI(90), PON(120), DISCARD(148) ID lookups
    assert_eq!(lookup_discard_id(0, false, false).unwrap(), 0);
    assert_eq!(lookup_discard_id(36, true, true).unwrap(), 147);
    assert_eq!(lookup_ankan_id(0).unwrap(), 148);
    assert_eq!(lookup_ankan_id(33).unwrap(), 181);
    assert_eq!(lookup_kakan_id(0).unwrap(), 182);
    assert_eq!(lookup_kakan_id(36).unwrap(), 218);
    assert_eq!(lookup_chi_id(1, 2, 3).unwrap(), 222);
    assert_eq!(lookup_pon_id(1, 1, 1, 1).unwrap(), 312);
}

#[test]
fn max_candidates_32_extreme_hands_across_all_four_phases() {
    let tracker = HanchanTracker::default();
    let rc = RuleContract::default_tenhou_houou();

    // 1) OWN_ACT Extreme: Turn-1 uninterrupted Kokushi 13-wait (14 discards + 14 riichi + TSUMO + KYUSHU = 30 <= 32)
    let mut obs_own = empty_obs(0);
    let kokushi_14: Vec<u8> = vec![
        0, 1, 32, 36, 68, 72, 104, // 1m(drawn=1), 1m(hand=0), 9m, 1p, 9p, 1s, 9s
        108, 112, 116, 120, 124, 128, 132, // E S W N P F C
    ];
    obs_own.hands[0] = kokushi_14.iter().map(|&x| u32::from(x)).collect();
    obs_own.drawn_tile = Some(1);
    obs_own.dora_indicators = vec![4]; // 2m
    let mut legal_own: Vec<Action> = kokushi_14
        .iter()
        .map(|&t| Action::new(ActionType::Discard, Some(t), vec![], Some(0)))
        .collect();
    legal_own.push(Action::new(ActionType::Riichi, None, vec![], Some(0)));
    legal_own.push(Action::new(ActionType::Tsumo, Some(1), vec![], Some(0)));
    legal_own.push(Action::new(ActionType::KyushuKyuhai, None, vec![], Some(0)));
    obs_own._legal_actions = legal_own;

    let vis_own = VisibleTileCounts::from_observation(&obs_own).unwrap();
    let ctx_own = RoundGlobalContext {
        phase: DecisionPhase::OwnAct,
        live_wall_left: 69,
        is_first_uninterrupted_turn: true,
        ..Default::default()
    };
    let enc_own = encode_candidates(
        &obs_own,
        &vis_own,
        FuritenState::default(),
        &tracker,
        &rc,
        &ctx_own,
        &WinEvalContext {
            is_tsumo_first_turn: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(enc_own.num_valid, 30);
    assert!(enc_own.num_valid <= MAX_CANDIDATES);
    // Verify strictly ascending cand_ids and distinct modes (POST_DISCARD=1, WIN_EVAL=5, KYUSHU=6)
    for i in 1..enc_own.num_valid {
        assert!(enc_own.cand_ids[i - 1] < enc_own.cand_ids[i]);
    }
    let idx_tsumo = enc_own.cand_ids.iter().position(|&id| id == 219).unwrap();
    let idx_kyushu = enc_own.cand_ids.iter().position(|&id| id == 220).unwrap();
    assert_eq!(enc_own.cand_mode[idx_tsumo], CandidateMode::WinEval as u8);
    assert_eq!(
        enc_own.cand_mode[idx_kyushu],
        CandidateMode::KyushuDrawEval as u8
    );
    assert!(enc_own.cand_dyn[idx_tsumo][3] == 1.0); // yakuman!

    // 2) DISCARD_RESP Extreme: Kamicha discards 4m(12); observer holds 2m,3m,4m,4m,4m,0m,5m,6m + winning shape
    // => PASS(1) + RON(1) + DAIMINKAN(1) + PON(1) + CHI(5: 23, 35, 30, 56, 06) = 9 (or with 5m claimed -> PON 2, CHI 3 = 8; max across all tiles <= 10!)
    let mut obs_resp = empty_obs(0);
    // 2m(4), 3m(8), 4m(13,14,15), 0m(16), 5m(17), 6m(20), 7p(60,61,62), 9s(104,105) -> 13 tiles
    obs_resp.hands[0] = vec![4, 8, 13, 14, 15, 16, 17, 20, 60, 61, 62, 104, 105];
    obs_resp.discards[3] = vec![12]; // Kamicha discarded 4m(12)
    obs_resp.last_discard = Some(12);
    obs_resp.dora_indicators = vec![0];
    obs_resp._legal_actions = vec![
        Action::new(ActionType::Pass, None, vec![], Some(0)),
        Action::new(ActionType::Ron, Some(12), vec![], Some(0)),
        Action::new(ActionType::Daiminkan, Some(12), vec![13, 14, 15], Some(0)),
        Action::new(ActionType::Pon, Some(12), vec![13, 14], Some(0)),
        Action::new(ActionType::Chi, Some(12), vec![4, 8], Some(0)),
        Action::new(ActionType::Chi, Some(12), vec![8, 17], Some(0)),
        Action::new(ActionType::Chi, Some(12), vec![8, 16], Some(0)),
        Action::new(ActionType::Chi, Some(12), vec![17, 20], Some(0)),
        Action::new(ActionType::Chi, Some(12), vec![16, 20], Some(0)),
    ];
    let vis_resp = VisibleTileCounts::from_observation(&obs_resp).unwrap();
    let ctx_resp = RoundGlobalContext {
        phase: DecisionPhase::DiscardResp,
        trigger_actor_rel: Some(3),
        live_wall_left: 50,
        ..Default::default()
    };
    let enc_resp = encode_candidates(
        &obs_resp,
        &vis_resp,
        FuritenState::default(),
        &tracker,
        &rc,
        &ctx_resp,
        &WinEvalContext::default(),
    )
    .unwrap();
    assert_eq!(enc_resp.num_valid, 9);
    assert!(enc_resp.num_valid <= 10);
    // Verify CHI with (4,8) (2m,3m claiming 4m) sets kuikae blocked ratio > 0 (1m/4m forbidden, and we hold three 4m!)
    let chi_23_id = lookup_chi_id(4, 2, 3).unwrap() as i16;
    let idx_chi23 = enc_resp
        .cand_ids
        .iter()
        .position(|&id| id == chi_23_id)
        .unwrap();
    assert_eq!(
        enc_resp.cand_mode[idx_chi23],
        CandidateMode::PreDiscardMeldEnvelope as u8
    );
    assert!(enc_resp.cand_dyn[idx_chi23][7] > 0.0); // kuikae blocked 3 copies of 4m!

    // 3) KAKAN_RESP & ANKAN_RESP: max 2 candidates (PASS + RON)
    let mut obs_kan = empty_obs(0);
    obs_kan.hands[0] = vec![0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 112, 113];
    obs_kan.last_discard = Some(114);
    obs_kan._legal_actions = vec![
        Action::new(ActionType::Pass, None, vec![], Some(0)),
        Action::new(ActionType::Ron, Some(114), vec![], Some(0)),
    ];
    let cands_kan = resolve_legal_candidates(&obs_kan, Some(1)).unwrap();
    assert_eq!(cands_kan.len(), 2);
}

// =========================================================================
// 7. P0-1/P0-2 (v2.4): Oracle permutation-invariance & Actor zero-leakage
// =========================================================================
#[test]
fn oracle_live_wall_is_permutation_invariant_and_actor_zero_leakage() {
    let mut state = GameState::new(0, true, Some(42), 0, GameRule::default_tenhou());
    let obs_before = state.get_observation(0);
    let vis_before = VisibleTileCounts::from_observation(&obs_before).unwrap();
    let fur_before = FuritenState::from_game_state(&state, 0);
    let tracker = HanchanTracker::default();
    let rc = RuleContract::default_tenhou_houou();
    let ctx = RoundGlobalContext {
        phase: DecisionPhase::OwnAct,
        live_wall_left: state.wall.drawable_count,
        is_first_uninterrupted_turn: state.is_first_turn,
        ..Default::default()
    };

    let actor_tile_before = encode_tile_feats(
        &obs_before,
        &vis_before,
        fur_before,
        &state.rule,
        DecisionPhase::OwnAct,
        &V2FeatureConfig::default(),
    )
    .unwrap();
    let (oracle_tile_before, oracle_player_before) = encode_oracle(&state, 0).unwrap();

    // Shuffle live wall order (indices 14..) 50 times -> BOTH Oracle and Actor MUST be bit-exact invariant!
    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
    for _ in 0..50 {
        if state.wall.tiles.len() > 14 {
            state.wall.tiles[14..].shuffle(&mut rng);
        }
        let (ot, op) = encode_oracle(&state, 0).unwrap();
        assert_eq!(
            ot, oracle_tile_before,
            "Oracle must be live-wall permutation invariant"
        );
        assert_eq!(op, oracle_player_before);
    }

    // Swap concealed hands between opponents 1 & 2 AND swap dead-wall uradora -> Actor MUST be bit-exact invariant!
    let tmp_hand = state.players[1].hand.clone();
    state.players[1].hand = state.players[2].hand.clone();
    state.players[2].hand = tmp_hand;
    if state.wall.tiles.len() > 7 {
        state.wall.tiles.swap(5, 7);
    }
    let obs_after = state.get_observation(0);
    let vis_after = VisibleTileCounts::from_observation(&obs_after).unwrap();
    let actor_tile_after = encode_tile_feats(
        &obs_after,
        &vis_after,
        fur_before,
        &state.rule,
        DecisionPhase::OwnAct,
        &V2FeatureConfig::default(),
    )
    .unwrap();
    assert_eq!(
        actor_tile_before, actor_tile_after,
        "Actor features MUST be bit-exact invariant to opponent hands & uradora"
    );

    // Confirm global/player/candidate encoders also produce zero NaN/Inf on live GameState
    let (g, p) = encode_global_player(&obs_after, fur_before, &tracker, &rc, &ctx).unwrap();
    let c = encode_candidates(
        &obs_after,
        &vis_after,
        fur_before,
        &tracker,
        &rc,
        &ctx,
        &WinEvalContext::default(),
    )
    .unwrap();
    assert!(g.iter().all(|x| x.is_finite()));
    assert!(p.iter().flatten().all(|x| x.is_finite()));
    assert!(c.cand_dyn.iter().flatten().all(|x| x.is_finite()));
    assert!(c.num_valid >= 1 && c.num_valid <= MAX_CANDIDATES);
}

// =========================================================================
// 8. End-to-end live multi-round simulation: physical conservation + HanchanTracker
// =========================================================================
#[test]
fn live_simulation_enforces_conservation_bounds_and_hanchan_tracker() {
    let mut state = GameState::new(0, true, Some(12345), 0, GameRule::default_tenhou());
    let mut tracker = HanchanTracker::default();
    let rc = RuleContract::default_tenhou_houou();
    let mut steps = 0usize;

    while !state.is_done && steps < 400 {
        let active: Vec<u8> = if state.phase == crate::action::Phase::WaitAct {
            vec![state.current_player]
        } else {
            state.active_players.clone()
        };
        let mut chosen_map = std::collections::HashMap::new();

        for pid in active {
            let obs = state.get_observation(pid);
            let vis = VisibleTileCounts::from_observation(&obs).unwrap();
            for b in 0..TILE_MAX {
                assert_eq!(
                    vis.self_hand_base34[b] + vis.visible_base34[b] + vis.unseen_base34[b],
                    4
                );
            }
            let fur = FuritenState::from_game_state(&state, pid);
            let (phase, trigger_rel) = if state.phase == crate::action::Phase::WaitAct {
                (DecisionPhase::OwnAct, None)
            } else {
                let src = state
                    .last_discard
                    .map(|(dp, _)| (dp + 4 - pid) % 4)
                    .or_else(|| state.pending_kan.as_ref().map(|(kp, _)| (kp + 4 - pid) % 4));
                (DecisionPhase::DiscardResp, src)
            };
            let ctx = RoundGlobalContext {
                phase,
                trigger_actor_rel: trigger_rel,
                live_wall_left: state.wall.drawable_count,
                is_first_uninterrupted_turn: state.is_first_turn,
                ..Default::default()
            };

            let tf = encode_tile_feats(
                &obs,
                &vis,
                fur,
                &state.rule,
                phase,
                &V2FeatureConfig::default(),
            )
            .unwrap();
            let (gf, pf) = encode_global_player(&obs, fur, &tracker, &rc, &ctx).unwrap();
            let cand = encode_candidates(
                &obs,
                &vis,
                fur,
                &tracker,
                &rc,
                &ctx,
                &WinEvalContext::default(),
            )
            .unwrap();
            let (ot, op) = encode_oracle(&state, pid).unwrap();

            assert!(tf.iter().all(|x| x.is_finite()));
            assert!(gf.iter().all(|x| x.is_finite()));
            assert!(pf.iter().flatten().all(|x| x.is_finite()));
            assert!(ot.iter().all(|x| x.is_finite()));
            assert!(op.iter().flatten().all(|x| x.is_finite()));
            assert!(cand.num_valid >= 1 && cand.num_valid <= MAX_CANDIDATES);

            // Pick a legal native action (prefer Pass or first Discard) to advance state cleanly
            let legal = obs.legal_actions_method();
            let act = legal
                .iter()
                .find(|a| a.action_type == ActionType::Pass)
                .or_else(|| legal.iter().find(|a| a.action_type == ActionType::Discard))
                .unwrap_or(&legal[0])
                .clone();
            chosen_map.insert(pid, act);
        }

        let prev_kyoku = (state.round_wind, state.kyoku_idx, state.honba);
        state.step(&chosen_map);
        let next_kyoku = (state.round_wind, state.kyoku_idx, state.honba);
        if next_kyoku != prev_kyoku {
            tracker.record_round_end(&RoundOutcomeSummary::default());
        }
        steps += 1;
    }
    assert!(
        steps > 50,
        "expected real multi-turn simulation, ran {steps} steps"
    );
}

// =========================================================================
// 9. P1-5: Cross-parity check against legacy 215-ch `encode_base_into`
//    (ONLY on semantically identical public channels per Section 4 table)
// =========================================================================
#[cfg(feature = "python")]
#[test]
fn tile_feats_public_channels_zero_error_parity_with_legacy_encode_base() {
    let mut state = GameState::new(0, true, Some(777), 0, GameRule::default_tenhou());
    let obs = state.get_observation(0);
    let vis = VisibleTileCounts::from_observation(&obs).unwrap();
    let fur = FuritenState::from_game_state(&state, 0);
    let tf = encode_tile_feats(
        &obs,
        &vis,
        fur,
        &state.rule,
        DecisionPhase::OwnAct,
        &V2FeatureConfig::default(),
    )
    .unwrap();

    let mut legacy = vec![0.0f32; 215 * 34];
    obs.encode_base_into(&mut legacy, 0);

    // Channels 0..4 in V2 (`self_closed_cnt_ge{1..4}`, `self_closed_has_red5`)
    // MUST match legacy `encode_base_into` ch 0..4 with ZERO error across all 34 tiles!
    for ch in 0..5 {
        for b in 0..TILE_MAX {
            let v2_val = tf[b * TILE_FEATS_DIM + ch];
            let legacy_val = legacy[ch * TILE_MAX + b];
            assert_eq!(
                v2_val, legacy_val,
                "parity mismatch on ch={ch}, base34={b}: v2={v2_val}, legacy={legacy_val}"
            );
        }
    }
}

// =========================================================================
// 10. Zero Train-Serve Skew Parity Test: Live `GameState::step` vs MJAI Replay
// =========================================================================
#[test]
fn live_vs_mjai_replay_zero_train_serve_skew_parity() {
    use super::{CAND_DYN_DIM, EVENT_BYTES, GLOBAL_FEATS_DIM, PLAYER_FEATS_DIM, TILE_DIM};
    use crate::replay::MjaiEvent;
    use std::collections::HashMap;

    struct Snapshot {
        mjai_len: usize,
        observer: u8,
        tile_feats: [f32; TILE_DIM * TILE_FEATS_DIM],
        global_feats: [f32; GLOBAL_FEATS_DIM],
        player_feats: [[f32; PLAYER_FEATS_DIM]; 4],
        cand_ids: [i16; MAX_CANDIDATES],
        cand_mask: [bool; MAX_CANDIDATES],
        cand_mode: [u8; MAX_CANDIDATES],
        cand_dyn: [[f32; CAND_DYN_DIM]; MAX_CANDIDATES],
        events: Vec<[u8; EVENT_BYTES]>,
        tracker: HanchanTracker,
    }

    let rc = RuleContract::default_tenhou_houou();
    let cfg = V2FeatureConfig::default();

    let encode_snapshot = |state: &mut GameState, observer: u8| -> Snapshot {
        let obs = state.get_observation(observer);
        let vis = VisibleTileCounts::from_observation(&obs).unwrap();
        let fur = FuritenState::from_game_state(state, observer);
        let ctx = state.v2_round_global_context(observer);
        let tf = encode_tile_feats(&obs, &vis, fur, &state.rule, ctx.phase, &cfg).unwrap();
        let (gf, pf) =
            encode_global_player(&obs, fur, &state.v2_hanchan_tracker, &rc, &ctx).unwrap();
        let cand = encode_candidates(
            &obs,
            &vis,
            fur,
            &state.v2_hanchan_tracker,
            &rc,
            &ctx,
            &WinEvalContext {
                is_rinshan: state.is_rinshan_flag,
                is_chankan: matches!(
                    ctx.phase,
                    DecisionPhase::KakanResp | DecisionPhase::AnkanResp
                ),
                is_haitei_or_houtei: state.wall.drawable_count == 0,
                is_tsumo_first_turn: state.is_first_turn,
                pao_payer: state.players[observer as usize]
                    .pao
                    .values()
                    .next()
                    .copied(),
            },
        )
        .unwrap();
        let events = state.v2_encode_round_events(observer).unwrap();
        Snapshot {
            mjai_len: state.mjai_log.len(),
            observer,
            tile_feats: tf,
            global_feats: gf,
            player_feats: pf,
            cand_ids: cand.cand_ids,
            cand_mask: cand.cand_mask,
            cand_mode: cand.cand_mode,
            cand_dyn: cand.cand_dyn,
            events,
            tracker: state.v2_hanchan_tracker.clone(),
        }
    };

    // Play a live 4P Hanchan game with MJAI logging enabled across >= 2 rounds,
    // exercising Riichi, Pon/Chi, Pass, Tsumo/Ron, and cross-round HanchanTracker updates.
    let mut live = GameState::new(2, false, Some(20261122), 0, GameRule::default_tenhou());
    let mut snapshots: Vec<Snapshot> = Vec::new();
    let mut saw_call = false;
    let mut saw_riichi = false;
    let mut saw_pass = false;
    let mut steps = 0usize;

    while !live.is_done && steps < 450 {
        if live.v2_hanchan_tracker.rounds_completed >= 2 && steps > 120 {
            break;
        }
        let active = live.active_players.clone();
        if active.is_empty() {
            break;
        }
        let mut chosen_map = HashMap::new();
        for &pid in &active {
            snapshots.push(encode_snapshot(&mut live, pid));
            let obs = live.get_observation(pid);
            let legal = obs.legal_actions_method();
            // Deterministic policy exercising Tsumo/Ron > Riichi > occasional Pon/Chi > Pass > Discard
            let act = legal
                .iter()
                .find(|a| matches!(a.action_type, ActionType::Tsumo | ActionType::Ron))
                .or_else(|| {
                    legal.iter().find(|a| {
                        if a.action_type == ActionType::Riichi {
                            saw_riichi = true;
                            true
                        } else {
                            false
                        }
                    })
                })
                .or_else(|| {
                    if !saw_call && steps.is_multiple_of(3) {
                        legal.iter().find(|a| {
                            if matches!(a.action_type, ActionType::Pon | ActionType::Chi) {
                                saw_call = true;
                                true
                            } else {
                                false
                            }
                        })
                    } else {
                        None
                    }
                })
                .or_else(|| {
                    legal.iter().find(|a| {
                        if a.action_type == ActionType::Pass {
                            saw_pass = true;
                            true
                        } else {
                            false
                        }
                    })
                })
                .or_else(|| legal.iter().find(|a| a.action_type == ActionType::Discard))
                .unwrap_or(&legal[0])
                .clone();
            chosen_map.insert(pid, act);
        }
        live.step(&chosen_map);
        steps += 1;
    }

    assert!(
        live.v2_hanchan_tracker.rounds_completed >= 1,
        "expected >= 1 completed round to verify HanchanTracker parity"
    );
    assert!(
        !snapshots.is_empty() && saw_pass,
        "expected multi-step snapshots with WaitResponse passes"
    );
    let _ = (saw_call, saw_riichi);

    // Replay the exact MJAI log into a fresh GameState and verify ZERO train-serve skew
    // at EVERY single captured decision snapshot!
    let mut replay = GameState::new(2, true, Some(99999), 0, GameRule::default_tenhou());
    let mut snap_idx = 0usize;
    for (ev_i, raw_json) in live.mjai_log.iter().enumerate() {
        let ev: MjaiEvent = serde_json::from_str(raw_json)
            .unwrap_or_else(|e| panic!("failed to parse mjai_log[{ev_i}]={raw_json}: {e}"));
        replay.apply_mjai_event(ev);
        assert!(
            replay.last_error.is_none(),
            "replay error at mjai_log[{ev_i}]={raw_json}: {:?}",
            replay.last_error
        );
        let cur_len = ev_i + 1;
        while snap_idx < snapshots.len() && snapshots[snap_idx].mjai_len == cur_len {
            let expected = &snapshots[snap_idx];
            let actual = encode_snapshot(&mut replay, expected.observer);

            assert_eq!(
                actual.tracker, expected.tracker,
                "HanchanTracker skew at snap #{snap_idx} (mjai_len={cur_len}, obs={})",
                expected.observer
            );
            assert_eq!(
                actual.events, expected.events,
                "v2_round_events skew at snap #{snap_idx} (mjai_len={cur_len}, obs={})",
                expected.observer
            );
            assert_eq!(
                actual.tile_feats, expected.tile_feats,
                "tile_feats skew at snap #{snap_idx} (mjai_len={cur_len}, obs={})",
                expected.observer
            );
            assert_eq!(
                actual.global_feats, expected.global_feats,
                "global_feats skew at snap #{snap_idx} (mjai_len={cur_len}, obs={})",
                expected.observer
            );
            assert_eq!(
                actual.player_feats, expected.player_feats,
                "player_feats skew at snap #{snap_idx} (mjai_len={cur_len}, obs={})",
                expected.observer
            );
            assert_eq!(
                actual.cand_ids, expected.cand_ids,
                "cand_ids skew at snap #{snap_idx} (mjai_len={cur_len}, obs={})",
                expected.observer
            );
            assert_eq!(
                actual.cand_mask, expected.cand_mask,
                "cand_mask skew at snap #{snap_idx} (mjai_len={cur_len}, obs={})",
                expected.observer
            );
            assert_eq!(
                actual.cand_mode, expected.cand_mode,
                "cand_mode skew at snap #{snap_idx} (mjai_len={cur_len}, obs={})",
                expected.observer
            );
            assert_eq!(
                actual.cand_dyn, expected.cand_dyn,
                "cand_dyn skew at snap #{snap_idx} (mjai_len={cur_len}, obs={})",
                expected.observer
            );
            snap_idx += 1;
        }
    }
    assert_eq!(
        snap_idx,
        snapshots.len(),
        "every single live snapshot must have been verified during MJAI replay"
    );
}

// =========================================================================
// 11. Post-Kan Dead-Wall Index Shift & Stage-3 BeliefTargets Exactness
// =========================================================================
#[test]
fn oracle_post_kan_dead_wall_shift_and_belief_targets_exactness() {
    use super::ORACLE_TILE_DIM;
    use super::oracle::encode_belief_targets;

    let mut state = GameState::new(0, true, Some(42), 0, GameRule::default_tenhou());
    // Simulate 1 rinshan draw (`wall.tiles.remove(0)`):
    // Dead wall now occupies indices 0..13:
    //   0..3 = remaining 3 rinshan tiles
    //   3, 5 = omote dora indicators 0 and 1 (`(4 + 2*i) - 1`)
    //   4, 6 = ura dora indicators 0 and 1 (`(5 + 2*i) - 1`)
    //   13.. = live wall
    state.wall.rinshan_draw_count = 1;
    let mut custom_wall = vec![0u8; 30];
    // Remaining rinshan tiles (indices 0..3): three copies of East (base34=27 -> 108,109,110)
    custom_wall[0] = 108;
    custom_wall[1] = 109;
    custom_wall[2] = 110;
    // Dora indicators at 3 and 5: 1m (0) and 2m (4)
    custom_wall[3] = 0;
    custom_wall[5] = 4;
    // Uradora indicators at 4 and 6: 9s (base34=26 -> dora=1s=18) and White (base34=31 -> dora=Green=32)
    custom_wall[4] = 26 * 4;
    custom_wall[6] = 31 * 4;
    // Live wall (13..30): put two copies of Red dragon (base34=33 -> 132,133) + red 5m (16)
    for slot in &mut custom_wall[13..30] {
        *slot = 8 * 4; // 9m filler
    }
    custom_wall[13] = 132;
    custom_wall[14] = 133;
    custom_wall[15] = 16; // red 5m
    state.wall.tiles = custom_wall;
    state.wall.dora_indicators = vec![0, 4];

    // Configure opponent 1 (rel=1 from observer 0):
    // Closed tenpai hand: 222m 456m 678m 23p 55s -> waits on 1p (9, NO yaku: 222m breaks Pinfu)
    // and 4p (12, Tanyao!).
    state.players[1].hand = vec![
        4,
        5,
        6,
        3 * 4 + 1,
        4 * 4 + 1,
        5 * 4,
        5 * 4 + 1,
        6 * 4,
        7 * 4,
        10 * 4,
        11 * 4,
        22 * 4 + 1,
        22 * 4 + 2,
    ];
    state.players[1].melds.clear();
    state.players[1].discards.clear();
    state.players[1].riichi_declared = false;

    // Configure opponent 2 (rel=2): tenpai on 4p (12) & 1p (9) too, BUT furiten (discarded 1p)!
    state.players[2].hand = state.players[1].hand.clone();
    state.players[2].melds.clear();
    state.players[2].discards = vec![9 * 4]; // discarded 1p -> permanent furiten!
    state.players[2].riichi_declared = false;

    // Configure opponent 3 (rel=3): non-tenpai hand
    state.players[3].hand = vec![0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 112, 116, 120];
    state.players[3].melds.clear();
    state.players[3].discards.clear();

    let (ot1, _) = encode_oracle(&state, 0).unwrap();
    // Uradora hits: 1s (base34=18) and Green (base34=32) each have count 1 -> 0.25
    assert!((ot1[18 * ORACLE_TILE_DIM + 20] - 0.25).abs() < 1e-6);
    assert!((ot1[32 * ORACLE_TILE_DIM + 20] - 0.25).abs() < 1e-6);
    // Remaining 3 rinshan tiles: East (base34=27) has count 3 -> 0.75
    assert!((ot1[27 * ORACLE_TILE_DIM + 21] - 0.75).abs() < 1e-6);
    // Live wall: Red dragon (base34=33) has 2 copies -> ch15=1, ch16=1, ch17=0; red 5m -> ch19=1 on base34=4
    assert_eq!(ot1[33 * ORACLE_TILE_DIM + 15], 1.0);
    assert_eq!(ot1[33 * ORACLE_TILE_DIM + 16], 1.0);
    assert_eq!(ot1[33 * ORACLE_TILE_DIM + 17], 0.0);
    assert_eq!(ot1[4 * ORACLE_TILE_DIM + 19], 1.0);

    // Shuffling live wall `[13..]` must produce bit-identical oracle_tile!
    let mut state2 = state.clone();
    state2.wall.tiles[13..].reverse();
    let (ot2, _) = encode_oracle(&state2, 0).unwrap();
    assert_eq!(ot1, ot2);

    // Stage-3 BeliefTargets exactness:
    let bt = encode_belief_targets(&state, 0).unwrap();
    assert_eq!(bt.opp_tenpai, [1, 1, 0]);
    // Opp 1 waits on 1p (9) and 4p (12)
    assert_eq!(bt.opp_waits[0][9], 1);
    assert_eq!(bt.opp_waits[0][12], 1);
    // 4p deals into Opp 1 (Tanyao), while 1p has NO yaku (Opp 1 not riichi, Opp 2 is furiten) -> 0!
    assert_eq!(bt.exact_dealin_now[12], 1);
    assert_eq!(bt.exact_dealin_now[9], 0);
    // Hand count sums match 13 per opponent
    for rel in 0..3 {
        let sum: u32 = bt.opp_hand_counts[rel].iter().map(|&c| u32::from(c)).sum();
        assert_eq!(sum, 13);
    }
}

#[test]
fn tile_feats_and_cand_dyn_match_bit_for_bit_across_furiten_and_red5_scenarios() {
    let tracker = HanchanTracker::default();
    let rc_tenhou = RuleContract::default_tenhou_houou();
    let rc_mjs = RuleContract::default_mjsoul_throne();

    // Hand (14 tiles): 1m 2m 3m | 4m 0m(16) 5m(17) 6m | 1p 1p 1p | 2s 3s | 9s 9s
    // Discarding 5m or 0m leaves 123m 456m 111p 23s 99s -> tenpai waiting on 1s(18), 4s(21).
    // Own river already contains 1s(72), so 1s/4s tenpai is permanent-furiten,
    // while discarding 4m leaves 123m 556m 111p 23s 99s (1-shanten, not tenpai).
    let mut obs = empty_obs(0);
    let hand_14: Vec<u8> = vec![
        0, 4, 8, // 1m 2m 3m
        12, 16, 17, 20, // 4m, 0m(red), 5m(normal), 6m
        36, 37, 38, // 1p 1p 1p
        76, 80, // 2s 3s
        104, 105, // 9s 9s
    ];
    obs.hands[0] = hand_14.iter().map(|&t| u32::from(t)).collect();
    obs.drawn_tile = Some(17); // drew normal 5m
    obs.discards[0] = vec![72]; // discarded 1s earlier -> permanent furiten on 1s/4s wait!
    obs.dora_indicators = vec![108]; // East -> South is dora
    obs._legal_actions = hand_14
        .iter()
        .map(|&t| Action::new(ActionType::Discard, Some(t), vec![], Some(0)))
        .collect();
    obs._legal_actions
        .push(Action::new(ActionType::Riichi, None, vec![], Some(0)));

    let vis = VisibleTileCounts::from_observation(&obs).unwrap();
    let ctx = RoundGlobalContext {
        phase: DecisionPhase::OwnAct,
        live_wall_left: 50,
        ..Default::default()
    };

    // Check across all 3 non-river furiten modes: clean, temporary-only, and riichi!
    for fur in [
        FuritenState {
            permanent: false,
            temporary: false,
            riichi: false,
        },
        FuritenState {
            permanent: false,
            temporary: true,
            riichi: false,
        },
        FuritenState {
            permanent: false,
            temporary: false,
            riichi: true,
        },
    ] {
        for rc in [&rc_tenhou, &rc_mjs] {
            let tf = encode_tile_feats(
                &obs,
                &vis,
                fur,
                &rc.rule,
                DecisionPhase::OwnAct,
                &V2FeatureConfig::default(),
            )
            .unwrap();
            let enc = encode_candidates(
                &obs,
                &vis,
                fur,
                &tracker,
                rc,
                &ctx,
                &WinEvalContext::default(),
            )
            .unwrap();

            // Strictly ascending cand_ids and bijective manifest decode
            for slot in 0..enc.num_valid {
                assert!(enc.cand_mask[slot]);
                if slot > 0 {
                    assert!(enc.cand_ids[slot - 1] < enc.cand_ids[slot]);
                }
                let gid = enc.cand_ids[slot] as u16;
                let decoded = decode_global_id_to_native(gid, &obs, None).unwrap();
                assert!(!decoded.is_empty());

                let entry = super::action_manifest::entry(gid).unwrap();
                if entry.kind == super::action_manifest::ManifestActionKind::Discard {
                    let b = entry.target_base34 as usize;
                    assert_eq!(enc.cand_mode[slot], CandidateMode::PostDiscardExact as u8);
                    // Exact bit-for-bit parity between tile_feats [57..62] and cand_dyn [0,2,3,4,5]
                    assert_eq!(tf[b * TILE_FEATS_DIM + 57], enc.cand_dyn[slot][0]);
                    assert_eq!(tf[b * TILE_FEATS_DIM + 58], enc.cand_dyn[slot][2]);
                    assert_eq!(tf[b * TILE_FEATS_DIM + 59], enc.cand_dyn[slot][3]);
                    assert_eq!(tf[b * TILE_FEATS_DIM + 60], enc.cand_dyn[slot][4]);
                    assert_eq!(tf[b * TILE_FEATS_DIM + 61], enc.cand_dyn[slot][5]);
                }
            }

            // Red 5m (tile37=0) vs normal 5m (tile37=5) have distinct IDs and distinct post_dora!
            let id_red5_dama = lookup_discard_id(0, false, false).unwrap() as i16;
            let id_norm5_tsumo = lookup_discard_id(5, true, false).unwrap() as i16;
            let id_norm5_riichi = lookup_discard_id(5, true, true).unwrap() as i16;
            let slot_red = enc
                .cand_ids
                .iter()
                .position(|&id| id == id_red5_dama)
                .unwrap();
            let slot_norm = enc
                .cand_ids
                .iter()
                .position(|&id| id == id_norm5_tsumo)
                .unwrap();
            let slot_riichi = enc
                .cand_ids
                .iter()
                .position(|&id| id == id_norm5_riichi)
                .unwrap();
            assert!(enc.cand_dyn[slot_red][6] < enc.cand_dyn[slot_norm][6]);
            assert_eq!(enc.cand_dyn[slot_norm][4], 1.0); // tenpai
            assert_eq!(enc.cand_dyn[slot_riichi][4], 1.0); // riichi tenpai
            assert_eq!(enc.cand_dyn[slot_norm][5], 1.0); // permanent furiten (1s in own river)
        }
    }

    // Remove 1s from own river: now temporary furiten MUST clear on own discard,
    // while riichi furiten MUST persist in BOTH tile_feats and cand_dyn!
    obs.discards[0].clear();
    let vis_clean = VisibleTileCounts::from_observation(&obs).unwrap();
    let tf_temp = encode_tile_feats(
        &obs,
        &vis_clean,
        FuritenState {
            permanent: false,
            temporary: true,
            riichi: false,
        },
        &rc_tenhou.rule,
        DecisionPhase::OwnAct,
        &V2FeatureConfig::default(),
    )
    .unwrap();
    let enc_temp = encode_candidates(
        &obs,
        &vis_clean,
        FuritenState {
            permanent: false,
            temporary: true,
            riichi: false,
        },
        &tracker,
        &rc_tenhou,
        &ctx,
        &WinEvalContext::default(),
    )
    .unwrap();
    let id_norm5_tsumo = lookup_discard_id(5, true, false).unwrap() as i16;
    let slot_norm = enc_temp
        .cand_ids
        .iter()
        .position(|&id| id == id_norm5_tsumo)
        .unwrap();
    assert_eq!(tf_temp[4 * TILE_FEATS_DIM + 61], 0.0);
    assert_eq!(enc_temp.cand_dyn[slot_norm][5], 0.0);

    let tf_riichi = encode_tile_feats(
        &obs,
        &vis_clean,
        FuritenState {
            permanent: false,
            temporary: false,
            riichi: true,
        },
        &rc_tenhou.rule,
        DecisionPhase::OwnAct,
        &V2FeatureConfig::default(),
    )
    .unwrap();
    let enc_riichi = encode_candidates(
        &obs,
        &vis_clean,
        FuritenState {
            permanent: false,
            temporary: false,
            riichi: true,
        },
        &tracker,
        &rc_tenhou,
        &ctx,
        &WinEvalContext::default(),
    )
    .unwrap();
    let slot_norm_r = enc_riichi
        .cand_ids
        .iter()
        .position(|&id| id == id_norm5_tsumo)
        .unwrap();
    assert_eq!(tf_riichi[4 * TILE_FEATS_DIM + 61], 1.0);
    assert_eq!(enc_riichi.cand_dyn[slot_norm_r][5], 1.0);
}
