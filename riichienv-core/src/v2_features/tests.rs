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
    obs_kakan.melds[0].push(Meld::new(MeldType::Kakan, vec![40, 41, 42, 43], true, 2, None));
    let vis_kakan = VisibleTileCounts::from_observation(&obs_kakan).unwrap();
    assert_eq!(vis_kakan.visible_base34[10], 4);

    // Orphan called_tile absent from rivers MUST be rejected
    let mut obs_orphan = empty_obs(0);
    obs_orphan.dora_indicators.clear();
    obs_orphan.melds[0].push(Meld::new(MeldType::Pon, vec![40, 41, 42], true, 2, Some(40)));
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

    // Now clear own river, test temporary furiten persisting across CHI/PON into post-discard!
    obs.discards[0].clear();
    let f_temp = FuritenState::from_observation(&obs, true, false, &rule);
    assert!(!f_temp.permanent);
    assert!(f_temp.temporary);
    assert!(f_temp.non_river_active());
    let own_disc = [false; TILE_MAX];
    // Even though wait 1m(0) was never discarded, temporary furiten marks post-discard as furiten!
    assert!(f_temp.is_post_discard_furiten(&[0, 3], &own_disc, Some(10)));

    // Riichi furiten stays active permanently
    let f_riichi = FuritenState::from_observation(&obs, false, true, &rule);
    assert!(f_riichi.riichi);
    assert!(f_riichi.is_post_discard_furiten(&[0], &own_disc, None));

    // Verify global_feats[19] reflects temporary || riichi
    let tracker = HanchanTracker::default();
    let rc = RuleContract::default_tenhou_houou();
    let ctx = RoundGlobalContext::default();
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

    let res_tenhou = eval.calc_with_rule(1, &[4], &[], Some(cond.clone()), &GameRule::default_tenhou());
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
            let do_riichi = !riichi_done[seat as usize] && melds_per_seat[seat as usize] == 0 && turn > 4;
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
    assert!(max_seen_len >= 120, "fuzz should exercise long rounds, max={max_seen_len}");

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
            let native = Action::new(ActionType::Daiminkan, Some(claimed_t136), consume_3.clone(), Some(0));
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
    assert_eq!(enc_own.cand_mode[idx_kyushu], CandidateMode::KyushuDrawEval as u8);
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
    let idx_chi23 = enc_resp.cand_ids.iter().position(|&id| id == chi_23_id).unwrap();
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
        assert_eq!(ot, oracle_tile_before, "Oracle must be live-wall permutation invariant");
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

            let tf = encode_tile_feats(&obs, &vis, fur, &state.rule, phase, &V2FeatureConfig::default())
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
    assert!(steps > 50, "expected real multi-turn simulation, ran {steps} steps");
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

