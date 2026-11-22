//! Global (48-dim) and 4-player (4x32-dim) state encoders (Contract v2.4 Section 1.4).

use crate::observation::Observation;
use crate::shanten::calculate_shanten;
use crate::types::MeldType;

use super::furiten::FuritenState;
use super::{
    DecisionPhase, GLOBAL_FEATS_DIM, HanchanTracker, PLAYER_FEATS_DIM, RuleContract,
    seat_rank_and_gaps,
};

/// Extra per-round public flags not duplicated on legacy `Observation`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoundGlobalContext {
    pub phase: DecisionPhase,
    /// Relative seat `1..=3` of trigger discarder/kan-declarer in response phases (`None` in OwnAct).
    pub trigger_actor_rel: Option<u8>,
    pub live_wall_left: u8,
    pub is_first_uninterrupted_turn: bool,
    pub double_riichi_declared: [bool; 4],
    pub ippatsu_active: [bool; 4],
    pub riichi_turn_index: [Option<u8>; 4],
}

/// Encode `global_feats: [f32; 48]` and `player_feats: [[f32; 32]; 4]`.
pub fn encode_global_player(
    obs: &Observation,
    furiten: FuritenState,
    tracker: &HanchanTracker,
    rule_contract: &RuleContract,
    ctx: &RoundGlobalContext,
) -> Result<([f32; GLOBAL_FEATS_DIM], [[f32; PLAYER_FEATS_DIM]; 4]), String> {
    let pid = obs.player_id as usize;
    if pid >= 4 {
        return Err(format!("invalid observer player_id {}", obs.player_id));
    }

    let mut global = [0.0f32; GLOBAL_FEATS_DIM];

    // [0..2]: round_wind one-hot (East=0, South=1, West=2)
    let rw = (obs.round_wind as usize).min(2);
    global[rw] = 1.0;

    // [3..6]: kyoku one-hot (0..3)
    let kyoku_local = (obs.kyoku_index % 4) as usize;
    global[3 + kyoku_local] = 1.0;

    // [7..8]: honba / 8, kyoutaku / 8 (clipped [0, 2])
    global[7] = (f32::from(obs.honba) / 8.0).min(2.0);
    global[8] = (obs.riichi_sticks as f32 / 8.0).min(2.0);

    // [9]: live_wall_left / 70
    global[9] = (f32::from(ctx.live_wall_left) / 70.0).clamp(0.0, 1.0);

    // [10..11]: is_all_last, is_extension_round
    let last_wind = if rule_contract.is_tonpuusen { 0u8 } else { 1u8 };
    let is_all_last = obs.round_wind == last_wind && (obs.kyoku_index % 4) == 3;
    let is_extension = obs.round_wind > last_wind;
    global[10] = if is_all_last { 1.0 } else { 0.0 };
    global[11] = if is_extension { 1.0 } else { 0.0 };

    // [12]: total_kans_in_round / 4
    let total_kans: usize = obs
        .melds
        .iter()
        .map(|ms| {
            ms.iter()
                .filter(|m| {
                    matches!(
                        m.meld_type,
                        MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan
                    )
                })
                .count()
        })
        .sum();
    global[12] = (total_kans as f32 / 4.0).min(1.0);

    // [13]: dora_indicator_cnt / 5
    global[13] = (obs.dora_indicators.len() as f32 / 5.0).min(1.0);

    // [14]: is_first_uninterrupted_turn
    global[14] = if ctx.is_first_uninterrupted_turn {
        1.0
    } else {
        0.0
    };

    // [15..18]: decision phase one-hot
    global[15 + ctx.phase as usize] = 1.0;

    // [19]: Contract v2.4 P1-2: self non-river furiten (temporary || riichi)
    global[19] = if furiten.non_river_active() { 1.0 } else { 0.0 };

    // [20..22]: trigger_actor_rel ∈ {1, 2, 3} one-hot (all 0 in OwnAct)
    if ctx.phase != DecisionPhase::OwnAct
        && let Some(src) = ctx.trigger_actor_rel
        && (1..=3).contains(&src)
    {
        global[20 + (src as usize - 1)] = 1.0;
    }

    // [23..25]: self_cur_shanten / 8, self_meld_cnt / 4, self_is_menzen
    let cur_shanten = calculate_shanten(&obs.hands[pid]).max(0);
    global[23] = (cur_shanten as f32 / 8.0).min(1.0);
    global[24] = (obs.melds[pid].len() as f32 / 4.0).min(1.0);
    let self_is_menzen = obs.melds[pid].iter().all(|m| !m.opened);
    global[25] = if self_is_menzen { 1.0 } else { 0.0 };

    // [26..36]: 1-to-1 mirror of GameRule 11 native switches
    let r = &rule_contract.rule;
    let b = |v: bool| if v { 1.0 } else { 0.0 };
    global[26] = b(r.allows_ron_on_ankan_for_kokushi_musou);
    global[27] = b(r.is_kokushi_musou_13machi_double);
    global[28] = b(r.is_suuankou_tanki_double);
    global[29] = b(r.is_junsei_chuurenpoutou_double);
    global[30] = b(r.is_daisuushii_double);
    global[31] = b(r.yakuman_pao_is_liability_only);
    global[32] = b(r.sanchaho_is_draw);
    global[33] = b(r.kuikae_forbidden);
    global[34] = b(r.dealer_first_discard_is_tedashi);
    global[35] = b(r.extension_dealer_repeat_priority);
    global[36] = b(r.allow_fifth_copy_tenpai);

    // [37..43]: 7 match/tabletop rule flags
    global[37] = b(rule_contract.is_tonpuusen);
    global[38] = b(rule_contract.has_aka_dora);
    global[39] = b(rule_contract.has_kuitan);
    global[40] = b(rule_contract.has_tobi_end);
    global[41] = b(rule_contract.has_agari_yame);
    global[42] = b(rule_contract.has_abortive_draws);
    global[43] = b(rule_contract.has_nagashi_mangan);

    // [44..47]: normalized 4-placement payoff vector w(R)
    global[44..48].copy_from_slice(&rule_contract.rank_payoffs);

    // ---------------------------------------------------------------------
    // player_feats [4, 32] ordered [0=Self, 1=Shimocha, 2=Toimen, 3=Kamicha]
    // ---------------------------------------------------------------------
    let mut players = [[0.0f32; PLAYER_FEATS_DIM]; 4];
    let self_score = obs.scores[pid];
    let n_rounds = tracker.rounds_completed;
    let denom_rounds = n_rounds.max(1) as f32;

    for rel in 0..4usize {
        let abs = (pid + rel) % 4;
        let p = &mut players[rel];

        // [0..3]: relative seat one-hot
        p[rel] = 1.0;

        // [4..7]: seat wind one-hot (E, S, W, N)
        let seat_wind = ((abs as u8 + 4 - obs.oya) % 4) as usize;
        p[4 + seat_wind] = 1.0;

        // [8]: is_dealer
        p[8] = if abs == obs.oya as usize { 1.0 } else { 0.0 };

        // [9..13]: score, diff from self, gap higher/lower, cur_rank
        let score = obs.scores[abs];
        p[9] = ((score - 25_000) as f32 / 25_000.0).clamp(-3.0, 3.0);
        p[10] = ((score - self_score) as f32 / 25_000.0).clamp(-3.0, 3.0);
        let (rank, gap_hi, gap_lo) =
            seat_rank_and_gaps(&obs.scores, &tracker.initial_seat_order, abs as u8);
        p[11] = (gap_hi as f32 / 25_000.0).clamp(0.0, 3.0);
        p[12] = (gap_lo as f32 / 25_000.0).clamp(0.0, 3.0);
        p[13] = f32::from(rank - 1) / 3.0;

        // [14..17]: riichi, double_riichi, ippatsu, riichi_turn_norm
        p[14] = b(obs.riichi_declared[abs]);
        p[15] = b(ctx.double_riichi_declared[abs]);
        p[16] = b(ctx.ippatsu_active[abs]);
        p[17] = ctx.riichi_turn_index[abs]
            .map(|idx| (f32::from(idx) / 24.0).min(1.0))
            .unwrap_or(0.0);

        // [18]: open_meld_cnt / 4
        let open_melds = obs.melds[abs].iter().filter(|m| m.opened).count();
        p[18] = (open_melds as f32 / 4.0).min(1.0);

        // [19]: tedashi ratio among last <= 6 discards
        let flags = &obs.tsumogiri_flags[abs];
        if !flags.is_empty() {
            let take = flags.len().min(6);
            let slice = &flags[flags.len() - take..];
            let tedashi_cnt = slice.iter().filter(|&&is_tsumo| !is_tsumo).count();
            p[19] = tedashi_cnt as f32 / take as f32;
        }

        // [20..31]: cross-round Hanchan behavioral summary (strictly 0 at East 1)
        if n_rounds > 0 {
            p[20] = (n_rounds as f32 / 12.0).min(2.0);
            p[21] = tracker.riichi_rounds[abs] as f32 / denom_rounds;
            p[22] = tracker.open_rounds[abs] as f32 / denom_rounds;
            p[23] = tracker.win_rounds[abs] as f32 / denom_rounds;
            p[24] = tracker.dealin_rounds[abs] as f32 / denom_rounds;
            p[25] = if tracker.win_rounds[abs] > 0 {
                ((tracker.win_pts_sum[abs] as f32 / tracker.win_rounds[abs] as f32) / 12_000.0)
                    .clamp(0.0, 4.0)
            } else {
                0.0
            };
            p[26] = if tracker.dealin_rounds[abs] > 0 {
                ((tracker.dealin_pts_sum[abs] as f32 / tracker.dealin_rounds[abs] as f32)
                    / 12_000.0)
                    .clamp(0.0, 4.0)
            } else {
                0.0
            };
            p[27] = (tracker.prev_round_delta[abs] as f32 / 12_000.0).clamp(-3.0, 3.0);
            p[28] = b(tracker.prev_round_won[abs]);
            p[29] = b(tracker.prev_round_dealin[abs]);
            p[30] = b(tracker.prev_round_riichi[abs]);
            p[31] = b(tracker.prev_round_opened[abs]);
        }
    }

    Ok((global, players))
}
