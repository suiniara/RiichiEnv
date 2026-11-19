"""Verify score selection through legal physical walls and settlement audit."""

import copy

import pytest

from riichienv import ActionType, Conditions, GameRule, HandEvaluator, Phase, RiichiEnv, convert
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.settlement import verify_settlements

TENHOU_HANDS = {
    "suuankou": ("1m 1m 1m 2p 2p 2p 3s 3s 3s E E E 5p 5p", 48),
    "chuuren": ("1p 1p 1p 2p 3p 4p 5p 5p 6p 7p 8p 9p 9p 9p", 47),
}


def take_tile(pool, name):
    tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == name)
    pool.remove(tile)
    return tile


def initial_tenhou_wall(players, kind, draw_position):
    all_tiles = [tile for tile in range(136) if players == 4 or tile < 4 or tile >= 32]
    pool = all_tiles.copy()
    full_hand = [take_tile(pool, name) for name in TENHOU_HANDS[kind][0].split()]
    hand = full_hand.copy()
    drawn_tile = hand.pop(draw_position)
    indicator = take_tile(pool, "C")
    hands = [hand] + [[pool.pop(0) for _ in range(13)] for _ in range(players - 1)]
    wall = [tile for start in (0, 4, 8) for hand in hands for tile in hand[start : start + 4]]
    wall += [hand[12] for hand in hands] + [drawn_tile] + pool + [indicator]
    indicator_index = 99 if players == 3 else len(wall) - 5
    wall[indicator_index], wall[-1] = wall[-1], wall[indicator_index]
    assert sorted(wall) == all_tiles
    return wall, full_hand, drawn_tile


def win_tenhou(players, preset, kind, draw_position):
    wall, full_hand, drawn_tile = initial_tenhou_wall(players, kind, draw_position)
    rule = getattr(GameRule, f"default_{preset}")()
    env = RiichiEnv(f"{players}p-red-single", seed=42, rule=rule)
    env.reset(wall=wall, oya=0)
    assert sorted(env.hands[0]) == sorted(full_hand)
    assert env.current_player == 0 and env.drawn_tile == drawn_tile
    tsumo = next(action for action in env.get_observation(0).legal_actions() if action.action_type == ActionType.TSUMO)
    env.step({0: tsumo})
    assert env.is_done
    assert not any(str(event.get("reason", "")).startswith("Error:") for event in env.mjai_log)
    return env


@pytest.mark.parametrize("players", [3, 4])
@pytest.mark.parametrize("preset", ["tenhou", "mjsoul"])
@pytest.mark.parametrize("kind", TENHOU_HANDS)
def test_tenhou_scoring_is_invariant_to_initial_deal_order(players, preset, kind):
    # Mahjong Soul ranked rules: at Tenhou, use the highest-scoring winning tile.
    # https://mahjongsoul.com/news/46/
    # The Tenhou profile still caps the individual double-yakuman patterns.
    units = 3 if preset == "mjsoul" else 2
    payment = 16000 * units
    for draw_position in range(14):
        env = win_tenhou(players, preset, kind, draw_position)
        result = env.win_results[0]
        assert result.is_win and result.yakuman
        assert result.han == 13 * units
        assert sorted(result.yaku) == [35, TENHOU_HANDS[kind][1]]
        assert result.tsumo_agari_ko == payment
        expected_deltas = [payment * (players - 1)] + [-payment] * (players - 1)
        assert env.score_deltas == expected_deltas
        starting_score = 35000 if players == 3 else 25000
        assert env.scores() == [starting_score + delta for delta in expected_deltas]


@pytest.mark.parametrize("kind", TENHOU_HANDS)
def test_tenhou_audit_accepts_correct_payment_and_rejects_old_underpayment(kind):
    env = win_tenhou(4, "mjsoul", kind, 0)
    live = LiveEventProjection()
    live.observe_native(env.rule_log)
    report = verify_settlements(live.events)
    assert report["status"] == "PASS", report
    assert_audit_rejects_underpayment(live.events, [96000, -32000, -32000, -32000], [121000, -7000, -7000, -7000])


def assert_audit_rejects_underpayment(events, deltas, scores):
    underpaid = copy.deepcopy(events)
    for event in underpaid:
        if event["type"] == "hora":
            event["deltas"] = deltas
            event["scores"] = scores
        elif event["type"] == "end_game":
            event["scores"] = scores
    report = verify_settlements(underpaid)
    assert report["status"] == "FAIL", report
    assert any(row.get("reason") == "independent_settlement_mismatch" for row in report["settlements"])


def high_bonus_wall():
    pool = list(range(136))
    winner = [take_tile(pool, name) for name in "1m 1m 1m 2m 2m 2m 3m 3m 3m 4m 4m 4m 5mr".split()]
    kan_hand = [take_tile(pool, name) for name in "P P P F F F C C C 6p 7p 8p E".split()]
    draws = [take_tile(pool, name) for name in "8s 9s P 7s 8s 5m".split()]
    dead_wall = [take_tile(pool, name) for name in "F C 6s 9s".split()]
    for dora, ura in [("9m", "1m"), ("9m", "2m"), ("9m", "3m"), ("9m", "4m")]:
        dead_wall.extend([take_tile(pool, dora), take_tile(pool, ura)])
    dead_wall.extend([take_tile(pool, "S"), take_tile(pool, "W")])
    hands = [[pool.pop(0) for _ in range(13)], winner, kan_hand, [pool.pop(0) for _ in range(13)]]
    wall = [tile for start in (0, 4, 8) for hand in hands for tile in hand[start : start + 4]]
    wall += [hand[12] for hand in hands] + draws + pool + list(reversed(dead_wall))
    assert sorted(wall) == list(range(136))
    return wall, winner


def test_true_double_yakuman_survives_bonus_han_and_ends_the_match():
    wall, winner = high_bonus_wall()
    env = RiichiEnv("4p-red-half", seed=42, rule=GameRule.default_mjsoul())
    env.reset(wall=wall, oya=0)
    live = LiveEventProjection()
    live.observe_native(env.rule_log)

    def step(actions):
        before = len(env.rule_log)
        env.step(actions)
        live.observe_native(env.rule_log[before:])
        assert not any(str(event.get("reason", "")).startswith("Error:") for event in env.mjai_log)

    def pass_responses():
        if env.phase == Phase.WaitResponse:
            step(
                {
                    seat: next(
                        action
                        for action in env.get_observation(seat).legal_actions()
                        if action.action_type == ActionType.PASS
                    )
                    for seat in env.active_players
                }
            )

    def play(kind, name=None):
        pass_responses()
        seat = env.current_player
        action = next(
            action
            for action in env.get_observation(seat).legal_actions()
            if action.action_type == kind and (name is None or convert.tid_to_mjai(action.tile) == name)
        )
        step({seat: action})
        return seat

    assert play(ActionType.DISCARD, "8s") == 0
    pass_responses()
    assert env.current_player == 1
    declaration = next(
        action for action in env.get_observation(1).legal_riichi_discards() if convert.tid_to_mjai(action.tile) == "9s"
    )
    before = len(env.rule_log)
    env.step_riichi_discard(1, declaration)
    live.observe_native(env.rule_log[before:])
    assert play(ActionType.ANKAN, "P") == 2
    assert play(ActionType.ANKAN, "F") == 2
    assert play(ActionType.ANKAN, "C") == 2
    assert play(ActionType.DISCARD, "6s") == 2
    assert play(ActionType.DISCARD, "7s") == 3
    assert play(ActionType.DISCARD, "8s") == 0
    pass_responses()
    assert env.current_player == 1 and convert.tid_to_mjai(env.drawn_tile) == "5m"

    conditions = Conditions(tsumo=True, riichi=True, double_riichi=True, player_wind=1)
    ura = [convert.mjai_to_tid(name) for name in env._get_ura_markers()]
    for dora, ura_indicators in [([], []), (env.dora_indicators, ura)]:
        result = HandEvaluator(winner).calc(env.drawn_tile, dora, conditions, ura_indicators)
        assert result.is_win and result.yakuman
        assert result.han == 26 and result.yaku == [48]
        assert result.tsumo_agari_oya == 32000 and result.tsumo_agari_ko == 16000

    assert play(ActionType.TSUMO) == 1
    assert env.is_done
    assert env.score_deltas == [-32000, 65000, -16000, -16000]
    assert env.scores() == [-7000, 89000, 9000, 9000]
    report = verify_settlements(live.events)
    assert report["status"] == "PASS", report

    # The old 34-han ordinary decomposition was capped to one yakuman. It
    # underpaid the win and incorrectly continued the match instead of busting.
    end = next(index for index, event in enumerate(live.events) if event["type"] == "end_kyoku")
    assert_audit_rejects_underpayment(
        live.events[: end + 1], [-16000, 33000, -8000, -8000], [9000, 57000, 17000, 17000]
    )
