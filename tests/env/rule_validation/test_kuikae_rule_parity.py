"""Live calls and native replay honor the same explicit kuikae rule."""

import pytest

from riichienv import Action, ActionType, GameRule, RiichiEnv, convert
from riichienv.kar.actions import legal_candidates
from tests.kar.test_decision_time_parity import _wall


@pytest.mark.parametrize("kuikae_forbidden", [False, True])
@pytest.mark.parametrize("kind", ["pon", "chi"])
def test_live_and_replay_call_discards_follow_kuikae_rule(kind, kuikae_forbidden):
    rule = GameRule.default_mjsoul()
    rule.kuikae_forbidden = kuikae_forbidden
    live = RiichiEnv("4p-red-single", seed=1, rule=rule)
    replay = RiichiEnv("4p-red-single", seed=1, rule=rule)
    held = ("1m", "1m", "1m") if kind == "pon" else ("1m", "2m", "3m", "4m")
    called = "1m" if kind == "pon" else "4m"
    consumed = ["1m", "1m"] if kind == "pon" else ["2m", "3m"]
    live.reset(wall=_wall({1: held}, (called,)), oya=0)
    discard = next(
        action
        for action in live.get_observation(0).legal_actions()
        if action.action_type == ActionType.DISCARD and convert.tid_to_mjai(action.tile) == called
    )
    live.step({0: discard})
    action_type = ActionType.PON if kind == "pon" else ActionType.CHI
    call = next(
        action
        for action in live.get_observation(1).legal_actions()
        if action.action_type == action_type
        and sorted(convert.tid_to_mjai(tile) for tile in action.consume_tiles) == consumed
    )
    live.step({seat: call if seat == 1 else Action(ActionType.PASS) for seat in live.active_players})
    assert live.current_player == 1
    for event in live.rule_log:
        replay.apply_event(event)

    live_candidates = legal_candidates(live.get_observation(1))
    replay_candidates = legal_candidates(replay.get_observation(1))
    assert {candidate.global_id for candidate in live_candidates} == {
        candidate.global_id for candidate in replay_candidates
    }
    discards = {
        convert.tid_to_mjai(candidate.actions[0].tile)
        for candidate in live_candidates
        if candidate.spec.action_type == "DISCARD"
    }
    # Chi of 4m using 23m restricts both 4m and the replacement sequence's 1m.
    for tile in ("1m",) if kind == "pon" else ("1m", "4m"):
        assert (tile in discards) == (not kuikae_forbidden)
    assert discards


def _sanma_pon_wall():
    # All 108 distinct sanma tiles, with three 1m in South's initial hand and
    # the fourth as the dealer's first draw. No hand/state injection is used.
    tiles = [tile for tile in range(136) if not 4 <= tile < 32]
    pool = [tile for tile in tiles if tile >= 4]
    hands = [[], [0, 1, 2], []]
    for hand in hands:
        hand.extend(pool.pop(0) for _ in range(13 - len(hand)))
    wall = [tile for start in (0, 4, 8) for hand in hands for tile in hand[start : start + 4]]
    wall += [hand[12] for hand in hands] + [3] + pool
    assert len(wall) == 108 and sorted(wall) == tiles
    return wall


@pytest.mark.parametrize("kuikae_forbidden", [False, True])
def test_sanma_live_and_replay_pon_follow_kuikae_rule(kuikae_forbidden):
    rule = GameRule.default_mjsoul()
    rule.kuikae_forbidden = kuikae_forbidden
    live = RiichiEnv("3p-red-single", seed=1, rule=rule)
    replay = RiichiEnv("3p-red-single", seed=1, rule=rule)
    live.reset(wall=_sanma_pon_wall(), oya=0)
    discard = next(
        action
        for action in live.get_observation(0).legal_actions()
        if action.action_type == ActionType.DISCARD and action.tile == 3
    )
    live.step({0: discard})
    pon = next(action for action in live.get_observation(1).legal_actions() if action.action_type == ActionType.PON)
    live.step({seat: pon if seat == 1 else Action(ActionType.PASS) for seat in live.active_players})
    assert live.current_player == 1
    for event in live.mjai_log:
        replay.apply_event(event)

    live_actions = live.get_observation(1).legal_actions()
    replay_actions = replay.get_observation(1).legal_actions()
    assert all(action.action_type == ActionType.DISCARD for action in [*live_actions, *replay_actions])
    live_discards = {convert.tid_to_mjai(action.tile) for action in live_actions}
    replay_discards = {convert.tid_to_mjai(action.tile) for action in replay_actions}
    assert live_discards == replay_discards
    assert ("1m" in live_discards) == (not kuikae_forbidden)
    assert live_discards
