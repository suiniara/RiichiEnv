from dataclasses import replace

import pytest

from riichienv import ActionType, GameRule, RiichiEnv
from riichienv.kar.actions import execute_candidate, execute_joint_responses, legal_candidates
from riichienv.kar.agent import functional_match
from riichienv.kar.live_events import LiveEventProjection


def _deal_with_hands(hands, drawn):
    used = [tile for hand in hands for tile in hand]
    assert len(set(used + [drawn])) == 53
    remainder = [tile for tile in range(136) if tile not in used and tile != drawn]
    wall = [tile for start in (0, 4, 8) for hand in hands for tile in hand[start : start + 4]]
    wall.extend(hand[12] for hand in hands)
    env = RiichiEnv("4p-red-single", seed=42, rule=GameRule.default_mjsoul())
    observations = env.reset(wall=wall + [drawn] + remainder, oya=0)
    projection = LiveEventProjection()
    projection.observe_native(list(env.mjai_log))
    return env, observations, projection


def _hands_except(specials, drawn):
    used = {drawn} | {tile for hand in specials.values() for tile in hand}
    pool = iter(tile for tile in range(136) if tile not in used)
    return [specials[seat] if seat in specials else [next(pool) for _ in range(13)] for seat in range(4)]


def _sync(env, projection, before):
    projection.observe_native(list(env.mjai_log[before:]))


def test_live_riichi_keeps_both_native_steps_and_owner_cutoffs():
    full = [36, 37, 38, 44, 45, 46, 52, 53, 54, 60, 61, 62, 108, 109]
    drawn = 53
    env, obs, projection = _deal_with_hands(_hands_except({0: [tile for tile in full if tile != drawn]}, drawn), drawn)
    selected = next(candidate for candidate in legal_candidates(obs[0]) if candidate.global_id == 61)
    before = len(env.mjai_log)
    result = execute_candidate(env, 0, selected)
    _sync(env, projection, before)
    assert len(result.step_observations) == 2
    assert projection.snapshot(0).riichi_declared[0]
    assert [event["type"] for event in projection.events[before:]][:2] == ["reach", "dahai"]
    for seat, observation in result.items():
        projection.ledgers[seat].check_observation(observation)


def test_all_responders_are_validated_before_one_joint_multi_ron_submission():
    first = [0, 4, 8, 48, 52, 56, 96, 100, 104, 124, 125, 126, 109]
    second = [1, 5, 9, 49, 53, 57, 97, 101, 105, 128, 129, 130, 110]
    drawn = 132
    hands = _hands_except({1: first, 2: second}, drawn)
    # Give seat zero the fourth distinct East without duplicating any physical tile.
    east = 108
    if east not in hands[0]:
        owner = next((seat for seat in (3,) if east in hands[seat]), None)
        if owner is not None:
            hands[owner][hands[owner].index(east)], hands[0][0] = hands[0][0], east
        else:
            hands[0][0] = east
    env, obs, projection = _deal_with_hands(hands, drawn)
    discard = next(
        action for action in obs[0].legal_actions() if action.action_type == ActionType.DISCARD and action.tile == east
    )
    before = len(env.mjai_log)
    responses = env.step({0: discard})
    _sync(env, projection, before)
    choices = {}
    for seat, observation in responses.items():
        mapped = legal_candidates(observation, source_player=0)
        choices[seat] = next(
            (candidate for candidate in mapped if candidate.spec.action_type == "RON"),
            next(candidate for candidate in mapped if candidate.global_id == 221),
        )
    assert {seat for seat, candidate in choices.items() if candidate.spec.action_type == "RON"} == {1, 2}
    before = len(env.mjai_log)
    bad = dict(choices)
    seat = next(iter(bad))
    bad[seat] = replace(bad[seat], _legal_signature=())
    with pytest.raises(ValueError):
        execute_joint_responses(env, bad)
    assert len(env.mjai_log) == before
    seat_three_private_hand = projection.snapshot(3).hand
    execute_joint_responses(env, choices)
    _sync(env, projection, before)
    wins = [event for event in projection.events if event["type"] == "hora"]
    assert len(wins) == 2 and len({event["source_record_index"] for event in wins}) == 1
    assert wins[0]["deltas"] == wins[1]["deltas"] and projection.public_scores == tuple(env.scores())
    assert all("public_hand" in event for event in wins)
    assert projection.snapshot(3).hand == seat_three_private_hand


def test_pon_is_followed_by_forced_tedashi_without_an_extra_draw():
    drawn = 132
    hands = _hands_except({1: [109, 110, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]}, drawn)
    if 108 not in hands[0]:
        hands[0][0] = 108
    env, obs, projection = _deal_with_hands(hands, drawn)
    discard = next(
        action for action in obs[0].legal_actions() if action.action_type == ActionType.DISCARD and action.tile == 108
    )
    before = len(env.mjai_log)
    responses = env.step({0: discard})
    _sync(env, projection, before)
    choices = {}
    for seat, observation in responses.items():
        mapped = legal_candidates(observation, source_player=0)
        choices[seat] = next(
            (candidate for candidate in mapped if seat == 1 and candidate.spec.action_type == "PON"),
            next(candidate for candidate in mapped if candidate.global_id == 221),
        )
    before = len(env.mjai_log)
    result = execute_joint_responses(env, choices)
    _sync(env, projection, before)
    assert set(result) == {1} and result[1].drawn_tile is None
    assert projection.snapshot(1).post_call_actor == 1
    assert all(candidate.global_id % 4 == 0 for candidate in legal_candidates(result[1]) if candidate.global_id < 148)
    assert not any(event["type"] == "tsumo" for event in env.mjai_log[before:])


def test_live_ankan_commits_only_after_public_replacement_boundary():
    drawn = 132
    hand = [0, 1, 2, 3, 4, 8, 12, 36, 40, 44, 72, 76, 80]
    env, observations, projection = _deal_with_hands(_hands_except({0: hand}, drawn), drawn)
    candidate = next(value for value in legal_candidates(observations[0]) if value.spec.action_type == "ANKAN")
    before = len(env.mjai_log)
    result = execute_candidate(env, 0, candidate)
    delta = list(env.mjai_log[before:])
    assert delta[0]["type"] == "ankan"
    projection.observe_native(delta[:1])
    pending = projection.snapshot(0)
    assert pending.pending_kan_actor == 0 and not pending.melds[0]
    projection.observe_native(delta[1:])
    committed = projection.snapshot(0)
    assert committed.pending_kan_actor is None and len(committed.melds[0]) == 1
    assert committed.melds[0][0].kind == "ankan"
    for seat, observation in result.items():
        projection.ledgers[seat].check_observation(observation)


def test_complete_fixed_policy_half_game_and_truncation_are_not_confused():
    truncated = functional_match(17, max_decisions=1)
    assert truncated["status"] == "BLOCKED" and not truncated["complete"]
    report = functional_match(17)
    assert report["status"] == "PASS" and report["complete"] and not report["parameter_updates_executed"]
    assert len(report["scores"]) == len(report["ranks"]) == 4 and sum(report["scores"]) == 100000
    assert report["rounds"] >= 8
    assert all(row["evidence_status"] == "unfitted_prior_only" for row in report["decisions"])
