"""Rules must work through native entry points without a Python flag bridge."""

from types import SimpleNamespace

import pytest

from riichienv import Action, ActionType, GameRule, MeldType, Phase, RiichiEnv, convert
from riichienv.kar.actions import legal_candidates, tile136_to_tile37
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.native_replay import NativeReplayBridge
from tests.kar.core.test_live_agent import _deal_with_hands, _hands_except
from tests.kar.test_actions import _env_with_hand, _ready_observation
from tests.kar.test_kan_preview import _ankan_env, _kakan_env
from tests.kar.test_native_replay import _claims, _kakan_prefix


def _state(env):
    return (
        env.hands,
        [[(meld.meld_type, meld.tiles, meld.opened, meld.from_who) for meld in player] for player in env.melds],
        env.scores(),
        env.riichi_declared,
        env.riichi_stage,
        env.ippatsu_cycle,
        env.double_riichi_declared,
        env.missed_agari_doujun,
        env.missed_agari_riichi,
        env.is_rinshan_flag,
        env.phase,
        env.current_player,
        env.active_players,
        env.drawn_tile,
        env.mjai_log,
        env.rule_log,
    )


def test_complete_riichi_candidates_equal_native_followup_discards():
    env, observation = _env_with_hand()
    complete = observation.legal_riichi_discards()
    assert complete
    assert all(action.action_type == ActionType.RIICHI and action.actor == 0 for action in complete)
    control = env.clone()
    control.step({0: Action(ActionType.RIICHI)})
    expected = {action.tile for action in control._get_legal_actions(0) if action.action_type == ActionType.DISCARD}
    assert {action.tile for action in complete} == expected
    # The frozen model projection consumes the native complete set without
    # exposing new intermediate reach decisions or changing its 546-action IDs.
    mapped = legal_candidates(observation)
    assert {candidate.spec.tile37 for candidate in mapped if candidate.spec.riichi} == {
        tile136_to_tile37(tile) for tile in expected
    }


def test_atomic_riichi_has_identical_state_and_event_deltas_to_legacy_steps():
    env, observation = _env_with_hand()
    declaration = next(action for action in observation.legal_riichi_discards() if action.tile == 53)
    control = env.clone()
    expected_reach = control.step({0: Action(ActionType.RIICHI, None, [], 0)})
    expected_discard = control.step({0: Action(ActionType.DISCARD, 53, [], 0)})
    reach, discard = env.step_riichi_discard(0, declaration)
    assert _state(env) == _state(control)
    assert {seat: obs.events for seat, obs in reach.items()} == {
        seat: obs.events for seat, obs in expected_reach.items()
    }
    assert {seat: obs.events for seat, obs in discard.items()} == {
        seat: obs.events for seat, obs in expected_discard.items()
    }


def test_atomic_riichi_rejects_non_tenpai_before_any_mutation():
    hand = [0, 4, 8, 48, 53, 56, 96, 100, 104, 76, 80, 54, 55]
    env, observations, _ = _deal_with_hands(_hands_except({0: hand}, 132), 132)
    observation = observations[0]
    valid = {action.tile for action in observation.legal_riichi_discards()}
    invalid = next(
        action.tile
        for action in observation.legal_actions()
        if action.action_type == ActionType.DISCARD and action.tile not in valid
    )
    before = _state(env)
    with pytest.raises(ValueError, match="not currently legal"):
        env.step_riichi_discard(0, Action(ActionType.RIICHI, invalid, [], 0))
    assert _state(env) == before


@pytest.mark.parametrize(
    "declaration",
    [
        Action(ActionType.RIICHI),
        Action(ActionType.RIICHI, 53, [], 1),
        Action(ActionType.DISCARD, 53, [], 0),
        Action(ActionType.RIICHI, 53, [53], 0),
    ],
)
def test_atomic_riichi_rejects_incomplete_or_modified_action(declaration):
    env, _ = _env_with_hand()
    before = _state(env)
    with pytest.raises(ValueError, match="complete riichi discard"):
        env.step_riichi_discard(0, declaration)
    assert _state(env) == before


def test_complete_riichi_fails_closed_with_an_old_extension():
    observation = _ready_observation(53)
    old = SimpleNamespace(
        **{
            name: getattr(observation, name)
            for name in (
                "player_id",
                "hands",
                "hand",
                "melds",
                "drawn_tile",
                "forced_tedashi",
                "riichi_declared",
                "legal_actions",
            )
        }
    )
    with pytest.raises(RuntimeError, match="rebuild"):
        legal_candidates(old)


def test_mjai_alone_owns_riichi_kan_declines_and_replacement_context():
    source = _kakan_prefix()
    env = RiichiEnv("4p-red-half", seed=47, rule=GameRule.default_mjsoul())
    for event in source.env.mjai_log:
        env.apply_event(event)
    assert env.ippatsu_cycle == source.env.ippatsu_cycle
    assert env.double_riichi_declared == source.env.double_riichi_declared
    assert env.ippatsu_cycle[1]
    declaration = {"type": "kakan", "actor": 0, "pai": "1s"}
    source.apply(declaration)
    # No NativeReplayBridge, snapshots, preview registration or flag setters.
    env.apply_event(declaration)
    assert _claims(env.replay_kan_responses) == _claims(source.bridge.kan_responses)
    assert env.ippatsu_cycle[1]
    env.apply_event({"type": "dora", "dora_marker": "9p"})
    assert env.replay_kan_responses is None
    assert env.ippatsu_cycle == [False] * 4
    assert env.missed_agari_doujun[1:3] == [True, True]
    assert env.missed_agari_riichi[1:3] == [True, False]
    env.apply_event({"type": "tsumo", "actor": 0, "pai": "6m"})
    assert env.is_rinshan_flag
    env.apply_event({"type": "dahai", "actor": 0, "pai": "6m", "tsumogiri": True})
    assert not env.is_rinshan_flag


def test_bridge_does_not_hold_or_write_rules_context():
    env, _ = _env_with_hand()
    bridge = NativeReplayBridge(env)
    # Reattaching the compatibility wrapper cannot restore cached flags over
    # newer native state; the wrapper has no second representation of them.
    env.ippatsu_cycle = [False, True, False, False]
    NativeReplayBridge(env)
    assert env.ippatsu_cycle == [False, True, False, False]
    assert bridge.kan_responses is None


@pytest.mark.parametrize("kind", ["ankan", "kakan"])
def test_raw_native_kan_window_defers_tiles_and_melds_until_commit(kind):
    source, action = _ankan_env() if kind == "ankan" else _kakan_env()
    env = RiichiEnv("4p-red-single", seed=19, rule=GameRule.default_mjsoul())
    for event in source.mjai_log:
        env.apply_event(event)
    with pytest.raises(ValueError, match="mandatory discard"):
        env.validate_replay_draw(0)
    before_hand = env.hands[0]
    before_melds = [(meld.meld_type, meld.tiles) for meld in env.melds[0]]
    event = {"type": kind, "actor": 0}
    if kind == "ankan":
        event["consumed"] = [convert.tid_to_mjai(tile) for tile in action.consume_tiles]
    else:
        event["pai"] = convert.tid_to_mjai(action.tile)
    observation = env.observe_event(event, 1)
    assert env.phase == Phase.WaitResponse
    assert observation is not None
    assert {value.action_type for value in observation.legal_actions()} == {ActionType.RON, ActionType.PASS}
    assert env.hands[0] == before_hand
    assert [(meld.meld_type, meld.tiles) for meld in env.melds[0]] == before_melds

    # Legacy server logs omit explicit commit acknowledgements. The observed
    # replacement draw is valid closure evidence while still in WaitResponse.
    before = _state(env)
    env.validate_replay_draw(0)
    with pytest.raises(ValueError, match="another player"):
        env.validate_replay_draw(1)
    assert _state(env) == before
    implicit = env.clone()
    implicit.apply_event({"type": "tsumo", "actor": 0, "pai": convert.tid_to_mjai(source.wall[0])})
    assert implicit.replay_kan_responses is None and implicit.is_rinshan_flag
    assert implicit.phase == Phase.WaitAct and not implicit.needs_tsumo

    robbed = env.clone()
    robbed.apply_event({"type": "hora", "actor": 1, "target": 0, "pai": convert.tid_to_mjai(action.tile)})
    assert robbed.hands[0] == before_hand
    assert [(meld.meld_type, meld.tiles) for meld in robbed.melds[0]] == before_melds
    with pytest.raises(ValueError, match="round termination"):
        robbed.validate_replay_draw(0)

    # Explicit canonical commit can precede dora/tsumo. No future draw is made.
    assert env.observe_event({"type": "kan_commit", "actor": 0}, 0) is None
    assert env.replay_kan_responses is None
    assert len(env.hands[0]) == len(before_hand) - (4 if kind == "ankan" else 1)
    assert env.melds[0][-1].meld_type == (MeldType.Ankan if kind == "ankan" else MeldType.Kakan)
    assert not env.active_players
    assert env.needs_tsumo
    assert not env.get_observation(0).legal_actions()


def test_live_kakan_also_preserves_the_pon_while_ron_is_unresolved():
    env, declaration = _kakan_env()
    before_hand = env.hands[0]
    before_melds = [(meld.meld_type, meld.tiles) for meld in env.melds[0]]
    env.step({0: declaration})
    assert env.phase == Phase.WaitResponse
    assert env.hands[0] == before_hand
    assert [(meld.meld_type, meld.tiles) for meld in env.melds[0]] == before_melds
    env.step({seat: Action(ActionType.PASS) for seat in env.active_players})
    assert env.melds[0][0].meld_type == MeldType.Kakan
    assert len(env.hands[0]) == len(before_hand)  # Commit removes one; replacement draw adds one.
    assert not env.needs_tsumo
    assert any(action.action_type == ActionType.DISCARD for action in env._get_legal_actions(0))


def test_replayed_ankan_can_resume_native_responses_without_redeclaring_in_rule_log():
    source, action = _ankan_env()
    env = RiichiEnv("4p-red-single", seed=19, rule=GameRule.default_mjsoul())
    for event in source.mjai_log:
        env.apply_event(event)
    # Resume against the known remaining wall; the replay reducer must not
    # invent it from an observed prefix.
    env.wall = source.wall
    projection = LiveEventProjection()
    projection.observe_native(env.rule_log)
    before = len(env.rule_log)
    env.apply_event(
        {"type": "ankan", "actor": 0, "consumed": [convert.tid_to_mjai(tile) for tile in action.consume_tiles]}
    )
    env.step({seat: Action(ActionType.PASS) for seat in env.active_players})
    delta = env.rule_log[before:]
    assert [event["type"] for event in delta] == ["ankan", "kan_commit", "dora", "tsumo"]
    assert env.replay_kan_responses is None
    assert not env.needs_tsumo
    projection.observe_native(delta)
    projection.ledgers[0].check_observation(env.get_observation(0))
