import copy

import pytest

from riichienv import Action, Action3P, ActionType, GameRule, Phase, RiichiEnv, convert

STANDARD = ["2m", "3m", "4m", "3p", "4p", "5p", "5s", "6s", "7s", "7p", "8p", "E", "E"]
NOT_READY = ["1m", "2m", "4m", "5m", "7m", "8m", "1p", "2p", "4p", "5p", "7p", "8p", "E"]
KOKUSHI = ["1m", "1m", "9m", "1p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"]


def _take(pool, value):
    tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == value)
    pool.remove(tile)
    return tile


def _wall(hands, draw_sequence, pool):
    for hand in hands:
        while len(hand) < 13:
            hand.append(pool.pop(0))
    deal = [tile for start in (0, 4, 8) for hand in hands for tile in hand[start : start + 4]]
    wall = deal + [hand[12] for hand in hands] + draw_sequence + pool
    assert sorted(wall) == list(range(136))
    return wall


def _ankan_env(*, kokushi_players=(1,), standard_players=(2,)):
    pool = list(range(136))
    for tile in (68, 69, 70, 71):
        pool.remove(tile)
    hands = [[68, 69, 70], [], [], []]
    for player in kokushi_players:
        hands[player] = [_take(pool, value) for value in KOKUSHI]
    for player in standard_players:
        hands[player] = [_take(pool, value) for value in STANDARD]
    env = RiichiEnv("4p-red-single", seed=19, rule=GameRule.default_mjsoul())
    env.reset(wall=_wall(hands, [71], pool), oya=0)
    declaration = next(
        action for action in env._get_legal_actions(0) if action.action_type == ActionType.ANKAN and action.tile == 68
    )
    return env, declaration


def _pass_responses(env):
    if env.phase == Phase.WaitResponse:
        env.step({player: Action(ActionType.PASS) for player in env.active_players})
    assert not env.is_done


def _discard(env, actor, tile):
    _pass_responses(env)
    assert env.current_player == actor
    action = next(
        action
        for action in env._get_legal_actions(actor)
        if action.action_type == ActionType.DISCARD and action.tile == tile
    )
    env.step({actor: action})
    assert not env.is_done


def _kakan_env(*, ready_players=(1, 2), stale_offer=False):
    pool = list(range(136))
    for tile in (68, 69, 70, 71):
        pool.remove(tile)
    hands = [[69, 70], [], [], [68]]
    for player in (1, 2):
        values = STANDARD if player in ready_players else NOT_READY
        if stale_offer:
            values = (
                ["7p" if value == "E" else value for value in values]
                if player == 1
                else ["6p" if value == "7p" else value for value in values]
            )
        hands[player] = [_take(pool, value) for value in values]
    stale_tile = _take(pool, "7p") if stale_offer else None
    if stale_tile is not None:
        hands[3].append(stale_tile)
    queue = [pool.pop() for _ in range(7)] + [71]
    env = RiichiEnv("4p-red-single", seed=37, rule=GameRule.default_mjsoul())
    env.reset(wall=_wall(hands, queue, pool), oya=0)
    for actor in (0, 1, 2):
        _pass_responses(env)
        _discard(env, actor, env.drawn_tile)
    _discard(env, 3, 68)
    pon = next(
        action for action in env._get_legal_actions(0) if action.action_type == ActionType.PON and action.tile == 68
    )
    env.step({player: pon if player == 0 else Action(ActionType.PASS) for player in env.active_players})
    discard = next(action for action in env._get_legal_actions(0) if action.action_type == ActionType.DISCARD)
    _discard(env, 0, discard.tile)
    for actor in (1, 2, 3):
        _pass_responses(env)
        _discard(env, actor, stale_tile if actor == 3 and stale_tile is not None else env.drawn_tile)
    _pass_responses(env)
    assert env.current_player == 0
    assert env.drawn_tile == 71
    declaration = next(
        action for action in env._get_legal_actions(0) if action.action_type == ActionType.KAKAN and action.tile == 71
    )
    return env, declaration


def _key(action):
    return int(action.action_type), action.tile, tuple(action.consume_tiles), action.actor


def _keys(actions):
    return tuple(_key(action) for action in actions)


def _snapshot(env):
    observations = env.clone().get_observations(list(range(4)))
    return {
        "hands": env.hands,
        "melds": [
            [(int(meld.meld_type), tuple(meld.tiles), meld.opened, meld.from_who) for meld in player]
            for player in env.melds
        ],
        "discards": env.discards,
        "wall": env.wall,
        "dora_indicators": env.dora_indicators,
        "scores": env.scores(),
        "phase": int(env.phase),
        "current_player": env.current_player,
        "active_players": env.active_players,
        "needs_tsumo": env.needs_tsumo,
        "is_done": env.is_done,
        "needs_initialize_next_round": env.needs_initialize_next_round,
        "last_discard": env.last_discard,
        "drawn_tile": env.drawn_tile,
        "current_claims": {player: _keys(actions) for player, actions in env.current_claims.items()},
        "mjai_log": copy.deepcopy(env.mjai_log),
        "observations": {player: obs.serialize_to_base64() for player, obs in observations.items()},
        "riichi_declared": env.riichi_declared,
        "riichi_stage": env.riichi_stage,
        "riichi_declaration_index": env.riichi_declaration_index,
        "missed_agari_doujun": env.missed_agari_doujun,
        "rinshan_draw_count": env.rinshan_draw_count,
        "pending_kan_dora_count": env.pending_kan_dora_count,
        "is_first_turn": env.is_first_turn,
        "is_rinshan_flag": env.is_rinshan_flag,
        "discard_from_hand": env.discard_from_hand,
        "discard_is_riichi": env.discard_is_riichi,
        "honba": env.honba,
        "riichi_sticks": env.riichi_sticks,
        "oya": env.oya,
        "round_wind": env.round_wind,
        "score_deltas": env.score_deltas,
        "turn_count": env.turn_count,
    }


@pytest.mark.parametrize("factory", [_ankan_env, _kakan_env])
def test_preview_matches_native_full_response_sets_without_exposing_any_other_data(factory):
    env, declaration = factory()
    actual = env.clone()
    actual.step({0: declaration})
    for player in (1, 2, 3):
        response = env.preview_kan_responses(0, declaration, player)
        expected = actual.get_observation(player).legal_actions()
        assert isinstance(response, list)
        assert _keys(response) == _keys(expected)
        assert all(
            action.action_type in (ActionType.RON, ActionType.PASS) and action.actor == player for action in response
        )


def test_ankan_only_offers_kokushi_not_other_ready_hands():
    env, declaration = _ankan_env()
    assert {action.action_type for action in env.preview_kan_responses(0, declaration, 1)} == {
        ActionType.RON,
        ActionType.PASS,
    }
    assert env.preview_kan_responses(0, declaration, 2) == []
    assert env.preview_kan_responses(0, declaration, 3) == []


def test_kakan_does_not_filter_to_one_actual_winner():
    env, declaration = _kakan_env()
    for player in (1, 2):
        assert {action.action_type for action in env.preview_kan_responses(0, declaration, player)} == {
            ActionType.RON,
            ActionType.PASS,
        }


def test_all_kokushi_responders_receive_their_own_complete_set():
    env, declaration = _ankan_env(kokushi_players=(1, 2), standard_players=())
    for player in (1, 2):
        assert {action.action_type for action in env.preview_kan_responses(0, declaration, player)} == {
            ActionType.RON,
            ActionType.PASS,
        }


@pytest.mark.parametrize("factory", [_ankan_env, _kakan_env])
def test_preview_does_not_change_original_state_input_action_or_future_execution(factory):
    env, declaration = factory()
    control = env.clone()
    before = _snapshot(env)
    original_action = _key(declaration)
    for player in (1, 2, 3, 1):
        env.preview_kan_responses(0, declaration, player)
        assert _snapshot(env) == before
        assert _key(declaration) == original_action
    env.step({0: declaration})
    control.step({0: declaration})
    assert _snapshot(env) == _snapshot(control)


@pytest.mark.parametrize("kind", ["ankan", "kakan"])
def test_no_robbery_returns_empty_even_when_native_clone_takes_a_replacement_draw(kind):
    env, declaration = (
        _ankan_env(kokushi_players=(), standard_players=(1, 2)) if kind == "ankan" else _kakan_env(ready_players=())
    )
    control = env.clone()
    before = _snapshot(env)
    for player in (1, 2, 3):
        assert env.preview_kan_responses(0, declaration, player) == []
    assert _snapshot(env) == before
    env.step({0: declaration})
    control.step({0: declaration})
    assert env.phase == Phase.WaitAct
    assert env.current_player == 0
    assert env.rinshan_draw_count > before["rinshan_draw_count"]
    assert _snapshot(env) == _snapshot(control)


@pytest.mark.parametrize("robbery", [True, False])
@pytest.mark.parametrize("kind", ["ankan", "kakan"])
def test_hidden_wall_cannot_influence_preview_responses(kind, robbery):
    if kind == "ankan":
        env, declaration = _ankan_env(kokushi_players=(1,) if robbery else (), standard_players=(2,))
    else:
        env, declaration = _kakan_env(ready_players=(1, 2) if robbery else ())
    alternate = env.clone()
    wall = alternate.wall
    hidden = [index for index, tile in enumerate(wall) if tile not in alternate.dora_indicators]
    wall[hidden[0]], wall[hidden[-1]] = wall[hidden[-1]], wall[hidden[0]]
    alternate.wall = wall
    assert alternate.wall != env.wall
    assert alternate.dora_indicators == env.dora_indicators
    assert alternate.hands == env.hands
    for player in (1, 2, 3):
        assert _keys(env.preview_kan_responses(0, declaration, player)) == _keys(
            alternate.preview_kan_responses(0, declaration, player)
        )


def test_replay_stale_offers_are_cleared_only_on_the_private_preview_clone():
    live, _ = _kakan_env(ready_players=(1,), stale_offer=True)
    replay = RiichiEnv("4p-red-single", seed=37, rule=GameRule.default_mjsoul())
    for event in live.mjai_log:
        replay.apply_event(event)
    assert replay.phase == Phase.WaitAct
    assert any(action.action_type == ActionType.PON for action in replay.current_claims[1])
    declaration = next(action for action in replay._get_legal_actions(0) if action.action_type == ActionType.KAKAN)
    before = _snapshot(replay)
    responses = replay.preview_kan_responses(0, declaration, 1)
    assert {action.action_type for action in responses} == {ActionType.RON, ActionType.PASS}
    assert _snapshot(replay) == before


@pytest.mark.parametrize("factory", [_ankan_env, _kakan_env])
def test_unspecified_actor_is_bound_only_in_the_native_copy(factory):
    env, declaration = factory()
    no_actor = Action(declaration.action_type, declaration.tile, declaration.consume_tiles)
    assert no_actor.actor is None
    assert _keys(env.preview_kan_responses(0, no_actor, 1)) == _keys(env.preview_kan_responses(0, declaration, 1))
    assert no_actor.actor is None


@pytest.mark.parametrize("factory", [_ankan_env, _kakan_env])
@pytest.mark.parametrize("bad", ["wrong_actor", "missing_tile", "missing_consumed", "wrong_consumed", "wrong_type"])
def test_only_a_complete_originally_legal_declaration_is_accepted(factory, bad):
    env, declaration = factory()
    bad_action = Action(declaration.action_type, declaration.tile, declaration.consume_tiles, 0)
    if bad == "wrong_actor":
        bad_action.actor = 1
    elif bad == "missing_tile":
        bad_action.tile = None
    elif bad == "missing_consumed":
        bad_action.consume_tiles = []
    elif bad == "wrong_consumed":
        bad_action.consume_tiles = [0, 1, 2, 3]
    else:
        bad_action.action_type = ActionType.DISCARD
    before = _snapshot(env)
    with pytest.raises(ValueError):
        env.preview_kan_responses(0, bad_action, 1)
    assert _snapshot(env) == before


@pytest.mark.parametrize("actor,responder", [(0, 0), (4, 1), (0, 4), (1, 2)])
def test_invalid_actor_or_responder_is_rejected(actor, responder):
    env, declaration = _ankan_env()
    with pytest.raises(ValueError):
        env.preview_kan_responses(actor, declaration, responder)


def test_three_player_environment_is_rejected():
    env = RiichiEnv("3p-red-single", seed=1, rule=GameRule.default_mjsoul())
    with pytest.raises(ValueError, match="four-player"):
        env.preview_kan_responses(0, Action(ActionType.ANKAN), 1)


def test_three_player_action_object_is_rejected():
    env, _ = _ankan_env()
    with pytest.raises(TypeError):
        env.preview_kan_responses(0, Action3P(ActionType.ANKAN), 1)


@pytest.mark.parametrize("bad", ["response_phase", "needs_draw", "done", "next_round"])
def test_invalid_pre_declaration_phase_is_rejected(bad):
    env, declaration = _ankan_env()
    if bad == "response_phase":
        env.phase = Phase.WaitResponse
    elif bad == "needs_draw":
        env.needs_tsumo = True
    elif bad == "done":
        env.is_done = True
    else:
        env.needs_initialize_next_round = True
    before = _snapshot(env)
    with pytest.raises(ValueError, match="pre-declaration"):
        env.preview_kan_responses(0, declaration, 1)
    assert _snapshot(env) == before
