import inspect
from dataclasses import FrozenInstanceError, replace
from itertools import permutations, product

import pytest

from riichienv import Action, ActionType, Conditions, GameRule, HandEvaluator, Meld, MeldType, Phase, RiichiEnv, convert
from riichienv.kar.payments import EvaluatedWin, draw_payments, end_game_pot, win_payments


def _ordinary_hand():
    return list(HandEvaluator.hand_from_text("123m456p789s111z2z").tiles_136), [], 113


def _daisangen_hand():
    melds = [
        Meld(MeldType.Pon, [124, 125, 126], True, 1),
        Meld(MeldType.Pon, [128, 129, 130], True, 1),
        Meld(MeldType.Pon, [132, 133, 134], True, 3),
    ]
    return [108, 109, 110, 112], melds, 113


def _daisuushi_hand():
    melds = [
        Meld(MeldType.Pon, [108, 109, 110], True, 1),
        Meld(MeldType.Pon, [112, 113, 114], True, 1),
        Meld(MeldType.Pon, [116, 117, 118], True, 1),
        Meld(MeldType.Pon, [120, 121, 122], True, 3),
    ]
    return [124], melds, 125


def _evaluate(winner, *, oya=0, tsumo=False, hand=None, dora=(), honba=0):
    tiles, melds, win_tile = hand or _ordinary_hand()
    result = HandEvaluator(list(tiles), list(melds)).calc(
        win_tile,
        list(dora),
        Conditions(tsumo=tsumo, player_wind=(winner - oya) % 4, round_wind=0, honba=honba),
    )
    assert result.is_win
    return result


def _live_payments(winner, *, oya, target, honba, sticks, hand, liable=None, liable_yaku=None):
    tiles, melds, win_tile = hand
    env = RiichiEnv("4p-red-single", seed=7, rule=GameRule.default_mjsoul())
    env.reset(oya=oya, honba=honba)
    env.set_scores([25000] * 4)
    env.riichi_sticks = sticks
    env.dora_indicators = []
    env.is_first_turn = False
    occupied = set(tiles) | {win_tile} | {tile for meld in melds for tile in meld.tiles}
    pool = [tile for tile in range(136) if tile not in occupied]
    hands = []
    offset = 0
    for player in range(4):
        if player == winner:
            hands.append(list(tiles) + ([win_tile] if target is None else []))
        else:
            hands.append(pool[offset : offset + 13] + ([win_tile] if player == target else []))
            offset += 13
    env.hands = hands
    env.melds = [list(melds) if player == winner else [] for player in range(4)]
    if liable is not None:
        pao = env.pao
        pao[winner] = {liable_yaku: liable}
        env.pao = pao
    actor = winner if target is None else target
    env.current_player = actor
    env.drawn_tile = win_tile
    env.needs_tsumo = False
    env.phase = Phase.WaitAct
    env.active_players = [actor]
    if target is None:
        env.step({winner: Action(ActionType.TSUMO, win_tile, [], winner)})
    else:
        env.step({target: Action(ActionType.DISCARD, win_tile, [], target)})
        assert env.phase == Phase.WaitResponse
        env.step({winner: Action(ActionType.RON, win_tile, [], winner)})
    assert winner in env.win_results
    return tuple(env.score_deltas)


@pytest.mark.parametrize("oya,winner", list(product(range(4), repeat=2)))
@pytest.mark.parametrize("honba,sticks", [(0, 0), (2, 3)])
def test_standard_tsumo_uses_independent_base_fields(oya, winner, honba, sticks):
    result = _evaluate(winner, oya=oya, tsumo=True)
    paid = win_payments([EvaluatedWin(winner, result)], oya=oya, target=None, honba=honba, riichi_sticks=sticks)
    expected = [0] * 4
    for player in range(4):
        if player != winner:
            expected[player] = -(result.tsumo_agari_oya if player == oya else result.tsumo_agari_ko) - honba * 100
    expected[winner] = -sum(expected) + sticks * 1000
    assert paid == tuple(expected)
    assert sum(paid) == sticks * 1000


@pytest.mark.parametrize(
    "oya,winner,target",
    [(oya, winner, target) for oya in range(4) for winner in range(4) for target in range(4) if winner != target],
)
def test_standard_ron_pays_only_discarder(oya, winner, target):
    result = _evaluate(winner, oya=oya)
    paid = win_payments([EvaluatedWin(winner, result)], oya=oya, target=target, honba=2, riichi_sticks=1)
    expected = [0] * 4
    expected[winner] = result.ron_agari + 600 + 1000
    expected[target] = -result.ron_agari - 600
    assert paid == tuple(expected)


def _three_ron_wins():
    texts = {1: "12m345p345s111z55z", 2: "12m678p678s666z55z", 3: "12m678m678p777z44z"}
    wins = []
    for player, text in texts.items():
        result = HandEvaluator.hand_from_text(text).calc(
            convert.mpsz_to_tid("3m"), [], Conditions(player_wind=player, round_wind=0)
        )
        assert result.is_win and result.ron_agari == 1600
        wins.append(EvaluatedWin(player, result))
    return wins


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_triple_ron_first_closest_takes_honba_and_pot_once(order):
    wins = _three_ron_wins()
    paid = win_payments([wins[index] for index in order], oya=0, target=0, honba=2, riichi_sticks=2)
    assert paid == (-5400, 4200, 1600, 1600)


def _physical_hands(texts):
    used = set()
    hands = []
    for text in texts:
        hand = []
        for tile in HandEvaluator.hand_from_text(text).tiles_136:
            choices = (tile,) if tile in (16, 52, 88) else range(tile // 4 * 4, tile // 4 * 4 + 4)
            chosen = next(
                candidate
                for candidate in choices
                if candidate not in used and (candidate == tile or candidate not in (16, 52, 88))
            )
            used.add(chosen)
            hand.append(chosen)
        hands.append(hand)
    return hands, used


@pytest.mark.parametrize("winners", [(1, 3), (1, 2, 3)])
def test_multiple_ron_honba_and_pot_equal_frozen_live_engine(winners):
    texts = [
        "3456789m129p129s",
        "12m345p345s111z55z",
        "12m678p678s666z55z",
        "12m678m678p777z44z",
    ]
    hands, used = _physical_hands(texts)
    drawn = next(tile for tile in range(32, 36) if tile not in used)
    hands[0].append(drawn)
    source_tile = next(tile for tile in hands[0] if tile // 4 == 2)
    env = RiichiEnv("4p-red-single", seed=5, rule=GameRule.default_mjsoul())
    env.reset(oya=0, honba=2)
    env.hands = hands
    env.dora_indicators = []
    env.riichi_sticks = 2
    env.is_first_turn = False
    env.current_player = 0
    env.drawn_tile = drawn
    env.needs_tsumo = False
    env.phase = Phase.WaitAct
    env.active_players = [0]
    expected = win_payments(
        [win for win in _three_ron_wins() if win.player_id in winners],
        oya=0,
        target=0,
        honba=2,
        riichi_sticks=2,
    )
    env.step({0: Action(ActionType.DISCARD, source_tile, [], 0)})
    assert env.phase == Phase.WaitResponse
    env.step({winner: Action(ActionType.RON, source_tile, [], winner) for winner in winners})
    assert set(env.win_results) == set(winners)
    assert tuple(env.score_deltas) == expected


def test_double_ron_closest_can_be_nonlowest_seat():
    wins = [EvaluatedWin(player, _evaluate(player)) for player in (1, 3)]
    paid = win_payments(wins, oya=0, target=2, honba=1, riichi_sticks=2)
    assert paid[3] == wins[1].result.ron_agari + 300 + 2000
    assert paid[1] == wins[0].result.ron_agari
    assert paid[2] == -wins[0].result.ron_agari - wins[1].result.ron_agari - 300
    assert paid[0] == 0


@pytest.mark.parametrize("winner", [0, 2])
@pytest.mark.parametrize("target", [None, 1])
@pytest.mark.parametrize("hand_factory,units", [(_daisangen_hand, 1), (_daisuushi_hand, 2)])
def test_composite_pao_only_charges_responsibility_portion(winner, target, hand_factory, units):
    result = _evaluate(winner, tsumo=target is None, hand=hand_factory())
    assert result.yakuman and result.han == 13 * (units + 1)
    paid = win_payments([EvaluatedWin(winner, result, 3, units)], oya=0, target=target, honba=2, riichi_sticks=1)
    unit = 48000 if winner == 0 else 32000
    if target is None:
        pao_only = units * unit + 600
        normal_share = 16000 if winner == 0 else 8000
        assert paid[3] == -pao_only - normal_share
        assert paid[winner] == (units + 1) * unit + 600 + 1000
    else:
        assert paid[3] == -(units * unit // 2 + 600)
        assert paid[1] == -result.ron_agari + units * unit // 2
        assert paid[winner] == result.ron_agari + 600 + 1000
    assert sum(paid) == 1000


def test_liable_discarder_is_charged_once_for_both_roles():
    result = _evaluate(2, hand=_daisangen_hand())
    paid = win_payments([EvaluatedWin(2, result, 1, 1)], oya=0, target=1, honba=3, riichi_sticks=2)
    assert paid == (0, -64900, 66900, 0)


def test_liability_can_cross_another_winners_payment():
    responsible_win = EvaluatedWin(1, _evaluate(1, hand=_daisangen_hand()), 2, 1)
    other_win = EvaluatedWin(2, _evaluate(2))
    paid = win_payments([other_win, responsible_win], oya=0, target=0, honba=1, riichi_sticks=2)
    assert sum(paid) == 2000
    assert paid[2] == other_win.result.ron_agari - 16000 - 300


@pytest.mark.parametrize("tsumo", [False, True])
def test_kazoe_many_han_is_not_multiple_true_yakuman(tsumo):
    hand = (list(HandEvaluator.hand_from_text("1111235678885m").tiles_136), [], 19)
    result = HandEvaluator(hand[0], []).calc(
        19,
        [32, 33, 34, 35, 25],
        Conditions(tsumo=tsumo, riichi=not tsumo, ippatsu=not tsumo, player_wind=1, round_wind=0),
    )
    assert result.is_win and result.han >= 26 and not result.yakuman
    paid = win_payments([EvaluatedWin(1, result)], oya=0, target=None if tsumo else 0, honba=0, riichi_sticks=0)
    assert paid[1] == 32000
    with pytest.raises(ValueError, match="true yakuman"):
        win_payments([EvaluatedWin(1, result, 2, 1)], oya=0, target=None if tsumo else 0, honba=0, riichi_sticks=0)


@pytest.mark.parametrize("winner", [0, 1])
@pytest.mark.parametrize("tsumo", [False, True])
def test_default_mjsoul_double_kokushi_is_preserved(winner, tsumo):
    result = HandEvaluator.hand_from_text("19m19p19s1234567z").calc(
        1, [], Conditions(tsumo=tsumo, player_wind=winner, round_wind=0)
    )
    assert result.yakuman and result.han == 26
    paid = win_payments([EvaluatedWin(winner, result)], oya=0, target=None if tsumo else 3, honba=0, riichi_sticks=0)
    assert paid[winner] == (96000 if winner == 0 else 64000)


@pytest.mark.parametrize("winner,target", [(0, None), (2, None), (0, 1), (2, 1)])
@pytest.mark.parametrize("honba", [0, 2])
def test_standard_payments_equal_frozen_live_engine(winner, target, honba):
    hand = _ordinary_hand()
    result = _evaluate(winner, tsumo=target is None, hand=hand)
    expected = win_payments([EvaluatedWin(winner, result)], oya=0, target=target, honba=honba, riichi_sticks=3)
    assert expected == _live_payments(winner, oya=0, target=target, honba=honba, sticks=3, hand=hand)


@pytest.mark.parametrize("winner,target", [(0, None), (2, None), (0, 1), (2, 1)])
@pytest.mark.parametrize("hand_factory,units,yaku", [(_daisangen_hand, 1, 37), (_daisuushi_hand, 2, 50)])
@pytest.mark.parametrize("honba", [0, 2])
def test_composite_pao_payments_equal_frozen_live_engine(winner, target, hand_factory, units, yaku, honba):
    hand = hand_factory()
    result = _evaluate(winner, tsumo=target is None, hand=hand)
    expected = win_payments(
        [EvaluatedWin(winner, result, 3, units)], oya=0, target=target, honba=honba, riichi_sticks=3
    )
    assert expected == _live_payments(
        winner, oya=0, target=target, honba=honba, sticks=3, hand=hand, liable=3, liable_yaku=yaku
    )


@pytest.mark.parametrize("tenpai", list(product((False, True), repeat=4)))
def test_all_noten_pool_patterns_and_abortive_draws(tenpai):
    paid = draw_payments(tenpai, oya=0)
    expected_by_count = {0: (0, 0), 1: (3000, -1000), 2: (1500, -1500), 3: (1000, -3000), 4: (0, 0)}
    gain, loss = expected_by_count[sum(tenpai)]
    assert paid == tuple(gain if ready else loss for ready in tenpai)
    assert sum(paid) == 0
    assert draw_payments(tenpai, oya=0, abortive=True) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "winners,expected",
    [
        ((0,), (12000, -4000, -4000, -4000)),
        ((1,), (-4000, 8000, -2000, -2000)),
        ((0, 1), (8000, 4000, -6000, -6000)),
        ((1, 2), (-8000, 6000, 6000, -4000)),
        ((0, 1, 2, 3), (0, 0, 0, 0)),
    ],
)
def test_nagashi_uses_mangan_tsumo_without_honba_pot_or_noten_mix(winners, expected):
    assert draw_payments([False] * 4, oya=0, nagashi_winners=winners) == expected
    assert draw_payments([True] * 4, oya=0, nagashi_winners=tuple(reversed(winners))) == expected


@pytest.mark.parametrize("order", list(permutations(range(4))))
def test_final_pot_uses_explicit_initial_order_for_ties(order):
    paid = end_game_pot([25000] * 4, 3, order)
    assert paid[order[0]] == 3000
    assert sum(paid) == 3000
    assert sum(value != 0 for value in paid) == 1


def test_final_pot_uses_scores_before_initial_order_and_allows_negative_scores():
    assert end_game_pot([-1000, 30000, 30000, 41000], 2, [2, 1, 0, 3]) == (0, 0, 0, 2000)
    assert end_game_pot([-1000, 30000, 30000, 30000], 2, [2, 1, 0, 3]) == (0, 0, 2000, 0)
    assert end_game_pot([25000] * 4, 0, [3, 2, 1, 0]) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("oya", -1),
        ("oya", 4),
        ("oya", True),
        ("target", 4),
        ("target", True),
        ("honba", -1),
        ("honba", True),
        ("riichi_sticks", -1),
        ("riichi_sticks", False),
    ],
)
def test_invalid_win_parameters_are_rejected(field, value):
    kwargs = {"oya": 0, "target": 0, "honba": 0, "riichi_sticks": 0}
    kwargs[field] = value
    with pytest.raises(ValueError):
        win_payments([EvaluatedWin(1, _evaluate(1))], **kwargs)


def test_invalid_winner_and_role_configurations_are_rejected():
    good = EvaluatedWin(1, _evaluate(1))
    for wins, target in (([], 0), ([good, good], 0), ([good], 1), ([good, EvaluatedWin(2, _evaluate(2))], None)):
        with pytest.raises(ValueError):
            win_payments(wins, oya=0, target=target, honba=0, riichi_sticks=0)
    for bad in (
        replace(good, player_id=4),
        replace(good, player_id=True),
        replace(good, liable_player=1, liable_yakuman_units=1),
        replace(good, liable_player=2, liable_yakuman_units=1),
    ):
        with pytest.raises(ValueError):
            win_payments([bad], oya=0, target=0, honba=0, riichi_sticks=0)
    with pytest.raises(ValueError, match="zero-honba"):
        win_payments([good], oya=1, target=0, honba=0, riichi_sticks=0)
    with pytest.raises(ValueError, match="zero-honba"):
        win_payments([EvaluatedWin(1, _evaluate(1, honba=1))], oya=0, target=0, honba=1, riichi_sticks=0)
    with pytest.raises(ValueError, match="zero-honba"):
        win_payments([EvaluatedWin(1, _evaluate(1, tsumo=True))], oya=0, target=0, honba=0, riichi_sticks=0)


@pytest.mark.parametrize("liable,units", [(None, 1), (2, 0), (2, -1), (2, True), (2, 3), (4, 1), (True, 1)])
def test_invalid_liability_configuration_is_rejected(liable, units):
    result = _evaluate(1, hand=_daisangen_hand())
    with pytest.raises(ValueError):
        win_payments([EvaluatedWin(1, result, liable, units)], oya=0, target=0, honba=0, riichi_sticks=0)


def test_nonresponsibility_yakuman_cannot_acquire_pao():
    result = HandEvaluator.hand_from_text("19m19p19s1234567z").calc(1, [], Conditions(player_wind=1))
    with pytest.raises(ValueError, match="responsibility"):
        win_payments([EvaluatedWin(1, result, 2, 1)], oya=0, target=0, honba=0, riichi_sticks=0)


@pytest.mark.parametrize(
    "tenpai,kwargs",
    [
        ([False] * 3, {}),
        ([0] * 4, {}),
        ([False] * 4, {"oya": 4}),
        ([False] * 4, {"nagashi_winners": (1, 1)}),
        ([False] * 4, {"nagashi_winners": (4,)}),
        ([False] * 4, {"abortive": 1}),
        ([False] * 4, {"abortive": True, "nagashi_winners": (1,)}),
    ],
)
def test_invalid_draw_configuration_is_rejected(tenpai, kwargs):
    with pytest.raises(ValueError):
        draw_payments(tenpai, **{"oya": 0, **kwargs})


@pytest.mark.parametrize(
    "scores,pot,order",
    [
        ([25000] * 3, 1, [0, 1, 2, 3]),
        ([True] * 4, 1, [0, 1, 2, 3]),
        ([25000] * 4, -1, [0, 1, 2, 3]),
        ([25000] * 4, True, [0, 1, 2, 3]),
        ([25000] * 4, 1, [0, 1, 2, 2]),
        ([25000] * 4, 1, [0, 1, 2, 4]),
    ],
)
def test_invalid_end_pot_configuration_is_rejected(scores, pot, order):
    with pytest.raises(ValueError):
        end_game_pot(scores, pot, order)


def test_evaluated_win_is_frozen_and_server_labels_have_no_input_channel():
    win = EvaluatedWin(1, _evaluate(1))
    with pytest.raises(FrozenInstanceError):
        win.player_id = 2
    assert tuple(inspect.signature(win_payments).parameters) == ("wins", "oya", "target", "honba", "riichi_sticks")
    assert tuple(inspect.signature(draw_payments).parameters) == ("tenpai", "oya", "nagashi_winners", "abortive")
    assert tuple(inspect.signature(end_game_pot).parameters) == ("scores", "riichi_sticks", "initial_seat_order")

    class Poison:
        def __int__(self):
            raise AssertionError("Server labels must never be consulted")

    for label in ("server_han", "server_fu", "server_scores", "server_deltas", "server_points"):
        with pytest.raises(TypeError):
            win_payments([win], oya=0, target=0, honba=0, riichi_sticks=0, **{label: Poison()})
