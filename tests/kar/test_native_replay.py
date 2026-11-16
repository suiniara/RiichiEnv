from collections import Counter

import pytest

from riichienv import ActionType, GameRule, RiichiEnv, convert
from riichienv.kar.native_replay import NativeReplayBridge, NativeReplayContextError
from riichienv.kar.observation import ObserverLedger

READY = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "2s", "3s", "5p", "5p"]
OTHER_READY = ["1p", "2p", "3p", "4m", "5m", "6m", "7m", "8m", "9m", "2s", "3s", "9p", "9p"]


def _hands(known, reserve=()):
    pool = list(range(136))

    def take(value):
        tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == value)
        pool.remove(tile)
        return tile

    take("C")
    for value in reserve:
        take(value)
    hands = [[take(value) for value in known.get(player, [])] for player in range(4)]
    for hand in hands:
        while len(hand) < 13:
            hand.append(pool.pop(0))
    return [[convert.tid_to_mjai(tile) for tile in hand] for hand in hands]


class Replay:
    def __init__(self, hands):
        self.env = RiichiEnv("4p-red-half", seed=47, rule=GameRule.default_mjsoul())
        self.bridge = NativeReplayBridge(self.env)
        self.ledgers = [ObserverLedger(player, (0, 1, 2, 3)) for player in range(4)]
        self.apply({"type": "start_game"})
        self.apply(
            {
                "type": "start_kyoku",
                "bakaze": "E",
                "kyoku": 1,
                "honba": 0,
                "kyotaku": 0,
                "oya": 0,
                "scores": [25000] * 4,
                "dora_marker": "C",
                "tehais": hands,
            }
        )

    def before(self):
        return [ledger.snapshot() for ledger in self.ledgers]

    def apply(self, event):
        if event["type"] in {"ankan", "kakan"}:
            actor = event["actor"]
            declaration = self.env.get_observation(actor).select_action_from_mjai(event)
            assert declaration is not None
            self.bridge.preview_kan(actor, declaration)
        self.bridge.prepare(event, self.before())
        self.env.apply_event(event)
        self.bridge.after_apply(event)
        for ledger in self.ledgers:
            ledger.apply_event(event)

    def turn(self, actor, tile, *, riichi=False, first=False):
        self.apply({"type": "tsumo", "actor": actor, "pai": tile})
        if riichi:
            assert any(action.action_type == ActionType.RIICHI for action in self.env._get_legal_actions(actor))
            self.apply({"type": "reach", "actor": actor})
        self.apply({"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": not first})
        if riichi:
            self.apply({"type": "reach_accepted", "actor": actor})


def _claims(cache):
    if cache is None:
        return None
    return {
        player: tuple((int(a.action_type), a.tile, tuple(a.consume_tiles), a.actor) for a in actions)
        for player, actions in cache.items()
    }


@pytest.mark.parametrize("mode,players", [("4p-red-single", 4), ("3p-red-single", 3)])
@pytest.mark.parametrize("field", ["ippatsu_cycle", "double_riichi_declared", "missed_agari_riichi"])
def test_native_context_flags_are_real_checked_length_properties(mode, players, field):
    env = RiichiEnv(mode, seed=17, rule=GameRule.default_mjsoul())
    values = [True] + [False] * (players - 1)
    setattr(env, field, values)
    assert getattr(env, field) == values
    returned = getattr(env, field)
    returned[0] = False
    assert getattr(env, field) == values
    for size in (0, players - 1, players + 1):
        with pytest.raises(ValueError, match="requires"):
            setattr(env, field, [False] * size)
        assert getattr(env, field) == values


def test_bridge_rejects_three_player_context():
    env = RiichiEnv("3p-red-single", seed=1)
    with pytest.raises(NativeReplayContextError, match="four-player"):
        NativeReplayBridge(env)


def test_first_discard_double_and_ippatsu_live_until_own_next_discard():
    replay = Replay(_hands({0: READY}, ("F", "P", "N", "W", "9m")))
    replay.turn(0, "F", riichi=True, first=True)
    assert replay.env.double_riichi_declared == [True, False, False, False]
    assert replay.env.ippatsu_cycle == [True, False, False, False]
    for actor, tile in ((1, "P"), (2, "N"), (3, "W")):
        replay.turn(actor, tile)
        assert replay.env.ippatsu_cycle[0] is True
    replay.apply({"type": "tsumo", "actor": 0, "pai": "9m"})
    assert replay.env.ippatsu_cycle[0] is True
    replay.apply({"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": True})
    assert replay.env.ippatsu_cycle[0] is False
    assert replay.env.double_riichi_declared[0] is True


def test_second_discard_riichi_is_not_double():
    replay = Replay(_hands({0: READY}, ("F", "P", "N", "W", "9m")))
    for actor, tile in enumerate(("F", "P", "N", "W")):
        replay.turn(actor, tile, first=actor == 0)
    replay.turn(0, "9m", riichi=True)
    assert replay.env.double_riichi_declared[0] is False
    assert replay.env.ippatsu_cycle[0] is True


@pytest.mark.parametrize("kind,consumed", [("pon", ["2m", "2m"]), ("chi", ["3m", "4m"])])
def test_accepted_call_cancels_ippatsu_and_committed_call_prevents_double(kind, consumed):
    hands = _hands({0: READY, 1: ["2m"], 2: consumed}, ("F", "P"))
    replay = Replay(hands)
    replay.turn(0, "F", riichi=True, first=True)
    replay.apply({"type": "tsumo", "actor": 1, "pai": "P"})
    replay.apply({"type": "dahai", "actor": 1, "pai": "2m", "tsumogiri": False})
    replay.apply({"type": kind, "actor": 2, "target": 1, "pai": "2m", "consumed": consumed})
    assert replay.env.ippatsu_cycle == [False] * 4
    second = Replay(_hands({0: ["P"], 1: ["P", "P", "9m"], 2: READY}, ("F", "N")))
    second.apply({"type": "tsumo", "actor": 0, "pai": "F"})
    second.apply({"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False})
    second.apply({"type": "pon", "actor": 1, "target": 0, "pai": "P", "consumed": ["P", "P"]})
    second.apply({"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": False})
    second.turn(2, "N", riichi=True)
    assert second.env.double_riichi_declared[2] is False
    assert second.env.ippatsu_cycle[2] is True


def _kakan_prefix():
    hands = _hands(
        {0: ["1s", "1s"], 1: READY, 2: OTHER_READY, 3: ["1s"]}, ("1s", "F", "P", "N", "W", "9m", "8m", "7m", "6m", "9p")
    )
    replay = Replay(hands)
    for actor, tile in ((0, "F"), (1, "P"), (2, "N")):
        replay.turn(actor, tile, first=actor == 0)
    replay.apply({"type": "tsumo", "actor": 3, "pai": "W"})
    replay.apply({"type": "dahai", "actor": 3, "pai": "1s", "tsumogiri": False})
    replay.apply({"type": "pon", "actor": 0, "target": 3, "pai": "1s", "consumed": ["1s", "1s"]})
    discard = next(tile for tile in hands[0] if tile != "1s")
    replay.apply({"type": "dahai", "actor": 0, "pai": discard, "tsumogiri": False})
    replay.turn(1, "9m", riichi=True)
    replay.turn(2, "8m")
    replay.turn(3, "7m")
    replay.apply({"type": "tsumo", "actor": 0, "pai": "1s"})
    return replay


def test_kan_declaration_keeps_ippatsu_and_cache_does_not_follow_true_winner():
    replay = _kakan_prefix()
    assert replay.env.ippatsu_cycle[1] is True
    replay.apply({"type": "kakan", "actor": 0, "pai": "1s"})
    cache = _claims(replay.bridge.kan_responses)
    assert set(cache) == {1, 2, 3}
    assert any(action.action_type == ActionType.RON for action in replay.bridge.kan_responses[1])
    assert any(action.action_type == ActionType.RON for action in replay.bridge.kan_responses[2])
    assert replay.env.ippatsu_cycle[1] is True
    caller_copy = replay.bridge.kan_responses
    caller_copy[1].clear()
    assert _claims(replay.bridge.kan_responses) == cache
    replay.apply({"type": "hora", "actor": 2, "target": 0, "pai": "1s"})
    assert _claims(replay.bridge.kan_responses) == cache
    assert replay.env.ippatsu_cycle[1] is True
    replay.apply({"type": "end_kyoku"})
    assert replay.bridge.kan_responses is None


def test_kan_commit_cancels_ippatsu_marks_all_ron_declines_and_switches_rinshan():
    replay = _kakan_prefix()
    replay.apply({"type": "kakan", "actor": 0, "pai": "1s"})
    replay.apply({"type": "dora", "dora_marker": "9p"})
    assert replay.env.ippatsu_cycle == [False] * 4
    assert replay.env.missed_agari_doujun[1:3] == [True, True]
    assert replay.env.missed_agari_riichi[1:3] == [True, False]
    assert replay.bridge.kan_responses is None
    replay.apply({"type": "tsumo", "actor": 0, "pai": "6m"})
    assert replay.env.is_rinshan_flag is True
    replay.apply({"type": "dahai", "actor": 0, "pai": "6m", "tsumogiri": True})
    assert replay.env.is_rinshan_flag is False
    replay.apply({"type": "tsumo", "actor": 1, "pai": "9p"})
    assert replay.env.is_rinshan_flag is False
    assert replay.env.missed_agari_doujun[1] is True
    replay.apply({"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": True})
    assert replay.env.missed_agari_doujun[1] is False
    assert replay.env.missed_agari_riichi[1] is True


def test_no_ron_next_draw_marks_temp_but_own_draw_does_not_reset_it():
    replay = Replay(_hands({0: ["1s"], 1: READY}, ("F", "9m")))
    replay.apply({"type": "tsumo", "actor": 0, "pai": "F"})
    replay.apply({"type": "dahai", "actor": 0, "pai": "1s", "tsumogiri": False})
    assert any(a.action_type == ActionType.RON for a in replay.env._get_legal_actions(1))
    replay.apply({"type": "tsumo", "actor": 1, "pai": "9m"})
    assert replay.env.missed_agari_doujun[1] is True
    replay.apply({"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True})
    assert replay.env.missed_agari_doujun[1] is False


def test_accepted_call_marks_every_prior_ron_offer_not_only_the_caller():
    replay = Replay(_hands({0: ["1s"], 1: READY, 2: ["1s", "1s"], 3: OTHER_READY}, ("F", "9m")))
    replay.apply({"type": "tsumo", "actor": 0, "pai": "F"})
    replay.apply({"type": "dahai", "actor": 0, "pai": "1s", "tsumogiri": False})
    offered = {
        player
        for player in range(4)
        if any(a.action_type == ActionType.RON for a in replay.env._get_legal_actions(player))
    }
    assert {1, 3} <= offered
    replay.apply({"type": "pon", "actor": 2, "target": 0, "pai": "1s", "consumed": ["1s", "1s"]})
    assert replay.env.missed_agari_doujun == [player in offered for player in range(4)]
    discard = next(value for value in replay.ledgers[2].snapshot().hand if value // 4 not in {18, 21})
    replay.apply({"type": "dahai", "actor": 2, "pai": convert.tid_to_mjai(discard), "tsumogiri": False})
    replay.apply({"type": "tsumo", "actor": 3, "pai": "9m"})
    assert replay.env.missed_agari_doujun[3] is True
    replay.apply({"type": "dahai", "actor": 3, "pai": "9m", "tsumogiri": True})
    assert replay.env.missed_agari_doujun == [False, True, False, False]


def test_collecting_ron_offers_does_not_consume_observation_event_deltas():
    replay = Replay(_hands({0: ["1s"], 1: READY}, ("F",)))
    replay.apply({"type": "tsumo", "actor": 0, "pai": "F"})
    replay.apply({"type": "dahai", "actor": 0, "pai": "1s", "tsumogiri": False})
    control = RiichiEnv("4p-red-half", seed=47, rule=GameRule.default_mjsoul())
    for event in replay.env.mjai_log:
        control.apply_event(event)
    assert replay.env.get_observation(1).events == control.get_observation(1).events


def test_prepare_only_changes_context_flags_and_protocol_order_is_checked():
    replay = Replay(_hands({0: READY}, ("F",)))
    event = {"type": "tsumo", "actor": 0, "pai": "F"}
    before = (replay.env.hands, replay.env.melds, replay.env.scores(), replay.env.wall, replay.env.current_claims)
    replay.bridge.prepare(event, replay.before())
    assert (
        replay.env.hands,
        replay.env.melds,
        replay.env.scores(),
        replay.env.wall,
        replay.env.current_claims,
    ) == before
    with pytest.raises(NativeReplayContextError, match="after_apply"):
        replay.bridge.prepare(event, replay.before())
    with pytest.raises(NativeReplayContextError, match="match"):
        replay.bridge.after_apply({"type": "tsumo", "actor": 1, "pai": "F"})
    replay.env.apply_event(event)
    replay.bridge.after_apply(event)


def test_new_round_resets_every_context_flag_and_response_cache():
    env = RiichiEnv("4p-red-half", seed=47, rule=GameRule.default_mjsoul())
    for field in ("ippatsu_cycle", "double_riichi_declared", "missed_agari_doujun", "missed_agari_riichi"):
        setattr(env, field, [True] * 4)
    env.is_rinshan_flag = True
    bridge = NativeReplayBridge(env)
    snapshots = [ObserverLedger(player, (0, 1, 2, 3)).snapshot() for player in range(4)]
    event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000] * 4,
        "dora_marker": "C",
        "tehais": _hands({0: READY}),
    }
    bridge.prepare(event, snapshots)
    env.apply_event(event)
    bridge.after_apply(event)
    assert env.ippatsu_cycle == [False] * 4
    assert env.double_riichi_declared == [False] * 4
    assert env.missed_agari_doujun == [False] * 4
    assert env.missed_agari_riichi == [False] * 4
    assert env.is_rinshan_flag is False
    assert bridge.kan_responses is None


def test_kan_requires_a_matching_pre_declaration_preview():
    replay = _kakan_prefix()
    with pytest.raises(NativeReplayContextError, match="preview"):
        replay.bridge.prepare({"type": "kakan", "actor": 0, "pai": "1s"}, replay.before())
    declaration = next(a for a in replay.env._get_legal_actions(0) if a.action_type == ActionType.KAKAN)
    replay.bridge.preview_kan(0, declaration)
    with pytest.raises(NativeReplayContextError, match="tile differs"):
        replay.bridge.prepare({"type": "kakan", "actor": 0, "pai": "2s"}, replay.before())
    assert Counter(a.action_type for a in replay.bridge.kan_responses[1])[ActionType.RON] == 1
