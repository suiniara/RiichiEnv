"""Replay the no-yaku miss from a native 136-tile wall through full-game export."""

import pytest

from riichienv import Phase
from riichienv.kar.actions import legal_candidates
from riichienv.kar.contracts import MJSOUL_RULESET, content_hash
from riichienv.kar.mjsoul import DecodedGame, normalize_records
from riichienv.kar.native_replay import NativeReplayBridge, NativeReplayInterfaceError
from riichienv.kar.observation import ObserverLedger
from tests.kar.test_decision_time_parity import _Episode, _server_records
from tests.kar.test_input_semantics import record

SEAT, PASS, RON_FROM_3 = 1, 221, 544


def _state(env, observations=None):
    candidates = ()
    if SEAT in env.active_players:
        obs = (observations if observations is not None else env.get_observations([SEAT]))[SEAT]
        source = env.last_discard[0] if env.phase == Phase.WaitResponse else None
        candidates = tuple(item.global_id for item in legal_candidates(obs, source_player=source))
    return env.missed_agari_doujun[SEAT], env.missed_agari_riichi[SEAT], candidates


def _complete_game(ep):
    for _ in range(1500):
        ep.resolve()
        if ep.env.done():
            break
        ep.act()
    else:
        pytest.fail("Bounded native hanchan did not finish")
    events = ep.live.events
    assert events[-1]["type"] == "end_game"
    starts = [i for i, event in enumerate(events) if event["type"] == "start_kyoku"]
    assert len(starts) >= 8
    records = [
        record(start + item.index - 1, item.name.removeprefix("lq."), item.data)
        for start, stop in zip(starts, [*starts[1:], len(events) - 1])
        for item in _server_records([events[0], *events[start:stop]])
    ]
    game_id = "synthetic-native-no-yaku-hanchan"
    normalized = normalize_records(records, game_id=game_id)
    # The native match ends on an exhaustive draw, an older legal format without
    # explicit GameEnd scores. The independent certificate must prove termination.
    assert [item["type"] for item in normalized] == [
        item["type"] for item in events[:-1] if item["type"] != "kan_commit"
    ]
    final_scores = events[-1]["scores"]
    assert list(ep.env.scores()) == final_scores
    metadata = {
        "uuid": game_id,
        "mode": 2,
        "category": 2,
        "account_seats": [0, 1, 2, 3],
        "start_time": 1,
        "end_time": 2,
        "final_standings": [{"seat": seat, "part_point_1": points} for seat, points in enumerate(final_scores)],
    }
    return DecodedGame(
        content_hash([item.data for item in records]), "b" * 64, game_id, metadata, tuple(records), normalized, "c" * 64
    )


@pytest.fixture(scope="module")
def episode():
    initial = ("3m", "4m", "4p", "5p", "6p", "6p", "7p", "8p", "2s", "3s", "5m", "5m", "F")
    # Give another seat a real call on each checked discard. This stops the
    # native step at the actual response boundary, not after an automatic draw.
    ep = _Episode(
        {0: ("5s", "6s"), SEAT: initial, 2: ("N", "N"), 3: ("2s", "3s")}, ("2m", "1s", "4s", "W", "N", "P", "4s")
    )
    ep.act(tile="2m")
    ep.resolve("CHI", seat=SEAT)
    ep.act(tile="F")
    states = {}

    def discard(seat, tile):
        start = len(ep.live.events)
        ep.act(tile=tile)
        index = ep._find_result(start, seat, "dahai")
        assert ep.env.phase == Phase.WaitResponse and index == len(ep.live.events) - 1
        states[index] = _state(ep.env, ep.obs)
        return index

    no_yaku = discard(2, "1s")
    blocked = discard(3, "4s")
    ep.act(tile="W")
    ep.resolve()
    draw = ep._find_result(blocked + 1, SEAT, "tsumo")
    states[draw] = _state(ep.env, ep.obs)
    cleared = discard(SEAT, "N")
    ep.act(tile="P")
    offered = discard(3, "4s")
    ep.resolve()
    assert states[no_yaku] == states[blocked] == (True, False, ())
    assert states[draw][:2] == (True, False) and states[draw][2]
    assert states[cleared] == (False, False, ())
    assert states[offered] == (False, False, (PASS, RON_FROM_3))
    game = _complete_game(ep)
    # Pair live and normalized state by source result identity, not a supplied
    # offline cutoff. The first draw can share a record with its round start.
    indices = {(event.get("source_record_index"), event["type"]): index for index, event in enumerate(game.events)}
    states = {indices[i, ep.live.events[i]["type"]]: value for i, value in states.items()}
    return game, states, indices[blocked, "dahai"], indices[offered, "dahai"]


@pytest.mark.parametrize("bridge_enabled", [False, True], ids=["native-mjai", "kar-bridge"])
def test_no_yaku_same_turn_furiten_matches_native_legal_sets(episode, bridge_enabled):
    game, states, _, _ = episode
    env = MJSOUL_RULESET.create_env(seed=0)
    bridge = NativeReplayBridge(env) if bridge_enabled else None
    ledgers = [ObserverLedger(seat, (0, 1, 2, 3)) for seat in range(4)]
    for index, event in enumerate(game.events[: max(states) + 1]):
        if bridge is not None:
            bridge.prepare(event, [ledger.snapshot() for ledger in ledgers])
        env.apply_event(event)
        if bridge is not None:
            bridge.after_apply(event)
            for ledger in ledgers:
                ledger.apply_event(event)
        if index in states:
            assert _state(env) == states[index], event


def test_kar_replay_fails_closed_with_pre_furiten_extension():
    # A double simulates only a missing capability, not any game state/flag.
    from types import SimpleNamespace  # noqa: PLC0415

    env = SimpleNamespace(
        ippatsu_cycle=[],
        double_riichi_declared=[],
        missed_agari_doujun=[],
        missed_agari_riichi=[],
        is_rinshan_flag=False,
        preview_kan_responses=lambda *_: [],
    )
    with pytest.raises(NativeReplayInterfaceError, match="rebuild.*_record_replay_kan_ron_declines"):
        NativeReplayBridge(env)
