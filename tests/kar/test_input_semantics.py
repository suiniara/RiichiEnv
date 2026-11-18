"""Bounded source/clock parity and native legal aborts, not real-log Golden certificates."""
import copy
from collections import Counter

import pytest

from riichienv import convert
from riichienv.kar.actions import execute_candidate, execute_joint_responses, legal_candidates
from riichienv.kar.contracts import MJSOUL_RULESET, content_hash
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.mjsoul import RecordError, ServerRecord, normalize_records
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.tokenize import TokenHistoryCache, encode_prefix


def record(index, name, data):
    return ServerRecord(index, "lq." + name, content_hash(data), data)


def new_round_data(start, draw):
    hands = copy.deepcopy(start["tehais"])
    hands[start["oya"]].append(draw["pai"])
    return {"chang": ("E", "S", "W", "N").index(start["bakaze"]), "ju": start["oya"],
            "ben": start["honba"], "liqibang": start.get("kyotaku", start.get("kyoutaku", 0)),
            "scores": start["scores"], "dora": start["dora_marker"],
            **{f"tiles{seat}": hand for seat, hand in enumerate(hands)}}


def opening_pair(seed=42):
    env = MJSOUL_RULESET.create_env(seed=seed)
    raw = list(env.mjai_log)
    start = next(event for event in raw if event["type"] == "start_kyoku")
    draw = next(event for event in raw if event["type"] == "tsumo")
    # A container closure is needed by the decoder, but is outside every tested cutoff.
    offline = normalize_records([record(0, "RecordNewRound", new_round_data(start, draw)),
                                 record(1, "RecordNoTile", {})])[:3]
    return env, raw, offline


def ledger(events, seat):
    result = ObserverLedger(seat, (0, 1, 2, 3))
    for event in events:
        result.apply_event(event)
    return result


def deal(specials, draws, *, rinshan=()):
    """Build a complete physical wall, without injecting native hands or flags."""
    pool = list(range(136))

    def take(name):
        tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == name)
        pool.remove(tile)
        return tile

    first_draws = [take(name) for name in draws]
    replacements = [take(name) for name in rinshan]
    hands = [[take(name) for name in specials.get(seat, ())] for seat in range(4)]
    for hand in hands:
        while len(hand) < 13:
            hand.append(pool.pop(0))
    wall = [tile for start in (0, 4, 8) for hand in hands for tile in hand[start:start + 4]]
    wall += [hand[12] for hand in hands] + first_draws + pool + list(reversed(replacements))
    assert sorted(wall) == list(range(136))
    env = MJSOUL_RULESET.create_env(seed=42)
    env.reset(wall=wall, oya=0)
    return env


@pytest.mark.parametrize("seat", range(4))
def test_dealer_opening_offline_and_live_have_identical_full_input(seat):
    env, raw, offline = opening_pair()
    live = LiveEventProjection()
    live.observe_native(raw)
    ids = tuple(item.global_id for item in legal_candidates(env.get_observation(0), drawn_tile=env.drawn_tile))
    first, second = encode_prefix(ledger(offline, seat).snapshot(), ids), encode_prefix(live.snapshot(seat), ids)
    assert first.rows == second.rows
    assert first.to_bytes() == second.to_bytes()
    assert first.logical_position == 4


@pytest.mark.parametrize("packing", ["single", "pairs", "whole"])
def test_source_packing_cannot_set_reach_dora_or_draw_positions(packing):
    _, raw, _ = opening_pair()
    drawn = raw[-1]["pai"]
    occupied = {tile for event in raw if event["type"] == "start_kyoku" for hand in event["tehais"] for tile in hand}
    marker = next(convert.tid_to_mjai(tile) for tile in range(136)
                  if convert.tid_to_mjai(tile) not in occupied and convert.tid_to_mjai(tile) != drawn)
    stream = raw + [{"type": "reach", "actor": 0},
                    {"type": "dahai", "actor": 0, "pai": drawn, "tsumogiri": True},
                    {"type": "reach_accepted", "actor": 0}, {"type": "dora", "dora_marker": marker},
                    {"type": "tsumo", "actor": 1, "pai": marker}]
    first, second = ObserverLedger(0, (0, 1, 2, 3)), ObserverLedger(0, (0, 1, 2, 3))
    cache = TokenHistoryCache()
    frozen = []
    for index, event in enumerate(stream):
        group = index if packing == "single" else index // 2 if packing == "pairs" else 0
        first.apply_event({**event, "source_record_index": group, "source_record_sha256": "a" * 64})
        second.apply_event({**event, "source_record_index": 1000 + index, "source_record_sha256": "b" * 64})
        if index >= 2:
            a = encode_prefix(first.snapshot(), (4, 8), history_cache=cache)
            b = encode_prefix(second.snapshot(), (4, 8))
            assert a.to_bytes() == b.to_bytes()
            frozen.append((a, a.to_bytes()))
    assert all(prefix.to_bytes() == value for prefix, value in frozen)


def test_kan_commit_ack_and_implicit_commit_share_semantic_positions():
    env = deal({0: ("E",) * 4}, ["9s"])
    raw = list(env.mjai_log)
    declaration = {"type": "ankan", "actor": 0, "consumed": ["E"] * 4}
    occupied = {tile for event in raw if event["type"] == "start_kyoku" for hand in event["tehais"] for tile in hand}
    indicator = next(convert.tid_to_mjai(tile) for tile in range(136) if convert.tid_to_mjai(tile) not in occupied)
    tail = [{"type": "dora", "dora_marker": indicator}, {"type": "tsumo", "actor": 0, "pai": indicator}]
    a = ledger(raw + [declaration] + tail, 0)
    b = ledger(raw + [declaration, {"type": "kan_commit", "actor": 0, "kan_type": "ankan"}] + tail, 0)
    assert encode_prefix(a.snapshot(), (4,)).to_bytes() == encode_prefix(b.snapshot(), (4,)).to_bytes()


def test_delayed_old_dora_and_optional_kan_ack_have_identical_history():
    from tests.kar.test_observation_response import round_events  # noqa: PLC0415

    hand = ["1m", "2m", "3m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "C"]
    head = round_events(hand) + [
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False},
        {"type": "pon", "actor": 2, "target": 1, "pai": "4m", "consumed": ["4m"] * 2},
        {"type": "dahai", "actor": 3, "pai": "9p", "tsumogiri": False},
        {"type": "daiminkan", "actor": 2, "target": 3, "pai": "9p", "consumed": ["9p"] * 3},
        {"type": "tsumo", "actor": 2, "pai": "4m"},
        {"type": "kakan", "actor": 2, "pai": "4m"},
    ]
    tail = [{"type": "dora", "dora_marker": "2s"}, {"type": "tsumo", "actor": 2, "pai": "1s"}]
    a, b = ledger(head, 0), ledger(head, 0)
    before = encode_prefix(a.snapshot(), (4, 8))
    assert not any(row["kind"] == 21 for row in before.rows)
    b.apply_event({"type": "kan_commit", "actor": 2, "kan_type": "kakan"})
    assert b.snapshot().pending_kan_actor is None  # actual ACK is valid state evidence
    for event in tail:
        a.apply_event(event)
        b.apply_event(event)
    first, second = encode_prefix(a.snapshot(), (4, 8)), encode_prefix(b.snapshot(), (4, 8))
    assert first.to_bytes() == second.to_bytes()
    assert sum(row["kind"] == 21 for row in first.rows) == 1
    assert not any(row["kind"] == 21 for row in before.rows)


@pytest.mark.parametrize("seat", range(4))
@pytest.mark.parametrize("chunks", [1, 2, 100])
def test_kyushu_public_reveal_and_zero_settlement_match_through_next_real_decision(seat, chunks):
    env = deal({0: ("1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C")}, ["2m"])
    initial = list(env.mjai_log)
    start = next(event for event in initial if event["type"] == "start_kyoku")
    draw = initial[-1]
    live = LiveEventProjection()
    live.observe_native(initial)
    old_snapshot = live.snapshot(seat)
    old_prefix = encode_prefix(old_snapshot, (220,))
    chosen = next(item for item in legal_candidates(env.get_observation(0), drawn_tile=env.drawn_tile)
                  if item.global_id == 220)
    execute_candidate(env, 0, chosen)
    terminal = list(env.mjai_log[len(initial):])
    # Advance the same actual environment to the next public round/first draw.
    if not any(event["type"] == "start_kyoku" for event in terminal):
        env.step({})
        terminal = list(env.mjai_log[len(initial):])
    next_start = next(event for event in terminal if event["type"] == "start_kyoku")
    next_draw = next(event for event in terminal[terminal.index(next_start) + 1:] if event["type"] == "tsumo")
    for begin in range(0, len(terminal), chunks):
        live.observe_native(terminal[begin:begin + chunks])
    records = [record(0, "RecordNewRound", new_round_data(start, draw)),
               record(1, "RecordLiuJu", {"type": 1, "seat": 0, "tiles": start["tehais"][0] + [draw["pai"]]}),
               record(2, "RecordNewRound", new_round_data(next_start, next_draw)),
               record(3, "RecordNoTile", {})]
    normalized = normalize_records(records)
    boundary = next(index for index, event in enumerate(normalized) if index > 2 and event["type"] == "tsumo")
    offline = ledger(normalized[:boundary + 1], seat)
    ids = tuple(item.global_id for item in legal_candidates(
        env.get_observation(env.current_player), drawn_tile=env.drawn_tile
    ))
    a, b = encode_prefix(offline.snapshot(), ids), encode_prefix(live.snapshot(seat), ids)
    assert a.to_bytes() == b.to_bytes()
    reveal = [row for row in a.rows if row["kind"] == 29]
    cash = [row for row in a.rows if row["kind"] == 25]
    assert len(reveal) == 3 and len(cash) == 4
    assert all(row["numeric_present"][1] and row["numeric"][1] == 0 for row in cash)
    assert not any(row["kind"] == 29 for row in old_prefix.rows)
    assert encode_prefix(old_snapshot, (220,)).to_bytes() == old_prefix.to_bytes()
    result = next(event for event in normalized if event["type"] == "ryukyoku")
    assert result["actor"] == 0
    assert all(hand is None for hand in result["public_hands"][1:])
    assert Counter(result["public_hands"][0]) == Counter(start["tehais"][0] + [draw["pai"]])


@pytest.mark.parametrize("bad", ["actor", "tiles", "late", "not-nine"])
def test_kyushu_reconstruction_rejects_unproven_identity_or_tiles(bad):
    env = deal({0: ("1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C")}, ["2m"])
    raw = list(env.mjai_log)
    start = next(event for event in raw if event["type"] == "start_kyoku")
    data = new_round_data(start, raw[-1])
    terminal = {"type": 1, "seat": 0, "tiles": data["tiles0"].copy()}
    records = [record(0, "RecordNewRound", data)]
    if bad == "actor":
        terminal["seat"] = 1
    elif bad == "tiles":
        terminal["tiles"][0] = "3m"
    elif bad == "late":
        records.append(record(1, "RecordDiscardTile", {"seat": 0, "tile": "2m", "moqie": True}))
    else:
        data["tiles0"] = ["2m"] * 14
        terminal["tiles"] = data["tiles0"].copy()
    records.append(record(2, "RecordLiuJu", terminal))
    with pytest.raises(RecordError, match="Nine|nine|declaring|hand|draw"):
        normalize_records(records)


def test_confirmed_private_pass_does_not_change_public_clock_or_other_observers():
    env = deal({0: ("E",), 1: ("E", "E")}, ["9s", "8s"])
    live = LiveEventProjection()
    live.observe_native(env.mjai_log)
    discard = next(item for item in legal_candidates(env.get_observation(0), drawn_tile=env.drawn_tile)
                   if item.global_id < 148 and item.actions[-1].tile // 4 == 27)
    before = len(env.mjai_log)
    responses = execute_candidate(env, 0, discard)
    live.observe_native(env.mjai_log[before:])
    choices = {seat: next(item for item in legal_candidates(observation, source_player=0) if item.global_id == 221)
               for seat, observation in responses.items() if observation.legal_actions()}
    assert any(item.spec.action_type == "PON" for item in legal_candidates(env.get_observation(1), source_player=0))
    receipt = tuple(trigger for trigger in live.capture_self_passes(choices) if trigger.observer_player_id == 1)
    old = live.snapshot(1)
    old_bytes = encode_prefix(old, (221,)).to_bytes()
    before = len(env.mjai_log)
    execute_joint_responses(env, choices)
    live.observe_native(env.mjai_log[before:], confirmed_self_passes=receipt)
    offline = ledger([{**event, "source_record_index": 0} for event in live.events], 1)
    trigger = offline._pass_triggers[receipt[0].trigger_event_id]
    offline.record_self_pass(trigger, provenance="inferred_unique_next_draw_pass")
    assert encode_prefix(offline.snapshot(), (4, 8)).to_bytes() == encode_prefix(live.snapshot(1), (4, 8)).to_bytes()
    assert old_bytes == encode_prefix(old, (221,)).to_bytes()
    for seat in (0, 2, 3):
        expected = ledger(live.events, seat).snapshot()
        assert live.snapshot(seat).to_bytes() == expected.to_bytes()
    assert live.ledgers[1].logical_clock == live.ledgers[0].logical_clock


@pytest.mark.parametrize("reverse_winners", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 100])
def test_actual_multi_ron_group_is_independent_of_native_batch_boundaries(chunk_size, reverse_winners):
    common = ("1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s")
    env = deal({0: ("E",), 1: common + ("P",) * 3 + ("E",),
                2: common + ("F",) * 3 + ("E",)}, ["C"])
    prefix = list(env.mjai_log)
    discard = next(item for item in legal_candidates(env.get_observation(0), drawn_tile=env.drawn_tile)
                   if item.global_id < 148 and not item.spec.riichi and item.actions[-1].tile // 4 == 27)
    responses = execute_candidate(env, 0, discard)
    public_before = list(env.mjai_log)
    choices = {seat: next(item for item in legal_candidates(observation, source_player=0)
                          if item.spec.action_type == ("RON" if seat in {1, 2} else "PASS"))
               for seat, observation in responses.items() if observation.legal_actions()}
    execute_joint_responses(env, choices)
    tail = list(env.mjai_log[len(public_before):])
    assert sum(event["type"] == "hora" for event in tail) == 2
    if reverse_winners:
        end = next(index for index, event in enumerate(tail) if event["type"] != "hora")
        tail = list(reversed(tail[:end])) + tail[end:]
    whole, streamed = LiveEventProjection(), LiveEventProjection()
    whole.observe_native(env.mjai_log)
    streamed.observe_native(public_before)
    old = streamed.snapshot(3)
    old_bytes = encode_prefix(old, (221,)).to_bytes()
    for start in range(0, len(tail), chunk_size):
        streamed.observe_native(tail[start:start + chunk_size])
        if streamed._pending_horas:
            with pytest.raises(ValueError, match="incomplete"):
                streamed.snapshot(3)
    ids = tuple(item.global_id for item in legal_candidates(
        env.get_observation(env.current_player), drawn_tile=env.drawn_tile
    ))
    for seat in range(4):
        a, b = encode_prefix(whole.snapshot(seat), ids), encode_prefix(streamed.snapshot(seat), ids)
        offline = ledger([{**event, "source_record_index": 0} for event in whole.events], seat)
        assert a.to_bytes() == b.to_bytes() == encode_prefix(offline.snapshot(), ids).to_bytes()
        assert sum(row["kind"] == 25 for row in a.rows) == 4
        wins = [row["positions"] for row in a.rows if row["kind"] == 27]
        assert len(wins) == 2 and len(set(wins)) == 1
    assert encode_prefix(old, (221,)).to_bytes() == old_bytes
    assert len(prefix) == 3
