"""Bounded synthetic *legal rounds*, not real-log Golden evidence.

Expected eligibility comes from Mahjong Soul's FAQ (id=47, 採用役は？):
https://mahjongsoul.com/faq/ and https://mahjongsoul.com/api/faq/list
Own calls/kans do not disqualify; middle discards and called-away discards do.
Every live action below must be offered by the native engine, using a full
permutation of 136 physical tiles. No hands/melds/eligibility flags are injected.
"""
from dataclasses import dataclass, replace

import pytest

from riichienv import Action, ActionType, Phase, convert
from riichienv.kar.actions import legal_candidates
from riichienv.kar.contracts import MJSOUL_RULESET, RESPONSE_MASK_VERSION, Response, content_hash
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.response import resolve_discard_response
from riichienv.kar.response_mask import _nagashi_player_possible, public_rule_response_mask
from riichienv.kar.runtime_contracts import RuntimeBinding, runtime_contract
from riichienv.kar.settlement import _Verifier
from riichienv.kar.support import EventSupport

KINDS = ("none", "chi", "pon", "daiminkan", "ankan", "kakan")
CALLS = {"chi": ActionType.CHI, "pon": ActionType.PON, "daiminkan": ActionType.DAIMINKAN}
NATIVE_KINDS = {**CALLS, "discard": ActionType.DISCARD, "ankan": ActionType.ANKAN, "kakan": ActionType.KAKAN}


def _terminal(tile):
    return tile // 4 >= 27 or tile // 4 % 9 in {0, 8}


def _take(pool, name):
    tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == name)
    pool.remove(tile)
    return tile


@dataclass(frozen=True)
class RoundPlan:
    wall: tuple[int, ...]
    actions: tuple[tuple[str, int, int, tuple[int, ...]], ...]
    oya: int = 2


def _plan(kind, *, middle_winner=False):  # noqa: PLR0915
    """Allocate every tile once, reserving P1's terminal draws and the dead wall."""
    pool = list(range(136))
    dora = _take(pool, "C")
    kan_dora = _take(pool, "F") if kind in {"daiminkan", "ankan", "kakan"} else None
    last_middle = _take(pool, "2m")
    needs_call = kind in {"chi", "pon", "daiminkan", "kakan"}
    called = _take(pool, "1m" if kind == "chi" else "E") if needs_call else None
    consumed_names = {
        "none": (), "chi": ("2m", "3m"), "pon": ("E", "E"),
        "daiminkan": ("E",) * 3, "ankan": ("E",) * 4, "kakan": ("E", "E"),
    }[kind]
    consumed = tuple(_take(pool, name) for name in consumed_names)
    spare = _take(pool, "9p")
    upgrade = _take(pool, "E") if kind == "kakan" else None
    hands = [[called] if called is not None else [], [*consumed, spare], [], []]
    terminal_draws = [tile for tile in pool if _terminal(tile)][:20]
    for tile in terminal_draws:
        pool.remove(tile)
    for hand in hands:
        while len(hand) < 13:
            hand.append(pool.pop(0))
    initial_hands = [hand.copy() for hand in hands]
    normal, replacements, actions = [], [], []
    draws = 0
    actor = 2
    call_done = kan_done = middle_done = False

    def draw(player, *, replacement=False, upgrade_draw=False):
        nonlocal draws, last_middle, middle_done
        if upgrade_draw:
            tile = upgrade
        elif player == 0 and draws == 69:
            tile, last_middle = last_middle, None
        elif player == 1 and not (middle_winner and not middle_done and call_done):
            tile = terminal_draws.pop()
        else:
            tile = next((tile for tile in pool if not _terminal(tile)), pool[0])
            pool.remove(tile)
            if player == 1:
                middle_done = True
        assert tile is not None
        (replacements if replacement else normal).append(tile)
        draws += 1
        return tile

    while draws < 70:
        use_upgrade = kind == "kakan" and call_done and not kan_done and actor == 1
        tile = draw(actor, upgrade_draw=use_upgrade)
        if actor == 1 and not kan_done and (kind == "ankan" or use_upgrade):
            actions.append((kind, actor, consumed[0] if kind == "ankan" else tile, consumed if kind == "ankan" else ()))
            kan_done = True
            tile = draw(actor, replacement=True)
        discard = called if needs_call and not call_done and actor == 0 else tile
        assert discard is not None
        actions.append(("discard", actor, discard, ()))
        if needs_call and not call_done and actor == 0:
            call_kind = "pon" if kind == "kakan" else kind
            actions.append((call_kind, 1, called, consumed))
            call_done = True
            if kind == "daiminkan":
                discard = draw(1, replacement=True)
                kan_done = True
            else:
                discard = spare
            actions.append(("discard", 1, discard, ()))
            actor = 2
        else:
            actor = (actor + 1) % 4
    assert not needs_call or call_done
    assert kind not in {"daiminkan", "ankan", "kakan"} or kan_done
    tail = [*pool, *terminal_draws, *replacements]
    if last_middle is not None:
        tail.append(last_middle)
    tail.append(dora)
    if kan_dora is not None:
        tail.append(kan_dora)
    tail.remove(dora)
    tail.insert(len(tail) - 4, dora)  # Initial marker is fifth from original end.
    if kan_dora is not None:
        tail.remove(kan_dora)
        tail.insert(len(tail) - 6, kan_dora)
    if replacements:
        replacement = replacements[0]
        tail.remove(replacement)
        tail.append(replacement)  # Native rinshan removes from internal wall front.
    # Set marker positions *after* moving rinshan, preserving the full permutation.
    for offset, tile in ((5, dora), (7, kan_dora)):
        if tile is not None:
            index = tail.index(tile)
            tail[index], tail[-offset] = tail[-offset], tail[index]
    wall = []
    for packet in range(3):
        for relative in range(4):
            hand = initial_hands[(2 + relative) % 4]
            wall.extend(hand[packet * 4:packet * 4 + 4])
    wall.extend(initial_hands[(2 + relative) % 4][12] for relative in range(4))
    wall.extend(normal)
    wall.extend(tail)
    assert sorted(wall) == list(range(136)), "no fifth copy, repeated physical tile or missing tile"
    assert sum(tile in {16, 52, 88} for tile in wall) == 3
    return RoundPlan(tuple(wall), tuple(actions))


def _run_round(kind, *, middle_winner=False, honba=0, pot=0):  # noqa: PLR0915
    plan = _plan(kind, middle_winner=middle_winner)
    scores = [25000, 25000, 25000 - 1000 * pot, 25000]
    env = MJSOUL_RULESET.create_env(seed=214)
    env.reset(wall=list(plan.wall), oya=plan.oya, round_wind=0, honba=honba, kyotaku=pot, scores=scores)
    projection = LiveEventProjection()
    projection.observe_native(env.mjai_log)
    contexts, claims = [], []

    def sync(**kwargs):
        projection.observe_native(env.mjai_log[projection.raw_cursor:], **kwargs)

    def pass_responses():
        if env.phase == Phase.WaitResponse:
            env.step({player: Action(ActionType.PASS) for player in env.active_players})
            sync()

    for kind_name, actor, tile, consumed in plan.actions:
        if kind_name not in CALLS:
            pass_responses()
        observation = env.get_observation(actor)
        offered = observation.legal_actions()
        selected = next(
            (action for action in offered
             if action.action_type == NATIVE_KINDS[kind_name] and action.tile == tile
             and (not consumed or tuple(sorted(action.consume_tiles)) == tuple(sorted(consumed)))),
            None,
        )
        assert selected is not None, (kind, kind_name, actor, tile, env.phase)
        before_count = len(projection.events)
        if kind_name == "discard":
            snapshot = projection.snapshot(actor)
            identifiers = [
                candidate.global_id for candidate in legal_candidates(observation, drawn_tile=env.drawn_tile)
                if len(candidate.actions) == 1 and candidate.actions[0].to_dict() == selected.to_dict()
            ]
            assert len(identifiers) == 1
            context = (snapshot, identifiers[0])
        elif kind_name in CALLS:
            before_claim = projection.snapshot(actor)
        declared = None
        if kind_name == "ankan":
            declared = {"type": "ankan", "actor": actor,
                        "consumed": [convert.tid_to_mjai(value) for value in consumed]}
        env.step({actor: selected})
        sync(declared_ankan=declared, pending_kan_window=env.phase == Phase.WaitResponse)
        if kind_name == "discard":
            trigger = next(index for index in range(before_count, len(projection.events))
                           if projection.events[index]["type"] == "dahai")
            contexts.append((*context, trigger))
        elif kind_name in CALLS:
            claims.append((before_claim, projection.snapshot(actor)))
    pass_responses()
    end = next(index for index, event in enumerate(projection.events) if event["type"] == "end_kyoku")
    events = projection.events[:end + 1]
    draw_event = next(event for event in events if event["type"] == "ryukyoku")
    # A legal seeded round is not claimed to be a complete hanchan. Initialize
    # the independent verifier with exactly the same public pre-round context.
    verifier = _Verifier(MJSOUL_RULESET)
    verifier.points, verifier.pot = tuple(scores), pot
    verifier.expected_round = (0, plan.oya, honba)
    for index, event in enumerate(events):
        if event["type"] == "ryukyoku":
            verifier.draw(event, index)
        else:
            verifier.observe(event, index)
    assert not verifier.errors and len(verifier.rows) == 1
    assert verifier.rows[0]["status"] == "PASS", verifier.rows
    assert all(len(hand) + 3 * len(melds) == 13 for hand, melds in zip(
        [ledger.snapshot().hand for ledger in verifier.views],
        [ledger.snapshot().melds[seat] for seat, ledger in enumerate(verifier.views)],
    ))
    return env, projection, events, draw_event, verifier, contexts, claims


@pytest.mark.parametrize("kind", KINDS)
def test_native_legal_meld_round_agrees_with_independent_settlement_and_public_mask(kind):
    env, projection, events, draw, verifier, contexts, _ = _run_round(kind)
    assert draw["reason"] == "nagashi_mangan"
    assert verifier.rows[0]["nagashi_winners"] == [1]
    # Native logs expose payments, not a winner list. These single-winner
    # scenarios independently specify P1 and the exact non-dealer mangan delta.
    delta = [-2000, 8000, -4000, -2000]
    assert draw["deltas"] == delta and sum(delta) == 0
    assert verifier.rows[0]["expected_deltas"] == tuple(delta)
    assert tuple(env.scores()) == verifier.points
    assert projection.public_scores == verifier.points
    observed = 0
    for snapshot, action_id, trigger in contexts:
        response = resolve_discard_response(events, trigger)
        if response.response_target == Response.NAGASHI_MANGAN:
            before = snapshot.to_bytes()
            assert public_rule_response_mask(snapshot, action_id)[Response.NAGASHI_MANGAN] is True
            assert snapshot.to_bytes() == before
            observed += 1
    assert observed == 1
    if kind == "pon":
        snapshot, action_id, _ = contexts[-1]
        assert snapshot.observer_player_id == 0 and events[-3]["actor"] == 0
        assert snapshot.melds[1][0].kind == "pon" and snapshot.melds[1][0].tiles[0] // 4 == 27
        assert snapshot.discard_history[1] == snapshot.current_rivers[1]
        assert all(any(not _terminal(tile) for tile in snapshot.discard_history[seat]) for seat in (0, 2, 3))
        assert action_id < 148 and not _terminal(convert.mjai_to_tid(events[-3]["pai"]))


@pytest.mark.parametrize("kind", ["chi", "pon", "daiminkan"])
def test_native_legal_call_disqualifies_discarder_not_caller(kind):
    *_, claims = _run_round(kind)
    assert len(claims) == 1
    before, after = claims[0]
    assert _nagashi_player_possible(before, 0, actor=1, tile=8) is True
    assert after.discard_history[0] == before.discard_history[0]
    assert len(after.current_rivers[0]) + 1 == len(before.current_rivers[0])
    assert _nagashi_player_possible(after, 0, actor=1, tile=8) is False
    assert _nagashi_player_possible(after, 1, actor=1, tile=8) is True


def test_native_open_meld_does_not_override_a_known_middle_discard():
    _, _, events, draw, verifier, contexts, _ = _run_round("pon", middle_winner=True)
    assert draw["reason"] == "exhaustive_draw"
    assert verifier.rows[0]["nagashi_winners"] == []
    snapshot, action_id, trigger = contexts[-1]
    assert snapshot.melds[1] and any(not _terminal(tile) for tile in snapshot.discard_history[1])
    assert public_rule_response_mask(snapshot, action_id)[Response.NAGASHI_MANGAN] is False
    assert resolve_discard_response(events, trigger).response_target == Response.EXHAUSTIVE_DRAW


def test_native_nagashi_keeps_honba_and_riichi_pot_payment_contract():
    env, projection, _, draw, verifier, _, _ = _run_round("pon", honba=3, pot=2)
    assert draw["deltas"] == [-2000, 8000, -4000, -2000]
    assert verifier.pot == 2 and sum(verifier.points) + 1000 * verifier.pot == 100000
    assert projection.public_pot == 2
    next_round = [event for event in env.mjai_log if event["type"] == "start_kyoku"][-1]
    assert next_round["honba"] == 4 and next_round["kyotaku"] == 2


def test_old_rule_contract_is_rejected_by_runtime_binding():
    old = replace(MJSOUL_RULESET, name="mahjong_soul_4p_red_half_v1")
    assert old.ruleset_hash != MJSOUL_RULESET.ruleset_hash
    assert RESPONSE_MASK_VERSION == "public_rule_response_mask_v3_abort_last_draw"
    current = runtime_contract()
    with pytest.raises(ValueError, match="contracts differ"):
        RuntimeBinding("a" * 64, "b" * 64, current["schema_hash"], old.ruleset_hash, current["action_manifest_hash"])


def test_old_support_contract_cannot_silently_cross_the_rule_fix():
    # Synthetic metadata only: no old data or manifest is changed to fake compatibility.
    old = replace(MJSOUL_RULESET, name="mahjong_soul_4p_red_half_v1")
    record = EventSupport("b" * 64, (500,) * 21, "synthetic-rule-migration-regression").to_record()
    record["contract"]["ruleset_hash"] = old.ruleset_hash
    record["support_sha256"] = content_hash({key: value for key, value in record.items() if key != "support_sha256"})
    binding = RuntimeBinding.current("a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="integrity/version/contract"):
        EventSupport.from_record(record, binding)
