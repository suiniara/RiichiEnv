"""Public-only masks against actual native legal actions on complete physical walls."""
from dataclasses import replace

import pytest

from riichienv import Phase
from riichienv.kar.actions import execute_candidate, execute_joint_responses, legal_candidates
from riichienv.kar.contracts import MJSOUL_RULESET, Response
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.observation import PublicMeld
from riichienv.kar.response import resolve_discard_response
from riichienv.kar.response_mask import public_rule_response_mask, validate_response_target
from tests.kar.test_input_semantics import deal
from tests.kar.test_nagashi_consistency import _run_round


def sync(env, live, begin):
    live.observe_native(env.rule_log[begin:])


def pass_window(env, live):
    if env.phase == Phase.WaitResponse:
        choices = {}
        for seat in env.active_players:
            view = live.snapshot(seat)
            source = view.pending_kan_actor if view.pending_kan_actor is not None else view.last_discard_actor
            choices[seat] = next(item for item in legal_candidates(env.get_observation(seat), source_player=source)
                                 if item.global_id == 221)
        begin = len(env.rule_log)
        execute_joint_responses(env, choices)
        sync(env, live, begin)


def perform(env, live, chosen):
    begin = len(env.rule_log)
    execute_candidate(env, env.current_player, chosen)
    sync(env, live, begin)
    pass_window(env, live)


def four_riichi(distinct_discards=False):
    shape = ("1m", "2m", "3m", "1p", "2p", "3p", "7p", "8p", "9p", "1s", "2s", "3s", "E")
    draws = ("6m", "7m", "8m", "9m") if distinct_discards else ("9m",) * 4
    env = deal({seat: shape for seat in range(4)}, draws)
    live = LiveEventProjection()
    live.observe_native(env.rule_log)
    for actor in range(3):
        assert env.current_player == actor
        chosen = next(item for item in legal_candidates(env.get_observation(actor), drawn_tile=env.drawn_tile)
                      if item.global_id < 148 and item.spec.riichi and item.actions[-1].tile == env.drawn_tile)
        perform(env, live, chosen)
    assert env.current_player == 3
    chosen = next(item for item in legal_candidates(env.get_observation(3), drawn_tile=env.drawn_tile)
                  if item.global_id == 39)
    return env, live, chosen


@pytest.mark.parametrize("distinct_discards", [False, True])
def test_fourth_native_riichi_discard_excludes_progress_but_not_publicly_possible_ron(distinct_discards):
    env, live, chosen = four_riichi(distinct_discards)
    before = live.snapshot(3)
    assert before.riichi_accepted == (True, True, True, False) and before.remaining_draws == 66
    allowed = public_rule_response_mask(before, chosen.global_id)
    assert allowed[Response.ABORT_FOUR_RIICHI]
    assert not any(allowed[:8])
    assert any(allowed[8:15]) == distinct_discards
    not_declaring = public_rule_response_mask(before, chosen.global_id - 1)
    assert not_declaring[Response.NEXT_DRAW] and not not_declaring[Response.ABORT_FOUR_RIICHI]
    trigger = len(live.events) + 1  # the legal two-step reach then discard
    perform(env, live, chosen)
    actual = resolve_discard_response(live.events, trigger)
    assert actual.response_observed and actual.response_target == Response.ABORT_FOUR_RIICHI
    validate_response_target(allowed, actual.response_target)


@pytest.mark.parametrize("remaining", [0, 1])
def test_native_last_discard_and_adjacent_boundary_have_no_impossible_calls(remaining):
    _, _, _, _, _, contexts, _ = _run_round("none")
    snapshot, action, _ = next(context for context in reversed(contexts) if context[0].remaining_draws == remaining)
    allowed = public_rule_response_mask(snapshot, action)
    if remaining == 0:
        assert not any(allowed[:8])
        assert allowed[Response.EXHAUSTIVE_DRAW] or allowed[Response.NAGASHI_MANGAN]
    else:
        assert allowed[Response.NEXT_DRAW]
        assert any(allowed[2:5])


def test_native_four_winds_is_terminal_after_ron_priority():
    shape = ("1m", "2m", "3m", "1p", "2p", "3p", "7p", "8p", "9p", "1s", "2s", "3s", "E")
    env = deal({seat: shape for seat in range(4)}, ("9m",) * 4)
    live = LiveEventProjection()
    live.observe_native(env.rule_log)
    for actor in range(4):
        chosen = next(item for item in legal_candidates(env.get_observation(actor), drawn_tile=env.drawn_tile)
                      if item.global_id < 148 and not item.spec.riichi and item.actions[-1].tile // 4 == 27)
        before = live.snapshot(actor)
        allowed = public_rule_response_mask(before, chosen.global_id)
        assert allowed[Response.ABORT_FOUR_WINDS] == (actor == 3)
        if actor == 3:
            assert not any(allowed[:8])
        else:
            assert allowed[Response.NEXT_DRAW]
        perform(env, live, chosen)
    assert any(event.get("reason") == "four_winds" for event in live.events)


@pytest.mark.parametrize("single_owner", [False, True])
def test_four_legal_native_kans_distinguish_one_and_multiple_owners(single_owner):
    specials = {0: ("E",) * 4 + ("S",) * 4 + ("W",) * 4 + ("N",)} if single_owner else {
        0: ("E",) * 4 + ("S",) * 4, 1: ("W",) * 4 + ("N",) * 4}
    env = deal(specials, ("N",) if single_owner else ("9m", "8m"),
               rinshan=("N", "N") if single_owner else ())
    live = LiveEventProjection()
    live.observe_native(env.rule_log)
    for owner, count in ((0, 4),) if single_owner else ((0, 2), (1, 2)):
        assert env.current_player == owner
        for _ in range(count):
            chosen = next(item for item in legal_candidates(env.get_observation(owner), drawn_tile=env.drawn_tile)
                          if item.spec.action_type == "ANKAN")
            perform(env, live, chosen)
        chosen = next(item for item in legal_candidates(env.get_observation(owner), drawn_tile=env.drawn_tile)
                      if item.global_id < 148 and not item.spec.riichi)
        before = live.snapshot(owner)
        counts = [sum(meld.kind in {"ankan", "kakan", "daiminkan"} for meld in melds) for melds in before.melds]
        allowed = public_rule_response_mask(before, chosen.global_id)
        if sum(counts) == 4:
            assert allowed[Response.ABORT_FOUR_KANS] == (not single_owner)
            assert allowed[Response.NEXT_DRAW] == single_owner
            assert not any(allowed[5:8])
        perform(env, live, chosen)
    assert any(event.get("reason") == "four_kans" for event in live.events) == (not single_owner)


@pytest.mark.parametrize("kind", ["winds", "riichi", "kans"])
def test_unknown_abort_evidence_does_not_falsely_suppress_progress(kind):
    _, live, _ = four_riichi()
    base = live.snapshot(3)
    state = replace(base, history_complete=False, riichi_accepted=(False,) * 4,
                    discard_history=((),) * 4, current_rivers=((),) * 4)
    allowed = public_rule_response_mask(state, 4)
    assert allowed[Response.NEXT_DRAW] and any(allowed[2:5])
    aborts = {"winds": Response.ABORT_FOUR_WINDS, "riichi": Response.ABORT_FOUR_RIICHI,
              "kans": Response.ABORT_FOUR_KANS}
    index = aborts[kind]
    # Non-wind candidate is positive evidence against four winds, not Unknown.
    assert allowed[index] == (kind != "winds")


def test_known_multi_owner_four_kan_proof_does_not_need_unknown_discard_history():
    _, live, _ = four_riichi()
    kans = [PublicMeld("ankan", (index * 4,) * 4) for index in range(4)]
    state = replace(live.snapshot(3), history_complete=False, melds=((kans[0], kans[1]), (kans[2], kans[3]), (), ()))
    allowed = public_rule_response_mask(state, 4)
    assert allowed[Response.ABORT_FOUR_KANS] and not any(allowed[:8])
    assert any(allowed[8:15])
    disabled = public_rule_response_mask(state, 4, replace(MJSOUL_RULESET, four_kans_abort=False))
    assert disabled[Response.NEXT_DRAW] and not disabled[Response.ABORT_FOUR_KANS]
