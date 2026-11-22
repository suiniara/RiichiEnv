from dataclasses import replace

import pytest

from riichienv import convert
from riichienv.kar.actions import tile136_to_tile37
from riichienv.kar.contracts import MJSOUL_RULESET, Response
from riichienv.kar.observation import ObserverLedger, PublicMeld, project_observer
from riichienv.kar.response import resolve_discard_response
from riichienv.kar.response_mask import (
    _nagashi_player_possible,
    _nagashi_possible,
    public_rule_response_mask,
    validate_response_target,
)
from riichienv.kar.tokenize import TokenizationUnavailableError, encode_prefix


def round_events(own=None):
    hand = own or ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    return [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000] * 4,
            "dora_marker": "9s",
            "tehais": [hand, ["1s"] * 13, ["2s"] * 13, ["3s"] * 13],
        },
    ]


def ledger(events):
    view = ObserverLedger(0, (0, 1, 2, 3))
    for event in events:
        view.apply_event(event)
    return view


def test_future_and_opponent_hidden_information_cannot_change_cutoff():
    original = round_events() + [{"type": "tsumo", "actor": 0, "pai": "5pr"}]
    changed = round_events() + [{"type": "tsumo", "actor": 0, "pai": "5pr"}, {"type": "hora", "actor": 2, "target": 0}]
    changed[1]["tehais"][1] = ["C"] * 13
    changed[1]["paishan"] = "never observable"
    changed[1]["ura_doras"] = ["1m"]
    a = project_observer(original, 0, 2, (0, 1, 2, 3))
    b = project_observer(changed, 0, 2, (0, 1, 2, 3))
    assert a.to_bytes() == b.to_bytes()
    assert b"never observable" not in b.to_bytes()


def test_opponent_draws_are_hidden():
    a = ledger(round_events() + [{"type": "tsumo", "actor": 1, "pai": "1m"}]).snapshot()
    b = ledger(round_events() + [{"type": "tsumo", "actor": 1, "pai": "5pr"}]).snapshot()
    assert a.to_bytes() == b.to_bytes()


def test_called_discard_keeps_history_but_moves_physical_region():
    events = round_events() + [
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False},
        {"type": "pon", "actor": 2, "target": 1, "pai": "5p", "consumed": ["5p", "5pr"]},
    ]
    state = ledger(events).snapshot()
    assert state.current_rivers[1] == ()
    assert len(state.discard_history[1]) == 1
    assert state.known_counts()[13] == 3


def test_kakan_is_only_committed_after_rob_window_and_never_creates_fifth_meld():
    events = round_events() + [
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False},
        {"type": "pon", "actor": 2, "target": 1, "pai": "5p", "consumed": ["5p", "5pr"]},
        {"type": "kakan", "actor": 2, "pai": "5p"},
    ]
    view = ledger(events)
    assert view.snapshot().melds[2][0].kind == "pon"
    assert view.snapshot().known_counts()[13] == 3
    view.apply_event({"type": "hora", "actor": 3, "target": 2})
    assert view.snapshot().melds[2][0].kind == "pon"
    committed = ledger(events)
    committed.apply_event({"type": "tsumo", "actor": 2, "pai": "1s"})
    assert len(committed.snapshot().melds[2]) == 1
    assert committed.snapshot().melds[2][0].kind == "kakan"
    assert committed.snapshot().known_counts()[13] == 4


def test_opponent_ankan_does_not_reveal_red_composition():
    a = ledger(round_events() + [{"type": "ankan", "actor": 2, "consumed": ["5p"] * 4}]).snapshot()
    b = ledger(round_events() + [{"type": "ankan", "actor": 2, "consumed": ["5pr", "5p", "5p", "5p"]}]).snapshot()
    assert a.to_bytes() == b.to_bytes()


def test_riichi_does_not_charge_before_acceptance():
    view = ledger(round_events())
    view.apply_event({"type": "reach", "actor": 0})
    view.apply_event({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False})
    assert view.snapshot().scores[0] == 25000
    view.apply_event({"type": "reach_accepted", "actor": 0})
    assert view.snapshot().scores[0] == 24000
    with pytest.raises(ValueError):
        view.apply_event({"type": "reach_accepted", "actor": 0})


@pytest.mark.parametrize("call,actor,expected", [("chi", 1, 1), ("pon", 2, 3), ("daiminkan", 3, 7)])
def test_accepted_call_closes_original_discard_window(call, actor, expected):
    events = [
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": call, "actor": actor, "target": 0, "pai": "1m"},
        {"type": "tsumo", "actor": actor, "pai": "2m"},
        {"type": "hora", "actor": actor, "target": actor},
    ]
    result = resolve_discard_response(events, 0)
    assert result.response_target == expected
    assert result.response_observed is True


def test_ron_waits_for_complete_winner_set_and_mjsoul_allows_triple_ron():
    events = [{"type": "dahai", "actor": 0, "pai": "1m"}]
    events += [{"type": "hora", "actor": p, "target": 0} for p in (1, 2, 3)]
    assert not resolve_discard_response(events, 0).response_observed
    events.append({"type": "end_kyoku"})
    assert resolve_discard_response(events, 0).response_target == Response.RON_MASK_7
    assert not resolve_discard_response(events, 0, replace(MJSOUL_RULESET, sanchaho_is_draw=True)).response_observed


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("exhaustive", 15),
        ("exhaustive_draw", 15),
        ("four_winds", 16),
        ("four_riichi", 17),
        ("four_kans", 18),
        ("nagashi_mangan", 20),
    ],
)
def test_direct_end_is_not_next_draw(reason, expected):
    result = resolve_discard_response(
        [{"type": "dahai", "actor": 0, "pai": "1m"}, {"type": "ryukyoku", "reason": reason}], 0
    )
    assert result.response_target == expected


def test_unknown_or_missing_response_is_not_fabricated():
    trigger = {"type": "dahai", "actor": 0, "pai": "1m"}
    for suffix in ([], [{"type": "end_kyoku"}], [{"type": "ryukyoku"}], [{"type": "ryukyoku", "reason": "triple_ron"}]):
        assert resolve_discard_response([trigger, *suffix], 0).response_observed is False
    assert resolve_discard_response([{"type": "ankan"}], 0).response_applicable is False


def test_public_mask_uses_visible_furiten_and_mjsoul_rule_only():
    view = ledger(round_events())
    view.apply_event({"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False})
    state = view.snapshot()
    allowed = public_rule_response_mask(state, 4)
    assert allowed[Response.ABORT_TRIPLE_RON] is False
    assert allowed[Response.RON_MASK_1] is False
    assert allowed[Response.RON_MASK_2] is True
    assert allowed[Response.RON_MASK_7] is False
    assert public_rule_response_mask(ledger(round_events()).snapshot(), 4)[Response.RON_MASK_7] is True
    with pytest.raises(ValueError):
        validate_response_target(allowed, Response.RON_MASK_1)
    assert not hasattr(state, "opponent_legal_actions")


def test_last_tile_masks_normal_draw_but_allows_end():
    # This is ordinary exhaustion: every player has visibly discarded a
    # middle tile, so complete history rules out nagashi before the last tile.
    rivers = ((4,), (40,), (76,), (8,))
    state = replace(ledger(round_events()).snapshot(), remaining_draws=0, discard_history=rivers, current_rivers=rivers)
    allowed = public_rule_response_mask(state, 4)
    assert allowed[Response.NEXT_DRAW] is False
    assert allowed[Response.EXHAUSTIVE_DRAW] is True
    assert allowed[5:8] == (False, False, False)
    with pytest.raises(ValueError):
        validate_response_target((False,) * 21, 0)
    with pytest.raises(ValueError):
        public_rule_response_mask(state, 219)


def test_unknown_history_cannot_imply_public_furiten():
    state = ledger(round_events()).snapshot()
    discards = ((0,), (0,), (), ())
    state = replace(state, history_complete=False, discard_history=discards)
    assert public_rule_response_mask(state, 4)[Response.RON_MASK_1] is True


def test_unknown_observer_hand_does_not_make_known_counts():
    state = ledger(round_events(["?"] * 13)).snapshot()
    assert state.physical_ledger_complete is False
    assert state.known_counts() is None
    assert convert.mjai_to_tid("5pr") // 4 == 13


def _nagashi_discard(tile):
    return tile136_to_tile37(convert.mjai_to_tid(tile)) * 4


def _nagashi_snapshot(eligible_player=0):
    history = tuple((convert.mjai_to_tid("9p" if player == eligible_player else "5s"),) for player in range(4))
    return replace(
        ledger(round_events()).snapshot(),
        discard_history=history,
        current_rivers=history,
        remaining_draws=0,
    )


def test_nagashi_no_meld_and_only_terminal_honor_discards_remains_possible():
    state = _nagashi_snapshot()
    original_history = state.discard_history
    assert not any(state.melds)
    assert public_rule_response_mask(state, _nagashi_discard("1m"))[Response.NAGASHI_MANGAN] is True
    assert state.discard_history == original_history


@pytest.mark.parametrize("player", range(4))
@pytest.mark.parametrize(
    "kind,tiles",
    [
        ("chi", ("1s", "2s", "3s")),
        ("pon", ("P",) * 3),
        ("daiminkan", ("P",) * 4),
        ("ankan", ("P",) * 4),
        ("kakan", ("P",) * 4),
    ],
    ids=["B-chi", "C-pon", "D-daiminkan", "E-ankan", "F-committed-kakan"],
)
def test_nagashi_own_committed_meld_does_not_disqualify_under_mjsoul(player, kind, tiles):
    # Mahjong Soul FAQ id=47 defines eligibility by own unclaimed discards,
    # not by menzen or meld count: https://mahjongsoul.com/faq/
    state = _nagashi_snapshot(player)
    melds = list(state.melds)
    melds[player] = (PublicMeld(kind, tuple(convert.mjai_to_tid(tile) for tile in tiles)),)
    called = replace(state, melds=tuple(melds))
    assert _nagashi_player_possible(called, player, actor=0, tile=0) is True
    assert public_rule_response_mask(called, _nagashi_discard("1m"))[Response.NAGASHI_MANGAN] is True
    # Preserve the original scenarios and assert their containers stay unchanged.
    assert called.discard_history == state.discard_history
    assert called.current_rivers == state.current_rivers


def test_nagashi_pending_kakan_preserves_the_existing_pon_not_a_committed_kan():
    view = ledger(
        round_events()
        + [
            {"type": "dahai", "actor": 2, "pai": "1s", "tsumogiri": False},
            {"type": "dahai", "actor": 3, "pai": "5s", "tsumogiri": False},
            {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": False},
            {"type": "pon", "actor": 2, "target": 1, "pai": "9p", "consumed": ["9p", "9p"]},
        ]
    )
    before = view.snapshot()
    # Only actor 0's 2m candidate, P1's called-away discard and P3's middle
    # tile disqualify those seats. P2's existing PON still permits nagashi.
    assert _nagashi_possible(before, actor=0, tile=1) is True
    view.apply_event({"type": "kakan", "actor": 2, "pai": "9p"})
    pending = view.snapshot()
    assert pending.pending_kan_actor == 2 and pending.pending_kan_kind == "kakan"
    assert pending.melds == before.melds
    assert pending.melds[2][0].kind == "pon" and len(pending.melds[2][0].tiles) == 3
    assert pending.known_counts()[17] == 3
    assert _nagashi_possible(pending, actor=0, tile=1) is True
    # A kan response window is still not a valid public discard-mask entry.
    with pytest.raises(ValueError, match="pending kan"):
        public_rule_response_mask(pending, _nagashi_discard("2m"))


@pytest.mark.parametrize("kind", ["chi", "pon", "daiminkan"])
def test_nagashi_called_away_terminal_stays_in_history_but_disqualifies_owner(kind):
    state = _nagashi_snapshot(eligible_player=1)
    rivers = list(state.current_rivers)
    rivers[1] = ()
    tiles = ("7p", "8p", "9p") if kind == "chi" else ("9p",) * (4 if kind == "daiminkan" else 3)
    melds = list(state.melds)
    melds[2] = (PublicMeld(kind, tuple(convert.mjai_to_tid(tile) for tile in tiles), source=1),)
    called = replace(state, current_rivers=tuple(rivers), melds=tuple(melds))
    assert called.discard_history[1] == (convert.mjai_to_tid("9p"),)
    assert called.current_rivers[1] == () and not called.melds[1]
    assert public_rule_response_mask(called, _nagashi_discard("1m"))[Response.NAGASHI_MANGAN] is False


@pytest.mark.parametrize("middle_source", ["history", "candidate"])
def test_nagashi_middle_tile_in_history_or_current_candidate_disqualifies(middle_source):
    state = _nagashi_snapshot()
    candidate = _nagashi_discard("2m" if middle_source == "candidate" else "1m")
    if middle_source == "history":
        history = ((convert.mjai_to_tid("2m"),), *state.discard_history[1:])
        state = replace(state, discard_history=history, current_rivers=history)
    assert public_rule_response_mask(state, candidate)[Response.NAGASHI_MANGAN] is False


@pytest.mark.parametrize("eligible_player", range(4))
def test_nagashi_checks_all_players_and_adds_candidate_only_to_its_actor(eligible_player):
    state = _nagashi_snapshot(eligible_player)
    assert public_rule_response_mask(state, _nagashi_discard("1m"))[Response.NAGASHI_MANGAN] is True
    assert public_rule_response_mask(state, _nagashi_discard("2m"))[Response.NAGASHI_MANGAN] is (eligible_player != 0)
    for player in range(4):
        assert _nagashi_player_possible(state, player, actor=0, tile=0) is (player == eligible_player)
    # Another player's meld must not disqualify the eligible player.
    melds = list(state.melds)
    melds[(eligible_player + 1) % 4] = (PublicMeld("pon", (convert.mjai_to_tid("P"),) * 3),)
    assert (
        public_rule_response_mask(replace(state, melds=tuple(melds)), _nagashi_discard("1m"))[Response.NAGASHI_MANGAN]
        is True
    )


def test_nagashi_all_four_players_are_independent_candidates():
    state = _nagashi_snapshot()
    history = ((convert.mjai_to_tid("9p"),),) * 4
    state = replace(state, discard_history=history, current_rivers=history)
    assert all(_nagashi_player_possible(state, player, actor=0, tile=0) for player in range(4))
    assert _nagashi_player_possible(state, 0, actor=0, tile=1) is False
    assert all(_nagashi_player_possible(state, player, actor=0, tile=1) for player in (1, 2, 3))
    assert public_rule_response_mask(state, _nagashi_discard("2m"))[Response.NAGASHI_MANGAN] is True


def _nagashi_incomplete_snapshot(actor):
    history = ((convert.mjai_to_tid("9p"),),) * 4
    return replace(
        _nagashi_snapshot(),
        observer_player_id=actor,
        history_complete=False,
        discard_history=history,
        current_rivers=history,
    )


@pytest.mark.parametrize("actor", range(4))
def test_nagashi_incomplete_history_with_definite_disqualifiers(actor):
    state = replace(_nagashi_snapshot(actor), observer_player_id=actor, history_complete=False)
    # The old early return ignored known facts: the actor's 2m candidate and
    # every other player's observed 5s. Missing events do not invalidate either.
    assert all(not _nagashi_player_possible(state, player, actor, tile=1) for player in range(4))
    assert _nagashi_possible(state, actor, tile=1) is False
    assert public_rule_response_mask(state, _nagashi_discard("2m"))[Response.NAGASHI_MANGAN] is False


@pytest.mark.parametrize("actor", range(4))
@pytest.mark.parametrize("known_tiles", [(), ("1m", "9p", "E")], ids=["empty", "terminals-and-honor"])
def test_nagashi_incomplete_history_without_definite_disqualifiers(actor, known_tiles):
    history = (tuple(convert.mjai_to_tid(tile) for tile in known_tiles),) * 4
    state = replace(_nagashi_incomplete_snapshot(actor), discard_history=history, current_rivers=history)
    assert all(_nagashi_player_possible(state, player, actor, tile=0) for player in range(4))
    assert _nagashi_possible(state, actor, tile=0) is True
    assert public_rule_response_mask(state, _nagashi_discard("1m"))[Response.NAGASHI_MANGAN] is True
    assert not public_rule_response_mask(state, _nagashi_discard("1m"), replace(MJSOUL_RULESET, nagashi_mangan=False))[
        Response.NAGASHI_MANGAN
    ]


@pytest.mark.parametrize("actor", range(4))
def test_nagashi_incomplete_history_own_melds_do_not_fabricate_disqualification(actor):
    # Official FAQ's unclaimed-discard condition does not add a menzen rule.
    state = _nagashi_incomplete_snapshot(actor)
    melds = tuple((PublicMeld("pon", (convert.mjai_to_tid(wind),) * 3),) for wind in ("E", "S", "W", "N"))
    called = replace(state, melds=melds)
    assert all(_nagashi_player_possible(called, player, actor, tile=0) for player in range(4))
    assert public_rule_response_mask(called, _nagashi_discard("1m"))[Response.NAGASHI_MANGAN] is True
    assert called.discard_history == state.discard_history and called.current_rivers == state.current_rivers


@pytest.mark.parametrize("actor", range(4))
@pytest.mark.parametrize("player", range(4))
@pytest.mark.parametrize("source", ["history", "river"])
def test_nagashi_incomplete_history_rejects_each_known_middle_discard_source(actor, player, source):
    state = _nagashi_incomplete_snapshot(actor)
    history, rivers = list(state.discard_history), list(state.current_rivers)
    middle = convert.mjai_to_tid("2m")
    # PublicSnapshot has no cross-container validation. A partial view can
    # retain either known public record; the other container cannot erase it.
    history[player] = (middle,) if source == "history" else ()
    rivers[player] = (middle,) if source == "river" else ()
    state = replace(state, discard_history=tuple(history), current_rivers=tuple(rivers))
    assert not state.melds[player]
    assert _nagashi_player_possible(state, player, actor, tile=0) is False


@pytest.mark.parametrize("actor", range(4))
@pytest.mark.parametrize("player", range(4))
def test_nagashi_incomplete_history_middle_candidate_only_disqualifies_its_actor(actor, player):
    state = _nagashi_incomplete_snapshot(actor)
    assert _nagashi_player_possible(state, player, actor, tile=1) is (player != actor)


@pytest.mark.parametrize("actor", range(4))
@pytest.mark.parametrize("player", range(4))
@pytest.mark.parametrize(
    "history_tiles,river_tiles",
    [(("9p", "E"), ("E",)), (("E",), ("9p", "E"))],
    ids=["history-longer", "river-longer"],
)
def test_nagashi_incomplete_history_length_difference_alone_is_not_called_away(
    actor, player, history_tiles, river_tiles
):
    state = _nagashi_incomplete_snapshot(actor)
    history, rivers = list(state.discard_history), list(state.current_rivers)
    history[player] = tuple(convert.mjai_to_tid(tile) for tile in history_tiles)
    rivers[player] = tuple(convert.mjai_to_tid(tile) for tile in river_tiles)
    state = replace(state, discard_history=tuple(history), current_rivers=tuple(rivers))
    assert len(state.discard_history[player]) != len(state.current_rivers[player])
    assert _nagashi_player_possible(state, player, actor, tile=0) is True


@pytest.mark.parametrize("actor", range(4))
@pytest.mark.parametrize("relative", [1, 2, 3])
def test_nagashi_incomplete_history_any_other_possible_player_keeps_category(actor, relative):
    eligible_player = (actor + relative) % 4
    state = _nagashi_incomplete_snapshot(actor)
    history = tuple((convert.mjai_to_tid("9p" if player in {actor, eligible_player} else "5s"),) for player in range(4))
    state = replace(state, discard_history=history, current_rivers=history)
    # Only the candidate disqualifies the actor; exactly one other seat remains.
    expected = [player == eligible_player for player in range(4)]
    assert [_nagashi_player_possible(state, player, actor, tile=1) for player in range(4)] == expected
    assert _nagashi_possible(state, actor, tile=1) is True
    assert public_rule_response_mask(state, _nagashi_discard("2m"))[Response.NAGASHI_MANGAN] is True


@pytest.mark.parametrize("actor", range(4))
@pytest.mark.parametrize("history_complete", [True, False])
@pytest.mark.parametrize("source", ["history", "river", "candidate"])
def test_nagashi_evidence_checks_are_read_only_and_preserve_other_response_categories(actor, history_complete, source):
    control = replace(_nagashi_snapshot(actor), observer_player_id=actor, history_complete=history_complete)
    melds = list(control.melds)
    melds[(actor + 1) % 4] = (PublicMeld("pon", (convert.mjai_to_tid("E"),) * 3),)
    control = replace(control, melds=tuple(melds))
    history, rivers = list(control.discard_history), list(control.current_rivers)
    if source == "history":
        history[actor] = (convert.mjai_to_tid("2m"),)
    elif source == "river":
        rivers[actor] = (convert.mjai_to_tid("2m"),)
    state = replace(control, discard_history=tuple(history), current_rivers=tuple(rivers))
    candidate = _nagashi_discard("2m" if source == "candidate" else "1m")
    before, control_before = state.to_bytes(), control.to_bytes()
    base_tile = convert.mjai_to_tid("2m" if source == "candidate" else "1m") // 4
    for player in range(4):
        assert _nagashi_player_possible(state, player, actor, base_tile) is False
    assert _nagashi_possible(state, actor, base_tile) is False
    expected = public_rule_response_mask(control, _nagashi_discard("1m"))
    actual = public_rule_response_mask(state, candidate)
    assert expected[Response.NAGASHI_MANGAN] is True
    assert actual[Response.NAGASHI_MANGAN] is False
    # Complete history proves that the control must end in nagashi absent Ron.
    # Disqualifying its only eligible player restores ordinary exhaustion.
    assert expected[Response.EXHAUSTIVE_DRAW] is (not history_complete)
    assert actual[Response.EXHAUSTIVE_DRAW] is True
    assert all(
        actual[index] == value
        for index, value in enumerate(expected)
        if index not in {Response.NAGASHI_MANGAN, Response.EXHAUSTIVE_DRAW}
    )
    assert state.to_bytes() == before and control.to_bytes() == control_before
    assert state.discard_history == tuple(history) and state.current_rivers == tuple(rivers)
    assert state.melds == control.melds


@pytest.mark.parametrize("history_complete", [True, False])
@pytest.mark.parametrize("physical_ledger_complete", [True, False])
@pytest.mark.parametrize("visibility_complete", [True, False])
def test_snapshot_known_counts_requires_physical_and_visibility_not_history(
    history_complete, physical_ledger_complete, visibility_complete
):
    state = replace(
        ledger(round_events()).snapshot(),
        history_complete=history_complete,
        physical_ledger_complete=physical_ledger_complete,
        visibility_complete=visibility_complete,
    )
    counts = state.known_counts()
    if physical_ledger_complete and visibility_complete:
        assert counts is not None and len(counts) == 34
        assert counts[0] == 1 and sum(counts) == 14
    else:
        assert counts is None


@pytest.mark.parametrize("missing", ["history_complete", "physical_ledger_complete", "visibility_complete"])
def test_tokenizer_still_rejects_each_incomplete_prefix_flag(missing):
    state = ledger(round_events() + [{"type": "tsumo", "actor": 0, "pai": "9p"}]).snapshot()
    candidates = (_nagashi_discard("1m"), _nagashi_discard("2m"))
    assert encode_prefix(state, candidates).rows
    with pytest.raises(TokenizationUnavailableError, match="complete audited observer prefix"):
        encode_prefix(replace(state, **{missing: False}), candidates)


@pytest.mark.parametrize("actor", range(4))
def test_response_v2_reads_only_the_frozen_public_fact_whitelist(actor):
    state = replace(_nagashi_snapshot(actor), observer_player_id=actor)
    allowed_names = {
        "observer_player_id",
        "discard_history",
        "current_rivers",
        "melds",
        "riichi_accepted",
        "remaining_draws",
        "history_complete",
        "pending_kan_actor",
    }

    class PublicFactsOnly:
        def __getattr__(self, name):
            assert name in allowed_names, f"Response mask read non-whitelisted field: {name}"
            return getattr(state, name)

    expected = public_rule_response_mask(state, _nagashi_discard("2m"))
    assert public_rule_response_mask(PublicFactsOnly(), _nagashi_discard("2m")) == expected
    poisoned = replace(
        state,
        hand=(135,) * 13,
        scores=(1, 2, 3, 4),
        dora_indicators=(0,),
        events=("future/hidden/private-legals must never be parsed here",),
    )
    assert public_rule_response_mask(poisoned, _nagashi_discard("2m")) == expected


def test_response_v2_freezes_abortive_draw_and_triple_ron_public_conditions():
    state = ledger(round_events()).snapshot()
    east = convert.mjai_to_tid("E")
    winds = ((), (east,), (east,), (east,))
    four_winds = replace(state, discard_history=winds, current_rivers=winds, remaining_draws=66)
    assert public_rule_response_mask(four_winds, _nagashi_discard("E"))[Response.ABORT_FOUR_WINDS]
    assert not public_rule_response_mask(four_winds, _nagashi_discard("S"))[Response.ABORT_FOUR_WINDS]
    reached = replace(state, riichi_accepted=(False, True, True, True))
    assert public_rule_response_mask(reached, _nagashi_discard("1m") + 1)[Response.ABORT_FOUR_RIICHI]
    assert not public_rule_response_mask(reached, _nagashi_discard("1m"))[Response.ABORT_FOUR_RIICHI]
    kans = tuple(PublicMeld("ankan", (convert.mjai_to_tid(wind),) * 4) for wind in ("E", "S", "W", "N"))
    split_kans = replace(state, melds=((kans[0], kans[1]), (kans[2], kans[3]), (), ()))
    single_owner = replace(state, melds=(kans, (), (), ()))
    assert public_rule_response_mask(split_kans, _nagashi_discard("1m"))[Response.ABORT_FOUR_KANS]
    assert not public_rule_response_mask(single_owner, _nagashi_discard("1m"))[Response.ABORT_FOUR_KANS]
    normal = public_rule_response_mask(state, _nagashi_discard("1m"))
    assert normal[Response.RON_MASK_7] and not normal[Response.ABORT_TRIPLE_RON]
    alternate = public_rule_response_mask(state, _nagashi_discard("1m"), replace(MJSOUL_RULESET, sanchaho_is_draw=True))
    assert alternate[Response.ABORT_TRIPLE_RON] and not alternate[Response.RON_MASK_7]
