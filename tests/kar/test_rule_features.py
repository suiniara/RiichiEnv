from dataclasses import FrozenInstanceError, replace

import pytest

from riichienv import GameRule, Phase, RiichiEnv, convert
from riichienv.kar.actions import legal_candidates
from riichienv.kar.contracts import Response
from riichienv.kar.observation import ObserverLedger, extract_rule_features
from riichienv.kar.response_mask import public_rule_response_mask
from riichienv.kar.rule_features import (
    OTHER_WAITS_BY_RANK,
    RYANMEN_PAIRS_BY_RANK,
    current_rank_features,
    full_suji_exclusion,
    genbutsu,
    known_count_base34,
    no_visible_possible_ryanmen_shape,
    public_dora_types,
)
from riichienv.kar.settlement import verify_settlements
from riichienv.kar.tokenize import TokenHistoryCache, encode_prefix
from tests.kar.test_decision_time_parity import _Episode

RANK_CASES = (
    (1, ((2, 3),), (4,)),
    (2, ((3, 4),), (5,)),
    (3, ((4, 5),), (6,)),
    (4, ((2, 3), (5, 6)), (1, 7)),
    (5, ((3, 4), (6, 7)), (2, 8)),
    (6, ((4, 5), (7, 8)), (3, 9)),
    (7, ((5, 6),), (4,)),
    (8, ((6, 7),), (5,)),
    (9, ((7, 8),), (6,)),
)


@pytest.mark.parametrize("rank,pairs,other_waits", RANK_CASES)
def test_frozen_ryanmen_tables(rank, pairs, other_waits):
    assert RYANMEN_PAIRS_BY_RANK[rank] == pairs
    assert OTHER_WAITS_BY_RANK[rank] == other_waits
    assert set(RYANMEN_PAIRS_BY_RANK) == set(range(1, 10))
    assert set(OTHER_WAITS_BY_RANK) == set(range(1, 10))


@pytest.mark.parametrize("suit_start", [0, 9, 18])
@pytest.mark.parametrize("rank,pairs,other_waits", RANK_CASES)
def test_full_suji_requires_every_opposite_wait_in_the_same_suit(suit_start, rank, pairs, other_waits):
    target = suit_start + rank - 1
    history = {suit_start + wait - 1 for wait in other_waits}
    assert full_suji_exclusion(target, set()) is False
    assert full_suji_exclusion(target, {target}) is False
    assert full_suji_exclusion(target, history) is True
    for wait in history:
        assert full_suji_exclusion(target, history - {wait}) is False
    another_suit = (suit_start + 9) % 27
    assert full_suji_exclusion(target, {another_suit + wait - 1 for wait in other_waits}) is False


@pytest.mark.parametrize("suit_start", [0, 9, 18])
@pytest.mark.parametrize("rank,pairs,other_waits", RANK_CASES)
def test_visible_ryanmen_requires_every_possible_pair_to_be_blocked(suit_start, rank, pairs, other_waits):
    target = suit_start + rank - 1
    counts = [0] * 34
    assert no_visible_possible_ryanmen_shape(target, counts) is False
    for left, _ in pairs:
        counts[suit_start + left - 1] = 4
    assert no_visible_possible_ryanmen_shape(target, counts) is True
    for left, _ in pairs:
        incomplete = counts.copy()
        incomplete[suit_start + left - 1] = 3
        assert no_visible_possible_ryanmen_shape(target, incomplete) is False

    counts = [0] * 34
    counts[target] = 4
    # A candidate discard can be one of these four visible target tiles.
    assert no_visible_possible_ryanmen_shape(target, counts) is False

    counts = [0] * 34
    another_suit = (suit_start + 9) % 27
    for left, _ in pairs:
        counts[another_suit + left - 1] = 4
    assert no_visible_possible_ryanmen_shape(target, counts) is False


def test_four_held_target_tiles_do_not_exclude_a_real_ryanmen_ron():
    # 23m + 456p + 789p + EEE + NN waits on 1m/4m; East supplies a yaku.
    # _Episode deals a complete, unique 136-tile wall and executes native actions.
    waiting_hand = ("2m", "3m", "4p", "5p", "6p", "7p", "8p", "9p", "E", "E", "E", "N", "N")
    episode = _Episode({0: ("1m",) * 4, 1: waiting_hand}, ("2s",))
    snapshot = episode.live.snapshot(0)
    counts = snapshot.known_counts()
    assert counts is not None and counts[0] == 4
    assert sum(tile // 4 == 0 for tile in snapshot.hand) == 4
    assert no_visible_possible_ryanmen_shape(0, counts) is False
    assert extract_rule_features(snapshot)["hand_count_flags"][0][6] == 1

    candidates = legal_candidates(episode.obs[0])
    discard = next(
        candidate
        for candidate in candidates
        if candidate.spec.action_type == "DISCARD" and candidate.spec.base34 == 0 and not candidate.spec.riichi
    )
    assert public_rule_response_mask(snapshot, discard.global_id)[Response.RON_MASK_1]
    ids = tuple(candidate.global_id for candidate in candidates)
    cache = TokenHistoryCache()
    for _ in range(2):
        prefix = encode_prefix(snapshot, ids, history_cache=cache)
        hand_count = next(row for row in prefix.rows if row["kind"] == 4 and row["tiles"][0] == discard.spec.tile37 + 1)
        assert hand_count["flags"][6] == 1
        assert prefix.to_bytes() == encode_prefix(snapshot, ids).to_bytes()

    episode.act(tile="1m")
    assert episode.env.phase == Phase.WaitResponse and 1 in episode.env.active_players
    assert any(candidate.spec.action_type == "RON" for candidate in legal_candidates(episode.obs[1], source_player=0))
    episode.resolve("RON", seat=1)
    end = next(index for index, event in enumerate(episode.live.events) if event["type"] == "end_kyoku")
    events = episode.live.events[: end + 1]
    win = next(event for event in events if event["type"] == "hora")
    assert (win["actor"], win["target"], win["pai"]) == (1, 0, "1m")
    assert win["scores"] == [23700, 26300, 25000, 25000]
    assert verify_settlements(events)["status"] == "PASS"


@pytest.mark.parametrize("suit_start", [0, 9, 18])
@pytest.mark.parametrize("target_rank,valid_block,penchan_block", [(3, 4, 1), (7, 5, 8)])
def test_penchan_pairs_neither_preserve_nor_block_ryanmen(suit_start, target_rank, valid_block, penchan_block):
    target = suit_start + target_rank - 1
    counts = [0] * 34
    counts[suit_start + valid_block - 1] = 4
    assert no_visible_possible_ryanmen_shape(target, counts) is True

    counts = [0] * 34
    counts[suit_start + penchan_block - 1] = 4
    assert no_visible_possible_ryanmen_shape(target, counts) is False


@pytest.mark.parametrize("target", range(27, 34))
def test_honor_shape_features_are_not_applicable(target):
    counts = [4] * 34
    assert full_suji_exclusion(target, set(range(34))) is None
    assert no_visible_possible_ryanmen_shape(target, counts) is None
    assert genbutsu(target, {target}) is True
    assert genbutsu(target, set()) is False


def test_missing_history_or_public_ledger_is_unknown_not_false():
    assert genbutsu(4, None) is None
    assert full_suji_exclusion(4, None) is None
    assert no_visible_possible_ryanmen_shape(4, None) is None
    assert genbutsu(4, set()) is False
    assert full_suji_exclusion(4, set()) is False
    assert no_visible_possible_ryanmen_shape(4, [0] * 34) is False


def test_known_count_preserves_distinct_copies_across_public_regions():
    counts = known_count_base34(
        observer_hand=[4],
        current_rivers=[[4], [4], [], []],
        current_melds=[[], [], [], []],
        dora_indicators=[4],
        other_public_reveals=[27],
    )
    assert isinstance(counts, tuple)
    assert len(counts) == 34
    assert counts[4] == 4
    assert counts[27] == 1
    assert sum(counts) == 5
    assert no_visible_possible_ryanmen_shape(4, counts) is False


def test_called_away_discard_remains_in_history_but_not_the_current_river():
    history = {4}
    counts = known_count_base34(
        observer_hand=[],
        current_rivers=[[], [], [], []],
        current_melds=[[], [], [4, 4, 4], []],
        dora_indicators=[],
    )
    assert genbutsu(4, history) is True
    assert counts[4] == 3
    assert history == {4}


def test_meld_tiles_do_not_become_the_owners_discard_history():
    counts = known_count_base34([], [[], [], [], []], [[], [4, 4, 4, 4], [], []], [])
    assert counts[4] == 4
    assert genbutsu(4, set()) is False


def test_red_and_nonred_fives_each_count_once_after_external_base34_conversion():
    counts = known_count_base34([4, 4], [[4], [], [], []], [[], [], [], []], [4])
    assert counts[4] == 4


def test_known_count_empty_regions_are_real_zero():
    assert known_count_base34([], [], [], []) == (0,) * 34


def test_known_count_rejects_more_than_four_visible_copies_without_clipping():
    with pytest.raises(ValueError, match="more than four"):
        known_count_base34([4], [[4, 4], [], [], []], [[], [], [], []], [4, 4])


@pytest.mark.parametrize("tile", [-1, 34, 136, 1.5, True, "4"])
def test_invalid_base34_tiles_are_rejected(tile):
    with pytest.raises(ValueError, match="base34"):
        genbutsu(tile, set())
    with pytest.raises(ValueError, match="base34"):
        full_suji_exclusion(tile, set())
    with pytest.raises(ValueError, match="base34"):
        no_visible_possible_ryanmen_shape(tile, [0] * 34)
    with pytest.raises(ValueError, match="base34"):
        public_dora_types([tile])
    with pytest.raises(ValueError, match="base34"):
        known_count_base34([tile], [], [], [])


@pytest.mark.parametrize("region", ["hand", "river", "meld", "indicator", "reveal"])
def test_all_visible_regions_validate_tile_ids(region):
    inputs = {
        "observer_hand": [],
        "current_rivers": [[], [], [], []],
        "current_melds": [[], [], [], []],
        "dora_indicators": [],
        "other_public_reveals": [],
    }
    if region == "hand":
        inputs["observer_hand"] = [34]
    elif region == "river":
        inputs["current_rivers"][2] = [34]
    elif region == "meld":
        inputs["current_melds"][1] = [34]
    elif region == "indicator":
        inputs["dora_indicators"] = [34]
    else:
        inputs["other_public_reveals"] = [34]
    with pytest.raises(ValueError, match="base34"):
        known_count_base34(**inputs)


@pytest.mark.parametrize("history", [{-1}, {34}, {False}, {1.5}])
def test_discard_history_is_validated(history):
    with pytest.raises(ValueError, match="base34"):
        genbutsu(4, history)
    with pytest.raises(ValueError, match="base34"):
        full_suji_exclusion(4, history)


@pytest.mark.parametrize("count", [-1, 5, 1.5, True])
def test_visible_count_entries_must_be_exact_integers_between_zero_and_four(count):
    counts = [0] * 34
    counts[12] = count
    with pytest.raises(ValueError, match="known counts"):
        no_visible_possible_ryanmen_shape(4, counts)


@pytest.mark.parametrize("length", [0, 33, 35])
def test_visible_count_vector_requires_all_34_tile_types(length):
    with pytest.raises(ValueError, match="34 entries"):
        no_visible_possible_ryanmen_shape(4, [0] * length)


@pytest.mark.parametrize("indicator", range(34))
def test_public_dora_cycles(indicator):
    if indicator < 27:
        expected = indicator + 1 if indicator % 9 != 8 else indicator - 8
    elif indicator < 31:
        expected = indicator + 1 if indicator != 30 else 27
    else:
        expected = indicator + 1 if indicator != 33 else 31
    assert public_dora_types([indicator]) == frozenset({expected})


def test_public_dora_uses_only_given_public_indicators():
    assert public_dora_types([]) == frozenset()
    assert public_dora_types([8, 8, 30, 33]) == frozenset({0, 27, 31})


@pytest.mark.parametrize("observer", range(4))
def test_equal_scores_follow_fixed_initial_order_and_keep_zero_gaps_present(observer):
    features = current_rank_features([25000] * 4, [0, 1, 2, 3], observer)
    assert features.rank == observer + 1
    assert features.gap_to_higher == 0.0
    assert features.gap_to_lower == 0.0
    assert features.gap_to_higher_present is (observer > 0)
    assert features.gap_to_lower_present is (observer < 3)


def test_tie_break_uses_explicit_initial_order_not_player_number():
    order = [3, 1, 0, 2]
    for rank, observer in enumerate(order, 1):
        assert current_rank_features([25000] * 4, order, observer).rank == rank
    assert order == [3, 1, 0, 2]


def test_rank_features_support_named_player_ids_and_normalize_raw_gaps():
    scores = {"east": 20000, "south": 35000, "west": 35000, "north": 10000}
    order = ["west", "east", "south", "north"]
    assert current_rank_features(scores, order, "west").rank == 1
    assert current_rank_features(scores, order, "south").rank == 2
    features = current_rank_features(scores, order, "east")
    assert features.rank == 3
    assert features.gap_to_higher_present is True
    assert features.gap_to_higher == pytest.approx(0.3)
    assert features.gap_to_lower_present is True
    assert features.gap_to_lower == pytest.approx(0.2)


def test_rank_features_do_not_clip_large_gaps_or_reject_negative_points():
    features = current_rank_features([-10000, 120000, 25000, 25000], [0, 1, 2, 3], 1)
    assert features.rank == 1
    assert features.gap_to_higher_present is False
    assert features.gap_to_lower_present is True
    assert features.gap_to_lower == pytest.approx(1.9)


def test_current_dealer_rotation_cannot_change_a_tie_break():
    env = RiichiEnv(seed=42, rule=GameRule.default_mjsoul())
    initial_order = [0, 1, 2, 3]
    for current_oya in range(4):
        env.reset(oya=current_oya, scores=[25000] * 4)
        observation = env.get_observation(0)
        assert observation.oya == current_oya
        actual = [current_rank_features(observation.scores, initial_order, player).rank for player in range(4)]
        assert actual == env.ranks() == [1, 2, 3, 4]


def test_rank_features_are_immutable():
    features = current_rank_features([25000] * 4, [0, 1, 2, 3], 0)
    with pytest.raises(FrozenInstanceError):
        features.rank = 4


@pytest.mark.parametrize("order", [[], [0, 1, 2], [0, 1, 2, 2], [0, 1, 2, 4]])
def test_rank_features_require_the_same_four_distinct_players(order):
    with pytest.raises(ValueError, match="four"):
        current_rank_features([25000] * 4, order, 0)


def test_rank_features_reject_unknown_observer():
    with pytest.raises(ValueError, match="observer"):
        current_rank_features([25000] * 4, [0, 1, 2, 3], 4)


@pytest.mark.parametrize("score", [25000.0, True, "25000"])
def test_rank_features_require_raw_integer_scores(score):
    with pytest.raises(ValueError, match="raw integer"):
        current_rank_features([score, 25000, 25000, 25000], [0, 1, 2, 3], 0)


def _rule_feature_snapshot():
    view = ObserverLedger(0, (0, 1, 2, 3))
    view.apply_event({"type": "start_game"})
    view.apply_event(
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000] * 4,
            "dora_marker": "5p",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "3p", "5p", "C"],
                ["?"] * 13,
                ["?"] * 13,
                ["?"] * 13,
            ],
        }
    )
    history = (
        (),
        tuple(convert.mjai_to_tid(tile) for tile in ("2p", "8p", "5p")),
        (convert.mjai_to_tid("5p"),),
        (),
    )
    return replace(view.snapshot(), discard_history=history, current_rivers=history)


@pytest.mark.parametrize("history_complete", [True, False])
@pytest.mark.parametrize("physical_ledger_complete", [True, False])
@pytest.mark.parametrize("visibility_complete", [True, False])
def test_rule_features_preserve_each_completeness_dependency(
    history_complete, physical_ledger_complete, visibility_complete
):
    state = replace(
        _rule_feature_snapshot(),
        history_complete=history_complete,
        physical_ledger_complete=physical_ledger_complete,
        visibility_complete=visibility_complete,
    )
    original_history = state.discard_history
    counts = state.known_counts()
    reliable_counts = physical_ledger_complete and visibility_complete
    if reliable_counts:
        assert counts is not None and len(counts) == 34 and counts[13] == 4
    else:
        assert counts is None
    features = extract_rule_features(state)
    flags = features["hand_count_flags"]
    assert len(flags) == 34 and all(len(row) == 16 for row in flags)
    # Genbutsu P1/P2/P3 then Full Suji P1/P2/P3: 0=Unknown, 1=False, 2=True.
    assert flags[13][:6] == ((2, 2, 1, 2, 1, 1) if history_complete else (0,) * 6)
    assert flags[13][6] == (1 if reliable_counts else 0)
    # The explicitly given public 5p indicator still identifies 6p as dora,
    # independently of missing history or unrelated physical/visibility gaps.
    assert flags[14][7] == 2 and flags[13][7] == 1
    assert all(row[8:] == (0,) * 8 for row in flags)
    assert state.discard_history == original_history


def test_missing_physical_information_keeps_complete_public_discard_features():
    state = replace(_rule_feature_snapshot(), physical_ledger_complete=False)
    features = extract_rule_features(state)["hand_count_flags"]
    assert features[13][:6] == (2, 2, 1, 2, 1, 1)
    assert all(row[6] == 0 for row in features)
    assert state.discard_history[1]


def test_missing_history_keeps_physical_counts_but_not_genbutsu_or_full_suji():
    state = replace(_rule_feature_snapshot(), history_complete=False)
    assert state.known_counts()[13] == 4
    flags = extract_rule_features(state)["hand_count_flags"]
    assert all(row[:6] == (0,) * 6 for row in flags)
    assert flags[13][6] == 1


def test_no_public_dora_indicator_is_unknown_not_confirmed_non_dora():
    state = replace(_rule_feature_snapshot(), dora_indicators=(), physical_ledger_complete=False)
    flags = extract_rule_features(state)["hand_count_flags"]
    assert all(row[7] == 0 for row in flags)
    assert flags[13][:6] == (2, 2, 1, 2, 1, 1)
