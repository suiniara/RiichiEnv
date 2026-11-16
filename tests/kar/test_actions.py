from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from riichienv import Action, ActionType, GameRule, HandEvaluator, Observation, RiichiEnv
from riichienv.kar.actions import (
    ACTION_COUNT,
    ACTION_MANIFEST_SHA256,
    MAX_CANDIDATES,
    AmbiguousDiscardError,
    action_id,
    action_spec,
    execute_candidate,
    legal_candidates,
    manifest,
    manifest_hash,
    tile37_to_base34,
    tile136_to_tile37,
)

_HAND = [36, 37, 38, 44, 45, 46, 52, 53, 54, 60, 61, 62, 108, 109]


def _tid(tile37):
    if tile37 >= 30:
        return 108 + 4 * (tile37 - 30)
    suit, rank = divmod(tile37, 10)
    return 36 * suit + (16 if rank == 0 else 17 if rank == 5 else 4 * (rank - 1))


def _observation(hand, actions, *, drawn=None, player=0, forced=False, declared=False, discards=None):
    hands = [[], [], [], []]
    hands[player] = hand
    riichi = [False] * 4
    riichi[player] = declared
    return Observation(
        player_id=player,
        hands=hands,
        melds=[[], [], [], []],
        discards=discards or [[], [], [], []],
        dora_indicators=[],
        scores=[25_000] * 4,
        riichi_declared=riichi,
        legal_actions=actions,
        events=[],
        honba=0,
        riichi_sticks=0,
        round_wind=0,
        oya=0,
        kyoku_index=0,
        waits=[],
        is_tenpai=False,
        riichi_sutehais=[None] * 4,
        last_tedashis=[None] * 4,
        last_discard=None,
        drawn_tile=drawn,
        forced_tedashi=forced,
    )


def _ready_observation(drawn, *, forced=False, discards=None):
    actions = [Action(ActionType.DISCARD, tile, [], 0) for tile in _HAND]
    actions.append(Action(ActionType.RIICHI, None, [], 0))
    return _observation(_HAND, actions, drawn=drawn, forced=forced, discards=discards)


def _env_with_hand(drawn=53):
    own = [tile for tile in _HAND if tile != drawn]
    pool = [tile for tile in range(136) if tile not in _HAND]
    hands = [own, pool[:13], pool[13:26], pool[26:39]]
    deal = [tile for start in (0, 4, 8) for hand in hands for tile in hand[start : start + 4]]
    deal.extend(hand[12] for hand in hands)
    wall = deal + [drawn] + pool[39:]
    assert sorted(wall) == list(range(136))
    env = RiichiEnv("4p-red-single", seed=42, rule=GameRule.default_mjsoul())
    return env, env.reset(wall=wall, oya=0)[0]


def test_manifest_is_complete_immutable_and_reversible():
    specs = manifest()
    assert len(specs) == ACTION_COUNT == 546
    assert [spec.global_id for spec in specs] == list(range(546))
    assert len({tuple(vars(spec).values()) for spec in specs}) == 546
    for global_id, spec in enumerate(specs):
        assert action_spec(global_id) is spec
        assert action_id(spec) == global_id
    with pytest.raises(FrozenInstanceError):
        specs[0].global_id = 1
    with pytest.raises(ValueError, match="disagrees"):
        action_id(replace(specs[0], global_id=1))
    assert len(ACTION_MANIFEST_SHA256) == 64
    assert manifest_hash() == ACTION_MANIFEST_SHA256


@pytest.mark.parametrize(
    "global_id,family",
    [
        (0, "DISCARD"),
        (147, "DISCARD"),
        (148, "ANKAN"),
        (181, "ANKAN"),
        (182, "KAKAN"),
        (218, "KAKAN"),
        (219, "TSUMO"),
        (220, "KYUSHU_KYUHAI"),
        (221, "PASS"),
        (222, "CHI"),
        (311, "CHI"),
        (312, "PON"),
        (431, "PON"),
        (432, "DAIMINKAN"),
        (542, "DAIMINKAN"),
        (543, "RON"),
        (545, "RON"),
    ],
)
def test_frozen_action_family_boundaries(global_id, family):
    assert action_spec(global_id).action_type == family


@pytest.mark.parametrize("tile37", range(37))
def test_tile_conversions_preserve_red_identity_and_base34(tile37):
    tile136 = _tid(tile37)
    assert tile136_to_tile37(tile136) == tile37
    assert tile37_to_base34(tile37) == tile136 // 4
    if tile37 in (0, 10, 20):
        assert tile136_to_tile37(tile136 + 1) == tile37 + 5


@pytest.mark.parametrize(
    "function,bad",
    [
        (tile136_to_tile37, -1),
        (tile136_to_tile37, 136),
        (tile136_to_tile37, True),
        (tile37_to_base34, -1),
        (tile37_to_base34, 37),
        (action_spec, 546),
    ],
)
def test_out_of_range_protocol_values_are_rejected(function, bad):
    with pytest.raises(ValueError):
        function(bad)


@pytest.mark.parametrize("drawn,required", [(52, {42, 43, 60, 61}), (53, {62, 63, 40, 41})])
def test_red_normal_five_tsumogiri_and_riichi_are_distinct(drawn, required):
    candidates = legal_candidates(_ready_observation(drawn))
    by_id = {candidate.global_id: candidate for candidate in candidates}
    assert required <= by_id.keys()
    for global_id in required:
        candidate = by_id[global_id]
        assert candidate.spec.tile37 in (10, 15)
        assert candidate.actions[-1].tile == (
            drawn
            if candidate.spec.tsumogiri
            else min(tile for tile in _HAND if tile136_to_tile37(tile) == candidate.spec.tile37 and tile != drawn)
        )
        assert len(candidate.actions) == (2 if candidate.spec.riichi else 1)


def test_legality_does_not_remove_structural_tenpai_with_exhausted_visible_waits():
    hand = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 108, 132]
    remaining = [tile for tile in hand if tile != 132]
    evaluator = HandEvaluator(remaining)
    assert evaluator.is_tenpai()
    assert evaluator.get_waits() == [27]
    obs = _observation(
        hand,
        [Action(ActionType.DISCARD, tile, [], 0) for tile in hand] + [Action(ActionType.RIICHI, None, [], 0)],
        drawn=132,
        discards=[[109, 110, 111], [], [], []],
    )
    assert 4 * 36 + 3 in {candidate.global_id for candidate in legal_candidates(obs)}


def test_mjsoul_forced_tedashi_never_becomes_tsumogiri():
    candidates = legal_candidates(_ready_observation(52, forced=True))
    assert {40, 41, 60, 61} <= {candidate.global_id for candidate in candidates}
    assert not any(candidate.spec.tsumogiri for candidate in candidates)


def test_post_riichi_only_maps_the_drawn_discard():
    obs = _observation(_HAND, [Action(ActionType.DISCARD, 53, [], 0)], drawn=53, declared=True)
    candidates = legal_candidates(obs)
    assert [candidate.global_id for candidate in candidates] == [62]
    assert not candidates[0].spec.riichi


def test_canonical_duplicate_draw_retains_both_ids_but_rejects_native_execution():
    hand = [53, 53]
    obs = _observation(hand, [Action(ActionType.DISCARD, 53, [], 0)], drawn=53)
    candidates = legal_candidates(obs)
    assert [candidate.global_id for candidate in candidates] == [60, 62]
    assert all(candidate.native_execution_ambiguous for candidate in candidates)
    env, _ = _env_with_hand()
    before = list(env.mjai_log)
    for candidate in candidates:
        with pytest.raises(AmbiguousDiscardError):
            execute_candidate(env, 0, candidate)
    assert env.mjai_log == before


def test_canonical_duplicate_draw_retains_riichi_variants_when_structurally_tenpai():
    hand = [53 if tile == 54 else tile for tile in _HAND]
    obs = _observation(
        hand,
        [Action(ActionType.DISCARD, tile, [], 0) for tile in hand] + [Action(ActionType.RIICHI, None, [], 0)],
        drawn=53,
    )
    by_id = {candidate.global_id: candidate for candidate in legal_candidates(obs)}
    assert {60, 61, 62, 63} <= by_id.keys()
    assert all(by_id[global_id].native_execution_ambiguous for global_id in (60, 61, 62, 63))


def test_canonical_forced_discard_has_no_execution_ambiguity():
    hand = [53, 53]
    actions = [Action(ActionType.DISCARD, 53, [], 0)]
    forced = legal_candidates(_observation(hand, actions, drawn=53, forced=True))
    declared = legal_candidates(_observation(hand, actions, drawn=53, declared=True))
    assert [candidate.global_id for candidate in forced] == [60]
    assert [candidate.global_id for candidate in declared] == [62]
    assert not forced[0].native_execution_ambiguous
    assert not declared[0].native_execution_ambiguous


def test_unambiguous_native_representative_is_preferred_for_the_same_semantic_id():
    hand = [53, 53, 54]
    actions = [Action(ActionType.DISCARD, tile, [], 0) for tile in hand]
    by_id = {candidate.global_id: candidate for candidate in legal_candidates(_observation(hand, actions, drawn=53))}
    assert not by_id[60].native_execution_ambiguous
    assert by_id[60].actions[0].tile == 54
    assert by_id[62].native_execution_ambiguous


def test_non_draw_duplicates_are_semantically_deduplicated():
    obs = _observation([0, 0, 4], [Action(ActionType.DISCARD, tile, [], 0) for tile in (0, 0, 4)], drawn=4)
    assert [candidate.global_id for candidate in legal_candidates(obs)] == [4, 10]


def test_native_candidate_order_does_not_change_global_ids_or_representatives():
    obs = _ready_observation(53)
    reverse = _observation(_HAND, list(reversed(obs.legal_actions())), drawn=53)
    first = legal_candidates(obs)
    second = legal_candidates(reverse)
    assert [(candidate.global_id, [a.to_dict() for a in candidate.actions]) for candidate in first] == [
        (candidate.global_id, [a.to_dict() for a in candidate.actions]) for candidate in second
    ]


@pytest.mark.parametrize("global_id", range(148))
def test_every_discard_manifest_id_is_reachable_from_structural_tenpai(global_id):
    spec = action_spec(global_id)
    ready = [0, 4, 8, 48, 53, 56, 96, 100, 104, 108, 109, 110, 112]
    tile = _tid(spec.tile37)
    drawn = tile if spec.tsumogiri else next(t for t in ready if tile136_to_tile37(t) != spec.tile37)
    obs = _observation(
        ready + [tile],
        [Action(ActionType.DISCARD, tile, [], 0), Action(ActionType.RIICHI, None, [], 0)],
        drawn=drawn,
    )
    candidates = {candidate.global_id: candidate for candidate in legal_candidates(obs)}
    assert candidates[global_id].spec == spec


@pytest.mark.parametrize("global_id", range(148, 546))
def test_every_non_discard_manifest_entry_maps_from_native_action(global_id):
    spec = action_spec(global_id)
    kind = getattr(ActionType, spec.action_type)
    if spec.action_type == "ANKAN":
        tile = spec.base34 * 4
        consumed = [tile + i for i in range(4)]
    elif spec.action_type == "KAKAN":
        tile = _tid(spec.tile37)
        consumed = [spec.base34 * 4 + i for i in range(4) if spec.base34 * 4 + i != tile]
    elif spec.action_type == "DAIMINKAN":
        tile = _tid(spec.tile37)
        consumed = [spec.base34 * 4 + i for i in range(4) if spec.base34 * 4 + i != tile]
    elif spec.action_type in ("CHI", "PON"):
        tile = _tid(spec.tile37)
        consumed = [_tid(value) for value in spec.consumed_tile37]
    else:
        tile, consumed = None, []
    player = 2
    obs = _observation(consumed, [Action(kind, tile, consumed, player)], player=player)
    source = None if spec.source is None else (player + spec.source) % 4
    candidates = legal_candidates(obs, source_player=source)
    assert len(candidates) == 1
    assert candidates[0].global_id == global_id
    assert candidates[0].spec == spec


@pytest.mark.parametrize(
    "global_id,claimed,consumed",
    [
        (228, 3, (0, 4)),
        (235, 0, (3, 4)),
        (243, 6, (0, 7)),
        (316, 5, (5, 5)),
        (317, 5, (0, 5)),
        (318, 0, (5, 5)),
        (328, 15, (10, 15)),
        (340, 20, (25, 25)),
    ],
)
def test_reference_red_tile_chi_and_pon_offsets(global_id, claimed, consumed):
    spec = action_spec(global_id)
    assert spec.tile37 == claimed
    assert spec.consumed_tile37 == consumed


def test_claims_require_public_source_and_chi_only_accepts_previous_player():
    action = Action(ActionType.CHI, 0, [4, 8], 0)
    obs = _observation([4, 8], [action])
    with pytest.raises(ValueError, match="source_player"):
        legal_candidates(obs)
    with pytest.raises(ValueError, match="previous"):
        legal_candidates(obs, source_player=1)
    assert legal_candidates(obs, source_player=3)[0].global_id == 222


def test_oversized_candidates_are_rejected_without_truncation():
    hand = [_tid(tile37) for tile37 in range(37)]
    obs = SimpleNamespace(
        player_id=0,
        hands=[hand, [], [], []],
        hand=hand,
        melds=[[], [], [], []],
        drawn_tile=None,
        forced_tedashi=True,
        riichi_declared=[False] * 4,
        legal_actions=lambda: [Action(ActionType.DISCARD, tile, [], 0) for tile in hand],
    )
    with pytest.raises(ValueError, match=f"exceeds {MAX_CANDIDATES}"):
        legal_candidates(obs)


def test_native_mjsoul_dealer_snapshot_and_discard_execution_roundtrip():
    env, obs = _env_with_hand()
    assert obs.forced_tedashi
    candidates = legal_candidates(obs)
    assert all(not candidate.spec.tsumogiri for candidate in candidates if candidate.spec.action_type == "DISCARD")
    selected = next(candidate for candidate in candidates if candidate.global_id == 60)
    execute_candidate(env, 0, selected)
    discard = next(event for event in reversed(env.mjai_log) if event["type"] == "dahai")
    assert discard == {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": False}
    assert not any("Illegal Action" in event.get("reason", "") for event in env.mjai_log)


def test_native_riichi_executes_reach_and_discard_without_losing_event_delta():
    env, obs = _env_with_hand()
    selected = next(candidate for candidate in legal_candidates(obs) if candidate.global_id == 61)
    assert [action.action_type for action in selected.actions] == [ActionType.RIICHI, ActionType.DISCARD]
    before = len(env.mjai_log)
    result = execute_candidate(env, 0, selected)
    assert len(result.step_observations) == 2
    assert [event["type"] for event in result.step_observations[0][0].events] == ["reach"]
    assert dict(result) == result.step_observations[-1]
    emitted = env.mjai_log[before:]
    assert [event["type"] for event in emitted[:2]] == ["reach", "dahai"]
    assert emitted[1]["pai"] == "5p"
    assert emitted[1]["tsumogiri"] is False
    assert not any("Illegal Action" in event.get("reason", "") for event in emitted)


def test_execute_rejects_stale_action_before_mutating_engine():
    env, obs = _env_with_hand()
    candidates = legal_candidates(obs)
    execute_candidate(env, 0, next(candidate for candidate in candidates if candidate.global_id == 60))
    before = list(env.mjai_log)
    with pytest.raises(ValueError, match="stale"):
        execute_candidate(env, 0, candidates[0])
    assert env.mjai_log == before


def test_drawn_context_must_belong_to_the_observer_for_discard_decisions():
    with pytest.raises(ValueError, match="not in the observer"):
        legal_candidates(_ready_observation(53), drawn_tile=0)


def test_execute_rejects_modified_native_payload_before_mutating_engine():
    env, obs = _env_with_hand()
    mapped = next(candidate for candidate in legal_candidates(obs) if candidate.global_id == 60)
    mapped.actions[0].tile = 52
    before = list(env.mjai_log)
    with pytest.raises(ValueError, match="modified"):
        execute_candidate(env, 0, mapped)
    assert env.mjai_log == before


def test_native_pon_source_and_followup_discard_execute_roundtrip():
    pool = [tile for tile in range(136) if tile not in (0, 1, 2, 3)]
    hands = [[0] + pool[:12], [1, 2] + pool[12:23], pool[23:36], pool[36:49]]
    deal = [tile for start in (0, 4, 8) for hand in hands for tile in hand[start : start + 4]]
    deal.extend(hand[12] for hand in hands)
    wall = deal + [3] + pool[49:]
    assert sorted(wall) == list(range(136))
    env = RiichiEnv("4p-red-single", seed=42, rule=GameRule.default_mjsoul())
    obs = env.reset(wall=wall, oya=0)[0]
    discard = next(candidate for candidate in legal_candidates(obs) if candidate.global_id == 4)
    responses = execute_candidate(env, 0, discard)
    pon = next(candidate for candidate in legal_candidates(responses[1], source_player=0) if candidate.global_id == 392)
    assert pon.spec.source == 3
    observations = execute_candidate(env, 1, pon)
    assert sorted(env.melds[1][0].tiles) == [0, 1, 2]
    followup = legal_candidates(observations[1])
    assert all(not candidate.spec.tsumogiri for candidate in followup)
    execute_candidate(env, 1, followup[0])
    assert env.discards[1]
    assert not any("Illegal Action" in event.get("reason", "") for event in env.mjai_log)


def test_seeded_game_maps_and_executes_native_legal_actions_to_completion():
    env = RiichiEnv("4p-red-single", seed=20261115, rule=GameRule.default_mjsoul())
    observations = env.reset()
    decisions = 0
    while not env.done() and decisions < 1000:
        actions = {}
        for player, obs in observations.items():
            source = env.last_discard[0] if env.last_discard is not None else None
            candidates = legal_candidates(obs, source_player=source)
            assert candidates
            selected = next((candidate for candidate in candidates if candidate.global_id == 221), candidates[0])
            assert len(selected.actions) == 1
            actions[player] = selected.actions[0]
        observations = env.step(actions)
        decisions += 1
    assert env.done()
    assert decisions < 1000
    assert not any("Illegal Action" in event.get("reason", "") for event in env.mjai_log)
