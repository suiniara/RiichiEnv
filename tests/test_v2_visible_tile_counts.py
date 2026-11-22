"""Python binding contract for KAR V2 physical visible tile counts (aligned with commit 5de9ff3)."""

import pytest

from riichienv import Meld, MeldType, Observation


def make_observation(*, hands=None, discards=None, dora_indicators=None):
    return Observation(
        player_id=0,
        hands=hands if hands is not None else [[0, 1, 16], [], [], []],
        melds=[[], [], [], []],
        discards=discards if discards is not None else [[], [], [], []],
        dora_indicators=dora_indicators if dora_indicators is not None else [],
        scores=[25_000] * 4,
        riichi_declared=[False] * 4,
        legal_actions=[],
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
    )


def decode_visible_counts(raw):
    assert len(raw) == 241
    data = memoryview(raw)
    seen = data[0:136]
    visible = data[136:170]
    own = data[170:204]
    unseen = data[204:238]
    red_unseen = data[238:241]
    return seen, visible, own, unseen, red_unseen


def test_visible_tile_counts_python_binary_layout_and_conservation():
    obs = make_observation()
    seen, visible, own, unseen, red_unseen = decode_visible_counts(obs.encode_visible_tile_counts())

    assert seen[0] == 1 and seen[1] == 1 and seen[16] == 1
    assert own[0] == 2 and own[4] == 1
    assert visible[0] == 0 and visible[4] == 0
    assert unseen[0] == 2 and unseen[4] == 3
    assert list(red_unseen) == [0, 1, 1]

    for tile_kind in range(34):
        assert own[tile_kind] + visible[tile_kind] + unseen[tile_kind] == 4


def test_visible_tile_counts_deduplicates_called_discard_in_meld():
    # Physical tile 4 is present in both the source river and the called meld.
    meld = Meld(MeldType.Chi, [4, 8, 12], True, 1, 4)
    obs = Observation(
        player_id=0,
        hands=[[0, 1, 16], [], [], []],
        melds=[[meld], [], [], []],
        discards=[[], [4], [], []],
        dora_indicators=[],
        scores=[25_000] * 4,
        riichi_declared=[False] * 4,
        legal_actions=[],
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
        last_discard=4,
    )
    _, visible, _, unseen, _ = decode_visible_counts(obs.encode_visible_tile_counts())
    assert visible[1] == 1
    assert visible[2] == 1
    assert visible[3] == 1
    assert unseen[1] == 3


def test_visible_tile_counts_rejects_duplicate_physical_discard():
    obs = make_observation(discards=[[20], [20], [], []])
    with pytest.raises(ValueError, match="duplicate physical tile"):
        obs.encode_visible_tile_counts()
