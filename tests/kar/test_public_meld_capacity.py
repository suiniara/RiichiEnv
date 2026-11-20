"""Public response capacity on short, physically valid native trajectories."""

import pytest

from riichienv import ActionType, convert
from riichienv.kar.actions import legal_candidates
from riichienv.kar.response_mask import public_rule_response_mask
from tests.kar.test_decision_time_parity import _Episode


def _pass_to_dealer(episode):
    for seat in (2, 3):
        episode.resolve()
        assert episode.env.current_player == seat
        episode.act(tile=convert.tid_to_mjai(episode.env.drawn_tile))
    episode.resolve()
    assert episode.env.current_player == 0


def _three_meld_episode(with_ankan):
    if with_ankan:
        episode = _Episode(
            {
                0: ("2m", "3m", "4m", "E"),
                1: ("1m",) * 4 + ("2m", "2m", "3m", "3m", "4m", "4m", "7s", "8s", "E"),
            },
            ("2p", "5s", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "6p"),
            ("6s",),
        )
        episode.act(tile="2p")
        episode.act("ANKAN")
        episode.act(tile="5s")
    else:
        episode = _Episode(
            {
                0: ("1m", "2m", "3m", "4m", "E"),
                1: ("1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5s", "6s", "7s", "8s", "E"),
            },
            ("1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s"),
        )
        episode.act(tile="1m")
        episode.resolve("PON", seat=1)
        episode.act(tile="5s")
    for called, discarded in (("2m", "6s"), ("3m", "7s")):
        _pass_to_dealer(episode)
        episode.act(tile=called)
        episode.resolve("PON", seat=1)
        episode.act(tile=discarded)
    _pass_to_dealer(episode)
    return episode


def _discard_mask(episode, tile):
    seat = episode.env.current_player
    snapshot = episode.live.snapshot(seat)
    candidate = next(
        candidate
        for candidate in legal_candidates(episode.env.get_observation(seat), drawn_tile=episode.env.drawn_tile)
        if candidate.spec.action_type == "DISCARD"
        and not candidate.spec.riichi
        and convert.tid_to_mjai(candidate.actions[0].tile) == tile
    )
    return snapshot, public_rule_response_mask(snapshot, candidate.global_id)


@pytest.mark.parametrize("with_ankan", [False, True], ids=["four-open-melds", "includes-ankan"])
def test_public_capacity_tracks_three_and_four_melds_without_masking_tanki_ron(with_ankan):
    episode = _three_meld_episode(with_ankan)
    snapshot, mask = _discard_mask(episode, "4m")
    assert len(snapshot.melds[1]) == 3
    assert any(meld.kind == "ankan" for meld in snapshot.melds[1]) == with_ankan
    assert mask[1] and mask[2] and mask[5]

    # A fourth Pon is actually available and executes from this three-meld state.
    episode.act(tile="4m")
    episode.resolve("PON", seat=1)
    episode.act(tile="8s")

    # Three consecutive real discard decisions place the same saturated player
    # at kamicha, toimen, and shimocha respectively. No snapshots are fabricated.
    for seat in (2, 3, 0):
        episode.resolve()
        assert episode.env.current_player == seat
        tile = convert.tid_to_mjai(episode.env.drawn_tile)
        snapshot, mask = _discard_mask(episode, tile)
        relative = (1 - seat) % 4
        assert len(snapshot.melds[1]) == 4
        assert not mask[1 + relative]
        assert not mask[4 + relative]
        if relative == 1:
            assert not mask[1]
        assert mask[7 + (1 << (relative - 1))]
        if seat != 0:
            episode.act(tile=tile)

    # Four sets leave a legal pair wait: suppressing calls must not hide Ron.
    _, mask = _discard_mask(episode, "E")
    assert mask[8]
    episode.act(tile="E")
    assert any(action.action_type == ActionType.RON for action in episode.obs[1].legal_actions())
    episode.resolve("RON", seat=1)
    assert any(event["type"] == "hora" and event["actor"] == 1 for event in episode.live.events)
