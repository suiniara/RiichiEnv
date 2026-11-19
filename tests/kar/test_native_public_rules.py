"""Public-mask isolation against different real native hidden responses.

These are bounded physical-wall/contract regressions, not training or Golden
evidence. Existing response-mask boundary tests supply terminal-rule coverage.
"""

from dataclasses import replace

import pytest

from riichienv import ActionType
from riichienv import _riichienv as native
from riichienv.kar.actions import execute_candidate, legal_candidates
from riichienv.kar.contracts import Response
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.response_mask import public_rule_response_mask, validate_response_target
from tests.kar.test_input_semantics import deal

PUBLIC_FIELDS = (
    "observer_player_id",
    "discard_history",
    "current_rivers",
    "melds",
    "riichi_accepted",
    "remaining_draws",
    "history_complete",
    "pending_kan_actor",
)


def _opening(*, can_ron):
    own = ("4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4s", "5s", "6s", "P")
    # 23m + 456p + 789p + EEE + NN has a real 1m/4m Ron with round-wind yaku.
    # Changing only the hidden 3m to 4m removes Ron on the same 1m discard.
    opponent = (
        "2m",
        "3m" if can_ron else "4m",
        "4p",
        "5p",
        "6p",
        "7p",
        "8p",
        "9p",
        "E",
        "E",
        "E",
        "N",
        "N",
    )
    env = deal({0: own, 1: opponent}, ("1m",))
    projection = LiveEventProjection()
    projection.observe_native(env.rule_log)
    snapshot = projection.snapshot(0)
    chosen = next(
        candidate
        for candidate in legal_candidates(env.get_observation(0), drawn_tile=env.drawn_tile)
        if candidate.global_id < 148 and candidate.spec.base34 == 0 and not candidate.spec.riichi
    )
    return env, snapshot, chosen


def test_public_mask_does_not_depend_on_actual_hidden_ron_or_future_events():
    left, left_snapshot, left_action = _opening(can_ron=True)
    right, right_snapshot, right_action = _opening(can_ron=False)
    assert left_action.global_id == right_action.global_id
    assert left_snapshot.hand == right_snapshot.hand
    assert left_snapshot.dora_indicators == right_snapshot.dora_indicators
    assert all(getattr(left_snapshot, key) == getattr(right_snapshot, key) for key in PUBLIC_FIELDS)
    before = left_snapshot.to_bytes()
    expected = public_rule_response_mask(left_snapshot, left_action.global_id)
    assert public_rule_response_mask(right_snapshot, right_action.global_id) == expected
    assert expected[Response.RON_MASK_1]

    actual = []
    for env, candidate in ((left, left_action), (right, right_action)):
        execute_candidate(env, 0, candidate)
        actual.append(any(action.action_type == ActionType.RON for action in env.get_observation(1).legal_actions()))
    assert actual == [True, False]
    # It is possible publicly in both states, even though only one hidden hand
    # really wins. The query must not be replaced by omniscient legal_actions.
    assert public_rule_response_mask(left_snapshot, left_action.global_id) == expected
    future = replace(
        left_snapshot,
        events=('{"type":"hora","actor":1,"target":0}',),
        hand=(135,) * 13,
        drawn_tile=135,
        scores=(1, 2, 3, 4),
    )
    assert public_rule_response_mask(future, left_action.global_id) == expected
    assert left_snapshot.to_bytes() == before
    validate_response_target(expected, Response.RON_MASK_1)


def test_native_category_order_matches_the_training_response_contract():
    assert tuple(native.KAR_PUBLIC_RESPONSE_NAMES) == tuple(response.name for response in Response)


def test_public_adapter_does_not_pass_a_snapshot_or_private_fields_to_native(monkeypatch):
    _, snapshot, action = _opening(can_ron=True)
    from riichienv.kar import response_mask as module  # noqa: PLC0415

    actual_query = module.kar_public_response_mask
    received = []

    class PublicOnly:
        def __getattr__(self, key):
            assert key in PUBLIC_FIELDS, f"Read outside public whitelist: {key}"
            return getattr(snapshot, key)

    def query(**facts):
        assert set(facts) == {
            "actor",
            "candidate_tile",
            "declares_riichi",
            "discard_history",
            "current_rivers",
            "meld_kinds",
            "riichi_accepted",
            "remaining_draws",
            "history_complete",
            "chi_allowed",
            "sanchaho_is_draw",
            "four_winds_abort",
            "four_riichi_abort",
            "four_kans_abort",
            "nagashi_mangan",
        }
        assert all(value is not snapshot for value in facts.values())
        received.append(facts)
        return actual_query(**facts)

    monkeypatch.setattr(module, "kar_public_response_mask", query)
    allowed = public_rule_response_mask(PublicOnly(), action.global_id)
    assert len(received) == 1 and allowed[Response.RON_MASK_1]
    # Even the native Python API has no parameter capable of receiving the
    # hidden state/future-event objects that this query must never inspect.
    with pytest.raises(TypeError):
        actual_query(**received[0], hidden_hands=object())
    with pytest.raises(TypeError):
        actual_query(**received[0], future_events=object())


@pytest.mark.parametrize("candidate", [True, False, -1, 148, 221, 1.0, "4", None])
def test_complete_action_contract_is_checked_before_snapshot_access(candidate):
    with pytest.raises(ValueError, match="complete discard action"):
        public_rule_response_mask(object(), candidate)


@pytest.mark.parametrize(("red", "ordinary"), [(0, 5), (10, 15), (20, 25)])
def test_red_tile_encoding_does_not_create_a_different_public_response_rule(red, ordinary):
    _, snapshot, _ = _opening(can_ron=False)
    for action_flags in range(4):
        assert public_rule_response_mask(snapshot, 4 * red + action_flags) == public_rule_response_mask(
            snapshot, 4 * ordinary + action_flags
        )
