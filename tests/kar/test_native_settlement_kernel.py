"""Physical-wall regressions for the shared online/audit settlement rules."""

from copy import deepcopy

from riichienv import GameRule, HandEvaluator, convert
from riichienv._riichienv import native_round_transition
from riichienv.kar.completion import _end_required
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.settlement import _meld, _Verifier, verify_settlements
from tests.kar.test_decision_time_parity import _Episode, _wall


def _finish_by_tsumogiri(episode):
    for _ in range(75):
        episode.resolve()
        if any(event["type"] == "end_kyoku" for event in episode.live.events):
            end = next(i for i, event in enumerate(episode.live.events) if event["type"] == "end_kyoku")
            return episode.live.events[: end + 1]
        view = episode.live.snapshot(episode.env.current_player)
        assert view.drawn_tile is not None
        episode.act(tile=convert.tid_to_mjai(view.drawn_tile))
    raise AssertionError("The physical round did not close within its live wall")


def test_four_owned_copies_do_not_create_a_fifth_copy_wait_or_noten_payment():
    hand = ("1m", "1m", "1m", "2p", "3p", "4p", "5p", "6p", "7p", "7s", "8s", "9s", "E")
    # _wall checks that every physical tile 0..135 occurs exactly once.
    episode = _Episode({1: hand}, ("1m",))
    episode.act(tile="1m")
    episode.resolve("PON", seat=1)
    episode.act(tile="E")
    view = episode.live.snapshot(1)
    calc = HandEvaluator(list(view.hand), [_meld(meld) for meld in view.melds[1]])
    assert (
        sum(tile // 4 == 0 for tile in view.hand) + sum(tile // 4 == 0 for meld in view.melds[1] for tile in meld.tiles)
        == 4
    )
    assert not calc.is_tenpai(rule=MJSOUL_RULESET.engine_rule())
    assert calc.get_waits(rule=MJSOUL_RULESET.engine_rule()) == []
    # Tenhou explicitly counts this form as ready; legacy unqualified hand
    # queries retain that original behavior instead of imposing KAR's rules.
    assert calc.is_tenpai(rule=GameRule.default_tenhou())
    assert calc.get_waits(rule=GameRule.default_tenhou()) == [0]
    assert calc.is_tenpai() and calc.get_waits() == [0]
    events = _finish_by_tsumogiri(episode)
    terminal = next(event for event in events if event["type"] == "ryukyoku")
    assert terminal["deltas"] == [1500, -1500, 1500, -1500]
    assert terminal["public_hands"][1] is None
    report = verify_settlements(events)
    assert report["status"] == "PASS", report
    assert report["settlements"][0]["tenpai"] == [True, False, True, False]
    # The historical wrong payment must now fail the independent source comparison.
    wrong = deepcopy(events)
    row = next(event for event in wrong if event["type"] == "ryukyoku")
    row["deltas"] = [1000, 1000, 1000, -3000]
    row["scores"] = [26000, 26000, 26000, 22000]
    assert verify_settlements(wrong)["status"] == "FAIL"


def test_west_draw_threshold_ends_online_and_in_audit_even_with_dealer_ready():
    hands = {
        0: ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "E"),
        1: ("1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "4p", "5p", "6p", "N"),
        2: ("1m", "1m", "2m", "2m", "4m", "4m", "5p", "5p", "7s", "E", "W", "W", "C"),
        3: ("9m", "9m", "3p", "3p", "5s", "5s", "6s", "8p", "8p", "S", "F", "F", "P"),
    }
    initial = [25000, 29500, 22500, 23000]
    assert sum(initial) == 100000 and max(initial) < 30000
    wall = _wall(hands, ("2s",))
    assert sorted(wall) == list(range(136))
    episode = _Episode(hands, ("2s",))
    episode.obs = episode.env.reset(wall=wall, oya=0, round_wind=2, honba=0, kyotaku=0, scores=initial)
    episode.live = LiveEventProjection()
    episode.live.observe_native(list(episode.env.rule_log))
    events = _finish_by_tsumogiri(episode)
    assert episode.env.done()
    assert list(episode.env.scores()) == [26500, 31000, 21000, 21500]
    assert episode.live.events[-1]["type"] == "end_game"
    # This is a valid West-1 boundary, not a claimed East-1-to-West full match.
    audit = _Verifier(MJSOUL_RULESET)
    audit.points, audit.pot, audit.expected_round = tuple(initial), 0, (2, 0, 0)
    for index, event in enumerate(events):
        if event["type"] == "ryukyoku":
            audit.draw(event, index)
        else:
            audit.observe(event, index)
    assert audit.rows[0]["status"] == "PASS", audit.rows
    assert audit.rule_requires_match_end is True
    start = next(event for event in events if event["type"] == "start_kyoku")
    terminal = next(event for event in events if event["type"] == "ryukyoku")
    assert _end_required(start, audit.rows[0], [terminal], tuple(episode.env.scores()))


def test_public_round_context_retains_dealer_multi_ron_and_abortive_exceptions():
    scores = [26500, 31000, 21000, 21500]
    mjsoul, tenhou = MJSOUL_RULESET.engine_rule(), GameRule.default_tenhou()
    # The KAR-specific adapter defaults to MjSoul, while explicit profiles
    # keep their distinct extension-round dealer priority.
    assert native_round_transition(scores, 0, 2, 0, 2, "exhaustive_draw", True)[4]
    assert native_round_transition(scores, 0, 2, 0, 2, "exhaustive_draw", True, rule=mjsoul)[4]
    assert not native_round_transition(scores, 0, 2, 0, 2, "exhaustive_draw", True, rule=tenhou)[4]
    for rule in (mjsoul, tenhou):
        assert not native_round_transition(scores, 0, 2, 0, 2, "win", True, 2, rule=rule)[4]
        assert not native_round_transition(scores, 0, 2, 0, 2, "abortive_draw", True, rule=rule)[4]


def test_new_rule_flags_keep_legacy_defaults_and_are_explicitly_configurable():
    for rule in (GameRule(), GameRule.default_tenhou()):
        assert rule.extension_dealer_repeat_priority is True
        assert rule.allow_fifth_copy_tenpai is True
    mjsoul = GameRule.default_mjsoul()
    assert mjsoul.extension_dealer_repeat_priority is False
    assert mjsoul.allow_fifth_copy_tenpai is False
    custom = GameRule(extension_dealer_repeat_priority=False, allow_fifth_copy_tenpai=False)
    assert custom.extension_dealer_repeat_priority is False
    assert custom.allow_fifth_copy_tenpai is False
