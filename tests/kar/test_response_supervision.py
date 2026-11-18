"""Full native half-games, not synthetic response labels or hand-filled cutoffs."""
import json
import sqlite3
from contextlib import closing

import pytest

from riichienv import Phase
from riichienv.kar.actions import action_spec, execute_joint_responses
from riichienv.kar.audit import replay_game
from riichienv.kar.learning.export import SQL_SCHEMA, GameWriter
from tests.kar.test_decision_time_parity import _comparison, _Episode
from tests.kar.test_native_furiten_export import _complete_game


@pytest.fixture(scope="module")
def completed():
    def play(ep, desired, seat, *, after):
        assert ep.env.phase == Phase.WaitResponse
        trigger = ep.live.ledgers[seat].pass_trigger()
        choices = {}
        selection = None
        for player in sorted(ep.env.active_players):
            choice, capture = ep._choose(player, desired.get(player, "PASS"), capture=player == seat)
            choices[player] = choice
            if capture is not None:
                selection = capture
        assert selection is not None
        frozen_bytes = selection.prefix.to_bytes()
        receipts = ep.live.capture_self_passes(choices)
        raw, public = len(ep.env.mjai_log), len(ep.live.events)
        ep.obs = execute_joint_responses(ep.env, choices)
        ep.live.observe_native(list(ep.env.mjai_log)[raw:], confirmed_self_passes=receipts)
        assert ep.live.events[public]["type"] == after
        family = desired.get(seat, "PASS")
        # Only a selected action that actually became public has a result ID.
        selection.record_index = (ep._find_result(public, seat, family.lower()) if family.lower() == after
                                  else trigger.trigger_event_id)
        assert selection.prefix.to_bytes() == frozen_bytes and not _marker(selection.snapshot, trigger.trigger_event_id)
        for _ in range(5):
            ep.resolve()
            if ep.env.current_player == seat:
                later = ep.act(capture=True)
                break
            ep.act()
        else:
            pytest.fail("The next owner decision was not reached")
        game = _complete_game(ep)
        return game, trigger.trigger_event_id, selection, later

    return play


def _marker(snapshot, trigger):
    markers = [row for raw in snapshot.events if (row := json.loads(raw))["type"] == "self_pass"
               and row["trigger_event_id"] == trigger]
    assert len(markers) <= 1
    return markers[0] if markers else None


def _export(game):
    rows = []
    with closing(sqlite3.connect(":memory:")) as db:
        db.executescript(SQL_SCHEMA)
        writer = GameWriter(db, game)

        def add(row):
            rows.append(row)
            writer(row)

        replay = replay_game(game, on_decision=add)
        assert replay["status"] == replay["completion"]["status"] == "PASS", replay["errors"]
        assert replay["independent_settlement"]["status"] == "PASS"
        receipt = writer.finish(replay, "train")
        assert db.execute("SELECT count(*) FROM games").fetchone() == (1,)
        samples = db.execute(
            "SELECT observer,trigger_event_id,action_target,provenance,diagnostic FROM samples"
        ).fetchall()
    return rows, receipt, samples


def _closed_slice(receipt, phase, kind):
    report = receipt["response_opportunities"]
    assert report["total"] == sum(report[key] for key in ("explicit", "uniquely_inferred", "censored"))
    return report["by_phase_and_close"][phase][kind]


def _check_pass(game, trigger, live, later, exported, *, kind, provenance, record_property):
    rows, receipt, samples = exported
    seat = live.snapshot.observer_player_id
    record_property("complete_rows", receipt["examples"])
    observed = [sample for sample in samples if sample[:3] == (seat, trigger, 221)]
    record_property("pass_rows_at_trigger", len(observed))
    record_property("live_has_self_pass", bool(_marker(later.snapshot, trigger)))
    offline_later = next(row for row in rows if row.snapshot.observer_player_id == seat
                         and game.events[row.trigger_event_id].get("source_record_index") == later.record_index)
    record_property("offline_has_self_pass", bool(_marker(offline_later.snapshot, trigger)))
    assert len(observed) == 1 and observed[0][3] == provenance
    diagnostic = json.loads(observed[0][4])
    assert diagnostic["response_window_trigger_event_id"] == trigger
    closing_event_id = diagnostic["response_label_evidence_event_id"]
    assert game.events[closing_event_id]["type"] == ("tsumo" if kind == "replacement_draw" else kind)
    assert _closed_slice(receipt, live.snapshot.decision_phase, kind)["uniquely_inferred"] >= 1
    prefix = _comparison((game.events, rows), live)
    assert not _marker(prefix.snapshot, trigger)
    compared = _comparison((game.events, rows), later)
    offline_marker, native_marker = _marker(compared.snapshot, trigger), _marker(later.snapshot, trigger)
    assert offline_marker == {**native_marker, "provenance": provenance}
    assert offline_marker["confirmed_at_event_id"] == closing_event_id
    assert (compared.prefix.to_record() == later.prefix.to_record()
            and compared.snapshot.history_positions == later.snapshot.history_positions)
    return prefix


def test_completed_robbery_pass_exports_and_matches_live_after_replacement(completed, record_property):
    hand = ("1m", "2m", "4p", "5p", "6p", "7p", "8p", "9p", "4s", "5s", "6s", "W", "W")
    ep = _Episode({1: ("3m", "3m", "S", "S", "S"), 2: hand}, ("3m", "8s", "9s", "S"), ("3m", "7s"))
    ep.act(tile="3m")
    ep.resolve("PON", seat=1)
    ep.act()
    for tile in ("8s", "9s", "S"):
        ep.act(tile=tile)
    ep.resolve("DAIMINKAN", seat=1)
    ep.act("KAKAN")
    assert set(ep.env.active_players) == {2}
    game, trigger, live, later = completed(ep, {2: "PASS"}, 2, after="tsumo")
    assert live.prefix.candidates == (221, 545)
    # The prior kan's public indicator can put this cutoff after its trigger.
    assert live.snapshot.cutoff_event_id > trigger
    pass_row = _check_pass(game, trigger, live, later, _export(game), kind="replacement_draw",
                           provenance="inferred_unique_kan_replacement_pass", record_property=record_property)
    assert pass_row.snapshot.cutoff_event_id > pass_row.trigger_event_id


def _competing_calls():
    ep = _Episode({1: ("2m", "4m"), 2: ("3m", "3m")}, ("3m",))
    ep.act(tile="3m")
    return ep


def test_completed_same_pon_pass_offer_does_not_disappear_when_lower_chi_wins(completed, record_property):
    pon_game, _, pon, _ = completed(_competing_calls(), {1: "CHI", 2: "PON"}, 2, after="pon")
    pass_game, trigger, passed, later = completed(_competing_calls(), {1: "CHI", 2: "PASS"}, 2, after="chi")
    assert {action_spec(action).action_type for action in pon.prefix.candidates} == {"PASS", "PON"}
    assert pon.prefix.to_bytes() == passed.prefix.to_bytes()
    pon_rows, pon_receipt, _ = _export(pon_game)
    explicit = _comparison((pon_game.events, pon_rows), pon)
    assert explicit.action_provenance == "explicit"
    assert _closed_slice(pon_receipt, "discard_response", "pon")["explicit"] >= 1
    _check_pass(pass_game, trigger, passed, later, _export(pass_game), kind="chi",
                provenance="inferred_unique_call_priority_pass", record_property=record_property)


def test_completed_higher_pon_censors_chi_even_when_actual_pass_is_private(completed):
    a = completed(_competing_calls(), {1: "CHI", 2: "PON"}, 1, after="pon")
    b = completed(_competing_calls(), {1: "PASS", 2: "PON"}, 1, after="pon")
    game, trigger, choice, later_call = a
    assert game.events == b[0].events and choice.prefix.to_bytes() == b[2].prefix.to_bytes()
    assert not _marker(later_call.snapshot, trigger) and _marker(b[3].snapshot, trigger)
    rows, receipt, samples = _export(game)
    offline_later = _comparison((game.events, rows), later_call)
    assert not any(row[:3] == (1, trigger, 221) for row in samples)
    assert not _marker(offline_later.snapshot, trigger)
    assert offline_later.prefix.to_bytes() != b[3].prefix.to_bytes()
    assert _closed_slice(receipt, "discard_response", "pon")["censored"] >= 1


@pytest.mark.parametrize("ids,result,proven", [
    ((221, 222), "pon", False), ((221, 312), "pon", False),
    ((221, 312), "chi", True), ((221, 312, 222), "chi", False),
    ((221, 543), "pon", True), ((221, 543), "replacement_draw", True),
    ((221, 543), "all_ron_results", True), ((221, 543, 312), "all_ron_results", False),
    ((221, 543), "no_ron_draw/exhaustive_draw", True), ((221, 543), "ron_results", False),
    ((221, 543), "unresolved", False), ((221,), "next_draw", False),
])
def test_only_disproved_nonpass_alternatives_are_eliminated(ids, result, proven):
    from riichienv.kar.response_supervision import unique_pass_provenance  # noqa: PLC0415

    assert bool(unique_pass_provenance(ids, result)) is proven
