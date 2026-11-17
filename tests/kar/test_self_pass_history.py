"""Bounded synthetic public streams and actual native Pass submissions; no stages."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from riichienv import convert
from riichienv.kar import agent as agent_module
from riichienv.kar.actions import execute_candidate, execute_joint_responses, legal_candidates
from riichienv.kar.audit import replay_game
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.tokenize import TokenHistoryCache, encode_prefix


def _scenario():
    pool = list(range(136))

    def take(name):
        tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == name)
        pool.remove(tile)
        return tile

    dora = take("C")
    draws = [take(name) for name in ("9s", "8s", "7s", "6s", "5s")]
    hands = [[take("E")], [take("E"), take("E")], [], []]
    for hand in hands:
        while len(hand) < 13:
            hand.append(pool.pop(0))
    start = {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
             "oya": 0, "scores": [25000] * 4, "dora_marker": "C",
             "tehais": [[convert.tid_to_mjai(tile) for tile in hand] for hand in hands]}
    events = [{"type": "start_game"}, start, {"type": "tsumo", "actor": 0, "pai": "9s"},
              {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
              {"type": "tsumo", "actor": 1, "pai": "8s"}]
    for actor, discard, next_actor, drawn in ((1, "8s", 2, "7s"), (2, "7s", 3, "6s"), (3, "6s", 0, "5s")):
        events.extend([{"type": "dahai", "actor": actor, "pai": discard, "tsumogiri": True},
                       {"type": "tsumo", "actor": next_actor, "pai": drawn}])
    wall = [tile for packet in range(3) for hand in hands for tile in hand[packet * 4:packet * 4 + 4]]
    wall.extend(hand[12] for hand in hands)
    wall.extend(draws)
    tail = [*pool, dora]
    index = tail.index(dora)
    tail[index], tail[-5] = tail[-5], tail[index]
    wall.extend(tail)
    assert sorted(wall) == list(range(136))
    return events, wall


def _views(events):
    views = [ObserverLedger(seat, (0, 1, 2, 3)) for seat in range(4)]
    for event in events:
        for view in views:
            view.apply_event(event)
    return views


def _ids(events, seat):
    env = MJSOUL_RULESET.create_env(seed=42)
    for event in events:
        env.apply_event(event)
    observation = env.get_observation(seat)
    source = (
        events[-1].get("actor") if events[-1]["type"] == "dahai"
        else env.last_discard[0] if env.last_discard else None
    )
    return tuple(candidate.global_id for candidate in legal_candidates(observation, source_player=source))


def _private_event():
    return {"type": "self_pass", "actor": 1, "target": 0, "pai": "E", "trigger_type": "dahai",
            "trigger_event_id": 3, "confirmed_at_event_id": 4, "confirmed": True,
            "provenance": "inferred_unique_next_draw_pass"}


def _markers(snapshot):
    return [json.loads(value) for value in snapshot.events if json.loads(value)["type"] == "self_pass"]


def test_ledger_accepts_confirmed_owner_pass_without_advancing_public_id_or_clock():
    events, _ = _scenario()
    views = _views(events[:5])
    before = [view.snapshot().to_bytes() for view in views]
    owner = views[1]
    public_id, clock = owner.event_id, owner.logical_clock
    owner.apply_event(_private_event())
    snapshot = owner.snapshot()
    assert owner.event_id == public_id == 4 and owner.logical_clock == clock
    assert len(_markers(snapshot)) == 1 and snapshot.history_positions[-1] == clock
    assert all(views[seat].snapshot().to_bytes() == before[seat] for seat in (0, 2, 3))
    sequence = encode_prefix(snapshot, _ids(events[:5], 1))
    rows = [row for row in sequence.rows if row["kind"] == 26]
    assert len(rows) == 1 and rows[0]["actor"] == 1 and rows[0]["source"] == 4
    assert rows[0]["tiles"][0] == 31


def test_private_pass_keeps_current_prefix_and_other_three_next_inputs_byte_identical():
    events, _ = _scenario()
    control, changed = _views(events[:4]), _views(events[:4])
    response_ids = _ids(events[:4], 1)
    old_snapshot = changed[1].snapshot()
    old_prefix = encode_prefix(old_snapshot, response_ids)
    old_bytes = old_prefix.to_bytes()
    for view in control + changed:
        view.apply_event(events[4])
    changed[1].apply_event(_private_event())
    assert old_prefix.to_bytes() == old_bytes
    assert encode_prefix(old_snapshot, response_ids).to_bytes() == old_bytes
    assert not any(row["kind"] == 26 for row in old_prefix.rows)
    next_owner = encode_prefix(changed[1].snapshot(), _ids(events[:5], 1))
    assert sum(row["kind"] == 26 for row in next_owner.rows) == 1
    for index in range(5, len(events)):
        for view in control + changed:
            view.apply_event(events[index])
        if events[index]["type"] == "tsumo" and events[index]["actor"] in {0, 2, 3}:
            seat = events[index]["actor"]
            ids = _ids(events[:index + 1], seat)
            assert (
                encode_prefix(control[seat].snapshot(), ids).to_bytes()
                == encode_prefix(changed[seat].snapshot(), ids).to_bytes()
            )


def test_repeated_confirmation_is_idempotent_and_incremental_history_matches_fresh():
    events, _ = _scenario()
    view = _views(events[:4])[1]
    cache = TokenHistoryCache()
    encode_prefix(view.snapshot(), _ids(events[:4], 1), history_cache=cache)
    view.apply_event(events[4])
    view.apply_event(_private_event())
    first = encode_prefix(view.snapshot(), _ids(events[:5], 1), history_cache=cache)
    view.apply_event(_private_event())
    assert len(_markers(view.snapshot())) == 1
    assert encode_prefix(view.snapshot(), _ids(events[:5], 1), history_cache=cache).to_bytes() == first.to_bytes()
    assert first.to_bytes() == encode_prefix(view.snapshot(), _ids(events[:5], 1)).to_bytes()


@pytest.mark.parametrize("field,value", [("actor", 2), ("target", 3), ("pai", "S"),
                                         ("trigger_event_id", 99), ("confirmed_at_event_id", 99),
                                         ("confirmed", False), ("provenance", "guessed_other_player_intent")])
def test_unconfirmed_foreign_or_wrong_source_pass_is_rejected_without_mutation(field, value):
    events, _ = _scenario()
    view = _views(events[:5])[1]
    before = view.snapshot().to_bytes()
    with pytest.raises(ValueError):
        view.apply_event({**_private_event(), field: value})
    assert view.snapshot().to_bytes() == before


def test_future_draw_cannot_authorize_a_marker_at_an_earlier_cutoff():
    events, _ = _scenario()
    view = _views(events[:4])[1]
    before = view.snapshot().to_bytes()
    with pytest.raises(ValueError):
        view.apply_event(_private_event())
    assert view.snapshot().to_bytes() == before


def test_private_pass_cannot_enter_the_public_projection_stream():
    events, _ = _scenario()
    projection = LiveEventProjection()
    projection.observe_native(events[:5])
    before = [projection.snapshot(seat).to_bytes() for seat in range(4)]
    public = list(projection.events)
    with pytest.raises(ValueError):
        projection.observe_native([_private_event()])
    assert projection.events == public
    assert all(projection.snapshot(seat).to_bytes() == before[seat] for seat in range(4))


def test_trusted_offline_label_is_not_added_to_current_prefix_but_enters_later_owner_history():
    events, _ = _scenario()
    decisions = []
    result = replay_game(SimpleNamespace(events=tuple(events)), on_decision=decisions.append)
    assert not result["errors"], result
    passes = [row for row in decisions if row.action_target_global == 221 and row.snapshot.observer_player_id == 1
              and row.trigger_event_id == 3]
    assert len(passes) == 1 and passes[0].action_provenance == "inferred_unique_next_draw_pass"
    assert not any(row["kind"] == 26 for row in passes[0].prefix.rows)
    later = next(row for row in decisions if row.snapshot.observer_player_id == 1 and row.trigger_event_id == 5)
    assert sum(row["kind"] == 26 for row in later.prefix.rows) == 1
    assert _markers(later.snapshot)[0]["confirmed_at_event_id"] == 4
    expected = _views(events[:5])[1]
    expected.apply_event(_private_event())
    assert later.prefix.to_bytes() == encode_prefix(expected.snapshot(), later.prefix.candidates).to_bytes()
    assert passes[0].snapshot.cutoff_event_id == 3
    without_export_hook = replay_game(SimpleNamespace(events=tuple(events)))
    assert without_export_hook["input_encoding"]["prefix_sha256"] == result["input_encoding"]["prefix_sha256"]
    assert (
        without_export_hook["input_encoding"]["conditional_input_sha256"]
        == result["input_encoding"]["conditional_input_sha256"]
    )


def _native_window():
    _, wall = _scenario()
    env = MJSOUL_RULESET.create_env(seed=42)
    observations = env.reset(wall=wall, oya=0)
    projection = LiveEventProjection()
    projection.observe_native(list(env.mjai_log))
    discard = next(candidate for candidate in legal_candidates(observations[0])
                   if candidate.spec.action_type == "DISCARD" and candidate.spec.base34 == 27
                   and not candidate.spec.riichi and not candidate.spec.tsumogiri)
    before = len(env.mjai_log)
    responses = execute_candidate(env, 0, discard)
    projection.observe_native(list(env.mjai_log[before:]))
    choices = {seat: next(candidate for candidate in legal_candidates(observation, source_player=0)
                          if candidate.spec.action_type == "PASS") for seat, observation in responses.items()}
    assert 1 in choices
    return env, projection, choices


def test_real_native_pass_submission_enters_only_its_owners_next_prefix():
    env, projection, choices = _native_window()
    receipt = next(trigger for trigger in projection.capture_self_passes(choices) if trigger.observer_player_id == 1)
    before_snapshot = projection.snapshot(1)
    ids = tuple(candidate.global_id for candidate in legal_candidates(env.get_observation(1), source_player=0))
    before_prefix = encode_prefix(before_snapshot, ids)
    old_bytes = before_prefix.to_bytes()
    assert not _markers(before_snapshot)
    before = len(env.mjai_log)
    observations = execute_joint_responses(env, choices)
    raw = list(env.mjai_log[before:])
    control = LiveEventProjection()
    control.observe_native(list(env.mjai_log[:before]))
    control.observe_native(raw)
    projection.observe_native(raw, confirmed_self_passes=(receipt,))
    assert projection.events == control.events and all(event["type"] != "self_pass" for event in projection.events)
    assert all(projection.snapshot(seat).to_bytes() == control.snapshot(seat).to_bytes() for seat in (0, 2, 3))
    next_ids = tuple(candidate.global_id for candidate in legal_candidates(observations[1], source_player=0))
    later = encode_prefix(projection.snapshot(1), next_ids)
    assert sum(row["kind"] == 26 for row in later.rows) == 1
    assert _markers(projection.snapshot(1))[0]["provenance"] == "native_committed_pass"
    assert before_prefix.to_bytes() == old_bytes and not _markers(before_snapshot)
    repeated = projection.snapshot(1).to_bytes()
    projection.observe_native([], confirmed_self_passes=(receipt,))
    assert projection.snapshot(1).to_bytes() == repeated


def test_batched_and_incremental_projection_materialize_identical_private_history():
    events, _ = _scenario()
    batch, incremental = LiveEventProjection(), LiveEventProjection()
    for projection in (batch, incremental):
        projection.observe_native(events[:4])
    batch_trigger = batch.ledgers[1].pass_trigger()
    incremental_trigger = incremental.ledgers[1].pass_trigger()
    batch.observe_native(events[4:7], confirmed_self_passes=(batch_trigger,))
    incremental.observe_native(events[4:5], confirmed_self_passes=(incremental_trigger,))
    incremental.observe_native(events[5:7])
    assert batch.events == incremental.events
    for seat in range(4):
        assert batch.snapshot(seat).to_bytes() == incremental.snapshot(seat).to_bytes()
        batch_history = TokenHistoryCache()._history(batch.snapshot(seat)).blob
        incremental_history = TokenHistoryCache()._history(incremental.snapshot(seat)).blob
        assert batch_history == incremental_history


def test_capture_cancel_and_failed_atomic_submission_never_record_success():
    env, projection, choices = _native_window()
    before = [projection.snapshot(seat).to_bytes() for seat in range(4)]
    public = len(env.mjai_log)
    projection.capture_self_passes(choices)  # Capture then cancel: no side effect.
    assert all(projection.snapshot(seat).to_bytes() == before[seat] for seat in range(4))
    bad = dict(choices)
    bad[1] = replace(bad[1], _legal_signature=())
    with pytest.raises(ValueError):
        execute_joint_responses(env, bad)
    assert len(env.mjai_log) == public
    assert all(projection.snapshot(seat).to_bytes() == before[seat] for seat in range(4))


class _FixedEnv:
    def __init__(self, env, wall):
        self.env, self.wall = env, wall

    def reset(self):
        return self.env.reset(wall=list(self.wall), oya=0)

    def __getattr__(self, name):
        return getattr(self.env, name)


class _SpyAgent:
    def __init__(self, seat, *, bad=False):
        self.seat, self.bad, self.seen = seat, bad, []
        self.first_discard = True

    def choose(self, observation, snapshot, _round):
        source = snapshot.pending_kan_actor if snapshot.pending_kan_actor is not None else snapshot.last_discard_actor
        candidates = legal_candidates(observation, source_player=source)
        self.seen.append((snapshot, tuple(candidate.global_id for candidate in candidates)))
        if snapshot.decision_phase in {"discard_response", "kan_response"}:
            selected = next(candidate for candidate in candidates if candidate.spec.action_type == "PASS")
            if self.bad and self.seat == 1:
                selected = replace(selected, _legal_signature=())
        elif self.seat == 0 and self.first_discard:
            selected = next(
                candidate for candidate in candidates if candidate.spec.action_type == "DISCARD"
                and candidate.spec.base34 == 27 and not candidate.spec.riichi and not candidate.spec.tsumogiri
            )
            self.first_discard = False
        else:
            selected = next(candidate for candidate in candidates if candidate.spec.action_type == "DISCARD"
                            and not candidate.spec.riichi)
        return selected, SimpleNamespace(evidence_status="synthetic-forward-only", fallback_reason=None, telemetry={})


def test_bounded_run_match_wires_confirmed_choices_into_later_owner_inputs(monkeypatch):
    _, wall = _scenario()
    env = _FixedEnv(MJSOUL_RULESET.create_env(seed=42), wall)
    monkeypatch.setattr(agent_module, "MJSOUL_RULESET", SimpleNamespace(create_env=lambda **_kwargs: env))
    agents = {seat: _SpyAgent(seat) for seat in range(4)}
    result = agent_module.run_match(agents, wall_seed=42, max_decisions=8)
    assert result["status"] == "BLOCKED" and result["complete"] is False
    assert result["reason"].startswith("decision_cap_reached")
    first_pass = next(snapshot for snapshot, _ in agents[1].seen if snapshot.decision_phase == "discard_response")
    assert not _markers(first_pass)
    later, ids = next((snapshot, ids) for snapshot, ids in agents[1].seen
                      if snapshot.decision_phase == "own_act" and snapshot.cutoff_event_id > first_pass.cutoff_event_id)
    assert sum(row["kind"] == 26 for row in encode_prefix(later, ids).rows) == 1


def test_failed_run_match_does_not_commit_private_history(monkeypatch):
    _, wall = _scenario()
    env = _FixedEnv(MJSOUL_RULESET.create_env(seed=42), wall)
    monkeypatch.setattr(agent_module, "MJSOUL_RULESET", SimpleNamespace(create_env=lambda **_kwargs: env))
    made = []

    def projection_factory():
        projection = LiveEventProjection()
        made.append(projection)
        return projection

    monkeypatch.setattr(agent_module, "LiveEventProjection", projection_factory)
    with pytest.raises(ValueError):
        agent_module.run_match({seat: _SpyAgent(seat, bad=True) for seat in range(4)}, wall_seed=42, max_decisions=8)
    assert len(made) == 1 and all(not _markers(made[0].snapshot(seat)) for seat in range(4))


def _fixed_native(known, draw):
    pool = list(range(136))

    def take(name):
        tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == name)
        pool.remove(tile)
        return tile

    dora, drawn = take("C"), take(draw)
    hands = [[take(name) for name in known.get(seat, ())] for seat in range(4)]
    for hand in hands:
        while len(hand) < 13:
            hand.append(pool.pop(0))
    wall = [tile for packet in range(3) for hand in hands for tile in hand[packet * 4:packet * 4 + 4]]
    wall.extend(hand[12] for hand in hands)
    wall.append(drawn)
    tail = [*pool, dora]
    index = tail.index(dora)
    tail[index], tail[-5] = tail[-5], tail[index]
    wall.extend(tail)
    assert sorted(wall) == list(range(136))
    env = MJSOUL_RULESET.create_env(seed=9)
    observations = env.reset(wall=wall, oya=0)
    projection = LiveEventProjection()
    projection.observe_native(list(env.mjai_log))
    return env, observations, projection


def test_actual_ankan_rob_window_pass_uses_public_trigger_and_keeps_others_private():
    kokushi = ["9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "C"]
    env, observations, projection = _fixed_native({0: ["1m"] * 4, 1: kokushi}, "6s")
    kan = next(candidate for candidate in legal_candidates(observations[0]) if candidate.spec.action_type == "ANKAN")
    declaration = {"type": "ankan", "actor": 0,
                   "consumed": [convert.tid_to_mjai(tile) for tile in kan.actions[0].consume_tiles]}
    before = len(env.mjai_log)
    responses = execute_candidate(env, 0, kan)
    projection.observe_native(list(env.mjai_log[before:]), declared_ankan=declaration, pending_kan_window=True)
    options = {seat: legal_candidates(observation, source_player=0) for seat, observation in responses.items()}
    assert 1 in options and any(candidate.spec.action_type == "RON" for candidate in options[1])
    choices = {seat: next(candidate for candidate in candidates if candidate.spec.action_type == "PASS")
               for seat, candidates in options.items()}
    receipt = next(trigger for trigger in projection.capture_self_passes(choices) if trigger.observer_player_id == 1)
    assert receipt.trigger_type == "ankan" and receipt.pai == "1m"
    earlier = projection.snapshot(1)
    before = len(env.mjai_log)
    execute_joint_responses(env, choices)
    projection.observe_native(list(env.mjai_log[before:]), confirmed_self_passes=(receipt,))
    assert not _markers(earlier)
    marker = _markers(projection.snapshot(1))[0]
    assert marker["trigger_type"] == "ankan" and marker["target"] == 0 and marker["pai"] == "1m"
    history = TokenHistoryCache()._history(projection.snapshot(1))
    assert sum(row["kind"] == 26 for row in history.rows) == 1
    assert all(not _markers(projection.snapshot(seat)) for seat in (0, 2, 3))


def test_overridden_chi_is_not_relabelled_as_private_pass_online_or_offline():
    env, observations, projection = _fixed_native({0: ["1m"], 1: ["2m", "3m"], 2: ["1m", "1m", "9p"]}, "9s")
    discard = next(candidate for candidate in legal_candidates(observations[0])
                   if candidate.spec.action_type == "DISCARD" and candidate.spec.base34 == 0
                   and not candidate.spec.riichi and not candidate.spec.tsumogiri)
    before = len(env.mjai_log)
    responses = execute_candidate(env, 0, discard)
    projection.observe_native(list(env.mjai_log[before:]))
    trigger = projection.ledgers[1].pass_trigger()
    choices = {}
    for seat, observation in responses.items():
        candidates = legal_candidates(observation, source_player=0)
        family = "CHI" if seat == 1 else "PON" if seat == 2 else "PASS"
        choices[seat] = next(candidate for candidate in candidates if candidate.spec.action_type == family)
    captured = projection.capture_self_passes(choices)
    assert all(receipt.observer_player_id != 1 for receipt in captured)
    before = len(env.mjai_log)
    execute_joint_responses(env, choices)
    projection.observe_native(list(env.mjai_log[before:]), confirmed_self_passes=captured)
    assert projection.events[-1]["type"] == "pon" and not _markers(projection.snapshot(1))
    with pytest.raises(ValueError, match="cannot be inferred"):
        projection.ledgers[1].record_self_pass(trigger, provenance="inferred_unique_next_draw_pass")
    decisions = []
    replay = replay_game(SimpleNamespace(events=tuple(projection.events)), on_decision=decisions.append)
    assert not replay["errors"], replay
    assert not any(row.action_target_global == 221 and row.snapshot.observer_player_id == 1
                   and row.trigger_event_id == trigger.trigger_event_id for row in decisions)


def test_kakan_ledger_trigger_keeps_original_pon_until_commit_and_never_infers_untrusted_pass():
    # Ledger transaction unit, not a claim that this is a complete native replay.
    pool = list(range(136))

    def take(name):
        tile = next(tile for tile in pool if convert.tid_to_mjai(tile) == name)
        pool.remove(tile)
        return tile

    take("C")
    for name in ("9s", "8s", "6s", "5s", "5pr", "7s"):
        take(name)
    known = {0: ["5p"], 2: ["5p", "5p", "9p"]}
    hands = [[take(name) for name in known.get(seat, ())] for seat in range(4)]
    for hand in hands:
        while len(hand) < 13:
            hand.append(pool.pop(0))
    events = [{"type": "start_game"}, {"type": "start_kyoku", "bakaze": "E", "kyoku": 1,
              "honba": 0, "kyotaku": 0, "oya": 0, "scores": [25000] * 4, "dora_marker": "C",
              "tehais": [[convert.tid_to_mjai(tile) for tile in hand] for hand in hands]},
              {"type": "tsumo", "actor": 0, "pai": "9s"},
              {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": False},
              {"type": "pon", "actor": 2, "target": 0, "pai": "5p", "consumed": ["5p", "5p"]},
              {"type": "dahai", "actor": 2, "pai": "9p", "tsumogiri": False}]
    for actor, tile in ((3, "8s"), (0, "6s"), (1, "5s")):
        events.extend([{"type": "tsumo", "actor": actor, "pai": tile},
                       {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": True}])
    events.extend([{"type": "tsumo", "actor": 2, "pai": "5pr"}, {"type": "kakan", "actor": 2, "pai": "5pr"}])
    view = _views(events)[1]
    trigger = view.pass_trigger()
    assert trigger.trigger_type == "kakan" and view.snapshot().melds[2][0].kind == "pon"
    assert len(view.snapshot().melds[2][0].tiles) == 3
    before = view.snapshot().to_bytes()
    with pytest.raises(ValueError, match="no observed closing"):
        view.record_self_pass(trigger, provenance="native_committed_pass")
    assert view.snapshot().to_bytes() == before
    view.apply_event({"type": "tsumo", "actor": 2, "pai": "7s"})
    assert view.snapshot().melds[2][0].kind == "kakan" and len(view.snapshot().melds[2][0].tiles) == 4
    with pytest.raises(ValueError, match="cannot be inferred"):
        view.record_self_pass(trigger, provenance="inferred_unique_next_draw_pass")
    assert view.record_self_pass(trigger, provenance="native_committed_pass")
    assert not view.record_self_pass(trigger, provenance="native_committed_pass")
    assert sum(row["kind"] == 26 for row in TokenHistoryCache()._history(view.snapshot()).rows) == 1
