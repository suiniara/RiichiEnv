"""Native choose/submit against Mahjong Soul result normalization, on physical walls.

The fixture serializes public results from one actually played round into server
records. It never supplies offline snapshots, candidates, or decision cutoffs.
"""
from dataclasses import dataclass

import pytest

from riichienv import Phase, convert
from riichienv.kar.actions import action_spec, execute_candidate, execute_joint_responses, tile136_to_tile37
from riichienv.kar.agent import KARAgent
from riichienv.kar.audit import replay_game
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.mjsoul import DecodedGame, normalize_records
from riichienv.kar.policy import PolicyResult
from riichienv.kar.settlement import verify_settlements
from riichienv.kar.tokenize import encode_prefix
from tests.kar.test_input_semantics import new_round_data, record


@dataclass
class _Selection:
    snapshot: object
    prefix: object
    target: int
    # Derived from the actual result event, not the observed decision cutoff.
    record_index: int = -1


class _Policy:
    def decide(self, snapshot, ids, context, *, bounds=None):
        family, tile, riichi, record_choice = self.want
        target = next(action for action in ids if action_spec(action).action_type == family
                      and bool(action_spec(action).riichi) == riichi
                      and (tile is None or action_spec(action).tile37 == tile))
        self.seen = _Selection(snapshot, encode_prefix(snapshot, ids), target) if record_choice else None
        return PolicyResult((), target, "fixture", None, {})


def _wall(specials, draws, replacements=(), *, dora_variant=False):
    pool = list(range(136))

    def take(name):
        tid = next(tile for tile in pool if convert.tid_to_mjai(tile) == name)
        pool.remove(tid)
        return tid

    normal = [take(name) for name in draws]
    rinshan = [take(name) for name in replacements]
    base, next1, next2 = (take(name) for name in ("C", "8p", "9p"))
    hands = [[take(name) for name in specials.get(seat, ())] for seat in range(4)]
    for hand in hands:
        hand.extend(pool.pop(0) for _ in range(13 - len(hand)))
    dealt = [tile for block in (0, 4, 8) for hand in hands for tile in hand[block:block + 4]]
    wall = dealt + [hand[-1] for hand in hands] + normal + pool + [base, next1, next2] + list(reversed(rinshan))
    # Hold the visible initial indicator, draws and replacement draws fixed.
    for index, tile in ((-5, base), (-7, next2 if dora_variant else next1)):
        found = wall.index(tile)
        wall[index], wall[found] = wall[found], wall[index]
    assert sorted(wall) == list(range(136))
    return wall


class _Episode:
    def __init__(self, specials, draws, replacements=(), *, dora_variant=False):
        self.env = MJSOUL_RULESET.create_env(seed=13)
        self.obs = self.env.reset(wall=_wall(specials, draws, replacements, dora_variant=dora_variant), oya=0)
        self.live = LiveEventProjection()
        self.live.observe_native(list(self.env.mjai_log))
        self.agents = [KARAgent(seat, _Policy()) for seat in range(4)]

    def _choose(self, seat, family, tile=None, *, riichi=False, capture=False):
        agent = self.agents[seat]
        view = self.live.snapshot(seat)
        self.live.ledgers[seat].check_observation(self.obs[seat])
        tile37 = None if tile is None else tile136_to_tile37(convert.mjai_to_tid(tile))
        agent.policy.want = family, tile37, riichi, capture
        action, _ = agent.choose(self.obs[seat], view, self.live.round_instance)
        return action, agent.policy.seen

    def act(self, family="DISCARD", tile=None, *, riichi=False, capture=False):
        self.resolve()
        assert self.env.phase == Phase.WaitAct
        seat = self.env.current_player
        action, selection = self._choose(seat, family, tile, riichi=riichi, capture=capture)
        public, raw = len(self.live.events), len(self.env.mjai_log)
        self.obs = execute_candidate(self.env, seat, action)
        declared_ankan = None
        if family == "ANKAN":
            declared_ankan = {"type": "ankan", "actor": seat,
                              "consumed": [convert.tid_to_mjai(t) for t in action.actions[0].consume_tiles]}
        self.live.observe_native(list(self.env.mjai_log)[raw:], declared_ankan=declared_ankan,
                                 pending_kan_window=self.env.phase == Phase.WaitResponse)
        if selection is not None:
            kind = {"DISCARD": "dahai", "TSUMO": "hora"}.get(family, family.lower())
            selection.record_index = self._find_result(public, seat, kind)
        return selection

    def _find_result(self, start, seat, kind):
        return next(i for i in range(start, len(self.live.events))
                    if self.live.events[i]["type"] == kind and self.live.events[i]["actor"] == seat)

    def resolve(self, family="PASS", *, seat=None, capture=False):
        if self.env.phase != Phase.WaitResponse:
            assert not capture
            return None
        choices, selected = {}, None
        for player in sorted(self.env.active_players):
            action, choice = self._choose(player, family if player == seat else "PASS",
                                          capture=capture and player == seat)
            choices[player] = action
            if choice is not None:
                selected = choice
        receipts = self.live.capture_self_passes(choices)
        public, raw = len(self.live.events), len(self.env.mjai_log)
        self.obs = execute_joint_responses(self.env, choices)
        self.live.observe_native(list(self.env.mjai_log)[raw:], confirmed_self_passes=receipts)
        if capture:
            assert selected is not None
            if family == "PASS":
                selected.record_index = next(i for i in range(public - 1, -1, -1)
                                             if self.live.events[i]["type"] == "dahai")
            else:
                selected.record_index = self._find_result(public, seat, "hora" if family == "RON" else family.lower())
        return selected

    def finish_round(self):
        # Bounded native completion of this one round only, stopping before any
        # next-round action. This supplies a real terminal server result.
        for _ in range(75):
            self.resolve()
            if any(event["type"] == "end_kyoku" for event in self.live.events):
                break
            self.act()
        else:
            raise AssertionError("The scripted first round never closed")
        end = next(i for i, event in enumerate(self.live.events) if event["type"] == "end_kyoku")
        return self.live.events[:end + 1]


def _server_records(events):  # noqa: PLR0915
    start, first = events[1:3]
    assert start["type"] == "start_kyoku" and first["type"] == "tsumo"
    records = [record(1, "RecordNewRound", new_round_data(start, first))]
    indicators, scores = [start["dora_marker"]], list(start["scores"])
    declaration, deposit = None, None
    for index, event in enumerate(events[3:], 3):
        kind = event["type"]
        if kind == "reach":
            declaration = event["actor"]
            continue
        if kind == "reach_accepted":
            seat = event["actor"]
            scores[seat] -= 1000
            deposit = {"seat": seat, "score": scores[seat]}
            continue
        if kind == "dora":
            indicators.append(event["dora_marker"])
            continue
        if kind in {"tsumo", "dahai"}:
            dealing = kind == "tsumo"
            data = {"seat": event["actor"], "tile": event["pai"], "doras": indicators.copy()}
            if not dealing:
                data.update(is_liqi=declaration == event["actor"], moqie=event["tsumogiri"])
                declaration = None
            name = "RecordDealTile" if dealing else "RecordDiscardTile"
        elif kind in {"chi", "pon", "daiminkan"}:
            data = {"seat": event["actor"], "type": ("chi", "pon", "daiminkan").index(kind),
                    "tiles": [event["pai"], *event["consumed"]],
                    "froms": [event["target"], *([event["actor"]] * len(event["consumed"]))]}
            name = "RecordChiPengGang"
        elif kind in {"ankan", "kakan"}:
            data = {"seat": event["actor"], "type": 3 if kind == "ankan" else 2,
                    "tiles": event["consumed"][0] if kind == "ankan" else event["pai"]}
            # Server kan records can include a marker made visible in that
            # transaction; dora before a discard instead belongs to its result.
            for later in events[index + 1:]:
                if later["type"] != "dora":
                    break
                data.setdefault("doras", indicators.copy()).append(later["dora_marker"])
            name = "RecordAnGangAddGang"
        elif kind == "hora":
            rules = verify_settlements(events)
            assert rules["status"] == "PASS", rules
            derived = rules["settlements"][-1]["derived_wins"]
            assert len(derived) == 1 and derived[0]["player_id"] == event["actor"]
            data = {"old_scores": scores.copy(), "scores": event["scores"], "delta_scores": event["deltas"],
                    "doras": indicators.copy(), "hules": [{"seat": event["actor"], "zimo": False,
                    "hu_tile": event["pai"], "count": derived[0]["computed_han"], "fu": derived[0]["computed_fu"]}]}
            name = "RecordHule"
        elif kind == "ryukyoku":
            assert event["reason"] in {"exhaustive_draw", "nagashi_mangan"}, event
            data = {"liujumanguan": event["reason"] == "nagashi_mangan",
                    "scores": [{"old_scores": scores.copy(), "delta_scores": event["deltas"]}],
                    "players": [{"tingpai": hand is not None, "hand": hand or []} for hand in event["public_hands"]]}
            name = "RecordNoTile"
        elif kind == "end_kyoku":
            break
        else:
            raise AssertionError((index, event))
        if deposit is not None:
            data["liqi"], deposit = deposit, None
        records.append(record(index, name, data))
    return records


def _offline(episode):
    live = episode.finish_round()
    records = _server_records(live)
    events = normalize_records(records)
    # No reordering or stripping of a public bonus reveal/payment to make
    # inputs match. Only server packing, not decision cutoffs, is supplied.
    assert [item["type"] for item in events] == [item["type"] for item in live]
    decisions = []
    game = DecodedGame("a" * 64, "b" * 64, "synthetic-native-wall", {}, tuple(records), events)
    report = replay_game(game, on_decision=decisions.append)
    assert not report["errors"], report["errors"]
    assert report["independent_settlement"]["status"] == "PASS", report["independent_settlement"]
    return events, decisions


def _comparison(offline, live):
    events, decisions = offline
    matches = [row for row in decisions if row.snapshot.observer_player_id == live.snapshot.observer_player_id
               and row.action_target_global == live.target
               and events[row.trigger_event_id].get("source_record_index") == live.record_index]
    assert len(matches) == 1
    row = matches[0]
    assert row.prefix.candidates == live.prefix.candidates
    assert row.prefix.to_bytes() == live.prefix.to_bytes(), {
        "live_cutoff": live.snapshot.cutoff_event_id, "offline_cutoff": row.snapshot.cutoff_event_id,
        "live_indicators": len(live.snapshot.dora_indicators), "offline_indicators": len(row.snapshot.dora_indicators),
        "live_scores": live.snapshot.scores, "offline_scores": row.snapshot.scores,
    }
    assert row.prefix.to_record() == live.prefix.to_record()  # all fields, including positions
    return row


def _kan_sequence(case, *, dora_variant=False):
    if case == "kakan":
        ep = _Episode({1: ("E", "E")}, ("E", "8s", "9s", "7m", "E"), ("7s",), dora_variant=dora_variant)
        ep.act(tile="E")
        ep.resolve("PON", seat=1)
        ep.act()
        for draw in ("8s", "9s", "7m"):
            ep.act(tile=draw)
        before = ep.act("KAKAN", capture=True)
    elif case in {"daiminkan", "consecutive"}:
        ep = _Episode({1: ("E",) * 3 + (("S",) * 4 if case == "consecutive" else ())}, ("E",), ("7s", "6s"),
                      dora_variant=dora_variant)
        ep.act(tile="E")
        before = ep.resolve("DAIMINKAN", seat=1, capture=True)
        if case == "consecutive":
            before = ep.act("ANKAN", capture=True)
    elif case == "ankan":
        ep = _Episode({0: ("S",) * 4}, ("E",), ("7s",))
        before = ep.act("ANKAN", capture=True)
    else:
        ep = _Episode({}, ("E",))
        before = None
    discard = ep.act(capture=True)
    after = ep.act(capture=True)
    return ep, before, discard, after


@pytest.mark.parametrize(
    "case,indicators", [("daiminkan", 1), ("kakan", 1), ("consecutive", 3), ("ankan", 2), ("ordinary", 1)]
)
def test_offline_kan_discard_decision_matches_native_choose_before_submission(case, indicators):
    ep, before, discard, after = _kan_sequence(case)
    assert len(discard.snapshot.dora_indicators) == indicators
    offline = _offline(ep)
    if before is not None:
        _comparison(offline, before)
    _comparison(offline, discard)
    _comparison(offline, after)


def test_actual_deferred_dora_variant_cannot_change_current_frozen_discard():
    episodes = [_kan_sequence("daiminkan", dora_variant=value) for value in (False, True)]
    pairs = [(_comparison((offline := _offline(ep)), discard), _comparison(offline, after))
             for ep, _, discard, after in episodes]
    (current0, next0), (current1, next1) = pairs
    assert current0.prefix.to_bytes() == current1.prefix.to_bytes()
    assert next0.snapshot.dora_indicators != next1.snapshot.dora_indicators
    assert next0.prefix.to_bytes() != next1.prefix.to_bytes()


@pytest.mark.parametrize("family", ["CHI", "PON", "DAIMINKAN", "RON", "PASS"])
def test_offline_riichi_declaration_response_is_frozen_before_deposit_and_resolution(family):
    tenpai = ("1p", "2p", "3p", "4p", "5p", "6p", "6s", "7s", "8s", "P", "P", "F", "F")
    offers = {"CHI": ("1m", "2m"), "PON": ("3m",) * 2, "DAIMINKAN": ("3m",) * 3,
              "RON": ("1m", "2m", "1s", "2s", "3s", "4s", "5s", "6s", "E", "E", "E", "N", "N"),
              "PASS": ("1m", "2m")}
    ep = _Episode({0: tenpai, 1: offers[family]}, ("3m",), ("7s",))
    declared = ep.act(riichi=True, tile="3m", capture=True)
    response = ep.resolve(family, seat=1, capture=True)
    assert response.snapshot.riichi_declared[0] and not response.snapshot.riichi_accepted[0]
    assert response.snapshot.scores[0] == 25000 and response.snapshot.riichi_sticks == 0
    subsequent = None if family == "RON" else ep.act(capture=True)
    offline = _offline(ep)
    _comparison(offline, declared)
    frozen = _comparison(offline, response)
    assert frozen.snapshot.scores[0] == 25000 and not frozen.snapshot.riichi_accepted[0]
    if subsequent is not None:
        downstream = _comparison(offline, subsequent)
        assert downstream.snapshot.scores[0] == 24000 and downstream.snapshot.riichi_sticks == 1
        assert ('"type":"self_pass"' in "".join(downstream.snapshot.events)) == (family == "PASS")
    else:
        assert not any(event["type"] == "reach_accepted" for event in offline[0])


def test_actual_pon_vs_ron_resolution_does_not_change_the_same_declaration_response_input():
    dealer = ("1p", "2p", "3p", "4p", "5p", "6p", "6s", "7s", "8s", "P", "P", "F", "F")
    # Both choices are legal on 3m: the player can Pon, or Ron to finish
    # the shanpon hand with an East triplet as yaku.
    responder = ("3m", "3m", "N", "N", "1s", "2s", "3s", "4s", "5s", "6s", "E", "E", "E")
    frozen = []
    for outcome in ("PON", "RON"):
        ep = _Episode({0: dealer, 1: responder}, ("3m",))
        ep.act(riichi=True, tile="3m")
        response = ep.resolve(outcome, seat=1, capture=True)
        after = ep.act(capture=True) if outcome == "PON" else None
        offline = _offline(ep)
        frozen.append(_comparison(offline, response))
        accepted = any(event["type"] == "reach_accepted" for event in offline[0])
        assert accepted == (outcome == "PON")
        if after is not None:
            assert _comparison(offline, after).snapshot.scores[0] == 24000
    assert frozen[0].prefix.to_bytes() == frozen[1].prefix.to_bytes()
    assert frozen[0].conditional.to_bytes() != frozen[1].conditional.to_bytes()


def test_consecutive_kakan_flushes_only_the_prior_indicator_before_robbery_choices():
    robber = ("1m", "2m", "4p", "5p", "6p", "7p", "8p", "9p", "4s", "5s", "6s", "W", "W")
    ep = _Episode({1: ("3m", "3m", "S", "S", "S"), 2: robber}, ("3m", "8s", "9s", "S"), ("3m",))
    ep.act(tile="3m")
    ep.resolve("PON", seat=1)
    ep.act()
    ep.act(tile="8s")
    ep.act(tile="9s")
    ep.act(tile="S")
    ep.resolve("DAIMINKAN", seat=1)
    upgrade = ep.act("KAKAN", capture=True)
    ron = ep.resolve("RON", seat=2, capture=True)
    assert len(upgrade.snapshot.dora_indicators) == 1
    assert len(ron.snapshot.dora_indicators) == 2  # the earlier daiminkan indicator, not a kakan indicator
    offline = _offline(ep)
    _comparison(offline, upgrade)
    _comparison(offline, ron)
