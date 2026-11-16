from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from riichienv import Conditions, HandEvaluator, Meld, MeldType, convert

from .contracts import MJSOUL_RULESET, Ruleset
from .kan_transactions import COMMITTED_KAN_KINDS, pending_kan_commits
from .observation import ObserverLedger, PublicMeld
from .payments import EvaluatedWin, draw_payments, end_game_pot, win_payments
from .response import draw_response

SETTLEMENT_VERSION = "independent_mjsoul_settlement_v1"


class SettlementError(ValueError):
    pass


class SettlementUnavailableError(ValueError):
    pass


@dataclass(frozen=True)
class WinFacts:
    player_id: int
    target_player_id: int
    winning_tile: int
    ura_markers: tuple[int, ...] | None


def _points(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise SettlementError("A score vector must have four entries")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise SettlementError("Score entries must be integer points")
    return tuple(value)


def _tile(value: str) -> int:
    return convert.mjai_to_tid(value)


def _meld(meld: PublicMeld) -> Meld:
    kinds = {
        "chi": MeldType.Chi,
        "pon": MeldType.Pon,
        "daiminkan": MeldType.Daiminkan,
        "ankan": MeldType.Ankan,
        "kakan": MeldType.Kakan,
    }
    if meld.kind not in kinds:
        raise SettlementError("Unknown committed meld type")
    expected = 3 if meld.kind in {"chi", "pon"} else 4
    bases = sorted(tile // 4 for tile in meld.tiles)
    if len(bases) != expected:
        raise SettlementError("Committed meld has the wrong tile count")
    if meld.kind == "chi":
        if bases[0] >= 27 or bases[-1] // 9 != bases[0] // 9 or bases != list(range(bases[0], bases[0] + 3)):
            raise SettlementError("Invalid sequence meld")
    elif len(set(bases)) != 1:
        raise SettlementError("Invalid triplet/kan meld")
    return Meld(kinds[meld.kind], list(meld.tiles), meld.kind != "ankan", -1 if meld.source is None else meld.source)


def _physical_tiles(tiles: list[int]) -> None:
    counts = Counter(tile // 4 for tile in tiles)
    if any(count > 4 for count in counts.values()):
        raise SettlementError("More than four physical copies of one tile type")
    if any(tiles.count(red) > 1 for red in (16, 52, 88)):
        raise SettlementError("More than one red five in the frozen ruleset")


class _Verifier:
    def __init__(self, ruleset: Ruleset):
        if ruleset != MJSOUL_RULESET:
            raise SettlementUnavailableError("This settlement contract is frozen to Mahjong Soul four-player hanchan")
        self.ruleset = ruleset
        self.views = [ObserverLedger(seat, (0, 1, 2, 3)) for seat in range(4)]
        self.points: tuple[int, ...] | None = (25000,) * 4
        self.pot = 0
        self.rows: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.prefix_checks: list[dict[str, Any]] = []
        self.pending_comparisons: list[dict[str, Any]] = []
        self.rule_requires_match_end: bool | None = False
        self.in_round = False
        self.explicit_end = False
        self.expected_round: tuple[int, int, int] | None = (0, 0, 0)
        self.expected_draw: int | None = None
        self.expected_discard: int | None = None
        self.oya = self.wind = self.honba = 0
        self.remaining = 70
        self.calls = False
        self.draws = [0] * 4
        self.discards = [0] * 4
        self.accepted = [False] * 4
        self.double = [False] * 4
        self.ippatsu = [False] * 4
        self.pending_reach: dict[int, bool] = {}
        self.pending_kan: tuple[str, int, int] | None = None
        self.replacement_actor: int | None = None
        self.last_draw: tuple[int, int, bool] | None = None
        self.last_trigger: tuple[int, int] | None = None
        self.pao: list[dict[int, int]] = [{} for _ in range(4)]
        self.end_pot_delta = (0, 0, 0, 0)

    def _balance(self) -> None:
        if self.points is not None and sum(self.points) + self.pot * 1000 != 100000:
            raise SettlementError("Independent point-and-deposit conservation failed")

    def _start_round(self, event: dict[str, Any], index: int) -> None:
        if self.in_round or self.rule_requires_match_end is True:
            raise SettlementError(
                "New round before terminal resolution or after independently derived match termination"
            )
        wind = ("E", "S", "W", "N").index(event["bakaze"])
        actual = wind, event["oya"], event["honba"]
        if event["kyoku"] != event["oya"] + 1:
            raise SettlementError("Round number and stable dealer seat disagree")
        if self.expected_round is not None and actual != self.expected_round:
            raise SettlementError("Round continuation disagrees with the independently derived outcome")
        if event.get("kyotaku", event.get("kyoutaku", 0)) != self.pot:
            raise SettlementError("New-round deposit count differs from the independent pot")
        source = _points(event["scores"])
        status = "BLOCKED" if self.points is None else "PASS" if source == self.points else "FAIL"
        self.prefix_checks.append(
            {"event_id": index, "status": status, "expected_scores": self.points, "observed_scores": source}
        )
        if status == "FAIL":
            raise SettlementError("New-round scores differ from the independent score ledger")
        for row in self.pending_comparisons:
            expected = row["expected_scores"]
            if expected is not None:
                row["status"] = "PASS" if tuple(expected) == source else "FAIL"
                row["comparison_source"] = "next_round_start_scores"
                row["observed_scores"] = source
                if row["status"] == "PASS":
                    row.pop("reason", None)
        self.pending_comparisons.clear()
        self.wind, self.oya, self.honba = actual
        self.remaining = 70
        self.calls = False
        self.draws = [0] * 4
        self.discards = [0] * 4
        self.accepted = [False] * 4
        self.double = [False] * 4
        self.ippatsu = [False] * 4
        self.pending_reach.clear()
        self.pending_kan = None
        self.replacement_actor = None
        self.last_draw = self.last_trigger = None
        self.pao = [{} for _ in range(4)]
        self.expected_draw = self.oya
        self.expected_discard = None
        self.in_round = True

    def _commit_kan(self) -> None:
        if self.pending_kan is None:
            return
        _, actor, _ = self.pending_kan
        self.calls = True
        self.ippatsu = [False] * 4
        self.replacement_actor = actor
        self.last_trigger = None
        self.pending_kan = None

    def observe(self, event: dict[str, Any], index: int) -> None:  # noqa: PLR0915
        typ = event["type"]
        if typ == "start_kyoku":
            self._start_round(event, index)
        elif typ == "tsumo":
            actor = event["actor"]
            if actor != self.expected_draw or not self.in_round or self.remaining <= 0:
                raise SettlementError("Draw is outside the independent turn context")
            self._commit_kan()
            rinshan = self.replacement_actor == actor
            self.remaining -= 1
            self.draws[actor] += 1
            self.last_draw = actor, _tile(event["pai"]), rinshan
            self.last_trigger = None
            self.replacement_actor = None
            self.expected_draw = None
            self.expected_discard = actor
        elif typ == "dahai":
            actor = event["actor"]
            if self.expected_draw is not None or self.expected_discard != actor or not self.in_round:
                raise SettlementError("Discard before a required draw or replacement draw, or by the wrong actor")
            self.expected_discard = None
            self.discards[actor] += 1
            self.ippatsu[actor] = False
            self.last_trigger = actor, _tile(event["pai"])
            self.last_draw = None
            self.expected_draw = (actor + 1) % 4
        elif typ == "reach":
            actor = event["actor"]
            if self.expected_discard != actor or self.expected_draw is not None:
                raise SettlementError("Riichi declaration outside the player's discard turn")
            if actor in self.pending_reach or self.accepted[actor]:
                raise SettlementError("Duplicate riichi declaration")
            self.pending_reach[actor] = self.discards[actor] == 0 and not self.calls
        elif typ == "reach_accepted":
            actor = event["actor"]
            if actor not in self.pending_reach:
                raise SettlementError("Riichi accepted without a pending declaration")
            self.double[actor] = self.pending_reach.pop(actor)
            self.accepted[actor] = self.ippatsu[actor] = True
            if self.points is not None:
                if self.points[actor] < 1000:
                    raise SettlementError("Insufficient independent points for riichi")
                self.points = tuple(score - (1000 if seat == actor else 0) for seat, score in enumerate(self.points))
            self.pot += 1
        elif typ in {"chi", "pon", "kan", "daiminkan"}:
            actor = event["actor"]
            if self.last_trigger != (event["target"], _tile(event["pai"])):
                raise SettlementError("Call source differs from the independent trigger")
            self.calls = True
            self.ippatsu = [False] * 4
            self.last_draw = self.last_trigger = None
            kan = typ in {"kan", "daiminkan"}
            self.expected_draw = actor if kan else None
            self.expected_discard = None if kan else actor
            self.replacement_actor = actor if kan else None
        elif typ in {"ankan", "kakan"}:
            actor = event["actor"]
            if self.last_draw is None or self.last_draw[0] != actor:
                raise SettlementError("Kan declaration without the player's draw")
            tile = _tile(event["pai"]) if typ == "kakan" else _tile(event["consumed"][0])
            self.pending_kan = typ, actor, tile
            self.last_trigger = actor, tile
            self.last_draw = None
            self.expected_draw = actor
            self.expected_discard = None
        elif typ in {"dora", "kan_commit"}:
            if self.pending_kan is not None:
                view = self.views[0].snapshot()
                if pending_kan_commits(
                    event,
                    actor=self.pending_kan[1],
                    committed_kans=sum(meld.kind in COMMITTED_KAN_KINDS for player in view.melds for meld in player),
                    indicators=len(view.dora_indicators),
                ):
                    self._commit_kan()
            elif typ == "kan_commit":
                raise SettlementError("Commit acknowledgement without a pending kan")
        elif typ == "end_game":
            if self.in_round or self.rule_requires_match_end is False:
                raise SettlementError("Match ends before independently derived hanchan termination")
            if self.rule_requires_match_end is None:
                raise SettlementUnavailableError(
                    "Match termination cannot be checked without a complete settlement chain"
                )
            if self.points is not None:
                self.end_pot_delta = end_game_pot(self.points, self.pot, (0, 1, 2, 3))
                self.points = tuple(score + delta for score, delta in zip(self.points, self.end_pot_delta))
            self.pot = 0
            self.explicit_end = True
        elif typ not in {"start_game", "end_kyoku"}:
            raise SettlementError(f"Unsupported settlement event {typ}")
        for view in self.views:
            view.apply_event(event)
        if typ in {"pon", "kan", "daiminkan"}:
            actor, tile, payer = event["actor"], _tile(event["pai"]) // 4, event["target"]
            melds = self.views[actor].snapshot().melds[actor]
            if tile >= 31 and sum(m.tiles[0] // 4 >= 31 for m in melds) == 3:
                self.pao[actor][37] = payer
            if 27 <= tile <= 30 and sum(27 <= m.tiles[0] // 4 <= 30 for m in melds) == 4:
                self.pao[actor][50] = payer
        self._balance()

    def _facts(self, event: dict[str, Any]) -> WinFacts:
        actor, target = event["actor"], event["target"]
        if actor not in range(4) or target not in range(4):
            raise SettlementError("Invalid winner or source seat")
        if actor == target:
            if self.last_draw is None or self.last_draw[0] != actor:
                raise SettlementError("Self-draw win without the player's draw")
            expected = self.last_draw[1]
        else:
            if self.last_trigger is None or self.last_trigger[0] != target:
                raise SettlementError("Ron has no matching discard/kan trigger")
            expected = self.last_trigger[1]
        tile = _tile(event["pai"]) if "pai" in event else expected
        if tile != expected:
            raise SettlementError("Winning tile differs from the independent trigger")
        ura = tuple(_tile(value) for value in event["ura_markers"]) if "ura_markers" in event else None
        return WinFacts(actor, target, tile, ura)

    def _score_win(self, fact: WinFacts) -> tuple[EvaluatedWin, dict[str, Any]]:
        seat = fact.player_id
        view = self.views[seat].snapshot()
        if not view.physical_ledger_complete:
            raise SettlementUnavailableError("Winner's pre-resolution private tiles are incomplete")
        tsumo = fact.target_player_id == seat
        hand = list(view.hand)
        if tsumo:
            hand.remove(fact.winning_tile)
        melds = list(view.melds[seat])
        if len(hand) + 3 * len(melds) != 13:
            raise SettlementError("Winning hand must contain exactly thirteen structural tiles before the win")
        body = hand + [fact.winning_tile] + [tile for meld in melds for tile in meld.tiles]
        _physical_tiles(body)
        calculator = HandEvaluator(hand, [_meld(meld) for meld in melds])
        waits = set(calculator.get_waits())
        if not tsumo and waits.intersection(tile // 4 for tile in view.discard_history[seat]):
            raise SettlementError("Winner is permanently furiten on the current wait set")
        ura = fact.ura_markers if self.accepted[seat] else ()
        if self.accepted[seat] and (ura is None or len(ura) != len(view.dora_indicators)):
            raise SettlementUnavailableError("Missing or incomplete actual ura indicator reveal for a riichi winner")
        assert ura is not None
        _physical_tiles(body + list(view.dora_indicators) + list(ura))
        chankan = not tsumo and self.pending_kan is not None
        conditions = Conditions(
            tsumo=tsumo,
            riichi=self.accepted[seat],
            double_riichi=self.double[seat],
            ippatsu=self.ippatsu[seat],
            haitei=tsumo and self.remaining == 0 and self.last_draw is not None and not self.last_draw[2],
            houtei=not tsumo and self.remaining == 0 and not chankan,
            rinshan=tsumo and self.last_draw is not None and self.last_draw[2],
            chankan=chankan,
            tsumo_first_turn=tsumo and self.draws[seat] == 1 and not self.calls,
            player_wind=(seat - self.oya) % 4,
            round_wind=self.wind,
            honba=0,
            riichi_sticks=0,
        )
        result = calculator.calc(fact.winning_tile, list(view.dora_indicators), conditions, list(ura))
        if not result.is_win:
            raise SettlementError("No independently valid winning shape and yaku")
        if chankan and self.pending_kan is not None and self.pending_kan[0] == "ankan":
            if not self.ruleset.allows_ron_on_ankan_for_kokushi_musou or not {42, 49}.intersection(result.yaku):
                raise SettlementError("Only kokushi may rob an ankan in the frozen ruleset")
        liabilities = [(self.pao[seat][yaku], 2 if yaku == 50 else 1) for yaku in result.yaku if yaku in self.pao[seat]]
        if len({payer for payer, _ in liabilities}) > 1:
            raise SettlementError("Ambiguous multiple responsibility payers")
        payer = liabilities[0][0] if liabilities else None
        win = EvaluatedWin(seat, result, payer, sum(units for _, units in liabilities))
        return win, {
            "player_id": seat,
            "target_player_id": fact.target_player_id,
            "computed_han": result.han,
            "computed_fu": result.fu,
            "computed_yaku": list(result.yaku),
            "yakuman": result.yakuman,
            "conditions": asdict(conditions),
            "ura_indicators_used": list(ura),
            "liable_player": payer,
            "liable_yakuman_units": win.liable_yakuman_units,
        }

    def _next_round(self, renchan: bool, draw: bool, abortive: bool = False) -> None:
        dealer = self.oya if renchan else (self.oya + 1) % 4
        wind = self.wind + int(not renchan and dealer == 0)
        honba = self.honba + 1 if renchan or draw else 0
        self.expected_round = wind, dealer, honba
        if self.points is None:
            self.rule_requires_match_end = None
            return
        points = self.points
        top = min(range(4), key=lambda seat: (-points[seat], seat))
        all_last = self.wind > 1 or (self.wind == 1 and self.oya == 3)
        agari_yame = not abortive and renchan and all_last and top == self.oya and self.points[self.oya] >= 30000
        regular_end = not abortive and not renchan and wind >= 2 and (max(self.points) >= 30000 or wind > 2)
        self.rule_requires_match_end = min(self.points) < 0 or agari_yame or regular_end

    def _compare(self, row: dict[str, Any], events: list[dict[str, Any]], delta: tuple[int, ...]) -> None:
        expected = None if self.points is None else tuple(score + change for score, change in zip(self.points, delta))
        row.update(expected_deltas=delta, expected_scores=expected, observed_deltas=[], observed_scores=[])
        matched = True
        targets = 0
        for event in events:
            if "delta" in event and "deltas" in event and _points(event["delta"]) != _points(event["deltas"]):
                raise SettlementError("Conflicting delta representations")
            source_delta = event.get("deltas", event.get("delta"))
            if source_delta is not None:
                observed = _points(source_delta)
                row["observed_deltas"].append(observed)
                targets += 1
                matched &= observed == delta
            if "scores" in event:
                observed = _points(event["scores"])
                row["observed_scores"].append(observed)
                targets += 1
                if expected is not None:
                    matched &= observed == expected
        row["status"] = "FAIL" if not matched else "BLOCKED" if expected is None or targets == 0 else "PASS"
        if not matched:
            row["reason"] = "independent_settlement_mismatch"
        elif expected is None:
            row["reason"] = "prior_settlement_chain_unavailable"
        elif targets == 0:
            row["reason"] = "settlement_comparison_target_missing"
            self.pending_comparisons.append(row)
        self.points = expected

    def win(self, events: list[dict[str, Any]], indices: list[int]) -> None:
        row: dict[str, Any] = {"kind": "win", "event_ids": indices, "derived_wins": []}
        try:
            facts = [self._facts(event) for event in events]
            if (
                len({fact.player_id for fact in facts}) != len(facts)
                or len({fact.target_player_id for fact in facts}) != 1
            ):
                raise SettlementError("Conflicting winner group")
            if len({event.get("source_record_index") for event in events}) > 1:
                raise SettlementError("Winner group spans different server results")
            target = None if facts[0].player_id == facts[0].target_player_id else facts[0].target_player_id
            if target is None and len(facts) != 1:
                raise SettlementError("A self-draw cannot have multiple winners")
            reveals = {fact.ura_markers for fact in facts if self.accepted[fact.player_id] and fact.ura_markers}
            if len(reveals) > 1:
                raise SettlementError("Riichi winners have conflicting ura indicator facts")
            scored = [self._score_win(fact) for fact in facts]
            row["derived_wins"] = [detail for _, detail in scored]
            delta = win_payments(
                [win for win, _ in scored], oya=self.oya, target=target, honba=self.honba, riichi_sticks=self.pot
            )
            self._compare(row, events, delta)
            self._next_round(self.oya in {fact.player_id for fact in facts}, False)
        except (SettlementUnavailableError, SettlementError, ValueError) as exc:
            row.update(
                status="BLOCKED" if isinstance(exc, SettlementUnavailableError) else "FAIL",
                reason=str(exc),
                expected_deltas=None,
                expected_scores=None,
            )
            self.points = None
            self.expected_round = None
            self.rule_requires_match_end = None
        self.pot = 0
        self.rows.append(row)
        self.in_round = False
        self.pending_kan = None
        self.pending_reach.clear()
        self.expected_draw = None
        self.expected_discard = None
        for event in events:
            for view in self.views:
                view.apply_event(event)
        self._balance()

    def draw(self, event: dict[str, Any], index: int) -> None:
        row: dict[str, Any] = {"kind": "draw", "event_ids": [index]}
        try:
            reason = event.get("reason")
            ordinary = reason in {"exhaustive", "exhaustive_draw", "fanpai", "nagashi_mangan"}
            if ordinary:
                if self.remaining != 0:
                    raise SettlementError("Exhaustive draw before the live wall is exhausted")
                tenpai = []
                nagashi = []
                for seat, ledger in enumerate(self.views):
                    view = ledger.snapshot()
                    if not view.physical_ledger_complete:
                        raise SettlementUnavailableError("A player's private tiles are missing for draw settlement")
                    hand, melds = list(view.hand), list(view.melds[seat])
                    _physical_tiles(hand + [tile for meld in melds for tile in meld.tiles])
                    if len(hand) + 3 * len(melds) != 13:
                        raise SettlementError("Draw hand does not have thirteen structural tiles")
                    tenpai.append(HandEvaluator(hand, [_meld(meld) for meld in melds]).is_tenpai())
                    river = view.discard_history[seat]
                    unclaimed = len(river) == len(view.current_rivers[seat])
                    if unclaimed and all(tile // 4 >= 27 or tile // 4 % 9 in {0, 8} for tile in river):
                        nagashi.append(seat)
                if (reason == "nagashi_mangan") != bool(nagashi):
                    raise SettlementError("Recorded draw type disagrees with independently derived nagashi eligibility")
                delta = draw_payments(tenpai, oya=self.oya, nagashi_winners=nagashi)
                row.update(tenpai=tenpai, nagashi_winners=nagashi)
                renchan = tenpai[self.oya]
            else:
                target = draw_response(reason)
                if reason not in {"kyushukyuhai", "kyushu_kyuhai"} and target not in {16, 17, 18}:
                    raise SettlementError("Unknown or disabled abortive draw")
                delta = draw_payments([False] * 4, oya=self.oya, abortive=True)
                renchan = True
            self._compare(row, [event], delta)
            self._next_round(renchan, True, abortive=not ordinary)
        except (SettlementUnavailableError, SettlementError, ValueError) as exc:
            row.update(
                status="BLOCKED" if isinstance(exc, SettlementUnavailableError) else "FAIL",
                reason=str(exc),
                expected_deltas=None,
                expected_scores=None,
            )
            self.points = None
            self.expected_round = None
            self.rule_requires_match_end = None
        self.rows.append(row)
        self.in_round = False
        self.pending_kan = None
        self.pending_reach.clear()
        self.expected_draw = None
        self.expected_discard = None
        for view in self.views:
            view.apply_event(event)
        self._balance()


def verify_settlements(
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    observed_final_scores: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    state = _Verifier(MJSOUL_RULESET)
    index = 0
    if not events or events[0].get("type") != "start_game":
        return {
            "status": "BLOCKED",
            "version": SETTLEMENT_VERSION,
            "reason": "missing_game_start",
            "settlements": [],
            "computed_final_scores": None,
        }
    while index < len(events):
        event = events[index]
        try:
            if event["type"] == "hora":
                end = index + 1
                while end < len(events) and events[end]["type"] == "hora":
                    end += 1
                state.win(list(events[index:end]), list(range(index, end)))
                index = end
                continue
            if event["type"] == "ryukyoku":
                state.draw(event, index)
            else:
                state.observe(event, index)
        except (SettlementUnavailableError, SettlementError, ValueError, KeyError) as exc:
            state.errors.append(
                {
                    "event_id": index,
                    "status": "BLOCKED" if isinstance(exc, SettlementUnavailableError) else "FAIL",
                    "reason": str(exc),
                }
            )
            break
        index += 1
    final_comparison: dict[str, Any] = {"status": "NOT_APPLICABLE", "is_completion_evidence": False}
    if observed_final_scores is not None and state.points is not None:
        try:
            actual = _points(observed_final_scores)
            final_comparison.update(
                observed_scores=actual, expected_scores=state.points, scores_match=actual == state.points
            )
            final_comparison["status"] = (
                ("PASS" if actual == state.points else "FAIL") if state.explicit_end else "DIAGNOSTIC_ONLY"
            )
            if not state.errors and state.rule_requires_match_end is True:
                before_award = (
                    tuple(score - delta for score, delta in zip(state.points, state.end_pot_delta))
                    if state.explicit_end
                    else state.points
                )
                remaining_award = (
                    state.end_pot_delta if state.explicit_end else end_game_pot(state.points, state.pot, (0, 1, 2, 3))
                )
                expected_endpoint = tuple(score + delta for score, delta in zip(before_award, remaining_award))
                if expected_endpoint == actual:
                    for row in state.pending_comparisons:
                        if row.get("expected_scores") is not None and tuple(row["expected_scores"]) == before_award:
                            row.update(
                                status="PASS",
                                comparison_source="final_raw_scores_after_independent_termination",
                                observed_scores_after_final_pot=actual,
                                comparison_end_pot_deltas=remaining_award,
                            )
                            row.pop("reason", None)
        except SettlementError as exc:
            final_comparison.update(status="FAIL", reason=str(exc))
    if state.explicit_end and any(state.end_pot_delta) and observed_final_scores is None:
        final_comparison.update(status="BLOCKED", reason="unclaimed_match_pot_requires_final_score_comparison")
    statuses = (
        [row["status"] for row in state.rows]
        + [row["status"] for row in state.errors]
        + [row["status"] for row in state.prefix_checks]
        + [final_comparison["status"]]
    )
    status = (
        "FAIL"
        if "FAIL" in statuses
        else "BLOCKED"
        if "BLOCKED" in statuses or not state.rows or state.points is None
        else "PASS"
    )
    return {
        "status": status,
        "version": SETTLEMENT_VERSION,
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "scope": "independently_calculated_settlements_in_observed_prefix_not_match_completion",
        "settlements": state.rows,
        "prefix_score_checks": state.prefix_checks,
        "errors": state.errors,
        "computed_final_scores": state.points,
        "computed_riichi_sticks": state.pot,
        "explicit_end_game": state.explicit_end,
        "independent_rule_requires_match_end": state.rule_requires_match_end,
        "end_game_pot_deltas": state.end_pot_delta,
        "final_score_comparison": final_comparison,
        "server_han_fu_fans_payment_labels_used_as_calculation_inputs": False,
        "native_apply_event_scores_used_as_calculation_inputs": False,
    }
