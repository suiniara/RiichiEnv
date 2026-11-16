from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from riichienv import WinResult, calculate_score


@dataclass(frozen=True)
class EvaluatedWin:
    player_id: int
    result: WinResult
    liable_player: int | None = None
    liable_yakuman_units: int = 0


def _seat(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 4:
        raise ValueError(f"{name} must be a four-player seat index")


def _counter(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_win(win: EvaluatedWin, *, oya: int, tsumo: bool) -> int:
    _seat(win.player_id, "winner")
    _counter(win.liable_yakuman_units, "liable_yakuman_units")
    result = win.result
    if not isinstance(result, WinResult) or not result.is_win:
        raise ValueError("A winner requires an independently evaluated winning result")
    if not 1 <= result.han <= 255 or not 0 <= result.fu <= 255:
        raise ValueError("Invalid independently evaluated han/fu")
    if result.yakuman and result.han % 13:
        raise ValueError("Yakuman han must identify a whole number of yakuman units")
    scoring_han = result.han if result.yakuman else min(result.han, 13)
    base = calculate_score(scoring_han, result.fu, win.player_id == oya, tsumo, 0, 4)
    actual = (result.ron_agari, result.tsumo_agari_oya, result.tsumo_agari_ko)
    expected = (base.pay_ron, base.pay_tsumo_oya, base.pay_tsumo_ko)
    if actual != expected or base.total == 0:
        raise ValueError("Result must be a zero-honba base evaluation for this win role")
    total_units = result.han // 13 if result.yakuman else 0
    if win.liable_player is None:
        if win.liable_yakuman_units:
            raise ValueError("Responsible yakuman units require a liable player")
        return total_units
    _seat(win.liable_player, "liable_player")
    if win.liable_player == win.player_id:
        raise ValueError("A winner cannot be their own liable payer")
    if not result.yakuman or not 0 < win.liable_yakuman_units <= total_units:
        raise ValueError("Liability requires supported units within a true yakuman result")
    possible_units = {0}
    for yaku, units in ((37, 1), (50, 2)):
        if yaku in result.yaku:
            possible_units |= {count + units for count in possible_units}
    if win.liable_yakuman_units not in possible_units:
        raise ValueError("Liable units disagree with the MjSoul responsibility yakuman")
    return total_units


def win_payments(
    wins: Sequence[EvaluatedWin],
    *,
    oya: int,
    target: int | None,
    honba: int,
    riichi_sticks: int,
) -> tuple[int, int, int, int]:
    _seat(oya, "oya")
    _counter(honba, "honba")
    _counter(riichi_sticks, "riichi_sticks")
    if not wins or len(wins) > 3:
        raise ValueError("A win requires one to three distinct winners")
    if len({win.player_id for win in wins}) != len(wins):
        raise ValueError("Duplicate winner")
    if target is None and len(wins) != 1:
        raise ValueError("A self-draw must have exactly one winner")
    if target is not None:
        _seat(target, "target")
        if any(win.player_id == target for win in wins):
            raise ValueError("The discard target cannot also be a ron winner")
    units = {win.player_id: _validate_win(win, oya=oya, tsumo=target is None) for win in wins}
    deltas = [0, 0, 0, 0]
    if target is None:
        win = wins[0]
        winner = win.player_id
        if win.liable_player is not None:
            payment = win.liable_yakuman_units * (48000 if winner == oya else 32000) + honba * 300
            deltas[win.liable_player] -= payment
            deltas[winner] += payment
            remaining_units = units[winner] - win.liable_yakuman_units
            for payer in range(4):
                if payer != winner:
                    payment = remaining_units * (16000 if oya in (winner, payer) else 8000)
                    deltas[payer] -= payment
                    deltas[winner] += payment
        else:
            for payer in range(4):
                if payer != winner:
                    base = win.result.tsumo_agari_oya if payer == oya else win.result.tsumo_agari_ko
                    payment = base + honba * 100
                    deltas[payer] -= payment
                    deltas[winner] += payment
        deltas[winner] += riichi_sticks * 1000
    else:
        ordered = sorted(wins, key=lambda win: (win.player_id - target) % 4)
        for index, win in enumerate(ordered):
            bonus = honba * 300 if index == 0 else 0
            payment = win.result.ron_agari + bonus
            liability = 0
            if win.liable_player is not None:
                liability = win.liable_yakuman_units * (48000 if win.player_id == oya else 32000) // 2 + bonus
                deltas[win.liable_player] -= liability
            deltas[target] -= payment - liability
            deltas[win.player_id] += payment
            if index == 0:
                deltas[win.player_id] += riichi_sticks * 1000
    assert sum(deltas) == riichi_sticks * 1000
    return deltas[0], deltas[1], deltas[2], deltas[3]


def draw_payments(
    tenpai: Sequence[bool],
    *,
    oya: int,
    nagashi_winners: Sequence[int] = (),
    abortive: bool = False,
) -> tuple[int, int, int, int]:
    _seat(oya, "oya")
    if len(tenpai) != 4 or any(not isinstance(value, bool) for value in tenpai):
        raise ValueError("Tenpai status must contain exactly four booleans")
    if not isinstance(abortive, bool):
        raise ValueError("abortive must be a boolean")
    if len(set(nagashi_winners)) != len(nagashi_winners):
        raise ValueError("Duplicate nagashi winner")
    for winner in nagashi_winners:
        _seat(winner, "nagashi winner")
    if abortive and nagashi_winners:
        raise ValueError("Abortive draws cannot also settle nagashi mangan")
    deltas = [0, 0, 0, 0]
    if not abortive:
        if nagashi_winners:
            for winner in nagashi_winners:
                base = calculate_score(5, 30, winner == oya, True, 0, 4)
                for payer in range(4):
                    if payer != winner:
                        payment = base.pay_tsumo_oya if payer == oya else base.pay_tsumo_ko
                        deltas[payer] -= payment
                        deltas[winner] += payment
        else:
            count = sum(tenpai)
            if 0 < count < 4:
                deltas = [3000 // count if ready else -3000 // (4 - count) for ready in tenpai]
    assert sum(deltas) == 0
    return deltas[0], deltas[1], deltas[2], deltas[3]


def end_game_pot(
    scores: Sequence[int], riichi_sticks: int, initial_seat_order: Sequence[int]
) -> tuple[int, int, int, int]:
    _counter(riichi_sticks, "riichi_sticks")
    if len(scores) != 4 or any(isinstance(score, bool) or not isinstance(score, int) for score in scores):
        raise ValueError("Scores must contain exactly four raw integer values")
    if len(initial_seat_order) != 4 or len(set(initial_seat_order)) != 4:
        raise ValueError("Initial seat order must identify four distinct players")
    for seat in initial_seat_order:
        _seat(seat, "initial seat")
    top = min(initial_seat_order, key=lambda seat: -scores[seat])
    deltas = [0, 0, 0, 0]
    deltas[top] = riichi_sticks * 1000
    return deltas[0], deltas[1], deltas[2], deltas[3]
