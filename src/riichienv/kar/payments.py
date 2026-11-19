from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from riichienv import WinResult
from riichienv import _riichienv as _native


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
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be a nonnegative u32 integer")


def win_payments(
    wins: Sequence[EvaluatedWin],
    *,
    oya: int,
    target: int | None,
    honba: int,
    riichi_sticks: int,
) -> tuple[int, int, int, int]:
    """Adapt audited base evaluations to the online engine's settlement kernel."""
    _seat(oya, "oya")
    _counter(honba, "honba")
    _counter(riichi_sticks, "riichi_sticks")
    if target is not None:
        _seat(target, "target")
    inputs = []
    for win in wins:
        if not isinstance(win, EvaluatedWin) or not isinstance(win.result, WinResult):
            raise ValueError("A winner requires an independently evaluated winning result")
        _seat(win.player_id, "winner")
        _counter(win.liable_yakuman_units, "liable_yakuman_units")
        if win.liable_player is not None:
            _seat(win.liable_player, "liable_player")
        inputs.append((win.player_id, win.result, win.liable_player, win.liable_yakuman_units))
    east, south, west, north = _native.native_win_payments(inputs, oya, target, honba, riichi_sticks)
    return east, south, west, north


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
    for winner in nagashi_winners:
        _seat(winner, "nagashi winner")
    east, south, west, north = _native.native_draw_payments(list(tenpai), oya, list(nagashi_winners), abortive)
    return east, south, west, north


def end_game_pot(
    scores: Sequence[int], riichi_sticks: int, initial_seat_order: Sequence[int]
) -> tuple[int, int, int, int]:
    _counter(riichi_sticks, "riichi_sticks")
    if len(scores) != 4 or any(
        isinstance(score, bool) or not isinstance(score, int) or not -(2**31) <= score < 2**31 for score in scores
    ):
        raise ValueError("Scores must contain exactly four raw i32 integer values")
    if len(initial_seat_order) != 4:
        raise ValueError("Initial seat order must identify four distinct players")
    for seat in initial_seat_order:
        _seat(seat, "initial seat")
    east, south, west, north = _native.native_end_game_pot(list(scores), riichi_sticks, list(initial_seat_order))
    return east, south, west, north
