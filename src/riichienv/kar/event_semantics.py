"""Source-independent public semantic time and verified terminal disclosures.

Source record IDs, server packaging and native callback boundaries are audit data,
not model time. No method here consults a future event or another player's hand.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from riichienv import convert

PUBLIC_CLOCK_VERSION = "public_semantic_clock_v1"
TERMINAL_PUBLIC_VERSION = "verified_abort_public_resolution_v1"
ABORT_REASONS = frozenset({"kyushukyuhai", "four_winds", "four_riichi", "four_kans", "sanchaho"})


class PublicEventClock:
    """A causal streaming clock for canonical public events.

    Each semantic event advances once; token expansion does not advance time.
    Consecutive wins on one trigger form one simultaneous resolution. Optional
    kan_commit transport acknowledgements carry state evidence, but no model
    history tick; history acceptance is anchored at common dora/draw evidence.
    Private Pass uses its confirmation anchor, never a new tick.
    """

    def __init__(self) -> None:
        self.position = 0
        self._win_trigger: tuple[Any, ...] | None = None

    def advance(self, event: Mapping[str, Any]) -> int:
        kind = event["type"]
        if kind in {"self_pass", "kan_commit"}:
            return self.position
        trigger = (event.get("target"), event.get("pai")) if kind == "hora" else None
        same_resolution = kind == "hora" and trigger == self._win_trigger
        if not same_resolution:
            self.position += 1
        self._win_trigger = trigger
        return self.position


def public_history_positions(events: Sequence[str]) -> tuple[int, ...]:
    clock = PublicEventClock()
    return tuple(clock.advance(json.loads(event)) for event in events)


def nine_terminals_public_hand(
    actor: int,
    hand: Sequence[str],
    *,
    current_draw_actor: int | None,
    draw_count: int,
    has_discarded: bool,
    calls_seen: bool,
    source_tiles: Sequence[str] | None = None,
) -> list[str]:
    """Reveal only the verified declaring hand, after the public abort.

    RecordLiuJu.tiles is checked when present; absent tiles may be reconstructed
    from the validated preceding owner events under the frozen public-reveal rule.
    This never authorizes publishing anyone else's concealed initial hand.
    """
    if type(actor) is not int or actor not in range(4) or actor != current_draw_actor:
        raise ValueError("Nine-terminals declaring actor differs from the current draw")
    if len(hand) != 14 or draw_count != 1 or has_discarded or calls_seen or "?" in hand:
        raise ValueError("Nine-terminals needs the verified first, uninterrupted 14-tile hand")
    bases = Counter(convert.mjai_to_tid(tile) // 4 for tile in hand)
    if any(count > 4 for count in bases.values()) or sum(base >= 27 or base % 9 in {0, 8} for base in bases) < 9:
        raise ValueError("Nine-terminals hand does not contain nine distinct terminal/honor types")
    if source_tiles and Counter(source_tiles) != Counter(hand):
        raise ValueError("Nine-terminals source hand differs from the verified declaring hand")
    return sorted(hand, key=convert.mjai_to_tid)
