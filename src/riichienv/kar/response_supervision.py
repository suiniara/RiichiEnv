"""Recover only responses determined by a frozen offer and its public result."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .actions import action_spec
from .decision import ReplayOpportunity
from .observation import (
    PASS_CALL_PRIORITY,
    PASS_KAN_REPLACEMENT,
    PASS_NEXT_DRAW,
    PASS_TERMINAL_NO_RON,
    ObserverLedger,
    SelfPassTrigger,
)
from .response import draw_response

RESPONSE_SUPERVISION_VERSION = "frozen_response_opportunities_v1"
PASS_ID = 221
# Only use *strict* priority. No guess at same-priority arbitration or
# at a lower call that could have been submitted but suppressed.
_CALL_PRIORITY = {"chi": 1, "pon": 2, "daiminkan": 2}
_ACTION_PRIORITY = {"CHI": 1, "PON": 2, "DAIMINKAN": 2, "RON": 3}
_COUNTS = ("total", "explicit", "uniquely_inferred", "censored")


def unique_pass_provenance(candidates: tuple[int, ...], closure: str) -> str | None:
    if PASS_ID not in candidates or len(candidates) <= 1:
        return None
    alternatives = {action_spec(action).action_type for action in candidates if action != PASS_ID}
    if not alternatives <= _ACTION_PRIORITY.keys():
        raise ValueError("Non-response action in a frozen response opportunity")
    if closure == "next_draw":
        return PASS_NEXT_DRAW
    if closure == "replacement_draw" and alternatives == {"RON"}:
        return PASS_KAN_REPLACEMENT
    if closure in _CALL_PRIORITY and all(_ACTION_PRIORITY[kind] > _CALL_PRIORITY[closure] for kind in alternatives):
        return PASS_CALL_PRIORITY
    if (closure == "all_ron_results" or closure.startswith("no_ron_draw/")) and alternatives == {"RON"}:
        return PASS_TERMINAL_NO_RON
    return None


@dataclass(frozen=True)
class ResponseOffer:
    choice: ReplayOpportunity
    trigger: SelfPassTrigger


@dataclass
class ResponseWindow:
    phase: str
    source: int
    trigger_event_id: int
    offers: dict[int, ResponseOffer]
    explicit: set[int] = field(default_factory=set)
    ron_seen: bool = False

    @classmethod
    def open(
        cls,
        ledgers: Sequence[ObserverLedger],
        choices: Mapping[int, ReplayOpportunity],
    ) -> ResponseWindow | None:
        if not choices:
            return None
        offers = {seat: ResponseOffer(choices[seat], ledgers[seat].pass_trigger()) for seat in sorted(choices)}
        keys = {
            (item.choice.snapshot.decision_phase, item.trigger.target, item.trigger.trigger_event_id)
            for item in offers.values()
        }
        if len(keys) != 1:
            raise ValueError("Frozen responders disagree on their public window")
        phase, source, trigger = keys.pop()
        if phase not in {"discard_response", "kan_response"} or any(
            PASS_ID not in item.choice.candidates or item.choice.snapshot.cutoff_event_id < trigger
            for item in offers.values()
        ):
            raise ValueError("Frozen response requires Pass and its public trigger")
        return cls(phase, source, trigger, offers)

    def select_explicit(self, actor: int, *, ron: bool) -> ResponseOffer:
        if actor not in self.offers or actor in self.explicit or (self.ron_seen and not ron):
            raise ValueError("Observed response does not match an unused frozen native offer")
        return self.offers[actor]

    def mark_explicit(self, actor: int, *, ron: bool) -> None:
        self.explicit.add(actor)
        self.ron_seen |= ron

    def closure(self, event: Mapping[str, Any], *, all_ron_accepted: bool) -> str | None:  # noqa: PLR0911
        typ = event["type"]
        if typ == "tsumo":
            kan = self.phase == "kan_response"
            if self.ron_seen or event["actor"] != (self.source if kan else (self.source + 1) % 4):
                raise ValueError("Draw does not close this frozen response window")
            return "replacement_draw" if kan else "next_draw"
        if typ in {"chi", "pon", "daiminkan", "kan"}:
            if self.phase != "discard_response" or event["target"] != self.source or self.ron_seen:
                raise ValueError("Call does not close this frozen response window")
            return "daiminkan" if typ == "kan" else typ
        if typ == "ryukyoku":
            reason = event.get("reason")
            response = draw_response(reason)
            if self.phase == "discard_response" and response is not None and response.name != "ABORT_TRIPLE_RON":
                return "no_ron_draw/" + response.name.lower()
            return "draw/other"
        if typ in {"end_kyoku", "end_game"}:
            if self.ron_seen:
                return "all_ron_results" if all_ron_accepted else "ron_results"
            return "unresolved"
        return None


@dataclass
class OpportunityCounts:
    by_phase_and_close: dict[str, dict[str, Counter[str]]] = field(default_factory=dict)

    def observe(self, phase: str, closure: str, status: str) -> None:
        if status not in _COUNTS[1:]:
            raise ValueError("Unknown response supervision status")
        counts = self.by_phase_and_close.setdefault(phase, {}).setdefault(closure, Counter())
        counts[status] += 1
        counts["total"] += 1

    def merge(self, record: Mapping[str, Any]) -> None:
        if record.get("version") != RESPONSE_SUPERVISION_VERSION:
            raise ValueError("Response opportunity report version differs")
        for phase, outcomes in record["by_phase_and_close"].items():
            for closure, counts in outcomes.items():
                if (
                    set(counts) != set(_COUNTS)
                    or any(type(n) is not int or n < 0 for n in counts.values())
                    or (counts["total"] != sum(counts[key] for key in _COUNTS[1:]))
                ):
                    raise ValueError("Invalid response opportunity counts")
                self.by_phase_and_close.setdefault(phase, {}).setdefault(closure, Counter()).update(counts)

    def report(self) -> dict[str, Any]:
        slices = {
            phase: {close: {key: counts[key] for key in _COUNTS} for close, counts in sorted(outcomes.items())}
            for phase, outcomes in sorted(self.by_phase_and_close.items())
        }
        totals = {
            key: sum(counts[key] for outcomes in slices.values() for counts in outcomes.values()) for key in _COUNTS
        }
        return {"version": RESPONSE_SUPERVISION_VERSION, **totals, "by_phase_and_close": slices}
