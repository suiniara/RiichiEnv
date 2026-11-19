from __future__ import annotations

from riichienv._riichienv import kar_public_nagashi_possibilities, kar_public_response_mask

from .actions import tile37_to_base34
from .contracts import MJSOUL_RULESET, Ruleset
from .observation import PublicSnapshot


def public_rule_response_mask(
    snapshot: PublicSnapshot, candidate_discard: int, ruleset: Ruleset = MJSOUL_RULESET
) -> tuple[bool, ...]:
    """Adapt the frozen public whitelist to Rust's conservative rule query.

    Never pass the snapshot itself or an engine to the native query: either
    could expose a private hand, future events, or an actual response list.
    This boundary translates the complete action ID, not Mahjong predicates.
    """
    if (
        isinstance(candidate_discard, bool)
        or not isinstance(candidate_discard, int)
        or not 0 <= candidate_discard < 148
    ):
        raise ValueError("Response mask requires a complete discard action ID")
    if snapshot.pending_kan_actor is not None:
        raise ValueError("A pending kan is not a discard decision")
    return tuple(
        kar_public_response_mask(
            actor=snapshot.observer_player_id,
            candidate_tile=tile37_to_base34(candidate_discard // 4),
            declares_riichi=bool(candidate_discard % 2),
            discard_history=snapshot.discard_history,
            current_rivers=snapshot.current_rivers,
            meld_kinds=tuple(tuple(meld.kind for meld in melds) for melds in snapshot.melds),
            riichi_accepted=snapshot.riichi_accepted,
            remaining_draws=snapshot.remaining_draws,
            history_complete=snapshot.history_complete,
            chi_allowed=ruleset.chi_allowed,
            sanchaho_is_draw=ruleset.sanchaho_is_draw,
            four_winds_abort=ruleset.four_winds_abort,
            four_riichi_abort=ruleset.four_riichi_abort,
            four_kans_abort=ruleset.four_kans_abort,
            nagashi_mangan=ruleset.nagashi_mangan,
        )
    )


def _nagashi_flags(snapshot: PublicSnapshot, actor: int, tile: int) -> tuple[bool, ...]:
    return tuple(
        kar_public_nagashi_possibilities(
            actor=actor,
            candidate_tile=tile,
            discard_history=snapshot.discard_history,
            current_rivers=snapshot.current_rivers,
            history_complete=snapshot.history_complete,
        )
    )


def _nagashi_possible(snapshot: PublicSnapshot, actor: int, tile: int) -> bool:
    return any(_nagashi_flags(snapshot, actor, tile))


def _nagashi_player_possible(snapshot: PublicSnapshot, player: int, actor: int, tile: int) -> bool:
    return _nagashi_flags(snapshot, actor, tile)[player]


def validate_response_target(allowed: tuple[bool, ...], target: int) -> None:
    if len(allowed) != 21 or not any(allowed):
        raise ValueError("Expected a nonempty 21-category public response mask")
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < 21 or not allowed[target]:
        raise ValueError("Observed response is excluded by the public rule mask")
