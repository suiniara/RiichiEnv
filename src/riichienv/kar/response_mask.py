from __future__ import annotations

from .actions import tile37_to_base34
from .contracts import MJSOUL_RULESET, Response, Ruleset
from .observation import PublicSnapshot


def public_rule_response_mask(
    snapshot: PublicSnapshot, candidate_discard: int, ruleset: Ruleset = MJSOUL_RULESET
) -> tuple[bool, ...]:
    if (
        isinstance(candidate_discard, bool)
        or not isinstance(candidate_discard, int)
        or not 0 <= candidate_discard < 148
    ):
        raise ValueError("Response mask requires a complete discard action ID")
    if snapshot.pending_kan_actor is not None:
        raise ValueError("A pending kan is not a discard decision")
    tile = tile37_to_base34(candidate_discard // 4)
    actor = snapshot.observer_player_id
    allowed = [True] * 21
    allowed[Response.CHI_P1] = ruleset.chi_allowed and tile < 27
    allowed[Response.ABORT_TRIPLE_RON] = ruleset.sanchaho_is_draw
    if ruleset.sanchaho_is_draw:
        allowed[Response.RON_MASK_7] = False
    for relative in range(1, 4):
        player = (actor + relative) % 4
        if snapshot.riichi_accepted[player]:
            allowed[1 + relative] = False
            allowed[4 + relative] = False
            if relative == 1:
                allowed[Response.CHI_P1] = False
        if snapshot.history_complete and any(t // 4 == tile for t in snapshot.discard_history[player]):
            for ron_mask in range(1, 8):
                if ron_mask & (1 << (relative - 1)):
                    allowed[7 + ron_mask] = False
    counts = [sum(m.kind in {"ankan", "kakan", "daiminkan"} for m in melds) for melds in snapshot.melds]
    if sum(counts) >= 4:
        allowed[5:8] = [False] * 3
    if snapshot.remaining_draws is not None:
        allowed[Response.NEXT_DRAW] = snapshot.remaining_draws > 0
        allowed[Response.EXHAUSTIVE_DRAW] = snapshot.remaining_draws == 0
        if snapshot.remaining_draws == 0:
            allowed[5:8] = [False] * 3
    if snapshot.history_complete:
        no_calls = not any(snapshot.melds)
        others = [(actor + relative) % 4 for relative in range(1, 4)]
        four_winds = (
            no_calls
            and 27 <= tile <= 30
            and not snapshot.discard_history[actor]
            and all(
                len(snapshot.discard_history[p]) == 1 and snapshot.discard_history[p][0] // 4 == tile for p in others
            )
        )
        allowed[Response.ABORT_FOUR_WINDS] = ruleset.four_winds_abort and four_winds
        allowed[Response.ABORT_FOUR_RIICHI] = (
            ruleset.four_riichi_abort
            and bool(candidate_discard % 2)
            and all(snapshot.riichi_accepted[p] for p in others)
        )
        allowed[Response.ABORT_FOUR_KANS] = (
            ruleset.four_kans_abort and sum(counts) == 4 and sum(c > 0 for c in counts) > 1
        )
    else:
        allowed[Response.ABORT_FOUR_WINDS] = ruleset.four_winds_abort
        allowed[Response.ABORT_FOUR_RIICHI] = ruleset.four_riichi_abort
        allowed[Response.ABORT_FOUR_KANS] = ruleset.four_kans_abort
    allowed[Response.NAGASHI_MANGAN] = (
        ruleset.nagashi_mangan and _nagashi_possible(snapshot, actor, tile) and snapshot.remaining_draws in {None, 0}
    )
    if not any(allowed):
        raise ValueError("Public response mask excludes every category")
    return tuple(allowed)


def _nagashi_possible(snapshot: PublicSnapshot, actor: int, tile: int) -> bool:
    return any(_nagashi_player_possible(snapshot, player, actor, tile) for player in range(4))


def _nagashi_player_possible(snapshot: PublicSnapshot, player: int, actor: int, tile: int) -> bool:
    """Return whether known public facts still permit this player's nagashi."""
    # Only committed melds count; a pending declaration does not change this tuple.
    if snapshot.melds[player]:
        return False
    # A set suffices for existence, never for physical counts or called-away counts.
    known_discards = {t // 4 for t in snapshot.discard_history[player]}
    known_discards.update(t // 4 for t in snapshot.current_rivers[player])
    if player == actor:
        known_discards.add(tile)
    if any(t < 27 and t % 9 not in {0, 8} for t in known_discards):
        return False
    if not snapshot.history_complete:
        return True  # Missing past events do not invalidate the known public facts above.
    if len(snapshot.current_rivers[player]) != len(snapshot.discard_history[player]):
        return False
    return True


def validate_response_target(allowed: tuple[bool, ...], target: int) -> None:
    if len(allowed) != 21 or not any(allowed):
        raise ValueError("Expected a nonempty 21-category public response mask")
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < 21 or not allowed[target]:
        raise ValueError("Observed response is excluded by the public rule mask")
