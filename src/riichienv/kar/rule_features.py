from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from numbers import Integral
from typing import cast

RYANMEN_PAIRS_BY_RANK: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((2, 3),),
    2: ((3, 4),),
    3: ((4, 5),),
    4: ((2, 3), (5, 6)),
    5: ((3, 4), (6, 7)),
    6: ((4, 5), (7, 8)),
    7: ((5, 6),),
    8: ((6, 7),),
    9: ((7, 8),),
}

OTHER_WAITS_BY_RANK: dict[int, tuple[int, ...]] = {
    1: (4,),
    2: (5,),
    3: (6,),
    4: (1, 7),
    5: (2, 8),
    6: (3, 9),
    7: (4,),
    8: (5,),
    9: (6,),
}


@dataclass(frozen=True, slots=True)
class RankFeatures:
    rank: int
    gap_to_higher: float
    gap_to_higher_present: bool
    gap_to_lower: float
    gap_to_lower_present: bool


def _validate_base34(tile: int) -> None:
    if isinstance(tile, bool) or not isinstance(tile, Integral) or not 0 <= tile < 34:
        raise ValueError("base34 tile must be an integer from 0 to 33")


def _validated_history(discard_history: Iterable[int]) -> frozenset[int]:
    history = frozenset(discard_history)
    for tile in history:
        _validate_base34(tile)
    return history


def _validated_counts(counts: Sequence[int]) -> tuple[int, ...]:
    if len(counts) != 34:
        raise ValueError("known counts must contain exactly 34 entries")
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, Integral) or not 0 <= count <= 4:
            raise ValueError("known counts must be integers from 0 to 4")
    return tuple(counts)


def genbutsu(target_base34: int, discard_history: set[int] | None) -> bool | None:
    _validate_base34(target_base34)
    if discard_history is None:
        return None
    return target_base34 in _validated_history(discard_history)


def full_suji_exclusion(target_base34: int, own_discard_history_base34: set[int] | None) -> bool | None:
    _validate_base34(target_base34)
    if own_discard_history_base34 is None:
        return None
    history = _validated_history(own_discard_history_base34)
    if target_base34 >= 27:
        return None
    suit_start = target_base34 // 9 * 9
    rank = target_base34 % 9 + 1
    return all(suit_start + wait - 1 in history for wait in OTHER_WAITS_BY_RANK[rank])


def no_visible_possible_ryanmen_shape(target_base34: int, known_count_base34: Sequence[int] | None) -> bool | None:
    _validate_base34(target_base34)
    if known_count_base34 is None:
        return None
    counts = _validated_counts(known_count_base34)
    if target_base34 >= 27:
        return None
    if counts[target_base34] == 4:
        return True
    suit_start = target_base34 // 9 * 9
    rank = target_base34 % 9 + 1
    return all(
        counts[suit_start + left - 1] == 4 or counts[suit_start + right - 1] == 4
        for left, right in RYANMEN_PAIRS_BY_RANK[rank]
    )


def known_count_base34(
    observer_hand: Iterable[int],
    current_rivers: Iterable[Iterable[int]],
    current_melds: Iterable[Iterable[int]],
    dora_indicators: Iterable[int],
    other_public_reveals: Iterable[int] = (),
) -> tuple[int, ...]:
    counts = [0] * 34
    regions = chain((observer_hand,), current_rivers, current_melds, (dora_indicators, other_public_reveals))
    # Regions must be disjoint; repeated base34 values can be distinct physical copies.
    for region in regions:
        for tile in region:
            _validate_base34(tile)
            counts[tile] += 1
            if counts[tile] > 4:
                raise ValueError(f"more than four visible copies of base34 tile {tile}")
    return tuple(counts)


def public_dora_types(indicators_base34: Iterable[int]) -> frozenset[int]:
    dora = set()
    for tile in indicators_base34:
        _validate_base34(tile)
        if tile < 27:
            suit_start = tile // 9 * 9
            dora.add(suit_start + (tile % 9 + 1) % 9)
        elif tile < 31:
            dora.add(27 + (tile - 27 + 1) % 4)
        else:
            dora.add(31 + (tile - 31 + 1) % 3)
    return frozenset(dora)


def current_rank_features(
    scores_by_player: Mapping[Hashable, int] | Sequence[int],
    initial_seat_order_player_ids: Sequence[Hashable],
    observer_player_id: Hashable,
) -> RankFeatures:
    scores: dict[Hashable, int] = {}
    if isinstance(scores_by_player, Mapping):
        for player_id, score in cast(Mapping[Hashable, int], scores_by_player).items():
            scores[player_id] = score
    else:
        for player_id, score in enumerate(scores_by_player):
            scores[player_id] = score
    initial_order = tuple(initial_seat_order_player_ids)
    if len(initial_order) != 4 or len(set(initial_order)) != 4:
        raise ValueError("initial seat order must contain four distinct player IDs")
    if set(scores) != set(initial_order):
        raise ValueError("scores and initial seat order must identify the same four players")
    if observer_player_id not in scores:
        raise ValueError("observer must be one of the four players")
    if any(isinstance(score, bool) or not isinstance(score, Integral) for score in scores.values()):
        raise ValueError("scores must be raw integer points")

    initial_indices = {player_id: index for index, player_id in enumerate(initial_order)}
    ranking = sorted(initial_order, key=lambda player_id: (-scores[player_id], initial_indices[player_id]))
    position = ranking.index(observer_player_id)
    own_score = scores[observer_player_id]
    higher_present = position > 0
    lower_present = position < 3
    higher_gap = (scores[ranking[position - 1]] - own_score) / 50000 if higher_present else 0.0
    lower_gap = (own_score - scores[ranking[position + 1]]) / 50000 if lower_present else 0.0
    return RankFeatures(position + 1, higher_gap, higher_present, lower_gap, lower_present)
