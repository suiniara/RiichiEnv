from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from riichienv import Action, ActionType, HandEvaluator, Observation, Phase, RiichiEnv

ACTION_COUNT = 546
MAX_CANDIDATES = 32
REFERENCE_COMMIT = "28356fe50efab57c9f05aaf2ec076ef128c6ea1f"


@dataclass(frozen=True)
class ActionSpec:
    global_id: int
    action_type: str
    tile37: int | None = None
    base34: int | None = None
    tsumogiri: bool | None = None
    riichi: bool | None = None
    source: int | None = None
    consumed_tile37: tuple[int, ...] = ()


NativeKey = tuple[int, int, tuple[int, ...], int]


@dataclass(frozen=True)
class MappedAction:
    global_id: int
    spec: ActionSpec
    actions: tuple[Action, ...]
    _legal_signature: tuple[NativeKey, ...] = field(repr=False)
    native_execution_ambiguous: bool = False
    _action_signature: tuple[NativeKey, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_action_signature", tuple(map(_native_key, self.actions)))


class ActionExecution(dict[int, Observation]):
    def __init__(self, step_observations: tuple[dict[int, Observation], ...]) -> None:
        super().__init__(step_observations[-1])
        self.step_observations = step_observations


class AmbiguousDiscardError(ValueError):
    pass


def _integer(value: int, upper: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper:
        raise ValueError(f"{name} must be an integer in [0, {upper})")
    return value


def tile136_to_tile37(tile: int) -> int:
    tile = _integer(tile, 136, "tile136")
    if tile < 108:
        suit, offset = divmod(tile, 36)
        return suit * 10 if offset == 16 else suit * 10 + offset // 4 + 1
    return 30 + (tile - 108) // 4


def tile37_to_base34(tile: int) -> int:
    tile = _integer(tile, 37, "tile37")
    if tile >= 30:
        return tile - 3
    suit, rank = divmod(tile, 10)
    return suit * 9 + (4 if rank == 0 else rank - 1)


_CHI_PATTERNS = (
    (2, 3, 1),
    (1, 3, 2),
    (3, 4, 2),
    (1, 2, 3),
    (2, 4, 3),
    (4, 5, 3),
    (4, 0, 3),
    (2, 3, 4),
    (3, 5, 4),
    (3, 0, 4),
    (5, 6, 4),
    (0, 6, 4),
    (3, 4, 5),
    (3, 4, 0),
    (4, 6, 5),
    (4, 6, 0),
    (6, 7, 5),
    (6, 7, 0),
    (4, 5, 6),
    (4, 0, 6),
    (5, 7, 6),
    (0, 7, 6),
    (7, 8, 6),
    (5, 6, 7),
    (0, 6, 7),
    (6, 8, 7),
    (8, 9, 7),
    (6, 7, 8),
    (7, 9, 8),
    (7, 8, 9),
)


def _pon_patterns() -> tuple[tuple[int, int, int], ...]:
    patterns = []
    for suit in range(3):
        for rank in range(1, 10):
            normal = suit * 10 + rank
            patterns.append((normal, normal, normal))
            if rank == 5:
                red = suit * 10
                patterns.extend(((red, normal, normal), (normal, normal, red)))
    patterns.extend((tile, tile, tile) for tile in range(30, 37))
    return tuple(patterns)


_PON_PATTERNS = _pon_patterns()


def _build_manifest() -> tuple[ActionSpec, ...]:
    specs = []
    for tile in range(37):
        for tsumogiri in (False, True):
            for riichi in (False, True):
                specs.append(
                    ActionSpec(
                        len(specs),
                        "DISCARD",
                        tile,
                        tile37_to_base34(tile),
                        tsumogiri,
                        riichi,
                    )
                )
    specs.extend(ActionSpec(148 + tile, "ANKAN", base34=tile) for tile in range(34))
    specs.extend(ActionSpec(182 + tile, "KAKAN", tile, tile37_to_base34(tile)) for tile in range(37))
    specs.extend((ActionSpec(219, "TSUMO"), ActionSpec(220, "KYUSHU_KYUHAI"), ActionSpec(221, "PASS")))
    for suit in range(3):
        for a, b, claimed in _CHI_PATTERNS:
            tile = suit * 10 + claimed
            specs.append(
                ActionSpec(
                    len(specs),
                    "CHI",
                    tile,
                    tile37_to_base34(tile),
                    source=3,
                    consumed_tile37=tuple(sorted((suit * 10 + a, suit * 10 + b))),
                )
            )
    for source in range(1, 4):
        for a, b, claimed in _PON_PATTERNS:
            specs.append(
                ActionSpec(
                    len(specs),
                    "PON",
                    claimed,
                    tile37_to_base34(claimed),
                    source=source,
                    consumed_tile37=tuple(sorted((a, b))),
                )
            )
    for source in range(1, 4):
        for tile in range(37):
            specs.append(
                ActionSpec(
                    len(specs),
                    "DAIMINKAN",
                    tile,
                    tile37_to_base34(tile),
                    source=source,
                )
            )
    specs.extend(ActionSpec(542 + source, "RON", source=source) for source in range(1, 4))
    return tuple(specs)


ACTION_MANIFEST = _build_manifest()
ACTION_MANIFEST_SHA256 = hashlib.sha256(
    json.dumps(
        [asdict(spec) for spec in ACTION_MANIFEST],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def manifest() -> tuple[ActionSpec, ...]:
    return ACTION_MANIFEST


def manifest_hash() -> str:
    return ACTION_MANIFEST_SHA256


def action_spec(global_id: int) -> ActionSpec:
    return ACTION_MANIFEST[_integer(global_id, ACTION_COUNT, "global_id")]


def _semantic_key(spec: ActionSpec) -> tuple:
    return (
        spec.action_type,
        spec.tile37,
        spec.base34,
        spec.tsumogiri,
        spec.riichi,
        spec.source,
        spec.consumed_tile37,
    )


_SPEC_IDS = {_semantic_key(spec): spec.global_id for spec in ACTION_MANIFEST}


def action_id(spec: ActionSpec) -> int:
    try:
        global_id = _SPEC_IDS[_semantic_key(spec)]
    except KeyError as error:
        raise ValueError("Action semantics are not in the frozen 546-action manifest") from error
    if spec.global_id != global_id:
        raise ValueError("global_id disagrees with its action semantics")
    return global_id


def _native_key(action: Action) -> NativeKey:
    return (
        int(action.action_type),
        -1 if action.tile is None else action.tile,
        tuple(sorted(action.consume_tiles)),
        -1 if action.actor is None else action.actor,
    )


def _copy_action(action: Action) -> Action:
    return Action(action.action_type, action.tile, list(action.consume_tiles), action.actor)


def _source(player_id: int, source_player: int | None) -> int:
    if source_player is None:
        raise ValueError("The public source_player is required for a claim action")
    source_player = _integer(source_player, 4, "source_player")
    relative = (source_player - player_id) % 4
    if relative == 0:
        raise ValueError("A claim source cannot be the observing player")
    return relative


def _claim_id(action: Action, player_id: int, source_player: int | None) -> int:
    kind = action.action_type
    if kind == ActionType.RON:
        return 542 + _source(player_id, source_player)
    if action.tile is None:
        raise ValueError("A tile-bearing action is missing its tile")
    tile = tile136_to_tile37(action.tile)
    if kind == ActionType.KAKAN:
        return 182 + tile
    if kind == ActionType.ANKAN:
        if len(action.consume_tiles) != 4 or any(t // 4 != action.tile // 4 for t in action.consume_tiles):
            raise ValueError("ANKAN requires four consumed tiles of one base34 type")
        return 148 + action.tile // 4
    source = _source(player_id, source_player)
    if kind == ActionType.DAIMINKAN:
        if len(action.consume_tiles) != 3 or any(t // 4 != action.tile // 4 for t in action.consume_tiles):
            raise ValueError("DAIMINKAN requires three consumed tiles of the called type")
        return 432 + 37 * (source - 1) + tile
    consumed = tuple(sorted(tile136_to_tile37(t) for t in action.consume_tiles))
    family = "CHI" if kind == ActionType.CHI else "PON"
    if kind not in (ActionType.CHI, ActionType.PON):
        raise ValueError(f"Unsupported four-player action type: {kind}")
    if kind == ActionType.CHI and source != 3:
        raise ValueError("CHI can only claim the previous player's discard")
    key = (family, tile, tile37_to_base34(tile), None, None, source, consumed)
    try:
        return _SPEC_IDS[key]
    except KeyError as error:
        raise ValueError(f"Invalid {family} tile combination") from error


def _discard_variants(
    action: Action,
    hand: list[int],
    drawn_tile: int | None,
    *,
    forced_tedashi: bool,
    declared: bool,
) -> tuple[tuple[int, bool], ...]:
    if action.tile is None or action.tile not in hand:
        raise ValueError("A legal discard must identify a tile in the observer's hand")
    if declared and action.tile != drawn_tile:
        raise ValueError("Post-riichi discard is not the drawn tile")
    tile = tile136_to_tile37(action.tile)
    if not declared and not forced_tedashi and drawn_tile == action.tile and hand.count(action.tile) > 1:
        return ((4 * tile, True), (4 * tile + 2, True))
    tsumogiri = not forced_tedashi and drawn_tile is not None and action.tile == drawn_tile
    return ((4 * tile + 2 * int(tsumogiri), False),)


def legal_candidates(
    observation: Observation,
    *,
    drawn_tile: int | None = None,
    source_player: int | None = None,
) -> tuple[MappedAction, ...]:
    player_id = _integer(observation.player_id, 4, "player_id")
    if len(observation.hands) != 4 or len(observation.melds) != 4:
        raise ValueError("KAR actions require a four-player observation")
    hand = list(observation.hand)
    if drawn_tile is None:
        drawn_tile = getattr(observation, "drawn_tile", None)
    if drawn_tile is not None:
        _integer(drawn_tile, 136, "drawn_tile")
    forced_tedashi = observation.forced_tedashi
    declared = observation.riichi_declared[player_id]
    native = sorted(observation.legal_actions(), key=_native_key)
    if drawn_tile is not None and any(a.action_type == ActionType.DISCARD for a in native) and drawn_tile not in hand:
        raise ValueError("The drawn discard context is not in the observer's hand")
    signature = tuple(_native_key(action) for action in native)
    reach = next((action for action in native if action.action_type == ActionType.RIICHI), None)
    mapped: dict[int, MappedAction] = {}

    def add(global_id: int, actions: tuple[Action, ...], *, ambiguous: bool = False) -> None:
        candidate = MappedAction(
            global_id,
            action_spec(global_id),
            tuple(_copy_action(a) for a in actions),
            signature,
            ambiguous,
        )
        previous = mapped.get(global_id)
        choice_key = (ambiguous, tuple(map(_native_key, candidate.actions)))
        if previous is None or choice_key < (
            previous.native_execution_ambiguous,
            tuple(map(_native_key, previous.actions)),
        ):
            mapped[global_id] = candidate

    for action in native:
        if action.actor is not None and action.actor != player_id:
            raise ValueError("Legal action actor does not match the observer")
        kind = action.action_type
        if kind == ActionType.RIICHI:
            continue
        if kind == ActionType.DISCARD:
            variants = _discard_variants(
                action,
                hand,
                drawn_tile,
                forced_tedashi=forced_tedashi,
                declared=declared,
            )
            can_reach = False
            if reach is not None:
                assert action.tile is not None
                remaining = list(hand)
                remaining.remove(action.tile)
                can_reach = HandEvaluator(remaining, list(observation.melds[player_id])).is_tenpai()
            for global_id, ambiguous in variants:
                add(global_id, (action,), ambiguous=ambiguous)
                if can_reach and reach is not None:
                    add(global_id + 1, (reach, action), ambiguous=ambiguous)
        elif kind == ActionType.TSUMO:
            add(219, (action,))
        elif kind == ActionType.KYUSHU_KYUHAI:
            add(220, (action,))
        elif kind == ActionType.PASS:
            add(221, (action,))
        else:
            add(_claim_id(action, player_id, source_player), (action,))
    if len(mapped) > MAX_CANDIDATES:
        raise ValueError(f"Complete legal candidate count {len(mapped)} exceeds {MAX_CANDIDATES}")
    return tuple(mapped[global_id] for global_id in sorted(mapped))


def execute_joint_responses(env: RiichiEnv, choices: dict[int, MappedAction]) -> ActionExecution:
    if env.phase != Phase.WaitResponse or set(choices) != set(env.active_players):
        raise ValueError("All active response choices must be submitted as one public window")
    native = {}
    for seat, mapped in choices.items():
        _integer(seat, 4, "player_id")
        if (
            mapped.spec != action_spec(mapped.global_id)
            or len(mapped.actions) != 1
            or mapped.native_execution_ambiguous
        ):
            raise ValueError("A joint response must have one unambiguous complete action")
        if mapped.spec.action_type not in {"CHI", "PON", "DAIMINKAN", "RON", "PASS"}:
            raise ValueError("Non-response action cannot enter a joint response window")
        current = tuple(sorted(_native_key(action) for action in env._get_legal_actions(seat)))
        if current != mapped._legal_signature or tuple(map(_native_key, mapped.actions)) != mapped._action_signature:
            raise ValueError("A joint response is stale or modified")
        action = mapped.actions[0]
        if _native_key(action) not in current or action.actor not in (None, seat):
            raise ValueError("Joint response does not belong to its legal observer")
        native[seat] = action
    return ActionExecution((env.step(native),))


def execute_candidate(env: RiichiEnv, player_id: int, mapped: MappedAction) -> ActionExecution:
    player_id = _integer(player_id, 4, "player_id")
    if mapped.spec != action_spec(mapped.global_id) or not mapped.actions:
        raise ValueError("Mapped action disagrees with the frozen manifest")
    if mapped.native_execution_ambiguous:
        raise AmbiguousDiscardError("Canonical duplicate draw IDs cannot execute the chosen discard provenance")
    if tuple(map(_native_key, mapped.actions)) != mapped._action_signature:
        raise ValueError("Mapped native actions were modified after projection")
    current = tuple(sorted(_native_key(action) for action in env._get_legal_actions(player_id)))
    if current != mapped._legal_signature:
        raise ValueError("Mapped action is stale for the current legal action set")
    for action in mapped.actions:
        if action.actor is not None and action.actor != player_id:
            raise ValueError("Mapped action belongs to another player")
    if mapped.spec.riichi:
        if len(mapped.actions) != 2 or mapped.actions[0].action_type != ActionType.RIICHI:
            raise ValueError("A riichi discard must contain reach followed by discard")
        discard = mapped.actions[1]
        if (
            discard.action_type != ActionType.DISCARD
            or discard.tile is None
            or tile136_to_tile37(discard.tile) != mapped.spec.tile37
        ):
            raise ValueError("Riichi discard disagrees with the mapped tile")
        if any(_native_key(action) not in current for action in mapped.actions):
            raise ValueError("Riichi sequence is not currently legal")
        # Reach consumes an intermediate observation delta that must remain available to the caller.
        reach_observations = env.step({player_id: mapped.actions[0]})
        discard_observations = env.step({player_id: discard})
        return ActionExecution((reach_observations, discard_observations))
    if len(mapped.actions) != 1 or _native_key(mapped.actions[0]) not in current:
        raise ValueError("Mapped native action is not currently legal")
    return ActionExecution((env.step({player_id: mapped.actions[0]}),))
