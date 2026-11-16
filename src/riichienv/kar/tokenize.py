from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from riichienv import convert

from .actions import action_spec, tile136_to_tile37
from .contracts import MAX_CANDIDATES
from .kan_transactions import COMMITTED_KAN_KINDS, pending_kan_commits
from .observation import PublicSnapshot, extract_rule_features
from .schema import ACTIVE_SCHEMA_ID, encode_token, load_schema, schema_hash, token_defaults


class TokenizationUnavailableError(ValueError):
    pass


@lru_cache(maxsize=1)
def _active_hash() -> str:
    return schema_hash(load_schema())


def _relative(player: int, observer: int) -> int:
    return (player - observer) % 4 + 1


def _tile(value: str) -> int:
    return tile136_to_tile37(convert.mjai_to_tid(value)) + 1


def _representative(base: int) -> int:
    return (base + 3 if base >= 27 else base // 9 * 10 + base % 9 + 1) + 1


def _flag(value: bool | None) -> int:
    return 0 if value is None else 2 if value else 1


def _numeric(row: dict[str, Any], index: int, value: float | int | None, divisor: int) -> None:
    if value is not None:
        row["numeric"][index] = float(value) / divisor
        row["numeric_present"][index] = True


def _row(kind: int, position: int, block: int = 0, **fields: Any) -> dict[str, Any]:
    row = token_defaults(kind, block=block, position=position)
    row.update(fields)
    return row


def _player_row(
    actor: int,
    observer: int,
    position: int,
    stage: int,
    points: Sequence[int],
    initial_order: Sequence[int],
    dealer: int,
    block: int,
) -> dict[str, Any]:
    row = _row(3, position, block, actor=_relative(actor, observer), stage=stage, wind=(actor - dealer) % 4 + 1)
    _numeric(row, 0, points[actor], 50000)
    _numeric(row, 2, points[actor] - points[observer], 50000)
    _numeric(row, 7, initial_order.index(actor), 3)
    row["flags"][2] = _flag(actor == dealer)
    return row


@dataclass(frozen=True, init=False)
class _EncodedToken(Mapping[str, Any]):
    _values: Mapping[str, Any] = field(repr=False)
    blob: bytes = field(repr=False)
    schema_hash: str

    def __init__(self, row: Mapping[str, Any], bound_hash: str):
        values = {}
        for key, source_value in row.items():
            if isinstance(source_value, Sequence) and not isinstance(source_value, (str, bytes, bytearray)):
                value = tuple(source_value)
                if any(type(item) not in (int, float, bool) for item in value):
                    raise ValueError("Token vectors require immutable builtin scalar values")
            else:
                value = source_value
                if type(value) not in (int, float, bool, str, type(None)):
                    raise ValueError("Token fields require immutable builtin values")
            values[key] = value
        blob = encode_token({key: list(value) if isinstance(value, tuple) else value for key, value in values.items()})
        object.__setattr__(self, "_values", MappingProxyType(values))
        object.__setattr__(self, "blob", blob)
        object.__setattr__(self, "schema_hash", bound_hash)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _freeze_token(row: Mapping[str, Any], bound_hash: str) -> _EncodedToken:
    if type(row) is _EncodedToken:
        if row.schema_hash != bound_hash:
            raise ValueError("Encoded token belongs to a different schema")
        return row
    return _EncodedToken(row, bound_hash)


def _sequence_metadata(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[int],
    history_rows: int,
    decide_index: int,
    logical_position: int,
) -> tuple[int, ...]:
    legal = tuple(candidates)
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 546 for value in legal):
        raise ValueError("Candidates must be complete global action IDs")
    if not 1 <= len(legal) <= MAX_CANDIDATES or legal != tuple(sorted(set(legal))):
        raise ValueError("Candidates must be a canonical complete legal set")
    for value in (history_rows, decide_index, logical_position):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 32:
            raise ValueError("Sequence metadata must fit unsigned 32-bit fields")
    if len(rows) not in {history_rows + 64, history_rows + 65} or decide_index != history_rows + 63:
        raise ValueError("Sequence must contain history, sixty-two state slots, DECIDE and an optional ACTION")
    if rows[0]["kind"] != 1 or rows[decide_index]["kind"] != 9:
        raise ValueError("Sequence has no canonical BOS/DECIDE positions")
    if len(rows) == history_rows + 65 and rows[-1]["kind"] != 10:
        raise ValueError("Only ACTION may follow DECIDE")
    return legal


@dataclass(frozen=True)
class TokenSequence:
    rows: tuple[Mapping[str, Any], ...]
    candidates: tuple[int, ...]
    history_rows: int
    decide_index: int
    logical_position: int
    schema_id: str = ACTIVE_SCHEMA_ID
    _bound_hash: str = field(init=False, repr=False, compare=False)
    _payload: bytes = field(init=False, repr=False, compare=False)
    _encoded_rows: tuple[_EncodedToken, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_id != ACTIVE_SCHEMA_ID:
            raise ValueError("Token sequence belongs to a different schema identity")
        bound_hash = _active_hash()
        rows = tuple(_freeze_token(row, bound_hash) for row in self.rows)
        legal = _sequence_metadata(rows, self.candidates, self.history_rows, self.decide_index, self.logical_position)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "candidates", legal)
        object.__setattr__(self, "_bound_hash", bound_hash)
        object.__setattr__(self, "_payload", b"".join(row.blob for row in rows))
        object.__setattr__(self, "_encoded_rows", rows)

    @classmethod
    def _from_validated(
        cls,
        rows: tuple[_EncodedToken, ...],
        candidates: tuple[int, ...],
        history_rows: int,
        decide_index: int,
        logical_position: int,
        payload: bytes,
        bound_hash: str,
    ) -> TokenSequence:
        legal = _sequence_metadata(rows, candidates, history_rows, decide_index, logical_position)
        sequence = object.__new__(cls)
        for name, value in (
            ("rows", rows),
            ("candidates", legal),
            ("history_rows", history_rows),
            ("decide_index", decide_index),
            ("logical_position", logical_position),
            ("schema_id", ACTIVE_SCHEMA_ID),
            ("_bound_hash", bound_hash),
            ("_payload", payload),
            ("_encoded_rows", rows),
        ):
            object.__setattr__(sequence, name, value)
        return sequence

    @property
    def schema_hash(self) -> str:
        return self._bound_hash

    def to_bytes(self) -> bytes:
        header = b"KAR1" + bytes.fromhex(self.schema_hash) + struct.pack("<II", len(self.rows), self.history_rows)
        candidates = self.candidates + (546,) * (MAX_CANDIDATES - len(self.candidates))
        return header + struct.pack("<32H", *candidates) + self._payload

    def to_record(self) -> dict[str, Any]:
        rows = [
            {key: list(value) if isinstance(value, tuple) else value for key, value in row.items()} for row in self.rows
        ]
        return {
            "schema_id": self.schema_id,
            "schema_hash": self.schema_hash,
            "history_rows": self.history_rows,
            "decide_index": self.decide_index,
            "rows": rows,
            "candidates": list(self.candidates),
        }


@dataclass
class _HistoryState:
    melds: list[list[dict[str, Any]]] = field(default_factory=lambda: [[] for _ in range(4)])
    pending: dict[str, Any] | None = None
    replacement: int | None = None
    declaration: int | None = None
    indicators: list[int] = field(default_factory=list)
    seen_cash: set[tuple[Any, ...]] = field(default_factory=set)
    seen_ura: set[tuple[Any, ...]] = field(default_factory=set)


@dataclass(frozen=True)
class _HistoryPrefix:
    rows: tuple[_EncodedToken, ...]
    blob: bytes
    schema_hash: str
    logical_position: int


class TokenHistoryCache:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._binding: tuple[int, tuple[int, ...], str, str] | None = None
        self._events: tuple[str, ...] = ()
        self._positions: tuple[int, ...] = ()
        self._state = _HistoryState()
        self._prefix: _HistoryPrefix | None = None

    @property
    def processed_events(self) -> int:
        return len(self._events)

    @property
    def history_rows(self) -> int:
        return 0 if self._prefix is None else len(self._prefix.rows) - 1

    @property
    def schema_hash(self) -> str | None:
        return None if self._binding is None else self._binding[3]

    def _history(self, snapshot: PublicSnapshot) -> _HistoryPrefix:
        _available(snapshot)
        events = tuple(snapshot.events)
        positions = (
            tuple(snapshot.history_positions) if snapshot.history_positions else tuple(range(1, len(events) + 1))
        )
        if len(events) != len(positions):
            self.reset()
            raise ValueError("Public event clock does not match visible history")
        binding = (
            snapshot.observer_player_id,
            tuple(snapshot.initial_seat_order_player_ids),
            ACTIVE_SCHEMA_ID,
            _active_hash(),
        )
        length = len(self._events)
        if (
            self._binding != binding
            or len(events) < length
            or events[:length] != self._events
            or positions[:length] != self._positions
        ):
            self.reset()
        if self._binding is None:
            self._binding = binding
            bos = _EncodedToken(_row(1, 0), binding[3])
            self._prefix = _HistoryPrefix((bos,), bos.blob, binding[3], 1)
        assert self._prefix is not None
        start = len(self._events)
        try:
            previous = self._positions[-1] if self._positions else 0
            for position in positions[start:]:
                if (
                    isinstance(position, bool)
                    or not isinstance(position, int)
                    or not previous <= position < (1 << 32) - 1
                    or position < 1
                ):
                    raise ValueError("Public history positions must form a nondecreasing visible clock")
                previous = position
            added = _history(snapshot, state=self._state, start_index=start, positions=positions, bound_hash=binding[3])
            prefix = self._prefix
            self._prefix = _HistoryPrefix(
                prefix.rows + tuple(added),
                prefix.blob + b"".join(row.blob for row in added),
                prefix.schema_hash,
                previous + 1,
            )
            self._events, self._positions = events, tuple(positions)
        except Exception:
            self.reset()
            raise
        return self._prefix


def _available(snapshot: PublicSnapshot) -> None:
    if not snapshot.history_complete or not snapshot.physical_ledger_complete or not snapshot.visibility_complete:
        raise TokenizationUnavailableError("A complete audited observer prefix is required")


def _history(  # noqa: PLR0915
    snapshot: PublicSnapshot,
    *,
    state: _HistoryState | None = None,
    start_index: int = 0,
    positions: tuple[int, ...] | None = None,
    bound_hash: str | None = None,
) -> list[_EncodedToken]:
    rows: list[_EncodedToken] = []
    observer = snapshot.observer_player_id
    events = [json.loads(value) for value in snapshot.events[start_index:]]
    positions = (
        positions if positions is not None else snapshot.history_positions or tuple(range(1, len(snapshot.events) + 1))
    )
    if len(snapshot.events) != len(positions):
        raise ValueError("Public event clock does not match visible history")
    state = state if state is not None else _HistoryState()
    bound_hash = bound_hash if bound_hash is not None else _active_hash()
    melds: list[list[dict[str, Any]]] = state.melds
    pending: dict[str, Any] | None = state.pending
    replacement: int | None = state.replacement
    declaration: int | None = state.declaration
    indicators: list[int] = state.indicators
    seen_cash, seen_ura = state.seen_cash, state.seen_ura

    def append(row: dict[str, Any]) -> None:
        rows.append(_EncodedToken(row, bound_hash))

    def slot(actor: int, index: int) -> int:
        if not 0 <= index < 4:
            raise ValueError("More than four committed melds")
        return ((_relative(actor, observer) - 1) * 4) + index + 1

    for event, position in zip(events, positions[start_index:]):
        typ = event["type"]
        if typ == "kan_commit" and (pending is None or event.get("kan_type") != pending["kind"]):
            raise ValueError("Kan commitment token lacks its matching declaration")
        if pending is not None and pending_kan_commits(
            event,
            actor=pending["actor"],
            committed_kans=sum(item.get("kind") in COMMITTED_KAN_KINDS for player in melds for item in player),
            indicators=len(indicators),
        ):
            actor = pending["actor"]
            index = pending["index"]
            melds[actor][index] = pending
            accepted = _row(
                19 if pending["kind"] == "ankan" else 21,
                position,
                actor=_relative(actor, observer),
                slot=slot(actor, index),
            )
            accepted["tiles"][: len(pending["tiles"])] = pending["tiles"]
            accepted["flags"][1] = _flag(actor == observer or pending["kind"] != "ankan")
            if pending["kind"] == "kakan":
                accepted["source"] = pending["source"]
                accepted["flags"][0] = 2
            append(accepted)
            replacement = actor
            pending = None
        if typ == "start_game":
            continue
        if typ == "start_kyoku":
            melds = [[] for _ in range(4)]
            pending = None
            replacement = declaration = None
            indicators = [_tile(event["dora_marker"])]
            row = _row(11, position, stage=1, wind=("E", "S", "W", "N").index(event["bakaze"]) + 1)
            row["tiles"][0] = indicators[0]
            _numeric(row, 3, event["honba"], 10)
            _numeric(row, 4, event.get("kyotaku", 0), 10)
            _numeric(row, 6, (row["wind"] - 1) * 4 + event["kyoku"], 12)
            append(row)
            for relative in range(4):
                actor = (observer + relative) % 4
                player = _player_row(
                    actor,
                    observer,
                    position,
                    1,
                    event["scores"],
                    snapshot.initial_seat_order_player_ids,
                    event["oya"],
                    0,
                )
                player["flags"][:2] = [1, 1]
                player["flags"][3] = 2
                append(player)
            hand = event["tehais"][observer]
            if len(hand) != 13 or "?" in hand:
                raise ValueError("Initial observer hand is incomplete")
            codes = sorted(_tile(value) for value in hand)
            for start in range(0, len(codes), 5):
                chunk = _row(11, position, actor=1, stage=2, slot=start // 5 + 1)
                chunk["tiles"][: len(codes[start : start + 5])] = codes[start : start + 5]
                append(chunk)
        elif typ == "kan_commit":
            continue
        elif typ == "tsumo":
            actor = event["actor"]
            row = _row(
                12 if actor == observer else 13,
                position,
                actor=_relative(actor, observer),
                stage=2 if replacement == actor else 1,
            )
            row["tiles"][0] = _tile(event["pai"]) if actor == observer else 38
            replacement = None
            append(row)
        elif typ == "dahai":
            actor = event["actor"]
            row = _row(14, position, actor=_relative(actor, observer))
            row["tiles"][0] = _tile(event["pai"])
            row["flags"][:2] = [_flag(event["tsumogiri"]), _flag(declaration == actor)]
            declaration = None
            append(row)
        elif typ in {"chi", "pon", "daiminkan", "kan"}:
            actor = event["actor"]
            tiles = [_tile(event["pai"]), *sorted(_tile(value) for value in event["consumed"])]
            item = {
                "actor": actor,
                "source": _relative(event["target"], observer),
                "tiles": tiles,
                "kind": "daiminkan" if typ == "kan" else typ,
                "index": len(melds[actor]),
            }
            melds[actor].append(item)
            row = _row(
                {"chi": 15, "pon": 16, "daiminkan": 17, "kan": 17}[typ],
                position,
                actor=_relative(actor, observer),
                source=item["source"],
                slot=slot(actor, item["index"]),
            )
            row["tiles"][: len(tiles)] = tiles
            append(row)
            if typ in {"daiminkan", "kan"}:
                replacement = actor
        elif typ in {"ankan", "kakan"}:
            actor = event["actor"]
            if typ == "ankan":
                tiles = sorted(_tile(value) for value in event["consumed"])
                index = len(melds[actor])
                melds[actor].append({})
                source = 0
            else:
                code = _tile(event["pai"])
                target_base = convert.mjai_to_tid(event["pai"]) // 4
                indices = [
                    index
                    for index, item in enumerate(melds[actor])
                    if item.get("kind") == "pon"
                    and (convert.mjai_to_tid(event["pai"]) // 4 == _code_base(item["tiles"][0]))
                ]
                if len(indices) != 1:
                    raise ValueError(f"Cannot identify the committed pon for base34 {target_base}")
                index = indices[0]
                source = melds[actor][index]["source"]
                tiles = [*melds[actor][index]["tiles"], code]
            pending = {"actor": actor, "source": source, "tiles": tiles, "kind": typ, "index": index}
            row = _row(
                18 if typ == "ankan" else 20, position, actor=_relative(actor, observer), slot=slot(actor, index)
            )
            if typ == "ankan":
                row["tiles"][:4] = tiles
                row["flags"][1] = _flag(actor == observer)
            else:
                row["tiles"][0] = tiles[-1]
            append(row)
        elif typ in {"reach", "reach_accepted"}:
            actor = event["actor"]
            row = _row(22 if typ == "reach" else 23, position, actor=_relative(actor, observer))
            if typ == "reach":
                declaration = actor
            else:
                _numeric(row, 1, -1000, 10000)
            append(row)
        elif typ == "dora":
            indicators.append(_tile(event["dora_marker"]))
            row = _row(24, position, stage=1, slot=len(indicators))
            row["tiles"][0] = indicators[-1]
            append(row)
        elif typ in {"hora", "ryukyoku"}:
            if event.get("visibility_complete") is False:
                raise TokenizationUnavailableError("Terminal public-hand visibility is not verified")
            if typ == "hora":
                row = _row(
                    27,
                    position,
                    actor=_relative(event["actor"], observer),
                    source=_relative(event["target"], observer),
                    stage=1 if event["actor"] == event["target"] else 2,
                )
                if "pai" in event:
                    row["tiles"][0] = _tile(event["pai"])
                append(row)
                public_hand = event.get("public_hand")
                if public_hand is not None:
                    _reveal_rows(public_hand, event["actor"], observer, position, 1, append)
                ura = tuple(event.get("public_ura_markers", ()))
                key = position, ura
                if ura and key not in seen_ura:
                    seen_ura.add(key)
                    for index, value in enumerate(ura):
                        row = _row(24, position, stage=2, slot=index + 1)
                        row["tiles"][0] = _tile(value)
                        append(row)
            else:
                for actor, hand in enumerate(event.get("public_hands", [None] * 4)):
                    if hand is not None:
                        _reveal_rows(hand, actor, observer, position, 2, append)
                reason = event.get("reason", "")
                row = _row(
                    28,
                    position,
                    stage=4
                    if reason == "nagashi_mangan"
                    else 2
                    if reason in {"exhaustive", "exhaustive_draw", "fanpai"}
                    else 3,
                )
                append(row)
            delta = event.get("deltas", event.get("delta"))
            scores = event.get("scores")
            key = position, None if delta is None else tuple(delta), None if scores is None else tuple(scores)
            if delta is not None and scores is not None and key not in seen_cash:
                seen_cash.add(key)
                for actor in range(4):
                    row = _row(25, position, actor=_relative(actor, observer))
                    _numeric(row, 0, scores[actor], 50000)
                    _numeric(row, 1, delta[actor], 10000)
                    append(row)
            pending = None
        elif typ in {"end_kyoku", "end_game"}:
            append(_row(28, position, stage=5 if typ == "end_game" else 0))
        elif typ == "self_pass":
            row = _row(26, position, actor=1, source=_relative(event["target"], observer))
            row["tiles"][0] = _tile(event["pai"])
            append(row)
        else:
            raise ValueError(f"Unsupported visible history event {typ}")
    state.melds, state.pending = melds, pending
    state.replacement, state.declaration = replacement, declaration
    state.indicators = indicators
    return rows


def _code_base(code: int) -> int:
    if not 1 <= code <= 37:
        raise ValueError("Expected a visible tile code")
    tile = code - 1
    return tile - 3 if tile >= 30 else tile // 10 * 9 + (4 if tile % 10 == 0 else tile % 10 - 1)


def _reveal_rows(values: Sequence[str], actor: int, observer: int, position: int, stage: int, append: Any) -> None:
    codes = sorted(_tile(value) for value in values)
    if not 1 <= len(codes) <= 14:
        raise ValueError("Public concealed hand must contain one to fourteen tiles")
    for start in range(0, len(codes), 5):
        row = _row(29, position, actor=_relative(actor, observer), stage=stage, slot=start // 5 + 1)
        row["tiles"][: len(codes[start : start + 5])] = codes[start : start + 5]
        append(row)


def encode_prefix(  # noqa: PLR0915
    snapshot: PublicSnapshot, candidates: Sequence[int], *, history_cache: TokenHistoryCache | None = None
) -> TokenSequence:
    _available(snapshot)
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 546 for value in candidates):
        raise ValueError("Candidates must be complete global action IDs")
    legal = tuple(sorted(set(candidates)))
    if not 1 <= len(legal) <= MAX_CANDIDATES:
        raise ValueError("Expected one to thirty-two complete legal candidates")
    history = (history_cache if history_cache is not None else TokenHistoryCache())._history(snapshot)
    logical = history.logical_position
    rows: list[dict[str, Any]] = []
    observer = snapshot.observer_player_id
    features = extract_rule_features(snapshot)
    round_row = _row(2, logical, 1, actor=1, stage=features["own_current_rank"], wind=snapshot.round_wind + 1)
    for index, value, divisor in (
        (3, snapshot.honba, 10),
        (4, snapshot.riichi_sticks, 10),
        (5, snapshot.remaining_draws, 70),
        (6, snapshot.round_wind * 4 + snapshot.oya + 1, 12),
        (18, features["gap_to_higher"] if features["gap_to_higher_present"] else None, 1),
        (19, features["gap_to_lower"] if features["gap_to_lower_present"] else None, 1),
    ):
        _numeric(round_row, index, value, divisor)
    rows.append(round_row)
    for relative in range(4):
        actor = (observer + relative) % 4
        row = _player_row(
            actor, observer, logical, 0, snapshot.scores, snapshot.initial_seat_order_player_ids, snapshot.oya, 1
        )
        row["flags"][:2] = [_flag(snapshot.riichi_declared[actor]), _flag(snapshot.riichi_accepted[actor])]
        row["flags"][3] = _flag(all(m.kind == "ankan" for m in snapshot.melds[actor]))
        for index, value, divisor in (
            (12, snapshot.public_hand_sizes[actor], 14),
            (13, len(snapshot.melds[actor]), 4),
            (14, len(snapshot.discard_history[actor]), 24),
            (15, snapshot.riichi_discard_ordinals[actor], 24),
            (17, snapshot.draw_counts[actor] if actor == observer else None, 70),
        ):
            _numeric(row, index, value, divisor)
        rows.append(row)
    known = snapshot.known_counts()
    own = [sum(tile // 4 == base for tile in snapshot.hand) for base in range(34)]
    red_unknown = {
        m.tiles[0] // 4
        for actor in range(4)
        if actor != observer
        for m in snapshot.melds[actor]
        if m.kind == "ankan" and m.tiles[0] // 4 in {4, 13, 22}
    }
    public = [
        *snapshot.hand,
        *snapshot.dora_indicators,
        *[t for river in snapshot.current_rivers for t in river],
        *[t for player in snapshot.melds for meld in player for t in meld.tiles],
    ]
    for base in range(34):
        row = _row(4, logical, 1, actor=1)
        row["tiles"][0] = _representative(base)
        row["flags"] = list(features["hand_count_flags"][base])
        _numeric(row, 8, own[base], 4)
        _numeric(row, 9, sum(tile in {16, 52, 88} and tile // 4 == base for tile in snapshot.hand), 1)
        _numeric(row, 10, None if known is None else known[base], 4)
        _numeric(
            row,
            11,
            None if base in red_unknown else sum(tile in {16, 52, 88} and tile // 4 == base for tile in public),
            1,
        )
        rows.append(row)
    stages = {"chi": 8, "pon": 9, "daiminkan": 10, "ankan": 11, "kakan": 12}
    for relative in range(4):
        actor = (observer + relative) % 4
        for index in range(4):
            if index >= len(snapshot.melds[actor]):
                rows.append(_row(0, logical, 1))
                continue
            meld = snapshot.melds[actor][index]
            row = _row(
                5,
                logical,
                1,
                actor=relative + 1,
                source=0 if meld.source is None else _relative(meld.source, observer),
                stage=stages[meld.kind],
                slot=relative * 4 + index + 1,
            )
            row["tiles"][: len(meld.tiles)] = [tile136_to_tile37(tile) + 1 for tile in meld.tiles]
            row["flags"][:2] = [_flag(meld.kind != "ankan"), _flag(actor == observer or meld.kind != "ankan")]
            rows.append(row)
    for index in range(5):
        row = _row(6, logical, 1, slot=index + 1) if index < len(snapshot.dora_indicators) else _row(0, logical, 1)
        if index < len(snapshot.dora_indicators):
            row["tiles"][0] = tile136_to_tile37(snapshot.dora_indicators[index]) + 1
        rows.append(row)
    phases = {"own_act": 1, "discard_response": 2, "kan_response": 3}
    if snapshot.decision_phase not in phases:
        raise ValueError("No decision is available in this observer phase")
    phase = phases[snapshot.decision_phase]
    trigger = snapshot.pending_kan_actor if phase == 3 else snapshot.last_discard_actor if phase == 2 else None
    context = _row(7, logical, 1, actor=1, stage=phase, source=0 if trigger is None else _relative(trigger, observer))
    if phase == 1 and snapshot.current_draw_actor == observer and snapshot.drawn_tile is not None:
        context["tiles"][0] = tile136_to_tile37(snapshot.drawn_tile) + 1
    trigger_tile = snapshot.pending_kan_tile if phase == 3 else snapshot.last_discard_tile if phase == 2 else None
    if trigger_tile is not None:
        context["tiles"][1] = tile136_to_tile37(trigger_tile) + 1
    context["flags"][:2] = [_flag(snapshot.forced_tedashi and phase == 1), _flag(snapshot.post_call_actor == observer)]
    rows.append(context)
    legal_row = _row(8, logical, 1, actor=1)
    _numeric(legal_row, 16, len(legal), 32)
    rows.append(legal_row)
    rows.append(_row(9, logical, 2, actor=1))
    if len(rows) != 63:
        raise ValueError("Snapshot must contain exactly sixty-two semantic slots")
    state_rows = tuple(_freeze_token(row, history.schema_hash) for row in rows)
    history_rows = len(history.rows) - 1
    return TokenSequence._from_validated(
        history.rows + state_rows,
        legal,
        history_rows,
        history_rows + 63,
        logical,
        history.blob + b"".join(row.blob for row in state_rows),
        history.schema_hash,
    )


def append_action(prefix: TokenSequence, action_id: int) -> TokenSequence:
    if prefix.schema_id != ACTIVE_SCHEMA_ID or prefix.schema_hash != _active_hash():
        raise ValueError("An ACTION suffix cannot mix schema versions")
    if prefix.rows[-1]["kind"] != 9 or action_id not in prefix.candidates:
        raise ValueError("An ACTION suffix requires a legal candidate and an unsuffixed DECIDE prefix")
    spec = action_spec(action_id)
    families = {
        "DISCARD": 1,
        "ANKAN": 2,
        "KAKAN": 3,
        "TSUMO": 4,
        "KYUSHU_KYUHAI": 5,
        "PASS": 6,
        "CHI": 7,
        "PON": 8,
        "DAIMINKAN": 9,
        "RON": 10,
    }
    row = _row(
        10, prefix.logical_position + 1, 3, actor=1, action_ids=action_id, action_type_ids=families[spec.action_type]
    )
    if spec.action_type == "DISCARD":
        assert spec.tile37 is not None
        row["action_tile37"] = spec.tile37 + 1
        row["action_tsumogiri"] = _flag(spec.tsumogiri)
        row["action_riichi"] = _flag(spec.riichi)
    encoded = _freeze_token(row, prefix.schema_hash)
    return TokenSequence._from_validated(
        (*prefix._encoded_rows, encoded),
        prefix.candidates,
        prefix.history_rows,
        prefix.decide_index,
        prefix.logical_position,
        prefix._payload + encoded.blob,
        prefix.schema_hash,
    )
