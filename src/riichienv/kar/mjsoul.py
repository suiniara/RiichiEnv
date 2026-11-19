from __future__ import annotations

import hashlib
import importlib.resources
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .event_semantics import ABORT_REASONS, nine_terminals_public_hand

PROTOCOL_IDENTITY_VERSION = "protobuf_source_and_descriptor_v1"
PROTOCOL_IDENTITY_FIELDS = ("protocol_identity_version", "protocol_sha256", "protocol_descriptor_sha256")


def protocol_identity(source_sha256: str, descriptor_sha256: str) -> dict[str, str]:
    """Bind the supplied source and the compiled schema, including imported files."""
    for value in (source_sha256, descriptor_sha256):
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Protocol identity requires source and complete descriptor SHA-256 values")
    return {
        "protocol_identity_version": PROTOCOL_IDENTITY_VERSION,
        "protocol_sha256": source_sha256,
        "protocol_descriptor_sha256": descriptor_sha256,
    }


def validate_protocol_identity(record: Mapping[str, Any], *, context: str) -> dict[str, str]:
    if record.get("protocol_identity_version") != PROTOCOL_IDENTITY_VERSION:
        raise ValueError(f"{context} lacks the complete source/descriptor protocol identity; rebuild from raw inputs")
    source, descriptor = record.get("protocol_sha256"), record.get("protocol_descriptor_sha256")
    if not isinstance(source, str) or not isinstance(descriptor, str):
        raise ValueError(f"Protocol identity requires source and complete descriptor SHA-256 values: {context}")
    return protocol_identity(source, descriptor)


class ProtocolError(ValueError):
    pass


class RecordError(ValueError):
    pass


_SUPPORTED_RECORDS = frozenset(
    {
        "lq.RecordNewRound",
        "lq.RecordDealTile",
        "lq.RecordDiscardTile",
        "lq.RecordChiPengGang",
        "lq.RecordAnGangAddGang",
        "lq.RecordHule",
        "lq.RecordNoTile",
        "lq.RecordLiuJu",
    }
)


@dataclass(frozen=True)
class Protocol:
    source_sha256: str
    descriptor_sha256: str
    source_name: str
    pool: Any = field(repr=False, compare=False)

    def identity_record(self) -> dict[str, str]:
        return protocol_identity(self.source_sha256, self.descriptor_sha256)

    @classmethod
    def from_descriptor_set(
        cls,
        descriptor_bytes: bytes,
        *,
        source_sha256: str | None = None,
        source_name: str = "synthetic_descriptor",
    ) -> Protocol:
        try:
            from google.protobuf import descriptor_pb2, descriptor_pool  # noqa: PLC0415
        except ImportError as error:
            raise ProtocolError("Protobuf parsing requires the protobuf dependency") from error
        descriptor_set = descriptor_pb2.FileDescriptorSet()
        try:
            descriptor_set.ParseFromString(descriptor_bytes)
        except Exception as error:
            raise ProtocolError("Invalid FileDescriptorSet") from error
        if not descriptor_set.file:
            raise ProtocolError("An empty FileDescriptorSet is not a protocol")
        pool = descriptor_pool.DescriptorPool()
        pending = list(descriptor_set.file)
        while pending:
            unresolved = []
            for item in pending:
                try:
                    pool.AddSerializedFile(item.SerializeToString())
                except (TypeError, ValueError):
                    unresolved.append(item)
            if len(unresolved) == len(pending):
                raise ProtocolError("Unresolved or conflicting Protobuf descriptors")
            pending = unresolved
        digest = hashlib.sha256(descriptor_bytes).hexdigest()
        return cls(source_sha256 or digest, digest, source_name, pool)

    def message(self, full_name: str, payload: bytes) -> Any:
        from google.protobuf import message_factory  # noqa: PLC0415
        from google.protobuf.message import DecodeError  # noqa: PLC0415

        try:
            descriptor = self.pool.FindMessageTypeByName(full_name)
        except KeyError as error:
            raise ProtocolError(f"Protocol lacks {full_name}") from error
        message = message_factory.GetMessageClass(descriptor)()
        try:
            message.ParseFromString(payload)
        except DecodeError as error:
            raise RecordError(f"Malformed {full_name}") from error
        return message

    def check_field(self, message_name: str, name: str, number: int, field_type: int) -> None:
        try:
            descriptor = self.pool.FindMessageTypeByName(message_name)
            actual = descriptor.fields_by_name[name]
        except KeyError as error:
            raise ProtocolError(f"Protocol lacks {message_name}.{name}") from error
        if actual.number != number or actual.type != field_type:
            raise ProtocolError(f"Unexpected wire contract for {message_name}.{name}")


def load_protocol(proto_path: str | Path) -> Protocol:
    source = Path(proto_path).resolve(strict=True)
    if not source.is_file():
        raise ProtocolError("proto_path must name a file")
    try:
        from grpc_tools import protoc  # noqa: PLC0415
    except ImportError as error:
        raise ProtocolError("Compiling a .proto requires grpcio-tools; generated pb2 files are not executed") from error
    with tempfile.TemporaryDirectory(prefix="kar-proto-") as directory:
        output = Path(directory) / "protocol.pb"
        standard = importlib.resources.files("grpc_tools") / "_proto"
        arguments = [
            "protoc",
            f"-I{source.parent}",
            f"-I{standard}",
            "--include_imports",
            f"--descriptor_set_out={output}",
            str(source),
        ]
        if protoc.main(arguments) != 0:
            raise ProtocolError("Failed to compile the supplied Protobuf schema")
        descriptor_bytes = output.read_bytes()
    return Protocol.from_descriptor_set(
        descriptor_bytes,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_name=str(source),
    )


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise RecordError("Truncated Protobuf varint")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise RecordError("Protobuf varint exceeds uint64")
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
    raise RecordError("Unterminated Protobuf varint")


def _wire_fields(data: bytes):
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire_type = tag >> 3, tag & 7
        if number == 0 or number >= 1 << 29:
            raise RecordError("Invalid Protobuf field number")
        if wire_type == 0:
            value, offset = _varint(data, offset)
        elif wire_type in (1, 2, 5):
            if wire_type == 2:
                size, offset = _varint(data, offset)
            else:
                size = 8 if wire_type == 1 else 4
            end = offset + size
            if end > len(data):
                raise RecordError("Truncated Protobuf field")
            value = data[offset:end]
            offset = end
        else:
            raise RecordError("Unsupported Protobuf wire type")
        yield number, wire_type, value


def _single_bytes(data: bytes, wanted: int, *, required: bool = True) -> bytes:
    matches = []
    for number, wire_type, value in _wire_fields(data):
        if number == wanted:
            if wire_type != 2:
                raise RecordError(f"Field {wanted} must be length-delimited")
            matches.append(value)
    if len(matches) > 1:
        raise RecordError(f"Duplicate singular field {wanted}")
    if not matches or not matches[0]:
        if required:
            raise RecordError(f"Missing nonempty field {wanted}")
        return b""
    return matches[0]


def _wrapper(data: bytes, protocol: Protocol, allowed_names: frozenset[str]) -> tuple[str, bytes]:
    protocol.check_field("lq.Wrapper", "name", 1, 9)
    protocol.check_field("lq.Wrapper", "data", 2, 12)
    try:
        name = _single_bytes(data, 1, required=False).decode("utf-8")
    except UnicodeDecodeError as error:
        raise RecordError("Invalid Wrapper.name UTF-8") from error
    canonical_name = name.removeprefix(".")
    if canonical_name not in allowed_names:
        raise RecordError(f"Unsupported Wrapper.name: {name!r}")
    return canonical_name, _single_bytes(data, 2)


@dataclass(frozen=True)
class ServerRecord:
    index: int
    name: str
    source_sha256: str
    data: dict[str, Any]


@dataclass(frozen=True)
class DecodedGame:
    source_sha256: str
    protocol_sha256: str
    game_id: str
    metadata: dict[str, Any]
    records: tuple[ServerRecord, ...]
    events: tuple[dict[str, Any], ...]
    protocol_descriptor_sha256: str

    @property
    def complete(self) -> bool:
        return bool(self.events and self.events[-1]["type"] == "end_game")


def _has(message: Any, name: str) -> bool:
    if name not in message.DESCRIPTOR.fields_by_name:
        return False
    try:
        return message.HasField(name)
    except ValueError:
        return bool(getattr(message, name))


def _copy_fields(message: Any, names: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for name in names:
        descriptor = message.DESCRIPTOR.fields_by_name.get(name)
        if descriptor is None:
            continue
        value = getattr(message, name)
        if descriptor.is_repeated:
            result[name] = list(value)
        elif descriptor.message_type is None:
            result[name] = value
    return result


def _record_data(message: Any, name: str) -> dict[str, Any]:
    fields = {
        "lq.RecordNewRound": (
            "chang",
            "ju",
            "ben",
            "dora",
            "scores",
            "liqibang",
            "tiles0",
            "tiles1",
            "tiles2",
            "tiles3",
            "left_tile_count",
            "doras",
        ),
        "lq.RecordDealTile": ("seat", "tile", "left_tile_count", "doras"),
        "lq.RecordDiscardTile": ("seat", "tile", "is_liqi", "is_wliqi", "moqie", "doras"),
        "lq.RecordChiPengGang": ("seat", "type", "tiles", "froms"),
        "lq.RecordAnGangAddGang": ("seat", "type", "tiles", "doras"),
        "lq.RecordHule": ("old_scores", "delta_scores", "scores", "doras"),
        "lq.RecordNoTile": ("liujumanguan", "gameend"),
        "lq.RecordLiuJu": ("type", "seat", "tiles"),
    }
    result = _copy_fields(message, fields[name])
    if _has(message, "liqi"):
        result["liqi"] = _copy_fields(message.liqi, ("seat", "score", "liqibang", "failed"))
    if name == "lq.RecordHule":
        result["hules"] = [
            _copy_fields(hule, ("seat", "hu_tile", "zimo", "count", "fu", "hand", "ming", "doras", "li_doras"))
            for hule in message.hules
        ]
    if name == "lq.RecordNoTile":
        result["scores"] = [_copy_fields(score, ("seat", "old_scores", "delta_scores")) for score in message.scores]
        if "players" in message.DESCRIPTOR.fields_by_name:
            result["players"] = [_copy_fields(player, ("tingpai", "hand")) for player in message.players]
    if name != "lq.RecordNoTile" and _has(message, "gameend"):
        if "scores" in message.gameend.DESCRIPTOR.fields_by_name:
            result["gameend_scores"] = list(message.gameend.scores)
    return result


def _head_metadata(protocol: Protocol, payload: bytes) -> tuple[str, dict[str, Any]]:
    if not payload:
        return "", {}
    head = protocol.message("lq.RecordGame", payload)
    metadata = _copy_fields(head, ("start_time", "end_time"))
    if _has(head, "config"):
        metadata["category"] = int(head.config.category)
        if _has(head.config, "mode"):
            metadata["mode"] = int(head.config.mode.mode)
        if _has(head.config, "meta"):
            metadata["mode_id"] = int(head.config.meta.mode_id)
    if "accounts" in head.DESCRIPTOR.fields_by_name:
        metadata["account_seats"] = [int(account.seat) for account in head.accounts]
    if _has(head, "result"):
        metadata["final_standings"] = [
            _copy_fields(player, ("seat", "total_point", "part_point_1", "part_point_2"))
            for player in head.result.players
        ]
    return str(head.uuid), metadata


def decode_record_bytes(  # noqa: PLR0915
    data: bytes,
    protocol: Protocol,
    *,
    container: Literal["majsoul_download", "res_game_record", "wrapper", "game_detail_records"] = "res_game_record",
) -> DecodedGame:
    if not isinstance(data, bytes) or not data:
        raise RecordError("A nonempty byte string is required")
    payload = data
    metadata: dict[str, Any] = {"container": container, "record_provenance": "server_result_only"}
    game_id = ""
    if container == "majsoul_download":
        if not payload.startswith(b"\x00\x00\x00") or len(payload) <= 3:
            raise RecordError("Archived download requires the verified three-zero-byte prefix")
        _, payload = _wrapper(payload[3:], protocol, frozenset({""}))
    elif container not in {"res_game_record", "wrapper", "game_detail_records"}:
        raise RecordError(f"Unsupported container: {container!r}")
    if container in {"majsoul_download", "res_game_record"}:
        protocol.check_field("lq.ResGameRecord", "data", 4, 12)
        error_data = _single_bytes(payload, 1, required=False)
        if error_data:
            error = protocol.message("lq.Error", error_data)
            if getattr(error, "code", 0):
                raise RecordError(f"Server returned error {error.code}")
        game_id, head = _head_metadata(protocol, _single_bytes(payload, 3, required=False))
        metadata.update(head)
        _, payload = _wrapper(_single_bytes(payload, 4), protocol, frozenset({"lq.GameDetailRecords"}))
    elif container == "wrapper":
        _, payload = _wrapper(payload, protocol, frozenset({"lq.GameDetailRecords"}))
    protocol.check_field("lq.GameDetailRecords", "records", 1, 12)
    protocol.check_field("lq.GameDetailRecords", "version", 2, 13)
    protocol.check_field("lq.GameDetailRecords", "actions", 3, 11)
    legacy = []
    actions = []
    versions = []
    for number, wire_type, value in _wire_fields(payload):
        if number in {1, 3}:
            if wire_type != 2:
                raise RecordError("Record/action entries must be length-delimited")
            (legacy if number == 1 else actions).append(value)
        elif number == 2:
            if wire_type != 0:
                raise RecordError("GameDetailRecords.version must be a varint")
            versions.append(value)
    if len(versions) > 1 or (legacy and actions):
        raise RecordError("Ambiguous GameDetailRecords storage")
    version = versions[0] if versions else 0
    missing_results = []
    if version == 0:
        if actions or not legacy:
            raise RecordError("Version 0 requires nonempty legacy records")
        results = list(enumerate(legacy))
    elif version == 210715:
        if legacy or not actions:
            raise RecordError("Version 210715 requires actions")
        protocol.check_field("lq.GameAction", "result", 3, 12)
        results = []
        for index, action_bytes in enumerate(actions):
            # Parse only the result wire field; opaque client input never enters a message parser.
            result = _single_bytes(action_bytes, 3, required=False)
            if result:
                results.append((index, result))
            else:
                missing_results.append(index)
        if not results:
            raise RecordError("No server results in actions")
    else:
        raise RecordError(f"Unsupported GameDetailRecords.version: {version}")
    records = []
    for index, result in results:
        name, body = _wrapper(result, protocol, _SUPPORTED_RECORDS)
        message = protocol.message(name, body)
        records.append(ServerRecord(index, name, hashlib.sha256(result).hexdigest(), _record_data(message, name)))
    metadata.update({"record_version": version, "resultless_action_indices": missing_results})
    standings = metadata.get("final_standings", [])
    metadata["header_reports_completion"] = bool(metadata.get("end_time") and len(standings) == 4)
    events = normalize_records(records, game_id=game_id)
    metadata["visibility_gaps"] = [
        {
            "event_id": index,
            "source_record_index": event["source_record_index"],
            "reason": event["visibility_gap"],
        }
        for index, event in enumerate(events)
        if event.get("visibility_complete") is False
    ]
    metadata["source_score_chain_complete"] = not any(event.get("settlement_labels_missing") for event in events)
    metadata["source_score_chain_gaps"] = [
        {
            "event_id": index,
            "source_record_index": event["source_record_index"],
            "reason": "source_settlement_labels_missing",
            "missing_fields": event["settlement_missing_fields"],
        }
        for index, event in enumerate(events)
        if event.get("settlement_labels_missing")
    ]
    metadata["observed_source_anchors_after_gaps"] = [
        {"event_id": index, "source_record_index": event["source_record_index"], "scores": event["scores"]}
        for index, event in enumerate(events)
        if event.get("source_score_anchor_after_gap")
    ]
    metadata["derived_riichi_acceptances"] = [
        {"event_id": index, "source_record_index": event["source_record_index"], "actor": event["actor"]}
        for index, event in enumerate(events)
        if event.get("acceptance_evidence") == "four_riichi_abort"
    ]
    metadata["completion_evidence"] = (
        "explicit_server_gameend" if events[-1]["type"] == "end_game" else "missing_explicit_server_gameend"
    )
    return DecodedGame(
        hashlib.sha256(data).hexdigest(),
        protocol.source_sha256,
        game_id,
        metadata,
        tuple(records),
        events,
        protocol.descriptor_sha256,
    )


def _seat(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 4:
        raise RecordError("Expected a four-player seat index")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordError(f"Expected integer {name}")
    return value


def _tile(value: Any) -> str:
    if not isinstance(value, str):
        raise RecordError("Tile must be a string")
    if value in {"E", "S", "W", "N", "P", "F", "C"}:
        return value
    if len(value) == 2 and value[1] in "mps" and value[0] in "0123456789":
        return f"5{value[1]}r" if value[0] == "0" else value
    if len(value) == 2 and value[1] == "z" and value[0] in "1234567":
        return ("E", "S", "W", "N", "P", "F", "C")[int(value[0]) - 1]
    if value in {"5mr", "5pr", "5sr"}:
        return value
    raise RecordError(f"Invalid tile: {value!r}")


def _base(tile: str) -> str:
    return tile.removesuffix("r")


def _scores(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise RecordError(f"{field_name} requires four scores")
    return [_integer(item, field_name) for item in value]


def _notile_public_hands(data: dict[str, Any], hands: list[list[str]]) -> dict[str, Any]:
    players = data.get("players")
    unknown: dict[str, Any] = {"public_hands": [None] * 4, "visibility_complete": False}
    if players is None:
        return {**unknown, "visibility_gap": "notile_players_field_missing"}
    if not isinstance(players, (tuple, list)):
        raise RecordError("NoTile players must be a repeated player-info sequence")
    if len(players) != 4:
        return {**unknown, "visibility_gap": "notile_four_player_entries_unavailable"}
    public: list[list[str] | None] = [None, None, None, None]
    gap = None
    for seat, player in enumerate(players):
        if not isinstance(player, dict):
            raise RecordError("NoTile player info must be a field mapping")
        revealed = player.get("tingpai")
        if revealed is None:
            gap = gap or "notile_public_status_unavailable"
        elif not isinstance(revealed, bool):
            raise RecordError("NoTile player tingpai must be an explicit boolean")
        raw = player.get("hand", [])
        if not isinstance(raw, (tuple, list)):
            raise RecordError("NoTile player hand must be a tile sequence")
        hand = [_tile(tile) for tile in raw]
        if hand:
            matches = [candidate for candidate in range(4) if Counter(hand) == Counter(hands[candidate])]
            if not matches:
                raise RecordError("NoTile player hand disagrees with every replayed concealed hand")
            # There is no seat field; bind positions only when the replayed tiles prove them.
            if seat not in matches:
                gap = gap or "notile_player_position_unverified"
            if revealed is True and matches != [seat]:
                gap = gap or "notile_public_hand_seat_binding_ambiguous"
        if revealed is True:
            if not hand:
                gap = gap or "notile_public_hand_unavailable"
            else:
                public[seat] = sorted(hand)
    if gap is not None:
        return {**unknown, "visibility_gap": gap}
    return {"public_hands": public, "visibility_complete": True}


def normalize_records(  # noqa: PLR0915
    records: list[ServerRecord] | tuple[ServerRecord, ...],
    *,
    game_id: str = "",
) -> tuple[dict[str, Any], ...]:
    if not records:
        raise RecordError("No records to normalize")
    events: list[dict[str, Any]] = [{"type": "start_game"}]
    if game_id:
        events[0]["id"] = game_id
    hands: list[list[str]] = [[], [], [], []]
    indicators: list[str] = []
    current_scores: list[int] | None = None
    previous_round: tuple[int, int] | None = None
    trigger: tuple[int, str] | None = None
    last_draw: tuple[int, str] | None = None
    pending_reach: set[int] = set()
    accepted_reach: set[int] = set()
    round_open = False
    terminal_match = False
    discard_history: list[list[str]] = [[], [], [], []]
    meld_seen = False
    kan_actors: list[int] = []
    draw_counts = [0] * 4

    def emit(event: dict[str, Any], record: ServerRecord) -> None:
        events.append({**event, "source_record_index": record.index, "source_record_sha256": record.source_sha256})

    def remove_tile(actor: int, tile: str) -> None:
        try:
            hands[actor].remove(tile)
        except ValueError as error:
            raise RecordError(f"Seat {actor} cannot consume {tile}") from error

    def accept_reach(data: dict[str, Any], record: ServerRecord) -> None:
        liqi = data.get("liqi")
        if liqi is None:
            return
        actor = _seat(liqi.get("seat"))
        if liqi.get("failed", False):
            pending_reach.discard(actor)
            return
        if actor not in pending_reach:
            raise RecordError("LiQiSuccess has no prior declaration")
        pending_reach.remove(actor)
        accepted_reach.add(actor)
        emit({"type": "reach_accepted", "actor": actor}, record)
        if current_scores is not None:
            current_scores[actor] -= 1000
            if "score" in liqi and liqi["score"] != current_scores[actor]:
                raise RecordError("LiQiSuccess score does not match the declared deposit")

    def accept_fourth_riichi(record: ServerRecord) -> None:
        if accepted_reach | pending_reach != set(range(4)):
            raise RecordError("Four-riichi abort lacks four distinct declarations")
        if not pending_reach:
            return
        if len(pending_reach) != 1 or trigger is None or trigger[0] not in pending_reach:
            raise RecordError("Four-riichi acceptance is not uniquely tied to the latest declaration discard")
        actor = trigger[0]
        pending_reach.remove(actor)
        accepted_reach.add(actor)
        if current_scores is not None:
            current_scores[actor] -= 1000
        emit({"type": "reach_accepted", "actor": actor, "acceptance_evidence": "four_riichi_abort"}, record)

    def reveal_dora(data: dict[str, Any], record: ServerRecord) -> None:
        values = data.get("doras", [])
        if not values:
            return
        markers = [_tile(item) for item in values]
        if markers[: len(indicators)] != indicators or len(markers) > 5:
            raise RecordError("Public dora indicators are not a monotone prefix")
        for marker in markers[len(indicators) :]:
            emit({"type": "dora", "dora_marker": marker}, record)
        indicators[:] = markers

    for record in records:
        data = record.data
        name = record.name
        if name not in _SUPPORTED_RECORDS:
            raise RecordError(f"Unsupported server record: {name}")
        if terminal_match:
            raise RecordError("Server records continue after match termination")
        if name == "lq.RecordNewRound":
            if round_open:
                raise RecordError("New round before previous round was resolved")
            actor = _seat(data.get("ju"))
            chang = _integer(data.get("chang"), "chang")
            if not 0 <= chang <= 3:
                raise RecordError("Invalid round wind")
            incoming_scores = _scores(data.get("scores"), "new-round scores")
            honba = _integer(data.get("ben"), "ben")
            riichi_sticks = _integer(data.get("liqibang"), "liqibang")
            if honba < 0 or riichi_sticks < 0:
                raise RecordError("Round counters must be nonnegative")
            round_index = chang * 4 + actor
            if previous_round is None:
                if (round_index, honba, riichi_sticks) != (0, 0, 0) or incoming_scores != [25000] * 4:
                    raise RecordError("not_initial_round: expected East 1, zero honba/sticks and four 25000 scores")
            else:
                old_round_index, old_honba = previous_round
                if round_index == old_round_index:
                    if honba != old_honba + 1:
                        raise RecordError("Round continuity requires exactly one additional honba for a repeat")
                elif round_index == old_round_index + 1:
                    if honba not in {0, old_honba + 1}:
                        raise RecordError("Round continuity has an invalid honba after advancing")
                else:
                    raise RecordError("Round continuity forbids skipped or backwards rounds")
            observed_anchor_after_gap = current_scores is None and previous_round is not None
            if current_scores is not None and incoming_scores != current_scores:
                raise RecordError("New-round scores disagree with the previous known settlement")
            current_scores = incoming_scores
            previous_round = round_index, honba
            hands = [[_tile(tile) for tile in data.get(f"tiles{seat}", [])] for seat in range(4)]
            if any(len(hand) != 13 for seat, hand in enumerate(hands) if seat != actor):
                raise RecordError("Four-player new round requires three 13-tile nondealer hands")
            if len(hands[actor]) not in {13, 14}:
                raise RecordError("Dealer must begin with 13 or 14 tiles")
            dealer_draw = hands[actor].pop() if len(hands[actor]) == 14 else None
            raw_indicators = data.get("doras") or ([data["dora"]] if data.get("dora") else [])
            indicators = [_tile(tile) for tile in raw_indicators]
            if not 1 <= len(indicators) <= 5:
                raise RecordError("Missing or invalid public dora indicators")
            emit(
                {
                    "type": "start_kyoku",
                    "bakaze": ("E", "S", "W", "N")[chang],
                    "kyoku": actor + 1,
                    "honba": honba,
                    "kyotaku": riichi_sticks,
                    "oya": actor,
                    "scores": current_scores.copy(),
                    "dora_marker": indicators[0],
                    "tehais": [hand.copy() for hand in hands],
                    **({"source_score_anchor_after_gap": True} if observed_anchor_after_gap else {}),
                },
                record,
            )
            for marker in indicators[1:]:
                emit({"type": "dora", "dora_marker": marker}, record)
            trigger = None
            last_draw = None
            pending_reach.clear()
            accepted_reach.clear()
            discard_history = [[], [], [], []]
            meld_seen = False
            kan_actors = []
            draw_counts = [0] * 4
            round_open = True
            if dealer_draw is not None:
                draw_counts[actor] += 1
                hands[actor].append(dealer_draw)
                last_draw = actor, dealer_draw
                emit({"type": "tsumo", "actor": actor, "pai": dealer_draw}, record)
            continue
        if not round_open:
            raise RecordError("Action outside an open round")
        accept_reach(data, record)
        if name == "lq.RecordDealTile":
            actor, tile = _seat(data.get("seat")), _tile(data.get("tile"))
            reveal_dora(data, record)
            draw_counts[actor] += 1
            hands[actor].append(tile)
            last_draw = actor, tile
            trigger = None
            emit({"type": "tsumo", "actor": actor, "pai": tile}, record)
        elif name == "lq.RecordDiscardTile":
            actor, tile = _seat(data.get("seat")), _tile(data.get("tile"))
            reveal_dora(data, record)
            moqie = data.get("moqie")
            if not isinstance(moqie, bool):
                raise RecordError("Discard requires explicit moqie semantics")
            if moqie and last_draw != (actor, tile):
                raise RecordError("Tsumogiri does not match the latest draw")
            if data.get("is_liqi") or data.get("is_wliqi"):
                if actor in pending_reach:
                    raise RecordError("Duplicate pending riichi declaration")
                pending_reach.add(actor)
                emit({"type": "reach", "actor": actor}, record)
            remove_tile(actor, tile)
            discard_history[actor].append(tile)
            emit({"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": moqie}, record)
            trigger = actor, tile
            last_draw = None
        elif name == "lq.RecordChiPengGang":
            actor = _seat(data.get("seat"))
            kind = data.get("type")
            if kind not in {0, 1, 2}:
                raise RecordError("Unsupported open-meld type")
            tiles = [_tile(tile) for tile in data.get("tiles", [])]
            sources = [_seat(seat) for seat in data.get("froms", [])]
            if len(tiles) != (4 if kind == 2 else 3) or len(sources) != len(tiles):
                raise RecordError("Invalid meld tile/source counts")
            called = [index for index, source in enumerate(sources) if source != actor]
            if len(called) != 1:
                raise RecordError("Open meld must have one externally called tile")
            index = called[0]
            target, tile = sources[index], tiles[index]
            if trigger != (target, tile) or (kind == 0 and target != (actor + 3) % 4):
                raise RecordError("Meld source does not match the public discard trigger")
            consumed = [tile for index_, tile in enumerate(tiles) if index_ != index]
            meld_seen = True
            if kind == 2:
                kan_actors.append(actor)
            for consumed_tile in consumed:
                remove_tile(actor, consumed_tile)
            emit(
                {
                    "type": ("chi", "pon", "daiminkan")[kind],
                    "actor": actor,
                    "target": target,
                    "pai": tile,
                    "consumed": consumed,
                },
                record,
            )
            trigger = None
            last_draw = None
        elif name == "lq.RecordAnGangAddGang":
            actor, tile = _seat(data.get("seat")), _tile(data.get("tiles"))
            kind = data.get("type")
            if kind == 3:
                consumed = [item for item in hands[actor] if _base(item) == _base(tile)]
                if len(consumed) != 4:
                    raise RecordError("Ankan requires four matching tiles in the source hand")
                for consumed_tile in consumed:
                    remove_tile(actor, consumed_tile)
                emit({"type": "ankan", "actor": actor, "consumed": consumed}, record)
            elif kind == 2:
                remove_tile(actor, tile)
                emit({"type": "kakan", "actor": actor, "pai": tile}, record)
            else:
                raise RecordError("Unsupported concealed/added-kan type")
            meld_seen = True
            kan_actors.append(actor)
            trigger = actor, tile
            last_draw = None
            reveal_dora(data, record)
        elif name == "lq.RecordHule":
            reveal_dora(data, record)
            winners = data.get("hules", [])
            if not winners or len(winners) > 3:
                raise RecordError("Hule requires one to three winners")
            scores = _scores(data.get("scores"), "hule scores")
            old = _scores(data.get("old_scores"), "hule old_scores")
            if current_scores is None or old != current_scores:
                raise RecordError("Hule old_scores disagree with the current known scores")
            delta = _scores(data.get("delta_scores"), "hule delta_scores")
            if any(before + change != after for before, change, after in zip(old, delta, scores)):
                raise RecordError("Hule settlement arithmetic is inconsistent")
            seats = [_seat(winner.get("seat")) for winner in winners]
            if len(set(seats)) != len(seats):
                raise RecordError("Duplicate hule winner")
            known_ura: list[str] | None = None
            ordered_winners = sorted(
                zip(winners, seats), key=lambda pair: (pair[1] - trigger[0]) % 4 if trigger else pair[1]
            )
            for winner, actor in ordered_winners:
                zimo = winner.get("zimo")
                if not isinstance(zimo, bool):
                    raise RecordError("Hule requires zimo semantics")
                tile = _tile(winner.get("hu_tile"))
                if zimo:
                    if len(winners) != 1 or last_draw != (actor, tile):
                        raise RecordError("Self-draw win does not match the last draw")
                    target = actor
                else:
                    if trigger is None or trigger[1] != tile or trigger[0] == actor:
                        raise RecordError("Ron does not match a public discard/kan trigger")
                    target = trigger[0]
                if winner.get("hand"):
                    expected_hand = Counter(hands[actor])
                    if zimo:
                        expected_hand[tile] -= 1
                        expected_hand += Counter()
                    if Counter(_tile(item) for item in winner["hand"]) != expected_hand:
                        raise RecordError("Hule hand disagrees with the replayed concealed hand")
                event = {
                    "type": "hora",
                    "actor": actor,
                    "target": target,
                    "pai": tile,
                    "public_hand": sorted(hands[actor] if zimo else [*hands[actor], tile]),
                    "han": _integer(winner.get("count"), "han"),
                    "fu": _integer(winner.get("fu"), "fu"),
                    "scores": scores.copy(),
                    "deltas": delta.copy(),
                }
                if "li_doras" in winner:
                    ura = [_tile(item) for item in winner["li_doras"]]
                    if actor in accepted_reach and ura:
                        if known_ura is not None and known_ura != ura:
                            raise RecordError("Riichi winners have conflicting ura indicators")
                        known_ura = ura
                    event["ura_markers"] = ura
                emit(event, record)
            current_scores = scores
            round_open = False
        elif name == "lq.RecordNoTile":
            score_entries = data.get("scores", [])
            delta = [0, 0, 0, 0]
            old = None
            missing_components = []
            for component_index, entry in enumerate(score_entries):
                before = _scores(entry.get("old_scores"), "notile old_scores")
                if current_scores is None or before != current_scores:
                    raise RecordError("NoTile old_scores disagree with the current known scores")
                old = before
                component = entry.get("delta_scores")
                if component is None or isinstance(component, (list, tuple)) and not component:
                    missing_components.append(component_index)
                else:
                    observed_delta = _scores(component, "notile delta")
                    delta = [left + right for left, right in zip(delta, observed_delta)]
            event = {
                "type": "ryukyoku",
                "reason": "nagashi_mangan" if data.get("liujumanguan") else "exhaustive_draw",
                **_notile_public_hands(data, hands),
            }
            if not score_entries or missing_components:
                event.update(
                    settlement_labels_missing=True,
                    settlement_missing_fields=["deltas", "scores"],
                    settlement_missing_component_indices=missing_components,
                )
                # A later public score is a new source observation, not proof of this missing transfer.
                current_scores = None
            else:
                assert old is not None
                current_scores = [before + change for before, change in zip(old, delta)]
                event.update(scores=current_scores.copy(), deltas=delta)
            emit(event, record)
            terminal_match = data.get("gameend") is True
            round_open = False
        elif name == "lq.RecordLiuJu":
            kind = data.get("type")
            if kind == 1:
                reason = "kyushukyuhai"
            elif kind == 4:
                accept_fourth_riichi(record)
                reason = "four_riichi"
            elif kind in {2, 3}:
                first = [river[0] for river in discard_history if len(river) == 1]
                four_winds = (
                    not meld_seen and len(first) == 4 and len(set(first)) == 1 and first[0] in {"E", "S", "W", "N"}
                )
                four_kans = len(kan_actors) == 4 and len(set(kan_actors)) > 1
                if four_winds == four_kans:
                    raise RecordError("Cannot resolve LiuJu 2/3 from public winds/kans invariants")
                reason = "four_winds" if four_winds else "four_kans"
            else:
                raise RecordError("Unfrozen LiuJu reason; no response category is guessed")
            event: dict[str, Any] = {"type": "ryukyoku", "reason": reason, "visibility_complete": True}
            if reason == "kyushukyuhai":
                actor = _seat(data.get("seat", None if last_draw is None else last_draw[0]))
                try:
                    revealed = nine_terminals_public_hand(
                        actor,
                        hands[actor],
                        current_draw_actor=None if last_draw is None else last_draw[0],
                        draw_count=draw_counts[actor],
                        has_discarded=bool(discard_history[actor]),
                        calls_seen=meld_seen,
                        source_tiles=[_tile(tile) for tile in data.get("tiles", ())],
                    )
                except ValueError as error:
                    raise RecordError(str(error)) from error
                shown: list[list[str] | None] = [None for _ in range(4)]
                shown[actor] = revealed
                event.update(
                    actor=actor,
                    public_hands=shown,
                    public_reveal_provenance="source_declaring_tiles_verified_against_prior_events"
                    if data.get("tiles")
                    else "frozen_nine_terminals_reveal_from_verified_prior_events",
                    declaring_actor_provenance="source_seat_and_current_draw"
                    if "seat" in data
                    else "unique_verified_current_draw",
                )
            if reason in ABORT_REASONS:
                # Frozen aborts have no transfer. This is rule evidence, not a
                # blanket zero imputation for missing NoTile/Hule settlements.
                event.update(deltas=[0] * 4, settlement_provenance="frozen_abort_zero_transfer")
                if current_scores is not None:
                    event["scores"] = current_scores.copy()
                else:
                    event.update(settlement_labels_missing=True, settlement_missing_fields=["scores"])
            emit(event, record)
            round_open = False
        end_scores = data.get("gameend_scores")
        if end_scores:
            if name not in {"lq.RecordHule", "lq.RecordLiuJu"}:
                raise RecordError("GameEnd scores are not defined for this server record")
            final = _scores(end_scores, "gameend scores")
            if current_scores is not None and final != current_scores:
                raise RecordError("GameEnd scores disagree with the terminal settlement")
            terminal_match = True
        if not round_open:
            emit({"type": "end_kyoku"}, record)
    if round_open:
        raise RecordError("Truncated record stream: final round is unresolved")
    if terminal_match:
        events.append({"type": "end_game"})
    return tuple(events)
