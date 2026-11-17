from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from .actions import action_spec, tile37_to_base34
from .contracts import REFERENCE_SCHEMA_ID, SCHEMA_OVERLAY_SHA256

ACTIVE_SCHEMA_ID = "kar-riichienv-v1"
SCHEMA_CANONICAL_SHA256 = "fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673"
_SCHEMA_NAME = "token_schema.riichienv.v1.json"
_FIELDS = (
    "kind",
    "actor",
    "source",
    "tiles",
    "stage",
    "wind",
    "slot",
    "flags",
    "numeric",
    "numeric_present",
    "action_ids",
    "action_type_ids",
    "action_tile37",
    "action_tsumogiri",
    "action_riichi",
    "block",
    "positions",
    "valid",
)
_KIND_NAMES = (
    "PAD",
    "BOS",
    "ROUND_STATE",
    "PLAYER_STATE",
    "HAND_COUNT",
    "MELD_STATE",
    "DORA_STATE",
    "DECISION_CONTEXT",
    "LEGAL_SET",
    "DECIDE",
    "ACTION",
    "ROUND_START",
    "DRAW_SELF",
    "DRAW_OTHER",
    "DISCARD",
    "CHI_ACCEPT",
    "PON_ACCEPT",
    "DAIMINKAN_ACCEPT",
    "ANKAN_DECLARE",
    "ANKAN_ACCEPT",
    "KAKAN_DECLARE",
    "KAKAN_ACCEPT",
    "RIICHI_DECLARE",
    "RIICHI_ACCEPT",
    "DORA_REVEAL",
    "SCORE_DELTA",
    "SELF_PASS",
    "WIN_RESOLVED",
    "ROUND_END",
    "PUBLIC_HAND_REVEAL",
)
_CATEGORIES = {
    "kind": 48,
    "actor": 5,
    "source": 5,
    "stage": 16,
    "wind": 5,
    "slot": 17,
    "action_ids": 547,
    "action_type_ids": 13,
    "action_tile37": 39,
    "action_tsumogiri": 3,
    "action_riichi": 3,
    "block": 4,
}
_EMBEDDINGS = (
    ("kind", 1, 48),
    ("actor", 1, 5),
    ("source", 1, 5),
    ("tiles", 5, 39),
    ("stage", 1, 16),
    ("wind", 1, 5),
    ("slot", 1, 17),
    ("flags", 16, 3),
    ("action_ids", 1, 547),
    ("action_type_ids", 1, 13),
    ("action_tile37", 1, 39),
    ("action_tsumogiri", 1, 3),
    ("action_riichi", 1, 3),
)
_ACTION_TYPES = {
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
_ACTION_FIELDS = ("action_ids", "action_type_ids", "action_tile37", "action_tsumogiri", "action_riichi")
ACTION_FACTOR_FIELDS = _ACTION_FIELDS
_REAL_TILES = frozenset(range(1, 38))
_RED_TILES = frozenset((1, 11, 21))
_BASE_TILES = _REAL_TILES - _RED_TILES
_TOKEN_STRUCT = struct.Struct("<" + "B" * 27 + "f" * 20 + "B" * 20 + "H" + "B" * 5 + "I" + "B")
TOKEN_BYTE_SIZE = _TOKEN_STRUCT.size


def action_factor_ids(action_id: int) -> tuple[int, int, int, int, int]:
    """Frozen ACTION factors, shared by suffix encoding and LEGAL_SET pooling."""
    spec = action_spec(action_id)
    family = _ACTION_TYPES[spec.action_type]
    if spec.action_type != "DISCARD":
        return action_id, family, 0, 0, 0
    assert spec.tile37 is not None and spec.tsumogiri is not None and spec.riichi is not None
    return action_id, family, spec.tile37 + 1, 1 + int(spec.tsumogiri), 1 + int(spec.riichi)


def schema_hash(schema: dict[str, Any]) -> str:
    if not isinstance(schema, dict) or "schema_hash" in schema:
        raise ValueError("A schema must be an object without an embedded schema_hash")
    encoded = json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_document(schema: dict[str, Any]) -> None:
    if schema.get("schema_id") != ACTIVE_SCHEMA_ID or schema.get("schema_version") != 1:
        raise ValueError("Unexpected active schema identity")
    if schema.get("parent_registry") is not None or schema.get("legacy_compatibility") != "not_claimed":
        raise ValueError("The new registry must not claim missing legacy compatibility")
    if (
        schema.get("reference_design_schema_id") != REFERENCE_SCHEMA_ID
        or schema.get("reference_overlay_sha256") != SCHEMA_OVERLAY_SHA256
    ):
        raise ValueError("Schema design/overlay references differ; they are not aliases for the host identity")
    if set(schema.get("fields", {})) != set(_FIELDS):
        raise ValueError("The schema must define every token field")
    kinds = schema.get("kinds", {})
    if set(kinds) != {str(index) for index in range(48)}:
        raise ValueError("The schema must define the complete 48-kind registry")
    for index, name in enumerate(_KIND_NAMES):
        if kinds[str(index)].get("name") != name or kinds[str(index)].get("reserved", False):
            raise ValueError(f"Incorrect active kind {index}")
    if any(kinds[str(index)].get("reserved") is not True for index in range(30, 48)):
        raise ValueError("Kinds 30..47 must remain reserved")
    actual_tables = tuple((item["field"], item["tables"], item["rows"]) for item in schema.get("embedding_tables", []))
    if actual_tables != _EMBEDDINGS or schema.get("input_embedding_rows") != 944:
        raise ValueError("Embedding tables must preserve the independent 944-row contract")
    if schema.get("numeric_projection") != {
        "input_width": 40,
        "operation": "concat(numeric,numeric_present)",
        "bias": False,
    }:
        raise ValueError("Numeric projection contract changed")
    if (
        schema.get("sequence", {}).get("snapshot_slots") != 62
        or schema.get("binary_contract", {}).get("token_bytes") != TOKEN_BYTE_SIZE
    ):
        raise ValueError("Snapshot or binary token size changed")
    if schema_hash(schema) != SCHEMA_CANONICAL_SHA256:
        raise ValueError("Schema content differs from the frozen canonical hash")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate schema object key: {key}")
        result[key] = value
    return result


def load_schema(path: Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else Path(__file__).with_name(_SCHEMA_NAME)
    schema = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(schema, dict):
        raise ValueError("Schema root must be an object")
    _validate_document(schema)
    return schema


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    return load_schema()


def schema_status(path: Path | None = None) -> dict[str, Any]:
    try:
        schema = load_schema(path)
    except FileNotFoundError:
        return {"status": "BLOCKED", "reason": "complete_schema_file_missing", "schema_id": ACTIVE_SCHEMA_ID}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {
            "status": "FAIL",
            "reason": "complete_schema_invalid",
            "detail": str(exc),
            "schema_id": ACTIVE_SCHEMA_ID,
        }
    return {
        "status": "PASS",
        "schema_id": schema["schema_id"],
        "schema_hash": schema_hash(schema),
        "embedding_rows": schema["input_embedding_rows"],
        "compatibility": schema["legacy_compatibility"],
        "reference_design_schema_id": schema["reference_design_schema_id"],
        "reference_overlay_sha256": schema["reference_overlay_sha256"],
        "complete_kind_registry": True,
        "token_bytes": TOKEN_BYTE_SIZE,
    }


def _integer(value: Any, limit: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < limit:
        raise ValueError(f"{field} must be an integer in [0,{limit})")
    return value


def _vector(value: Any, length: int, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} entries")
    return value


def _float32(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric entries must be real float32-compatible numbers")
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("numeric entries must be finite")
        number = struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("numeric entry overflows float32") from exc
    if not math.isfinite(number):
        raise ValueError("numeric entry overflows float32")
    return 0.0 if number == 0 else number


def token_defaults(kind: int = 0, *, block: int = 0, position: int = 0, valid: bool | None = None) -> dict[str, Any]:
    _integer(kind, 30, "writable kind")
    _integer(block, 4, "block")
    _integer(position, 1 << 32, "positions")
    if valid is not None and not isinstance(valid, bool):
        raise ValueError("valid must be boolean")
    defaults = _registry()["fields"]
    row = {
        field: list(item["default"]) if isinstance(item["default"], list) else item["default"]
        for field, item in defaults.items()
    }
    row.update(kind=kind, block=block, positions=position, valid=kind != 0 if valid is None else valid)
    return row


def _profile(row: Mapping[str, Any]) -> Mapping[str, Any]:
    definition = _registry()["kinds"][str(row["kind"])]
    if row["block"] not in definition["blocks"] or row["stage"] not in definition["stages"]:
        raise ValueError("Block/stage is not defined for this kind")
    return definition.get("stage_profiles", {}).get(str(row["stage"]), definition)


def _field_usage(row: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    for field in ("actor", "source", "wind", "slot"):
        if row[field] not in profile[field]:
            raise ValueError(f"{field} is not defined for this kind/stage")
    modes = profile["tiles"]
    permitted = {"real": _REAL_TILES, "optional_real": _REAL_TILES | {0}, "hidden": {38}, "base34": _BASE_TILES}
    for index, tile in enumerate(row["tiles"]):
        if tile not in (permitted[modes[index]] if index < len(modes) else {0}):
            raise ValueError(f"tiles[{index}] is not defined for this kind/stage")
    for index, flag in enumerate(row["flags"]):
        if str(index) not in profile["flags"] and flag != 0:
            raise ValueError(f"flags[{index}] is reserved for this kind/stage")
    for index, (value, present) in enumerate(zip(row["numeric"], row["numeric_present"])):
        if index not in profile["numeric"] and (present or value != 0):
            raise ValueError(f"numeric[{index}] is not applicable to this kind/stage")
        if index in profile["required_numeric"] and not present:
            raise ValueError(f"numeric[{index}] must be present for this kind/stage")


def _validate_action(row: Mapping[str, Any]) -> None:
    if row["kind"] != 10:
        defaults = _registry()["fields"]
        if any(row[field] != defaults[field]["default"] for field in _ACTION_FIELDS):
            raise ValueError("Action factors may only appear on ACTION tokens")
        return
    if row["action_ids"] == 546:
        raise ValueError("ACTION requires a complete global action ID")
    spec = action_spec(row["action_ids"])
    if row["action_type_ids"] != _ACTION_TYPES[spec.action_type]:
        raise ValueError("Action type disagrees with the complete action ID")
    actual = row["action_tile37"], row["action_tsumogiri"], row["action_riichi"]
    if spec.action_type == "DISCARD":
        assert spec.tile37 is not None and spec.tsumogiri is not None and spec.riichi is not None
        expected = spec.tile37 + 1, 1 + int(spec.tsumogiri), 1 + int(spec.riichi)
    else:
        expected = (0, 0, 0)
    if actual != expected:
        raise ValueError("Action tile/tsumogiri/riichi factors disagree with the complete action ID")


def _same_meld(row: Mapping[str, Any], family: str, count: int) -> None:
    tiles = row["tiles"]
    if any(tile not in _REAL_TILES for tile in tiles[:count]) or any(tiles[count:]):
        raise ValueError("Meld token has an incorrect number of visible tiles")
    bases = sorted(tile37_to_base34(tile - 1) for tile in tiles[:count])
    if family == "chi":
        if row["source"] and row["source"] != (row["actor"] + 2) % 4 + 1:
            raise ValueError("CHI must claim the previous player's discard")
        if bases[0] >= 27 or bases[-1] // 9 != bases[0] // 9 or bases != list(range(bases[0], bases[0] + 3)):
            raise ValueError("CHI token is not a same-suit sequence")
    elif len(set(bases)) != 1:
        raise ValueError("Triplet/kan token mixes base34 types")
    if sum(tile in _RED_TILES for tile in tiles[:count]) > 1:
        raise ValueError("Meld token has more than one red five")
    if family == "ankan" and row["actor"] != 1 and bases[0] in {4, 13, 22}:
        if any(tile in _RED_TILES for tile in tiles[:count]) or row["flags"][1] == 2:
            raise ValueError("Opponent ankan red composition must remain hidden")


def _cross_fields(row: Mapping[str, Any]) -> None:  # noqa: PLR0915
    kind, stage = row["kind"], row["stage"]
    if kind == 2:
        for index, required in ((18, stage != 1), (19, stage != 4)):
            if row["numeric_present"][index] is not required:
                raise ValueError("Rank-gap presence disagrees with current rank")
    elif kind == 3:
        if row["block"] != (1 if stage == 0 else 0):
            raise ValueError("PLAYER_STATE stage/block disagree")
        if row["actor"] != 1 and row["numeric_present"][17]:
            raise ValueError("Observer draw count cannot be written on another player")
        if row["actor"] == 1 and row["numeric_present"][2] and row["numeric"][2] != 0:
            raise ValueError("Observer's score difference from self must be zero")
    elif kind == 4:
        base = tile37_to_base34(row["tiles"][0] - 1)
        if base >= 27 and any(row["flags"][3:7]):
            raise ValueError("Honor tiles have no suji/Ryanmen flags")
        for index, scale in ((8, 4), (9, 1), (10, 4), (11, 1)):
            if row["numeric_present"][index] and not float(row["numeric"][index] * scale).is_integer():
                raise ValueError("Tile counts must identify whole physical copies")
        if base not in {4, 13, 22} and any(row["numeric"][index] for index in (9, 11)):
            raise ValueError("Only five tile types can have red counts")
        if row["numeric"][9] > row["numeric"][8] * 4:
            raise ValueError("Own red count exceeds own tile count")
        if row["numeric_present"][10] and row["numeric"][10] < row["numeric"][8]:
            raise ValueError("Known count cannot be less than own count")
        if row["numeric_present"][11] and row["numeric"][11] < row["numeric"][9]:
            raise ValueError("Known red count cannot be less than own red count")
        if row["numeric_present"][10] and row["numeric"][11] > row["numeric"][10] * 4:
            raise ValueError("Known red count exceeds known tile count")
    elif kind == 5:
        family = {8: "chi", 9: "pon", 10: "daiminkan", 11: "ankan", 12: "kakan"}[stage]
        _same_meld(row, family, 3 if stage in {8, 9} else 4)
        if row["flags"][0] != (1 if stage == 11 else 2):
            raise ValueError("Committed meld opened flag disagrees with its type")
        if stage == 11 and row["source"] != 0:
            raise ValueError("Ankan has no external source")
        if stage != 11 and row["source"] == row["actor"]:
            raise ValueError("An open meld cannot be called from self")
    elif kind == 8:
        count = row["numeric"][16] * 32
        if not 1 <= count <= 32 or not float(count).is_integer():
            raise ValueError("Complete legal candidate count must be 1..32")
    elif kind == 11 and stage == 2:
        count = 3 if row["slot"] == 3 else 5
        if any(tile not in _REAL_TILES for tile in row["tiles"][:count]) or any(row["tiles"][count:]):
            raise ValueError("Initial hand chunks must contain 5,5,3 tiles")
        if tuple(row["tiles"][:count]) != tuple(sorted(row["tiles"][:count])):
            raise ValueError("Initial hand chunks must be ordered by tile37")
    elif kind in {15, 16, 17, 18, 19, 21}:
        family = {15: "chi", 16: "pon", 17: "daiminkan", 18: "ankan", 19: "ankan", 21: "kakan"}[kind]
        _same_meld(row, family, 3 if kind in {15, 16} else 4)
        if kind in {15, 16, 17} and row["source"] == row["actor"]:
            raise ValueError("Accepted call cannot claim from self")
        if kind == 21 and (row["flags"][0] != 2 or row["source"] == row["actor"]):
            raise ValueError("Committed kakan must be an open meld with an external or unknown source")
    elif kind == 23:
        if _float32(row["numeric"][1]) != _float32(-0.1):
            raise ValueError("RIICHI_ACCEPT must record the actual -1000 point deposit")
    elif kind == 27:
        if (row["actor"] == row["source"]) is not (stage == 1):
            raise ValueError("Win source disagrees with tsumo/ron stage")
    elif kind == 29:
        tiles = row["tiles"]
        used = sum(tile != 0 for tile in tiles)
        if used == 0 or any(tile == 0 for tile in tiles[:used]) or any(tiles[used:]):
            raise ValueError("Public hand chunks require nonempty right-padded tiles")
        if tuple(tiles[:used]) != tuple(sorted(tiles[:used])):
            raise ValueError("Public hand chunks must be ordered by tile37")


def validate_token(row: Mapping[str, Any]) -> None:
    if not isinstance(row, Mapping) or set(row) != set(_FIELDS):
        raise ValueError("Token fields must exactly match the complete schema; labels and unknown fields are forbidden")
    for field, limit in _CATEGORIES.items():
        _integer(row[field], limit, field)
    _integer(row["positions"], 1 << 32, "positions")
    kind = row["kind"]
    if kind >= 30:
        raise ValueError("Reserved kinds cannot be written")
    if not isinstance(row["valid"], bool) or row["valid"] is not (kind != 0):
        raise ValueError("Only PAD tokens may be invalid")
    for tile in _vector(row["tiles"], 5, "tiles"):
        _integer(tile, 39, "tile")
    for flag in _vector(row["flags"], 16, "flags"):
        _integer(flag, 3, "flag")
    values = _vector(row["numeric"], 20, "numeric")
    presence = _vector(row["numeric_present"], 20, "numeric_present")
    for index, (value, present) in enumerate(zip(values, presence)):
        _float32(value)
        if not isinstance(present, bool):
            raise ValueError("numeric_present entries must be boolean")
        if not present and value != 0:
            raise ValueError("Unknown numeric entries must be zero before encoding")
        domain = _registry()["numeric_fields"][str(index)]
        if present and (
            (domain["minimum"] is not None and value < domain["minimum"])
            or (domain["maximum"] is not None and value > domain["maximum"])
        ):
            raise ValueError(f"numeric[{index}] violates its physical domain; values are never clipped")
    _field_usage(row, _profile(row))
    _validate_action(row)
    _cross_fields(row)


def encode_token(row: Mapping[str, Any]) -> bytes:
    validate_token(row)
    return _TOKEN_STRUCT.pack(
        row["kind"],
        row["actor"],
        row["source"],
        *row["tiles"],
        row["stage"],
        row["wind"],
        row["slot"],
        *row["flags"],
        *(_float32(value) for value in row["numeric"]),
        *row["numeric_present"],
        row["action_ids"],
        row["action_type_ids"],
        row["action_tile37"],
        row["action_tsumogiri"],
        row["action_riichi"],
        row["block"],
        row["positions"],
        row["valid"],
    )


def decode_token(data: bytes) -> dict[str, Any]:
    if not isinstance(data, bytes) or len(data) != TOKEN_BYTE_SIZE:
        raise ValueError(f"A binary token must contain exactly {TOKEN_BYTE_SIZE} bytes")
    values = _TOKEN_STRUCT.unpack(data)
    if any(value not in (0, 1) for value in (*values[47:67], values[74])):
        raise ValueError("Binary presence/valid bytes must be canonical booleans")
    row = {
        "kind": values[0],
        "actor": values[1],
        "source": values[2],
        "tiles": list(values[3:8]),
        "stage": values[8],
        "wind": values[9],
        "slot": values[10],
        "flags": list(values[11:27]),
        "numeric": list(values[27:47]),
        "numeric_present": [bool(value) for value in values[47:67]],
        "action_ids": values[67],
        "action_type_ids": values[68],
        "action_tile37": values[69],
        "action_tsumogiri": values[70],
        "action_riichi": values[71],
        "block": values[72],
        "positions": values[73],
        "valid": bool(values[74]),
    }
    validate_token(row)
    if encode_token(row) != data:
        raise ValueError("Binary token uses a noncanonical float32 representation")
    return row
