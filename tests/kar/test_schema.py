import copy
import json
import math
import struct

import pytest

from riichienv.kar.actions import action_spec
from riichienv.kar.schema import (
    ACTIVE_SCHEMA_ID,
    SCHEMA_CANONICAL_SHA256,
    TOKEN_BYTE_SIZE,
    decode_token,
    encode_token,
    load_schema,
    schema_hash,
    schema_status,
    token_defaults,
    validate_token,
)


def _row(kind, stage=None):  # noqa: PLR0915
    block = 1 if 2 <= kind <= 8 else 2 if kind == 9 else 3 if kind == 10 else 0
    row = token_defaults(kind, block=block)
    if kind in {2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29}:
        row["actor"] = 1
    if kind == 2:
        row.update(stage=1, wind=1)
        _numeric(row, 3, 0)
        _numeric(row, 4, 0)
        _numeric(row, 6, 1 / 12)
        _numeric(row, 19, 0)
    elif kind == 3:
        row["wind"] = 1
        for index, value in ((0, 0.5), (2, 0), (7, 0)):
            _numeric(row, index, value)
    elif kind == 4:
        row["tiles"][0] = 2
        _numeric(row, 8, 0)
        _numeric(row, 9, 0)
    elif kind == 5:
        row.update(stage=9, source=2, slot=1)
        row["tiles"][:3] = [2] * 3
        row["flags"][:2] = [2, 2]
    elif kind == 6:
        row["tiles"][0], row["slot"] = 2, 1
    elif kind == 7:
        row["stage"] = 1
    elif kind == 8:
        _numeric(row, 16, 1 / 32)
    elif kind == 10:
        row.update(action_ids=0, action_type_ids=1, action_tile37=1, action_tsumogiri=1, action_riichi=1)
    elif kind == 11:
        row.update(stage=1, wind=1)
        row["tiles"][0] = 2
        for index, value in ((3, 0), (4, 0), (6, 1 / 12)):
            _numeric(row, index, value)
    elif kind == 12:
        row["tiles"][0], row["stage"] = 2, 1
    elif kind == 13:
        row.update(actor=2, stage=1)
        row["tiles"][0] = 38
    elif kind == 14:
        row["tiles"][0] = 2
    elif kind in {15, 16, 17}:
        row["source"] = 4 if kind == 15 else 2
        row["tiles"][:3] = [2, 3, 4] if kind == 15 else [2] * 3
        if kind == 17:
            row["tiles"][3] = 2
    elif kind in {18, 19}:
        row["tiles"][:4] = [2] * 4
        row["flags"][1] = 2
    elif kind == 20:
        row["tiles"][0] = 2
    elif kind == 21:
        row["source"] = 2
        row["tiles"][:4] = [2] * 4
        row["flags"][:2] = [2, 2]
    elif kind == 23:
        _numeric(row, 1, -0.1)
    elif kind == 24:
        row.update(stage=1, slot=1)
        row["tiles"][0] = 2
    elif kind == 25:
        _numeric(row, 0, 0.5)
        _numeric(row, 1, 0)
    elif kind == 27:
        row.update(stage=2, source=2)
        row["tiles"][0] = 2
    elif kind == 29:
        row.update(stage=1, slot=1)
        row["tiles"][:3] = [2, 3, 4]
    if stage is not None:
        row["stage"] = stage
    return row


def _numeric(row, index, value):
    row["numeric"][index] = value
    row["numeric_present"][index] = True


def test_complete_registry_and_unchanged_tensor_contract():
    schema = load_schema()
    assert schema["schema_id"] == ACTIVE_SCHEMA_ID == "kar-riichienv-v1"
    assert schema["parent_registry"] is None
    assert schema["legacy_compatibility"] == "not_claimed"
    assert len(schema["kinds"]) == 48
    assert set(schema["numeric_fields"]) == {str(i) for i in range(20)}
    assert sum(item["rows"] * item["tables"] for item in schema["embedding_tables"]) == 944
    assert sum(count for _, count in schema["sequence"]["snapshot_layout"]) == 62
    assert schema["numeric_projection"]["input_width"] == 40
    assert schema["numeric_projection"]["bias"] is False
    assert schema["embedding_tables"][3] == {"field": "tiles", "tables": 5, "rows": 39}
    assert schema_hash(schema) == SCHEMA_CANONICAL_SHA256
    assert schema_status()["status"] == "PASS"
    assert schema_status()["compatibility"] == "not_claimed"


@pytest.mark.parametrize("kind", range(30))
def test_every_defined_kind_has_an_encodable_profile(kind):
    row = _row(kind)
    validate_token(row)
    payload = encode_token(row)
    assert len(payload) == TOKEN_BYTE_SIZE == 139
    assert encode_token(decode_token(payload)) == payload


@pytest.mark.parametrize("global_id", range(546))
def test_all_action_id_factorizations_are_verified(global_id):
    spec = action_spec(global_id)
    row = _row(10)
    types = {
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
    row.update(
        action_ids=global_id,
        action_type_ids=types[spec.action_type],
        action_tile37=0,
        action_tsumogiri=0,
        action_riichi=0,
    )
    if spec.action_type == "DISCARD":
        row.update(
            action_tile37=spec.tile37 + 1, action_tsumogiri=1 + int(spec.tsumogiri), action_riichi=1 + int(spec.riichi)
        )
    assert decode_token(encode_token(row))["action_ids"] == global_id
    row["action_type_ids"] = 11
    with pytest.raises(ValueError):
        validate_token(row)


@pytest.mark.parametrize("kind", range(30, 48))
def test_reserved_kinds_cannot_be_written(kind):
    row = _row(1)
    row["kind"] = kind
    with pytest.raises(ValueError, match="Reserved"):
        validate_token(row)
    with pytest.raises(ValueError):
        token_defaults(kind)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", True),
        ("actor", True),
        ("source", -1),
        ("stage", 16),
        ("wind", 5),
        ("slot", 17),
        ("action_ids", 547),
        ("action_type_ids", True),
        ("action_tile37", 39),
        ("action_tsumogiri", 3),
        ("action_riichi", -1),
        ("block", 4),
        ("positions", 1 << 32),
        ("valid", 1),
    ],
)
def test_scalar_dtypes_and_bounds_are_strict(field, value):
    row = _row(1)
    row[field] = value
    with pytest.raises(ValueError):
        validate_token(row)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tiles", [0] * 4),
        ("tiles", [True] * 5),
        ("tiles", [39] * 5),
        ("flags", [0] * 15),
        ("flags", [3] * 16),
        ("flags", [False] * 16),
        ("numeric", [0] * 19),
        ("numeric_present", [False] * 19),
        ("numeric_present", [0] * 20),
    ],
)
def test_vector_shapes_and_dtypes_are_strict(field, value):
    row = _row(1)
    row[field] = value
    with pytest.raises(ValueError):
        validate_token(row)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1e100, True, "0"])
def test_nonfinite_overflow_and_fake_numeric_dtypes_are_rejected(value):
    row = _row(3)
    row["numeric"][0] = value
    with pytest.raises(ValueError):
        encode_token(row)


def test_unknown_numeric_is_not_real_zero_and_never_clipped():
    zero = _row(3)
    absent = copy.deepcopy(zero)
    absent["numeric_present"][12] = False
    _numeric(zero, 12, 0)
    assert encode_token(zero) != encode_token(absent)
    absent["numeric"][12] = 0.1
    with pytest.raises(ValueError, match="Unknown"):
        encode_token(absent)
    round_row = _row(2)
    _numeric(round_row, 3, 9.5)
    assert decode_token(encode_token(round_row))["numeric"][3] == 9.5
    _numeric(round_row, 5, 1.1)
    with pytest.raises(ValueError, match="never clipped"):
        encode_token(round_row)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_rank_gap_presence_keeps_actual_zero(rank):
    row = _row(2, rank)
    for index, present in ((18, rank != 1), (19, rank != 4)):
        row["numeric_present"][index] = present
    validate_token(row)
    index = 18 if rank == 1 else 19 if rank == 4 else 18
    row["numeric_present"][index] = not row["numeric_present"][index]
    with pytest.raises(ValueError, match="Rank-gap"):
        validate_token(row)


def test_stage_is_kind_specific_and_unknown_fields_cannot_smuggle_labels():
    context = _row(7, 2)
    round_row = _row(2, 2)
    round_row["numeric_present"][18] = True
    validate_token(context)
    validate_token(round_row)
    for row in (context, round_row):
        row["response_target"] = 0
        with pytest.raises(ValueError, match="labels"):
            validate_token(row)
    bad = _row(3)
    bad["stage"] = 8
    with pytest.raises(ValueError):
        validate_token(bad)
    other = _row(3)
    other["actor"] = 2
    _numeric(other, 17, 0.1)
    with pytest.raises(ValueError):
        validate_token(other)


@pytest.mark.parametrize("index", range(8, 16))
def test_hand_flags_reserved_positions_remain_na(index):
    row = _row(4)
    row["flags"][index] = 1
    with pytest.raises(ValueError, match="reserved"):
        validate_token(row)


def test_hand_count_identity_red_counts_and_honor_shapes():
    row = _row(4)
    row["tiles"][0] = 1
    with pytest.raises(ValueError):
        validate_token(row)
    row["tiles"][0] = 16
    _numeric(row, 8, 0.5)
    _numeric(row, 9, 1)
    _numeric(row, 10, 0.75)
    _numeric(row, 11, 1)
    validate_token(row)
    _numeric(row, 10, 0.25)
    with pytest.raises(ValueError, match="Known count"):
        validate_token(row)
    honor = _row(4)
    honor["tiles"][0] = 31
    honor["flags"][6] = 2
    with pytest.raises(ValueError, match="Honor"):
        validate_token(honor)


@pytest.mark.parametrize("kind,stage", [(5, 11), (18, 0), (19, 0)])
def test_opponent_five_ankan_never_reveals_red_composition(kind, stage):
    row = _row(kind, stage)
    row.update(actor=2, source=0)
    if kind == 5:
        row["flags"][0] = 1
    row["flags"][1] = 1
    row["tiles"][:4] = [16] * 4
    validate_token(row)
    row["tiles"][0] = 11
    with pytest.raises(ValueError, match="Opponent"):
        validate_token(row)
    row["tiles"][0] = 16
    row["flags"][1] = 2
    with pytest.raises(ValueError, match="Opponent"):
        validate_token(row)


@pytest.mark.parametrize("stage", [8, 9, 10, 11, 12])
def test_all_committed_meld_stages_have_checked_shape(stage):
    row = _row(5, stage)
    row["tiles"] = [2, 3, 4, 0, 0] if stage == 8 else [2, 2, 2, 0, 0] if stage == 9 else [2, 2, 2, 2, 0]
    row["source"] = 0 if stage == 11 else 4 if stage == 8 else 2
    row["flags"][0] = 1 if stage == 11 else 2
    validate_token(row)
    row["tiles"][1] = 8
    with pytest.raises(ValueError):
        validate_token(row)


@pytest.mark.parametrize("slot,count", [(1, 5), (2, 5), (3, 3)])
def test_initial_hand_chunks_are_exactly_thirteen_values(slot, count):
    row = _row(11)
    row.update(stage=2, actor=1, wind=0, slot=slot)
    row["numeric"] = [0.0] * 20
    row["numeric_present"] = [False] * 20
    row["tiles"] = [2, 3, 4, 5, 6][:count] + [0] * (5 - count)
    validate_token(row)
    row["tiles"][count - 1] = 0
    with pytest.raises(ValueError):
        validate_token(row)


def test_public_chunks_and_winning_sources_are_canonical():
    row = _row(29)
    row["tiles"] = [2, 0, 3, 0, 0]
    with pytest.raises(ValueError):
        validate_token(row)
    win = _row(27)
    validate_token(win)
    win["stage"] = 1
    with pytest.raises(ValueError):
        validate_token(win)
    win["source"] = 1
    validate_token(win)


def test_binary_layout_float32_roundtrip_and_negative_zero():
    row = _row(3)
    row["numeric"][0] = 0.50000001
    packed = encode_token(row)
    assert packed[0:3] == bytes((3, 1, 0))
    assert struct.unpack_from("<H", packed, 127)[0] == 546
    expected = struct.unpack("<f", struct.pack("<f", 0.50000001))[0]
    assert decode_token(packed)["numeric"][0] == expected
    row["numeric"][0] = -0.0
    packed = encode_token(row)
    assert packed[27:31] == b"\0\0\0\0"
    negative_zero = packed[:27] + b"\0\0\0\x80" + packed[31:]
    with pytest.raises(ValueError, match="noncanonical"):
        decode_token(negative_zero)


@pytest.mark.parametrize("offset", [107, 126, 138])
def test_binary_boolean_bytes_are_not_truthiness_cast(offset):
    packed = bytearray(encode_token(_row(1)))
    packed[offset] = 2
    with pytest.raises(ValueError, match="booleans"):
        decode_token(bytes(packed))


@pytest.mark.parametrize("data", [b"", b"\0" * 138, b"\0" * 140, bytearray(139)])
def test_binary_size_and_payload_type_are_strict(data):
    with pytest.raises(ValueError):
        decode_token(data)


def test_schema_hash_is_canonical_and_not_a_self_hash(tmp_path):
    document = load_schema()
    changed_order = dict(reversed(list(document.items())))
    assert schema_hash(changed_order) == schema_hash(document)
    path = tmp_path / "copied.json"
    path.write_text(json.dumps(changed_order, ensure_ascii=False, indent=3), encoding="utf-8")
    assert schema_status(path)["status"] == "PASS"
    assert schema_status(tmp_path / "missing.json")["status"] == "BLOCKED"
    document["schema_hash"] = "not-allowed"
    with pytest.raises(ValueError):
        schema_hash(document)
    with pytest.raises(ValueError):
        schema_hash({"numeric": math.nan})


@pytest.mark.parametrize("mutation", ["missing_kind", "renamed_semantics", "wrong_rows", "claim_parent", "identity"])
def test_incomplete_or_reinterpreted_schema_cannot_pass(tmp_path, mutation):
    document = load_schema()
    if mutation == "missing_kind":
        document["kinds"].pop("29")
    elif mutation == "renamed_semantics":
        document["numeric_fields"]["0"]["name"] = "future_score"
    elif mutation == "wrong_rows":
        document["embedding_tables"][3]["tables"] = 1
    elif mutation == "claim_parent":
        document["parent_registry"] = "unavailable-v1.2.1"
    else:
        document["schema_id"] = "kar-v1.2.2"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert schema_status(path)["status"] == "FAIL"


def test_schema_json_rejects_duplicate_keys_even_when_the_final_value_is_correct(tmp_path):
    source = json.dumps(load_schema())
    duplicate = source.replace(
        '"schema_id": "kar-riichienv-v1"', '"schema_id":"wrong","schema_id":"kar-riichienv-v1"', 1
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    assert schema_status(path)["status"] == "FAIL"


def test_chunks_and_chi_sources_cannot_use_noncanonical_combinations():
    chi = _row(15)
    chi["source"] = 2
    with pytest.raises(ValueError, match="previous"):
        validate_token(chi)
    reveal = _row(29)
    reveal["tiles"][:3] = [4, 3, 2]
    with pytest.raises(ValueError, match="ordered"):
        validate_token(reveal)
    deposit = _row(23)
    deposit["numeric"][1] = struct.unpack("<f", struct.pack("<I", 0xBDCCCCCE))[0]
    with pytest.raises(ValueError, match="-1000"):
        validate_token(deposit)


def test_padding_preserves_bookkeeping_but_contains_no_model_features():
    row = token_defaults(block=1, position=731)
    assert decode_token(encode_token(row))["positions"] == 731
    row["actor"] = 1
    with pytest.raises(ValueError):
        validate_token(row)


def test_loaded_documents_and_default_lists_do_not_mutate_the_private_registry():
    document = load_schema()
    document["kinds"]["2"]["actor"] = [4]
    assert load_schema()["kinds"]["2"]["actor"] == [1]
    a, b = token_defaults(), token_defaults()
    a["flags"][0] = 2
    assert b["flags"][0] == 0
    assert schema_status()["status"] == "PASS"
