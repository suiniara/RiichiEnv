from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest
from google.protobuf import descriptor_pb2, message_factory

from riichienv.kar.mjsoul import (
    Protocol,
    ProtocolError,
    RecordError,
    ServerRecord,
    decode_record_bytes,
    load_protocol,
    normalize_records,
)
from riichienv.kar.observation import project_observer

F = descriptor_pb2.FieldDescriptorProto


def _protocol(*, include_ura=True, include_players=True) -> Protocol:
    file = descriptor_pb2.FileDescriptorProto(name="synthetic.proto", package="lq", syntax="proto3")
    definitions = {
        "Wrapper": [("name", 1, F.TYPE_STRING), ("data", 2, F.TYPE_BYTES)],
        "GameUserInput": [("secret", 1, F.TYPE_STRING)],
        "GameAction": [("result", 3, F.TYPE_BYTES), ("user_input", 4, "GameUserInput")],
        "GameDetailRecords": [
            ("records", 1, F.TYPE_BYTES, True),
            ("version", 2, F.TYPE_UINT32),
            ("actions", 3, "GameAction", True),
        ],
        "LiQiSuccess": [
            ("seat", 1, F.TYPE_UINT32),
            ("score", 2, F.TYPE_INT32),
            ("liqibang", 3, F.TYPE_UINT32),
            ("failed", 4, F.TYPE_BOOL),
        ],
        "GameEnd": [("scores", 1, F.TYPE_INT32, True)],
        "RecordNewRound": [
            ("chang", 1, F.TYPE_UINT32),
            ("ju", 2, F.TYPE_UINT32),
            ("ben", 3, F.TYPE_UINT32),
            ("dora", 4, F.TYPE_STRING),
            ("scores", 5, F.TYPE_INT32, True),
            ("liqibang", 6, F.TYPE_UINT32),
            ("tiles0", 7, F.TYPE_STRING, True),
            ("tiles1", 8, F.TYPE_STRING, True),
            ("tiles2", 9, F.TYPE_STRING, True),
            ("tiles3", 10, F.TYPE_STRING, True),
            ("paishan", 14, F.TYPE_STRING),
            ("left_tile_count", 15, F.TYPE_UINT32),
            ("doras", 16, F.TYPE_STRING, True),
        ],
        "RecordDealTile": [
            ("seat", 1, F.TYPE_UINT32),
            ("tile", 2, F.TYPE_STRING),
            ("left_tile_count", 3, F.TYPE_UINT32),
            ("liqi", 5, "LiQiSuccess"),
            ("doras", 6, F.TYPE_STRING, True),
        ],
        "RecordDiscardTile": [
            ("seat", 1, F.TYPE_UINT32),
            ("tile", 2, F.TYPE_STRING),
            ("is_liqi", 3, F.TYPE_BOOL),
            ("moqie", 5, F.TYPE_BOOL),
            ("doras", 8, F.TYPE_STRING, True),
            ("is_wliqi", 9, F.TYPE_BOOL),
        ],
        "RecordChiPengGang": [
            ("seat", 1, F.TYPE_UINT32),
            ("type", 2, F.TYPE_UINT32),
            ("tiles", 3, F.TYPE_STRING, True),
            ("froms", 4, F.TYPE_UINT32, True),
            ("liqi", 5, "LiQiSuccess"),
        ],
        "RecordAnGangAddGang": [
            ("seat", 1, F.TYPE_UINT32),
            ("type", 2, F.TYPE_UINT32),
            ("tiles", 3, F.TYPE_STRING),
            ("doras", 6, F.TYPE_STRING, True),
        ],
        "HuleInfo": [
            ("hand", 1, F.TYPE_STRING, True),
            ("ming", 2, F.TYPE_STRING, True),
            ("hu_tile", 3, F.TYPE_STRING),
            ("seat", 4, F.TYPE_UINT32),
            ("zimo", 5, F.TYPE_BOOL),
            ("doras", 8, F.TYPE_STRING, True),
            ("li_doras", 9, F.TYPE_STRING, True),
            ("count", 11, F.TYPE_UINT32),
            ("fu", 13, F.TYPE_UINT32),
        ],
        "RecordHule": [
            ("hules", 1, "HuleInfo", True),
            ("old_scores", 2, F.TYPE_INT32, True),
            ("delta_scores", 3, F.TYPE_INT32, True),
            ("scores", 5, F.TYPE_INT32, True),
            ("gameend", 6, "GameEnd"),
            ("doras", 7, F.TYPE_STRING, True),
        ],
        "NoTileScoreInfo": [
            ("seat", 1, F.TYPE_UINT32),
            ("old_scores", 2, F.TYPE_INT32, True),
            ("delta_scores", 3, F.TYPE_INT32, True),
        ],
        "TingPaiInfo": [("tile", 1, F.TYPE_STRING), ("private_operation", 2, F.TYPE_STRING)],
        "NoTilePlayerInfo": [
            ("tingpai", 3, F.TYPE_BOOL),
            ("hand", 4, F.TYPE_STRING, True),
            ("tings", 5, "TingPaiInfo", True),
            ("already_hule", 6, F.TYPE_BOOL),
        ],
        "RecordNoTile": [
            ("liujumanguan", 1, F.TYPE_BOOL),
            ("players", 2, "NoTilePlayerInfo", True),
            ("scores", 3, "NoTileScoreInfo", True),
            ("gameend", 4, F.TYPE_BOOL),
        ],
        "RecordLiuJu": [
            ("type", 1, F.TYPE_UINT32),
            ("gameend", 2, "GameEnd"),
            ("seat", 3, F.TYPE_UINT32),
            ("liqi", 5, "LiQiSuccess"),
        ],
        "GameEndPlayer": [("seat", 1, F.TYPE_UINT32), ("total_point", 2, F.TYPE_INT32)],
        "GameEndResult": [("players", 1, "GameEndPlayer", True)],
        "RecordGame": [
            ("uuid", 1, F.TYPE_STRING),
            ("start_time", 2, F.TYPE_UINT32),
            ("end_time", 3, F.TYPE_UINT32),
            ("result", 12, "GameEndResult"),
        ],
        "Error": [("code", 1, F.TYPE_UINT32)],
        "ResGameRecord": [
            ("error", 1, "Error"),
            ("head", 3, "RecordGame"),
            ("data", 4, F.TYPE_BYTES),
            ("data_url", 5, F.TYPE_STRING),
        ],
    }
    for name, fields in definitions.items():
        message = file.message_type.add(name=name)
        for item in fields:
            if name == "HuleInfo" and not include_ura and item[0] == "li_doras":
                continue
            if name == "RecordNoTile" and not include_players and item[0] == "players":
                continue
            field = message.field.add(name=item[0], number=item[1])
            field.label = F.LABEL_REPEATED if len(item) == 4 and item[3] else F.LABEL_OPTIONAL
            if isinstance(item[2], str):
                field.type = F.TYPE_MESSAGE
                field.type_name = ".lq." + item[2]
            else:
                field.type = item[2]
    descriptor = descriptor_pb2.FileDescriptorSet()
    descriptor.file.add().CopyFrom(file)
    return Protocol.from_descriptor_set(descriptor.SerializeToString(), source_name="synthetic_fixture")


@pytest.fixture
def protocol():
    return _protocol()


def _message(protocol, message_name, **values):
    cls = message_factory.GetMessageClass(protocol.pool.FindMessageTypeByName("lq." + message_name))
    return cls(**values)


def _result(protocol, name, **values):
    message = _message(protocol, name, **values)
    return _message(protocol, "Wrapper", name=".lq." + name, data=message.SerializeToString()).SerializeToString()


def _varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _bytes_field(number, value):
    return _varint(number * 8 + 2) + _varint(len(value)) + value


def _new_round():
    deck = []
    for suit in "mps":
        for rank in range(1, 10):
            deck.extend([f"{rank}{suit}"] * 4)
            if rank == 5:
                deck[-4] = "0" + suit
    for rank in range(1, 8):
        deck.extend([f"{rank}z"] * 4)
    deck.remove("0p")
    hands = [deck[index * 13 : (index + 1) * 13] for index in range(4)]
    hands[0].append("0p")
    return {
        "chang": 0,
        "ju": 0,
        "ben": 0,
        "dora": "9s",
        "scores": [25000] * 4,
        "liqibang": 0,
        **{f"tiles{seat}": hand for seat, hand in enumerate(hands)},
        "left_tile_count": 69,
        "paishan": "not-a-model-input",
    }


def _round_at(round_index, honba=0, riichi_sticks=0, scores=None):
    data = _new_round()
    dealer_draw = data["tiles0"].pop()
    dealer = round_index % 4
    data[f"tiles{dealer}"].append(dealer_draw)
    data.update(chang=round_index // 4, ju=dealer, ben=honba, liqibang=riichi_sticks)
    if scores is not None:
        data["scores"] = scores
    return data


def _results(protocol):
    return [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordNoTile", gameend=True),
    ]


def _legacy(protocol, results=None):
    return _message(protocol, "GameDetailRecords", records=results or _results(protocol)).SerializeToString()


def _actions(results, client_input=b"\xff\xff\xff"):
    actions = [_bytes_field(3, result) + _bytes_field(4, client_input) for result in results]
    return _varint(2 * 8) + _varint(210715) + b"".join(_bytes_field(3, action) for action in actions)


def test_legacy_decodes_red_tiles_and_splits_dealer_draw(protocol):
    game = decode_record_bytes(_legacy(protocol), protocol, container="game_detail_records")
    assert game.metadata["record_version"] == 0
    assert game.complete is True
    assert game.events[1]["type"] == "start_kyoku"
    assert all(len(hand) == 13 for hand in game.events[1]["tehais"])
    assert game.events[2]["type"] == "tsumo"
    assert game.events[2]["pai"] == "5pr"
    assert "paishan" not in game.records[0].data
    assert "paishan" not in repr(game.events)
    assert game.protocol_sha256 == protocol.source_sha256
    assert game.source_sha256 == hashlib.sha256(_legacy(protocol)).hexdigest()


def test_actions_only_decode_server_result_and_skip_poisoned_client_input(protocol):
    left = decode_record_bytes(_actions(_results(protocol), b"\xff\xff\xff"), protocol, container="game_detail_records")
    right = decode_record_bytes(
        _actions(_results(protocol), b"anything-private"), protocol, container="game_detail_records"
    )
    assert left.events == right.events
    assert left.records == right.records
    assert left.source_sha256 != right.source_sha256
    assert left.metadata["record_version"] == 210715


def test_resultless_client_action_is_recorded_not_invented_as_next_draw(protocol):
    payload = _actions(_results(protocol)) + _bytes_field(3, _bytes_field(4, b"\xff"))
    game = decode_record_bytes(payload, protocol, container="game_detail_records")
    assert game.metadata["resultless_action_indices"] == [2]
    assert [event["type"] for event in game.events].count("tsumo") == 1


def test_all_resultless_actions_are_rejected(protocol):
    payload = _varint(16) + _varint(210715) + _bytes_field(3, _bytes_field(4, b"\xff"))
    with pytest.raises(RecordError, match="No server results"):
        decode_record_bytes(payload, protocol, container="game_detail_records")


@pytest.mark.parametrize("payload", [b"\x00", b"\x0a\x80", b"\x0a\x05a", b"\x0b", b"\x80" * 11])
def test_invalid_wire_payloads_are_rejected(protocol, payload):
    with pytest.raises(RecordError):
        decode_record_bytes(payload, protocol, container="game_detail_records")


def test_mixed_legacy_actions_and_unknown_version_are_rejected(protocol):
    mixed = _legacy(protocol) + _actions(_results(protocol))
    with pytest.raises(RecordError, match="Ambiguous"):
        decode_record_bytes(mixed, protocol, container="game_detail_records")
    unknown = _varint(16) + _varint(999) + _bytes_field(1, _results(protocol)[0])
    with pytest.raises(RecordError, match="version"):
        decode_record_bytes(unknown, protocol, container="game_detail_records")


def test_unknown_wrapper_is_not_dynamically_dispatched(protocol):
    wrapper = _message(protocol, "Wrapper", name=".lq.UntrustedMessage", data=b"opaque").SerializeToString()
    with pytest.raises(RecordError, match="Wrapper.name"):
        decode_record_bytes(_bytes_field(1, wrapper), protocol, container="game_detail_records")


def test_res_game_record_and_verified_archive_container(protocol):
    detail = _message(protocol, "Wrapper", name=".lq.GameDetailRecords", data=_legacy(protocol)).SerializeToString()
    response = _message(protocol, "ResGameRecord", data=detail).SerializeToString()
    plain = decode_record_bytes(response, protocol)
    outer = _message(protocol, "Wrapper", data=response).SerializeToString()
    archived = decode_record_bytes(b"\0\0\0" + outer, protocol, container="majsoul_download")
    assert plain.events == archived.events
    with pytest.raises(RecordError, match="three-zero"):
        decode_record_bytes(b"\x03\x01\x00" + outer, protocol, container="majsoul_download")
    named = _message(protocol, "Wrapper", name=".lq.Other", data=response).SerializeToString()
    with pytest.raises(RecordError, match="Wrapper.name"):
        decode_record_bytes(b"\0\0\0" + named, protocol, container="majsoul_download")


def test_missing_response_data_and_server_error_are_explicit(protocol):
    response = _message(protocol, "ResGameRecord", data_url="https://unused.invalid").SerializeToString()
    with pytest.raises(RecordError, match="Missing nonempty"):
        decode_record_bytes(response, protocol)
    error = _message(protocol, "ResGameRecord", error={"code": 23}).SerializeToString()
    with pytest.raises(RecordError, match="Server returned error 23"):
        decode_record_bytes(error, protocol)


def test_riichi_deposit_only_after_explicit_liqi_success(protocol):
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="0p", is_liqi=True, moqie=True),
        _result(protocol, "RecordDealTile", seat=1, tile="9s", liqi={"seat": 0, "score": 24000, "liqibang": 1}),
        _result(protocol, "RecordNoTile", gameend=True),
    ]
    game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    types = [event["type"] for event in game.events]
    assert types[3:7] == ["reach", "dahai", "reach_accepted", "tsumo"]
    before = project_observer(list(game.events), 0, types.index("dahai"), (0, 1, 2, 3))
    accepted = project_observer(list(game.events), 0, types.index("reach_accepted"), (0, 1, 2, 3))
    assert before.scores == (25000,) * 4
    assert accepted.scores == (24000, 25000, 25000, 25000)
    terminal = game.events[-3]
    assert "scores" not in terminal and "deltas" not in terminal
    assert terminal["settlement_labels_missing"] is True
    assert game.metadata["source_score_chain_complete"] is False


def test_multiple_ron_winners_share_the_original_trigger(protocol):
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="1m", moqie=False),
        _result(
            protocol,
            "RecordHule",
            hules=[
                {"seat": 1, "hu_tile": "1m", "zimo": False, "count": 1, "fu": 30},
                {"seat": 2, "hu_tile": "1m", "zimo": False, "count": 1, "fu": 30},
            ],
            old_scores=[25000] * 4,
            delta_scores=[-2000, 1000, 1000, 0],
            scores=[23000, 26000, 26000, 25000],
        ),
    ]
    game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    winners = [event for event in game.events if event["type"] == "hora"]
    assert [(event["actor"], event["target"]) for event in winners] == [(1, 0), (2, 0)]
    assert [event["type"] for event in game.events][-1] == "end_kyoku"
    assert game.complete is False


def test_unknown_liuju_is_not_guessed(protocol):
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordLiuJu", type=99),
    ]
    with pytest.raises(RecordError, match="Unfrozen LiuJu"):
        decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")


@pytest.mark.parametrize("liuju_type", [2, 3])
def test_ambiguous_liuju_requires_public_resolution(protocol, liuju_type):
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordLiuJu", type=liuju_type),
    ]
    with pytest.raises(RecordError, match="public winds/kans"):
        decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")


def test_four_winds_reason_is_verified_from_public_discards(protocol):
    round_data = _new_round()
    for seat in range(4):
        round_data[f"tiles{seat}"][0] = "1z"
    results = [_result(protocol, "RecordNewRound", **round_data)]
    for seat in range(4):
        if seat:
            results.append(_result(protocol, "RecordDealTile", seat=seat, tile="9s"))
        results.append(_result(protocol, "RecordDiscardTile", seat=seat, tile="1z", moqie=False))
    results.append(_result(protocol, "RecordLiuJu", type=2))
    game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    assert game.events[-2]["reason"] == "four_winds"


def test_ankan_uses_source_hand_red_identity_without_publishing_wall(protocol):
    round_data = _new_round()
    round_data["tiles0"][:4], round_data["tiles1"][3:7] = round_data["tiles1"][3:7], round_data["tiles0"][:4]
    results = [
        _result(protocol, "RecordNewRound", **round_data),
        _result(protocol, "RecordAnGangAddGang", seat=0, type=3, tiles="5m"),
        _result(protocol, "RecordNoTile", gameend=True),
    ]
    game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    event = next(event for event in game.events if event["type"] == "ankan")
    assert event["consumed"] == ["5mr", "5m", "5m", "5m"]
    assert "paishan" not in game.records[0].data


def test_truncated_round_and_bad_four_player_hands_are_rejected(protocol):
    with pytest.raises(RecordError, match="unresolved"):
        decode_record_bytes(_legacy(protocol, _results(protocol)[:1]), protocol, container="game_detail_records")
    round_data = _new_round()
    round_data["tiles3"] = []
    with pytest.raises(RecordError, match="13-tile"):
        decode_record_bytes(
            _legacy(protocol, [_result(protocol, "RecordNewRound", **round_data), _results(protocol)[1]]),
            protocol,
            container="game_detail_records",
        )


def test_normalizer_rejects_bad_settlement_arithmetic():
    records = [
        ServerRecord(0, "lq.RecordNewRound", "synthetic", _new_round()),
        ServerRecord(1, "lq.RecordDiscardTile", "synthetic", {"seat": 0, "tile": "1m", "moqie": False}),
        ServerRecord(
            2,
            "lq.RecordHule",
            "synthetic",
            {
                "hules": [{"seat": 1, "hu_tile": "1m", "zimo": False, "count": 1, "fu": 30}],
                "old_scores": [25000] * 4,
                "delta_scores": [0] * 4,
                "scores": [24000, 26000, 25000, 25000],
            },
        ),
    ]
    with pytest.raises(RecordError, match="arithmetic"):
        normalize_records(records)


@pytest.mark.parametrize("terminal_kind", ["notile", "hule"])
def test_header_completion_does_not_supply_server_gameend(protocol, terminal_kind):
    results = [_result(protocol, "RecordNewRound", **_new_round())]
    if terminal_kind == "notile":
        results.append(
            _result(
                protocol,
                "RecordNoTile",
                gameend=False,
                scores=[{"old_scores": [25000] * 4, "delta_scores": [0] * 4}],
            )
        )
    else:
        results.extend(
            [
                _result(protocol, "RecordDiscardTile", seat=0, tile="1m", moqie=False),
                _result(
                    protocol,
                    "RecordHule",
                    hules=[{"seat": 1, "hu_tile": "1m", "zimo": False, "count": 1, "fu": 30}],
                    old_scores=[25000] * 4,
                    delta_scores=[-1000, 1000, 0, 0],
                    scores=[24000, 26000, 25000, 25000],
                ),
            ]
        )
    detail = _message(protocol, "Wrapper", name=".lq.GameDetailRecords", data=_legacy(protocol, results))
    response = _message(
        protocol,
        "ResGameRecord",
        data=detail.SerializeToString(),
        head={
            "uuid": "synthetic",
            "start_time": 1,
            "end_time": 2,
            "result": {"players": [{"seat": seat, "total_point": 25000} for seat in range(4)]},
        },
    )
    game = decode_record_bytes(response.SerializeToString(), protocol)
    assert game.metadata["header_reports_completion"] is True
    assert game.metadata["completion_evidence"] == "missing_explicit_server_gameend"
    assert game.events[-1]["type"] == "end_kyoku"
    assert game.complete is False
    assert all(event["type"] != "end_game" for event in game.events)


def test_external_match_complete_override_is_not_an_accepted_argument():
    records = [
        ServerRecord(0, "lq.RecordNewRound", "synthetic", _new_round()),
        ServerRecord(1, "lq.RecordNoTile", "synthetic", {"gameend": False, "scores": []}),
    ]
    with pytest.raises(TypeError, match="match_complete"):
        normalize_records(records, match_complete=True)


@pytest.mark.parametrize("gameend_scores", [None, [], [24000, 26000, 25000, 25000]])
def test_hule_only_explicit_valid_gameend_scores_finish_match(protocol, gameend_scores):
    settlement = {
        "hules": [{"seat": 1, "hu_tile": "1m", "zimo": False, "count": 1, "fu": 30}],
        "old_scores": [25000] * 4,
        "delta_scores": [-1000, 1000, 0, 0],
        "scores": [24000, 26000, 25000, 25000],
    }
    if gameend_scores is not None:
        settlement["gameend"] = {"scores": gameend_scores}
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="1m", moqie=False),
        _result(protocol, "RecordHule", **settlement),
    ]
    game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    assert game.complete is bool(gameend_scores)
    assert (game.events[-1]["type"] == "end_game") is bool(gameend_scores)


@pytest.mark.parametrize("terminal_kind", ["hule", "notile"])
def test_arithmetic_consistent_old_scores_cannot_rewrite_current_scores(protocol, terminal_kind):
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="1m", moqie=False),
    ]
    old = [26000, 24000, 25000, 25000]
    delta = [-1000, 1000, 0, 0]
    if terminal_kind == "hule":
        results.append(
            _result(
                protocol,
                "RecordHule",
                hules=[{"seat": 1, "hu_tile": "1m", "zimo": False, "count": 1, "fu": 30}],
                old_scores=old,
                delta_scores=delta,
                scores=[25000] * 4,
            )
        )
    else:
        results.append(_result(protocol, "RecordNoTile", scores=[{"old_scores": old, "delta_scores": delta}]))
    with pytest.raises(RecordError, match="current known scores"):
        decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")


@pytest.mark.parametrize("score_jump", [False, True])
def test_new_round_scores_must_continue_the_known_settlement(protocol, score_jump):
    round_data = _new_round()
    round_data["scores"] = [25000] * 4 if score_jump else [24000, 26000, 25000, 25000]
    round_data["ben"] = 1
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="1m", moqie=False),
        _result(
            protocol,
            "RecordHule",
            hules=[{"seat": 1, "hu_tile": "1m", "zimo": False, "count": 1, "fu": 30}],
            old_scores=[25000] * 4,
            delta_scores=[-1000, 1000, 0, 0],
            scores=[24000, 26000, 25000, 25000],
        ),
        _result(protocol, "RecordNewRound", **round_data),
        _result(
            protocol,
            "RecordNoTile",
            gameend=False,
            scores=[{"old_scores": round_data["scores"], "delta_scores": [0] * 4}],
        ),
    ]
    if score_jump:
        with pytest.raises(RecordError, match="previous known settlement"):
            decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    else:
        game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
        assert [event for event in game.events if event["type"] == "start_kyoku"][-1]["scores"] == round_data["scores"]


@pytest.mark.parametrize("terminal_kind", ["hule", "notile"])
def test_accepted_riichi_deposit_is_part_of_the_score_chain(protocol, terminal_kind):
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="0p", is_liqi=True, moqie=True),
        _result(protocol, "RecordDealTile", seat=1, tile="9s", liqi={"seat": 0, "score": 24000, "liqibang": 1}),
        _result(protocol, "RecordDiscardTile", seat=1, tile="9s", moqie=True),
    ]
    if terminal_kind == "hule":
        results.append(
            _result(
                protocol,
                "RecordHule",
                hules=[{"seat": 2, "hu_tile": "9s", "zimo": False, "count": 1, "fu": 30}],
                old_scores=[25000] * 4,
                delta_scores=[0] * 4,
                scores=[25000] * 4,
            )
        )
    else:
        results.append(_result(protocol, "RecordNoTile", scores=[{"old_scores": [25000] * 4, "delta_scores": [0] * 4}]))
    with pytest.raises(RecordError, match="current known scores"):
        decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")


def test_gameend_cannot_silently_add_unexplained_points(protocol):
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="1m", moqie=False),
        _result(
            protocol,
            "RecordHule",
            hules=[{"seat": 1, "hu_tile": "1m", "zimo": False, "count": 1, "fu": 30}],
            old_scores=[25000] * 4,
            delta_scores=[-1000, 1000, 0, 0],
            scores=[24000, 26000, 25000, 25000],
            gameend={"scores": [24000, 27000, 25000, 25000]},
        ),
    ]
    with pytest.raises(RecordError, match="GameEnd scores disagree"):
        decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")


@pytest.mark.parametrize(
    "round_index,honba,riichi_sticks,scores",
    [
        (7, 0, 0, [25000] * 4),
        (1, 0, 0, [25000] * 4),
        (0, 1, 0, [25000] * 4),
        (0, 0, 1, [25000] * 4),
        (0, 0, 0, [26000, 24000, 25000, 25000]),
    ],
)
def test_explicit_terminal_result_cannot_make_a_missing_start_complete(
    protocol, round_index, honba, riichi_sticks, scores
):
    results = [
        _result(protocol, "RecordNewRound", **_round_at(round_index, honba, riichi_sticks, scores)),
        _result(protocol, "RecordNoTile", gameend=True),
    ]
    with pytest.raises(RecordError, match="not_initial_round"):
        decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")


@pytest.mark.parametrize(
    "rounds",
    [
        [(0, 0), (2, 0)],
        [(0, 0), (0, 0)],
        [(0, 0), (0, 2)],
        [(0, 0), (1, 2)],
        [(0, 0), (1, 0), (0, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
    ],
)
def test_equal_scores_do_not_hide_missing_or_inconsistent_rounds(protocol, rounds):
    results = []
    for round_index, honba in rounds:
        results.extend(
            [
                _result(protocol, "RecordNewRound", **_round_at(round_index, honba)),
                _result(
                    protocol,
                    "RecordNoTile",
                    gameend=False,
                    scores=[{"old_scores": [25000] * 4, "delta_scores": [0] * 4}],
                ),
            ]
        )
    with pytest.raises(RecordError, match="Round continuity"):
        decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")


@pytest.mark.parametrize(
    "rounds",
    [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    ],
)
def test_conservative_round_continuity_accepts_valid_local_transitions(protocol, rounds):
    results = []
    for round_index, honba in rounds:
        results.extend(
            [
                _result(protocol, "RecordNewRound", **_round_at(round_index, honba)),
                _result(
                    protocol,
                    "RecordNoTile",
                    gameend=False,
                    scores=[{"old_scores": [25000] * 4, "delta_scores": [0] * 4}],
                ),
            ]
        )
    game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    assert len([event for event in game.events if event["type"] == "start_kyoku"]) == len(rounds)
    assert game.complete is False


def test_protocol_contract_mismatch_and_empty_descriptor_are_rejected(protocol):
    with pytest.raises(ProtocolError, match="Unexpected wire contract"):
        protocol.check_field("lq.GameAction", "result", 4, F.TYPE_BYTES)
    with pytest.raises(ProtocolError, match="empty"):
        Protocol.from_descriptor_set(b"")


def test_load_protocol_only_generates_a_descriptor(tmp_path):
    proto = tmp_path / "minimal.proto"
    proto.write_text('syntax = "proto3"; package lq; message Example { string text = 1; }', encoding="utf-8")
    (tmp_path / "minimal_pb2.py").write_text("raise RuntimeError('must not import')", encoding="utf-8")
    protocol = load_protocol(proto)
    assert protocol.message("lq.Example", b"\x0a\x02ok").text == "ok"
    assert protocol.source_sha256 == hashlib.sha256(proto.read_bytes()).hexdigest()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["minimal.proto", "minimal_pb2.py"]


def _ron_results(protocol, *, winner_fields=None, record_doras=None):
    winner = {"seat": 1, "hu_tile": "1m", "zimo": False, "count": 1, "fu": 30}
    winner.update(winner_fields or {})
    settlement = {
        "hules": [winner],
        "old_scores": [25000] * 4,
        "delta_scores": [-1000, 1000, 0, 0],
        "scores": [24000, 26000, 25000, 25000],
    }
    if record_doras is not None:
        settlement["doras"] = record_doras
    return [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="1m", moqie=False),
        _result(protocol, "RecordHule", **settlement),
    ]


@pytest.mark.parametrize("markers,expected", [([], []), (["0p"], ["5pr"]), (["1z", "0m"], ["E", "5mr"])])
def test_hule_ura_facts_are_preserved_only_at_the_winning_event(protocol, markers, expected):
    results = _ron_results(protocol, winner_fields={"li_doras": markers})
    game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    win = next(event for event in game.events if event["type"] == "hora")
    assert win["ura_markers"] == expected
    assert game.records[-1].data["hules"][0]["li_doras"] == markers
    assert all("ura_markers" not in event for event in game.events if event["type"] != "hora")


def test_ura_schema_absence_is_distinct_from_a_known_empty_field():
    old_protocol = _protocol(include_ura=False)
    game = decode_record_bytes(
        _legacy(old_protocol, _ron_results(old_protocol)), old_protocol, container="game_detail_records"
    )
    win = next(event for event in game.events if event["type"] == "hora")
    assert "ura_markers" not in win
    assert "li_doras" not in game.records[-1].data["hules"][0]


def test_hule_ura_facts_do_not_decode_poisoned_user_input(protocol):
    results = _ron_results(protocol, winner_fields={"li_doras": ["0p"]})
    poisoned = decode_record_bytes(_actions(results, b"\xff\xff\xff"), protocol, container="game_detail_records")
    changed = decode_record_bytes(
        _actions(results, b"private-client-decision"), protocol, container="game_detail_records"
    )
    assert poisoned.events == changed.events
    assert poisoned.records == changed.records
    assert "private-client-decision" not in repr(changed.records)
    assert next(event for event in poisoned.events if event["type"] == "hora")["ura_markers"] == ["5pr"]


def test_hule_record_doras_reveal_at_terminal_time_not_the_previous_cutoff(protocol):
    results = _ron_results(
        protocol,
        winner_fields={"doras": ["9s", "1p"], "ming": ["kezi(3p,3p,3p)"], "li_doras": ["0p"]},
        record_doras=["9s", "1p"],
    )
    original = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    changed = decode_record_bytes(
        _legacy(protocol, _ron_results(protocol, record_doras=["9s", "2p"])),
        protocol,
        container="game_detail_records",
    )
    types = [event["type"] for event in original.events]
    assert types[-3:] == ["dora", "hora", "end_kyoku"]
    assert original.events[-3]["dora_marker"] == "1p"
    assert original.records[-1].data["doras"] == ["9s", "1p"]
    assert original.records[-1].data["hules"][0]["ming"] == ["kezi(3p,3p,3p)"]
    assert "ming" not in original.events[-2]
    cutoff = types.index("dahai")
    assert (
        project_observer(list(original.events), 0, cutoff, (0, 1, 2, 3)).to_bytes()
        == project_observer(list(changed.events), 0, cutoff, (0, 1, 2, 3)).to_bytes()
    )


def test_hule_record_doras_cannot_replace_an_already_public_indicator(protocol):
    with pytest.raises(RecordError, match="monotone prefix"):
        decode_record_bytes(
            _legacy(protocol, _ron_results(protocol, record_doras=["8s"])),
            protocol,
            container="game_detail_records",
        )


@pytest.mark.parametrize("zimo", [False, True])
@pytest.mark.parametrize("include_win", [False, True])
def test_hule_hand_is_an_excluded_win_tile_cross_check_not_a_replacement(protocol, zimo, include_win):
    new_round = _new_round()
    if zimo:
        hand = list(new_round["tiles0"][:-1])
        tile = new_round["tiles0"][-1]
        if include_win:
            hand.append(tile)
        results = [
            _result(protocol, "RecordNewRound", **new_round),
            _result(
                protocol,
                "RecordHule",
                hules=[{"seat": 0, "hu_tile": tile, "zimo": True, "count": 1, "fu": 30, "hand": hand}],
                old_scores=[25000] * 4,
                delta_scores=[3000, -1000, -1000, -1000],
                scores=[28000, 24000, 24000, 24000],
            ),
        ]
    else:
        hand = list(new_round["tiles1"])
        if include_win:
            hand.append("1m")
        results = _ron_results(protocol, winner_fields={"hand": hand})
    if include_win:
        with pytest.raises(RecordError, match="replayed concealed hand"):
            decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    else:
        game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
        assert "hand" not in next(event for event in game.events if event["type"] == "hora")


def test_hule_hand_same_length_mismatch_is_not_used_to_repair_replay(protocol):
    hand = list(_new_round()["tiles1"])
    hand[0] = "7z"
    with pytest.raises(RecordError, match="replayed concealed hand"):
        decode_record_bytes(
            _legacy(protocol, _ron_results(protocol, winner_fields={"hand": hand})),
            protocol,
            container="game_detail_records",
        )


def _two_riichi_ron_results(protocol, markers):
    return [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="0p", is_liqi=True, moqie=True),
        _result(protocol, "RecordDealTile", seat=1, tile="9s", liqi={"seat": 0, "score": 24000, "liqibang": 1}),
        _result(protocol, "RecordDiscardTile", seat=1, tile="9s", is_liqi=True, moqie=True),
        _result(protocol, "RecordDealTile", seat=2, tile="8s", liqi={"seat": 1, "score": 24000, "liqibang": 2}),
        _result(protocol, "RecordDiscardTile", seat=2, tile="8s", moqie=True),
        _result(
            protocol,
            "RecordHule",
            hules=[
                {"seat": seat, "hu_tile": "8s", "zimo": False, "count": 1, "fu": 30, "li_doras": marker}
                for seat, marker in enumerate(markers)
            ],
            old_scores=[24000, 24000, 25000, 25000],
            delta_scores=[3000, 1000, -2000, 0],
            scores=[27000, 25000, 23000, 25000],
        ),
    ]


def test_multiple_accepted_riichi_winners_cannot_reveal_conflicting_ura(protocol):
    with pytest.raises(RecordError, match="conflicting ura"):
        decode_record_bytes(
            _legacy(protocol, _two_riichi_ron_results(protocol, [["1p"], ["2p"]])),
            protocol,
            container="game_detail_records",
        )


@pytest.mark.parametrize("markers", [[["1p"], ["1p"]], [[], ["1p"]], [[], []]])
def test_missing_or_equal_multiwinner_ura_does_not_become_a_zero_value_claim(protocol, markers):
    game = decode_record_bytes(
        _legacy(protocol, _two_riichi_ron_results(protocol, markers)), protocol, container="game_detail_records"
    )
    wins = [event for event in game.events if event["type"] == "hora"]
    assert [event["ura_markers"] for event in wins] == markers
    assert [event["type"] for event in game.events].count("reach_accepted") == 2


@pytest.mark.parametrize("zimo", [False, True])
def test_winners_public_hand_is_reconstructed_with_one_winning_instance(protocol, zimo):
    new_round = _new_round()
    if zimo:
        source_hand = list(new_round["tiles0"][:-1])
        tile = new_round["tiles0"][-1]
        results = [
            _result(protocol, "RecordNewRound", **new_round),
            _result(
                protocol,
                "RecordHule",
                hules=[{"seat": 0, "hu_tile": tile, "zimo": True, "count": 1, "fu": 30, "hand": source_hand}],
                old_scores=[25000] * 4,
                delta_scores=[3000, -1000, -1000, -1000],
                scores=[28000, 24000, 24000, 24000],
            ),
        ]
    else:
        source_hand = list(new_round["tiles1"])
        tile = "1m"
        results = _ron_results(protocol, winner_fields={"hand": source_hand})
    game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    win = next(event for event in game.events if event["type"] == "hora")
    expected = Counter(_canonical(value) for value in source_hand)
    expected[_canonical(tile)] += 1
    assert Counter(win["public_hand"]) == expected
    assert len(win["public_hand"]) == len(source_hand) + 1
    assert "hand" not in win and "ming" not in win and "fans" not in win
    assert all("public_hand" not in event for event in game.events if event["type"] != "hora")


def _canonical(value):
    if value.endswith("z"):
        return ("E", "S", "W", "N", "P", "F", "C")[int(value[0]) - 1]
    return "5" + value[1] + "r" if value.startswith("0") else value


def _notile_results(protocol, players=None, *, round_data=None, score_entries=None):
    terminal = {"gameend": True}
    if score_entries is not None:
        terminal["scores"] = score_entries
    if players is not None:
        terminal["players"] = players
    return [
        _result(protocol, "RecordNewRound", **(round_data or _new_round())),
        _result(protocol, "RecordDiscardTile", seat=0, tile="0p", moqie=True),
        _result(protocol, "RecordNoTile", **terminal),
    ]


def _notile_players(flags=(False, True, False, True)):
    data = _new_round()
    return [
        {
            "tingpai": flag,
            "hand": list(data[f"tiles{seat}"][:13]),
            "tings": [{"tile": "1m", "private_operation": f"must-not-copy-{seat}"}],
            "already_hule": False,
        }
        for seat, flag in enumerate(flags)
    ]


def test_notile_reveals_only_verified_tenpai_hands_not_private_false_hands(protocol):
    players = _notile_players()
    game = decode_record_bytes(
        _legacy(protocol, _notile_results(protocol, players)), protocol, container="game_detail_records"
    )
    event = next(event for event in game.events if event["type"] == "ryukyoku")
    assert event["visibility_complete"] is True
    assert event["public_hands"][0] is None and event["public_hands"][2] is None
    assert Counter(event["public_hands"][1]) == Counter(_canonical(tile) for tile in players[1]["hand"])
    assert Counter(event["public_hands"][3]) == Counter(_canonical(tile) for tile in players[3]["hand"])
    assert all(set(player) == {"tingpai", "hand"} for player in game.records[-1].data["players"])
    assert "must-not-copy" not in repr(game.records) and "must-not-copy" not in repr(event)
    assert game.metadata["visibility_gaps"] == []


@pytest.mark.parametrize("observed_labels", [False, True])
def test_notile_visibility_is_independent_of_score_inputs(protocol, observed_labels):
    score_entries = [{"old_scores": [25000] * 4, "delta_scores": [0] * 4}] if observed_labels else None
    private = decode_record_bytes(
        _legacy(protocol, _notile_results(protocol, _notile_players((False,) * 4), score_entries=score_entries)),
        protocol,
        container="game_detail_records",
    )
    shown = decode_record_bytes(
        _legacy(protocol, _notile_results(protocol, _notile_players((True,) * 4), score_entries=score_entries)),
        protocol,
        container="game_detail_records",
    )
    left = next(event for event in private.events if event["type"] == "ryukyoku")
    right = next(event for event in shown.events if event["type"] == "ryukyoku")
    if observed_labels:
        assert left["scores"] == right["scores"] and left["deltas"] == right["deltas"]
        assert "settlement_labels_missing" not in left and "settlement_labels_missing" not in right
    else:
        assert "scores" not in left and "scores" not in right
        assert "deltas" not in left and "deltas" not in right
        assert left["settlement_labels_missing"] is True and right["settlement_labels_missing"] is True
    assert left["public_hands"] == [None] * 4
    assert left["visibility_complete"] is True
    assert right["visibility_complete"] is True
    assert all(hand is not None for hand in right["public_hands"])


@pytest.mark.parametrize("kind", ["missing_field", "empty", "short", "missing_hand", "order", "missing_status"])
def test_unproven_notile_visibility_stays_unknown(protocol, kind):
    players = _notile_players()
    expected = {
        "missing_field": "notile_players_field_missing",
        "empty": "notile_four_player_entries_unavailable",
        "short": "notile_four_player_entries_unavailable",
        "missing_hand": "notile_public_hand_unavailable",
        "order": "notile_player_position_unverified",
        "missing_status": "notile_public_status_unavailable",
    }[kind]
    if kind == "missing_field":
        protocol = _protocol(include_players=False)
        results = _notile_results(protocol)
        game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
        event = next(event for event in game.events if event["type"] == "ryukyoku")
        gaps = game.metadata["visibility_gaps"]
    else:
        if kind == "empty":
            players = []
        elif kind == "short":
            players.pop()
        elif kind == "missing_hand":
            players[1]["hand"] = []
        elif kind == "order":
            players[0], players[1] = players[1], players[0]
        elif kind == "missing_status":
            players[1].pop("tingpai")
        records = [
            ServerRecord(0, "lq.RecordNewRound", "synthetic", _new_round()),
            ServerRecord(1, "lq.RecordDiscardTile", "synthetic", {"seat": 0, "tile": "0p", "moqie": True}),
            ServerRecord(2, "lq.RecordNoTile", "synthetic", {"scores": [], "players": players}),
        ]
        event = next(event for event in normalize_records(records) if event["type"] == "ryukyoku")
        gaps = [{"reason": event["visibility_gap"]}]
    assert event["visibility_complete"] is False
    assert event["visibility_gap"] == expected
    assert event["public_hands"] == [None] * 4
    assert gaps[0]["reason"] == expected


def test_notile_mismatched_hand_cannot_repair_a_replayed_seat(protocol):
    players = _notile_players()
    players[2]["hand"] = ["7z"] * 13
    with pytest.raises(RecordError, match="every replayed concealed hand"):
        decode_record_bytes(
            _legacy(protocol, _notile_results(protocol, players)), protocol, container="game_detail_records"
        )


def test_identical_private_hands_cannot_prove_a_public_players_seat(protocol):
    values = ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "1z", "2z", "3z", "4z"]
    data = _new_round()
    for seat in range(4):
        data[f"tiles{seat}"] = values.copy()
    data["tiles0"].append("0p")
    players = [{"tingpai": seat == 1, "hand": values.copy()} for seat in range(4)]
    game = decode_record_bytes(
        _legacy(protocol, _notile_results(protocol, players, round_data=data)),
        protocol,
        container="game_detail_records",
    )
    event = next(event for event in game.events if event["type"] == "ryukyoku")
    assert event["visibility_complete"] is False
    assert event["visibility_gap"] == "notile_public_hand_seat_binding_ambiguous"
    assert event["public_hands"] == [None] * 4


def test_public_terminal_facts_are_not_added_before_the_previous_cutoff(protocol):
    with_facts = decode_record_bytes(
        _legacy(protocol, _notile_results(protocol, _notile_players())), protocol, container="game_detail_records"
    )
    missing = decode_record_bytes(
        _legacy(protocol, _notile_results(protocol)), protocol, container="game_detail_records"
    )
    cutoff = next(index for index, event in enumerate(with_facts.events) if event["type"] == "dahai")
    assert (
        project_observer(list(with_facts.events), 0, cutoff, (0, 1, 2, 3)).to_bytes()
        == project_observer(list(missing.events), 0, cutoff, (0, 1, 2, 3)).to_bytes()
    )
    assert all(
        "public_hands" not in event and "visibility_complete" not in event for event in with_facts.events[: cutoff + 1]
    )


def test_notile_public_facts_ignore_poisoned_client_input(protocol):
    results = _notile_results(protocol, _notile_players())
    a = decode_record_bytes(_actions(results, b"\xff\xff"), protocol, container="game_detail_records")
    b = decode_record_bytes(_actions(results, b"private-choice"), protocol, container="game_detail_records")
    assert a.events == b.events and a.records == b.records
    assert a.metadata["visibility_gaps"] == b.metadata["visibility_gaps"] == []


@pytest.mark.parametrize(
    "entries", [[], [{"old_scores": [25000] * 4}], [{"old_scores": [25000] * 4, "delta_scores": []}]]
)
def test_absent_notile_delta_is_not_filled_as_an_observed_zero(protocol, entries):
    game = decode_record_bytes(
        _legacy(protocol, _notile_results(protocol, score_entries=entries)), protocol, container="game_detail_records"
    )
    event = next(event for event in game.events if event["type"] == "ryukyoku")
    assert "deltas" not in event and "scores" not in event
    assert event["settlement_labels_missing"] is True
    assert event["settlement_missing_fields"] == ["deltas", "scores"]
    assert game.metadata["source_score_chain_complete"] is False
    assert len(game.metadata["source_score_chain_gaps"]) == 1
    assert game.complete is True


def test_mixed_notile_components_do_not_turn_partial_deltas_into_an_aggregate(protocol):
    entries = [
        {"old_scores": [25000] * 4, "delta_scores": [1000, -1000, 0, 0]},
        {"old_scores": [25000] * 4, "delta_scores": []},
    ]
    game = decode_record_bytes(
        _legacy(protocol, _notile_results(protocol, score_entries=entries)), protocol, container="game_detail_records"
    )
    event = next(event for event in game.events if event["type"] == "ryukyoku")
    assert "deltas" not in event and "scores" not in event
    assert event["settlement_missing_component_indices"] == [1]


@pytest.mark.parametrize("component", [[0, 0], [0, 0, 0, 0]])
def test_observed_notile_components_still_require_valid_length_and_current_old_scores(protocol, component):
    entries = [{"old_scores": [26000, 24000, 25000, 25000], "delta_scores": component}]
    with pytest.raises(RecordError, match="current known scores"):
        decode_record_bytes(
            _legacy(protocol, _notile_results(protocol, score_entries=entries)),
            protocol,
            container="game_detail_records",
        )
    if len(component) != 4:
        entries[0]["old_scores"] = [25000] * 4
        with pytest.raises(RecordError, match="notile delta requires four scores"):
            decode_record_bytes(
                _legacy(protocol, _notile_results(protocol, score_entries=entries)),
                protocol,
                container="game_detail_records",
            )


def test_public_round_anchor_after_missing_labels_does_not_repair_source_continuity(protocol):
    anchor = [26000, 24000, 25000, 25000]
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordNoTile", scores=[{"old_scores": [25000] * 4}]),
        _result(protocol, "RecordNewRound", **_round_at(1, 1, scores=anchor)),
        _result(protocol, "RecordNoTile", scores=[{"old_scores": anchor, "delta_scores": [0] * 4}]),
    ]
    game = decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")
    assert game.metadata["source_score_chain_complete"] is False
    assert len(game.metadata["source_score_chain_gaps"]) == 1
    assert len(game.metadata["observed_source_anchors_after_gaps"]) == 1
    start = [event for event in game.events if event["type"] == "start_kyoku"][-1]
    assert start["source_score_anchor_after_gap"] is True
    assert start["scores"] == anchor
    terminal = [event for event in game.events if event["type"] == "ryukyoku"][-1]
    assert terminal["deltas"] == [0] * 4
    assert "settlement_labels_missing" not in terminal
    results[-1] = _result(protocol, "RecordNoTile", scores=[{"old_scores": [25000] * 4, "delta_scores": [0] * 4}])
    with pytest.raises(RecordError, match="current known scores"):
        decode_record_bytes(_legacy(protocol, results), protocol, container="game_detail_records")


def _four_riichi_results(
    protocol, *, explicit_fourth=False, missing_declaration=False, multiple_pending=False, stale_trigger=False
):
    results = [
        _result(protocol, "RecordNewRound", **_new_round()),
        _result(protocol, "RecordDiscardTile", seat=0, tile="0p", is_liqi=True, moqie=True),
    ]
    for seat, tile in ((1, "9s"), (2, "8s"), (3, "7s")):
        ack = {} if multiple_pending and seat == 2 else {"liqi": {"seat": seat - 1, "score": 24000, "liqibang": seat}}
        results.extend(
            [
                _result(protocol, "RecordDealTile", seat=seat, tile=tile, **ack),
                _result(
                    protocol,
                    "RecordDiscardTile",
                    seat=seat,
                    tile=tile,
                    is_liqi=not (missing_declaration and seat == 3),
                    moqie=True,
                ),
            ]
        )
    if stale_trigger:
        results.append(_result(protocol, "RecordDealTile", seat=0, tile="6s"))
    fourth = {"liqi": {"seat": 3, "score": 24000, "liqibang": 4}} if explicit_fourth else {}
    results.append(_result(protocol, "RecordLiuJu", type=4, **fourth))
    results.extend(
        [
            _result(protocol, "RecordNewRound", **_round_at(0, 1, 4, [24000] * 4)),
            _result(protocol, "RecordNoTile", scores=[{"old_scores": [24000] * 4, "delta_scores": [0] * 4}]),
        ]
    )
    return results


@pytest.mark.parametrize("explicit_fourth", [False, True])
def test_four_riichi_abort_confirms_only_the_unique_last_declaration(protocol, explicit_fourth):
    game = decode_record_bytes(
        _legacy(protocol, _four_riichi_results(protocol, explicit_fourth=explicit_fourth)),
        protocol,
        container="game_detail_records",
    )
    accepted = [event for event in game.events if event["type"] == "reach_accepted"]
    assert [event["actor"] for event in accepted] == [0, 1, 2, 3]
    assert len(game.metadata["derived_riichi_acceptances"]) == (0 if explicit_fourth else 1)
    if not explicit_fourth:
        assert accepted[-1]["acceptance_evidence"] == "four_riichi_abort"
        assert game.metadata["derived_riichi_acceptances"][0]["actor"] == 3
    next_round = [event for event in game.events if event["type"] == "start_kyoku"][-1]
    assert next_round["scores"] == [24000] * 4
    assert next_round["kyotaku"] == 4
    assert game.metadata["source_score_chain_complete"] is True


@pytest.mark.parametrize("bad", ["missing_declaration", "multiple_pending", "stale_trigger"])
def test_four_riichi_does_not_guess_an_unproven_acceptance_from_next_round_money(protocol, bad):
    with pytest.raises(RecordError, match="Four-riichi"):
        decode_record_bytes(
            _legacy(protocol, _four_riichi_results(protocol, **{bad: True})), protocol, container="game_detail_records"
        )


@pytest.mark.parametrize(
    "game_id,kind,actor",
    [
        ("200401-075abcea-14a0-4904-b608-901612af5bb8", "missing_delta", None),
        ("200401-1abb541c-edec-4552-be36-d03d31dcb754", "four_riichi", 3),
        ("200403-34b5b9c2-5ce6-478c-9ebd-afd22d9832f3", "four_riichi", 1),
    ],
)
def test_real_legacy_missing_labels_and_four_riichi_formats_are_preserved(game_id, kind, actor):
    root = Path(__file__).parents[3] / "kar-m0-assets"
    raw = root / "jade-catalog-1000" / (game_id + ".bin")
    if not raw.is_file():
        pytest.skip("Private real regression data is not distributed with source")
    game = decode_record_bytes(
        raw.read_bytes(), load_protocol(root / "mahjongsoul.proto"), container="majsoul_download"
    )
    if kind == "missing_delta":
        missing = [event for event in game.events if event.get("settlement_labels_missing")]
        assert missing and all("scores" not in event and "deltas" not in event for event in missing)
        assert game.metadata["source_score_chain_complete"] is False
        assert game.metadata["observed_source_anchors_after_gaps"]
    else:
        derived = game.metadata["derived_riichi_acceptances"]
        assert len(derived) == 1 and derived[0]["actor"] == actor
        assert game.events[derived[0]["event_id"]]["acceptance_evidence"] == "four_riichi_abort"
