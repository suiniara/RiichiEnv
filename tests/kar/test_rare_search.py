import base64
import hashlib
import io
import json
import sqlite3
from types import SimpleNamespace

import pytest
from google.protobuf import descriptor_pb2

from riichienv.kar import corpus_fetch as transport
from riichienv.kar import rare_search as rare
from riichienv.kar.mjsoul import Protocol


def protocol():
    file = descriptor_pb2.FileDescriptorProto(name="synthetic-rare.proto", package="lq", syntax="proto3")
    definitions = {
        "Wrapper": [("name", 5, 9, False, None), ("data", 9, 12, False, None)],
        "ResGameRecord": [("data", 7, 12, False, None)],
        "GameDetailRecords": [
            ("records", 3, 12, True, None),
            ("version", 6, 13, False, None),
            ("actions", 9, 11, True, ".lq.GameAction"),
        ],
        "GameAction": [("result", 8, 12, False, None), ("user_input", 11, 11, False, ".lq.ClientOpaque")],
        "ClientOpaque": [("unused", 1, 13, False, None)],
        "RecordAnGangAddGang": [
            ("seat", 7, 13, False, None),
            ("type", 8, 13, False, None),
            ("tiles", 9, 9, False, None),
        ],
        "RecordHule": [("hules", 5, 11, True, ".lq.DifferentWinnerName")],
        "DifferentWinnerName": [
            ("seat", 4, 13, False, None),
            ("hu_tile", 11, 9, False, None),
            ("zimo", 15, 8, False, None),
        ],
        "RecordChiPengGang": [("seat", 6, 13, False, None), ("type", 7, 13, False, None)],
        "RecordLiuJu": [("type", 9, 13, False, None)],
        "RecordNewRound": [],
        "RecordDealTile": [],
        "RecordDiscardTile": [],
        "RecordNoTile": [],
        "RecordAnalysisedData": [],
    }
    for name, fields in definitions.items():
        message = file.message_type.add(name=name)
        for field_name, number, kind, repeated, message_name in fields:
            field = message.field.add(name=field_name, number=number, type=kind, label=3 if repeated else 1)
            if message_name:
                field.type_name = message_name
    data = descriptor_pb2.FileDescriptorSet()
    data.file.add().CopyFrom(file)
    return Protocol.from_descriptor_set(data.SerializeToString())


def varint(value):
    output = bytearray()
    while value > 127:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def field(number, value):
    if isinstance(value, str):
        value = value.encode()
    if isinstance(value, bytes):
        return varint(number * 8 + 2) + varint(len(value)) + value
    return varint(number * 8) + varint(int(value))


def body(mapping, message, **values):
    return b"".join(
        field(mapping["messages"][message]["fields"][name]["number"], value) for name, value in values.items()
    )


def wrapped(mapping, name, payload):
    return body(mapping, "wrapper", name="." + name if name else "", data=payload)


def game(mapping, records, *, version=0, poisoned_input=False):
    result = []
    for name, payload in records:
        if name is None:
            result.append(None)
        else:
            result.append(wrapped(mapping, name, payload or field(99, 1)))
    detail = body(mapping, "detail", version=version)
    if version == 0:
        detail += b"".join(
            field(mapping["messages"]["detail"]["fields"]["records"]["number"], item) for item in result if item
        )
    else:
        for item in result:
            action = body(mapping, "action", result=item) if item else b""
            if poisoned_input:
                action += field(11, b"\xff\xff\x80malformed client data")
            detail += field(mapping["messages"]["detail"]["fields"]["actions"]["number"], action)
    response = body(mapping, "response", data=wrapped(mapping, mapping["messages"]["detail"]["name"], detail))
    return b"\0\0\0" + wrapped(mapping, "", response)


def namespace():
    scope = {"__name__": "trusted_rare_script_test"}
    exec(compile(rare.REMOTE_SCRIPT, "trusted_rare_script", "exec"), scope)
    return scope


def records(mapping, *, tile="1m", zimo=False, winner=1):
    names = mapping["records"]
    return [
        (names["round"], b""),
        (names["kan"], body(mapping, "kan", seat=0, type=3, tiles=tile)),
        (names["win"], body(mapping, "hule", hules=body(mapping, "winner", seat=winner, hu_tile=tile, zimo=zimo))),
    ]


def test_mapping_comes_from_descriptor_not_remembered_huleinfo_numbers():
    mapping = rare.freeze_field_map(protocol())
    assert mapping["messages"]["winner"]["name"] == "lq.DifferentWinnerName"
    assert mapping["messages"]["winner"]["fields"]["hu_tile"]["number"] == 11
    assert set(mapping["messages"]["action"]["fields"]) == {"result"}
    assert "user_input" not in json.dumps(mapping)
    namespace()["validate_map"](mapping)


@pytest.mark.parametrize("tile", ["1m", "5pr", "N"])
def test_ankan_ron_is_only_a_candidate_even_without_a_kokushi_hand(tile):
    mapping = rare.freeze_field_map(protocol())
    result = namespace()["detect"](game(mapping, records(mapping, tile=tile)), mapping)
    assert result["candidate_tags"] == ["possible_kokushi_ankan_ron"]
    assert result["candidate_is_confirmed_positive"] is False
    assert result["semantic_verification_performed"] is False


def test_red_and_normal_five_have_the_same_basic_tile_for_candidate_matching():
    mapping = rare.freeze_field_map(protocol())
    items = records(mapping, tile="5mr")
    items[-1] = (
        mapping["records"]["win"],
        body(mapping, "hule", hules=body(mapping, "winner", seat=1, hu_tile="5m", zimo=False)),
    )
    assert namespace()["detect"](game(mapping, items), mapping)["candidate_tags"]


@pytest.mark.parametrize("intervening", ["draw", "discard", "call", "notile", "round"])
def test_turn_or_round_advance_cancels_the_robbed_ankan_window(intervening):
    mapping = rare.freeze_field_map(protocol())
    items = records(mapping)
    payload = body(mapping, "call", seat=2, type=1) if intervening == "call" else field(99, 1)
    items.insert(2, (mapping["records"][intervening], payload))
    assert namespace()["detect"](game(mapping, items), mapping)["candidate_tags"] == []


def test_neutral_analysis_record_does_not_change_the_declaration_window():
    mapping = rare.freeze_field_map(protocol())
    items = records(mapping)
    items.insert(2, ("lq.RecordAnalysisedData", field(99, 1)))
    assert namespace()["detect"](game(mapping, items), mapping)["candidate_tags"]


@pytest.mark.parametrize("zimo,winner", [(True, 1), (False, 0)])
def test_self_draw_and_the_declarer_are_not_ankan_ron_candidates(zimo, winner):
    mapping = rare.freeze_field_map(protocol())
    result = namespace()["detect"](game(mapping, records(mapping, zimo=zimo, winner=winner)), mapping)
    assert result["candidate_tags"] == []


def test_poisoned_user_input_never_enters_the_wire_result_parser():
    mapping = rare.freeze_field_map(protocol())
    items = records(mapping)
    items.insert(1, (None, None))
    result = namespace()["detect"](game(mapping, items, version=210715, poisoned_input=True), mapping)
    assert result["candidate_tags"] == ["possible_kokushi_ankan_ron"]
    assert result["candidate_evidence"][0]["declaration_record_index"] == 2
    assert result["candidate_evidence"][0]["record_index"] == 3


@pytest.mark.parametrize("liuju", [2, 3])
def test_four_kan_candidate_counts_declarations_and_daiminkan(liuju):
    mapping = rare.freeze_field_map(protocol())
    names = mapping["records"]
    items = [(names["round"], b"")]
    items += [
        (names["kan"], body(mapping, "kan", seat=actor, type=kind, tiles="1m"))
        for actor, kind in ((0, 3), (1, 3), (2, 2))
    ]
    items.append((names["call"], body(mapping, "call", seat=2, type=2)))
    items.append((names["abort"], body(mapping, "abort", type=liuju)))
    result = namespace()["detect"](game(mapping, items), mapping)
    assert result["candidate_tags"] == ["possible_four_kans"]
    assert result["candidate_evidence"][0]["kan_declarations"] == 4


def test_kan_count_does_not_leak_between_rounds():
    mapping = rare.freeze_field_map(protocol())
    names = mapping["records"]
    items = [(names["round"], b"")]
    items += [(names["kan"], body(mapping, "kan", seat=actor, type=3, tiles="1m")) for actor in (0, 1, 2)]
    items += [
        (names["round"], b""),
        (names["kan"], body(mapping, "kan", seat=3, type=3, tiles="9m")),
        (names["abort"], body(mapping, "abort", type=2)),
    ]
    assert namespace()["detect"](game(mapping, items), mapping)["candidate_tags"] == []


def test_four_kans_by_one_actor_is_not_a_multi_actor_abort_candidate():
    mapping = rare.freeze_field_map(protocol())
    names = mapping["records"]
    items = [(names["round"], b"")] + [(names["kan"], body(mapping, "kan", seat=0, type=3, tiles="1m"))] * 4
    items.append((names["abort"], body(mapping, "abort", type=3)))
    assert namespace()["detect"](game(mapping, items), mapping)["candidate_tags"] == []


def test_four_riichi_is_also_only_a_candidate():
    mapping = rare.freeze_field_map(protocol())
    items = [(mapping["records"]["round"], b""), (mapping["records"]["abort"], body(mapping, "abort", type=4))]
    assert namespace()["detect"](game(mapping, items), mapping)["candidate_tags"] == ["possible_four_riichi"]


@pytest.mark.parametrize("case", ["unknown_wrapper", "unknown_version", "truncated", "bad_mapping"])
def test_bad_protocol_or_unknown_record_is_an_error_not_a_no_hit(case):
    mapping = rare.freeze_field_map(protocol())
    items = records(mapping)
    raw = game(mapping, items)
    if case == "unknown_wrapper":
        raw = game(mapping, [("lq.UnknownRecord", field(99, 1))])
    elif case == "unknown_version":
        raw = game(mapping, items, version=42)
    elif case == "truncated":
        raw = raw[:-1]
    else:
        mapping["messages"]["winner"]["fields"]["hu_tile"]["type"] = 13
    with pytest.raises(ValueError):
        namespace()["detect"](raw, mapping)


class Input(io.BytesIO):
    def close(self):
        if not self.closed:
            self.written = self.getvalue()
        super().close()


class Process:
    def __init__(self, output, code=0):
        self.stdin, self.stdout = Input(), io.BytesIO(output)
        self.returncode = None
        self.code = code

    def wait(self, timeout=None):
        self.returncode = self.code if self.returncode is None else self.returncode
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class SSH:
    def __init__(self, builders):
        self.builders = list(builders)
        self.calls = []
        self.processes = []

    def __call__(self, command, **kwargs):
        wire = json.loads(base64.urlsafe_b64decode(command[-1].rsplit(" ", 1)[1]))
        self.calls.append((command, wire))
        output = self.builders.pop(0)(wire)
        process = output if isinstance(output, Process) else Process(output)
        self.processes.append(process)
        return process


def config(tmp_path, **kwargs):
    values = dict(
        host="example-user@192.0.2.86",
        ssh_executable="ssh.exe",
        remote_python=r"C:\Users\Example\kanachan-data-work\runtime\venv\Scripts\python.exe",
        catalog=r"C:\Users\Example\kanachan-data-work\catalog\jade-hanchan-v2\catalog.sqlite3",
        proto=tmp_path / "synthetic.proto",
        max_scanned=100,
    )
    values.update(kwargs)
    return rare.SearchConfig(**values)


def envelope(wire, entries=(), *, summary=None):
    header = {
        "kind": "header",
        "protocol": rare.VERSION,
        "scope_sha256": wire["scope_sha256"],
        "field_map_sha256": wire["field_map_sha256"],
        "after_uuid": wire["after_uuid"],
        "max_scanned": wire["max_scanned"],
        "mode_id": wire["mode_id"],
        "old_source_split": wire["source_split"],
        "locator_read_mode": "immutable_main_ignores_wal",
    }
    rows = [header, *entries]
    if summary is not None:
        rows.append(
            {
                "kind": "summary",
                **summary,
                "selection_complete": True,
                "all_disk_scanned": False,
                "semantic_verification_performed": False,
            }
        )
    return b"".join(transport._canonical(row) + b"\n" for row in rows)


def position(scanned, cursor, hits=0, errors=0):
    return {"scanned": scanned, "hits": hits, "error_count": errors, "last_scanned_uuid": cursor}


def candidate(wire, uuid, *, scanned=1, hits=1, errors=0):
    raw = b"opaque raw candidate " + uuid.encode()
    sha = hashlib.sha256(raw).hexdigest()
    path = rf"D:\majsoul\record-data-iter\{uuid}"
    return {
        "kind": "candidate",
        "uuid": uuid,
        "source_path": path,
        "old_source_split": wire["source_split"],
        "mode_id": wire["mode_id"],
        "locator_sha256": sha,
        "source_sha256": sha,
        "bytes": len(raw),
        "locator_metadata": {"uuid": uuid, "path": path, "mode_id": wire["mode_id"], "sha256": sha, "bytes": len(raw)},
        "raw_base64": base64.b64encode(raw).decode(),
        "candidate_tags": ["possible_four_kans"],
        "candidate_evidence": [],
        "candidate_is_confirmed_positive": False,
        "semantic_verification_performed": False,
        "field_map_sha256": wire["field_map_sha256"],
        "search_protocol_sha256": wire["field_map"]["source_proto_sha256"],
        **position(scanned, uuid, hits, errors),
    }


def patch_protocol(monkeypatch):
    proto = protocol()
    monkeypatch.setattr(rare, "load_protocol", lambda _: proto)
    return proto


def read_checkpoint(root):
    return json.loads((root / "checkpoint.json").read_text())


def test_single_session_copies_hits_only_and_keeps_all_errors_auditable(tmp_path, monkeypatch):
    patch_protocol(monkeypatch)

    def output(wire):
        rows = [
            candidate(wire, "game-0020", scanned=20),
            {
                "kind": "input_error",
                "uuid": "game-0030",
                "reason": "unknown_wrapper",
                **position(30, "game-0030", 1, 1),
            },
            {"kind": "progress", **position(50, "game-0050", 1, 1)},
        ]
        return envelope(wire, rows, summary={**position(100, "game-0100", 1, 1), "catalog_exhausted": False})

    ssh = SSH([output])
    monkeypatch.setattr(rare.subprocess, "Popen", ssh)
    root = tmp_path / "search"
    state = rare.search(config(tmp_path), root)
    assert state["selection_complete"] is True and state["scan_complete"] is False
    assert state["scanned"] == 100 and state["hits"] == 1 and state["error_count"] == 1
    assert state["no_candidate_inputs"] == 98
    assert state["no_candidate_is_semantic_negative_proof"] is False
    assert len(list(root.glob("*.bin"))) == 1
    assert len((root / "errors.jsonl").read_text().splitlines()) == 1
    manifest = json.loads((root / "manifest.jsonl").read_text())
    assert manifest["selection_role"] == "golden_only" and manifest["training_eligible"] is False
    assert manifest["candidate_is_confirmed_positive"] is False
    assert ssh.processes[0].stdin.written == rare.REMOTE_SCRIPT.encode()
    assert len(ssh.calls) == 1


def test_interrupted_stream_resumes_scanned_cursor_not_last_hit(tmp_path, monkeypatch):
    patch_protocol(monkeypatch)
    ssh = SSH(
        [
            lambda wire: envelope(
                wire, [{"kind": "progress", **position(50, "game-0050")}, candidate(wire, "game-0051", scanned=51)]
            ),
            lambda wire: envelope(wire, summary={**position(49, "game-0100"), "catalog_exhausted": False}),
        ]
    )
    monkeypatch.setattr(rare.subprocess, "Popen", ssh)
    root = tmp_path / "resume"
    first = rare.search(config(tmp_path), root)
    assert first["scanned"] == 51 and not first["selection_complete"]
    stamp = (root / "game-0051.bin").stat().st_mtime_ns
    second = rare.search(config(tmp_path), root, resume=True)
    assert second["scan_complete"] and second["scanned"] == 100 and second["hits"] == 1
    assert ssh.calls[1][1]["after_uuid"] == "game-0051"
    assert ssh.calls[1][1]["max_scanned"] == 49
    assert (root / "game-0051.bin").stat().st_mtime_ns == stamp


def test_missing_footer_at_bound_requires_successful_zero_input_resume(tmp_path, monkeypatch):
    patch_protocol(monkeypatch)
    ssh = SSH(
        [
            lambda wire: envelope(wire, [{"kind": "progress", **position(100, "game-0100")}]),
            lambda wire: envelope(wire, summary={**position(0, "game-0100"), "catalog_exhausted": False}),
        ]
    )
    monkeypatch.setattr(rare.subprocess, "Popen", ssh)
    root = tmp_path / "footer"
    assert not rare.search(config(tmp_path), root)["scan_complete"]
    assert rare.search(config(tmp_path), root, resume=True)["scan_complete"]
    assert ssh.calls[1][1]["max_scanned"] == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.update(source_path=r"C:\Users\Example\.ssh\secret"),
        lambda row: row.update(uuid="../escape"),
        lambda row: row.update(source_sha256="0" * 64),
        lambda row: row.update(candidate_is_confirmed_positive=True),
    ],
)
def test_unsafe_candidates_never_advance_scan_cursor(tmp_path, monkeypatch, change):
    patch_protocol(monkeypatch)

    def output(wire):
        row = candidate(wire, "game-0001")
        change(row)
        return envelope(wire, [row])

    monkeypatch.setattr(rare.subprocess, "Popen", SSH([output]))
    root = tmp_path / "unsafe"
    state = rare.search(config(tmp_path), root)
    assert not state["scan_complete"] and state["scanned"] == 0
    assert not list(root.glob("*.bin"))


def test_resume_checks_candidate_sha_and_scope_before_any_remote_work(tmp_path, monkeypatch):
    patch_protocol(monkeypatch)
    ssh = SSH(
        [
            lambda wire: envelope(
                wire,
                [candidate(wire, "game-0001")],
                summary={**position(100, "game-0100", 1), "catalog_exhausted": False},
            )
        ]
    )
    monkeypatch.setattr(rare.subprocess, "Popen", ssh)
    root = tmp_path / "integrity"
    cfg = config(tmp_path)
    rare.search(cfg, root)
    with pytest.raises(rare.RareSearchError, match="scope"):
        rare.search(config(tmp_path, mode_id=16), root, resume=True)
    (root / "game-0001.bin").write_bytes(b"tampered")
    with pytest.raises(rare.RareSearchError, match="SHA"):
        rare.search(cfg, root, resume=True)
    assert read_checkpoint(root)["scan_complete"] is False
    assert len(ssh.calls) == 1


def test_progress_cannot_hide_a_missing_hit_or_exceed_the_scan_bound(tmp_path, monkeypatch):
    patch_protocol(monkeypatch)
    for number, stats in enumerate((position(50, "game-0050", 1), position(101, "game-0101"))):
        ssh = SSH([lambda wire, stats=stats: envelope(wire, [{"kind": "progress", **stats}])])
        monkeypatch.setattr(rare.subprocess, "Popen", ssh)
        state = rare.search(config(tmp_path), tmp_path / str(number))
        assert not state["scan_complete"] and state["scanned"] == 0


def test_bound_and_connection_parameters_are_validated_before_output_creation(tmp_path, monkeypatch):
    patch_protocol(monkeypatch)
    for number, values in enumerate(
        ({"max_scanned": 10001}, {"max_scanned": 0}, {"host": "user@host & calc"}, {"after_uuid": "x' OR 1=1--"})
    ):
        root = tmp_path / str(number)
        with pytest.raises(ValueError):
            rare.search(config(tmp_path, **values), root)
        assert not root.exists()


def test_remote_raw_paths_cannot_escape_authorized_roots():
    ns = namespace()
    for path in (
        r"C:\Users\Example\.ssh\safe",
        r"D:\majsoul\record-data-iter-evil\safe",
        r"D:\majsoul\record-data\..\..\safe",
        r"D:\majsoul\record-data\safe:secret",
    ):
        with pytest.raises(ValueError):
            ns["safe_path"](path, "safe")
    ns["Path"] = lambda _: SimpleNamespace(resolve=lambda **kwargs: "C:\\Users\\Example\\.ssh\\safe")
    raw = b"opaque"
    sha = hashlib.sha256(raw).hexdigest()
    md = {"uuid": "safe", "mode_id": 12, "path": r"D:\majsoul\record-data-iter\safe", "sha256": sha, "bytes": len(raw)}
    with pytest.raises(ValueError, match="outside_allowed_roots"):
        ns["read_raw"](("safe", md["path"], "train", 12, json.dumps(md)), {"mode_id": 12, "source_split": "train"})


def test_existing_candidate_collision_is_not_overwritten(tmp_path, monkeypatch):
    patch_protocol(monkeypatch)
    root = tmp_path / "collision"

    def output(wire):
        (root / "game-0001.bin").write_bytes(b"unknown user file")
        return envelope(wire, [candidate(wire, "game-0001")])

    monkeypatch.setattr(rare.subprocess, "Popen", SSH([output]))
    report = rare.search(config(tmp_path), root)
    assert not report["scan_complete"] and report["scanned"] == 0
    assert (root / "game-0001.bin").read_bytes() == b"unknown user file"


def test_journal_ahead_of_checkpoint_is_recovered_without_recopied_hits(tmp_path, monkeypatch):
    proto = patch_protocol(monkeypatch)
    ssh = SSH(
        [
            lambda wire: envelope(wire, [candidate(wire, "game-0050", scanned=50)]),
            lambda wire: envelope(wire, summary={**position(50, "game-0100"), "catalog_exhausted": False}),
        ]
    )
    monkeypatch.setattr(rare.subprocess, "Popen", ssh)
    root = tmp_path / "ahead"
    cfg = config(tmp_path)
    rare.search(cfg, root)
    transport._atomic_checkpoint(root, rare._initial(cfg.scope(rare.freeze_field_map(proto))))
    stamp = (root / "game-0050.bin").stat().st_mtime_ns
    result = rare.search(cfg, root, resume=True)
    assert result["scan_complete"] and result["scanned"] == 100 and result["hits"] == 1
    assert ssh.calls[1][1]["after_uuid"] == "game-0050"
    assert (root / "game-0050.bin").stat().st_mtime_ns == stamp


def test_keyboard_interrupt_does_not_complete_the_search(tmp_path, monkeypatch):
    patch_protocol(monkeypatch)

    class Interrupted(io.BytesIO):
        def readline(self, size=-1):
            raise KeyboardInterrupt

    process = Process(b"")
    process.stdout = Interrupted()
    monkeypatch.setattr(rare.subprocess, "Popen", lambda *args, **kwargs: process)
    root = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        rare.search(config(tmp_path), root)
    assert read_checkpoint(root)["status"] == "INTERRUPTED"
    assert read_checkpoint(root)["scan_complete"] is False


def test_remote_scan_limit_counts_all_inputs_not_just_hits_and_reports_decode_errors(tmp_path):
    mapping = rare.freeze_field_map(protocol())
    db = tmp_path / "catalog.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE games(uuid TEXT PRIMARY KEY,path TEXT,split TEXT,"
        "mode_id INTEGER,conflicted INTEGER,metadata TEXT)"
    )
    conn.execute("CREATE INDEX game_split ON games(conflicted,split,uuid)")
    for number in range(200):
        conn.execute("INSERT INTO games VALUES(?,?,?,?,?,?)", (f"game-{number:04}", "unused", "train", 12, 0, "{}"))
    conn.execute("INSERT INTO games VALUES(?,?,?,?,?,?)", ("game-0010-golden", "unused", "train", 16, 0, "{}"))
    conn.commit()
    conn.close()
    ns = namespace()
    emitted, calls = [], []
    ns["emit"] = emitted.append

    def connect(uri_path, **kwargs):
        assert uri_path.endswith("?mode=ro&immutable=1") and kwargs["uri"] is True
        calls.append(uri_path)
        return sqlite3.connect(uri_path, uri=True)

    ns["sqlite3"] = SimpleNamespace(connect=connect)
    ns["read_raw"] = lambda row, cfg: (row[0].encode(), {"uuid": row[0]})

    def detect(raw, _):
        if raw == b"game-0030":
            raise ValueError("unknown_record")
        return {
            "candidate_tags": ["possible_four_kans"] if raw == b"game-0020" else [],
            "candidate_evidence": [],
            "candidate_is_confirmed_positive": False,
            "semantic_verification_performed": False,
        }

    ns["detect"] = detect
    result = ns["run"](
        {
            "catalog": str(db),
            "mode_id": 12,
            "source_split": "train",
            "after_uuid": "game-0009",
            "max_scanned": 100,
            "scope_sha256": "fixture",
            "field_map_sha256": transport._digest(mapping),
            "field_map": mapping,
        }
    )
    assert result == 0 and len(calls) == 1
    assert emitted[-1]["scanned"] == 100 and emitted[-1]["hits"] == 1 and emitted[-1]["error_count"] == 1
    assert emitted[-1]["last_scanned_uuid"] == "game-0109"
    assert len([row for row in emitted if row["kind"] == "progress"]) == 2
    assert len([row for row in emitted if "raw_base64" in row]) == 1
    assert (
        next(row for row in emitted if row["kind"] == "input_error")["classification"]
        == "unknown_not_a_verified_negative"
    )
    assert "OFFSET" not in ns["SQL"] and ns["SQL"].count("?") == 4
