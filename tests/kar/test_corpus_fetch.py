import base64
import hashlib
import io
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from riichienv.kar import corpus_fetch as fetch


def config(**kwargs):
    base = fetch.FetchConfig(
        host="example-user@192.0.2.86",
        ssh_executable="ssh.exe",
        remote_python=r"C:\Users\Example\kanachan-data-work\runtime\venv\Scripts\python.exe",
        catalog=r"C:\Users\Example\kanachan-data-work\catalog\jade-hanchan-v2\catalog.sqlite3",
        limit=3,
    )
    return replace(base, **kwargs)


def record(uuid, *, mode=12, source_split="train", data=None):
    raw = data if data is not None else b"opaque-" + uuid.encode()
    sha = hashlib.sha256(raw).hexdigest()
    path = rf"D:\majsoul\record-data-iter\{uuid}"
    return {
        "kind": "record",
        "uuid": uuid,
        "source_path": path,
        "old_source_split": source_split,
        "mode_id": mode,
        "locator_sha256": sha,
        "source_sha256": sha,
        "bytes": len(raw),
        "locator_metadata": {
            "uuid": uuid,
            "path": path,
            "sha256": sha,
            "bytes": len(raw),
            "mode_id": mode,
            "record_version": 0,
            "split": source_split,
            "schema_sha256": "old_generated_module_hash_not_protocol",
        },
        "raw_base64": base64.b64encode(raw).decode(),
    }


def jsonl(rows):
    return b"".join(fetch._canonical(row) + b"\n" for row in rows)


def protocol(wire, records=(), *, footer=True, exhausted=False):
    header = {
        "kind": "header",
        "protocol": fetch.VERSION,
        "scope_sha256": wire["scope_sha256"],
        "after_uuid": wire["after_uuid"],
        "limit": wire["limit"],
        "mode_id": wire["mode_id"],
        "old_source_split": wire["source_split"],
        "locator_read_mode": "immutable_main_ignores_wal",
    }
    rows = [header, *records]
    if footer:
        rows.append(
            {
                "kind": "summary",
                "selection_complete": True,
                "records": len(records),
                "last_uuid": records[-1]["uuid"] if records else wire["after_uuid"],
                "catalog_exhausted": exhausted,
                "all_disk_scanned": False,
            }
        )
    return jsonl(rows)


class RecordingInput(io.BytesIO):
    def close(self):
        if not self.closed:
            self.written = self.getvalue()
        super().close()


class Process:
    def __init__(self, output, *, code=0):
        self.stdin = RecordingInput()
        self.stdout = io.BytesIO(output)
        self.code = code
        self.returncode = None
        self.terminated = False

    def wait(self, timeout=None):
        self.returncode = self.code if self.returncode is None else self.returncode
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
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
        self.calls.append((command, kwargs, wire))
        result = self.builders.pop(0)(wire)
        process = result if isinstance(result, Process) else Process(result)
        self.processes.append(process)
        return process


def checkpoint(root):
    return json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))


def manifest(root):
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


def test_single_session_stdin_transport_keeps_old_metadata_non_authoritative(tmp_path, monkeypatch):
    ssh = SSH([lambda wire: protocol(wire, [record("a"), record("b"), record("c")])])
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "new-output"
    result = fetch.fetch_corpus(config(), root)
    assert result["selection_complete"] is True
    assert result["scan_complete"] is True
    assert result["all_disk_scanned"] is False
    assert result["processed"] == 3 and result["last_uuid"] == "c"
    assert len(ssh.calls) == 1
    assert not list(root.glob(".checkpoint-*.tmp"))
    command = ssh.calls[0][0]
    assert '"' + config().remote_python + '" -I -B -S - ' in command[-1]
    assert config().catalog not in command[-1]
    assert ssh.processes[0].stdin.written == fetch.REMOTE_SCRIPT.encode()
    for row in manifest(root):
        assert row["selection_role"] == "main_cohort_candidate"
        assert row["kar_split"] is None and row["protocol_sha256"] is None
        assert row["training_eligible"] is False
        assert row["old_schema_hash_is_protocol_hash"] is False
        assert row["source_sha256"] == hashlib.sha256((root / row["local_file"]).read_bytes()).hexdigest()


def test_golden_room_is_never_marked_as_main_training_data(tmp_path, monkeypatch):
    ssh = SSH([lambda wire: protocol(wire, [record("a", mode=16)], exhausted=True)])
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "golden"
    result = fetch.fetch_corpus(config(mode_id=16), root)
    assert result["processed"] == 1
    assert manifest(root)[0]["selection_role"] == "golden_only"
    assert manifest(root)[0]["training_eligible"] is False


@pytest.mark.parametrize(
    "values",
    [
        {"host": "user@host & calc"},
        {"host": "-oProxyCommand=calc"},
        {"host": "user@host\ncalc"},
        {"remote_python": r"C:\Python\python.exe & calc.exe"},
        {"remote_python": r"C:\Python\python.exe%COMSPEC%.exe"},
        {"remote_python": r"C:\Windows\System32\cmd.exe"},
        {"catalog": "file:C:/db?mode=rw"},
        {"source_split": "train' OR 1=1 --"},
        {"after_uuid": "x' OR 1=1 --"},
        {"limit": 1001},
        {"limit": 0},
        {"limit": True},
        {"mode_id": 9},
    ],
)
def test_unsafe_parameters_are_rejected_before_creating_output(tmp_path, values):
    root = tmp_path / "no-output"
    with pytest.raises(fetch.CorpusFetchError):
        fetch.fetch_corpus(config(**values), root)
    assert not root.exists()


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\Example\.ssh\safe",
        r"\\server\share\safe",
        r"\\?\D:\majsoul\record-data\safe",
        r"D:\majsoul\record-data-iter-evil\safe",
        r"D:\majsoul\record-data\..\..\safe",
        r"D:\majsoul\record-data\safe:secret",
        r"D:\majsoul\record-data\account.yml",
    ],
)
def test_client_raw_path_scope_rejects_escapes(path):
    with pytest.raises(fetch.CorpusFetchError):
        fetch.validate_raw_path(path, "safe")


def test_normalized_in_scope_raw_paths_are_accepted():
    assert fetch.validate_raw_path(r"D:\majsoul\record-data\2020\safe", "safe").endswith(r"2020\safe")
    assert fetch.validate_raw_path("d:/majsoul/record-data-iter/safe", "safe").endswith("safe")


@pytest.mark.parametrize("uuid", ["../x", "x/y", r"x\y", ".", "..", "NUL", "COM1", "LPT9", "x.bin"])
def test_unsafe_remote_uuid_never_becomes_a_local_filename(tmp_path, monkeypatch, uuid):
    ssh = SSH([lambda wire: protocol(wire, [record(uuid)], exhausted=True)])
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "bad-uuid"
    result = fetch.fetch_corpus(config(), root)
    assert result["selection_complete"] is False
    assert not list(root.glob("*.bin"))


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.update(raw_base64="not base64!"),
        lambda row: row.update(source_sha256="0" * 64),
        lambda row: row.update(locator_sha256="0" * 64),
        lambda row: row.update(bytes=999),
        lambda row: row.update(mode_id=16),
        lambda row: row.update(old_source_split="test"),
        lambda row: row.update(source_path=r"C:\Users\Example\account.yml"),
    ],
)
def test_unverified_stream_rows_fail_without_advancing_cursor(tmp_path, monkeypatch, change):
    row = record("a")
    change(row)
    ssh = SSH([lambda wire: protocol(wire, [row], exhausted=True)])
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "bad-row"
    result = fetch.fetch_corpus(config(), root)
    assert result["processed"] == 0 and result["last_uuid"] == ""
    assert result["scan_complete"] is False
    assert result["errors"]
    assert not (root / "a.bin").exists()


def test_partial_stream_resume_rechecks_and_reuses_committed_raw(tmp_path, monkeypatch):
    ssh = SSH(
        [
            lambda wire: protocol(wire, [record("a")], footer=False),
            lambda wire: protocol(wire, [record("b"), record("c")]),
        ]
    )
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "resume"
    first = fetch.fetch_corpus(config(), root)
    assert not first["scan_complete"] and first["processed"] == 1
    stamp = (root / "a.bin").stat().st_mtime_ns
    second = fetch.fetch_corpus(config(), root, resume=True)
    assert second["selection_complete"] and second["processed"] == 3
    assert ssh.calls[1][2]["after_uuid"] == "a" and ssh.calls[1][2]["limit"] == 2
    assert (root / "a.bin").stat().st_mtime_ns == stamp
    assert second["errors"] == [] and second["error_history"]
    assert [row["uuid"] for row in manifest(root)] == ["a", "b", "c"]


def test_missing_footer_at_the_limit_still_requires_a_successful_resume_summary(tmp_path, monkeypatch):
    ssh = SSH(
        [
            lambda wire: protocol(wire, [record("a"), record("b")], footer=False),
            protocol,
        ]
    )
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "footer"
    cfg = config(limit=2)
    first = fetch.fetch_corpus(cfg, root)
    assert first["processed"] == 2 and first["scan_complete"] is False
    second = fetch.fetch_corpus(cfg, root, resume=True)
    assert second["scan_complete"] is True
    assert ssh.calls[1][2]["limit"] == 0


def test_completed_resume_verifies_sha_but_does_not_start_another_ssh(tmp_path, monkeypatch):
    ssh = SSH([lambda wire: protocol(wire, [record("a")])])
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "complete"
    cfg = config(limit=1)
    fetch.fetch_corpus(cfg, root)
    result = fetch.fetch_corpus(cfg, root, resume=True)
    assert result["scan_complete"] and len(ssh.calls) == 1
    (root / "a.bin").write_bytes(b"tampered")
    with pytest.raises(fetch.CorpusFetchError, match="SHA-256"):
        fetch.fetch_corpus(cfg, root, resume=True)
    assert checkpoint(root)["scan_complete"] is False


def test_existing_output_requires_explicit_resume_and_scope_cannot_change(tmp_path, monkeypatch):
    ssh = SSH([lambda wire: protocol(wire, [record("a")])])
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "existing"
    cfg = config(limit=1)
    fetch.fetch_corpus(cfg, root)
    with pytest.raises(FileExistsError):
        fetch.fetch_corpus(cfg, root)
    with pytest.raises(fetch.CorpusFetchError, match="scope"):
        fetch.fetch_corpus(replace(cfg, mode_id=16), root, resume=True)
    assert checkpoint(root)["selection"]["mode_id"] == 12


def test_manifest_ahead_of_checkpoint_is_recovered_without_recopy(tmp_path, monkeypatch):
    ssh = SSH([lambda wire: protocol(wire, [record("a"), record("b")]), protocol])
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "ahead"
    cfg = config(limit=2)
    fetch.fetch_corpus(cfg, root)
    fetch._atomic_checkpoint(root, fetch._initial_state(cfg.scope()))
    stamp = (root / "b.bin").stat().st_mtime_ns
    result = fetch.fetch_corpus(cfg, root, resume=True)
    assert result["scan_complete"] and result["processed"] == 2
    assert ssh.calls[1][2]["after_uuid"] == "b" and ssh.calls[1][2]["limit"] == 0
    assert (root / "b.bin").stat().st_mtime_ns == stamp


def test_partial_manifest_tail_can_be_recovered_only_on_explicit_resume(tmp_path, monkeypatch):
    ssh = SSH(
        [
            lambda wire: protocol(wire, [record("a")], footer=False),
            lambda wire: protocol(wire, [record("b"), record("c")]),
        ]
    )
    monkeypatch.setattr(fetch.subprocess, "Popen", ssh)
    root = tmp_path / "partial-manifest"
    fetch.fetch_corpus(config(), root)
    with (root / "manifest.jsonl").open("ab") as target:
        target.write(b'{"uuid":')
    result = fetch.fetch_corpus(config(), root, resume=True)
    assert result["scan_complete"] and len(manifest(root)) == 3


def test_unknown_file_collision_is_never_overwritten(tmp_path, monkeypatch):
    root = tmp_path / "collision"

    def stream(wire):
        (root / "a.bin").write_bytes(b"unknown")
        return protocol(wire, [record("a")], exhausted=True)

    monkeypatch.setattr(fetch.subprocess, "Popen", SSH([stream]))
    result = fetch.fetch_corpus(config(), root)
    assert not result["scan_complete"] and result["processed"] == 0
    assert (root / "a.bin").read_bytes() == b"unknown"


def test_orphan_with_matching_bytes_can_be_verified_on_resume(tmp_path, monkeypatch):
    root = tmp_path / "orphan"
    cfg = config(limit=1)
    root.mkdir()
    (root / "manifest.jsonl").write_bytes(b"")
    (root / "a.bin").write_bytes(b"opaque-a")
    fetch._atomic_checkpoint(root, fetch._initial_state(cfg.scope()))
    monkeypatch.setattr(fetch.subprocess, "Popen", SSH([lambda wire: protocol(wire, [record("a")])]))
    result = fetch.fetch_corpus(cfg, root, resume=True)
    assert result["scan_complete"]
    assert manifest(root)[0]["reused_verified"] is True


def test_nonzero_ssh_exit_or_partial_summary_cannot_complete(tmp_path, monkeypatch):
    for number, builder in enumerate(
        [
            lambda wire: Process(protocol(wire, [record("a")], exhausted=True), code=1),
            lambda wire: protocol(wire, [record("a")], exhausted=False),
            lambda wire: protocol(wire, [record("a")], footer=False)[:-1],
        ]
    ):
        monkeypatch.setattr(fetch.subprocess, "Popen", SSH([builder]))
        result = fetch.fetch_corpus(config(), tmp_path / f"failure-{number}")
        assert result["scan_complete"] is False


def test_keyboard_interrupt_persists_a_noncomplete_checkpoint(tmp_path, monkeypatch):
    class Interrupted(io.BytesIO):
        def readline(self, size=-1):
            raise KeyboardInterrupt

    process = Process(b"")
    process.stdout = Interrupted()
    monkeypatch.setattr(fetch.subprocess, "Popen", lambda *args, **kwargs: process)
    root = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        fetch.fetch_corpus(config(), root)
    assert checkpoint(root)["scan_complete"] is False
    assert checkpoint(root)["status"] == "INTERRUPTED"
    assert process.terminated


def remote_namespace():
    namespace = {"__name__": "trusted_remote_script_test"}
    exec(compile(fetch.REMOTE_SCRIPT, "trusted_corpus_fetch_remote", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\Example\.ssh\safe",
        r"D:\majsoul\record-data-iter-evil\safe",
        r"D:\majsoul\record-data\..\..\safe",
        r"D:\majsoul\record-data\safe:stream",
    ],
)
def test_remote_path_validator_rejects_lexical_escapes(path):
    with pytest.raises(ValueError):
        remote_namespace()["normalized_raw_path"](path, "safe")


def test_remote_resolved_junction_escape_is_rejected_before_open():
    namespace = remote_namespace()
    namespace["Path"] = lambda _: SimpleNamespace(resolve=lambda **kwargs: Path(r"C:\Users\Example\.ssh\safe"))
    row = record("safe")
    metadata = row["locator_metadata"]
    with pytest.raises(ValueError, match="outside_allowed_roots"):
        namespace["read_record"](
            ("safe", row["source_path"], "train", 12, json.dumps(metadata)), {"mode_id": 12, "source_split": "train"}
        )


def test_remote_sql_uses_bound_keyset_pagination_and_no_writes(tmp_path):
    db = tmp_path / "catalog.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE games(uuid TEXT PRIMARY KEY,path TEXT,split TEXT,"
        "mode_id INTEGER,conflicted INTEGER,metadata TEXT)"
    )
    conn.execute("CREATE INDEX game_split ON games(conflicted,split,uuid)")
    for number in range(150):
        conn.execute("INSERT INTO games VALUES(?,?,?,?,?,?)", (f"game-{number:04}", "unused", "train", 12, 0, "{}"))
    conn.execute("INSERT INTO games VALUES(?,?,?,?,?,?)", ("game-0001-golden", "unused", "train", 16, 0, "{}"))
    conn.execute("INSERT INTO games VALUES(?,?,?,?,?,?)", ("game-0001-test", "unused", "test", 12, 0, "{}"))
    conn.execute("INSERT INTO games VALUES(?,?,?,?,?,?)", ("game-0001-conflict", "unused", "train", 12, 1, "{}"))
    conn.commit()
    conn.close()
    namespace = remote_namespace()
    emitted, queries, connections = [], [], []

    def connect(uri_path, **kwargs):
        assert kwargs.get("uri") is True
        assert uri_path.endswith("?mode=ro&immutable=1")
        connections.append(uri_path)
        value = sqlite3.connect(uri_path, uri=True)
        value.set_trace_callback(queries.append)
        return value

    namespace["sqlite3"] = SimpleNamespace(connect=connect)
    namespace["emit"] = emitted.append
    namespace["read_record"] = lambda row, cfg: {"kind": "record", "uuid": row[0]}
    code = namespace["run"](
        {
            "catalog": str(db),
            "after_uuid": "game-0009",
            "source_split": "train",
            "mode_id": 12,
            "limit": 130,
            "scope_sha256": "fixture",
        }
    )
    assert code == 0
    rows = [row["uuid"] for row in emitted if row["kind"] == "record"]
    assert rows == [f"game-{number:04}" for number in range(10, 140)]
    assert emitted[-1]["records"] == 130 and emitted[-1]["all_disk_scanned"] is False
    assert len(connections) == 1 and len(queries) == 3
    assert "OFFSET" not in namespace["SQL"]
    assert namespace["SQL"].count("?") == 4
    readonly = sqlite3.connect(db.as_uri() + "?mode=ro&immutable=1", uri=True)
    assert readonly.execute(namespace["SQL"], ("train' OR 1=1 --", "", 12, 5)).fetchall() == []
    with pytest.raises(sqlite3.OperationalError):
        readonly.execute("DELETE FROM games")
    readonly.close()
