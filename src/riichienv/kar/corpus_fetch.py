from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import ntpath
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "kar_catalog_fetch_v1"
MAX_RECORD_BYTES = 64 * 1024 * 1024
MAX_LIMIT = 1000
MAX_LINE_BYTES = ((MAX_RECORD_BYTES + 2) // 3) * 4 + 256 * 1024
RAW_ROOTS = (r"D:\majsoul\record-data", r"D:\majsoul\record-data-iter")
_UUID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\Z", re.IGNORECASE)
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
_HOST = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]{0,252}\Z")
_PYTHON = re.compile(r"[A-Za-z]:[\\/][A-Za-z0-9 _./\\-]+\.exe\Z", re.IGNORECASE)

REMOTE_SCRIPT = r'''
import base64
import hashlib
import json
import ntpath
import os
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import sys

VERSION = "kar_catalog_fetch_v1"
MAX_RECORD_BYTES = 64 * 1024 * 1024
ROOTS = (r"D:\majsoul\record-data", r"D:\majsoul\record-data-iter")
UUID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\Z",re.IGNORECASE)
SQL = """SELECT uuid,path,split,mode_id,
    CASE WHEN length(metadata)<=65536 THEN metadata ELSE NULL END
    FROM games WHERE conflicted=0 AND split=? AND uuid>? AND mode_id=?
    ORDER BY uuid LIMIT ?"""

def emit(value):
    print(json.dumps(value,ensure_ascii=False,separators=(",",":")),flush=True)

def normalized_raw_path(value, uuid):
    if not isinstance(value,str) or len(value)>4096 or any(ord(c)<32 for c in value):
        raise ValueError("invalid_raw_path")
    drive, tail = ntpath.splitdrive(value)
    if drive.lower()!="d:" or not tail.startswith(("\\","/")) or ":" in tail:
        raise ValueError("raw_path_outside_allowed_roots")
    candidate = ntpath.normcase(ntpath.abspath(value))
    roots = tuple(ntpath.normcase(ntpath.abspath(root)) for root in ROOTS)
    if not any(ntpath.commonpath((candidate,root))==root for root in roots):
        raise ValueError("raw_path_outside_allowed_roots")
    path = PureWindowsPath(candidate)
    bad_device = any(DEVICE.fullmatch(part.rstrip(" .").split(".")[0]) for part in path.parts[1:])
    if path.name.lower()!=uuid.lower() or bad_device:
        raise ValueError("raw_basename_is_not_the_game_uuid")
    return candidate

def read_record(row, cfg):
    uuid,path,split,mode,encoded = row
    if not isinstance(uuid,str) or UUID.fullmatch(uuid) is None:
        raise ValueError("unsafe_catalog_uuid")
    if split!=cfg["source_split"] or mode!=cfg["mode_id"] or encoded is None:
        raise ValueError("catalog_scope_or_metadata_invalid")
    metadata = json.loads(encoded)
    if not isinstance(metadata,dict) or metadata.get("uuid")!=uuid or metadata.get("mode_id")!=mode:
        raise ValueError("catalog_metadata_identity_mismatch")
    expected = metadata.get("sha256")
    if not isinstance(expected,str) or SHA.fullmatch(expected) is None:
        raise ValueError("catalog_raw_sha256_missing")
    lexical = normalized_raw_path(path,uuid)
    if "path" in metadata and normalized_raw_path(metadata["path"],uuid)!=lexical:
        raise ValueError("catalog_metadata_path_mismatch")
    resolved = Path(lexical).resolve(strict=True)
    normalized_raw_path(str(resolved),uuid)
    if not resolved.is_file() or not 0<resolved.stat().st_size<=MAX_RECORD_BYTES:
        raise ValueError("raw_record_size_or_type_invalid")
    with resolved.open("rb") as source:
        raw = source.read(MAX_RECORD_BYTES+1)
    if not 0<len(raw)<=MAX_RECORD_BYTES:
        raise ValueError("raw_record_size_limit")
    actual = hashlib.sha256(raw).hexdigest()
    if actual!=expected.lower() or metadata.get("bytes")!=len(raw):
        raise ValueError("raw_record_differs_from_locator")
    return {"kind":"record","uuid":uuid,"source_path":path,"resolved_source_path":str(resolved),
        "old_source_split":split,"mode_id":mode,"locator_sha256":expected.lower(),
        "source_sha256":actual,"bytes":len(raw),"locator_metadata":metadata,
        "raw_base64":base64.b64encode(raw).decode("ascii")}

def run(cfg):
    if cfg.get("mode_id") not in (12,16) or cfg.get("source_split") not in ("train","validation","test"):
        raise ValueError("invalid_selection_scope")
    limit = cfg.get("limit")
    cursor = cfg.get("after_uuid")
    if isinstance(limit,bool) or not isinstance(limit,int) or not 0<=limit<=1000:
        raise ValueError("invalid_selection_limit")
    if not isinstance(cursor,str) or (cursor and UUID.fullmatch(cursor) is None):
        raise ValueError("invalid_initial_cursor")
    catalog = Path(cfg["catalog"]).resolve(strict=True)
    if not catalog.is_file() or catalog.suffix.lower() not in (".sqlite3",".db"):
        raise ValueError("invalid_catalog_file")
    emit({"kind":"header","protocol":VERSION,"scope_sha256":cfg["scope_sha256"],
        "after_uuid":cursor,"limit":limit,"mode_id":cfg["mode_id"],
        "old_source_split":cfg["source_split"],"locator_read_mode":"immutable_main_ignores_wal"})
    count = 0
    exhausted = False
    db = sqlite3.connect(catalog.as_uri()+"?mode=ro&immutable=1",uri=True)
    try:
        while count<limit:
            rows = db.execute(SQL,(cfg["source_split"],cursor,cfg["mode_id"],min(64,limit-count))).fetchall()
            if not rows:
                exhausted = True
                break
            for row in rows:
                try:
                    record = read_record(row,cfg)
                except Exception as error:
                    emit({"kind":"error","uuid":row[0],"reason":str(error)[:4096],"error_type":type(error).__name__})
                    emit({"kind":"summary","selection_complete":False,"records":count,"last_uuid":cursor,
                        "catalog_exhausted":False,"all_disk_scanned":False})
                    return 1
                emit(record)
                cursor = row[0]
                count += 1
        emit({"kind":"summary","selection_complete":True,"records":count,"last_uuid":cursor,
            "catalog_exhausted":exhausted,"all_disk_scanned":False})
        return 0
    finally:
        db.close()

def main():
    try:
        cfg = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode("utf-8"))
        return run(cfg)
    except Exception as error:
        emit({"kind":"error","uuid":None,"reason":str(error)[:4096],"error_type":type(error).__name__})
        return 1

if __name__=="__main__":
    raise SystemExit(main())
'''


class CorpusFetchError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _uuid(value: Any, *, empty: bool = False) -> str:
    if not isinstance(value, str) or not ((empty and not value) or _UUID.fullmatch(value)) or _DEVICE.fullmatch(value):
        raise CorpusFetchError("Unsafe game UUID")
    return value


def _absolute_windows(value: str) -> str:
    drive, tail = ntpath.splitdrive(value)
    if len(drive) != 2 or drive[1] != ":" or not tail.startswith(("\\", "/")):
        raise CorpusFetchError("An absolute Windows drive path is required")
    if ":" in tail or any(ord(char) < 32 for char in value):
        raise CorpusFetchError("Unsafe Windows path")
    return ntpath.normpath(value)


def validate_raw_path(value: str, uuid: str) -> str:
    candidate = ntpath.normcase(_absolute_windows(value))
    roots = tuple(ntpath.normcase(root) for root in RAW_ROOTS)
    if ntpath.splitdrive(candidate)[0] != "d:" or not any(
        ntpath.commonpath((candidate, root)) == root for root in roots
    ):
        raise CorpusFetchError("Raw path escapes the authorized data roots")
    if ntpath.basename(candidate) != uuid.lower():
        raise CorpusFetchError("Raw basename does not match UUID")
    return candidate


@dataclass(frozen=True)
class FetchConfig:
    host: str
    ssh_executable: str
    remote_python: str
    catalog: str
    limit: int = 1000
    mode_id: int = 12
    source_split: str = "train"
    after_uuid: str = ""

    def scope(self) -> dict[str, Any]:
        if not _HOST.fullmatch(self.host):
            raise CorpusFetchError("Unsafe SSH host; use a plain hostname or user@hostname")
        if not _PYTHON.fullmatch(self.remote_python) or ntpath.basename(self.remote_python).lower() not in {
            "python.exe",
            "python3.exe",
        }:
            raise CorpusFetchError("Remote Python must be a shell-safe absolute Windows Python executable")
        catalog = _absolute_windows(self.catalog)
        if ntpath.splitext(catalog)[1].lower() not in {".sqlite3", ".db"}:
            raise CorpusFetchError("Catalog must name a SQLite file")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or not 1 <= self.limit <= MAX_LIMIT:
            raise CorpusFetchError("Selection limit must be between 1 and 1000")
        if self.mode_id not in {12, 16} or isinstance(self.mode_id, bool):
            raise CorpusFetchError("Only main mode12 or separate golden-only mode16 is supported")
        if self.source_split not in {"train", "validation", "test"}:
            raise CorpusFetchError("Invalid old catalog source split")
        _uuid(self.after_uuid, empty=True)
        return {
            "version": VERSION,
            "host": self.host,
            "remote_python": ntpath.normpath(self.remote_python),
            "catalog": catalog,
            "limit": self.limit,
            "mode_id": self.mode_id,
            "old_source_split": self.source_split,
            "initial_after_uuid": self.after_uuid,
            "selection_role": "main_cohort_candidate" if self.mode_id == 12 else "golden_only",
            "raw_roots": list(RAW_ROOTS),
            "locator_read_mode": "immutable_main_ignores_wal",
            "all_disk_statistics": False,
            "remote_script_sha256": hashlib.sha256(REMOTE_SCRIPT.encode()).hexdigest(),
        }


def _atomic_checkpoint(root: Path, state: dict[str, Any]) -> None:
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".checkpoint-", suffix=".tmp", dir=root, delete=False
        ) as target:
            name = target.name
            target.write(_canonical(state) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, root / "checkpoint.json")
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _local_file(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise CorpusFetchError("Local artifact escapes the output directory")
    return path


def _file_sha(path: Path) -> str:
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_RECORD_BYTES:
        raise CorpusFetchError("Local raw file size/type is invalid")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _initial_state(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "selection": scope,
        "scope_sha256": _digest(scope),
        "last_uuid": scope["initial_after_uuid"],
        "processed": 0,
        "errors": [],
        "error_history": [],
        "selection_complete": False,
        "scan_complete": False,
        "scan_complete_scope": "bounded_selected_catalog_input_only",
        "all_disk_scanned": False,
        "kar_split_assigned": False,
        "protocol_sha256": None,
        "protocol_descriptor_sha256": None,
    }


def _recover_manifest(root: Path, state: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
    manifest = _local_file(root, "manifest.jsonl")
    if not manifest.is_file() or manifest.stat().st_size > MAX_LIMIT * 128 * 1024:
        raise CorpusFetchError("Missing or oversized manifest journal")
    previous_count = state.get("processed")
    if (
        isinstance(previous_count, bool)
        or not isinstance(previous_count, int)
        or not 0 <= previous_count <= scope["limit"]
    ):
        raise CorpusFetchError("Invalid checkpoint count")
    if state.get("scan_complete") is not state.get("selection_complete"):
        raise CorpusFetchError("Inconsistent checkpoint completion flags")
    lines = manifest.read_bytes().splitlines(keepends=True)
    last = scope["initial_after_uuid"]
    count = offset = 0
    truncated = False
    for number, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if number != len(lines) - 1 or state.get("selection_complete"):
                raise CorpusFetchError("Corrupted completed manifest")
            truncated = True
            break
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            raise CorpusFetchError("Corrupted manifest journal") from error
        uuid = _uuid(row.get("uuid"))
        if uuid <= last or row.get("scope_sha256") != state["scope_sha256"]:
            raise CorpusFetchError("Manifest order or scope is invalid")
        expected_name = f"{uuid}.bin"
        if row.get("local_file") != expected_name or row.get("selection_role") != scope["selection_role"]:
            raise CorpusFetchError("Manifest identity or role is invalid")
        if row.get("mode_id") != scope["mode_id"] or row.get("old_source_split") != scope["old_source_split"]:
            raise CorpusFetchError("Manifest room/split differs from the frozen scope")
        metadata = row.get("locator_metadata", {})
        actual = _file_sha(_local_file(root, expected_name))
        if (
            actual != row.get("source_sha256")
            or actual != row.get("locator_sha256")
            or not isinstance(metadata, dict)
            or str(metadata.get("sha256", "")).lower() != actual
        ):
            raise CorpusFetchError("Resume raw SHA-256 mismatch")
        if metadata.get("uuid") != uuid or metadata.get("mode_id") != scope["mode_id"]:
            raise CorpusFetchError("Resume locator identity mismatch")
        if row.get("bytes") != (root / expected_name).stat().st_size or metadata.get("bytes") != row.get("bytes"):
            raise CorpusFetchError("Resume byte count mismatch")
        validate_raw_path(row["source_path"], uuid)
        if "resolved_source_path" in row:
            validate_raw_path(row["resolved_source_path"], uuid)
        last, count = uuid, count + 1
        offset += len(line)
        if count == previous_count and state.get("last_uuid") != last:
            raise CorpusFetchError("Checkpoint cursor is inconsistent")
    if count > scope["limit"] or previous_count > count:
        raise CorpusFetchError("Checkpoint/manifest count is inconsistent")
    if previous_count == 0 and state.get("last_uuid") != scope["initial_after_uuid"]:
        raise CorpusFetchError("Invalid empty checkpoint cursor")
    if state.get("selection_complete") and (count != previous_count or truncated):
        raise CorpusFetchError("Completed checkpoint differs from its manifest")
    if truncated:
        with manifest.open("r+b") as target:
            target.truncate(offset)
            target.flush()
            os.fsync(target.fileno())
    state.update(last_uuid=last, processed=count)
    if truncated or count != previous_count:
        state.update(selection_complete=False, scan_complete=False)
    return state


def _resume(root: Path, scope: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _local_file(root, "checkpoint.json")
    if not checkpoint.is_file():
        raise CorpusFetchError("Resume requires an initialized checkpoint")
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (
        not isinstance(state, dict)
        or state.get("version") != VERSION
        or state.get("scope_sha256") != _digest(scope)
        or state.get("selection") != scope
    ):
        raise CorpusFetchError("Resume selection differs from the frozen fetch scope")
    try:
        state = _recover_manifest(root, state, scope)
    except (CorpusFetchError, OSError, ValueError, KeyError) as error:
        state.update(selection_complete=False, scan_complete=False, status="FAILED")
        state["errors"].append({"error_type": type(error).__name__, "detail": str(error), "phase": "resume_integrity"})
        _atomic_checkpoint(root, state)
        raise CorpusFetchError(str(error)) from error
    _atomic_checkpoint(root, state)
    return state


def _wire_config(config: FetchConfig, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "catalog": state["selection"]["catalog"],
        "mode_id": config.mode_id,
        "source_split": config.source_split,
        "limit": config.limit - state["processed"],
        "after_uuid": state["last_uuid"],
        "scope_sha256": state["scope_sha256"],
    }


def ssh_command(config: FetchConfig, wire: dict[str, Any]) -> list[str]:
    config.scope()
    payload = base64.urlsafe_b64encode(_canonical(wire)).decode("ascii")
    command = f'"{config.remote_python}" -I -B -S - {payload}'
    return [
        config.ssh_executable,
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        "--",
        config.host,
        command,
    ]


def _write_record(root: Path, state: dict[str, Any], row: dict[str, Any], *, resume: bool) -> None:  # noqa: PLR0915
    uuid = _uuid(row.get("uuid"))
    if (
        uuid <= state["last_uuid"]
        or row.get("mode_id") != state["selection"]["mode_id"]
        or row.get("old_source_split") != state["selection"]["old_source_split"]
    ):
        raise CorpusFetchError("Remote record is outside the requested keyset scope")
    validate_raw_path(row["source_path"], uuid)
    if "resolved_source_path" in row:
        validate_raw_path(row["resolved_source_path"], uuid)
    metadata = row.get("locator_metadata")
    expected = row.get("locator_sha256")
    if not isinstance(metadata, dict) or metadata.get("uuid") != uuid or metadata.get("mode_id") != row["mode_id"]:
        raise CorpusFetchError("Locator identity mismatch")
    if "path" in metadata and validate_raw_path(metadata["path"], uuid) != validate_raw_path(row["source_path"], uuid):
        raise CorpusFetchError("Locator path differs from the raw source path")
    if (
        not isinstance(expected, str)
        or not _SHA.fullmatch(expected)
        or metadata.get("sha256", "").lower() != expected.lower()
    ):
        raise CorpusFetchError("Missing or inconsistent old locator SHA-256")
    try:
        data = base64.b64decode(row["raw_base64"], validate=True)
    except (ValueError, TypeError, binascii.Error) as error:
        raise CorpusFetchError("Invalid raw base64 payload") from error
    actual = hashlib.sha256(data).hexdigest()
    if not 0 < len(data) <= MAX_RECORD_BYTES or row.get("bytes") != len(data) or metadata.get("bytes") != len(data):
        raise CorpusFetchError("Raw byte count does not match the locator")
    if actual != expected.lower() or row.get("source_sha256") != actual:
        raise CorpusFetchError("Raw payload SHA-256 mismatch")
    name = f"{uuid}.bin"
    target = _local_file(root, name)
    reused = target.exists()
    if reused:
        if not resume or _file_sha(target) != actual:
            raise CorpusFetchError("Existing raw file is unknown or has a different SHA-256")
    else:
        created = False
        try:
            with target.open("xb") as destination:
                created = True
                destination.write(data)
                destination.flush()
                os.fsync(destination.fileno())
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
    manifest_row = {key: value for key, value in row.items() if key not in {"raw_base64", "kind"}}
    manifest_row.update(
        scope_sha256=state["scope_sha256"],
        local_file=name,
        selection_role=state["selection"]["selection_role"],
        reused_verified=reused,
        provenance="opaque_raw_copy_from_old_catalog_locator",
        training_eligible=False,
        kar_split=None,
        protocol_sha256=None,
        protocol_descriptor_sha256=None,
        old_schema_hash_is_protocol_hash=False,
    )
    with _local_file(root, "manifest.jsonl").open("ab") as manifest:
        manifest.write(_canonical(manifest_row) + b"\n")
        manifest.flush()
        os.fsync(manifest.fileno())
    state.update(last_uuid=uuid, processed=state["processed"] + 1)
    _atomic_checkpoint(root, state)


def _stream(
    process: Any, root: Path, config: FetchConfig, state: dict[str, Any], wire: dict[str, Any], *, resume: bool
) -> dict[str, Any]:
    if process.stdin is None or process.stdout is None:
        raise CorpusFetchError("SSH process did not expose its streams")
    process.stdin.write(REMOTE_SCRIPT.encode("utf-8"))
    process.stdin.close()
    header_seen = False
    footer = None
    received = 0
    for _ in range(wire["limit"] + 3):
        line = process.stdout.readline(MAX_LINE_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
            raise CorpusFetchError("Truncated or oversized remote JSONL line")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise CorpusFetchError("Remote JSONL entry is not an object")
        kind = row.get("kind")
        if not header_seen:
            expected = {
                "kind": "header",
                "protocol": VERSION,
                "scope_sha256": state["scope_sha256"],
                "after_uuid": wire["after_uuid"],
                "limit": wire["limit"],
                "mode_id": config.mode_id,
                "old_source_split": config.source_split,
                "locator_read_mode": "immutable_main_ignores_wal",
            }
            if row != expected:
                raise CorpusFetchError("Missing or mismatched remote selection header")
            header_seen = True
        elif footer is not None:
            raise CorpusFetchError("Remote data continues after its final summary")
        elif kind == "error":
            raise CorpusFetchError(f"Remote selection failure for {row.get('uuid')}: {row.get('reason')}")
        elif kind == "record":
            if received >= wire["limit"]:
                raise CorpusFetchError("Remote record count exceeds the requested limit")
            _write_record(root, state, row, resume=resume)
            received += 1
        elif kind == "summary":
            footer = row
        else:
            raise CorpusFetchError("Unexpected remote JSONL entry")
    if process.stdout.read(1):
        raise CorpusFetchError("Remote output exceeded its bounded protocol")
    exit_code = process.wait(timeout=30)
    if not header_seen or footer is None or exit_code != 0:
        raise CorpusFetchError("Remote selection ended without a successful final summary")
    if (
        footer.get("selection_complete") is not True
        or footer.get("all_disk_scanned") is not False
        or footer.get("records") != received
        or footer.get("last_uuid") != state["last_uuid"]
    ):
        raise CorpusFetchError("Remote final summary disagrees with committed local records")
    if not isinstance(footer.get("catalog_exhausted"), bool) or (
        received < wire["limit"] and not footer["catalog_exhausted"]
    ):
        raise CorpusFetchError("Partial selection cannot be marked complete")
    return footer


def fetch_corpus(config: FetchConfig, output_dir: str | Path, *, resume: bool = False) -> dict[str, Any]:
    scope = config.scope()
    root = Path(output_dir).resolve()
    if resume:
        if not root.is_dir():
            raise CorpusFetchError("Resume output directory does not exist")
        state = _resume(root, scope)
    else:
        root.mkdir(parents=True, exist_ok=False)
        with (root / "manifest.jsonl").open("xb"):
            pass
        state = _initial_state(scope)
        _atomic_checkpoint(root, state)
    if state.get("selection_complete"):
        return state
    wire = _wire_config(config, state)
    process = None
    state["error_history"].extend(state["errors"])
    state["errors"] = []
    state.update(selection_complete=False, scan_complete=False, status="RUNNING")
    _atomic_checkpoint(root, state)
    with tempfile.TemporaryFile(mode="w+b") as diagnostics:
        try:
            process = subprocess.Popen(
                ssh_command(config, wire), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics
            )
            footer = _stream(process, root, config, state, wire, resume=resume)
            state.update(selection_complete=True, scan_complete=True, status="COMPLETE", remote_summary=footer)
        except BaseException as error:
            state.update(
                selection_complete=False,
                scan_complete=False,
                status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
            )
            state["errors"].append(
                {"after_uuid": state["last_uuid"], "error_type": type(error).__name__, "detail": str(error)[:4096]}
            )
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            diagnostics.seek(0)
            state["ssh_diagnostic"] = diagnostics.read(4096).decode("utf-8", errors="replace")
            _atomic_checkpoint(root, state)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is not None and not stream.closed:
                        stream.close()
    _atomic_checkpoint(root, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded read-only Mahjong Soul catalog fetch; not a KAR dataset audit"
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--ssh-executable", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--mode-id", type=int, choices=(12, 16), default=12)
    parser.add_argument("--source-split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--after-uuid", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = FetchConfig(
        args.host,
        args.ssh_executable,
        args.remote_python,
        args.catalog,
        args.limit,
        args.mode_id,
        args.source_split,
        args.after_uuid,
    )
    try:
        result = fetch_corpus(config, args.output, resume=args.resume)
    except (CorpusFetchError, OSError, ValueError) as error:
        parser.exit(1, f"Fetch failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["selection_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
