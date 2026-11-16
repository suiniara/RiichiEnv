from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import corpus_fetch as transport
from .mjsoul import Protocol, load_protocol

VERSION = "kar_rare_candidate_search_v1"
MAX_SCANNED = 10000
TAGS = frozenset({"possible_kokushi_ankan_ron", "possible_four_kans", "possible_four_riichi"})

REMOTE_SCRIPT = r'''
import base64
import hashlib
import json
import ntpath
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import sys

VERSION = "kar_rare_candidate_search_v1"
MAX_BYTES = 64*1024*1024
ROOTS = (r"D:\majsoul\record-data",r"D:\majsoul\record-data-iter")
UUID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\Z",re.IGNORECASE)
SQL = """SELECT uuid,path,split,mode_id,
    CASE WHEN length(metadata)<=65536 THEN metadata ELSE NULL END
    FROM games WHERE conflicted=0 AND split=? AND uuid>? AND mode_id=?
    ORDER BY uuid LIMIT ?"""
EXPECTED = {
    "wrapper":{"name":(9,False),"data":(12,False)},
    "response":{"data":(12,False)},
    "detail":{"records":(12,True),"version":(13,False),"actions":(11,True)},
    "action":{"result":(12,False)},
    "kan":{"seat":(13,False),"type":(13,False),"tiles":(9,False)},
    "hule":{"hules":(11,True)},
    "winner":{"seat":(13,False),"hu_tile":(9,False),"zimo":(8,False)},
    "call":{"seat":(13,False),"type":(13,False)},
    "abort":{"type":(13,False)},
}

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()

def emit(value):
    print(json.dumps(value,ensure_ascii=False,separators=(",",":")),flush=True)

def validate_map(mapping):
    if mapping.get("version")!="kar_rare_wire_map_v1":
        raise ValueError("unknown_field_mapping_version")
    messages = mapping["messages"]
    for key,fields in EXPECTED.items():
        message = messages[key]
        seen = set()
        for name,(kind,repeated) in fields.items():
            field = message["fields"][name]
            number = field["number"]
            if isinstance(number,bool) or not isinstance(number,int) or not 0<number<(1<<29):
                raise ValueError("invalid_frozen_field_number")
            if field["type"]!=kind or field["repeated"] is not repeated or number in seen:
                raise ValueError("incompatible_frozen_field_type")
            seen.add(number)
    if messages["detail"]["fields"]["actions"].get("message")!=messages["action"]["name"]:
        raise ValueError("action_message_mapping_mismatch")
    if messages["hule"]["fields"]["hules"].get("message")!=messages["winner"]["name"]:
        raise ValueError("winner_message_mapping_mismatch")

def varint(data,offset):
    value = 0
    for shift in range(0,70,7):
        if offset>=len(data):
            raise ValueError("truncated_varint")
        byte = data[offset]
        offset += 1
        if shift==63 and byte>1:
            raise ValueError("oversized_varint")
        value |= (byte&127)<<shift
        if not byte&128:
            return value,offset
    raise ValueError("unterminated_varint")

def fields(data):
    offset = 0
    while offset<len(data):
        tag,offset = varint(data,offset)
        number,wire = tag>>3,tag&7
        if not 0<number<(1<<29):
            raise ValueError("invalid_field_number")
        if wire==0:
            value,offset = varint(data,offset)
        elif wire in (1,2,5):
            if wire==2:
                size,offset = varint(data,offset)
            else:
                size = 8 if wire==1 else 4
            end = offset+size
            if end>len(data):
                raise ValueError("truncated_wire_field")
            value = data[offset:end]
            offset = end
        else:
            raise ValueError("unsupported_wire_type")
        yield number,wire,value

def values(data,mapping,message,name):
    spec = mapping["messages"][message]["fields"][name]
    wanted = spec["number"]
    expected_wire = 2 if spec["type"] in (9,11,12) else 0
    found = []
    for number,wire,value in fields(data):
        if number==wanted:
            if wire!=expected_wire:
                raise ValueError("mapped_wire_type_mismatch")
            found.append(value)
    if not spec["repeated"] and len(found)>1:
        raise ValueError("duplicate_singular_field")
    if spec["repeated"]:
        return found
    return found[0] if found else b"" if expected_wire==2 else 0

def text(data,mapping,message,name):
    return values(data,mapping,message,name).decode("utf-8")

def wrapper(data,mapping,allowed):
    name = text(data,mapping,"wrapper","name").removeprefix(".")
    body = values(data,mapping,"wrapper","data")
    if name not in allowed or not body:
        raise ValueError("unknown_or_empty_wrapper:"+name)
    return name,body

def results(raw,mapping):
    if not raw.startswith(b"\0\0\0") or len(raw)<=3:
        raise ValueError("unsupported_archive_prefix")
    _,response = wrapper(raw[3:],mapping,{""})
    detail_wrapper = values(response,mapping,"response","data")
    _,detail = wrapper(detail_wrapper,mapping,{mapping["messages"]["detail"]["name"]})
    legacy = values(detail,mapping,"detail","records")
    actions = values(detail,mapping,"detail","actions")
    version = values(detail,mapping,"detail","version")
    if legacy and actions:
        raise ValueError("mixed_record_storage")
    if version==0:
        if actions or not legacy:
            raise ValueError("invalid_legacy_storage")
        return version,list(enumerate(legacy))
    if version==210715:
        if legacy or not actions:
            raise ValueError("invalid_action_storage")
        extracted = []
        for index,action in enumerate(actions):
            result = values(action,mapping,"action","result")
            if result:
                extracted.append((index,result))
        if not extracted:
            raise ValueError("no_server_result")
        return version,extracted
    raise ValueError("unknown_record_version:"+str(version))

def tile_base(value):
    honors = {"E":27,"S":28,"W":29,"N":30,"P":31,"F":32,"C":33}
    if value in honors:
        return honors[value]
    if value in ("5mr","5pr","5sr"):
        value = value[:2]
    if len(value)==2 and value[1] in "mps" and value[0] in "0123456789":
        rank = 5 if value[0]=="0" else int(value[0])
        return "mps".index(value[1])*9+rank-1
    if len(value)==2 and value[1]=="z" and value[0] in "1234567":
        return 26+int(value[0])
    raise ValueError("invalid_tile_string")

def seat(value):
    if not isinstance(value,int) or not 0<=value<4:
        raise ValueError("invalid_four_player_seat")
    return value

def detect(raw,mapping):
    validate_map(mapping)
    version,records = results(raw,mapping)
    names = mapping["records"]
    allowed = set(names.values())|set(mapping["neutral_records"])
    round_index = -1
    pending = None
    kan_actors = []
    evidence = []
    tags = set()
    for index,result in records:
        name,body = wrapper(result,mapping,allowed)
        if name==names["round"]:
            round_index += 1
            pending = None
            kan_actors = []
            continue
        if round_index<0:
            raise ValueError("record_before_first_round")
        if name in mapping["neutral_records"]:
            continue
        if name==names["kan"]:
            actor = seat(values(body,mapping,"kan","seat"))
            kind = values(body,mapping,"kan","type")
            tile = tile_base(text(body,mapping,"kan","tiles"))
            if kind not in (2,3):
                raise ValueError("unknown_kan_type")
            kan_actors.append(actor)
            pending = (actor,tile,index) if kind==3 else None
        elif name==names["call"]:
            actor = seat(values(body,mapping,"call","seat"))
            kind = values(body,mapping,"call","type")
            if kind not in (0,1,2):
                raise ValueError("unknown_open_meld_type")
            if kind==2:
                kan_actors.append(actor)
            pending = None
        elif name==names["win"]:
            winners = values(body,mapping,"hule","hules")
            if not winners:
                raise ValueError("empty_winner_list")
            matched = []
            for winner in winners:
                actor = seat(values(winner,mapping,"winner","seat"))
                zimo = values(winner,mapping,"winner","zimo")
                if zimo not in (0,1):
                    raise ValueError("invalid_win_boolean")
                tile = tile_base(text(winner,mapping,"winner","hu_tile"))
                if pending is not None and not zimo and actor!=pending[0] and tile==pending[1]:
                    matched.append(actor)
            if matched:
                tags.add("possible_kokushi_ankan_ron")
                evidence.append({"tag":"possible_kokushi_ankan_ron","round_index":round_index,
                    "record_index":index,"declaration_record_index":pending[2],"winner_seats":matched})
            pending = None
        elif name==names["abort"]:
            kind = values(body,mapping,"abort","type")
            if kind in (2,3) and len(kan_actors)>=4 and len(set(kan_actors))>1:
                tags.add("possible_four_kans")
                evidence.append({"tag":"possible_four_kans","round_index":round_index,
                    "record_index":index,"liuju_type":kind,"kan_declarations":len(kan_actors),
                    "kan_actors":kan_actors[:16]})
            if kind==4:
                tags.add("possible_four_riichi")
                evidence.append({"tag":"possible_four_riichi","round_index":round_index,"record_index":index})
            pending = None
        else:
            pending = None
        if len(evidence)>128:
            raise ValueError("too_many_candidate_occurrences")
    return {"record_version":version,"candidate_tags":sorted(tags),"candidate_evidence":evidence,
        "semantic_verification_performed":False,"candidate_is_confirmed_positive":False}

def safe_path(value,uuid):
    if not isinstance(value,str) or len(value)>4096 or any(ord(c)<32 for c in value):
        raise ValueError("unsafe_raw_path")
    drive,tail = ntpath.splitdrive(value)
    if drive.lower()!="d:" or not tail.startswith(("\\","/")) or ":" in tail:
        raise ValueError("raw_path_outside_allowed_roots")
    actual = ntpath.normcase(ntpath.abspath(value))
    roots = [ntpath.normcase(ntpath.abspath(root)) for root in ROOTS]
    if not any(ntpath.commonpath((actual,root))==root for root in roots):
        raise ValueError("raw_path_outside_allowed_roots")
    path = PureWindowsPath(actual)
    if path.name.lower()!=uuid.lower() or any(DEVICE.fullmatch(p.rstrip(" .").split(".")[0]) for p in path.parts[1:]):
        raise ValueError("invalid_raw_basename")
    return actual

def read_raw(row,cfg):
    uuid,path,split,mode,encoded = row
    if not isinstance(uuid,str) or UUID.fullmatch(uuid) is None or DEVICE.fullmatch(uuid):
        raise ValueError("unsafe_catalog_uuid")
    if split!=cfg["source_split"] or mode!=cfg["mode_id"] or encoded is None:
        raise ValueError("catalog_scope_or_metadata_mismatch")
    md = json.loads(encoded)
    if not isinstance(md,dict) or md.get("uuid")!=uuid or md.get("mode_id")!=mode:
        raise ValueError("catalog_metadata_identity_mismatch")
    sha = md.get("sha256")
    if not isinstance(sha,str) or SHA.fullmatch(sha) is None:
        raise ValueError("locator_sha_missing")
    path = safe_path(path,uuid)
    if "path" in md and safe_path(md["path"],uuid)!=path:
        raise ValueError("locator_path_mismatch")
    resolved = Path(path).resolve(strict=True)
    safe_path(str(resolved),uuid)
    if not resolved.is_file() or not 0<resolved.stat().st_size<=MAX_BYTES:
        raise ValueError("raw_size_or_type_invalid")
    with resolved.open("rb") as source:
        raw = source.read(MAX_BYTES+1)
    actual = hashlib.sha256(raw).hexdigest()
    if not 0<len(raw)<=MAX_BYTES or len(raw)!=md.get("bytes") or actual!=sha.lower():
        raise ValueError("raw_sha_or_size_differs_from_locator")
    return raw,{"uuid":uuid,"source_path":path,"resolved_source_path":str(resolved),
        "old_source_split":split,"mode_id":mode,"locator_sha256":sha.lower(),
        "source_sha256":actual,"bytes":len(raw),"locator_metadata":md}

def run(cfg):
    mapping = cfg["field_map"]
    validate_map(mapping)
    if hashlib.sha256(canonical(mapping)).hexdigest()!=cfg["field_map_sha256"]:
        raise ValueError("field_map_hash_mismatch")
    maximum = cfg["max_scanned"]
    cursor = cfg["after_uuid"]
    if isinstance(maximum,bool) or not isinstance(maximum,int) or not 0<=maximum<=10000:
        raise ValueError("invalid_scan_bound")
    if cfg["mode_id"] not in (12,16) or cfg["source_split"] not in ("train","validation","test"):
        raise ValueError("invalid_selection_scope")
    if not isinstance(cursor,str) or (cursor and UUID.fullmatch(cursor) is None):
        raise ValueError("invalid_scan_cursor")
    catalog = Path(cfg["catalog"]).resolve(strict=True)
    emit({"kind":"header","protocol":VERSION,"scope_sha256":cfg["scope_sha256"],
        "field_map_sha256":cfg["field_map_sha256"],"after_uuid":cursor,"max_scanned":maximum,
        "mode_id":cfg["mode_id"],"old_source_split":cfg["source_split"],
        "locator_read_mode":"immutable_main_ignores_wal"})
    scanned = hits = errors = 0
    exhausted = False
    db = sqlite3.connect(catalog.as_uri()+"?mode=ro&immutable=1",uri=True)
    def progress(kind,**extra):
        return {"kind":kind,"scanned":scanned,"hits":hits,"error_count":errors,
            "last_scanned_uuid":cursor,**extra}
    try:
        while scanned<maximum:
            rows = db.execute(SQL,(cfg["source_split"],cursor,cfg["mode_id"],min(64,maximum-scanned))).fetchall()
            if not rows:
                exhausted = True
                break
            for row in rows:
                uuid = row[0]
                if not isinstance(uuid,str) or UUID.fullmatch(uuid) is None or DEVICE.fullmatch(uuid):
                    raise ValueError("catalog_uuid_cannot_be_checkpointed")
                scanned += 1
                cursor = uuid
                try:
                    raw,metadata = read_raw(row,cfg)
                    detection = detect(raw,mapping)
                    if detection["candidate_tags"]:
                        hits += 1
                        emit(progress("candidate",**metadata,**detection,
                            raw_base64=base64.b64encode(raw).decode("ascii"),
                            search_protocol_sha256=mapping["source_proto_sha256"],
                            field_map_sha256=cfg["field_map_sha256"]))
                except Exception as error:
                    errors += 1
                    source_path = row[1][:4096] if isinstance(row[1],str) else None
                    emit(progress("input_error",uuid=uuid,source_path=source_path,
                        error_type=type(error).__name__,reason=str(error)[:2048],
                        classification="unknown_not_a_verified_negative"))
                if scanned%50==0:
                    emit(progress("progress"))
        emit(progress("summary",selection_complete=True,catalog_exhausted=exhausted,
            all_disk_scanned=False,semantic_verification_performed=False))
        return 0
    finally:
        db.close()

def main():
    try:
        cfg = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())
        return run(cfg)
    except Exception as error:
        emit({"kind":"fatal","error_type":type(error).__name__,"reason":str(error)[:2048]})
        return 1

if __name__=="__main__":
    raise SystemExit(main())
'''


class RareSearchError(ValueError):
    pass


_EXPECTED = {
    "wrapper": {"name": (9, False), "data": (12, False)},
    "response": {"data": (12, False)},
    "detail": {"records": (12, True), "version": (13, False), "actions": (11, True)},
    "action": {"result": (12, False)},
    "kan": {"seat": (13, False), "type": (13, False), "tiles": (9, False)},
    "hule": {"hules": (11, True)},
    "winner": {"seat": (13, False), "hu_tile": (9, False), "zimo": (8, False)},
    "call": {"seat": (13, False), "type": (13, False)},
    "abort": {"type": (13, False)},
}


def freeze_field_map(protocol: Protocol) -> dict[str, Any]:
    pool = protocol.pool
    descriptors = {
        key: pool.FindMessageTypeByName(name)
        for key, name in {
            "wrapper": "lq.Wrapper",
            "response": "lq.ResGameRecord",
            "detail": "lq.GameDetailRecords",
            "kan": "lq.RecordAnGangAddGang",
            "hule": "lq.RecordHule",
            "call": "lq.RecordChiPengGang",
            "abort": "lq.RecordLiuJu",
        }.items()
    }
    descriptors["action"] = descriptors["detail"].fields_by_name["actions"].message_type
    descriptors["winner"] = descriptors["hule"].fields_by_name["hules"].message_type
    messages = {}
    for key, wanted in _EXPECTED.items():
        descriptor = descriptors[key]
        fields = {}
        for name, (kind, repeated) in wanted.items():
            field = descriptor.fields_by_name[name]
            if field.type != kind or field.is_repeated is not repeated:
                raise RareSearchError(f"Unsupported descriptor contract: {descriptor.full_name}.{name}")
            fields[name] = {"number": field.number, "type": field.type, "repeated": field.is_repeated}
            if field.message_type is not None:
                fields[name]["message"] = field.message_type.full_name
        messages[key] = {"name": descriptor.full_name, "fields": fields}
    records = {
        "round": "lq.RecordNewRound",
        "draw": "lq.RecordDealTile",
        "discard": "lq.RecordDiscardTile",
        "call": descriptors["call"].full_name,
        "kan": descriptors["kan"].full_name,
        "win": descriptors["hule"].full_name,
        "notile": "lq.RecordNoTile",
        "abort": descriptors["abort"].full_name,
    }
    for name in records.values():
        pool.FindMessageTypeByName(name)
    neutral = []
    try:
        pool.FindMessageTypeByName("lq.RecordAnalysisedData")
    except KeyError:
        pass
    else:
        neutral.append("lq.RecordAnalysisedData")
    return {
        "version": "kar_rare_wire_map_v1",
        "messages": messages,
        "records": records,
        "neutral_records": neutral,
        "source_proto_sha256": protocol.source_sha256,
        "descriptor_sha256": protocol.descriptor_sha256,
    }


@dataclass(frozen=True)
class SearchConfig:
    host: str
    ssh_executable: str
    remote_python: str
    catalog: str
    proto: Path
    max_scanned: int = MAX_SCANNED
    mode_id: int = 12
    source_split: str = "train"
    after_uuid: str = ""

    def connection(self) -> transport.FetchConfig:
        return transport.FetchConfig(
            self.host,
            self.ssh_executable,
            self.remote_python,
            self.catalog,
            1,
            self.mode_id,
            self.source_split,
            self.after_uuid,
        )

    def scope(self, mapping: dict[str, Any]) -> dict[str, Any]:
        base = self.connection().scope()
        if (
            isinstance(self.max_scanned, bool)
            or not isinstance(self.max_scanned, int)
            or not 1 <= self.max_scanned <= MAX_SCANNED
        ):
            raise RareSearchError("max_scanned must be between 1 and 10000")
        base.update(
            version=VERSION,
            limit=self.max_scanned,
            max_scanned=self.max_scanned,
            selection_role="golden_only",
            source_proto_sha256=mapping["source_proto_sha256"],
            field_map_sha256=transport._digest(mapping),
            remote_script_sha256=hashlib.sha256(REMOTE_SCRIPT.encode()).hexdigest(),
        )
        return base


def _initial(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "selection": scope,
        "scope_sha256": transport._digest(scope),
        "scanned": 0,
        "hits": 0,
        "error_count": 0,
        "processed": 0,
        "last_scanned_uuid": scope["initial_after_uuid"],
        "last_uuid": scope["initial_after_uuid"],
        "selection_complete": False,
        "scan_complete": False,
        "all_disk_scanned": False,
        "semantic_verification_performed": False,
        "training_eligible": False,
        "errors": [],
        "error_history": [],
        "status": "INITIALIZED",
        "scan_complete_scope": "bounded_light_scan_not_confirmed_golden_coverage",
    }


def _journal(root: Path, name: str, completed: bool) -> list[dict[str, Any]]:
    path = transport._local_file(root, name)
    rows = []
    offset = 0
    with path.open("rb") as source:
        while line := source.readline(256 * 1024 + 1):
            if len(line) > 256 * 1024:
                raise RareSearchError("Oversized local search journal row")
            if not line.endswith(b"\n"):
                if completed or source.read(1):
                    raise RareSearchError("Corrupted search journal")
                break
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RareSearchError("Search journal entry must be an object")
            rows.append(row)
            offset += len(line)
            if len(rows) > MAX_SCANNED:
                raise RareSearchError("Search journal exceeds its input bound")
    if path.stat().st_size != offset:
        with path.open("r+b") as target:
            target.truncate(offset)
            target.flush()
            os.fsync(target.fileno())
    return rows


def _resume(root: Path, scope: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
    checkpoint = transport._local_file(root, "checkpoint.json")
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (
        state.get("version") != VERSION
        or state.get("selection") != scope
        or state.get("scope_sha256") != transport._digest(scope)
    ):
        raise RareSearchError("Resume scope differs from the frozen search")
    try:
        hits = _journal(root, "manifest.jsonl", state["selection_complete"])
        failures = _journal(root, "errors.jsonl", state["selection_complete"])
        positions = [state]
        last_hit = scope["initial_after_uuid"]
        for row in hits:
            uuid = transport._uuid(row["uuid"])
            if uuid <= last_hit or row.get("scope_sha256") != state["scope_sha256"]:
                raise RareSearchError("Candidate journal identity/order mismatch")
            if row.get("selection_role") != "golden_only" or row.get("mode_id") != scope["mode_id"]:
                raise RareSearchError("Candidate room/role mismatch")
            if row.get("local_file") != f"{uuid}.bin":
                raise RareSearchError("Unsafe local candidate filename")
            path = transport._local_file(root, row["local_file"])
            metadata = row.get("locator_metadata", {})
            actual = transport._file_sha(path)
            if (
                actual != row.get("source_sha256")
                or actual != row.get("locator_sha256")
                or not isinstance(metadata, dict)
                or str(metadata.get("sha256", "")).lower() != actual
            ):
                raise RareSearchError("Candidate resume SHA-256 mismatch")
            if metadata.get("uuid") != uuid or metadata.get("mode_id") != scope["mode_id"]:
                raise RareSearchError("Candidate locator identity mismatch")
            if row.get("old_source_split") != scope["old_source_split"] or row.get("bytes") != path.stat().st_size:
                raise RareSearchError("Candidate split or byte count mismatch")
            if (
                row.get("search_protocol_sha256") != scope["source_proto_sha256"]
                or row.get("field_map_sha256") != scope["field_map_sha256"]
            ):
                raise RareSearchError("Candidate frozen protocol differs from checkpoint")
            tags = row.get("candidate_tags")
            if not isinstance(tags, list) or not tags or not set(tags) <= TAGS:
                raise RareSearchError("Invalid stored candidate tags")
            if row.get("candidate_is_confirmed_positive") is not False or row.get("training_eligible") is not False:
                raise RareSearchError("Stored lightweight candidate claimed acceptance")
            transport.validate_raw_path(row["source_path"], uuid)
            if "resolved_source_path" in row:
                transport.validate_raw_path(row["resolved_source_path"], uuid)
            positions.append(row["scan_position"])
            last_hit = uuid
        for row in failures:
            transport._uuid(row["uuid"])
            if row.get("scope_sha256") != state["scope_sha256"]:
                raise RareSearchError("Error journal scope mismatch")
            positions.append(row["scan_position"])
        journal_state = {"scanned": 0, "hits": 0, "error_count": 0, "last_scanned_uuid": scope["initial_after_uuid"]}
        zero = {"scanned": 0, "hits": 0, "error_count": 0}
        entries = [(row, "candidate") for row in hits] + [(row, "input_error") for row in failures]
        for row, kind in sorted(entries, key=lambda entry: entry[0]["scan_position"]["scanned"]):
            recovered = _position(row["scan_position"], zero, journal_state, scope["max_scanned"], kind)
            if recovered["last_scanned_uuid"] != row["uuid"]:
                raise RareSearchError("Journal scan position differs from its UUID")
            journal_state.update(recovered)
        best = max(positions, key=lambda row: row["scanned"])
        if best["hits"] != len(hits) or best["error_count"] != len(failures):
            raise RareSearchError("Checkpoint/journal counters disagree")
        if not 0 <= best["scanned"] <= scope["max_scanned"] or best["scanned"] < len(hits) + len(failures):
            raise RareSearchError("Invalid resumed scan count")
        transport._uuid(best["last_scanned_uuid"], empty=True)
        if state["scan_complete"] is not (state["selection_complete"] and state["error_count"] == 0):
            raise RareSearchError("Checkpoint completion flags contradict input errors")
        if state["selection_complete"] and best["scanned"] != state["scanned"]:
            raise RareSearchError("Completed search has uncommitted journal data")
        if best["scanned"] != state["scanned"]:
            state.update(selection_complete=False, scan_complete=False)
        state.update(
            scanned=best["scanned"],
            hits=len(hits),
            error_count=len(failures),
            processed=len(hits),
            last_uuid=last_hit,
            last_scanned_uuid=best["last_scanned_uuid"],
        )
    except (ValueError, OSError, KeyError) as error:
        state.update(selection_complete=False, scan_complete=False, status="FAILED")
        state["errors"].append({"phase": "resume_integrity", "detail": str(error)})
        transport._atomic_checkpoint(root, state)
        raise RareSearchError(str(error)) from error
    transport._atomic_checkpoint(root, state)
    return state


def _position(
    message: dict[str, Any], base: dict[str, Any], state: dict[str, Any], maximum: int, kind: str
) -> dict[str, Any]:
    for key in ("scanned", "hits", "error_count"):
        if isinstance(message.get(key), bool) or not isinstance(message.get(key), int) or message[key] < 0:
            raise RareSearchError("Invalid remote scan counters")
    absolute = {key: base[key] + message[key] for key in ("scanned", "hits", "error_count")}
    cursor = transport._uuid(message.get("last_scanned_uuid"), empty=absolute["scanned"] == 0)
    if not state["scanned"] <= absolute["scanned"] <= maximum:
        raise RareSearchError("Remote scanner exceeded or rewound its scan bound")
    if absolute["scanned"] == state["scanned"]:
        if cursor != state["last_scanned_uuid"] or kind in {"candidate", "input_error"}:
            raise RareSearchError("Repeated or inconsistent scanned UUID")
    elif cursor <= state["last_scanned_uuid"]:
        raise RareSearchError("Scanned cursor did not advance")
    expected_hits = state["hits"] + int(kind == "candidate")
    expected_errors = state["error_count"] + int(kind == "input_error")
    if absolute["hits"] != expected_hits or absolute["error_count"] != expected_errors:
        raise RareSearchError("Remote output omitted a hit or error")
    if absolute["hits"] + absolute["error_count"] > absolute["scanned"]:
        raise RareSearchError("Scan outcome counters exceed inputs")
    absolute["last_scanned_uuid"] = cursor
    return absolute


def _append_error(root: Path, state: dict[str, Any], row: dict[str, Any], position: dict[str, Any]) -> None:
    uuid = transport._uuid(row.get("uuid"))
    if uuid != position["last_scanned_uuid"]:
        raise RareSearchError("Error UUID differs from its cursor")
    output = {
        **row,
        "scope_sha256": state["scope_sha256"],
        "scan_position": position,
        "classification": "unknown_not_a_verified_negative",
    }
    with transport._local_file(root, "errors.jsonl").open("ab") as target:
        target.write(transport._canonical(output) + b"\n")
        target.flush()
        os.fsync(target.fileno())


def _consume(  # noqa: PLR0915
    process: Any, root: Path, config: SearchConfig, state: dict[str, Any], wire: dict[str, Any], *, resume: bool
) -> dict[str, Any]:
    process.stdin.write(REMOTE_SCRIPT.encode())
    process.stdin.close()
    base = {key: state[key] for key in ("scanned", "hits", "error_count")}
    header = False
    footer = None
    maximum_lines = 2 * wire["max_scanned"] + 10
    for _ in range(maximum_lines):
        line = process.stdout.readline(transport.MAX_LINE_BYTES + 1)
        if not line:
            break
        if len(line) > transport.MAX_LINE_BYTES or not line.endswith(b"\n"):
            raise RareSearchError("Truncated/oversized remote search line")
        row = json.loads(line)
        kind = row.get("kind")
        if not header:
            expected = {
                "kind": "header",
                "protocol": VERSION,
                "scope_sha256": state["scope_sha256"],
                "field_map_sha256": wire["field_map_sha256"],
                "after_uuid": wire["after_uuid"],
                "max_scanned": wire["max_scanned"],
                "mode_id": config.mode_id,
                "old_source_split": config.source_split,
                "locator_read_mode": "immutable_main_ignores_wal",
            }
            if row != expected:
                raise RareSearchError("Unexpected search scope header")
            header = True
            continue
        if footer is not None:
            raise RareSearchError("Output continues after final search summary")
        if kind == "fatal":
            raise RareSearchError(f"Remote scanner failed: {row.get('reason')}")
        if kind not in {"candidate", "input_error", "progress", "summary"}:
            raise RareSearchError("Unknown remote search entry")
        position = _position(row, base, state, config.max_scanned, kind)
        if kind == "candidate":
            if row.get("uuid") != position["last_scanned_uuid"]:
                raise RareSearchError("Candidate UUID differs from its cursor")
            tags = row.get("candidate_tags")
            if not isinstance(tags, list) or not tags or not set(tags) <= TAGS:
                raise RareSearchError("Unknown/empty candidate tags")
            if (
                row.get("candidate_is_confirmed_positive") is not False
                or row.get("semantic_verification_performed") is not False
            ):
                raise RareSearchError("A lightweight candidate must not claim semantic acceptance")
            if (
                row.get("field_map_sha256") != wire["field_map_sha256"]
                or row.get("search_protocol_sha256") != wire["field_map"]["source_proto_sha256"]
            ):
                raise RareSearchError("Candidate protocol mapping changed")
            row["scan_position"] = position
            transport._write_record(root, state, row, resume=resume)
        elif kind == "input_error":
            _append_error(root, state, row, position)
        elif kind == "summary":
            footer = row
        state.update(position)
        transport._atomic_checkpoint(root, state)
    if process.stdout.read(1):
        raise RareSearchError("Remote scanner exceeded its bounded output")
    result = process.wait(timeout=30)
    if not header or footer is None or result != 0 or footer.get("selection_complete") is not True:
        raise RareSearchError("Interrupted search has no successful final summary")
    if footer.get("all_disk_scanned") is not False or footer.get("semantic_verification_performed") is not False:
        raise RareSearchError("Invalid global-verification claim")
    if not isinstance(footer.get("catalog_exhausted"), bool) or (
        state["scanned"] < config.max_scanned and not footer["catalog_exhausted"]
    ):
        raise RareSearchError("Partial search cannot be marked selection-complete")
    return footer


def search(config: SearchConfig, output_dir: str | Path, *, resume: bool = False) -> dict[str, Any]:
    mapping = freeze_field_map(load_protocol(config.proto))
    scope = config.scope(mapping)
    root = Path(output_dir).resolve()
    if resume:
        state = _resume(root, scope)
    else:
        root.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.jsonl", "errors.jsonl"):
            with (root / name).open("xb"):
                pass
        state = _initial(scope)
        transport._atomic_checkpoint(root, state)
    if state["selection_complete"]:
        return state
    wire = {
        "catalog": scope["catalog"],
        "mode_id": config.mode_id,
        "source_split": config.source_split,
        "max_scanned": config.max_scanned - state["scanned"],
        "after_uuid": state["last_scanned_uuid"],
        "scope_sha256": state["scope_sha256"],
        "field_map_sha256": scope["field_map_sha256"],
        "field_map": mapping,
    }
    state["error_history"].extend(state["errors"])
    state["errors"] = []
    state.update(status="RUNNING", selection_complete=False, scan_complete=False)
    transport._atomic_checkpoint(root, state)
    process = None
    with tempfile.TemporaryFile(mode="w+b") as diagnostics:
        try:
            process = subprocess.Popen(
                transport.ssh_command(config.connection(), wire),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
            )
            footer = _consume(process, root, config, state, wire, resume=resume)
            state.update(
                selection_complete=True,
                scan_complete=state["error_count"] == 0,
                status="COMPLETE" if state["error_count"] == 0 else "COMPLETE_WITH_INPUT_ERRORS",
                remote_summary=footer,
            )
            state["no_candidate_inputs"] = state["scanned"] - state["hits"] - state["error_count"]
            state["no_candidate_is_semantic_negative_proof"] = False
        except BaseException as error:
            state.update(
                selection_complete=False,
                scan_complete=False,
                status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
            )
            state["errors"].append({"error_type": type(error).__name__, "detail": str(error)[:4096]})
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            diagnostics.seek(0)
            state["ssh_diagnostic"] = diagnostics.read(4096).decode(errors="replace")
            transport._atomic_checkpoint(root, state)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is not None and not stream.closed:
                        stream.close()
    transport._atomic_checkpoint(root, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only bounded rare-event candidate search, not Golden acceptance")
    parser.add_argument("--host", required=True)
    parser.add_argument("--ssh-executable", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--proto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scanned", type=int, default=MAX_SCANNED)
    parser.add_argument("--mode-id", type=int, choices=(12, 16), default=12)
    parser.add_argument("--source-split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--after-uuid", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = SearchConfig(
        args.host,
        args.ssh_executable,
        args.remote_python,
        args.catalog,
        args.proto,
        args.max_scanned,
        args.mode_id,
        args.source_split,
        args.after_uuid,
    )
    try:
        result = search(config, args.output, resume=args.resume)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Search failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["scan_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
