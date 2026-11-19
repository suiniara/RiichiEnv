"""Real protoc imports must stay bound across audits and derived artifacts."""

import hashlib
import json

import pytest
from test_mjsoul import _message, _new_round, _result

from riichienv.kar import acceptance, corpus_audit, golden
from riichienv.kar.contracts import content_hash, game_split
from riichienv.kar.learning.export import export_dataset, learning_contract
from riichienv.kar.mjsoul import decode_record_bytes, load_protocol

SALT = "protocol-dependency-regression"
TOP_LEVEL = """syntax = "proto3";
package lq;
import "records.proto";
message Wrapper { string name = 1; bytes data = 2; }
message GameAction { bytes result = 3; }
message GameDetailRecords {
  repeated bytes records = 1;
  uint32 version = 2;
  repeated GameAction actions = 3;
}
message RecordNewRound {
  uint32 chang = 1; uint32 ju = 2; uint32 ben = 3; string dora = 4;
  repeated int32 scores = 5; uint32 liqibang = 6;
  repeated string tiles0 = 7; repeated string tiles1 = 8;
  repeated string tiles2 = 9; repeated string tiles3 = 10;
  string paishan = 14; uint32 left_tile_count = 15;
  repeated string doras = 16;
}
message RecordGame { string uuid = 1; uint32 start_time = 2; uint32 end_time = 3; }
message ResGameRecord { RecordGame head = 3; bytes data = 4; }
message NoTileScoreInfo { repeated int32 old_scores = 2; repeated int32 delta_scores = 3; }
message RecordNoTile { repeated NoTileScoreInfo scores = 3; bool gameend = 4; }
"""
DEPENDENCY = """syntax = "proto3";
package lq;
message RecordDiscardTile {
  uint32 seat = 1; string tile = 2; bool is_liqi = 3;
  bool moqie = 5; repeated string doras = 8; bool is_wliqi = 9;
}
"""


def _save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def multifile(tmp_path):
    schema = tmp_path / "schema"
    schema.mkdir()
    proto, dependency = schema / "protocol.proto", schema / "records.proto"
    proto.write_text(TOP_LEVEL, encoding="utf-8")
    dependency.write_text(DEPENDENCY, encoding="utf-8")
    protocol = load_protocol(proto)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    ids = [f"dependency-{index}" for index in range(100) if game_split(f"dependency-{index}", SALT) == "train"][:2]
    rows = []
    for game_id in ids:
        results = [
            _result(protocol, "RecordNewRound", **_new_round()),
            _result(protocol, "RecordDiscardTile", seat=0, tile="0p", moqie=True),
            _result(
                protocol, "RecordNoTile", gameend=False, scores=[{"old_scores": [25000] * 4, "delta_scores": [0] * 4}]
            ),
        ]
        detail = _message(protocol, "GameDetailRecords", records=results).SerializeToString()
        wrapped = _message(protocol, "Wrapper", name=".lq.GameDetailRecords", data=detail).SerializeToString()
        response = _message(protocol, "ResGameRecord", head={"uuid": game_id}, data=wrapped).SerializeToString()
        payload = b"\0\0\0" + _message(protocol, "Wrapper", data=response).SerializeToString()
        path = raw_dir / f"{game_id}.bin"
        path.write_bytes(payload)
        rows.append({"uuid": game_id, "local_file": path.name, "source_sha256": hashlib.sha256(payload).hexdigest()})
    manifest = raw_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return proto, dependency, protocol, manifest, ids


def _change_dependency(dependency):
    dependency.write_text(DEPENDENCY.replace("bool moqie = 5;", "bool moqie = 6;"), encoding="utf-8")


def _audit(multifile, output, monkeypatch, *, resume=False, interrupt=False):
    proto, _, _, manifest, ids = multifile
    monkeypatch.setattr(corpus_audit, "find_golden_candidates", lambda game: [])

    def replay(game):
        if interrupt and game.game_id == ids[1]:
            raise KeyboardInterrupt
        # Deliberately incomplete synthetic games are never accepted for training.
        return {"status": "BLOCKED", "response_categories_observed": []}

    monkeypatch.setattr(corpus_audit, "replay_game", replay)
    return corpus_audit.audit_corpus(manifest, proto, output, mode="audit", split_salt=SALT, resume=resume)


def test_imported_field_change_alters_decoding_but_not_top_level_source_hash(multifile):
    proto, dependency, before, manifest, ids = multifile
    source = manifest.parent / f"{ids[0]}.bin"
    payload = source.read_bytes()
    first = decode_record_bytes(payload, before, container="majsoul_download")
    _change_dependency(dependency)
    after = load_protocol(proto)
    second = decode_record_bytes(payload, after, container="majsoul_download")
    assert before.source_sha256 == after.source_sha256 == hashlib.sha256(proto.read_bytes()).hexdigest()
    assert before.descriptor_sha256 != after.descriptor_sha256
    assert first.source_sha256 == second.source_sha256
    assert first.records[1].data["moqie"] is True and second.records[1].data["moqie"] is False
    assert next(event for event in first.events if event["type"] == "dahai")["tsumogiri"] is True
    assert next(event for event in second.events if event["type"] == "dahai")["tsumogiri"] is False
    assert first.protocol_sha256 == second.protocol_sha256 == before.source_sha256
    assert first.protocol_descriptor_sha256 == before.descriptor_sha256
    assert second.protocol_descriptor_sha256 == after.descriptor_sha256


@pytest.mark.parametrize("interrupted", [False, True])
def test_audit_resume_rejects_import_change_before_reusing_completed_rows(
    multifile, tmp_path, monkeypatch, interrupted
):
    output = tmp_path / "audit"
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            _audit(multifile, output, monkeypatch, interrupt=True)
    else:
        _audit(multifile, output, monkeypatch)
    original = (output / "checkpoint.json").read_bytes()
    state = json.loads(original)
    assert len(state["rows"]) == (1 if interrupted else 2)
    assert state["scope"]["protocol_descriptor_sha256"] == multifile[2].descriptor_sha256
    _change_dependency(multifile[1])
    with pytest.raises(ValueError, match="Checkpoint scope differs"):
        _audit(multifile, output, monkeypatch, resume=True)
    assert (output / "checkpoint.json").read_bytes() == original


def test_public_merge_rejects_same_top_level_proto_with_different_imports(multifile, tmp_path, monkeypatch):
    before, after = tmp_path / "before", tmp_path / "after"
    first = _audit(multifile, before, monkeypatch)
    _change_dependency(multifile[1])
    second = _audit(multifile, after, monkeypatch)
    assert first["scope"]["protocol_sha256"] == second["scope"]["protocol_sha256"]
    assert first["scope"]["protocol_descriptor_sha256"] != second["scope"]["protocol_descriptor_sha256"]
    assert all(row["status"] == "BLOCKED" for row in first["rows"] + second["rows"])
    with pytest.raises(ValueError, match="incompatible.*protocol"):
        acceptance.consolidate_shards([(before, multifile[3]), (after, multifile[3])], [], tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def _accepted_selection(multifile, directory):
    _, _, protocol, manifest, ids = multifile
    directory.mkdir()
    scope = {**learning_contract(), **protocol.identity_record(), "split_salt": SALT}
    source = manifest.parent / f"{ids[0]}.bin"
    row = {
        "game_id": ids[0],
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        **protocol.identity_record(),
    }
    accepted = {
        "status": "PASS",
        "training_allowed": True,
        "source_scope": scope,
        "golden_game_ids_reserved": [],
        "split_counts": {"train": 1, "validation": 0, "test": 0},
    }
    _save(directory / "audit.json", {**accepted, "acceptance_sha256": content_hash(accepted)})
    for split in ("train", "validation", "test"):
        games = [row] if split == "train" else []
        _save(
            directory / f"{split}.manifest.json",
            {"scope": scope, "games": games, "manifest_sha256": content_hash(games)},
        )


@pytest.mark.parametrize("resume", [False, True])
def test_export_rejects_schema_that_differs_from_accepted_inputs_before_writes(multifile, tmp_path, resume):
    accepted, output = tmp_path / "accepted", tmp_path / "export"
    _accepted_selection(multifile, accepted)
    sentinel = b"previously exported state must not be opened or replaced"
    if resume:
        output.mkdir()
        (output / "decisions.sqlite3").write_bytes(sentinel)
    _change_dependency(multifile[1])
    with pytest.raises(ValueError, match="imported descriptors differ from accepted inputs"):
        export_dataset(accepted, multifile[0], output, resume=resume)
    if resume:
        assert (output / "decisions.sqlite3").read_bytes() == sentinel
    else:
        assert not output.exists()


def test_golden_recompiles_imports_before_replaying_saved_evidence(multifile, monkeypatch):
    proto, dependency, protocol, manifest, ids = multifile
    source = manifest.parent / f"{ids[0]}.bin"
    game = decode_record_bytes(source.read_bytes(), protocol, container="majsoul_download")
    candidate = next(
        row for row in golden.find_golden_candidates(game) if row["case_id"] == "four_observers_same_split"
    )
    evidence = {
        **candidate,
        "source_path": str(source),
        "proto_path": str(proto),
        "source_provenance": "real_raw_record",
        "container": "majsoul_download",
    }
    _change_dependency(dependency)
    monkeypatch.setattr(golden, "decode_record_bytes", lambda *args, **kwargs: pytest.fail("Changed protocol was used"))
    result = golden.verify_golden_evidence(game, {}, evidence)
    assert result["status"] == "FAIL" and result["reason"] == "protocol_descriptor_hash_mismatch"
    # A self-signed saved status still has to pass the same descriptor verification.
    saved = {**evidence, "status": "PASS", "verified": True, "verification_version": golden.GOLDEN_VERSION}
    saved["evidence_sha256"] = content_hash(saved)
    coverage = golden.golden_coverage([saved])
    assert coverage["verified_count"] == 0
    assert coverage["failed"][0]["reason"] == "protocol_descriptor_hash_mismatch"


@pytest.mark.parametrize("location", ["scope", "row", "receipt"])
def test_legacy_audit_without_dependency_identity_is_rejected(multifile, tmp_path, monkeypatch, location):
    output = tmp_path / "audit"
    _audit(multifile, output, monkeypatch)
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    report = json.loads((output / "report.json").read_text())
    if location == "scope":
        target = checkpoint["scope"]
    elif location == "row":
        target = checkpoint["rows"][0]
    else:
        path = output / "receipts" / f"{multifile[4][0]}.json"
        target = json.loads(path.read_text())
    target.pop("protocol_descriptor_sha256")
    target.pop("protocol_identity_version")
    if location == "receipt":
        _save(path, target)
        checkpoint["rows"][0]["receipt_sha256"] = content_hash(target)
    checkpoint.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = content_hash(checkpoint)
    report.update(scope=checkpoint["scope"], rows=checkpoint["rows"])
    _save(output / "checkpoint.json", checkpoint)
    _save(output / "report.json", report)
    with pytest.raises(ValueError, match="complete source/descriptor protocol identity"):
        acceptance.consolidate(output, multifile[3], [], tmp_path / "rejected")
