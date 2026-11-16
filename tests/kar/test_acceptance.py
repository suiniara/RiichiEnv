import hashlib
import json

import pytest

from riichienv.kar import acceptance
from riichienv.kar.actions import manifest_hash
from riichienv.kar.contracts import MJSOUL_RULESET, cohort_hash, content_hash, game_split
from riichienv.kar.schema import schema_status


def _save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path, monkeypatch, status="PASS"):
    raw_root, corpus = tmp_path / "raw", tmp_path / "producer"
    raw_root.mkdir()
    corpus.mkdir()
    (corpus / "receipts").mkdir()
    salt = "acceptance-unit"
    ids = [f"fixture-{index}" for index in range(100) if game_split(f"fixture-{index}", salt) == "train"][:3]
    inputs, rows = [], []
    for identity in ids:
        raw = identity.encode()
        source = raw_root / f"{identity}.bin"
        source.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        inputs.append({"uuid": identity, "local_file": source.name, "source_sha256": digest})
        replay = {
            "status": status,
            "errors": [],
            "events_applied": 100,
            "events_total": 100,
            "discard_decisions_checked": 40,
            "complete_response_windows": 40,
            **{name: {"status": "PASS"} for name in ("input_encoding", "completion", "independent_settlement")},
        }
        receipt = {"game_id": identity, "source_sha256": digest, "replay": replay}
        _save(corpus / "receipts" / f"{identity}.json", receipt)
        rows.append(
            {
                "game_id": identity,
                "source": str(source),
                "source_sha256": digest,
                "receipt_sha256": content_hash(receipt),
                "status": status,
                "protocol_sha256": "a" * 64,
                "metadata": {"category": 2, "mode": 2, "mode_id": 12, "account_seats": [0, 1, 2, 3]},
                "response_categories": [0],
            }
        )
    manifest = raw_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in inputs), encoding="utf-8")
    scope = {
        "mode": "audit",
        "schema_hash": schema_status()["schema_hash"],
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "cohort_hash": cohort_hash(),
        "action_manifest_hash": manifest_hash(),
        "protocol_sha256": "a" * 64,
        "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "limit": 3,
        "split_salt": salt,
        "implementation_sha256": "f" * 64,
    }
    state = {"scope": scope, "rows": rows, "selection_complete": True}
    state["checkpoint_sha256"] = content_hash(state)
    _save(corpus / "checkpoint.json", state)
    _save(
        corpus / "report.json",
        {
            "status": "BLOCKED",
            "scope": scope,
            "rows": rows,
            "selection_complete": True,
            "golden_game_ids_reserved": [ids[0]],
        },
    )
    golden = tmp_path / "golden.json"
    _save(golden, {"evidence": []})
    monkeypatch.setattr(acceptance, "golden_coverage", lambda _: {"status": "PASS", "golden_game_ids": []})
    return corpus, manifest, golden, ids


def test_new_verified_golden_can_complete_previously_blocked_scoped_report(tmp_path, monkeypatch):
    corpus, manifest, golden, ids = _fixture(tmp_path, monkeypatch)
    original = (corpus / "report.json").read_bytes()
    report = acceptance.consolidate(corpus, manifest, [golden], tmp_path / "accepted")
    assert report["status"] == "PASS" and report["training_allowed"] is True
    assert report["N_eligible_half_games"] == report["N_train_half_games"] == 2
    assert report["response_support_independent_games"]["0"] == 2
    assert report["golden_game_ids_reserved"] == [ids[0]]
    assert report["producer_report_status_preserved"] == "BLOCKED"
    assert report["native_replay_receipts_reexecuted"] is False
    assert (corpus / "report.json").read_bytes() == original
    payload = {key: value for key, value in report.items() if key != "acceptance_sha256"}
    assert report["acceptance_sha256"] == content_hash(payload)


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED"])
def test_additional_golden_never_excuses_a_bad_or_unresolved_main_game(tmp_path, monkeypatch, status):
    corpus, manifest, golden, _ = _fixture(tmp_path, monkeypatch, status=status)
    report = acceptance.consolidate(corpus, manifest, [golden], tmp_path / "accepted")
    assert report["status"] == status and report["training_allowed"] is False
    assert report["N_train_half_games"] == 0


@pytest.mark.parametrize("changed", ["checkpoint", "receipt", "raw", "manifest", "contract", "incomplete"])
def test_changed_or_incomplete_producer_evidence_is_rejected(tmp_path, monkeypatch, changed):
    corpus, manifest, golden, ids = _fixture(tmp_path, monkeypatch)
    if changed == "checkpoint":
        target = corpus / "checkpoint.json"
        state = json.loads(target.read_text(encoding="utf-8"))
        state["rows"][0]["status"] = "FAIL"
        _save(target, state)
    elif changed == "receipt":
        _save(corpus / "receipts" / f"{ids[0]}.json", {})
    elif changed == "raw":
        (manifest.parent / f"{ids[0]}.bin").write_bytes(b"changed source")
    elif changed == "manifest":
        with manifest.open("a", encoding="utf-8") as output:
            output.write("\n")
    elif changed in {"contract", "incomplete"}:
        state = json.loads((corpus / "checkpoint.json").read_text(encoding="utf-8"))
        original = json.loads((corpus / "report.json").read_text(encoding="utf-8"))
        if changed == "contract":
            state["scope"]["schema_hash"] = original["scope"]["schema_hash"] = "0" * 64
        else:
            state["selection_complete"] = original["selection_complete"] = False
        state.pop("checkpoint_sha256")
        state["checkpoint_sha256"] = content_hash(state)
        _save(corpus / "checkpoint.json", state)
        _save(corpus / "report.json", original)
    with pytest.raises(ValueError):
        acceptance.consolidate(corpus, manifest, [golden], tmp_path / "accepted")


def test_incomplete_real_golden_still_blocks_acceptance(tmp_path, monkeypatch):
    corpus, manifest, golden, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(acceptance, "golden_coverage", lambda _: {"status": "BLOCKED", "golden_game_ids": []})
    report = acceptance.consolidate(corpus, manifest, [golden], tmp_path / "accepted")
    assert report["status"] == "BLOCKED" and report["training_allowed"] is False


def test_outputs_cannot_modify_producer_directory(tmp_path, monkeypatch):
    corpus, manifest, golden, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="separate"):
        acceptance.consolidate(corpus, manifest, [golden], corpus / "nested-output")
