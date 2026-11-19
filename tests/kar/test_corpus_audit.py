import hashlib
import json
from types import SimpleNamespace

import pytest

from riichienv.kar import corpus_audit as corpus
from riichienv.kar.contracts import content_hash, game_split
from riichienv.kar.mjsoul import Protocol

SALT = "corpus-unit-split"


def _setup(tmp_path, monkeypatch, *, statuses=None, golden=False, room=12):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    ids = [f"fixture-{index}" for index in range(100) if game_split(f"fixture-{index}", SALT) == "train"][:3]
    rows = []
    for game_id in ids:
        payload = game_id.encode()
        (raw_root / f"{game_id}.bin").write_bytes(payload)
        rows.append(
            {"uuid": game_id, "local_file": f"{game_id}.bin", "source_sha256": hashlib.sha256(payload).hexdigest()}
        )
    manifest = raw_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    proto = tmp_path / "fixture.proto"
    proto.write_text("synthetic protocol fixture", encoding="utf-8")
    monkeypatch.setattr(corpus, "load_protocol", lambda _: Protocol("a" * 64, "b" * 64, "synthetic_fixture", None))

    def decode(payload, *_args, **_kwargs):
        game_id = payload.decode()
        # The producer hashes normalized events before invoking the mocked replay.
        # Keep distinct semantic facts, not only distinct IDs (IDs are excluded).
        events = (
            {"type": "start_game", "id": game_id},
            {"type": "tsumo", "actor": 0, "pai": ("1m", "2m", "3m")[ids.index(game_id)]},
        )
        return SimpleNamespace(
            game_id=game_id,
            events=events,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            protocol_sha256="a" * 64,
            protocol_descriptor_sha256="b" * 64,
            metadata={"category": 2, "mode": 2, "mode_id": room, "account_seats": [0, 1, 2, 3]},
        )

    def replay(game):
        return {
            "status": (statuses or {}).get(game.game_id, "PASS"),
            "completion": {"status": "PASS"},
            "input_encoding": {"status": "PASS"},
            "response_categories_observed": [0],
        }

    monkeypatch.setattr(corpus, "decode_record_bytes", decode)
    monkeypatch.setattr(corpus, "replay_game", replay)
    monkeypatch.setattr(
        corpus,
        "find_golden_candidates",
        lambda game: [{"case_id": "pon_then_discard"}] if golden and game.game_id == ids[0] else [],
    )
    monkeypatch.setattr(
        corpus,
        "verify_golden_evidence",
        lambda game, _replay, evidence: {"status": "PASS", "case_id": evidence["case_id"], "game_id": game.game_id},
    )
    monkeypatch.setattr(corpus, "golden_coverage", lambda _: {"status": "PASS"})
    return manifest, proto, ids, replay


def _audit(manifest, proto, output, **kwargs):
    return corpus.audit_corpus(manifest, proto, output, mode="audit", split_salt=SALT, **kwargs)


def test_unique_games_are_split_and_golden_games_are_reserved(tmp_path, monkeypatch):
    manifest, proto, ids, _ = _setup(tmp_path, monkeypatch, golden=True)
    report = _audit(manifest, proto, tmp_path / "audit")
    assert report["status"] == "PASS"
    assert report["N_eligible_half_games"] == report["N_train_half_games"] == 2
    assert report["golden_game_ids_reserved"] == [ids[0]]
    assert report["response_support_independent_games"]["0"] == 2
    assert report["selection_complete"] is True and report["all_disk_scanned"] is False
    assert len({row["normalized_game_sha256"] for row in report["rows"]}) == len(ids)
    for row in report["rows"]:
        receipt = json.loads((tmp_path / "audit" / "receipts" / f"{row['game_id']}.json").read_text(encoding="utf-8"))
        assert receipt["normalized_game_sha256"] == row["normalized_game_sha256"]
    train = json.loads((tmp_path / "audit" / "train.manifest.json").read_text(encoding="utf-8"))
    assert {game["game_id"] for game in train["games"]} == set(ids[1:])


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED"])
def test_a_bad_or_unresolved_game_cannot_be_filtered_into_pass(tmp_path, monkeypatch, status):
    manifest, proto, ids, replay = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        corpus,
        "replay_game",
        lambda game: {**replay(game), "status": status} if game.game_id == ids[1] else replay(game),
    )
    report = _audit(manifest, proto, tmp_path / "audit")
    assert report["status"] == status
    assert report["training_allowed"] is False
    assert report["N_eligible_half_games"] == 2


def test_other_room_is_not_a_training_cohort(tmp_path, monkeypatch):
    manifest, proto, _, _ = _setup(tmp_path, monkeypatch, room=16)
    report = _audit(manifest, proto, tmp_path / "audit")
    assert report["status"] == "BLOCKED"
    assert report["N_eligible_half_games"] == 0
    assert all(row["cohort_matches"] is False for row in report["rows"])


def test_explicit_golden_only_sources_never_become_training_data(tmp_path, monkeypatch):
    manifest, proto, _, _ = _setup(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["selection_role"] = "golden_only"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = _audit(manifest, proto, tmp_path / "audit")
    assert report["N_eligible_half_games"] == 0
    assert report["training_allowed"] is False
    assert all(row["source_role"] == "golden_only" and row["training_eligible"] is False for row in report["rows"])


def test_complete_resume_is_identical_without_replaying(tmp_path, monkeypatch):
    manifest, proto, _, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "audit"
    first = _audit(manifest, proto, output)
    monkeypatch.setattr(corpus, "replay_game", lambda _: pytest.fail("Completed game was replayed"))
    second = _audit(manifest, proto, output, resume=True)
    assert content_hash(first) == content_hash(second)


def test_interrupted_resume_matches_clean_run(tmp_path, monkeypatch):
    manifest, proto, ids, replay = _setup(tmp_path, monkeypatch)

    def interrupt(game):
        if game.game_id == ids[1]:
            raise KeyboardInterrupt
        return replay(game)

    monkeypatch.setattr(corpus, "replay_game", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _audit(manifest, proto, tmp_path / "resumed")
    checkpoint = json.loads((tmp_path / "resumed" / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["rows"]) == 1 and checkpoint["selection_complete"] is False
    monkeypatch.setattr(corpus, "replay_game", replay)
    resumed = _audit(manifest, proto, tmp_path / "resumed", resume=True)
    clean = _audit(manifest, proto, tmp_path / "clean")
    assert content_hash(resumed) == content_hash(clean)


@pytest.mark.parametrize("changed", ["source", "receipt", "checkpoint"])
def test_resume_rejects_changed_evidence(tmp_path, monkeypatch, changed):
    manifest, proto, ids, _ = _setup(tmp_path, monkeypatch)
    output = tmp_path / "audit"
    _audit(manifest, proto, output)
    if changed == "source":
        target = manifest.parent / f"{ids[0]}.bin"
    elif changed == "receipt":
        target = output / "receipts" / f"{ids[0]}.json"
    else:
        target = output / "checkpoint.json"
    if changed == "checkpoint":
        state = json.loads(target.read_text(encoding="utf-8"))
        state["rows"][0]["status"] = "BLOCKED"
        target.write_text(json.dumps(state), encoding="utf-8")
    else:
        target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed|hash mismatch"):
        _audit(manifest, proto, output, resume=True)


def test_duplicate_views_do_not_increase_game_counts(tmp_path, monkeypatch):
    manifest, _, _, _ = _setup(tmp_path, monkeypatch)
    original = manifest.read_text(encoding="utf-8")
    manifest.write_text(original + original, encoding="utf-8")
    assert len(corpus._inputs(manifest)) == 3
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["source_sha256"] = "f" * 64
    with manifest.open("a", encoding="utf-8") as output:
        output.write(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="Conflicting"):
        corpus._inputs(manifest)


@pytest.mark.parametrize(
    "key,value", [("local_file", "../escape.bin"), ("uuid", "../escape"), ("uuid", "CON"), ("source_sha256", "z" * 64)]
)
def test_manifest_paths_ids_and_hashes_are_checked(tmp_path, monkeypatch, key, value):
    manifest, _, _, _ = _setup(tmp_path, monkeypatch)
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    row[key] = value
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        corpus._inputs(manifest)


def test_outputs_never_write_inside_the_raw_input_directory(tmp_path, monkeypatch):
    manifest, proto, _, _ = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside raw"):
        _audit(manifest, proto, manifest.parent / "outputs")


def test_selected_golden_manifest_references_unchanged_raw_root(tmp_path, monkeypatch):
    manifest, _, ids, _ = _setup(tmp_path, monkeypatch)
    discovery = tmp_path / "discovery.json"
    discovery.write_text(
        json.dumps(
            {
                "rows": [
                    {"game_id": ids[0], "candidate_case_ids": ["x", "y"]},
                    {"game_id": ids[1], "candidate_case_ids": ["x"]},
                    {"game_id": ids[2], "candidate_case_ids": ["z"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "golden-selected.jsonl"
    assert corpus.select_golden_inputs(discovery, manifest, output) == sorted([ids[0], ids[2]])
    assert {row["game_id"] for row in corpus._inputs(output)} == {ids[0], ids[2]}
    with pytest.raises(ValueError, match="outside"):
        corpus.select_golden_inputs(discovery, manifest, manifest.parent / "selected.jsonl")


@pytest.mark.parametrize("limit", [0, -1, 10001, True, 1.0])
def test_batch_size_is_explicitly_bounded(tmp_path, limit):
    with pytest.raises(ValueError, match="bounded"):
        _audit(tmp_path / "unused", tmp_path / "unused.proto", tmp_path / "out", limit=limit)
