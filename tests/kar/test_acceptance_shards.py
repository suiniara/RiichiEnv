import hashlib
import json

import pytest
from test_acceptance import _fixture, _save

from riichienv.kar import acceptance
from riichienv.kar.actions import manifest_hash
from riichienv.kar.contracts import MJSOUL_RULESET, cohort_hash, content_hash, game_split
from riichienv.kar.corpus_audit import SEMANTIC_GAME_VERSION, normalized_game_sha256
from riichienv.kar.mjsoul import protocol_identity
from riichienv.kar.schema import schema_status
from riichienv.kar.versioning import artifact_contract


def _upgrade_fixture(corpus):
    state = json.loads((corpus / "checkpoint.json").read_text())
    report = json.loads((corpus / "report.json").read_text())
    state["scope"]["normalized_game_digest_version"] = SEMANTIC_GAME_VERSION
    for row in state["rows"]:
        semantic = hashlib.sha256(("synthetic-semantic/" + row["game_id"]).encode()).hexdigest()
        row["normalized_game_sha256"] = semantic
        path = corpus / "receipts" / f"{row['game_id']}.json"
        receipt = json.loads(path.read_text())
        receipt["normalized_game_sha256"] = semantic
        receipt["replay"]["response_categories_observed"] = row["response_categories"]
        _save(path, receipt)
        row["receipt_sha256"] = content_hash(receipt)
    state.pop("checkpoint_sha256")
    state["checkpoint_sha256"] = content_hash(state)
    report["scope"] = state["scope"]
    report["rows"] = state["rows"]
    _save(corpus / "checkpoint.json", state)
    _save(corpus / "report.json", report)


def test_public_shard_consolidation_deduplicates_identical_games_and_preserves_sources(tmp_path, monkeypatch):
    shards = []
    for index in range(2):
        directory = tmp_path / f"part-{index}"
        directory.mkdir()
        corpus, manifest, golden, ids = _fixture(directory, monkeypatch)
        _upgrade_fixture(corpus)
        shards.append((corpus, manifest))
    report = acceptance.consolidate_shards(shards, [], tmp_path / "accepted")
    assert report["status"] == "PASS" and report["selected_inputs"] == 6 and report["unique_input_game_ids"] == 3
    assert report["N_train_half_games"] == 2 and len(report["producer_shards"]) == 2
    assert report["source_scope"]["mode"] == "merged_audit_shards_v1"
    document = json.loads((tmp_path / "accepted/train.manifest.json").read_text())
    assert document["manifest_sha256"] == content_hash(document["games"])
    assert len(document["games"][0]["producer_receipts"]) == 2


def test_shards_reject_unverified_report_fields_and_incompatible_implementations(tmp_path, monkeypatch):
    shards = []
    for index in range(2):
        directory = tmp_path / f"part-{index}"
        directory.mkdir()
        corpus, manifest, _, _ = _fixture(directory, monkeypatch)
        _upgrade_fixture(corpus)
        shards.append((corpus, manifest))
    target = shards[1][0] / "report.json"
    document = json.loads(target.read_text())
    document["rows"][0]["response_categories"] = [8]
    _save(target, document)
    with pytest.raises(ValueError, match="authenticated checkpoint"):
        acceptance.consolidate_shards(shards, [], tmp_path / "bad")


def _trusted_shards(tmp_path, count, *, semantic_alias=False):
    # Aggregation unit fixture only. Public consolidation tests above exercise the
    # raw/checkpoint/receipt validator; these rows stand for its authenticated output.
    salt = "large-shard-unit"
    ids = []
    index = 0
    while len(ids) < count:
        identity = f"synthetic-large-{index}"
        if game_split(identity, salt) == "train":
            ids.append(identity)
        index += 1
    common = {
        **artifact_contract(),
        "mode": "audit",
        "schema_hash": schema_status()["schema_hash"],
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "cohort_hash": cohort_hash(),
        "action_manifest_hash": manifest_hash(),
        **protocol_identity("a" * 64, "b" * 64),
        "split_salt": salt,
        "implementation_sha256": "b" * 64,
        "normalized_game_digest_version": SEMANTIC_GAME_VERSION,
        "limit": 10000,
        "input_manifest_sha256": "c" * 64,
    }
    metadata = {"category": 2, "mode": 2, "mode_id": 12, "account_seats": [0, 1, 2, 3]}
    shards = []
    for start in range(0, count, 10000):
        number = start // 10000
        directory = tmp_path / f"producer-{number}"
        sources = {}
        rows = []
        for identity in ids[start : start + 10000]:
            raw = hashlib.sha256(identity.encode()).hexdigest()
            sources[identity] = {
                "game_id": identity,
                "source": str(tmp_path / "raw" / f"{identity}.bin"),
                "source_sha256": raw,
                "source_role": "main_cohort_candidate",
            }
            rows.append(
                {
                    **sources[identity],
                    "status": "PASS",
                    **protocol_identity("a" * 64, "b" * 64),
                    "metadata": metadata,
                    "response_categories": [0],
                    "receipt_sha256": "d" * 64,
                    "normalized_game_sha256": hashlib.sha256(("semantic/" + identity).encode()).hexdigest(),
                }
            )
        scope = {**common, "limit": len(rows)}
        shards.append(
            acceptance._Shard(
                directory,
                tmp_path / f"input-{number}.jsonl",
                scope,
                tuple(rows),
                sources,
                {"status": "PASS"},
                "e" * 64,
                "f" * 64,
            )
        )
    return shards


@pytest.mark.parametrize("count,tier", [(30000, "medium"), (50000, "large")])
def test_merged_producer_has_no_single_shard_model_tier_ceiling(tmp_path, monkeypatch, count, tier):
    monkeypatch.setattr(acceptance, "golden_coverage", lambda evidence: {"status": "PASS", "golden_game_ids": []})
    shards = _trusted_shards(tmp_path, count)
    assert all(len(shard.rows) <= 10000 for shard in shards)
    report = acceptance._finalize(shards, [], tmp_path / "accepted", merged=True)
    assert report["N_train_half_games"] == count and report["provisional_model_tier"] == tier
    assert report["response_support_independent_games"]["0"] == count


def test_semantic_alias_of_any_reserved_golden_is_also_excluded(tmp_path, monkeypatch):
    shard = _trusted_shards(tmp_path, 2)
    rows = list(shard[0].rows)
    rows[1]["normalized_game_sha256"] = rows[0]["normalized_game_sha256"]
    held = rows[1]["game_id"]
    monkeypatch.setattr(acceptance, "golden_coverage", lambda evidence: {"status": "PASS", "golden_game_ids": [held]})
    report = acceptance._finalize(shard, [], tmp_path / "accepted", merged=True)
    assert report["N_train_half_games"] == 0 and set(report["golden_game_ids_reserved"]) == {
        row["game_id"] for row in rows
    }


def test_semantic_game_digest_ignores_only_identity_and_source_coordinates():
    first = [
        {"type": "start_game", "id": "alias-A"},
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False, "source_record_index": 3},
    ]
    second = [
        {"type": "start_game", "id": "alias-B"},
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False, "source_record_index": 300},
    ]
    assert normalized_game_sha256(first) == normalized_game_sha256(second)
    second[1]["pai"] = "5m"
    assert normalized_game_sha256(first) != normalized_game_sha256(second)
