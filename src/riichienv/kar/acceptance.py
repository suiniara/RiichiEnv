from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import manifest_hash
from .audit import response_support_evidence
from .contracts import (
    MJSOUL_RULESET,
    PRIMARY_COHORT,
    SCHEMA_ID,
    cohort_hash,
    content_hash,
    game_split,
    matches_primary_cohort,
    model_tier,
)
from .corpus_audit import SEMANTIC_GAME_VERSION, _inputs, _write
from .golden import golden_coverage
from .schema import schema_status


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _Shard:
    directory: Path
    input_manifest: Path
    scope: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    sources: dict[str, dict[str, Any]]
    original: dict[str, Any]
    checkpoint_sha256: str
    report_sha256: str

    def descriptor(self) -> dict[str, Any]:
        return {
            "corpus": str(self.directory.resolve()),
            "input_manifest": str(self.input_manifest.resolve()),
            "input_manifest_sha256": self.scope["input_manifest_sha256"],
            "checkpoint_sha256": self.checkpoint_sha256,
            "report_sha256": self.report_sha256,
            "producer_scope": self.scope,
            "producer_status": self.original["status"],
        }


def _load_shard(directory: Path, input_manifest: Path) -> _Shard:
    original = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint_sha = content_hash({key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"})
    if checkpoint.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Completed corpus checkpoint hash mismatch")
    scope = checkpoint["scope"]
    if original.get("scope") != scope or scope.get("mode") != "audit":
        raise ValueError("Expected a matching full-audit report and checkpoint")
    if not checkpoint.get("selection_complete") or not original.get("selection_complete"):
        raise ValueError("Corpus input scope is not completely processed")
    registry = schema_status()
    expected = {
        "schema_hash": registry.get("schema_hash"),
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "cohort_hash": cohort_hash(),
        "action_manifest_hash": manifest_hash(),
    }
    if registry["status"] != "PASS" or any(scope.get(key) != value for key, value in expected.items()):
        raise ValueError("Corpus contracts differ from active frozen contracts")
    if _file_sha(input_manifest) != scope["input_manifest_sha256"]:
        raise ValueError("Corpus input manifest changed")
    sources = _inputs(input_manifest)[: scope["limit"]]
    source_by_id = {source["game_id"]: source for source in sources}
    reported, stored = original["rows"], checkpoint["rows"]
    prior = {row["game_id"]: row for row in stored}
    if (
        len(reported) != len(sources)
        or len({row["game_id"] for row in reported}) != len(reported)
        or len(prior) != len(stored)
        or set(prior) != set(source_by_id)
    ):
        raise ValueError("Each shard must cover every selected unique game exactly once")
    for report_row in reported:
        identity = report_row["game_id"]
        row, source = prior[identity], source_by_id[identity]
        # Only final derived eligibility differs legitimately from the checkpoint.
        if {key: value for key, value in report_row.items() if key != "training_eligible"} != {
            key: value for key, value in row.items() if key != "training_eligible"
        }:
            raise ValueError("Report material evidence conflicts with its authenticated checkpoint")
        if row["source_sha256"] != source["source_sha256"]:
            raise ValueError("Shard source hash differs from its input selection")
        if row.get("protocol_sha256", scope["protocol_sha256"]) != scope["protocol_sha256"]:
            raise ValueError("Receipt protocol differs from its corpus scope")
        raw = Path(source["source"])
        if raw.stat().st_size > 64 * 1024 * 1024 or _file_sha(raw) != source["source_sha256"]:
            raise ValueError("An audited raw source changed")
        if row.get("receipt_sha256"):
            receipt = json.loads((directory / "receipts" / f"{identity}.json").read_text(encoding="utf-8"))
            if (
                content_hash(receipt) != row["receipt_sha256"]
                or receipt["game_id"] != identity
                or receipt["source_sha256"] != source["source_sha256"]
                or receipt["replay"]["status"] != row["status"]
            ):
                raise ValueError("Native receipt identity/status/hash mismatch")
            replay = receipt["replay"]
            if row["status"] == "PASS" and (
                replay.get("errors") != []
                or replay.get("events_applied") != replay.get("events_total")
                or replay.get("discard_decisions_checked") != replay.get("complete_response_windows")
                or any(
                    replay.get(component, {}).get("status") != "PASS"
                    for component in ("input_encoding", "completion", "independent_settlement")
                )
            ):
                raise ValueError("A PASS receipt lacks required full-replay components")
            if (
                "response_categories_observed" in replay
                and row.get("response_categories") != replay["response_categories_observed"]
            ):
                raise ValueError("Response support does not match the authenticated replay")
            semantic = row.get("normalized_game_sha256")
            if semantic is not None and (receipt.get("normalized_game_sha256") != semantic or len(semantic) != 64):
                raise ValueError("Normalized-game digest differs from its source-bound receipt")
        elif row["status"] != "FAIL":
            raise ValueError("Native replay receipt missing")
        if any(type(event) is not int or not 0 <= event < 21 for event in row.get("response_categories", [])):
            raise ValueError("Invalid response category evidence")
    return _Shard(
        directory,
        input_manifest,
        scope,
        tuple(dict(row) for row in stored),
        source_by_id,
        original,
        checkpoint_sha,
        _file_sha(directory / "report.json"),
    )


def _golden_inputs(
    shards: Sequence[_Shard], manifests: Sequence[Path]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    paths = [
        *manifests,
        *(
            shard.directory / "golden.manifest.json"
            for shard in shards
            if (shard.directory / "golden.manifest.json").is_file()
        ),
    ]
    evidence, descriptors = [], []
    seen: set[Path] = set()
    for supplied in paths:
        path = supplied.resolve()
        if path in seen:
            continue
        seen.add(path)
        payload = path.read_bytes()
        if len(payload) > 16 * 1024 * 1024:
            raise ValueError("Golden evidence manifest exceeds bounded size")
        evidence.extend(json.loads(payload)["evidence"])
        descriptors.append({"source": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
    return evidence, descriptors


def _finalize(  # noqa: PLR0915
    shards: Sequence[_Shard], golden_manifests: Sequence[Path], output_dir: Path, *, merged: bool
) -> dict[str, Any]:
    if not shards:
        raise ValueError("At least one completely audited shard is required")
    first = shards[0].scope
    shared = (
        "schema_hash",
        "ruleset_hash",
        "cohort_hash",
        "action_manifest_hash",
        "protocol_sha256",
        "split_salt",
        "implementation_sha256",
    )
    if any(any(shard.scope.get(key) != first.get(key) for key in shared) for shard in shards):
        raise ValueError("Audit shards have incompatible contracts, protocol, split salt or implementation")
    if merged and any(shard.scope.get("normalized_game_digest_version") != SEMANTIC_GAME_VERSION for shard in shards):
        raise ValueError(
            "Multi-shard aggregation needs source-bound semantic digests; re-audit legacy shards explicitly"
        )
    rows: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    lineage: dict[str, list[dict[str, str]]] = {}
    raw_ids: dict[str, str] = {}
    for shard in shards:
        for row in shard.rows:
            identity = row["game_id"]
            old = rows.get(identity)
            if old is not None and any(
                old.get(key) != row.get(key)
                for key in (
                    "source_sha256",
                    "status",
                    "protocol_sha256",
                    "metadata",
                    "response_categories",
                    "normalized_game_sha256",
                )
            ):
                raise ValueError("Conflicting evidence for the same game identity across shards")
            digest = row["source_sha256"]
            if digest in raw_ids and raw_ids[digest] != identity:
                raise ValueError("Identical raw bytes have conflicting game identities")
            raw_ids[digest] = identity
            source = shard.sources[identity]
            rows.setdefault(identity, row)
            sources.setdefault(identity, source)
            if source["source_role"] == "golden_only":
                sources[identity] = source
            lineage.setdefault(identity, []).append(
                {
                    "corpus": str(shard.directory.resolve()),
                    "checkpoint_sha256": shard.checkpoint_sha256,
                    "receipt_sha256": row.get("receipt_sha256", ""),
                    "receipt": str((shard.directory / "receipts" / f"{identity}.json").resolve()),
                }
            )
    evidence, evidence_inputs = _golden_inputs(shards, golden_manifests)
    golden = golden_coverage(evidence)
    reserved = set(golden["golden_game_ids"])
    for shard in shards:
        reserved.update(shard.original.get("golden_game_ids_reserved", []))
    reserved.update(item["game_id"] for item in evidence if item.get("source_role") == "golden_only")
    reserved.update(identity for identity, source in sources.items() if source["source_role"] == "golden_only")
    groups: dict[str, list[str]] = {}
    for identity, row in rows.items():
        semantic = row.get("normalized_game_sha256")
        if merged and row["status"] == "PASS" and (not isinstance(semantic, str) or len(semantic) != 64):
            raise ValueError("A successful shard row lacks its normalized semantic digest")
        key = "semantic/" + semantic if semantic is not None else "raw/" + row["source_sha256"]
        groups.setdefault(key, []).append(identity)
    representative: dict[str, str] = {}
    for aliases in groups.values():
        canonical = min(aliases)
        representative.update({identity: canonical for identity in aliases})
        if any(identity in reserved for identity in aliases):
            reserved.update(aliases)
    descriptors = [shard.descriptor() for shard in shards]
    scope = (
        first
        if not merged
        else {
            **{key: first[key] for key in shared},
            "mode": "merged_audit_shards_v1",
            "normalized_game_digest_version": SEMANTIC_GAME_VERSION,
            "shards_sha256": content_hash(descriptors),
        }
    )
    split_games: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    support: dict[int, set[str]] = {event: set() for event in range(21)}
    selected = []
    salt = scope["split_salt"]
    for identity, row in sorted(rows.items()):
        eligible = (
            row["status"] == "PASS"
            and matches_primary_cohort(row.get("metadata", {}))
            and identity not in reserved
            and representative[identity] == identity
        )
        selected.append(
            {
                "game_id": identity,
                "source_sha256": row["source_sha256"],
                "status": row["status"],
                "training_eligible": eligible,
                "deduplicated_to": representative[identity],
            }
        )
        if eligible:
            source = {
                **sources[identity],
                "protocol_sha256": row["protocol_sha256"],
                "receipt_sha256": row["receipt_sha256"],
                "producer_receipts": lineage[identity],
            }
            if row.get("normalized_game_sha256") is not None:
                source["normalized_game_sha256"] = row["normalized_game_sha256"]
            split_games[game_split(identity, salt)].append(source)
            for event in row["response_categories"]:
                support[event].add(identity)
    replay_status = (
        "FAIL"
        if any(row["status"] == "FAIL" for row in rows.values())
        else "BLOCKED"
        if not rows or any(row["status"] != "PASS" for row in rows.values())
        else "PASS"
    )
    n_train = len(split_games["train"])
    checks = [
        {"name": "frozen_contracts", "status": "PASS", **{key: first[key] for key in shared}},
        {"name": "unchanged_source_bound_receipts", "status": "PASS", "games": len(rows)},
        {"name": "all_selected_game_replays", "status": replay_status},
        {"name": "real_golden_coverage", "status": golden["status"]},
        {"name": "eligible_training_games", "status": "PASS" if n_train else "BLOCKED"},
    ]
    status = (
        "FAIL"
        if any(check["status"] == "FAIL" for check in checks)
        else "BLOCKED"
        if any(check["status"] == "BLOCKED" for check in checks)
        else "PASS"
    )
    output_dir = output_dir.resolve()
    if any(output_dir.is_relative_to(shard.directory.resolve()) for shard in shards) or any(
        output_dir.is_relative_to(Path(source["source"]).parent) for source in sources.values()
    ):
        raise ValueError("Acceptance output must remain separate from raw and producer reports")
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, games in split_games.items():
        _write(
            output_dir / f"{split}.manifest.json",
            {"scope": scope, "games": games, "manifest_sha256": content_hash(games)},
        )
    _write(output_dir / "golden.manifest.json", {"evidence": evidence, "source_manifests": evidence_inputs})
    result = {
        "status": status,
        "training_allowed": status == "PASS",
        "schema_id": SCHEMA_ID,
        "schema_hash": scope["schema_hash"],
        "cohort": PRIMARY_COHORT,
        "checks": checks,
        "golden": golden,
        "source_scope": scope,
        "producer_shards": descriptors,
        "native_replay_receipts_reexecuted": False,
        "golden_native_replays_reexecuted": True,
        "golden_game_ids_reserved": sorted(reserved),
        "golden_evidence_inputs": evidence_inputs,
        "selection_complete": True,
        "all_disk_scanned": False,
        "selected_inputs": sum(len(shard.sources) for shard in shards),
        "unique_input_game_ids": len(rows),
        "semantic_duplicate_aliases": {
            identity: canonical for identity, canonical in representative.items() if identity != canonical
        },
        "N_eligible_half_games": sum(map(len, split_games.values())),
        "N_train_half_games": n_train,
        "model_tier_basis": "N_train_half_games",
        "provisional_model_tier": model_tier(n_train),
        "split_counts": {split: len(games) for split, games in split_games.items()},
        **response_support_evidence(support, salt),
        "games": selected,
        "cloud_resources_started": False,
    }
    if not merged:
        result.update(
            producer_report_sha256=shards[0].report_sha256,
            producer_checkpoint_sha256=shards[0].checkpoint_sha256,
            producer_report_status_preserved=shards[0].original["status"],
        )
    result["acceptance_sha256"] = content_hash(result)
    _write(output_dir / "audit.json", result)
    return result


def consolidate(main_dir: Path, input_manifest: Path, golden_manifests: list[Path], output_dir: Path) -> dict[str, Any]:
    return _finalize([_load_shard(main_dir, input_manifest)], golden_manifests, output_dir, merged=False)


def consolidate_shards(
    shards: Sequence[tuple[Path, Path]], golden_manifests: Sequence[Path], output_dir: Path
) -> dict[str, Any]:
    if not shards or len({directory.resolve() for directory, _ in shards}) != len(shards):
        raise ValueError("Provide distinct completed audit shard directories")
    loaded = [_load_shard(directory, manifest) for directory, manifest in shards]
    return _finalize(loaded, golden_manifests, output_dir, merged=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize one audit or aggregate source-bound bounded audit shards")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--shard", type=Path, nargs=2, action="append", metavar=("CORPUS", "INPUT_MANIFEST"))
    parser.add_argument("--golden-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shard:
        if args.corpus is not None or args.input_manifest is not None:
            parser.error("Use repeated --shard pairs or the single-corpus inputs, not both")
        report = consolidate_shards([(pair[0], pair[1]) for pair in args.shard], args.golden_manifest, args.output)
    else:
        if args.corpus is None or args.input_manifest is None:
            parser.error("Supply --corpus/--input-manifest or one or more --shard pairs")
        report = consolidate(args.corpus, args.input_manifest, args.golden_manifest, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "games"}, ensure_ascii=False, indent=2))
    return {"PASS": 0, "BLOCKED": 2, "FAIL": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
