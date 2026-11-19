from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .actions import manifest_hash
from .audit import replay_game, response_support_evidence
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
from .golden import find_golden_candidates, golden_coverage, verify_golden_evidence
from .mjsoul import (
    PROTOCOL_IDENTITY_FIELDS,
    decode_record_bytes,
    load_protocol,
    protocol_identity,
    validate_protocol_identity,
)
from .schema import schema_status
from .versioning import MODEL_TIER_BASIS, artifact_contract

SEMANTIC_GAME_VERSION = "result_event_semantics_v1"


def normalized_game_sha256(events: Sequence[dict[str, Any]]) -> str:
    gameplay = [
        {
            key: value
            for key, value in event.items()
            if not key.startswith("source_") and not (event["type"] == "start_game" and key == "id")
        }
        for event in events
    ]
    return content_hash({"version": SEMANTIC_GAME_VERSION, "events": gameplay})


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint(path: Path, state: dict[str, Any]) -> None:
    state["checkpoint_sha256"] = content_hash(
        {key: value for key, value in state.items() if key != "checkpoint_sha256"}
    )
    _write(path, state)


def _inputs(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        root = Path(row.get("raw_root", path.parent)).resolve()
        name, game_id = row.get("local_file"), row.get("uuid")
        if not isinstance(game_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", game_id) is None:
            raise ValueError("Input manifest requires a safe game identity")
        if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", game_id, re.IGNORECASE):
            raise ValueError("Input manifest game identity is a reserved device name")
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Fetch manifest local filename must remain inside its raw root")
        raw = (root / name).resolve()
        if not raw.is_relative_to(root) or not raw.is_file():
            raise ValueError("Fetch manifest raw source is missing or escapes its root")
        digest = row.get("source_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Fetch manifest requires the raw byte hash")
        if game_id in seen:
            if seen[game_id] != digest:
                raise ValueError("Conflicting raw bytes for the same game identity")
            continue
        seen[game_id] = digest
        rows.append(
            {
                "game_id": game_id,
                "source": str(raw),
                "source_sha256": digest,
                "source_role": row.get("selection_role", "main_cohort_candidate"),
            }
        )
    return rows


def audit_corpus(  # noqa: PLR0915
    manifest_path: Path,
    proto_path: Path,
    output_dir: Path,
    *,
    mode: str = "discover",
    limit: int = 1000,
    split_salt: str = "kar-riichienv-v1-games",
    resume: bool = False,
) -> dict[str, Any]:
    if (
        mode not in {"discover", "audit"}
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 10000
    ):
        raise ValueError("Expected discover/audit and a bounded input limit")
    sources = _inputs(manifest_path)[:limit]
    output_dir = output_dir.resolve()
    if any(output_dir.is_relative_to(Path(row["source"]).parent) for row in sources):
        raise ValueError("Audit outputs must remain outside raw input directories")
    protocol = load_protocol(proto_path)
    registry = schema_status()
    if registry["status"] != "PASS":
        raise ValueError("Active full schema is not verified")
    package = Path(__file__).parent
    implementation = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in (
            "actions.py",
            "audit.py",
            "completion.py",
            "contracts.py",
            "golden.py",
            "input_audit.py",
            "mjsoul.py",
            "native_replay.py",
            "observation.py",
            "payments.py",
            "response.py",
            "response_supervision.py",
            "response_mask.py",
            "rule_features.py",
            "versioning.py",
            "kan_transactions.py",
            "schema.py",
            "settlement.py",
            "tokenize.py",
            "corpus_audit.py",
        )
    }
    scope = {
        **artifact_contract(),
        "mode": mode,
        "implementation_sha256": content_hash(implementation),
        "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        **protocol.identity_record(),
        "schema_hash": registry["schema_hash"],
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "cohort_hash": cohort_hash(),
        "action_manifest_hash": manifest_hash(),
        "limit": limit,
        "split_salt": split_salt,
        "normalized_game_digest_version": SEMANTIC_GAME_VERSION,
    }
    checkpoint_path = output_dir / "checkpoint.json"
    state: dict[str, Any]
    if resume:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected = content_hash({key: value for key, value in state.items() if key != "checkpoint_sha256"})
        if state.get("checkpoint_sha256") != expected:
            raise ValueError("Checkpoint content hash mismatch")
        if state.get("scope") != scope:
            raise ValueError("Checkpoint scope differs from the selected inputs or contracts")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "receipts").mkdir()
        state = {
            "scope": scope,
            "rows": [],
            "golden_by_case": {},
            "selection_complete": False,
            "all_disk_scanned": False,
        }
        _checkpoint(checkpoint_path, state)
    prior_rows = {row["game_id"]: row for row in state["rows"]}
    if len(prior_rows) != len(state["rows"]) or not set(prior_rows).issubset(source["game_id"] for source in sources):
        raise ValueError("Checkpoint contains duplicate or unselected game identities")
    processed = set(prior_rows)
    for source in sources:
        if source["game_id"] in processed:
            if hashlib.sha256(Path(source["source"]).read_bytes()).hexdigest() != source["source_sha256"]:
                raise ValueError("A previously processed source changed")
            previous = prior_rows[source["game_id"]]
            if validate_protocol_identity(previous, context="Completed corpus row") != protocol.identity_record():
                raise ValueError("A completed row protocol differs from its corpus scope")
            if previous.get("receipt_sha256"):
                receipt = json.loads(
                    (output_dir / "receipts" / f"{source['game_id']}.json").read_text(encoding="utf-8")
                )
                if content_hash(receipt) != previous["receipt_sha256"]:
                    raise ValueError("A completed native replay receipt changed")
                if validate_protocol_identity(receipt, context="Native receipt") != protocol.identity_record():
                    raise ValueError("A completed native receipt protocol differs from its corpus scope")
            elif mode == "audit" and previous["status"] != "FAIL":
                raise ValueError("A completed native replay receipt is missing")
            continue
        row: dict[str, Any] = {**source, **protocol.identity_record()}
        try:
            raw_path = Path(source["source"])
            if raw_path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("Not a bounded raw game file")
            raw = raw_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != source["source_sha256"]:
                raise ValueError("Raw source differs from its fetched byte hash")
            game = decode_record_bytes(raw, protocol, container="majsoul_download")
            if game.game_id != source["game_id"]:
                raise ValueError("Raw game identity differs from input manifest")
            candidates = find_golden_candidates(game)
            row.update(
                **protocol_identity(game.protocol_sha256, game.protocol_descriptor_sha256),
                normalized_game_sha256=normalized_game_sha256(game.events),
                metadata=game.metadata,
                cohort_matches=matches_primary_cohort(game.metadata),
                candidate_case_ids=sorted({case["case_id"] for case in candidates}),
                candidates=candidates,
                status="CANDIDATE_ONLY",
                training_eligible=False,
            )
            if mode == "audit":
                replay = replay_game(game)
                receipt = {
                    "game_id": game.game_id,
                    "source_sha256": game.source_sha256,
                    **protocol_identity(game.protocol_sha256, game.protocol_descriptor_sha256),
                    "source": str(raw_path),
                    "normalized_game_sha256": row["normalized_game_sha256"],
                    "replay": replay,
                }
                _write(output_dir / "receipts" / f"{game.game_id}.json", receipt)
                row.update(
                    receipt_sha256=content_hash(receipt),
                    status=replay["status"],
                    completion=replay.get("completion"),
                    input_encoding=replay.get("input_encoding"),
                    response_categories=replay["response_categories_observed"],
                )
                verified = []
                if replay["status"] == "PASS":
                    for candidate in candidates:
                        if candidate["case_id"] in state["golden_by_case"]:
                            continue
                        evidence = {
                            **candidate,
                            "source_path": str(raw_path),
                            "proto_path": str(proto_path.resolve()),
                            "source_provenance": "real_raw_record",
                            "container": "majsoul_download",
                        }
                        result = verify_golden_evidence(game, replay, evidence)
                        if result["status"] == "PASS":
                            state["golden_by_case"][result["case_id"]] = result
                            verified.append(result["case_id"])
                row["golden_case_ids"] = verified
        except Exception as exc:
            row.update(status="FAIL", reason=type(exc).__name__, detail=str(exc), training_eligible=False)
        state["rows"].append(row)
        processed.add(source["game_id"])
        _checkpoint(checkpoint_path, state)
    state["selection_complete"] = len(state["rows"]) == len(sources)
    _checkpoint(checkpoint_path, state)
    evidence_by_case: dict[str, dict[str, Any]] = state["golden_by_case"]
    golden_ids = {evidence["game_id"] for evidence in evidence_by_case.values()}
    split_games: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "validation", "test")}
    support: dict[int, set[str]] = {event: set() for event in range(21)}
    for row in state["rows"]:
        eligible = (
            mode == "audit"
            and row["status"] == "PASS"
            and row.get("cohort_matches")
            and row.get("source_role") != "golden_only"
            and row["game_id"] not in golden_ids
        )
        row["training_eligible"] = bool(eligible)
        if eligible:
            split = game_split(row["game_id"], split_salt)
            split_games[split].append(
                {key: row[key] for key in ("game_id", "source", "source_sha256", *PROTOCOL_IDENTITY_FIELDS)}
            )
            for event in row.get("response_categories", []):
                support[event].add(row["game_id"])
    for split, games in split_games.items():
        _write(
            output_dir / f"{split}.manifest.json",
            {"scope": scope, "games": games, "manifest_sha256": content_hash(games)},
        )
    evidence = list(evidence_by_case.values())
    _write(output_dir / "golden.manifest.json", {"scope": scope, "evidence": evidence})
    golden_result = golden_coverage(evidence)
    failure = any(row["status"] == "FAIL" for row in state["rows"])
    counts: dict[str, int] = {}
    for row in state["rows"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    n_train = len(split_games["train"])
    status = (
        "FAIL"
        if failure
        else "PASS"
        if mode == "audit"
        and not counts.get("BLOCKED")
        and golden_result["status"] == "PASS"
        and n_train
        and state["selection_complete"]
        else "BLOCKED"
    )
    report = {
        "status": status,
        "schema_id": SCHEMA_ID,
        "schema_hash": registry["schema_hash"],
        "scope": scope,
        "cohort": PRIMARY_COHORT,
        "mode": mode,
        "selection_complete": state["selection_complete"],
        "all_disk_scanned": False,
        "selected_inputs": len(sources),
        "processed": len(state["rows"]),
        "counts": counts,
        "candidate_case_ids": sorted({case for row in state["rows"] for case in row.get("candidate_case_ids", [])}),
        "golden": golden_result,
        "golden_game_ids_reserved": sorted(golden_ids),
        "N_eligible_half_games": sum(len(games) for games in split_games.values()),
        "N_train_half_games": n_train,
        "model_tier_basis": MODEL_TIER_BASIS,
        "provisional_model_tier": model_tier(n_train),
        **response_support_evidence(support, split_salt),
        "training_allowed": status == "PASS",
        "cloud_resources_started": False,
        "rows": state["rows"],
    }
    _write(output_dir / "report.json", report)
    return report


def select_golden_inputs(discovery_report: Path, fetch_manifest: Path, output_manifest: Path) -> list[str]:
    report = json.loads(discovery_report.read_text(encoding="utf-8"))
    rows = [row for row in report["rows"] if row.get("candidate_case_ids")]
    uncovered = {case for row in rows for case in row["candidate_case_ids"]}
    selected: set[str] = set()
    while uncovered:
        best = max(rows, key=lambda row: len(uncovered.intersection(row["candidate_case_ids"])))
        gain = uncovered.intersection(best["candidate_case_ids"])
        if not gain:
            break
        selected.add(best["game_id"])
        uncovered.difference_update(gain)
    original = [json.loads(line) for line in fetch_manifest.read_text(encoding="utf-8").splitlines()]
    selected_rows = [
        {**row, "raw_root": str(fetch_manifest.parent.resolve())} for row in original if row["uuid"] in selected
    ]
    if len(selected_rows) != len(selected) or output_manifest.resolve().is_relative_to(fetch_manifest.parent.resolve()):
        raise ValueError("Selected manifests must stay outside the unchanged raw directory")
    with output_manifest.open("x", encoding="utf-8") as output:
        for row in selected_rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded result-only corpus discovery and KAR M0 audit")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--proto", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("discover", "audit"), default="discover")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--split-salt", default="kar-riichienv-v1-games")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = audit_corpus(
        args.manifest,
        args.proto,
        args.output,
        mode=args.mode,
        limit=args.limit,
        split_salt=args.split_salt,
        resume=args.resume,
    )
    print(
        json.dumps({key: value for key, value in report.items() if key not in {"rows"}}, ensure_ascii=False, indent=2)
    )
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
