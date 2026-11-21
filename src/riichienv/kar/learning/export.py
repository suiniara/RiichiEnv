from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from ..actions import action_spec
from ..audit import replay_game
from ..calibration_roles import CALIBRATION_PARTITION_VERSION, CALIBRATION_SPLIT_SALT
from ..contracts import cohort_hash, content_hash, game_split, matches_primary_cohort
from ..decision import ReplayDecision
from ..kan_transactions import KAN_TRANSACTION_VERSION
from ..mjsoul import decode_record_bytes, load_protocol, protocol_identity, validate_protocol_identity
from ..response import resolve_discard_response
from ..response_mask import public_rule_response_mask, validate_response_target
from ..response_supervision import RESPONSE_SUPERVISION_VERSION, OpportunityCounts
from ..schema import TOKEN_BYTE_SIZE
from ..versioning import artifact_contract, game_count_contract, validate_artifact_contract

DATASET_VERSION = "kar_decision_sqlite_v2_candidate_branches"
ACTION_LABEL_VERSION = "explicit_and_unique_response_pass_v3"
SQL_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE games (game_id TEXT PRIMARY KEY, split TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    final_ranks TEXT NOT NULL, replay_sha256 TEXT NOT NULL);
CREATE TABLE histories (game_id TEXT NOT NULL, observer INTEGER NOT NULL, tokens BLOB NOT NULL,
                        PRIMARY KEY(game_id,observer));
CREATE TABLE samples (
 row_id TEXT PRIMARY KEY, game_id TEXT NOT NULL, observer INTEGER NOT NULL, round_instance INTEGER NOT NULL,
 ordinal INTEGER NOT NULL, cutoff_event_id INTEGER NOT NULL, trigger_event_id INTEGER NOT NULL,
 history_rows INTEGER NOT NULL, state BLOB NOT NULL, action BLOB NOT NULL, candidates BLOB NOT NULL,
 action_target INTEGER NOT NULL, action_valid INTEGER NOT NULL, response_applicable INTEGER NOT NULL,
 response_observed INTEGER NOT NULL, response_target INTEGER NOT NULL, final_rank INTEGER NOT NULL,
 value_valid INTEGER NOT NULL, response_allowed BLOB NOT NULL, provenance TEXT NOT NULL, diagnostic TEXT NOT NULL);
CREATE INDEX samples_game ON samples(game_id,observer,ordinal);
CREATE TABLE branch_samples (
 branch_id TEXT PRIMARY KEY, source_game_id TEXT NOT NULL, source_decision_row_id TEXT NOT NULL,
 split TEXT NOT NULL, candidate_action_id INTEGER NOT NULL, replicate INTEGER NOT NULL,
 status TEXT NOT NULL, tokens BLOB NOT NULL, payload TEXT NOT NULL);
CREATE INDEX branch_samples_source ON branch_samples(source_game_id,source_decision_row_id);
"""


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def learning_contract() -> dict[str, Any]:
    return {
        **artifact_contract(),
        "dataset_version": DATASET_VERSION,
        "cohort_hash": cohort_hash(),
        "kan_transaction_version": KAN_TRANSACTION_VERSION,
        "calibration_partition_version": CALIBRATION_PARTITION_VERSION,
        "calibration_split_salt": CALIBRATION_SPLIT_SALT,
        "action_label_version": ACTION_LABEL_VERSION,
        "response_supervision_version": RESPONSE_SUPERVISION_VERSION,
        "rank_tiebreak": "initial_seat_order_east_south_west_north",
        "initial_seat_order": [0, 1, 2, 3],
    }


class GameWriter:
    def __init__(
        self, connection: sqlite3.Connection, game: Any, *, split: str = "train",
        branch_probe_decisions: int = 8, branch_replicates: int = 1,
        branch_seed: int = 17, branch_max_decisions: int = 10000,
    ):
        self.db, self.game, self.split = connection, game, split
        self.branch_probe_decisions = branch_probe_decisions
        self.branch_replicates = branch_replicates
        self.branch_seed = branch_seed
        self.branch_max_decisions = branch_max_decisions
        self.branch_opportunities = 0
        self.branch_counts: Counter[str] = Counter()
        self.histories: dict[int, bytes] = {}
        self.ordinals: Counter[tuple[int, int]] = Counter()
        self.round_at = []
        current = -1
        for event in game.events:
            current += event["type"] == "start_kyoku"
            self.round_at.append(current)
        self.responses: dict[int, Any] = {}
        self.examples = self.response_count = self.value_count = 0
        self.provenance: Counter[str] = Counter()
        self.response_labels: Counter[str] = Counter()

    def on_opportunity(self, native_env: Any, opportunity: Any, game_id: str, cutoff_event_id: int) -> None:
        """Persist all candidate branches for the configured source decisions."""
        if self.branch_probe_decisions <= 0 or opportunity.snapshot.decision_phase != "own_act":
            return
        if not any(0 <= action < 148 for action in opportunity.candidates):
            return
        if self.branch_opportunities >= self.branch_probe_decisions:
            return
        from ..candidate_branches import simulate_candidate_branches  # noqa: PLC0415

        actor = opportunity.snapshot.observer_player_id
        round_instance = self.round_at[cutoff_event_id]
        ordinal = self.ordinals[round_instance, actor]
        source_row_id = f"{self.game.game_id}/round{round_instance}/seat{actor}/decision{ordinal}"
        rows = simulate_candidate_branches(
            native_env,
            opportunity,
            source_game_id=self.game.game_id,
            cutoff_event_id=cutoff_event_id,
            master_seed=self.branch_seed,
            replicates=self.branch_replicates,
            max_decisions=self.branch_max_decisions,
        )
        self.branch_opportunities += 1
        for row in rows:
            row["source_decision_row_id"] = source_row_id
            row["split"] = self.split
            row.pop("branch_sha256", None)
            row["branch_sha256"] = content_hash(row)
            branch_id = content_hash(
                {
                    "source_decision_row_id": source_row_id,
                    "candidate_action_id": row["candidate_action_id"],
                    "replicate": row["replicate"],
                    "seed": row["seed"],
                    "continuation_policy": row["continuation_policy"],
                }
            )
            tokens = bytes.fromhex(row["tokens_hex"])
            payload = {key: value for key, value in row.items() if key != "tokens_hex"}
            self.db.execute(
                "INSERT INTO branch_samples VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    branch_id,
                    self.game.game_id,
                    source_row_id,
                    self.split,
                    row["candidate_action_id"],
                    row["replicate"],
                    row["status"],
                    tokens,
                    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
                ),
            )
            self.branch_counts[row["status"]] += 1

    def _decision_diagnostics(self, event: ReplayDecision) -> dict[str, int]:
        snapshot, trigger, evidence = event.snapshot, event.trigger_event_id, event.response_evidence
        response = snapshot.decision_phase in {"discard_response", "kan_response"}
        cutoff, length = snapshot.cutoff_event_id, len(self.game.events)
        if not 0 <= trigger < length or response != (evidence is not None):
            raise ValueError("Decision lacks its source or response window evidence")
        if evidence is None:
            if cutoff > trigger:
                raise ValueError("Decision cutoff is after its source action")
            return {}
        opening, closing = evidence.window_trigger_event_id, evidence.label_evidence_event_id
        explicit = event.action_provenance == "explicit"
        expected_kinds = {"dahai"} if snapshot.decision_phase == "discard_response" else {"ankan", "kakan"}
        if (
            not 0 <= opening <= cutoff <= closing < length
            or opening >= closing
            or self.game.events[opening]["type"] not in expected_kinds
            or self.round_at[opening] != self.round_at[closing]
            or trigger != (closing if explicit else opening)
            or not explicit
            and event.action_target_global != 221
        ):
            raise ValueError("Response label is not bound to its frozen opening and public result")
        if len(event.prefix.candidates) > 1:
            self.response_labels["explicit" if explicit else "uniquely_inferred"] += 1
        return {"response_window_trigger_event_id": opening, "response_label_evidence_event_id": closing}

    def __call__(self, event: ReplayDecision) -> None:
        snapshot, prefix = event.snapshot, event.prefix
        actor, trigger = snapshot.observer_player_id, event.trigger_event_id
        response_evidence = self._decision_diagnostics(event)
        round_instance = self.round_at[trigger]
        ordinal = self.ordinals[round_instance, actor]
        self.ordinals[round_instance, actor] += 1
        row_id = f"{self.game.game_id}/round{round_instance}/seat{actor}/decision{ordinal}"
        payload = prefix.to_bytes()[108:]
        history_end = (prefix.history_rows + 1) * TOKEN_BYTE_SIZE
        history, state = payload[:history_end], payload[history_end:]
        old = self.histories.get(actor, b"")
        if not history.startswith(old):
            raise ValueError("Observer history was rewritten during a game")
        self.histories[actor] = history
        if len(state) != 63 * TOKEN_BYTE_SIZE:
            raise ValueError("Expected exactly 62 snapshot rows and DECIDE")
        action = event.conditional.to_bytes()[-TOKEN_BYTE_SIZE:]
        applicable = action_spec(event.action_target_global).action_type == "DISCARD"
        observed, target, allowed, resolution_ids = False, -1, [True] * 21, []
        if applicable:
            if self.game.events[trigger]["type"] != "dahai":
                raise ValueError("Discard label is not bound to a discard trigger")
            resolution = self.responses.setdefault(trigger, resolve_discard_response(list(self.game.events), trigger))
            observed, target = resolution.response_observed, resolution.response_target
            allowed = list(public_rule_response_mask(snapshot, event.action_target_global))
            if observed:
                assert target is not None
                validate_response_target(tuple(allowed), target)
            target = -1 if target is None else int(target)
            resolution_ids = list(resolution.response_resolution_event_ids)
        diagnostic = {
            **response_evidence,
            "score_before_action": snapshot.scores[actor],
            "response_resolution_event_ids": resolution_ids,
            "round_return_valid": False,
            "match_return_valid": False,
        }
        legal = prefix.candidates + (546,) * (32 - len(prefix.candidates))
        values = (
            row_id,
            self.game.game_id,
            actor,
            round_instance,
            ordinal,
            snapshot.cutoff_event_id,
            trigger,
            prefix.history_rows,
            state,
            action,
            struct.pack("<32H", *legal),
            event.action_target_global,
            1,
            int(applicable),
            int(observed),
            target,
            -1,
            0,
            bytes(allowed),
            event.action_provenance,
            json.dumps(diagnostic, separators=(",", ":")),
        )
        self.db.execute("INSERT INTO samples VALUES (" + ",".join("?" for _ in values) + ")", values)
        self.examples += 1
        self.response_count += int(applicable and observed)
        self.provenance[event.action_provenance] += 1

    def finish(self, replay: dict[str, Any], split: str) -> dict[str, Any]:
        identity = protocol_identity(self.game.protocol_sha256, self.game.protocol_descriptor_sha256)
        if replay["status"] != "PASS" or not replay.get("final_ranks"):
            raise ValueError(f"Game failed complete export replay: {replay.get('errors')}")
        counters = replay["response_opportunities"]
        if counters["version"] != RESPONSE_SUPERVISION_VERSION or any(
            counters[key] != self.response_labels[key] for key in ("explicit", "uniquely_inferred")
        ):
            raise ValueError("Response opportunity evidence differs from emitted action supervision")
        ranks = replay["final_ranks"]
        for actor, rank in enumerate(ranks):
            if isinstance(rank, bool) or rank not in range(1, 5):
                raise ValueError("Complete rank must be explicitly 1..4")
            self.db.execute(
                "UPDATE samples SET final_rank=?,value_valid=1 WHERE game_id=? AND observer=? "
                "AND response_applicable=1 AND response_observed=1",
                (rank - 1, self.game.game_id, actor),
            )
        endings = {}
        for settlement in replay["independent_settlement"]["settlements"]:
            if settlement.get("expected_scores") is not None:
                endings[self.round_at[settlement["event_ids"][0]]] = settlement["expected_scores"]
        records = self.db.execute(
            "SELECT row_id,observer,round_instance,diagnostic FROM samples WHERE game_id=?", (self.game.game_id,)
        ).fetchall()
        for row_id, actor, round_instance, diagnostic_json in records:
            diagnostic = json.loads(diagnostic_json)
            if round_instance in endings:
                after = endings[round_instance][actor]
                diagnostic.update(
                    score_after_same_round=after,
                    round_delta_points=after - diagnostic["score_before_action"],
                    round_return_valid=True,
                )
            diagnostic.update(match_return_valid=True, final_rank=ranks[actor])
            self.db.execute(
                "UPDATE samples SET diagnostic=? WHERE row_id=?",
                (json.dumps(diagnostic, separators=(",", ":")), row_id),
            )
        self.db.execute(
            "INSERT INTO games VALUES (?,?,?,?,?)",
            (self.game.game_id, split, self.game.source_sha256, json.dumps(ranks), content_hash(replay)),
        )
        for actor, payload in self.histories.items():
            self.db.execute("INSERT INTO histories VALUES (?,?,?)", (self.game.game_id, actor, payload))
        self.value_count = self.response_count
        return {
            "game_id": self.game.game_id,
            "split": split,
            "source_sha256": self.game.source_sha256,
            **identity,
            "examples": self.examples,
            "response_valid": self.response_count,
            "value_valid": self.value_count,
            "action_provenance_counts": dict(self.provenance),
            "response_opportunities": counters,
            "replay_sha256": content_hash(replay),
            "unlabelled_response_intents": "overridden_or_unobserved_private_claims_not_guessed",
            "candidate_branch_opportunities": self.branch_opportunities,
            "candidate_branch_rows": dict(self.branch_counts),
        }


def _effective_supervision_counts(db: sqlite3.Connection, split: str) -> tuple[int, int, int, dict[int, int]]:
    """Count at most one auxiliary-supervision unit per source decision."""
    response_count = value_count = complete_source_decisions = 0
    action_source_counts: Counter[int] = Counter()
    branch_groups: dict[str, list[tuple[int, int, str, bytes, dict[str, Any]]]] = {}
    for source_row_id, action, replicate, status, tokens, payload in db.execute(
        "SELECT source_decision_row_id,candidate_action_id,replicate,status,tokens,payload "
        "FROM branch_samples WHERE split=? ORDER BY source_decision_row_id,candidate_action_id,replicate",
        (split,),
    ):
        branch_groups.setdefault(source_row_id, []).append((action, replicate, status, tokens, json.loads(payload)))
    rows = db.execute(
        "SELECT s.row_id,s.game_id,s.cutoff_event_id,s.observer,s.candidates,"
        "s.response_applicable,s.response_observed,s.value_valid "
        "FROM samples s JOIN games g ON s.game_id=g.game_id WHERE g.split=? ORDER BY s.row_id",
        (split,),
    )
    for row_id, game_id, cutoff_event_id, observer, candidate_blob, applicable, observed, value_valid in rows:
        legal = tuple(value for value in struct.unpack("<32H", candidate_blob) if value != 546)
        branches = branch_groups.get(row_id, [])
        complete = False
        if branches:
            first = branches[0][4]
            actions = tuple(first.get("expected_branch_action_ids", ()))
            replicates = first.get("replicates")
            expected = {
                (action, replicate)
                for action in actions
                for replicate in range(replicates if type(replicates) is int else 0)
            }
            actual = {(item[0], item[1]) for item in branches}
            complete = (
                bool(actions)
                and actions == tuple(action for action in legal if action < 148)
                and type(replicates) is int
                and replicates > 0
                and actual == expected
                and all(item[2] == "PASS" for item in branches)
                and all(
                    tuple(item[4].get("expected_branch_action_ids", ())) == actions
                    and item[4].get("replicates") == replicates
                    and item[4].get("candidates") == list(legal)
                    and item[4].get("candidate_action_id") == item[0]
                    and item[4].get("source_decision_row_id") == row_id
                    and item[4].get("source_game_id") == game_id
                    and item[4].get("source_cutoff_event_id") == cutoff_event_id
                    and item[4].get("observer") == observer
                    and item[4].get("source_state_identity") == first.get("source_state_identity")
                    and item[4].get("split") == split
                    and item[4].get("branch_sha256")
                    == content_hash(
                        {
                            key: value
                            for key, value in {**item[4], "tokens_hex": item[3].hex()}.items()
                            if key != "branch_sha256"
                        }
                    )
                    for item in branches
                )
            )
        if complete:
            response_count += 1
            value_count += 1
            complete_source_decisions += 1
            action_source_counts.update(actions)
        else:
            response_count += int(bool(applicable) and bool(observed))
            value_count += int(bool(value_valid))
    return response_count, value_count, complete_source_decisions, dict(action_source_counts)


def _selection(acceptance_dir: Path, limits: dict[str, int | None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accepted = json.loads((acceptance_dir / "audit.json").read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in accepted.items() if key != "acceptance_sha256"}
    if (
        accepted.get("acceptance_sha256") != content_hash(unsigned)
        or accepted.get("status") != "PASS"
        or accepted.get("training_allowed") is not True
    ):
        raise ValueError("M1 requires a source-bound PASS acceptance manifest")
    validate_artifact_contract(accepted.get("source_scope", {}), context="M0 acceptance for dataset export")
    protocol = validate_protocol_identity(accepted["source_scope"], context="M0 acceptance for dataset export")
    contract = learning_contract()
    if any(
        accepted["source_scope"].get(key) != contract[key]
        for key in ("schema_hash", "ruleset_hash", "cohort_hash", "action_manifest_hash")
    ):
        raise ValueError("M0 acceptance contract differs from active M1")
    selected, seen = [], set()
    reserved = set(accepted["golden_game_ids_reserved"])
    for split in ("train", "validation", "test"):
        limit = limits[split]
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise ValueError("Subset limits must be nonnegative or None for all")
        document = json.loads((acceptance_dir / f"{split}.manifest.json").read_text(encoding="utf-8"))
        rows = document["games"]
        if document.get("manifest_sha256") != content_hash(rows) or document["scope"] != accepted["source_scope"]:
            raise ValueError("Train/validation/test manifest was modified")
        for source in rows[:limit] if limit is not None else rows:
            if validate_protocol_identity(source, context="Selected game") != protocol:
                raise ValueError("Selected game protocol differs from its accepted scope")
            identity = source["game_id"]
            if (
                identity in reserved
                or identity in seen
                or source.get("source_role") == "golden_only"
                or game_split(identity, document["scope"]["split_salt"]) != split
            ):
                raise ValueError("Duplicate, Golden or cross-split game cannot enter M1")
            seen.add(identity)
            selected.append({**source, "split": split})
    return {
        "acceptance_sha256": accepted["acceptance_sha256"],
        **protocol,
        "split_salt": accepted["source_scope"]["split_salt"],
        "available_split_counts": accepted["split_counts"],
        "subset_limits": limits,
    }, selected


def export_dataset(  # noqa: PLR0915
    acceptance_dir: Path,
    proto_path: Path,
    output_dir: Path,
    *,
    train_games: int | None = 6,
    validation_games: int | None = 2,
    test_games: int | None = 0,
    resume: bool = False,
    branch_probe_decisions: int = 8,
    branch_replicates: int = 1,
    branch_seed: int = 17,
    branch_max_decisions: int = 10000,
) -> dict[str, Any]:
    selection, sources = _selection(
        acceptance_dir, {"train": train_games, "validation": validation_games, "test": test_games}
    )
    if (
        type(branch_probe_decisions) is not int or branch_probe_decisions < 1
        or type(branch_replicates) is not int or not 1 <= branch_replicates <= 32
        or type(branch_seed) is not int or not 0 <= branch_seed < 2**64
        or type(branch_max_decisions) is not int or branch_max_decisions < 1
    ):
        raise ValueError("Candidate-action branch generation must be enabled with at least one decision per game")
    if not any(source["split"] == "train" for source in sources):
        raise ValueError("No training games selected")
    output_dir = output_dir.resolve()
    if any(output_dir.is_relative_to(Path(source["source"]).resolve().parent) for source in sources):
        raise ValueError("Learning output must stay outside raw directories")
    protocol = load_protocol(proto_path)
    if protocol.identity_record() != validate_protocol_identity(selection, context="Dataset selection"):
        raise ValueError("Export protocol source or imported descriptors differ from accepted inputs")
    package = Path(__file__).parents[1]
    implementation = {
        name: _file_sha(package / name)
        for name in (
            "learning/export.py",
            "candidate_branches.py",
            "audit.py",
            "completion.py",
            "mjsoul.py",
            "settlement.py",
            "input_audit.py",
            "decision.py",
            "observation.py",
            "schema.py",
            "tokenize.py",
            "response.py",
            "response_supervision.py",
            "response_mask.py",
            "rule_features.py",
            "versioning.py",
            "native_replay.py",
            "kan_transactions.py",
        )
    }
    scope = {
        **learning_contract(),
        **selection,
        "producer_implementation_sha256": content_hash(implementation),
        **protocol.identity_record(),
        "sources": sources,
        "label_scope": ACTION_LABEL_VERSION,
        "candidate_branch_generation": {
            "version": "kar_candidate_action_branch_v2",
            "enabled": True,
            "max_decisions_per_game": branch_probe_decisions,
            "replicates": branch_replicates,
            "master_seed": branch_seed,
            "max_continuation_decisions": branch_max_decisions,
            "continuation_policy": "native_riichi_baseline_v2",
            "wall_conditioning": "native_unknown_tiles_sampled_once_per_source_replicate_shared_across_candidates",
            "scope": "candidate_action_branch_supervision",
        },
    }
    if not resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not output_dir.is_dir():
        raise ValueError("No export directory to resume")
    db_path = output_dir / "decisions.sqlite3"
    db = sqlite3.connect(db_path)
    rows = []
    try:
        if not resume:
            db.executescript(SQL_SCHEMA)
            db.execute("INSERT INTO metadata VALUES ('scope',?)", (json.dumps(scope, sort_keys=True),))
            db.commit()
        elif json.loads(db.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()[0]) != scope:
            raise ValueError("Export resume contract/input scope differs")
        for source in sources:
            path = Path(source["source"])
            if path.stat().st_size > 64 * 1024 * 1024 or _file_sha(path) != source["source_sha256"]:
                raise ValueError("Original audited game bytes changed")
            prior = db.execute("SELECT value FROM metadata WHERE key=?", ("game:" + source["game_id"],)).fetchone()
            if prior:
                record = json.loads(prior[0])
                if validate_protocol_identity(record, context="Completed exported game") != protocol.identity_record():
                    raise ValueError("Completed export game protocol differs from its dataset scope")
                rows.append(record)
                continue
            game = decode_record_bytes(path.read_bytes(), protocol, container="majsoul_download")
            if game.game_id != source["game_id"] or not matches_primary_cohort(game.metadata):
                raise ValueError("Source identity or primary cohort differs")
            writer = GameWriter(
                db, game, split=source["split"], branch_probe_decisions=branch_probe_decisions,
                branch_replicates=branch_replicates, branch_seed=branch_seed,
                branch_max_decisions=branch_max_decisions,
            )
            try:
                replay = replay_game(game, on_decision=writer, on_opportunity=writer.on_opportunity)
                record = writer.finish(replay, source["split"])
                db.execute(
                    "INSERT INTO metadata VALUES (?,?)", ("game:" + game.game_id, json.dumps(record, sort_keys=True))
                )
                db.commit()
                rows.append(record)
            except BaseException:
                db.rollback()
                raise
        totals = {
            split: {"games": 0, "examples": 0, "response_valid": 0, "value_valid": 0}
            for split in ("train", "validation", "test")
        }
        supervision = {split: OpportunityCounts() for split in totals}
        for row in rows:
            split = row["split"]
            total = totals[split]
            total["games"] += 1
            supervision[split].merge(row["response_opportunities"])
            for key in ("examples", "response_valid", "value_valid"):
                total[key] += row[key]
        effective_supervision = {
            split: _effective_supervision_counts(db, split) for split in ("train", "validation", "test")
        }
        for split, (response_count, value_count, _, _) in effective_supervision.items():
            totals[split]["response_valid"] = response_count
            totals[split]["value_valid"] = value_count
        train = totals["train"]
        if not train["response_valid"] or not train["value_valid"]:
            raise ValueError("Three-head recipe requires positive fixed training rho_E/rho_V")
        manifest = {
            **game_count_contract(
                selection["available_split_counts"], {split: totals[split]["games"] for split in totals}
            ),
            "status": "PASS",
            "scope": scope,
            "games": rows,
            "counts": totals,
            "candidate_branch_counts": {
                "opportunities": sum(row.get("candidate_branch_opportunities", 0) for row in rows),
                "rows": {
                    status: sum(row.get("candidate_branch_rows", {}).get(status, 0) for row in rows)
                    for status in sorted({status for row in rows for status in row.get("candidate_branch_rows", {})})
                },
                "complete_source_decisions_by_split": {
                    split: effective_supervision[split][2] for split in ("train", "validation", "test")
                },
                "candidate_action_source_counts_by_split": {
                    split: {
                        str(action): count
                        for action, count in sorted(effective_supervision[split][3].items())
                    }
                    for split in ("train", "validation", "test")
                },
                "training_supervision_enabled": True,
            },
            "response_opportunities_by_split": {split: value.report() for split, value in supervision.items()},
            "rho_E": train["response_valid"] / train["examples"],
            "rho_V": train["value_valid"] / train["examples"],
            "is_probe_subset": any(
                totals[split]["games"] != selection["available_split_counts"][split]
                for split in ("train", "validation")
            ),
            "complete": True,
            "source_data_read_only": True,
        }
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('complete', 'true')")
        db.commit()
    finally:
        db.close()
    manifest["database_sha256"] = _file_sha(db_path)
    manifest["manifest_sha256"] = content_hash(manifest)
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export source-bound M0 decisions into shared-history M1 data")
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--proto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-games", type=int, default=6)
    parser.add_argument("--validation-games", type=int, default=2)
    parser.add_argument("--test-games", type=int, default=0)
    parser.add_argument("--all-games", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--branch-probe-decisions", "--candidate-branch-decisions-per-game", dest="branch_probe_decisions", type=int, default=8, help="Maximum own-action decision states branched per source game (default: enabled, 8)")
    parser.add_argument("--branch-replicates", type=int, default=1)
    parser.add_argument("--branch-seed", type=int, default=17)
    parser.add_argument("--branch-max-decisions", type=int, default=10000)
    args = parser.parse_args()
    manifest = export_dataset(
        args.acceptance,
        args.proto,
        args.output,
        train_games=None if args.all_games else args.train_games,
        validation_games=None if args.all_games else args.validation_games,
        test_games=None if args.all_games else args.test_games,
        resume=args.resume,
        branch_probe_decisions=args.branch_probe_decisions,
        branch_replicates=args.branch_replicates,
        branch_seed=args.branch_seed,
        branch_max_decisions=args.branch_max_decisions,
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "games"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
