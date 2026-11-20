from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from riichienv import Action, Phase, convert

from .actions import tile136_to_tile37
from .completion import certify_completion, observed_raw_final_scores
from .contracts import (
    ENGINE_COMMIT,
    MJSOUL_RULESET,
    PRIMARY_COHORT,
    REFERENCE_SCHEMA_ID,
    SCHEMA_ID,
    SCHEMA_OVERLAY_SHA256,
    cohort_hash,
    game_split,
    matches_primary_cohort,
    model_tier,
)
from .decision import ReplayDecision, ReplayOpportunity, ResponseEvidence
from .golden import golden_coverage
from .input_audit import InputAudit
from .kan_transactions import COMMITTED_KAN_KINDS
from .native_replay import NativeReplayBridge
from .observation import ObserverLedger, PublicSnapshot, extract_rule_features
from .response import draw_response, resolve_discard_response
from .response_mask import public_rule_response_mask, validate_response_target
from .response_supervision import PASS_ID, OpportunityCounts, ResponseWindow, unique_pass_provenance
from .schema import schema_status
from .settlement import verify_settlements
from .versioning import MODEL_TIER_BASIS, artifact_contract


def _check(name: str, status: str, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "status": status, **evidence}


def check_overlay(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return _check("schema_overlay", "BLOCKED", reason="overlay_file_missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SCHEMA_OVERLAY_SHA256:
        return _check("schema_overlay", "FAIL", reason="overlay_sha256_mismatch", actual_sha256=digest)
    overlay = json.loads(path.read_text(encoding="utf-8"))
    if overlay.get("schema_id") != REFERENCE_SCHEMA_ID:
        return _check("schema_overlay", "FAIL", reason="schema_id_mismatch")
    return _check("schema_overlay", "PASS", sha256=digest, is_complete_schema=False)


def compare_generated_descriptor(protocol: Any, path: Path) -> dict[str, Any]:
    from google.protobuf.descriptor_pb2 import FileDescriptorProto  # noqa: PLC0415

    module = ast.parse(path.read_text(encoding="utf-8"))
    descriptors = [
        node.args[0].value
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "AddSerializedFile"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, bytes)
    ]
    if len(descriptors) != 1:
        return _check("protocol_descriptor_provenance", "FAIL", reason="expected_one_static_descriptor")
    descriptor = FileDescriptorProto.FromString(descriptors[0])
    actual = FileDescriptorProto()
    protocol.pool.FindFileByName(descriptor.name).CopyToProto(actual)
    for message in (descriptor, actual):
        message.ClearField("source_code_info")
        for item in message.message_type:
            _clear_json_names(item)
    equal = descriptor.SerializeToString(deterministic=True) == actual.SerializeToString(deterministic=True)
    return _check(
        "protocol_descriptor_provenance",
        "PASS" if equal else "FAIL",
        generated_module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        comparison="static_AST_only; generated Python was not executed",
    )


def _clear_json_names(message: Any) -> None:
    for field in message.field:
        field.ClearField("json_name")
    for nested in message.nested_type:
        _clear_json_names(nested)


def _require_own(own: ReplayOpportunity | None, actor: int) -> ReplayOpportunity:
    if own is None or own.snapshot.observer_player_id != actor:
        raise ValueError("Observed action has no matching frozen own-action opportunity")
    return own


def _freeze_kan_window(ledgers: list[ObserverLedger], preview: dict[int, list[Action]]) -> dict[int, ReplayOpportunity]:
    return {
        seat: ReplayOpportunity.from_kan_preview(ledgers[seat].snapshot(), actions)
        for seat, actions in preview.items()
        if actions
    }


def _finish_response(
    window: ResponseWindow,
    closure: str,
    index: int,
    inputs: InputAudit,
    ledgers: list[ObserverLedger],
    counts: OpportunityCounts,
) -> None:
    evidence = ResponseEvidence(window.trigger_event_id, index)
    for seat, offer in window.offers.items():
        choice = offer.choice
        if len(choice.candidates) <= 1:
            continue
        if seat in window.explicit:
            status = "explicit"
        elif provenance := unique_pass_provenance(choice.candidates, closure):
            # Score the frozen prefix before inserting the private marker into
            # the live ledger. This cannot affect any earlier opportunity.
            inputs.inspect(
                choice.snapshot, choice.candidates, PASS_ID, provenance=provenance, response_evidence=evidence
            )
            ledgers[seat].record_self_pass(offer.trigger, provenance=provenance)
            status = "uniquely_inferred"
        else:
            status = "censored"
        counts.observe(window.phase, closure, status)


def replay_game(\n    game: Any,\n    *,\n    on_decision: Callable[[ReplayDecision], None] | None = None,\n    on_opportunity: Callable[[Any, ReplayOpportunity, str, int], None] | None = None,\n) -> dict[str, Any]:  # noqa: PLR0915
    env = MJSOUL_RULESET.create_env(seed=0)
    ledgers = [ObserverLedger(seat, (0, 1, 2, 3)) for seat in range(4)]
    own: ReplayOpportunity | None = None
    declaring = False
    responses: ResponseWindow | None = None
    opportunities = OpportunityCounts()
    # Kakan flushes earlier queued indicators before its rob window opens.
    # A robbable ankan does not: that flush happens only after the window.
    kakan_opening: tuple[int, dict[int, list[Action]]] | None = None
    inputs = InputAudit(on_decision)
    bridge = NativeReplayBridge(env)
    errors: list[dict[str, Any]] = []
    settlement_gaps: list[dict[str, Any]] = []
    categories: set[int] = set()
    discard_count = 0
    full_windows = 0
    events = list(game.events)
    applied = 0
    final_server_scores: list[int] | None = None
    pending_discard: tuple[PublicSnapshot, int] | None = None
    next_draw_player: int | None = None
    for index, event in enumerate(events):
        inputs.trigger_event_id = index
        typ = event["type"]
        if typ in {"start_kyoku", "tsumo"}:
            pending_discard = None
        preview: dict[int, list[Action]] | None = None
        try:
            if typ == "start_kyoku":
                if responses is not None or kakan_opening is not None:
                    raise ValueError("New round before the preceding response window closed")
                own, declaring = None, False
            if declaring and typ not in {"dora", "dahai"}:
                raise ValueError("Riichi declaration must resolve as its own discard")
            if typ == "start_kyoku":
                next_draw_player = event["oya"]
            elif typ == "tsumo":
                env.validate_replay_draw(event["actor"])
                if event["actor"] != next_draw_player:
                    raise ValueError("Draw actor does not match the public turn or replacement-draw order")
                next_draw_player = None
            elif typ == "reach":
                choice = _require_own(own, event["actor"])
                if not any(action < 148 and action % 2 for action in choice.candidates):
                    raise ValueError("Observed reach is not legal before its declaration")
                declaring = True
            elif typ == "dahai":
                if env.needs_tsumo or env.done() or next_draw_player is not None:
                    raise ValueError("Discard before a required draw or replacement draw")
                actor = event["actor"]
                choice = _require_own(own, actor)
                snapshot = choice.snapshot
                target = (
                    4 * tile136_to_tile37(convert.mjai_to_tid(event["pai"]))
                    + 2 * int(event["tsumogiri"])
                    + int(declaring)
                )
                if target not in choice.candidates:
                    raise ValueError(f"Observed complete action {target} is absent from frozen legal candidates")
                extract_rule_features(snapshot)
                inputs.inspect(snapshot, choice.candidates, target)
                own, declaring = None, False
                discard_count += 1
                pending_discard = snapshot, target
                next_draw_player = (actor + 1) % 4
                if responses is not None or kakan_opening is not None:
                    raise ValueError("New discard before the preceding response window closed")
                resolution = resolve_discard_response(events, index)
                if resolution.response_observed:
                    assert resolution.response_target is not None
                    allowed = public_rule_response_mask(snapshot, target)
                    validate_response_target(allowed, resolution.response_target)
                    categories.add(resolution.response_target)
                    full_windows += 1
                else:
                    errors.append({"event_id": index, "reason": resolution.error})
            elif typ == "hora":
                actor = event["actor"]
                if actor == event["target"]:
                    choice = _require_own(own, actor)
                    inputs.inspect(choice.snapshot, choice.candidates, choice.win_target(event))
                else:
                    if responses is None:
                        raise ValueError("Observed win is absent from the pre-resolution native legal actions")
                    choice = responses.select_explicit(actor, ron=True).choice
                    inputs.inspect(
                        choice.snapshot,
                        choice.candidates,
                        choice.win_target(event),
                        response_evidence=ResponseEvidence(responses.trigger_event_id, index),
                    )
                    responses.mark_explicit(actor, ron=True)
                own = None  # Keep responders frozen for other winners in this same window.
            elif typ == "ryukyoku":
                reason = event.get("reason")
                if reason in {"kyushukyuhai", "kyushu_kyuhai"}:
                    choice = _require_own(own, env.current_player)
                    if 220 not in choice.candidates:
                        raise ValueError("Nine-terminals abort is not legal in the frozen native state")
                    inputs.inspect(choice.snapshot, choice.candidates, 220)
                else:
                    response = draw_response(reason)
                    if response is None or pending_discard is None:
                        raise ValueError("Draw termination has no matching public discard window")
                    snapshot, target = pending_discard
                    validate_response_target(public_rule_response_mask(snapshot, target), int(response))
                own = None
            elif typ in {"chi", "pon", "daiminkan", "kan", "ankan", "kakan"}:
                actor = event["actor"]
                if typ in {"ankan", "kakan"}:
                    choice = _require_own(own, actor)
                    target, selected = choice.select(event)
                    inputs.inspect(choice.snapshot, choice.candidates, target)
                    preview = bridge.preview_kan(actor, selected)
                else:
                    if responses is None or responses.phase != "discard_response":
                        raise ValueError(f"Observed {typ} has no matching frozen native legal action")
                    choice = responses.select_explicit(actor, ron=False).choice
                    target, _ = choice.select(event)
                    inputs.inspect(
                        choice.snapshot,
                        choice.candidates,
                        target,
                        response_evidence=ResponseEvidence(responses.trigger_event_id, index),
                    )
                    responses.mark_explicit(actor, ron=False)
                own = None
                next_draw_player = actor if typ in {"daiminkan", "kan", "ankan", "kakan"} else None

            # Always apply public results in source order to the current state.
            # In particular, neither a dora reveal nor a deposit refreshes an
            # already-open decision opportunity.
            bridge.prepare(event)
            env.apply_event(event)
            bridge.after_apply(event)
            for ledger in ledgers:
                ledger.apply_event(event)
            if responses is not None:
                closure = responses.closure(event, all_ron_accepted=not MJSOUL_RULESET.sanchaho_is_draw)
                if closure is not None:
                    _finish_response(responses, closure, index, inputs, ledgers, opportunities)
                    responses = None
            if typ in {"tsumo", "chi", "pon"}:
                actor = event["actor"]
                if kakan_opening is not None:
                    raise ValueError("Draw/call before the frozen robbing window could open")
                observation = env.get_observations([actor])[actor]
                own = ReplayOpportunity.observe(ledgers[actor], observation)\n                if on_opportunity is not None and own.snapshot.decision_phase == "own_act":\n                    # Capture a deep native clone at the decision cutoff, before\n                    # the factual replay applies the expert action. The callback\n                    # may branch this clone but must never mutate the replay env.\n                    on_opportunity(env, own, game.game_id, index)
            elif typ == "dahai":
                seats = sorted(env.active_players) if env.phase == Phase.WaitResponse else []
                observations = env.get_observations(seats) if seats else {}
                responses = ResponseWindow.open(
                    ledgers, {seat: ReplayOpportunity.observe(ledgers[seat], observations[seat]) for seat in seats}
                )
            elif preview is not None:
                if responses is not None:
                    raise ValueError("Kan declaration overlaps an earlier response window")
                view = ledgers[event["actor"]].snapshot()
                committed = sum(meld.kind in COMMITTED_KAN_KINDS for player in view.melds for meld in player)
                earlier_dora = committed + 1 - len(view.dora_indicators) if typ == "kakan" else 0
                if earlier_dora > 0:
                    kakan_opening = earlier_dora, preview
                else:
                    responses = ResponseWindow.open(ledgers, _freeze_kan_window(ledgers, preview))
            elif typ == "dora" and kakan_opening is not None:
                remaining, offers = kakan_opening
                if remaining == 1:
                    responses = ResponseWindow.open(ledgers, _freeze_kan_window(ledgers, offers))
                    kakan_opening = None
                else:
                    kakan_opening = remaining - 1, offers
            applied += 1
            if typ in {"hora", "ryukyoku"} and "scores" in event:
                final_server_scores = list(event["scores"])
                native = list(env.scores())
                settlement_gaps.append(
                    {
                        "event_id": index,
                        "server_scores": final_server_scores,
                        "native_scores": native,
                        "scores_match": native == final_server_scores,
                        "reason": "native_apply_event_does_not_independently_compute_settlement",
                    }
                )
        except Exception as exc:
            errors.append({"event_id": index, "event_type": typ, "reason": type(exc).__name__, "detail": str(exc)})
            break
    if responses is not None:
        _finish_response(responses, "unresolved", applied, inputs, ledgers, opportunities)
    complete = bool(events and events[0]["type"] == "start_game" and events[-1]["type"] == "end_game")
    independent: dict[str, Any] = {
        "status": "BLOCKED",
        "reason": "action_or_phase_contract_failed",
        "settlements": [],
        "computed_final_scores": None,
    }
    if not errors:
        try:
            independent = verify_settlements(events, observed_final_scores=observed_raw_final_scores(game))
        except ValueError as exc:
            independent = {
                "status": "FAIL",
                "reason": "invalid_header_raw_points",
                "settlements": [],
                "computed_final_scores": None,
            }
            errors.append({"component": "independent_settlement", "reason": type(exc).__name__, "detail": str(exc)})
    if errors or independent["status"] == "FAIL":
        status = "FAIL"
    elif not complete or independent["status"] != "PASS":
        status = "BLOCKED"
    else:
        status = "PASS"
    input_report = inputs.report()
    result: dict[str, Any] = {
        "status": status,
        "events_total": len(events),
        "draw_phase_contract": "native_observed_draw_and_public_turn_v2",
        "events_applied": applied,
        "discard_decisions_checked": discard_count,
        "complete_response_windows": full_windows,
        "response_opportunities": opportunities.report(),
        "response_categories_observed": sorted(categories),
        "complete_game_boundaries": complete,
        "server_final_scores": final_server_scores,
        "computed_final_scores": independent.get("computed_final_scores"),
        "independent_settlement": independent,
        "errors": errors,
        "native_settlement_gaps": settlement_gaps,
        "native_apply_event_scores_are_diagnostic_only": True,
        "settlement_note": (
            "HandEvaluator and an independent score ledger compute payments; server labels are comparison targets only"
        ),
        "full_token_byte_parity": input_report["status"],
        "input_encoding": input_report,
    }
    certificate = certify_completion(game, result)
    result["completion"] = certificate
    result["normalized_end_game_present"] = complete
    result["complete_game_boundaries"] = certificate["status"] == "PASS"
    result["final_ranks"] = certificate.get("final_ranks")
    if certificate["status"] == "PASS":
        result["computed_final_scores"] = certificate["expected_final_scores"]
    if errors or independent["status"] == "FAIL" or certificate["status"] == "FAIL" or input_report["status"] == "FAIL":
        result["status"] = "FAIL"
    elif independent["status"] == "PASS" and certificate["status"] == "PASS" and input_report["status"] == "PASS":
        result["status"] = "PASS"
    else:
        result["status"] = "BLOCKED"
    return result


def run_audit(  # noqa: PLR0915
    *,
    overlay_path: Path | None = None,
    schema_path: Path | None = None,
    golden_manifest_path: Path | None = None,
    proto_path: Path | None = None,
    records: tuple[Path, ...] = (),
    generated_module_path: Path | None = None,
    container: Literal["majsoul_download", "res_game_record", "wrapper", "game_detail_records"] = "majsoul_download",
    split_salt: str = "kar-m0-audit-v1",
    cohort_mode_ids: tuple[int, ...] = (12,),
    record_provenance: Literal["unverified", "real_raw_record", "synthetic"] = "unverified",
) -> dict[str, Any]:
    requested_modes = tuple(sorted(set(cohort_mode_ids))) if cohort_mode_ids else (12,)
    if requested_modes != (12,):
        raise ValueError("The frozen primary cohort is Mahjong Soul jade hanchan mode_id=12 only")
    registry = schema_status(schema_path)
    checks = [
        _check(
            "complete_schema", registry["status"], **{key: value for key, value in registry.items() if key != "status"}
        )
    ]
    if overlay_path is not None:
        checks.append(check_overlay(overlay_path))
    checks.append(_check("fixed_skill_cohort", "PASS", cohort=PRIMARY_COHORT, cohort_hash=cohort_hash()))
    checks.append(
        _check("mjsoul_rules", "PASS", ruleset_hash=MJSOUL_RULESET.ruleset_hash, rules=asdict(MJSOUL_RULESET))
    )
    golden_items: list[dict[str, Any]] = []
    if golden_manifest_path is not None:
        if golden_manifest_path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Golden manifest exceeds its bounded evidence limit")
        golden_document = json.loads(golden_manifest_path.read_text(encoding="utf-8"))
        golden_items = golden_document.get("evidence", [])
        if not isinstance(golden_items, list) or any(not isinstance(item, dict) for item in golden_items):
            raise ValueError("Golden manifest must contain evidence records")
    golden_result = golden_coverage(golden_items)
    reserved_golden_ids = {
        item["game_id"]
        for item in golden_items
        if item.get("status") == "PASS" and isinstance(item.get("game_id"), str)
    }
    games: list[dict[str, Any]] = []
    eligible: set[str] = set()
    response_game_ids: dict[int, set[str]] = {event: set() for event in range(21)}
    protocol = None
    if proto_path is None or not proto_path.is_file():
        checks.append(_check("protobuf_protocol", "BLOCKED", reason="actual_proto_missing"))
    else:
        try:
            from .mjsoul import load_protocol  # noqa: PLC0415

            protocol = load_protocol(proto_path)
            checks.append(
                _check(
                    "protobuf_protocol",
                    "PASS",
                    source=str(proto_path),
                    sha256=hashlib.sha256(proto_path.read_bytes()).hexdigest(),
                    **protocol.identity_record(),
                )
            )
            if generated_module_path is not None:
                checks.append(compare_generated_descriptor(protocol, generated_module_path))
        except Exception as exc:
            checks.append(_check("protobuf_protocol", "FAIL", reason=type(exc).__name__, detail=str(exc)))
    if not records:
        checks.append(_check("real_game_inputs", "BLOCKED", reason="no_real_raw_records_supplied"))
    if record_provenance != "real_raw_record":
        checks.append(_check("real_record_provenance", "BLOCKED", reason="origin_not_declared_as_real_raw_record"))
    seen: set[str] = set()
    for path in records:
        row: dict[str, Any] = {"source": str(path), "provenance": record_provenance, "eligible": False}
        if protocol is None:
            row.update(status="BLOCKED", reason="protocol_unavailable")
        else:
            row.update(protocol.identity_record())
            try:
                from .mjsoul import decode_record_bytes, protocol_identity  # noqa: PLC0415

                if path.stat().st_size > 64 * 1024 * 1024:
                    raise ValueError("Input exceeds raw-game record size limit; do not pass prepared datasets")
                game = decode_record_bytes(path.read_bytes(), protocol, container=container)
                if game.game_id in seen:
                    row.update(status="EXCLUDED_DUPLICATE", game_id=game.game_id)
                    games.append(row)
                    continue
                seen.add(game.game_id)
                result = replay_game(game)
                cohort_ok = matches_primary_cohort(game.metadata)
                row.update(
                    status=result["status"],
                    game_id=game.game_id,
                    source_sha256=game.source_sha256,
                    **protocol_identity(game.protocol_sha256, game.protocol_descriptor_sha256),
                    metadata=game.metadata,
                    replay=result,
                    cohort_matches=cohort_ok,
                    split=game_split(game.game_id, split_salt),
                )
                if (
                    result["status"] == "PASS"
                    and cohort_ok
                    and record_provenance == "real_raw_record"
                    and game.game_id not in reserved_golden_ids
                ):
                    eligible.add(game.game_id)
                    row["eligible"] = True
                    for category in result["response_categories_observed"]:
                        response_game_ids[category].add(game.game_id)
            except Exception as exc:
                row.update(status="FAIL", reason=type(exc).__name__, detail=str(exc))
        games.append(row)
    n_train = sum(game_split(game_id, split_salt) == "train" for game_id in eligible)
    checks.append(
        _check(
            "eligible_training_games",
            "PASS" if n_train else "BLOCKED",
            N_eligible_half_games=len(eligible),
            N_train_half_games=n_train,
        )
    )
    if any(row["status"] == "FAIL" for row in games):
        checks.append(_check("real_game_replay", "FAIL", reason="one_or_more_records_failed"))
    elif not games or any(row["status"] == "BLOCKED" for row in games):
        checks.append(_check("real_game_replay", "BLOCKED", reason="full_replay_not_verified"))
    else:
        checks.append(_check("real_game_replay", "PASS", unique_games=len(seen)))
    money = [
        row.get("replay", {}).get("independent_settlement", {})
        for row in games
        if row["status"] != "EXCLUDED_DUPLICATE"
    ]
    money_status = (
        "FAIL"
        if any(item.get("status") == "FAIL" for item in money)
        else "PASS"
        if money and all(item.get("status") == "PASS" for item in money)
        else "BLOCKED"
    )
    checks.append(
        _check(
            "independent_settlement",
            money_status,
            verified_settlements=sum(
                sum(result["status"] == "PASS" for result in item.get("settlements", [])) for item in money
            ),
            scope="observed_settlements_not_game_completion_or_all_disk_data",
        )
    )
    checks.append(
        _check(
            "real_golden_coverage",
            golden_result["status"],
            **{key: value for key, value in golden_result.items() if key != "status"},
        )
    )
    status = (
        "FAIL"
        if any(check["status"] == "FAIL" for check in checks)
        else "BLOCKED"
        if any(check["status"] == "BLOCKED" for check in checks)
        else "PASS"
    )
    return {
        **artifact_contract(),
        "schema_id": SCHEMA_ID,
        "host": "RiichiEnv",
        "reference_engine_commit": ENGINE_COMMIT,
        "status": status,
        "training_allowed": status == "PASS",
        "cloud_resources_started": False,
        "checks": checks,
        "games": games,
        "inventory_scope": "explicit_input_files_only_not_a_full_remote_disk_census",
        "N_eligible_half_games": len(eligible),
        "N_train_half_games": n_train,
        "model_tier_basis": MODEL_TIER_BASIS,
        "provisional_model_tier": model_tier(n_train),
        "split_salt": split_salt,
        "cohort_mode_ids": list(requested_modes),
        "cohort": PRIMARY_COHORT,
        "cohort_hash": cohort_hash(),
        "schema_hash": registry.get("schema_hash"),
        "golden_game_ids_reserved": sorted(reserved_golden_ids),
        **response_support_evidence(response_game_ids, split_salt),
    }


def response_support_evidence(response_game_ids: dict[int, set[str]], split_salt: str) -> dict[str, Any]:
    counts = {split: {str(event): 0 for event in range(21)} for split in ("train", "validation", "test")}
    for event, game_ids in response_game_ids.items():
        for game_id in game_ids:
            counts[game_split(game_id, split_salt)][str(event)] += 1
    return {
        "response_support_split": "train",
        "response_support_independent_games": counts["train"],
        "response_support_counts_by_split": counts,
        "response_value_supported": {event: count >= 500 for event, count in counts["train"].items()},
        "support_is_uniform_error_certificate": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only KAR M0 audit for Mahjong Soul raw records")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--golden-manifest", type=Path)
    parser.add_argument("--proto", type=Path)
    parser.add_argument("--generated-module", type=Path)
    parser.add_argument("--record", type=Path, action="append", default=[])
    parser.add_argument(
        "--container",
        default="majsoul_download",
        choices=("majsoul_download", "res_game_record", "wrapper", "game_detail_records"),
    )
    parser.add_argument(
        "--record-provenance", default="unverified", choices=("unverified", "real_raw_record", "synthetic")
    )
    parser.add_argument("--cohort-mode-id", type=int, action="append", default=[])
    parser.add_argument("--split-salt", default="kar-m0-audit-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(
        overlay_path=args.overlay,
        schema_path=args.schema,
        golden_manifest_path=args.golden_manifest,
        proto_path=args.proto,
        records=tuple(args.record),
        generated_module_path=args.generated_module,
        container=args.container,
        cohort_mode_ids=tuple(args.cohort_mode_id),
        split_salt=args.split_salt,
        record_provenance=args.record_provenance,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return {"PASS": 0, "BLOCKED": 2, "FAIL": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
