from __future__ import annotations

import hashlib
import random
from typing import Any

from riichienv import ActionType, Phase, RiichiEnv

from .actions import MappedAction, execute_candidate, legal_candidates
from .contracts import MJSOUL_RULESET, content_hash
from .decision import ReplayOpportunity
from .response import resolve_discard_response
from .response_mask import public_rule_response_mask, validate_response_target
from .tokenize import append_action, encode_prefix

BRANCH_DATA_VERSION = "kar_candidate_branch_probe_v1"
CONTINUATION_POLICY_VERSION = "uniform_native_legal_v1"


def _wall_seed(master_seed: int, game_id: str, cutoff_event_id: int, replicate: int) -> int:
    payload = "\\0".join(
        (str(master_seed), game_id, str(cutoff_event_id), str(replicate), "shared-wall", CONTINUATION_POLICY_VERSION)
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _seed(
    master_seed: int, game_id: str, cutoff_event_id: int, action_id: int, replicate: int
) -> int:
    payload = "\0".join(
        (str(master_seed), game_id, str(cutoff_event_id), str(action_id), str(replicate), CONTINUATION_POLICY_VERSION)
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _continue_uniform_native(
    env: RiichiEnv, rng: random.Random, *, max_decisions: int
) -> tuple[bool, int]:
    """Finish one branch using only native legal actions; no Python rule logic."""
    decisions = 0
    while not env.done():
        if decisions >= max_decisions:
            return False, decisions
        if env.phase == Phase.WaitResponse:
            seats = list(env.active_players)
            if not seats:
                raise RuntimeError("Native response phase has no active players")
            observations = env.get_observations(seats)
            choices = {}
            for seat in seats:
                legal = list(observations[seat].legal_actions())
                if not legal:
                    raise RuntimeError("Active native responder has no legal action")
                choices[seat] = rng.choice(legal)
            before = len(env.rule_log)
            env.step(choices)
            if len(env.rule_log) == before and not env.done():
                raise RuntimeError("Native response step did not advance the branch")
        elif env.needs_tsumo or env.needs_initialize_next_round:
            before = len(env.rule_log)
            env.step({})
            if len(env.rule_log) == before and not env.done():
                raise RuntimeError("Native automatic transition did not advance the branch")
        else:
            seat = int(env.current_player)
            observation = env.get_observations([seat])[seat]
            mapped = legal_candidates(observation)
            usable = [candidate for candidate in mapped if not candidate.native_execution_ambiguous]
            if usable:
                execute_candidate(env, seat, rng.choice(usable))
            else:
                legal = [action for action in observation.legal_actions() if action.action_type != ActionType.RIICHI]
                if not legal:
                    raise RuntimeError("Native active player has no executable legal action")
                before = len(env.rule_log)
                env.step({seat: rng.choice(legal)})
                if len(env.rule_log) == before and not env.done():
                    raise RuntimeError("Native own-action step did not advance the branch")
        decisions += 1
    return True, decisions


def _branch_one(
    source_env: RiichiEnv,
    opportunity: ReplayOpportunity,
    candidate: MappedAction,
    *,
    source_game_id: str,
    cutoff_event_id: int,
    master_seed: int,
    replicate: int,
    max_decisions: int,
    prefix: Any,
) -> dict[str, Any]:
    action_id = candidate.global_id
    seed = _seed(master_seed, source_game_id, cutoff_event_id, action_id, replicate)
    wall_seed = _wall_seed(master_seed, source_game_id, cutoff_event_id, replicate)
    conditional = append_action(prefix, action_id)
    row: dict[str, Any] = {
        "version": BRANCH_DATA_VERSION,
        "tokens_hex": conditional.to_bytes()[108:].hex(),
        "history_rows": conditional.history_rows,
        "decide_index": conditional.decide_index,
        "action_index": conditional.history_rows + 64,
        "candidates": list(conditional.candidates),
        "action_target": -1,
        "action_valid": False,
        "response_applicable": True,
        "response_observed": True,
        "value_valid": True,
        "source_game_id": source_game_id,
        "source_cutoff_event_id": cutoff_event_id,
        "observer": opportunity.snapshot.observer_player_id,
        "candidate_action_id": action_id,
        "replicate": replicate,
        "seed": seed,
        "wall_seed": wall_seed,
        "wall_conditioning": "native_unknown_tiles_sampled_once_per_source_replicate_shared_across_candidates",
        "continuation_policy": CONTINUATION_POLICY_VERSION,
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "source_candidates": list(opportunity.candidates),
        "status": "BLOCKED",
        "complete": False,
        "response_target": None,
        "response_allowed": None,
        "final_rank": None,
        "utility": None,
    }
    if action_id not in opportunity.candidates or not 0 <= action_id < 148:
        row["reason"] = "not_a_legal_discard_candidate"
        return row
    if candidate.native_execution_ambiguous:
        row["reason"] = "native_discard_provenance_ambiguous"
        return row
    if candidate.spec.action_type != "DISCARD":
        row["reason"] = "candidate_is_not_discard"
        return row
    if candidate.spec.riichi and len(candidate.actions) != 2:
        row["reason"] = "invalid_riichi_transaction"
        return row

    env = source_env.clone()
    before = len(env.rule_log)
    try:
        env.prepare_replay_continuation_wall(wall_seed)
        execute_candidate(env, opportunity.snapshot.observer_player_id, candidate)
        after_action = list(env.rule_log)
        trigger_ids = [
            index
            for index, event in enumerate(after_action[before:], start=before)
            if event.get("type") == "dahai" and event.get("actor") == opportunity.snapshot.observer_player_id
        ]
        if len(trigger_ids) != 1:
            raise ValueError("Candidate branch did not emit exactly one native discard")
        trigger_id = trigger_ids[0]
        allowed = public_rule_response_mask(opportunity.snapshot, action_id)
        complete, continuation_decisions = _continue_uniform_native(
            env, random.Random(seed), max_decisions=max_decisions
        )
        if not complete:
            row.update(
                status="INCOMPLETE",
                reason="continuation_decision_cap",
                continuation_decisions=continuation_decisions,
            )
            return row
        events = list(env.rule_log)
        resolution = resolve_discard_response(events, trigger_id)
        if not resolution.response_observed or resolution.response_target is None:
            raise ValueError("Completed branch has no unambiguous native discard response")
        validate_response_target(tuple(allowed), resolution.response_target)
        ranks = list(env.ranks())
        rank = ranks[opportunity.snapshot.observer_player_id]
        if rank not in (1, 2, 3, 4):
            raise ValueError("Native final rank is outside 1..4")
        row.update(
            status="PASS",
            complete=True,
            reason=None,
            continuation_decisions=continuation_decisions,
            branch_trigger_event_id=trigger_id,
            response_target=int(resolution.response_target),
            response_resolution_event_ids=list(resolution.response_resolution_event_ids),
            response_allowed=[bool(value) for value in allowed],
            final_rank=rank - 1,
            utility=2.5 - rank,
            final_scores=list(env.scores()),
            branch_event_count=len(events) - before,
        )
        row["branch_sha256"] = content_hash(row)
        return row
    except Exception as error:
        row.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
        return row


def simulate_candidate_branches(
    source_env: RiichiEnv,
    opportunity: ReplayOpportunity,
    *,
    source_game_id: str,
    cutoff_event_id: int,
    master_seed: int,
    replicates: int = 1,
    max_decisions: int = 10000,
) -> list[dict[str, Any]]:
    """Enumerate legal discard candidates from one immutable native decision state.

    This probe uses a deliberately simple, versioned uniform-legal continuation
    policy. Its outcomes are conditional on that policy and are not production
    counterfactual ground truth for arbitrary opponents.
    """
    if not source_game_id or type(cutoff_event_id) is not int or cutoff_event_id < 0:
        raise ValueError("A branch requires a source game and nonnegative decision cutoff")
    if type(master_seed) is not int or not 0 <= master_seed < 2**64:
        raise ValueError("Branch master seed must be an unsigned 64-bit integer")
    if type(replicates) is not int or not 1 <= replicates <= 32:
        raise ValueError("Probe replicates must be in 1..32")
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("Branch continuation cap must be positive")
    if opportunity.snapshot.decision_phase != "own_act":
        raise ValueError("Candidate branch generation only supports own-action decisions")
    if source_env.done() or source_env.phase != Phase.WaitAct:
        raise ValueError("Native source environment is not at an own-action decision")
    if int(source_env.current_player) != opportunity.snapshot.observer_player_id:
        raise ValueError("Native source actor and public observer differ")
    if tuple(sorted(item.global_id for item in opportunity.mapped)) != opportunity.candidates:
        raise ValueError("Opportunity mapped actions differ from its frozen candidate set")

    candidates = [item for item in opportunity.mapped if 0 <= item.global_id < 148]
    prefix = encode_prefix(opportunity.snapshot, opportunity.candidates)
    source_identity = content_hash(
        {
            "game_id": source_game_id,
            "cutoff_event_id": cutoff_event_id,
            "observer": opportunity.snapshot.observer_player_id,
            "candidates": opportunity.candidates,
            "native_rule_log_sha256": content_hash(list(source_env.rule_log)),
        }
    )
    rows = []
    for candidate in candidates:
        for replicate in range(replicates):
            row = _branch_one(
                source_env,
                opportunity,
                candidate,
                source_game_id=source_game_id,
                cutoff_event_id=cutoff_event_id,
                master_seed=master_seed,
                replicate=replicate,
                max_decisions=max_decisions,
                prefix=prefix,
            )
            row["source_state_identity"] = source_identity
            row["expected_branch_action_ids"] = [action for action in opportunity.candidates if 0 <= action < 148]
            row["replicates"] = replicates
            row.pop("branch_sha256", None)
            row["branch_sha256"] = content_hash(row)
            rows.append(row)
    return rows
