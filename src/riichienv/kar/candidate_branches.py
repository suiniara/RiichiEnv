from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from riichienv import ActionType, Phase, RiichiEnv

from .actions import MappedAction, execute_candidate, legal_candidates
from .contracts import MJSOUL_RULESET, content_hash
from .decision import ReplayOpportunity
from .response import resolve_discard_response
from .response_mask import public_rule_response_mask, validate_response_target
from .tokenize import append_action, encode_prefix

BRANCH_DATA_VERSION = "kar_candidate_action_branch_v2"
CONTINUATION_POLICY_VERSION = "native_riichi_baseline_v2"


def _wall_seed(master_seed: int, game_id: str, cutoff_event_id: int, replicate: int) -> int:
    payload = "\0".join(
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


def _choose_response_action(legal: list[Any], rng: random.Random) -> Any:
    wins = [action for action in legal if action.action_type == ActionType.RON]
    if wins:
        return wins[0]
    passes = [action for action in legal if action.action_type == ActionType.PASS]
    calls = [action for action in legal if action.action_type not in {ActionType.PASS, ActionType.RON}]
    if calls and rng.random() < 0.15:
        return rng.choice(calls)
    if passes:
        return passes[0]
    if legal:
        return rng.choice(legal)
    raise RuntimeError("Native responder has no legal action")


def _tile_base(tile136: int) -> int:
    if not 0 <= tile136 < 136:
        raise ValueError("Native observation contains an invalid physical tile")
    if tile136 < 108:
        suit, offset = divmod(tile136, 36)
        rank = offset // 4 + 1
        return suit * 9 + rank - 1
    return 27 + (tile136 - 108) // 4


def _dora_base(indicator136: int) -> int:
    base = _tile_base(indicator136)
    if base < 27:
        suit, rank = divmod(base, 9)
        return suit * 9 + ((rank + 1) % 9)
    return 27 + ((base - 27 + 1) % 7)


def _discard_efficiency_score(hand: list[int], discard_tile: int, dora_bases: set[int]) -> float:
    """Small transparent hand-efficiency heuristic; native legality remains authoritative."""
    base = _tile_base(discard_tile)
    counts = [0] * 34
    for tile in hand:
        counts[_tile_base(tile)] += 1
    if counts[base] <= 0:
        raise ValueError("Candidate discard is not present in the active player's hand")
    counts[base] -= 1

    shape = 0.0
    for tile, count in enumerate(counts):
        if count >= 2:
            shape += 2.4 if tile < 27 else 1.8
        if count >= 3:
            shape += 1.2
        if tile >= 27 or count == 0:
            continue
        rank = tile % 9
        if rank < 8 and counts[tile + 1]:
            shape += 0.85
        if rank < 7 and counts[tile + 2]:
            shape += 0.45
        if rank > 0 and counts[tile - 1]:
            shape += 0.85
        if rank > 1 and counts[tile - 2]:
            shape += 0.45

    isolated = 0
    if base < 27:
        rank = base % 9
        neighbours = sum(
            counts[index]
            for index in (base - 2, base - 1, base + 1, base + 2)
            if 0 <= index < 27 and abs(index // 9 - base // 9) == 0
        )
        isolated = int(neighbours == 0)
    else:
        isolated = int(counts[base] == 0)

    dora_penalty = 4.0 if base in dora_bases else 0.0
    red_penalty = 1.25 if discard_tile in {16, 52, 88} else 0.0
    return shape - dora_penalty - red_penalty - 0.6 * isolated


def _choose_own_candidate(
    candidates: list[MappedAction], observation: Any, rng: random.Random
) -> MappedAction:
    wins = [candidate for candidate in candidates if candidate.spec.action_type == "TSUMO"]
    if wins:
        return wins[0]

    discards = [candidate for candidate in candidates if candidate.spec.action_type == "DISCARD"]
    if discards:
        hand = list(observation.hands[observation.player_id])
        dora_bases = {_dora_base(int(tile)) for tile in observation.dora_indicators}
        scored = [
            (
                _discard_efficiency_score(hand, int(candidate.actions[-1].tile), dora_bases)
                + (0.35 if candidate.spec.riichi and observation.is_tenpai else 0.0),
                candidate,
            )
            for candidate in discards
        ]
        best = max(score for score, _ in scored)
        tied = [candidate for score, candidate in scored if math.isclose(score, best, abs_tol=1e-9)]
        return rng.choice(tied)

    usable = [candidate for candidate in candidates if not candidate.native_execution_ambiguous]
    if not usable:
        raise RuntimeError("Native active player has no unambiguous executable action")
    return rng.choice(usable)


def _continue_baseline_native(
    env: RiichiEnv, rng: random.Random, *, max_decisions: int
) -> tuple[bool, int]:
    """Continue with a legal, deliberately simple riichi-aware baseline policy."""
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
                choices[seat] = _choose_response_action(legal, rng)
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
            selected = _choose_own_candidate(usable, observation, rng)
            execute_candidate(env, seat, selected)
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
        complete, continuation_decisions = _continue_baseline_native(
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

    This branch generator uses a versioned riichi-aware native-legal baseline continuation
    policy. Outcomes estimate candidate values conditional on that fixed baseline,
    not policy-independent counterfactual ground truth.
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
