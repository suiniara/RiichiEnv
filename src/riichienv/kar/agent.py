from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from riichienv import Observation, Phase

from .actions import MappedAction, execute_candidate, execute_joint_responses, legal_candidates
from .contracts import MJSOUL_RULESET, content_hash
from .live_events import LIVE_EVENT_VERSION, LiveEventProjection
from .observation import PublicSnapshot
from .policy import BoundEvidence, PolicyOptions, PolicyResult
from .runtime import RuntimePolicy
from .runtime_contracts import RuntimeBinding


class StaticLogitPredictor:
    """Fixed, model-free integration fixture; never claims trained policy quality."""

    def __init__(self):
        self.logits = tuple(
            8.0 if action in {219, 543, 544, 545} else 2.0 if action == 221 else 0.0 for action in range(546)
        )
        self.binding = RuntimeBinding.current(
            content_hash(self.logits), hashlib.sha256(b"static-functional-fixture-no-training-data").hexdigest()
        )
        self.candidate_call_count = 0

    def policy(self, snapshot: PublicSnapshot, candidates: tuple[int, ...], context: str) -> tuple[float, ...]:
        return self.logits

    def candidates(
        self, snapshot: PublicSnapshot, candidates: tuple[int, ...], supported_discards: tuple[int, ...], context: str
    ) -> dict[int, tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]]:
        self.candidate_call_count += 1
        return {action: ((0.0,) * 21, ((0.0,) * 4,) * 21) for action in supported_discards}


class KARAgent:
    def __init__(self, seat: int, policy: RuntimePolicy):
        if type(seat) is not int or not 0 <= seat < 4:
            raise ValueError("KAR requires a stable four-player observer ID")
        self.seat, self.policy = seat, policy
        self.decisions = 0
        self.last_result: PolicyResult | None = None

    def choose(
        self,
        observation: Observation,
        snapshot: PublicSnapshot,
        round_instance: int,
        *,
        bounds: BoundEvidence | None = None,
    ) -> tuple[MappedAction, PolicyResult]:
        if observation.player_id != self.seat or snapshot.observer_player_id != self.seat:
            raise ValueError("Agent cannot act for another observer")
        source = snapshot.pending_kan_actor if snapshot.pending_kan_actor is not None else snapshot.last_discard_actor
        mapped = legal_candidates(observation, source_player=source)
        if any(candidate.native_execution_ambiguous for candidate in mapped):
            raise ValueError("A live agent cannot hide an ambiguous native action by filtering its probability")
        context = f"live/round{round_instance}/seat{self.seat}/event{snapshot.cutoff_event_id}/decision{self.decisions}"
        result = self.policy.decide(snapshot, [candidate.global_id for candidate in mapped], context, bounds=bounds)
        selected = next((candidate for candidate in mapped if candidate.global_id == result.action_id), None)
        if selected is None:
            raise ValueError("Runtime sampled an action outside the complete legal set")
        self.decisions += 1
        self.last_result = result
        return selected, result


def run_match(agents: dict[int, KARAgent], *, wall_seed: int, max_decisions: int = 10000) -> dict[str, Any]:  # noqa: PLR0915
    if set(agents) != set(range(4)) or any(agent.seat != seat for seat, agent in agents.items()):
        raise ValueError("Match needs four explicitly assigned agents")
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("Match decision cap must be positive")
    if type(wall_seed) is not int or not 0 <= wall_seed < 2**64:
        raise ValueError("A frozen functional match requires an explicit unsigned wall seed")
    env = MJSOUL_RULESET.create_env(seed=wall_seed)
    observations = env.reset()
    projection = LiveEventProjection()
    projection.observe_native(list(env.rule_log))
    decisions = []
    started = time.perf_counter()
    while not env.done():
        if len(decisions) >= max_decisions:
            return {
                "status": "BLOCKED",
                "reason": "decision_cap_reached_not_a_complete_match",
                "complete": False,
                "wall_seed": wall_seed,
                "decisions": decisions,
                "parameter_updates_executed": False,
            }
        choices = {}
        for seat in sorted(observations):
            observation = observations[seat]
            if not observation.legal_actions():
                continue
            snapshot = projection.snapshot(seat)
            projection.ledgers[seat].check_observation(observation)
            decision_started = time.perf_counter()
            selected, result = agents[seat].choose(observation, snapshot, projection.round_instance)
            decision_latency = time.perf_counter() - decision_started
            choices[seat] = selected
            decisions.append(
                {
                    "seat": seat,
                    "round_instance": projection.round_instance,
                    "cutoff_event_id": snapshot.cutoff_event_id,
                    "action_id": selected.global_id,
                    "evidence_status": result.evidence_status,
                    "fallback_reason": result.fallback_reason,
                    "telemetry": dict(result.telemetry),
                    "decision_latency_seconds": decision_latency,
                }
            )
        before = len(env.rule_log)
        projected_before = len(projection.events)
        pending_passes = projection.capture_self_passes(choices) if env.phase == Phase.WaitResponse else ()
        if env.phase == Phase.WaitResponse:
            if not choices:
                raise ValueError("Native response window has no observer choices")
            observations = dict(execute_joint_responses(env, choices))
        elif choices:
            if len(choices) != 1 or env.current_player not in choices:
                raise ValueError("Own-action phase must have one active observer")
            observations = dict(execute_candidate(env, env.current_player, choices[env.current_player]))
        elif env.needs_tsumo or env.needs_initialize_next_round:
            observations = env.step({})
        else:
            raise ValueError("Native game is stuck without a legal decision or automatic transition")
        new_events = list(env.rule_log[before:])
        projection.observe_native(
            new_events,
            confirmed_self_passes=pending_passes,
        )
        if len(projection.events) == projected_before:
            raise ValueError("Native execution did not advance a public event or validated kan transaction")
    if not projection.ended or projection.public_scores != tuple(env.scores()):
        raise ValueError("Projected public completion/scores differ from native match result")
    elapsed = time.perf_counter() - started
    report = {
        "status": "PASS",
        "complete": True,
        "scope": "fixed_policy_functional_match_not_strength_evaluation",
        "wall_seed": wall_seed,
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "live_event_version": LIVE_EVENT_VERSION,
        "scores": list(env.scores()),
        "ranks": list(env.ranks()),
        "rounds": projection.round_instance + 1,
        "decision_count": len(decisions),
        "event_count": len(projection.events),
        "elapsed_seconds": elapsed,
        "matches_per_second": 1 / elapsed if elapsed else None,
        "events": projection.events,
        "decisions": decisions,
        "parameter_updates_executed": False,
        "policy_rng_final_sha256": {
            str(seat): content_hash(agent.policy.rng.getstate()) for seat, agent in agents.items()
        },
    }
    logical = {key: value for key, value in report.items() if key not in {"elapsed_seconds", "matches_per_second"}}
    logical["decisions"] = [
        {key: value for key, value in row.items() if key != "decision_latency_seconds"} for row in decisions
    ]
    report["match_sha256"] = content_hash(logical)
    return report


def functional_match(seed: int = 17, *, max_decisions: int = 10000) -> dict[str, Any]:
    agents = {
        seat: KARAgent(seat, RuntimePolicy(StaticLogitPredictor(), options=PolicyOptions(), seed=seed * 4 + seat))
        for seat in range(4)
    }
    return run_match(agents, wall_seed=seed, max_decisions=max_decisions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed-logit, no-learning KAR functional match")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite existing functional match evidence")
    report = functional_match(args.seed, max_decisions=args.max_decisions)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"events", "decisions"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
