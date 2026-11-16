from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .probability import complete_candidates, finite_vector, probability_vector
from .runtime_contracts import BOUND_VERSION, RuntimeBinding


def support_discards(
    candidates: Sequence[int], log_prior: Sequence[float], log_threshold: float | None
) -> tuple[int, ...]:
    ids = complete_candidates(candidates)
    if len(log_prior) != 546:
        raise ValueError("Log prior requires all global action identities")
    if log_threshold is None:
        return ()
    if isinstance(log_threshold, bool) or not math.isfinite(log_threshold) or log_threshold > 0:
        raise ValueError("The gate requires an authoritative finite nonpositive log threshold")
    if any(not math.isfinite(log_prior[action]) or log_prior[action] > 1e-12 for action in ids):
        raise ValueError("Legal action log probabilities must be finite and nonpositive")
    return tuple(action for action in ids if action < 148 and log_prior[action] >= log_threshold)


def tau_minimum(max_prior_amplification: float) -> float:
    if (
        isinstance(max_prior_amplification, bool)
        or not math.isfinite(max_prior_amplification)
        or max_prior_amplification <= 1
    ):
        raise ValueError("R_max must be finite and greater than one")
    return 3 / math.log(max_prior_amplification)


def fixed_tau_grid(
    values: Sequence[float] = (math.inf, 1.0, 0.5, 0.25), *, max_prior_amplification: float | None = None
) -> tuple[float, ...]:
    if not values or len(set(values)) != len(values) or not any(value == math.inf for value in values):
        raise ValueError("A fixed tau grid must contain distinct positive temperatures and the infinity baseline")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value) or value <= 0:
            raise ValueError("Tau grid entries must be positive temperatures or infinity")
        if max_prior_amplification is not None and value < tau_minimum(max_prior_amplification):
            raise ValueError("A grid temperature violates the declared R_max")
    return tuple(sorted((float(value) for value in values), reverse=True))


@dataclass(frozen=True)
class RestrictedUpdate:
    probabilities: tuple[float, ...]
    actions: tuple[int, ...]
    mass: float
    q0: tuple[float, ...]
    qtau: tuple[float, ...]
    kl: float
    model_gain: float


def restricted_kl(
    prior: Sequence[float],
    actions: Sequence[int],
    values: Sequence[float],
    tau: float,
    *,
    max_prior_amplification: float | None = None,
) -> RestrictedUpdate:
    base = probability_vector(prior, 546, "complete prior")
    ids = tuple(actions)
    if (
        len(ids) > 32
        or len(set(ids)) != len(ids)
        or any(type(action) is not int or not 0 <= action < 148 for action in ids)
    ):
        raise ValueError("Restricted value updates require distinct complete discard IDs")
    scores = finite_vector(values, len(ids), "candidate values")
    if any(value < -1.5 - 1e-12 or value > 1.5 + 1e-12 for value in scores):
        raise ValueError("Average-rank candidate utility lies in [-1.5,1.5]")
    if isinstance(tau, bool) or math.isnan(tau) or tau <= 0:
        raise ValueError("tau must be positive or infinity")
    if max_prior_amplification is not None and tau < tau_minimum(max_prior_amplification):
        raise ValueError("tau violates the declared prior amplification limit")
    mass = math.fsum(base[action] for action in ids)
    q0 = tuple(base[action] / mass for action in ids) if mass > 0 else ()
    if len(ids) < 2 or mass == 0 or math.isinf(tau):
        return RestrictedUpdate(base, ids, mass, q0, q0, 0.0, 0.0)
    maximum = max(score for score, probability in zip(scores, q0) if probability > 0)
    weights = tuple(
        math.log(probability) + (score - maximum) / tau if probability > 0 else -math.inf
        for score, probability in zip(scores, q0)
    )
    offset = max(weights)
    exponentials = tuple(math.exp(value - offset) for value in weights)
    total = math.fsum(exponentials)
    q = tuple(value / total for value in exponentials)
    result = list(base)
    for action, probability in zip(ids, q):
        result[action] = mass * probability
    # Only support entries were written. Correct residual floating-point mass inside S.
    residual = mass - math.fsum(result[action] for action in ids)
    largest = ids[max(range(len(ids)), key=lambda index: q[index])]
    result[largest] += residual
    q = tuple(result[action] / mass for action in ids)
    kl = math.fsum(
        probability * (math.log(probability) - math.log(original))
        for probability, original in zip(q, q0)
        if probability > 0
    )
    if kl < -1e-12:
        raise ArithmeticError("KL cannot be negative")
    kl = max(0.0, kl)
    gain = math.fsum((current - original) * score for current, original, score in zip(q, q0, scores))
    probability_vector(result, 546, "restricted complete policy")
    return RestrictedUpdate(tuple(result), ids, mass, q0, q, kl, gain)


@dataclass(frozen=True)
class BoundEvidence:
    binding_sha256: str
    decision_context_id: str
    action_ids: tuple[int, ...]
    b: tuple[float, ...]
    delta_pi: float
    source_reference: str
    coverage_statement: str
    confidence_statement: str
    version: str = BOUND_VERSION

    def validate(self, binding: RuntimeBinding, context: str, ids: tuple[int, ...]) -> None:
        finite_vector(self.b, len(ids), "candidate error upper bounds")
        if (
            self.version != BOUND_VERSION
            or self.binding_sha256 != binding.digest
            or self.decision_context_id != context
            or self.action_ids != ids
        ):
            raise ValueError("Error-bound evidence does not cover this model/decision/candidate set")
        if (
            any(value < 0 for value in self.b)
            or isinstance(self.delta_pi, bool)
            or not math.isfinite(self.delta_pi)
            or not 0 <= self.delta_pi <= 1
        ):
            raise ValueError("Error bounds must be nonnegative and delta_pi in [0,1]")
        if not all(
            value.strip() for value in (self.source_reference, self.coverage_statement, self.confidence_statement)
        ):
            raise ValueError("Certification requires explicit source,coverage and confidence statements")


def one_sided_bounds(
    update: RestrictedUpdate, unsupported_mass: Sequence[float], b: Sequence[float], delta_pi: float, tau: float
) -> tuple[float, float]:
    if len(update.actions) < 2 or update.mass <= 0 or not math.isfinite(tau) or tau <= 0:
        raise ValueError("Finite-error bounds require a nonempty finite-tau update")
    errors = finite_vector(b, len(update.actions), "candidate error bounds")
    rho = finite_vector(unsupported_mass, len(update.actions), "unsupported mass")
    if (
        any(value < 0 for value in errors)
        or any(not 0 <= value <= 1 + 1e-12 for value in rho)
        or not math.isfinite(delta_pi)
        or not 0 <= delta_pi <= 1
    ):
        raise ValueError("Invalid error or unsupported-mass bounds")
    e1 = (
        update.mass
        * (
            tau * update.kl
            - math.fsum((new + old) * error for new, old, error in zip(update.qtau, update.q0, errors))
            - 3 * math.fsum(old * value for old, value in zip(update.q0, rho))
        )
        - 3 * delta_pi
    )
    e2 = (
        update.mass
        * (
            update.model_gain
            - math.fsum(abs(new - old) * error for new, old, error in zip(update.qtau, update.q0, errors))
            - 3 * math.fsum(max(0.0, old - new) * value for old, new, value in zip(update.q0, update.qtau, rho))
        )
        - 3 * delta_pi
    )
    return e1, e2


@dataclass(frozen=True)
class PolicyOptions:
    mode: str = "prior_only"
    tau: float = math.inf
    max_prior_amplification: float | None = None
    diagnostic_argmax: bool = False

    def __post_init__(self) -> None:
        if (
            self.mode not in {"prior_only", "empirical_only", "certified"}
            or isinstance(self.tau, bool)
            or math.isnan(self.tau)
            or self.tau <= 0
        ):
            raise ValueError("Invalid policy evidence mode or tau")
        if type(self.diagnostic_argmax) is not bool:
            raise ValueError("Diagnostic action selection must be explicit boolean metadata")
        if self.mode == "certified" and self.diagnostic_argmax:
            raise ValueError("Argmax is a different policy and cannot inherit a stochastic-policy certificate")
        if self.max_prior_amplification is not None and self.tau < tau_minimum(self.max_prior_amplification):
            raise ValueError("Policy tau violates R_max")


@dataclass(frozen=True)
class PolicyResult:
    probabilities: tuple[float, ...]
    action_id: int
    evidence_status: str
    fallback_reason: str | None
    telemetry: Mapping[str, Any] = field(default_factory=dict)


def sample_complete(
    probabilities: Sequence[float], candidates: Sequence[int], rng: random.Random, *, diagnostic_argmax: bool = False
) -> int:
    values = probability_vector(probabilities, 546)
    ids = complete_candidates(candidates)
    if any(values[action] != 0 for action in range(546) if action not in ids):
        raise ValueError("Illegal actions must have exactly zero sampling probability")
    if diagnostic_argmax:
        return max(ids, key=lambda action: (values[action], -action))
    draw, cumulative = rng.random(), 0.0
    for action in ids:
        cumulative += values[action]
        if draw < cumulative:
            return action
    return next(action for action in reversed(ids) if values[action] > 0)
