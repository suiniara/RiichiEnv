from __future__ import annotations

import math
from collections.abc import Sequence

from .contracts import MAX_CANDIDATES


def finite_vector(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if len(values) != size or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values
    ):
        raise ValueError(f"{name} must contain {size} finite real values")
    return tuple(float(value) for value in values)


def complete_candidates(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if (
        not 1 <= len(result) <= MAX_CANDIDATES
        or len(set(result)) != len(result)
        or any(type(value) is not int or not 0 <= value < 546 for value in result)
    ):
        raise ValueError("Expected one to thirty-two distinct complete global action IDs")
    return tuple(sorted(result))


def masked_log_softmax(logits: Sequence[float], allowed: Sequence[bool], temperature: float = 1.0) -> tuple[float, ...]:
    values = finite_vector(logits, len(allowed), "logits")
    if not allowed or any(type(value) is not bool for value in allowed) or not any(allowed):
        raise ValueError("A boolean probability mask must allow at least one class")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Temperature must be finite and strictly positive")
    maximum = max(value for value, valid in zip(values, allowed) if valid)
    centered = tuple((value - maximum) / temperature if valid else -math.inf for value, valid in zip(values, allowed))
    if any(valid and not math.isfinite(value) for value, valid in zip(centered, allowed)):
        raise ValueError("Finite logits/temperature exceeded stable numeric range")
    log_total = math.log(math.fsum(math.exp(value) for value, valid in zip(centered, allowed) if valid))
    return tuple(value - log_total if valid else -math.inf for value, valid in zip(centered, allowed))


def masked_softmax(logits: Sequence[float], allowed: Sequence[bool], temperature: float = 1.0) -> tuple[float, ...]:
    return tuple(
        math.exp(value) if valid else 0.0
        for value, valid in zip(masked_log_softmax(logits, allowed, temperature), allowed)
    )


def legal_prior(
    logits: Sequence[float], candidates: Sequence[int], temperature: float = 1.0
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    ids = complete_candidates(candidates)
    finite_vector(logits, 546, "policy logits")
    permitted = tuple(index in ids for index in range(546))
    logs = masked_log_softmax(logits, permitted, temperature)
    return tuple(math.exp(value) if allowed else 0.0 for value, allowed in zip(logs, permitted)), logs


def probability_vector(values: Sequence[float], size: int, name: str = "probabilities") -> tuple[float, ...]:
    result = finite_vector(values, size, name)
    if any(value < 0 or value > 1 for value in result) or not math.isclose(
        math.fsum(result), 1.0, rel_tol=0.0, abs_tol=2e-12
    ):
        raise ValueError(f"{name} must be nonnegative and sum to one")
    return result
