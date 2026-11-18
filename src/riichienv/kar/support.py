from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import content_hash, game_split
from .probability import masked_softmax
from .runtime_contracts import SUPPORT_VERSION, UTILITIES, RuntimeBinding, runtime_contract, sha256_value
from .versioning import validate_artifact_contract


@dataclass(frozen=True)
class EventSupport:
    train_manifest_sha256: str
    train_game_counts: tuple[int, ...]
    source_reference: str

    def __post_init__(self) -> None:
        sha256_value(self.train_manifest_sha256, "train_manifest_sha256")
        if len(self.train_game_counts) != 21 or any(
            type(value) is not int or value < 0 for value in self.train_game_counts
        ):
            raise ValueError("Event support requires twenty-one independent train game counts")
        if not self.source_reference.strip():
            raise ValueError("Event support requires explicit evidence provenance")

    @property
    def supported(self) -> tuple[bool, ...]:
        """Train-count support only; never a claim of branch calibration."""
        return tuple(count >= 500 for count in self.train_game_counts)

    def value_supported(self, branch_statuses: Sequence[str] | None) -> tuple[bool, ...]:
        states = tuple(branch_statuses) if branch_statuses is not None else ("unavailable",) * 21
        if len(states) != 21 or any(state not in {"passed", "failed", "unavailable"} for state in states):
            raise ValueError("Value support needs twenty-one explicit branch quality states")
        return tuple(trained and state == "passed" for trained, state in zip(self.supported, states))

    def to_record(self) -> dict[str, Any]:
        document = {
            "version": SUPPORT_VERSION,
            "contract": runtime_contract(),
            "train_manifest_sha256": self.train_manifest_sha256,
            "train_game_counts": list(self.train_game_counts),
            "counting_unit": "distinct_complete_train_game_id",
            "support_threshold": 500,
            "value_gate": "train_count_and_preregistered_branch_quality",
            "source_reference": self.source_reference,
            "uniform_error_certificate": False,
        }
        return {**document, "support_sha256": content_hash(document)}

    @property
    def digest(self) -> str:
        return self.to_record()["support_sha256"]

    @classmethod
    def from_record(cls, document: Mapping[str, Any], binding: RuntimeBinding) -> EventSupport:
        unsigned = {key: value for key, value in document.items() if key != "support_sha256"}
        if (
            document.get("support_sha256") != content_hash(unsigned)
            or document.get("contract") != runtime_contract()
            or document.get("version") != SUPPORT_VERSION
        ):
            raise ValueError("Event support integrity/version/contract mismatch")
        if (
            document.get("support_threshold") != 500
            or document.get("counting_unit") != "distinct_complete_train_game_id"
            or document.get("uniform_error_certificate") is not False
        ):
            raise ValueError("Support counts are a train-only engineering gate, not an error certificate")
        if document["train_manifest_sha256"] != binding.train_manifest_sha256:
            raise ValueError("Support data differs from the model training manifest")
        result = cls(
            document["train_manifest_sha256"], tuple(document["train_game_counts"]), document["source_reference"]
        )
        if result.to_record() != dict(document):
            raise ValueError("Unexpected event-support fields")
        return result


def support_from_train_games(
    train_manifest: Mapping[str, Any], observed_events: Mapping[str, Sequence[int]], *, source_reference: str
) -> EventSupport:
    digest = train_manifest.get("manifest_sha256")
    if not isinstance(digest, str) or digest != content_hash(train_manifest["games"]):
        raise ValueError("Training-selection manifest integrity mismatch")
    scope = train_manifest["scope"]
    validate_artifact_contract(scope, context="training support selection")
    current = runtime_contract()
    if any(scope.get(key) != current[key] for key in ("schema_hash", "ruleset_hash", "action_manifest_hash")):
        raise ValueError("Training selection uses another public protocol")
    rows = train_manifest["games"]
    identities = [row["game_id"] for row in rows]
    salt = scope["split_salt"]
    if len(set(identities)) != len(identities) or any(game_split(game, salt) != "train" for game in identities):
        raise ValueError("Support may count distinct train games only")
    if any(row.get("source_role") != "main_cohort_candidate" for row in rows):
        raise ValueError("Golden/foreign games cannot increase training support")
    if not set(observed_events) <= set(identities):
        raise ValueError("Validation/test/unknown games cannot fill a 499-game support gap")
    counts = [0] * 21
    for events in observed_events.values():
        if any(type(event) is not int or not 0 <= event < 21 for event in events):
            raise ValueError("Unknown response event in training evidence")
        for event in set(events):
            counts[event] += 1
    return EventSupport(digest, tuple(counts), source_reference)


@dataclass(frozen=True)
class CandidateValue:
    action_id: int
    response_probabilities: tuple[float, ...]
    conditional_utilities: tuple[float, ...]
    marginal_value: float
    conservative_value: float
    unsupported_mass: float
    rule_allowed: tuple[bool, ...]
    train_supported: tuple[bool, ...]
    branch_quality_statuses: tuple[str, ...]
    value_supported: tuple[bool, ...]


def candidate_value(
    action_id: int,
    response_logits: Sequence[float],
    rank_logits: Sequence[Sequence[float]],
    allowed: Sequence[bool],
    support: EventSupport,
    *,
    response_temperature: float = 1.0,
    rank_temperature: float = 1.0,
    branch_quality_statuses: Sequence[str] | None = None,
) -> CandidateValue:
    if type(action_id) is not int or not 0 <= action_id < 148:
        raise ValueError("Only complete discard actions have response/rank value semantics")
    if len(allowed) != 21 or len(rank_logits) != 21:
        raise ValueError("Expected twenty-one response branches")
    probabilities = masked_softmax(response_logits, allowed, response_temperature)
    conditional = tuple(
        math.fsum(
            probability * utility
            for probability, utility in zip(masked_softmax(row, (True,) * 4, rank_temperature), UTILITIES)
        )
        for row in rank_logits
    )
    marginal = math.fsum(probability * utility for probability, utility in zip(probabilities, conditional))
    states = tuple(branch_quality_statuses) if branch_quality_statuses is not None else ("unavailable",) * 21
    covered = support.value_supported(states)
    conservative = math.fsum(
        probability * (utility if supported else -1.5)
        for probability, utility, supported in zip(probabilities, conditional, covered)
    )
    unsupported = math.fsum(probability for probability, supported in zip(probabilities, covered) if not supported)
    return CandidateValue(action_id, probabilities, conditional, marginal, conservative, unsupported,
                          tuple(allowed), support.supported, states, covered)
