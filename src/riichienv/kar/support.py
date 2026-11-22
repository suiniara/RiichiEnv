from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import Response, content_hash, game_split
from .probability import masked_softmax
from .runtime_contracts import SUPPORT_VERSION, UTILITIES, RuntimeBinding, runtime_contract, sha256_value
from .versioning import validate_artifact_contract

CANDIDATE_ACTION_SUPPORT_MIN_SOURCES = 50


@dataclass(frozen=True)
class EventSupport:
    train_manifest_sha256: str
    train_game_counts: tuple[int, ...]
    source_reference: str
    candidate_branch_source_decisions: int = 0
    candidate_action_source_counts: tuple[int, ...] = (0,) * 148

    def __post_init__(self) -> None:
        sha256_value(self.train_manifest_sha256, "train_manifest_sha256")
        if len(self.train_game_counts) != 21 or any(
            type(value) is not int or value < 0 for value in self.train_game_counts
        ):
            raise ValueError("Event support requires twenty-one independent train game counts")
        if not self.source_reference.strip():
            raise ValueError("Event support requires explicit evidence provenance")
        if (
            type(self.candidate_branch_source_decisions) is not int
            or self.candidate_branch_source_decisions < 0
            or len(self.candidate_action_source_counts) != 148
            or any(type(value) is not int or value < 0 for value in self.candidate_action_source_counts)
        ):
            raise ValueError("Candidate branch support requires 148 nonnegative complete-source counts")
        if self.candidate_branch_source_decisions == 0 and any(self.candidate_action_source_counts):
            raise ValueError("Candidate action counts require complete branch source decisions")
        if any(value > self.candidate_branch_source_decisions for value in self.candidate_action_source_counts):
            raise ValueError("A candidate action cannot occur in more complete sources than exist")

    @property
    def supported(self) -> tuple[bool, ...]:
        """Train-count support only; never a claim of branch calibration."""
        return tuple(count >= 500 for count in self.train_game_counts)

    def candidate_supported(self, action_ids: Sequence[int]) -> bool:
        ids = tuple(action_ids)
        return bool(ids) and all(
            type(action) is int and 0 <= action < 148 and self.candidate_action_source_counts[action] >= CANDIDATE_ACTION_SUPPORT_MIN_SOURCES
            for action in ids
        )

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
            "candidate_branch_version": "kar_candidate_action_branch_v2",
            "candidate_branch_counting_unit": "distinct_complete_source_decision",
            "candidate_action_support_min_sources": CANDIDATE_ACTION_SUPPORT_MIN_SOURCES,
            "candidate_branch_source_decisions": self.candidate_branch_source_decisions,
            "candidate_action_source_counts": list(self.candidate_action_source_counts),
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
            or document.get("candidate_action_support_min_sources") != CANDIDATE_ACTION_SUPPORT_MIN_SOURCES
            or document.get("uniform_error_certificate") is not False
        ):
            raise ValueError("Support counts are a train-only engineering gate, not an error certificate")
        if document["train_manifest_sha256"] != binding.train_manifest_sha256:
            raise ValueError("Support data differs from the model training manifest")
        result = cls(
            document["train_manifest_sha256"],
            tuple(document["train_game_counts"]),
            document["source_reference"],
            document["candidate_branch_source_decisions"],
            tuple(document["candidate_action_source_counts"]),
        )
        if result.to_record() != dict(document):
            raise ValueError("Unexpected event-support fields")
        return result


def support_from_train_games(
    train_manifest: Mapping[str, Any],
    observed_events: Mapping[str, Sequence[int]],
    *,
    source_reference: str,
    dataset_manifest: Mapping[str, Any] | None = None,
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
    bound_digest = digest
    branch_source = train_manifest
    if dataset_manifest is not None:
        unsigned = {key: value for key, value in dataset_manifest.items() if key != "manifest_sha256"}
        ds_digest = dataset_manifest.get("manifest_sha256")
        if not isinstance(ds_digest, str) or ds_digest != content_hash(unsigned):
            raise ValueError("Exported dataset manifest integrity mismatch")
        ds_train = [row["game_id"] for row in dataset_manifest.get("games", ()) if row.get("split") == "train"]
        if set(ds_train) != set(identities):
            raise ValueError("Exported dataset train games differ from training-selection manifest")
        bound_digest = ds_digest
        branch_source = dataset_manifest
    branch_counts = branch_source.get("candidate_branch_counts", {})
    complete_by_split = branch_counts.get("complete_source_decisions_by_split", {})
    actions_by_split = branch_counts.get("candidate_action_source_counts_by_split", {})
    complete_sources = complete_by_split.get("train", 0)
    action_counts_record = actions_by_split.get("train", {})
    if type(complete_sources) is not int or complete_sources < 0 or not isinstance(action_counts_record, Mapping):
        raise ValueError("Candidate branch coverage evidence is missing or malformed")
    action_counts = [0] * 148
    for key, value in action_counts_record.items():
        action = int(key)
        if not 0 <= action < 148 or type(value) is not int or value < 0:
            raise ValueError("Candidate action branch coverage contains invalid identities/counts")
        action_counts[action] = value
    return EventSupport(bound_digest, tuple(counts), source_reference, complete_sources, tuple(action_counts))


def unsupported_response_mass(probabilities: Sequence[float], supported: Sequence[bool]) -> float:
    """Sum already-masked response probabilities outside the supplied support set."""
    if len(probabilities) != 21 or len(supported) != 21:
        raise ValueError("Unsupported mass requires twenty-one response branches")
    return math.fsum(probability for probability, covered in zip(probabilities, supported) if not covered)


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


_NEUTRAL_DRAW_EVENTS = frozenset(
    {
        int(Response.EXHAUSTIVE_DRAW),
        int(Response.ABORT_FOUR_WINDS),
        int(Response.ABORT_FOUR_RIICHI),
        int(Response.ABORT_FOUR_KANS),
        int(Response.NAGASHI_MANGAN),
    }
)


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
    unsupported_fallback: str = "worst_case",
) -> CandidateValue:
    if type(action_id) is not int or not 0 <= action_id < 148:
        raise ValueError("Only complete discard actions have response/rank value semantics")
    if len(allowed) != 21 or len(rank_logits) != 21:
        raise ValueError("Expected twenty-one response branches")
    if unsupported_fallback not in {"worst_case", "neutral_draws"}:
        raise ValueError("Unsupported branch fallback must be 'worst_case' or 'neutral_draws'")
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
        probability
        * (
            utility
            if supported
            else (0.0 if unsupported_fallback == "neutral_draws" and event in _NEUTRAL_DRAW_EVENTS else -1.5)
        )
        for event, (probability, utility, supported) in enumerate(zip(probabilities, conditional, covered))
    )
    unsupported = unsupported_response_mass(probabilities, covered)
    return CandidateValue(
        action_id,
        probabilities,
        conditional,
        marginal,
        conservative,
        unsupported,
        tuple(allowed),
        support.supported,
        states,
        covered,
    )
