from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

from .decision import ReplayDecision
from .observation import PublicSnapshot
from .schema import ACTIVE_SCHEMA_ID, schema_status
from .tokenize import TokenHistoryCache, TokenizationUnavailableError, append_action, encode_prefix


class InputAudit:
    def __init__(self, on_decision: Callable[[ReplayDecision], None] | None = None) -> None:
        self.on_decision = on_decision
        self.trigger_event_id: int | None = None
        self.caches = [TokenHistoryCache() for _ in range(4)]
        self.prefix_hash = hashlib.sha256()
        self.conditional_hash = hashlib.sha256()
        self.decisions = 0
        self.bytes_checked = 0
        self.gaps: list[dict[str, Any]] = []
        self.phases: dict[str, int] = {}

    def inspect(
        self, snapshot: PublicSnapshot, candidates: Sequence[int], target: int, *, provenance: str = "explicit"
    ) -> None:
        try:
            prefix = encode_prefix(snapshot, candidates, history_cache=self.caches[snapshot.observer_player_id])
            conditional = append_action(prefix, target)
            prefix_bytes, conditional_bytes = prefix.to_bytes(), conditional.to_bytes()
        except TokenizationUnavailableError as exc:
            self.gaps.append(
                {
                    "cutoff_event_id": snapshot.cutoff_event_id,
                    "observer_player_id": snapshot.observer_player_id,
                    "reason": str(exc),
                }
            )
            return
        self.prefix_hash.update(prefix_bytes)
        self.conditional_hash.update(conditional_bytes)
        self.decisions += 1
        self.bytes_checked += len(prefix_bytes) + len(conditional_bytes)
        self.phases[snapshot.decision_phase] = self.phases.get(snapshot.decision_phase, 0) + 1
        if self.on_decision is not None:
            if self.trigger_event_id is None:
                raise ValueError("Training export requires an explicit source trigger event")
            self.on_decision(ReplayDecision(snapshot, prefix, conditional, target, self.trigger_event_id, provenance))

    def report(self) -> dict[str, Any]:
        registry = schema_status()
        return {
            "status": "FAIL"
            if registry["status"] != "PASS"
            else "BLOCKED"
            if self.gaps or not self.decisions
            else "PASS",
            "schema_id": ACTIVE_SCHEMA_ID,
            "schema_hash": registry.get("schema_hash"),
            "decisions_checked": self.decisions,
            "bytes_checked": self.bytes_checked,
            "prefix_sha256": self.prefix_hash.hexdigest(),
            "conditional_input_sha256": self.conditional_hash.hexdigest(),
            "phases": self.phases,
            "gaps": self.gaps,
            "labels_in_prefix": False,
        }
