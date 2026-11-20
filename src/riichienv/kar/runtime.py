from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from .calibration import CalibrationArtifact
from .contracts import content_hash
from .event_semantics import PUBLIC_CLOCK_VERSION
from .observation import PublicSnapshot
from .policy import (
    BoundEvidence,
    PolicyOptions,
    PolicyResult,
    one_sided_bounds,
    restricted_kl,
    sample_complete,
    support_discards,
)
from .probability import complete_candidates, legal_prior
from .response_mask import public_rule_response_mask
from .runtime_contracts import RuntimeBinding
from .support import EventSupport, candidate_value


class LogitPredictor(Protocol):
    binding: RuntimeBinding

    def policy(self, snapshot: PublicSnapshot, candidates: tuple[int, ...], context: str) -> Sequence[float]: ...
    def candidates(
        self, snapshot: PublicSnapshot, candidates: tuple[int, ...], supported_discards: tuple[int, ...], context: str
    ) -> Mapping[int, tuple[Sequence[float], Sequence[Sequence[float]]]]: ...


class RuntimePolicy:
    def __init__(
        self,
        predictor: LogitPredictor,
        *,
        options: PolicyOptions | None = None,
        support: EventSupport | None = None,
        calibration: CalibrationArtifact | None = None,
        seed: int = 0,
        allow_diagnostic_calibration: bool = False,
    ):
        self.predictor, self.binding = predictor, predictor.binding
        self.options = options or PolicyOptions()
        self.support, self.calibration = support, calibration
        self.allow_diagnostic_calibration = allow_diagnostic_calibration
        self.rng = random.Random(seed)
        if support is not None and support.train_manifest_sha256 != self.binding.train_manifest_sha256:
            raise ValueError("Support/model training data mismatch")
        if calibration is not None:
            if support is None:
                raise ValueError("Calibrated inference requires bound event-support evidence")
            calibration.validate(self.binding, support, allow_diagnostic=allow_diagnostic_calibration)

    def decide(  # noqa: PLR0915
        self, snapshot: PublicSnapshot, candidates: Sequence[int], context: str, *, bounds: BoundEvidence | None = None
    ) -> PolicyResult:
        if not context or self.predictor.binding != self.binding:
            raise ValueError("Inference model identity changed or decision identity is empty")
        ids = complete_candidates(candidates)
        if (
            not snapshot.history_complete
            or not snapshot.visibility_complete
            or getattr(snapshot, "public_clock_version", None) != PUBLIC_CLOCK_VERSION
        ):
            raise ValueError("A runtime decision requires a complete visible prefix and current semantic clock")
        artifact = self.calibration
        branch_report = artifact.branch_quality_report if artifact is not None else None
        branch_states = (
            tuple(branch_report["branches"][str(event)]["status"] for event in range(21))
            if branch_report
            else ("unavailable",) * 21
        )
        temperatures = artifact.temperatures if artifact is not None else (1.0, 1.0, 1.0)
        logits = self.predictor.policy(snapshot, ids, context)
        prior, logs = legal_prior(logits, ids, temperatures[0])
        threshold = artifact.log_threshold if artifact is not None else None
        selected = support_discards(ids, logs, threshold)
        mass = math.fsum(prior[action] for action in selected)
        telemetry: dict[str, Any] = {
            "context_id": context,
            "binding_sha256": self.binding.digest,
            "candidate_ids": list(ids),
            "prior": {str(action): prior[action] for action in ids},
            "supported_discards": list(selected),
            "support_mass": mass,
            "temperatures": list(temperatures),
            "log_threshold": threshold,
            "calibration_status": "unfitted"
            if artifact is None
            else "diagnostic_fixture"
            if artifact.diagnostic_fixture
            else "heldout_artifact",
            "decision_mode": self.options.mode,
            "tau": "infinity" if math.isinf(self.options.tau) else self.options.tau,
            "max_prior_amplification": self.options.max_prior_amplification,
            "diagnostic_argmax": self.options.diagnostic_argmax,
            "calibration_quality_status": artifact.quality_report["status"] if artifact is not None else "unavailable",
            "branch_quality_report": branch_report,
            "candidate_forward_count": 0,
            "KL": 0.0,
            "model_gain": 0.0,
            "bound_E1": None,
            "bound_E2": None,
            "error_bound_source": None,
        }
        final, reason, status = prior, None, "certificate_unavailable"
        if self.options.mode == "prior_only":
            reason, status = "configured_prior_only", "unfitted_prior_only" if artifact is None else "prior_only"
        elif math.isinf(self.options.tau):
            reason, status = "infinite_tau_baseline", "unfitted_prior_only" if artifact is None else "prior_only"
        elif artifact is None or not artifact.gate_valid:
            reason = "calibration_or_support_gate_unavailable"
        elif not artifact.empirical_check_valid:
            reason = (
                "calibration_quality_failed"
                if artifact.quality_report["status"] == "failed"
                else "independent_empirical_check_unavailable"
            )
        elif self.support is None:
            reason = "event_support_manifest_unavailable"
        elif not any(self.support.value_supported(branch_states)):
            reason = "no_quality_qualified_response_branches"
        elif len(selected) < 2 or mass == 0:
            reason = "fewer_than_two_supported_discards_or_zero_mass"
        elif not self.support.candidate_supported(selected) and not (
            artifact.diagnostic_fixture and self.allow_diagnostic_calibration
        ):
            reason = "candidate_action_branch_supervision_unavailable"
        elif self.options.mode == "certified" and bounds is None:
            reason = "error_bound_certificate_unavailable"
        else:
            valid_bounds = True
            if self.options.mode == "certified":
                assert bounds is not None
                try:
                    bounds.validate(self.binding, context, selected)
                except ValueError as error:
                    valid_bounds, reason = False, "error_bound_coverage_invalid"
                    telemetry["error_bound_rejection"] = str(error)
            if valid_bounds:
                raw_values = self.predictor.candidates(snapshot, ids, selected, context)
                if set(raw_values) != set(selected):
                    raise ValueError("Candidate predictor must score every S action exactly once, without Top-K")
                scores, rho, details = [], [], {}
                for action in selected:
                    response, ranks = raw_values[action]
                    allowed = public_rule_response_mask(snapshot, action)
                    value = candidate_value(
                        action,
                        response,
                        ranks,
                        allowed,
                        self.support,
                        response_temperature=temperatures[1],
                        rank_temperature=temperatures[2],
                        branch_quality_statuses=branch_states,
                    )
                    scores.append(value.conservative_value)
                    rho.append(value.unsupported_mass)
                    details[str(action)] = asdict(value)
                update = restricted_kl(
                    prior,
                    selected,
                    scores,
                    self.options.tau,
                    max_prior_amplification=self.options.max_prior_amplification,
                )
                telemetry.update(
                    candidate_forward_count=len(selected),
                    candidates=details,
                    KL=update.kl,
                    model_gain=update.model_gain,
                    q0=list(update.q0),
                    qtau=list(update.qtau),
                )
                if self.options.mode == "certified":
                    assert bounds is not None
                    e1, e2 = one_sided_bounds(update, rho, bounds.b, bounds.delta_pi, self.options.tau)
                    telemetry.update(
                        bound_E1=e1,
                        bound_E2=e2,
                        error_bound_source=bounds.source_reference,
                        bound_coverage_statement=bounds.coverage_statement,
                        bound_confidence_statement=bounds.confidence_statement,
                        guarantee_scope="conditional_on_the_callers_valid_error_bounds",
                    )
                    if e2 > 0:
                        final, status = update.probabilities, "certified"
                    else:
                        reason, status = "nonpositive_conditional_E2_bound", "certified_bound_fallback"
                else:
                    final, status = update.probabilities, "empirical_only"
                    telemetry["guarantee_scope"] = "no_mathematical_safety_certificate"
        telemetry["distribution_changed"] = any(final[item] != prior[item] for item in ids)
        telemetry["distribution_change_comparison"] = "actual_float_probabilities_not_a_gain_certificate"
        telemetry["rng_before_sha256"] = content_hash(self.rng.getstate())
        action = sample_complete(final, ids, self.rng, diagnostic_argmax=self.options.diagnostic_argmax)
        telemetry.update(
            rng_after_sha256=content_hash(self.rng.getstate()),
            final={str(item): final[item] for item in ids},
            action_id=action,
            evidence_status=status,
            fallback_reason=reason,
        )
        return PolicyResult(final, action, status, reason, telemetry)
