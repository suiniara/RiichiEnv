from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import chain
from types import MappingProxyType
from typing import Any

from .calibration_quality import CalibrationQualityPolicy, evaluate_branch_quality, evaluate_quality
from .calibration_roles import CALIBRATION_SPLIT_SALT, calibration_is_fit
from .contracts import canonical_bytes, content_hash, game_split
from .policy import fixed_tau_grid, support_discards
from .probability import complete_candidates, finite_vector, legal_prior, masked_log_softmax
from .runtime_contracts import CALIBRATION_VERSION, RuntimeBinding, runtime_contract, sha256_value
from .selection_quality import TAU_SELECTION_VERSION, TauSelectionPolicy, paired_tau_evidence
from .statistics import ClassificationMetrics, FullDiagnostics
from .support import EventSupport
from .versioning import validate_artifact_contract


@dataclass(frozen=True)
class CalibrationSplit:
    validation_games: tuple[str, ...]
    fit_games: tuple[str, ...]
    check_games: tuple[str, ...]
    salt: str
    validation_manifest_sha256: str | None = None
    main_split_salt: str | None = None

    def validate(self) -> None:
        if self.validation_manifest_sha256 is not None:
            sha256_value(self.validation_manifest_sha256, "validation_manifest_sha256")
            if self.salt != CALIBRATION_SPLIT_SALT:
                raise ValueError("Production calibration cannot change the role split after training selection")
            if self.main_split_salt != "kar-riichienv-v1-games" or any(
                game_split(game, self.main_split_salt) != "validation" for game in self.validation_games
            ):
                raise ValueError("Production calibration requires the actual frozen validation split")
        elif self.main_split_salt is not None:
            raise ValueError("Split provenance needs the source manifest hash")
        values = (self.validation_games, self.fit_games, self.check_games)
        if not self.salt or any(
            len(set(items)) != len(items) or any(not isinstance(item, str) or not item for item in items)
            for items in values
        ):
            raise ValueError("Calibration split needs distinct complete game IDs and salt")
        fit_set, check_set = set(self.fit_games), set(self.check_games)
        if fit_set & check_set or fit_set | check_set != set(self.validation_games):
            raise ValueError("cal_fit/cal_check must partition validation without overlap")
        for game in self.validation_games:
            if calibration_is_fit(game, salt=self.salt) != (game in fit_set):
                raise ValueError("Calibration membership differs from the frozen game hash")


def calibration_split(
    validation_games: Sequence[str],
    *,
    train_games: Sequence[str] = (),
    test_games: Sequence[str] = (),
    salt: str = CALIBRATION_SPLIT_SALT,
) -> CalibrationSplit:
    games = tuple(sorted(validation_games))
    if set(games) & (set(train_games) | set(test_games)):
        raise ValueError("Train/test games cannot enter validation calibration")
    fit = tuple(game for game in games if calibration_is_fit(game, salt=salt))
    fit_set = set(fit)
    result = CalibrationSplit(games, fit, tuple(game for game in games if game not in fit_set), salt)
    result.validate()
    return result


def calibration_split_from_manifest(
    document: Mapping[str, Any], *, salt: str = CALIBRATION_SPLIT_SALT
) -> CalibrationSplit:
    digest = document.get("manifest_sha256")
    if not isinstance(digest, str) or digest != content_hash(document["games"]):
        raise ValueError("Validation selection manifest integrity mismatch")
    validate_artifact_contract(document["scope"], context="validation calibration selection")
    current = runtime_contract()
    if any(
        document["scope"].get(key) != current[key] for key in ("schema_hash", "ruleset_hash", "action_manifest_hash")
    ):
        raise ValueError("Validation calibration uses another public protocol")
    if any(row.get("source_role") != "main_cohort_candidate" for row in document["games"]):
        raise ValueError("Reserved Golden/foreign games cannot enter calibration")
    split = calibration_split([row["game_id"] for row in document["games"]], salt=salt)
    result = replace(split, validation_manifest_sha256=digest, main_split_salt=document["scope"]["split_salt"])
    result.validate()
    return result


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def choose_validation_tau(
    outcomes: Mapping[float, Mapping[str, float]],
    *,
    split: CalibrationSplit,
    binding: RuntimeBinding,
    grid: Sequence[float] = (math.inf, 1.0, 0.5, 0.25),
    max_prior_amplification: float | None = None,
    diagnostic_fixture: bool = False,
    policy_identity: Mapping[str, Any] | None = None,
    selection_policy: TauSelectionPolicy | None = None,
) -> dict[str, Any]:
    """Select a frozen validation-fit grid from supplied outcomes, never cal_check/test.

    This interface does not run matches, update weights, or turn validation wins into
    an error-bound certificate. A complete M3 paired-match scheduler is separate.
    """
    split.validate()
    if not diagnostic_fixture and split.validation_manifest_sha256 is None:
        raise ValueError("Production tau choice requires bound validation selection")
    if type(diagnostic_fixture) is not bool:
        raise ValueError("Tau-selection fixture provenance must be explicit")
    if not diagnostic_fixture and selection_policy is None:
        raise ValueError("Preregister paired validation-gain criteria before selecting a production tau")
    if not diagnostic_fixture and policy_identity is None:
        raise ValueError("Production tau selection must identify the calibrated policy, not only its weights")
    if policy_identity is not None:
        validate_policy_identity(policy_identity, binding)
        if (
            policy_identity["max_prior_amplification"] != max_prior_amplification
            or policy_identity["sampling"] != "stochastic"
            or policy_identity["mode"] not in {"empirical_only", "certified"}
        ):
            raise ValueError("Tau outcomes do not identify the registered stochastic empirical policy")
    temperatures = fixed_tau_grid(grid, max_prior_amplification=max_prior_amplification)
    if set(outcomes) != set(temperatures):
        raise ValueError("Score every predeclared grid entry including the infinity baseline")
    game_ids = tuple(sorted(outcomes[temperatures[0]]))
    if not game_ids or not set(game_ids) <= set(split.fit_games):
        raise ValueError("Tau selection cannot use cal_check, test or foreign games")
    means = {}
    for tau in temperatures:
        if tuple(sorted(outcomes[tau])) != game_ids:
            raise ValueError("Tau comparisons require the same validation games")
        utilities = finite_vector(
            tuple(outcomes[tau][game] for game in game_ids), len(game_ids), "validation utility outcomes"
        )
        if any(not -1.5 <= value <= 1.5 for value in utilities):
            raise ValueError("Tau units and outcomes must use frozen average-rank utility")
        means[tau] = math.fsum(utilities) / len(utilities)

    def key(tau: float) -> str:
        return "infinity" if math.isinf(tau) else str(tau)

    paired = paired_tau_evidence(outcomes, temperatures, selection_policy) if selection_policy is not None else None
    eligible = tuple(
        tau for tau in temperatures if math.isinf(tau) or paired is None or paired[key(tau)]["reproducible_gain"]
    )
    selected = max(eligible, key=lambda tau: (means[tau], tau))
    record = {
        "version": TAU_SELECTION_VERSION,
        "selection_policy": selection_policy.to_record() if selection_policy is not None else None,
        "paired_validation_evidence": paired,
        "reproducible_gain": not math.isinf(selected)
        and paired is not None
        and paired[key(selected)]["reproducible_gain"],
        "binding_sha256": binding.digest,
        "policy_identity": dict(policy_identity) if policy_identity is not None else None,
        "validation_manifest_sha256": split.validation_manifest_sha256,
        "calibration_split_salt": split.salt,
        "selection_partition": "cal_fit",
        "game_ids": list(game_ids),
        "grid": [key(tau) for tau in temperatures],
        "mean_utility": {key(tau): means[tau] for tau in temperatures},
        "selected_tau": key(selected),
        "max_prior_amplification": max_prior_amplification,
        "diagnostic_fixture": diagnostic_fixture,
        "evidence_status": "empirical_only_no_error_certificate",
    }
    return {**record, "tau_selection_sha256": content_hash(record)}


@dataclass(frozen=True)
class CalibrationRow:
    game_id: str
    row_id: str
    candidates: tuple[int, ...]
    policy_logits: tuple[float, ...]
    action_target: int
    action_valid: bool = True
    response_logits: tuple[float, ...] | None = None
    rank_logits: tuple[tuple[float, ...], ...] | None = None
    response_allowed: tuple[bool, ...] | None = None
    response_target: int = -1
    response_valid: bool = False
    rank_target: int = -1
    value_valid: bool = False
    threat_slice: str = "unspecified"
    current_rank: int | None = None

    def validate(self) -> None:
        if self.current_rank is not None and (type(self.current_rank) is not int or not 1 <= self.current_rank <= 4):
            raise ValueError("Current-rank diagnostic slice must be public rank 1..4 or unknown")
        ids = complete_candidates(self.candidates)
        finite_vector(self.policy_logits, 546, "calibration policy logits")
        if (
            not self.game_id
            or not self.row_id
            or any(type(value) is not bool for value in (self.action_valid, self.response_valid, self.value_valid))
        ):
            raise ValueError("Calibration row requires game/decision identity and boolean masks")
        if self.action_valid and (type(self.action_target) is not int or self.action_target not in ids):
            raise ValueError("Valid calibration action is illegal")
        if self.value_valid and not self.response_valid:
            raise ValueError("Value calibration requires a valid response")
        if self.response_valid:
            if (
                not self.action_valid
                or not 0 <= self.action_target < 148
                or self.response_logits is None
                or self.response_allowed is None
            ):
                raise ValueError("Response calibration only applies to actual complete discard actions")
            finite_vector(self.response_logits, 21, "response logits")
            if (
                len(self.response_allowed) != 21
                or any(type(value) is not bool for value in self.response_allowed)
                or not any(self.response_allowed)
            ):
                raise ValueError("Invalid public response mask")
            if (
                type(self.response_target) is not int
                or not 0 <= self.response_target < 21
                or not self.response_allowed[self.response_target]
            ):
                raise ValueError("Calibration target is excluded by the frozen public mask")
        if self.value_valid:
            if (
                self.rank_logits is None
                or len(self.rank_logits) != 21
                or type(self.rank_target) is not int
                or not 0 <= self.rank_target < 4
            ):
                raise ValueError("Invalid observed response/rank branch")
            for row in self.rank_logits:
                finite_vector(row, 4, "rank logits")


def _observed_log_probabilities(
    row: CalibrationRow, head: str, temperature: float
) -> tuple[tuple[float, ...], int] | None:
    if head == "action":
        return (
            (legal_prior(row.policy_logits, row.candidates, temperature)[1], row.action_target)
            if row.action_valid
            else None
        )
    if head == "response":
        if not row.response_valid:
            return None
        assert row.response_logits is not None and row.response_allowed is not None
        return masked_log_softmax(row.response_logits, row.response_allowed, temperature), row.response_target
    if head == "rank":
        if not row.value_valid:
            return None
        assert row.rank_logits is not None
        return masked_log_softmax(row.rank_logits[row.response_target], (True,) * 4, temperature), row.rank_target
    raise ValueError("Unknown calibration head")


def temperature_nll(rows: Sequence[CalibrationRow], head: str, temperature: float) -> dict[str, float | int | None]:
    # Grid scoring is a pure function of provided arrays; no network or optimizer is involved.
    masked_log_softmax((0.0,), (True,), temperature)
    if head not in {"action", "response", "rank"}:
        raise ValueError("Unknown calibration head")
    losses = []
    for row in rows:
        row.validate()
        observed = _observed_log_probabilities(row, head, temperature)
        if observed is not None:
            logs, target = observed
            losses.append(-logs[target])
    return {"count": len(losses), "nll": math.fsum(losses) / len(losses) if losses else None}


def select_static_temperature(
    rows: Sequence[CalibrationRow], head: str, temperatures: Sequence[float]
) -> tuple[float, dict[str, float | int | None]]:
    if not temperatures or len(set(temperatures)) != len(temperatures):
        raise ValueError("Use a nonempty fixed temperature grid")
    scored = [(value, temperature_nll(rows, head, value)) for value in temperatures]
    if not scored[0][1]["count"]:
        raise ValueError("Cannot fit a temperature without valid observations")
    best = min(scored, key=lambda item: (float(item[1]["nll"]), item[0]))
    return best


def gate_log_threshold(rows: Sequence[CalibrationRow], action_temperature: float) -> tuple[float | None, int]:
    logs = []
    for row in rows:
        row.validate()
        discards = tuple(action for action in row.candidates if action < 148)
        if row.action_valid and row.action_target < 148 and len(discards) >= 2:
            logs.append(legal_prior(row.policy_logits, row.candidates, action_temperature)[1][row.action_target])
    return (sorted(logs)[len(logs) // 100], len(logs)) if logs else (None, 0)


def calibration_metrics(
    rows: Sequence[CalibrationRow],
    temperatures: tuple[float, float, float],
    log_threshold: float | None,
    *,
    support: EventSupport | None = None,
    bootstrap_seed: int = 17,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    diagnostics = FullDiagnostics()
    branch_metrics = {head: {str(event): ClassificationMetrics(classes) for event in range(21)}
                      for head, classes in (("response", 21), ("rank", 4))}
    branch_games: dict[str, dict[str, set[str]]] = {
        head: {str(event): set() for event in range(21)} for head in branch_metrics
    }
    gate_rows = retained = small = 0
    mass_sum = 0.0
    events = {str(index): 0 for index in range(21)}
    threat: dict[str, dict[str, int]] = {}
    rank_slices: dict[str, dict[str, int]] = {}
    for row in rows:
        row.validate()
        slices = ("threat/" + row.threat_slice, "current_rank/" + str(row.current_rank or "unknown"))
        for head, temperature in zip(("action", "response", "rank"), temperatures):
            observed = _observed_log_probabilities(row, head, temperature)
            if observed is None:
                continue
            logs, target = observed
            probabilities = [math.exp(value) for value in logs]
            diagnostics.add(
                game=row.game_id,
                head=head,
                probabilities=probabilities,
                target=target,
                log_probability=logs[target],
                slices=slices,
                event=row.response_target if head == "rank" else None,
            )
            if head in branch_metrics:
                branch_metrics[head][str(row.response_target)].add(probabilities, target, log_probability=logs[target])
                branch_games[head][str(row.response_target)].add(row.game_id)
            if head == "response" and support is not None:
                diagnostics.add_unsupported(
                    math.fsum(value for value, covered in zip(probabilities, support.supported) if not covered)
                )
        if row.response_valid:
            events[str(row.response_target)] += 1
        if row.action_valid and row.action_target < 148 and sum(action < 148 for action in row.candidates) >= 2:
            prior, logs = legal_prior(row.policy_logits, row.candidates, temperatures[0])
            selected = support_discards(row.candidates, logs, log_threshold)
            mass = math.fsum(prior[action] for action in selected)
            diagnostics.add_gate(
                retained=row.action_target in selected, support_size=len(selected), mass=mass, slices=slices
            )
            gate_rows += 1
            retained += row.action_target in selected
            small += len(selected) < 2
            mass_sum += mass
            for table, key in ((threat, row.threat_slice), (rank_slices, str(row.current_rank or "unknown"))):
                item = table.setdefault(key, {"count": 0, "retained": 0})
                item["count"] += 1
                item["retained"] += row.action_target in selected
    details = diagnostics.report(bootstrap_seed=bootstrap_seed, bootstrap_replicates=bootstrap_replicates)
    report: dict[str, Any] = {}
    for head in ("action", "response", "rank"):
        values = details["heads"][head]
        report[head] = {
            "count": values["count"],
            "nll": values["nll"],
            "brier": values["brier"],
            "ece": values["confidence_reliability"]["ece"],
        }
    report["branch_metrics"] = {}
    for head, table in branch_metrics.items():
        report["branch_metrics"][head] = {}
        for event, classifier in table.items():
            values = classifier.report()
            report["branch_metrics"][head][event] = {
                "count": values["count"], "independent_games": len(branch_games[head][event]),
                "nll": values["nll"], "brier": values["brier"], "ece": values["confidence_reliability"]["ece"],
            }
    report.update(
        gate_multi_discard_count=gate_rows,
        gate_retention=retained / gate_rows if gate_rows else None,
        fewer_than_two_support_fraction=small / gate_rows if gate_rows else None,
        mean_support_mass=mass_sum / gate_rows if gate_rows else None,
        response_counts=events,
        gate_by_threat=threat,
        gate_by_current_rank=rank_slices,
        diagnostics=details,
        evidence_status="empirical_only_not_uniform_error_bounds",
    )
    return report


@dataclass(frozen=True)
class CalibrationArtifact:
    binding: RuntimeBinding
    support_sha256: str
    split: CalibrationSplit
    temperatures: tuple[float, float, float]
    log_threshold: float | None
    fit_rows_sha256: str
    check_rows_sha256: str
    fit_metrics: Mapping[str, Any]
    check_metrics: Mapping[str, Any]
    source_reference: str
    diagnostic_fixture: bool = False
    quality_policy: CalibrationQualityPolicy | None = None
    inference_device_type: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_metrics", _freeze_json(self.fit_metrics))
        object.__setattr__(self, "check_metrics", _freeze_json(self.check_metrics))
        if type(self.diagnostic_fixture) is not bool:
            raise ValueError("Calibration fixture provenance must be explicit boolean metadata")
        sha256_value(self.fit_rows_sha256, "fit_rows_sha256")
        sha256_value(self.check_rows_sha256, "check_rows_sha256")

    @property
    def gate_valid(self) -> bool:
        return self.log_threshold is not None and self.fit_metrics.get("gate_multi_discard_count", 0) > 0

    @property
    def quality_report(self) -> dict[str, Any]:
        return evaluate_quality(self.check_metrics, self.quality_policy)

    @property
    def branch_quality_report(self) -> dict[str, Any]:
        policy = None if self.quality_policy is None else self.quality_policy.branch_policy
        return evaluate_branch_quality(self.check_metrics, policy)

    @property
    def value_branch_statuses(self) -> tuple[str, ...]:
        branches = self.branch_quality_report["branches"]
        return tuple(branches[str(event)]["status"] for event in range(21))

    @property
    def digest(self) -> str:
        return self.to_record()["calibration_sha256"]

    @property
    def empirical_check_valid(self) -> bool:
        return self.gate_valid and self.quality_report["status"] == "passed"

    def to_record(self) -> dict[str, Any]:
        document = {
            "version": CALIBRATION_VERSION,
            "contract": runtime_contract(),
            "binding": self.binding.to_record(),
            "support_sha256": self.support_sha256,
            "split": asdict(self.split),
            "temperatures": list(self.temperatures),
            "log_threshold": self.log_threshold,
            "fit_rows_sha256": self.fit_rows_sha256,
            "check_rows_sha256": self.check_rows_sha256,
            "fit_metrics": _thaw_json(self.fit_metrics),
            "check_metrics": _thaw_json(self.check_metrics),
            "source_reference": self.source_reference,
            "diagnostic_fixture": self.diagnostic_fixture,
            "quality_policy": self.quality_policy.to_record() if self.quality_policy is not None else None,
            "quality_report": self.quality_report,
            "branch_quality_report": self.branch_quality_report,
            "value_branch_statuses": list(self.value_branch_statuses),
            "inference_device_type": self.inference_device_type,
            "gate_valid": self.gate_valid,
            "empirical_check_valid": self.empirical_check_valid,
            "certified_error_bounds": False,
        }
        # Canonicalize tuples before equality checks against JSON-loaded artifacts.
        import json  # noqa: PLC0415

        document = json.loads(json.dumps(document, allow_nan=False))
        return {**document, "calibration_sha256": content_hash(document)}

    def validate(self, binding: RuntimeBinding, support: EventSupport, *, allow_diagnostic: bool = False) -> None:
        self.split.validate()
        if not self.diagnostic_fixture and (
            self.quality_policy is None or self.inference_device_type not in {"cpu", "cuda"}
        ):
            raise ValueError("Production calibration needs preregistered quality criteria and its originating backend")
        if not self.diagnostic_fixture and self.split.validation_manifest_sha256 is None:
            raise ValueError("Unbound static arrays cannot masquerade as production validation calibration")
        if self.binding != binding or self.support_sha256 != support.digest or not self.source_reference.strip():
            raise ValueError("Calibration/model/support provenance mismatch")
        if support.train_manifest_sha256 != binding.train_manifest_sha256:
            raise ValueError("Calibration uses another training dataset")
        if self.diagnostic_fixture and not allow_diagnostic:
            raise ValueError("Diagnostic calibration cannot masquerade as a production artifact")
        if len(self.temperatures) != 3:
            raise ValueError("Exactly three global temperatures required")
        for value in self.temperatures:
            masked_log_softmax((0.0,), (True,), value)
        if self.log_threshold is not None and (not math.isfinite(self.log_threshold) or self.log_threshold > 0):
            raise ValueError("Invalid authoritative log gate threshold")

    @classmethod
    def from_record(
        cls,
        document: Mapping[str, Any],
        binding: RuntimeBinding,
        support: EventSupport,
        *,
        allow_diagnostic: bool = False,
    ) -> CalibrationArtifact:
        unsigned = {key: value for key, value in document.items() if key != "calibration_sha256"}
        if (
            document.get("version") != CALIBRATION_VERSION
            or document.get("contract") != runtime_contract()
            or document.get("calibration_sha256") != content_hash(unsigned)
        ):
            raise ValueError("Calibration integrity/version/contract mismatch")
        split = document["split"]
        result = cls(
            RuntimeBinding(**document["binding"]),
            document["support_sha256"],
            CalibrationSplit(
                tuple(split["validation_games"]),
                tuple(split["fit_games"]),
                tuple(split["check_games"]),
                split["salt"],
                split.get("validation_manifest_sha256"),
                split.get("main_split_salt"),
            ),
            tuple(document["temperatures"]),
            document["log_threshold"],
            document["fit_rows_sha256"],
            document["check_rows_sha256"],
            document["fit_metrics"],
            document["check_metrics"],
            document["source_reference"],
            document["diagnostic_fixture"],
            CalibrationQualityPolicy.from_record(document["quality_policy"])
            if document["quality_policy"] is not None
            else None,
            document["inference_device_type"],
        )
        result.validate(binding, support, allow_diagnostic=allow_diagnostic)
        if result.to_record() != dict(document):
            raise ValueError("Calibration derived flags or field set were modified")
        return result


def calibrated_policy_identity(
    binding: RuntimeBinding,
    support: EventSupport,
    calibration: CalibrationArtifact,
    *,
    mode: str = "empirical_only",
    max_prior_amplification: float | None = None,
    diagnostic_argmax: bool = False,
) -> dict[str, Any]:
    calibration.validate(binding, support, allow_diagnostic=calibration.diagnostic_fixture)
    if mode not in {"empirical_only", "certified"} or type(diagnostic_argmax) is not bool:
        raise ValueError("Invalid finite-policy identity")
    if max_prior_amplification is not None:
        from .policy import tau_minimum  # noqa: PLC0415

        tau_minimum(max_prior_amplification)
    record = {
        "version": "calibrated_policy_identity_v2_branches",
        "binding_sha256": binding.digest,
        "calibration_sha256": calibration.digest,
        "support_sha256": support.digest,
        "branch_quality_policy_sha256": calibration.branch_quality_report["branch_quality_policy_sha256"],
        "value_branch_statuses": list(calibration.value_branch_statuses),
        "inference_device_type": calibration.inference_device_type,
        "mode": mode,
        "sampling": "diagnostic_argmax" if diagnostic_argmax else "stochastic",
        "max_prior_amplification": max_prior_amplification,
    }
    return {**record, "policy_identity_sha256": content_hash(record)}


def validate_policy_identity(record: Mapping[str, Any], binding: RuntimeBinding) -> None:
    unsigned = {key: value for key, value in record.items() if key != "policy_identity_sha256"}
    if (
        record.get("version") != "calibrated_policy_identity_v2_branches"
        or record.get("binding_sha256") != binding.digest
        or record.get("policy_identity_sha256") != content_hash(unsigned)
    ):
        raise ValueError("Calibrated-policy identity integrity or model mismatch")
    fields = {
        "version", "binding_sha256", "calibration_sha256", "support_sha256",
        "branch_quality_policy_sha256", "value_branch_statuses", "inference_device_type",
        "mode", "sampling", "max_prior_amplification", "policy_identity_sha256",
    }
    if set(record) != fields:
        raise ValueError("Calibrated-policy identity requires its full branch-quality contract")
    states = record["value_branch_statuses"]
    if not isinstance(states, (list, tuple)) or len(states) != 21 or any(
        state not in {"passed", "failed", "unavailable"} for state in states
    ):
        raise ValueError("Invalid calibrated-policy branch quality states")
    if record["branch_quality_policy_sha256"] is not None:
        sha256_value(record["branch_quality_policy_sha256"], "branch_quality_policy_sha256")
    for key in ("calibration_sha256", "support_sha256"):
        sha256_value(record[key], key)


def calibration_rows_sha256(rows: Sequence[CalibrationRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_bytes(asdict(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def calibrate_static_rows(
    fit_rows: Sequence[CalibrationRow],
    check_rows: Sequence[CalibrationRow],
    *,
    split: CalibrationSplit,
    binding: RuntimeBinding,
    support: EventSupport,
    temperatures: Sequence[float] = (0.5, 1.0, 2.0),
    source_reference: str,
    diagnostic_fixture: bool = False,
    quality_policy: CalibrationQualityPolicy | None = None,
    inference_device_type: str = "unspecified",
) -> CalibrationArtifact:
    split.validate()
    if any(row.game_id not in split.fit_games for row in fit_rows) or any(
        row.game_id not in split.check_games for row in check_rows
    ):
        raise ValueError("cal_check cannot fit temperatures or p_min, and foreign games are forbidden")
    if len({row.row_id for row in chain(fit_rows, check_rows)}) != len(fit_rows) + len(check_rows):
        raise ValueError("Duplicate calibration decision identities")
    fitted = (
        select_static_temperature(fit_rows, "action", temperatures)[0],
        select_static_temperature(fit_rows, "response", temperatures)[0],
        select_static_temperature(fit_rows, "rank", temperatures)[0],
    )
    threshold, _ = gate_log_threshold(fit_rows, fitted[0])
    result = CalibrationArtifact(
        binding,
        support.digest,
        split,
        fitted,
        threshold,
        calibration_rows_sha256(fit_rows),
        calibration_rows_sha256(check_rows),
        calibration_metrics(fit_rows, fitted, threshold, support=support),
        calibration_metrics(check_rows, fitted, threshold, support=support),
        source_reference,
        diagnostic_fixture,
        quality_policy,
        inference_device_type,
    )
    result.validate(binding, support, allow_diagnostic=diagnostic_fixture)
    return result
