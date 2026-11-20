from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast, overload

import numpy as np
import torch

from ..authorization import update_implementation_sha256
from ..budget import BudgetLedger, BudgetPlan
from ..calibration import CalibrationArtifact, CalibrationRow, calibrate_static_rows, calibration_split_from_manifest
from ..calibration_quality import CalibrationQualityPolicy
from ..contracts import Response, content_hash
from ..evaluation_authorization import require_evaluation_authorization
from ..support import EventSupport
from .batch import rule_feature_view
from .dataset import DecisionDataset, collate_examples
from .evaluation_startup import evaluation_job
from .inference import FrozenModelPredictor
from .losses import loss_sums


class CalibrationRows(Sequence[CalibrationRow]):
    """Readonly lazy arrays; fit/check iterators never expand all logits into RAM."""

    def __init__(self, path: Path, partition: str):
        if partition not in {"fit", "check"}:
            raise ValueError("Unknown calibration partition")
        self.path, self.partition = path, partition
        self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        self.indices = [
            row[0]
            for row in self.connection.execute(
                "SELECT row_id FROM logits WHERE partition=? ORDER BY row_id", (partition,)
            )
        ]

    def __len__(self) -> int:
        return len(self.indices)

    @overload
    def __getitem__(self, index: int) -> CalibrationRow: ...
    @overload
    def __getitem__(self, index: slice) -> list[CalibrationRow]: ...
    def __getitem__(self, index: int | slice) -> CalibrationRow | list[CalibrationRow]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        row = self.connection.execute(
            "SELECT metadata,policy,response,rank,allowed FROM logits WHERE row_id=?", (self.indices[index],)
        ).fetchone()
        if row is None:
            raise ValueError("Archived calibration row disappeared")
        return self._decode(row)

    @staticmethod
    def _decode(row: tuple) -> CalibrationRow:
        metadata = json.loads(row[0])
        metadata["candidates"] = tuple(metadata["candidates"])
        return CalibrationRow(
            **metadata,
            policy_logits=tuple(float(value) for value in np.frombuffer(row[1], dtype="<f4")),
            response_logits=tuple(float(value) for value in np.frombuffer(row[2], dtype="<f4")),
            rank_logits=tuple(
                tuple(float(value) for value in part) for part in np.frombuffer(row[3], dtype="<f4").reshape(21, 4)
            ),
            response_allowed=tuple(bool(value) for value in row[4]),
        )

    def __iter__(self) -> Iterator[CalibrationRow]:
        cursor = self.connection.execute(
            "SELECT metadata,policy,response,rank,allowed FROM logits WHERE partition=? ORDER BY row_id",
            (self.partition,),
        )
        for row in cursor:
            yield self._decode(row)

    def close(self) -> None:
        self.connection.close()


def dataset_event_support(dataset: DecisionDataset) -> EventSupport:
    counts = [0] * 21
    rows = dataset._db().execute(
        "SELECT s.response_target,COUNT(DISTINCT s.game_id) FROM samples s JOIN games g ON s.game_id=g.game_id "
        "WHERE g.split='train' AND s.action_valid=1 AND s.response_applicable=1 "
        "AND s.response_observed=1 GROUP BY s.response_target"
    )
    for event, count in rows:
        if not 0 <= event < 21:
            raise ValueError("Invalid event in completed training evidence")
        counts[event] = count
    return EventSupport(
        dataset.manifest["manifest_sha256"],
        tuple(counts),
        f"completed immutable database {dataset.manifest['database_sha256']}; distinct train game_id only",
    )


def calibration_job_plan(
    predictor: FrozenModelPredictor,
    dataset_manifest: dict[str, Any],
    validation_manifest: dict[str, Any],
    *,
    quality_policy: CalibrationQualityPolicy,
    temperatures: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, Any]:
    split = calibration_split_from_manifest(validation_manifest)
    record = {
        "job_version": "heldout_logits_calibration_job_v3_single_prefix",
        "source_sha256": update_implementation_sha256(),
        "binding_sha256": predictor.binding.digest,
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "validation_manifest_sha256": split.validation_manifest_sha256,
        "split_salt": split.salt,
        "temperatures": list(temperatures),
        "use_rule_features": predictor.use_rule_features,
        "inference_device_type": next(predictor.model.parameters()).device.type,
        "backend_precision": predictor.backend_precision.to_record(),
        "quality_policy": quality_policy.to_record(),
    }
    return {**record, "plan_sha256": content_hash(record)}


def _archive_and_fit(
    predictor: FrozenModelPredictor,
    validation: DecisionDataset,
    validation_manifest: dict[str, Any],
    output: Path,
    *,
    ledger: BudgetLedger,
    quality_policy: CalibrationQualityPolicy,
    temperatures: tuple[float, ...],
    microbatch: int,
    plan_sha256: str,
) -> tuple[CalibrationArtifact, EventSupport]:
    split = calibration_split_from_manifest(validation_manifest)
    support = dataset_event_support(validation)
    path = output / "logits.sqlite3"
    before = predictor._frozen_identity()
    model, device = predictor.model, next(predictor.model.parameters()).device
    db = sqlite3.connect(path)
    try:
        db.execute(
            "CREATE TABLE logits(row_id TEXT PRIMARY KEY, partition TEXT, game_id TEXT,metadata TEXT,"
            "policy BLOB,response BLOB,rank BLOB,allowed BLOB)"
        )
        fit_games = set(split.fit_games)
        for start in range(0, len(validation), microbatch):
            stop = min(start + microbatch, len(validation))
            ledger.reserve_next(
                (stop - start) / ledger.plan.forecast.measured_units_per_second * ledger.plan.forecast.safety_multiplier
            )
            rows = [validation[index] for index in range(start, stop)]
            batch = collate_examples(rows).to(device)
            inputs = rule_feature_view(
                batch.inputs,
                torch.zeros(batch.inputs.size, dtype=torch.bool, device=device),
                enabled=predictor.use_rule_features,
            )
            with predictor.frozen_forward():
                outputs = model(inputs)
            # A strict gate must be fitted from the deployed single-prefix
            # arithmetic, not a batch-shaped dense forward with teacher ACTION.
            outputs = replace(outputs, policy_logits=predictor.policy_logits_from_rows(rows))
            loss_sums(outputs, batch.labels, inputs.legal_mask)
            values = {
                key: value.detach().float().cpu().numpy().astype("<f4", copy=False)
                for key, value in (
                    ("policy", outputs.policy_logits),
                    ("response", outputs.response_logits),
                    ("rank", outputs.rank_logits),
                )
            }
            label = {key: value.cpu().tolist() for key, value in batch.labels.__dict__.items()}
            for index, game in enumerate(batch.game_ids):
                base = batch.inputs.fields
                fields = inputs.fields
                riichi = bool(
                    (
                        (fields["kind"][index] == 3)
                        & (fields["actor"][index] > 1)
                        & (fields["flags"][index, :, 1] == 2)
                    ).any()
                )
                called = bool(
                    (
                        (fields["kind"][index] == 5)
                        & (fields["actor"][index] > 1)
                        & (fields["flags"][index, :, 0] == 2)
                    ).any()
                )
                rank_values = base["stage"][index][base["kind"][index] == 2]
                rank = int(rank_values.item()) if rank_values.numel() == 1 else 0
                metadata = {
                    "game_id": game,
                    "row_id": batch.inputs.context_ids[index],
                    "candidates": [action for action in inputs.candidates[index].cpu().tolist() if action != 546],
                    "action_target": label["action_target"][index],
                    "action_valid": label["action_valid"][index],
                    "response_target": label["response_target"][index],
                    "response_valid": bool(batch.labels.response_valid[index]),
                    "rank_target": label["final_rank_target"][index],
                    "value_valid": label["value_valid"][index],
                    "threat_slice": "mixed"
                    if riichi and called
                    else "riichi"
                    if riichi
                    else "open_meld"
                    if called
                    else "none",
                    "current_rank": rank if 1 <= rank <= 4 else None,
                }
                db.execute(
                    "INSERT INTO logits VALUES (?,?,?,?,?,?,?,?)",
                    (
                        metadata["row_id"],
                        "fit" if game in fit_games else "check",
                        game,
                        json.dumps(metadata),
                        values["policy"][index].tobytes(),
                        values["response"][index].tobytes(),
                        values["rank"][index].tobytes(),
                        bytes(label["response_allowed"][index]),
                    ),
                )
            db.commit()
            ledger.settle_elapsed(f"validation_logits/{start}")
    finally:
        db.close()
    if predictor._frozen_identity() != before:
        raise ValueError("Frozen calibration changed model state/configuration")
    fit, check = CalibrationRows(path, "fit"), CalibrationRows(path, "check")
    try:
        artifact = calibrate_static_rows(
            fit,
            check,
            split=split,
            binding=predictor.binding,
            support=support,
            temperatures=temperatures,
            source_reference=f"frozen archive {path.name}; plan {plan_sha256}",
            diagnostic_fixture=False,
            quality_policy=quality_policy,
            inference_device_type=device.type,
        )
    finally:
        fit.close()
        check.close()
    return artifact, support


@torch.no_grad()
def collect_and_calibrate(
    predictor: FrozenModelPredictor | Callable[[], FrozenModelPredictor],
    dataset_dir: Path,
    validation_manifest: dict[str, Any],
    output: Path,
    *,
    budget: BudgetPlan,
    plan_sha256: str,
    authorization: dict[str, Any] | None = None,
    temperatures: tuple[float, ...] = (0.5, 1.0, 2.0),
    microbatch: int = 1,
    quality_policy: CalibrationQualityPolicy | None = None,
) -> dict[str, Any]:
    require_evaluation_authorization(authorization, operation="calibrate_frozen_logits", plan_sha256=plan_sha256)
    if quality_policy is None:
        raise ValueError("Preregister calibration-quality limits before computing or inspecting cal_check")
    if microbatch < 1 or budget.phase not in {"M1", "M2"} or budget.forecast.recipe_sha256 != plan_sha256:
        raise ValueError("Calibration requires its registered full-validation budget")
    with evaluation_job(
        output, budget, operation="calibrate_frozen_logits", plan_sha256=plan_sha256, authorization=authorization
    ) as ledger:
        predictor = cast(Callable[[], FrozenModelPredictor], predictor)() if callable(predictor) else predictor
        predictor._unchanged()
        validation = DecisionDataset(dataset_dir, "validation", include_candidate_branches=False)
        plan = calibration_job_plan(
            predictor,
            validation.manifest,
            validation_manifest,
            quality_policy=quality_policy,
            temperatures=temperatures,
        )
        if (
            plan["plan_sha256"] != plan_sha256
            or predictor.binding.train_manifest_sha256 != validation.manifest["manifest_sha256"]
        ):
            raise ValueError("Calibration uses a different model, quality policy or training selection")
        split = calibration_split_from_manifest(validation_manifest)
        actual = {row["game_id"] for row in validation.manifest["games"] if row["split"] == "validation"}
        if actual != set(split.validation_games):
            raise ValueError("Production calibration cannot silently use a probe validation subset")
        # The measured whole-job rate covers loading, hashes, index materialization,
        # canonical policy forwards, batched auxiliary heads, archives, and fit/check.
        if budget.forecast.work_units < len(validation):
            raise ValueError("Calibration cost forecast must cover every validation decision row")
        ledger.settle_elapsed("calibration_startup")
        artifact, support = _archive_and_fit(
            predictor,
            validation,
            validation_manifest,
            output,
            ledger=ledger,
            quality_policy=quality_policy,
            temperatures=temperatures,
            microbatch=microbatch,
            plan_sha256=plan_sha256,
        )
    result = {
        "status": "PASS" if artifact.empirical_check_valid else "CALIBRATION_REJECTED",
        "plan": plan,
        "calibration": artifact.to_record(),
        "support": support.to_record(),
        "quality_report": artifact.quality_report,
        "budget": ledger.state_dict(),
        "validation_rows": len(validation),
        "model_parameters_updated": False,
        "error_certificate_estimated": False,
        "response_event_names": [event.name for event in Response],
    }
    result["job_sha256"] = content_hash(result)
    (output / "calibration.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
