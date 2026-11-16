from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from ..contracts import content_hash


def performance_summary(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item) or item < 0
        for item in value.values()
    ):
        raise ValueError("Run performance counters must be measured finite nonnegative totals")
    seconds, examples, tokens, padded = (
        value[key] for key in ("training_seconds", "examples", "real_tokens", "padded_real_tokens")
    )
    if (examples and seconds <= 0) or padded < tokens:
        raise ValueError("Run throughput needs actual elapsed time and consistent real-token counts")
    return {
        **value,
        "decision_samples_per_second": examples / seconds if seconds else None,
        "tokens_per_second": tokens / seconds if seconds else None,
        "padding_fraction_real_microbatches": 1 - tokens / padded if padded else None,
        "averaging": "sum_of_real_work_over_sum_of_committed_update_wall_times",
        "dummy_work_in_denominator_not_real_token_numerator": True,
    }


def completed_run_record(
    state: dict[str, Any],
    *,
    phase: str,
    planned_epochs: int,
    runtime: dict[str, Any],
    contract: dict[str, Any],
    budget: dict[str, Any],
    dataset_sha256: str,
    final_weights_sha256: str,
    checkpoint: dict[str, Any],
    recovered: bool,
) -> dict[str, Any]:
    if state.get("finished") is not True or not state.get("validation_history") or state.get("completed_epochs", 0) < 1:
        raise ValueError("Only a committed finished training state can publish completion")
    if checkpoint["metadata"]["step"] != state["step"] or checkpoint["metadata"]["release_contract"] != contract:
        raise ValueError("Completion publication differs from the committed checkpoint")
    if state.get("performance") is not None and (
        state["performance"]["examples"] != state["processed_examples"]
        or state["performance"]["updates"] != state["step"]
    ):
        raise ValueError("Committed run performance counters differ from checkpoint progress")
    if state.get("final_weights_sha256", final_weights_sha256) != final_weights_sha256:
        raise ValueError("Finished checkpoint model differs from its recorded final weights")
    return {
        "status": "PASS",
        "phase": phase,
        "completed_epochs": state["completed_epochs"],
        "planned_epochs": planned_epochs,
        "step": state["step"],
        "processed_examples": state["processed_examples"],
        "performance": performance_summary(state.get("performance")),
        "full_epoch_training": True,
        "initial_weights_sha256": state["initial_weights_sha256"],
        "final_weights_sha256": final_weights_sha256,
        "epoch_order_sha256": state["epoch_order_sha256"],
        "third_epoch_decision": state.get("third_epoch_decision"),
        "runtime": runtime,
        "release_contract": contract,
        "final_validation": state["validation_history"][-1],
        "budget_ledger": budget,
        "dataset_manifest_sha256": dataset_sha256,
        **state.get("execution_telemetry", {}),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_file": checkpoint["checkpoint_file"],
        "recovered_completed_checkpoint": recovered,
    }


def publish_run_manifest(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in record.items() if key != "run_manifest_sha256"}
    signed = {**body, "run_manifest_sha256": content_hash(body)}
    if path.exists():
        previous = path.read_bytes()
        if previous == (json.dumps(signed, indent=2, allow_nan=False) + "\n").encode("utf-8"):
            return signed
        archived = path.with_name(path.stem + ".previous-" + hashlib.sha256(previous).hexdigest()[:20] + ".json")
        if archived.exists():
            if archived.read_bytes() != previous:
                raise ValueError("Prior publication archive conflicts")
        else:
            with archived.open("xb") as stream:
                stream.write(previous)
                stream.flush()
                os.fsync(stream.fileno())
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(signed, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return signed
