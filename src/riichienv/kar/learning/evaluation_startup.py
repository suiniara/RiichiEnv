from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..budget import BudgetLedger, BudgetPlan
from ..contracts import content_hash
from ..evaluation_authorization import require_evaluation_authorization


@contextmanager
def evaluation_job(
    output: Path,
    budget: BudgetPlan,
    *,
    operation: str,
    plan_sha256: str,
    authorization: dict[str, Any] | None,
) -> Iterator[BudgetLedger]:
    """Own one immutable output and account for its complete preparation/work.

    These jobs do not resume partially collected logits or match selections.
    Existing outputs, including interrupted attempts, must be preserved for
    explicit reconciliation; a second invocation never replaces their ledger.
    """
    require_evaluation_authorization(authorization, operation=operation, plan_sha256=plan_sha256)
    if budget.phase not in {"M1", "M2"} or budget.forecast.recipe_sha256 != plan_sha256:
        raise ValueError("Frozen evaluation requires its registered M1/M2 budget and plan")
    assert authorization is not None
    identity = {
        "version": "frozen_evaluation_startup_v1",
        "operation": operation,
        "plan_sha256": plan_sha256,
        "source_sha256": authorization["source_sha256"],
        "output": str(output.resolve()),
        "budget": budget.to_record(),
    }
    identity["startup_sha256"] = content_hash(identity)
    output.mkdir(parents=True, exist_ok=False)
    # No model/data preparation occurs until both identity and compute attempt
    # are durable. An incomplete publication leaves the directory blocked.
    with (output / "startup.json").open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    ledger = BudgetLedger(output / "costs.jsonl", budget)
    with ledger.record_compute(operation + "/startup_and_work"):
        yield ledger
