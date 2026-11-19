from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import content_hash, model_tier
from .evaluation_authorization import exclusive_job

PHASE_CAPS = {"M1": 100.0, "M2": 650.0, "M3": 250.0}
TOTAL_CAP_USD = 1000.0
BUDGET_VERSION = "budget_ladder_v1"
LEDGER_VERSION = "durable_compute_cost_ledger_v3"


@dataclass(frozen=True)
class CostForecast:
    phase: str
    work_units: int
    measured_units_per_second: float
    resource_usd_per_hour: float
    fixed_cost_usd: float
    measurement_reference: str
    recipe_sha256: str
    safety_multiplier: float = 1.2

    def __post_init__(self) -> None:
        if self.phase not in PHASE_CAPS or type(self.work_units) is not int or self.work_units <= 0:
            raise ValueError("A forecast needs an explicit phase and complete planned work")
        values = (
            self.measured_units_per_second,
            self.resource_usd_per_hour,
            self.fixed_cost_usd,
            self.safety_multiplier,
        )
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("A forecast cannot omit or invent throughput/pricing")
        if (
            self.measured_units_per_second <= 0
            or min(self.resource_usd_per_hour, self.fixed_cost_usd) < 0
            or self.safety_multiplier < 1
        ):
            raise ValueError("Invalid throughput, price or safety factor")
        if not self.measurement_reference.strip() or len(self.recipe_sha256) != 64:
            raise ValueError("Forecast must bind a measured recipe and its evidence")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CostForecast:
        names = set(cls.__dataclass_fields__)
        extras = set(record) - names
        if extras - {"version", "forecast_seconds", "forecast_usd", "forecast_sha256"} or (
            extras and "forecast_sha256" not in record
        ):
            raise ValueError("Unknown or unsigned derived cost forecast fields")
        value = cls(**{key: item for key, item in record.items() if key in names})
        if "forecast_sha256" in record and value.to_record() != record:
            raise ValueError("Forecast derived fields were modified")
        return value

    @property
    def seconds(self) -> float:
        return self.work_units / self.measured_units_per_second * self.safety_multiplier

    @property
    def usd(self) -> float:
        return self.seconds * self.resource_usd_per_hour / 3600 + self.fixed_cost_usd

    def to_record(self) -> dict[str, Any]:
        result = {"version": BUDGET_VERSION, **asdict(self), "forecast_seconds": self.seconds, "forecast_usd": self.usd}
        return {**result, "forecast_sha256": content_hash(result)}


@dataclass(frozen=True)
class FixedCostReceipt:
    receipt_id: str
    usd: float
    category: str
    reference: str

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.reference or not self.category or self.category == "compute":
            raise ValueError("A non-compute invoice needs a unique ID, category and actual receipt reference")
        if isinstance(self.usd, bool) or not math.isfinite(self.usd) or self.usd < 0:
            raise ValueError("Actual non-compute charges must be finite and nonnegative")


@dataclass(frozen=True)
class BudgetPlan:
    phase: str
    budget_cap_usd: float
    budget_spent_usd: float
    forecast: CostForecast
    other_phase_spent_usd: float = 0.0
    fixed_cost_receipts: tuple[FixedCostReceipt, ...] = ()

    def __post_init__(self) -> None:
        values = (self.budget_cap_usd, self.budget_spent_usd, self.other_phase_spent_usd)
        if (
            self.phase not in PHASE_CAPS
            or self.forecast.phase != self.phase
            or any(isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in values)
        ):
            raise ValueError("Missing/inconsistent per-phase budget")
        if self.budget_cap_usd > PHASE_CAPS[self.phase]:
            raise ValueError("Phase caps are independent; no implicit budget transfer")
        if len({receipt.receipt_id for receipt in self.fixed_cost_receipts}) != len(self.fixed_cost_receipts):
            raise ValueError("Duplicate fixed-cost invoice ID")
        actual = math.fsum(receipt.usd for receipt in self.fixed_cost_receipts)
        self.admit(self.forecast.usd + max(0.0, actual - self.forecast.fixed_cost_usd))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> BudgetPlan:
        values = dict(record)
        values["forecast"] = CostForecast.from_record(values["forecast"])
        values["fixed_cost_receipts"] = tuple(
            FixedCostReceipt(**item) for item in values.get("fixed_cost_receipts", ())
        )
        return cls(**values)

    def admit(self, incremental_usd: float) -> None:
        if not math.isfinite(incremental_usd) or incremental_usd < 0:
            raise ValueError("Every cost must be explicit and nonnegative")
        if self.budget_spent_usd + incremental_usd > self.budget_cap_usd + 1e-9:
            raise BudgetExceededError("Phase would exceed its registered cap; do not silently reduce work")
        if self.other_phase_spent_usd + self.budget_spent_usd + incremental_usd > TOTAL_CAP_USD + 1e-9:
            raise BudgetExceededError("Total project budget would exceed USD1000")

    def to_record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "budget_cap_usd": self.budget_cap_usd,
            "budget_spent_usd": self.budget_spent_usd,
            "other_phase_spent_usd": self.other_phase_spent_usd,
            "forecast": self.forecast.to_record(),
            "fixed_cost_receipts": [asdict(receipt) for receipt in self.fixed_cost_receipts],
        }


class BudgetExceededError(RuntimeError):
    pass


class PendingComputeError(RuntimeError):
    """An interrupted computation needs an external receipt before more work."""


class BudgetLedger:
    """Append-only charges, fixed estimates, and unfinished compute attempts.

    A compute estimate is never an invoice. A live attempt survives intermediate
    settlements, and only its owner can finish it using the local monotonic clock.
    Reopening a ledger cannot recover that clock; an external receipt must close
    the old attempt before any new computation, including a differently named job.
    """

    def __init__(self, path: Path, plan: BudgetPlan, *, job_id: str | None = None, allow_create: bool = True):
        self.path, self.plan = path, plan
        self.job_id = job_id or plan.forecast.to_record()["forecast_sha256"]
        self.entries: list[dict[str, Any]] = []
        self._heads: dict[str, int] = {}
        self._reservations: dict[str, float] = {}
        self._pending_compute: dict[str, dict[str, Any]] = {}
        self._owned_compute: str | None = None
        self._fixed_paid: dict[str, float] = {}
        self._invoices: dict[str, dict[str, Any]] = {}
        self._sum = self._compensation = 0.0
        self._clock = time.monotonic()
        self._byte_count = 0
        self._write_failed = False
        identity = {
            "version": LEDGER_VERSION,
            "phase": plan.phase,
            "budget_cap_usd": plan.budget_cap_usd,
            "budget_spent_usd": plan.budget_spent_usd,
        }
        if path.exists():
            raw = path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise ValueError("Incomplete budget-ledger tail; preserve it for explicit audited recovery")
            self._byte_count = len(raw)
            for line in raw.decode("utf-8").splitlines():
                row = json.loads(line)
                unsigned = {key: value for key, value in row.items() if key != "entry_sha256"}
                if (
                    row.get("entry_sha256") != content_hash(unsigned)
                    or row.get("previous_sha256") != self.head
                    or row.get("phase") != plan.phase
                ):
                    raise ValueError("Budget ledger is modified or belongs to another phase")
                self._apply(row)
            if not self.entries or self.entries[0].get("identity") != identity:
                raise ValueError("Ledger identity/version changed; legacy ledgers need explicit audited migration")
        else:
            if not allow_create:
                raise ValueError("A resumed job has receipts but its required phase ledger is missing")
            self._append("identity", identity=identity, category="control", reference="registered phase budget")
        reservation = self._reservations.get(self.job_id)
        if reservation is None:
            if self._pending_compute:
                raise PendingComputeError(
                    "Finish or reconcile the existing compute attempt before registering another ledger job"
                )
            self._check_room(plan.forecast.fixed_cost_usd)
            self._append(
                "fixed_reservation",
                reserved_usd=plan.forecast.fixed_cost_usd,
                forecast_sha256=plan.forecast.to_record()["forecast_sha256"],
                category="control",
                reference="unbilled fixed-cost estimate",
            )
        elif reservation != plan.forecast.fixed_cost_usd:
            raise ValueError("A fixed-cost reservation cannot be replaced on resume")
        for receipt in plan.fixed_cost_receipts:
            self.charge(
                receipt.usd, category=receipt.category, reference=receipt.reference, receipt_id=receipt.receipt_id
            )
        # An already-recorded overrun must remain readable/reconcilable. New
        # work is denied by assert_room/reserve_next, not by opening its evidence.

    @property
    def head(self) -> str | None:
        return self.entries[-1]["entry_sha256"] if self.entries else None

    def _apply(self, row: dict[str, Any]) -> None:
        if (
            row.get("sequence") != len(self.entries)
            or not isinstance(row.get("usd"), (int, float))
            or isinstance(row["usd"], bool)
            or not math.isfinite(row["usd"])
            or row["usd"] < 0
        ):
            raise ValueError("Invalid charge sequence or amount")
        kind = row.get("kind")
        if kind not in {
            "identity",
            "fixed_reservation",
            "charge",
            "compute_reservation",
            "compute_settlement",
            "compute_reconciliation",
        } or (kind == "identity") != (len(self.entries) == 0):
            raise ValueError("Invalid ledger event or identity position")
        job = row["job_id"]
        if kind == "fixed_reservation":
            amount = row["reserved_usd"]
            if not math.isfinite(amount) or amount < 0 or job in self._reservations or row["usd"] != 0:
                raise ValueError("Invalid or duplicated fixed reservation")
            self._reservations[job] = amount
        elif kind == "compute_reservation":
            self._apply_compute_reservation(row)
        elif kind in {"compute_settlement", "compute_reconciliation"}:
            attempt = row["attempt_id"]
            previous = self._pending_compute.get(attempt)
            if previous is None or previous["job_id"] != job or row["category"] != "compute":
                raise ValueError("Compute settlement has no matching unfinished attempt")
            if kind == "compute_reconciliation":
                invoice = row["receipt_id"]
                if not invoice or invoice in self._invoices or row["basis"] != "external_unsettled_cost_receipt":
                    raise ValueError("External compute reconciliation requires a unique actual receipt")
                self._invoices[invoice] = row
                del self._pending_compute[attempt]
            else:
                elapsed = row["elapsed_seconds"]
                if (
                    isinstance(elapsed, bool)
                    or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed)
                    or elapsed < 0
                    or type(row["closed"]) is not bool
                    or row["rate_usd_per_hour"] != previous["rate_usd_per_hour"]
                    or row["usd"] != elapsed * previous["rate_usd_per_hour"] / 3600
                    or row["basis"] != "measured_elapsed_at_registered_rate"
                ):
                    raise ValueError("Invalid measured compute settlement")
                if row["closed"]:
                    del self._pending_compute[attempt]
                else:
                    self._pending_compute[attempt] = {
                        **previous,
                        "reserved_usd": max(0.0, previous["reserved_usd"] - row["usd"]),
                        "settled_usd": previous["settled_usd"] + row["usd"],
                    }
        elif kind == "charge" and row["category"] != "compute":
            receipt_id = row["receipt_id"]
            if not receipt_id or receipt_id in self._invoices:
                raise ValueError("Non-compute receipt was duplicated")
            self._invoices[receipt_id] = row
            self._fixed_paid[job] = self._fixed_paid.get(job, 0.0) + row["usd"]
        total = self._sum + row["usd"]
        self._compensation += (
            (self._sum - total) + row["usd"] if abs(self._sum) >= abs(row["usd"]) else (row["usd"] - total) + self._sum
        )
        self._sum = total
        self.entries.append(row)
        self._heads[row["entry_sha256"]] = len(self.entries)

    def _apply_compute_reservation(self, row: dict[str, Any]) -> None:
        attempt, amount, rate, job = row["attempt_id"], row["reserved_usd"], row["rate_usd_per_hour"], row["job_id"]
        if (
            not isinstance(attempt, str)
            or not attempt
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                for value in (amount, rate)
            )
            or row["usd"] != 0
        ):
            raise ValueError("Invalid compute reservation")
        previous = self._pending_compute.get(attempt)
        if previous is None:
            if any(value["job_id"] == job for value in self._pending_compute.values()):
                raise ValueError("A job cannot replace an unfinished compute attempt")
            settled = 0.0
        else:
            if (
                previous["job_id"] != job
                or previous["rate_usd_per_hour"] != rate
                or previous["forecast_sha256"] != row["forecast_sha256"]
                or amount < previous["reserved_usd"]
            ):
                raise ValueError("A live compute estimate cannot change identity or silently shrink")
            settled = previous["settled_usd"]
        self._pending_compute[attempt] = {
            "attempt_id": attempt,
            "job_id": job,
            "reserved_usd": amount,
            "rate_usd_per_hour": rate,
            "forecast_sha256": row["forecast_sha256"],
            "settled_usd": settled,
        }

    def _append(self, kind: str, *, usd: float = 0.0, **fields: Any) -> dict[str, Any]:
        if self._write_failed:
            raise ValueError("A budget write had an uncertain outcome; reopen and verify the retained ledger")
        row = {
            "sequence": len(self.entries),
            "phase": self.plan.phase,
            "kind": kind,
            "job_id": self.job_id,
            "usd": usd,
            "previous_sha256": self.head,
            **fields,
        }
        row["entry_sha256"] = content_hash(row)
        encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        with exclusive_job(self.path.with_name(self.path.name + ".lock")):
            size = self.path.stat().st_size if self.path.exists() else 0
            if size != self._byte_count:
                raise ValueError("Another writer changed the budget ledger; reopen before appending")
            try:
                with self.path.open("ab") as stream:
                    if stream.write(encoded) != len(encoded):
                        raise OSError("Incomplete budget journal write")
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                self._write_failed = True
                raise
        self._byte_count += len(encoded)
        self._apply(row)
        return row

    @property
    def spent_usd(self) -> float:
        return self.plan.budget_spent_usd + self._sum + self._compensation

    @property
    def reserved_fixed_usd(self) -> float:
        return math.fsum(
            max(0.0, amount - self._fixed_paid.get(job, 0.0)) for job, amount in self._reservations.items()
        )

    @property
    def committed_usd(self) -> float:
        return self.spent_usd + self.reserved_fixed_usd + self.reserved_compute_usd

    @property
    def reserved_compute_usd(self) -> float:
        return math.fsum(value["reserved_usd"] for value in self._pending_compute.values())

    @property
    def pending_compute(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(self._pending_compute[key]) for key in sorted(self._pending_compute))

    def _require_owned_compute(self) -> None:
        if self._write_failed:
            raise ValueError("A budget write had an uncertain outcome; reopen and verify the retained ledger")
        if self.path.stat().st_size != self._byte_count:
            raise ValueError("Another writer changed the budget ledger; reopen before computing")
        if any(attempt != self._owned_compute for attempt in self._pending_compute):
            raise PendingComputeError(
                "Unfinished compute from a previous owner requires reconcile_compute() with an actual external "
                "receipt; estimates and a new process clock cannot close it"
            )

    def assert_room(self, extra_usd: float) -> None:
        self._require_owned_compute()
        self._check_room(extra_usd)

    def _check_room(self, extra_usd: float) -> None:
        if not math.isfinite(extra_usd) or extra_usd < 0:
            raise ValueError("A budget reservation cannot be missing, negative or non-finite")
        total = self.committed_usd + extra_usd
        if total > self.plan.budget_cap_usd + 1e-9 or total + self.plan.other_phase_spent_usd > TOTAL_CAP_USD + 1e-9:
            raise BudgetExceededError("Actual charges plus outstanding reservations exhausted this phase")

    def reserve_next(self, seconds: float, *, fixed_usd: float = 0.0) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in (seconds, fixed_usd)
        ):
            raise ValueError("Next-work reservation requires an explicit cost estimate")
        self._require_owned_compute()
        elapsed = time.monotonic() - self._clock
        if elapsed < 0 or not math.isfinite(elapsed):
            raise ValueError("The local monotonic compute clock moved backwards")
        rate = self.plan.forecast.resource_usd_per_hour
        previous = self._pending_compute.get(self._owned_compute or "")
        held = previous["reserved_usd"] if previous is not None else 0.0
        requested = max(held, (elapsed + seconds) * rate / 3600)
        self._check_room(requested - held + fixed_usd)
        if previous is not None and requested == held:
            return
        attempt = self._owned_compute or uuid.uuid4().hex
        self._append(
            "compute_reservation",
            attempt_id=attempt,
            reserved_usd=requested,
            rate_usd_per_hour=rate,
            forecast_sha256=self.plan.forecast.to_record()["forecast_sha256"],
            category="control",
            reference="unfinished compute estimate, not an actual charge",
        )
        self._owned_compute = attempt

    def charge(self, usd: float, *, category: str, reference: str, receipt_id: str | None = None) -> dict[str, Any]:
        if isinstance(usd, bool) or not math.isfinite(usd) or usd < 0 or not category or not reference:
            raise ValueError("Actual cost, category and reference are required")
        if category == "compute" and self._pending_compute:
            raise PendingComputeError("Use settle_elapsed/finish_compute, or reconcile an interrupted attempt")
        invoice = (
            None
            if category == "compute"
            else receipt_id or content_hash({"job_id": self.job_id, "category": category, "reference": reference})
        )
        if invoice in self._invoices:
            previous = self._invoices[invoice]
            if any(
                previous[key] != value
                for key, value in (
                    ("usd", usd),
                    ("category", category),
                    ("reference", reference),
                    ("job_id", self.job_id),
                )
            ):
                raise ValueError("An actual receipt ID was reused with different content")
            return previous
        row = self._append("charge", usd=usd, category=category, reference=reference, receipt_id=invoice)
        # Actual invoices are retained even when they reveal an overrun.
        self._check_room(0.0)
        return row

    def settle_elapsed(self, reference: str) -> None:
        """Atomically charge measured work and reduce its estimate, keeping the attempt open."""
        self._settle_elapsed(reference, close=False)

    def finish_compute(self, reference: str) -> None:
        """Atomically settle and close only the attempt owned by this live ledger object."""
        self._settle_elapsed(reference, close=True)

    def _settle_elapsed(self, reference: str, *, close: bool) -> None:
        self._require_owned_compute()
        if not reference:
            raise ValueError("Measured compute settlement needs a reference")
        now = time.monotonic()
        elapsed = now - self._clock
        if elapsed < 0 or not math.isfinite(elapsed):
            raise ValueError("The local monotonic compute clock moved backwards")
        rate = self.plan.forecast.resource_usd_per_hour
        usd = elapsed * rate / 3600
        if self._owned_compute is not None:
            self._append(
                "compute_settlement",
                usd=usd,
                attempt_id=self._owned_compute,
                closed=close,
                elapsed_seconds=elapsed,
                rate_usd_per_hour=rate,
                basis="measured_elapsed_at_registered_rate",
                category="compute",
                reference=reference,
            )
            if close:
                self._owned_compute = None
        else:
            self._append("charge", usd=usd, category="compute", reference=reference, receipt_id=None)
        # Advance after the durable transaction, even if it reveals an overrun.
        # A subsequent abort handler must not charge this interval twice.
        self._clock = now
        self._check_room(0.0)

    def reconcile_compute(
        self,
        attempt_id: str,
        *,
        usd: float,
        receipt_id: str,
        reference: str,
    ) -> dict[str, Any]:
        """Close an interrupted attempt with its additional, previously unrecorded actual cost.

        `usd` is not the estimate or the whole job invoice: subtract any charges
        already recorded for this attempt. The external receipt/reference must
        establish that remainder, including zero if no additional cost accrued.
        """
        if (
            isinstance(usd, bool)
            or not isinstance(usd, (int, float))
            or not math.isfinite(usd)
            or usd < 0
            or not attempt_id
            or not receipt_id
            or not reference
        ):
            raise ValueError("Interrupted compute needs an actual nonnegative remainder and unique receipt")
        if receipt_id in self._invoices:
            previous = self._invoices[receipt_id]
            if previous.get("kind") != "compute_reconciliation" or any(
                previous.get(key) != value
                for key, value in (("attempt_id", attempt_id), ("usd", usd), ("reference", reference))
            ):
                raise ValueError("An actual receipt ID was reused with different content")
            return previous
        pending = self._pending_compute.get(attempt_id)
        if pending is None:
            raise ValueError("No unfinished compute attempt has that identity")
        if attempt_id == self._owned_compute:
            raise ValueError("A live owned attempt must finish using its measured clock")
        row = self._append(
            "compute_reconciliation",
            usd=usd,
            job_id=pending["job_id"],
            attempt_id=attempt_id,
            receipt_id=receipt_id,
            reference=reference,
            category="compute",
            basis="external_unsettled_cost_receipt",
        )
        self._clock = time.monotonic()
        self._check_room(0.0)
        return row

    @contextmanager
    def record_compute(self, reference: str) -> Generator[BudgetLedger, None, None]:
        # Even a zero estimate leaves a durable unfinished marker. An explicit
        # reserve_next can enlarge it; a new process clock cannot prove it free.
        self.reserve_next(0.0)
        try:
            yield self
        except BaseException as error:
            try:
                self.finish_compute(reference + "/aborted")
            except Exception as accounting_error:
                note = getattr(error, "add_note", None)
                message = f"Cost settlement also failed: {type(accounting_error).__name__}: {accounting_error}"
                if callable(note):
                    note(message)
                else:
                    import logging  # noqa: PLC0415

                    logging.getLogger(__name__).error(message)
            raise
        else:
            self.finish_compute(reference + "/complete")

    def verify_anchor(self, head: str | None, *, entries: int | None = None) -> None:
        if head is None or head not in self._heads:
            raise ValueError("Required cost-ledger history is missing or has been rewound")
        if entries is not None and (type(entries) is not int or self._heads[head] != entries):
            raise ValueError("Checkpoint cost-ledger position does not match its anchor")

    def verify_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != LEDGER_VERSION:
            raise ValueError("Checkpoint budget-ledger version differs; legacy history needs audited migration")
        self.verify_anchor(state.get("ledger_head"), entries=state.get("entries"))
        if state.get("spent_usd", math.inf) > self.spent_usd + 1e-9:
            raise ValueError("Actual phase charges cannot decrease after resume")

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": LEDGER_VERSION,
            "plan": self.plan.to_record(),
            "job_id": self.job_id,
            "spent_usd": self.spent_usd,
            "reserved_fixed_usd": self.reserved_fixed_usd,
            "reserved_compute_usd": self.reserved_compute_usd,
            "pending_compute": list(self.pending_compute),
            "committed_usd": self.committed_usd,
            "ledger_head": self.head,
            "entries": len(self.entries),
            "billing_status": (
                "compute_reconciliation_required"
                if any(attempt != self._owned_compute for attempt in self._pending_compute)
                else "compute_in_progress"
                if self._pending_compute
                else "fixed_receipts_pending"
                if self.reserved_fixed_usd
                else "recorded_actual_charges"
            ),
        }


def choose_training_tier(
    n_train_half_games: int,
    forecasts: dict[str, CostForecast],
    *,
    budget_spent_usd: float,
    fixed_phase_cap_usd: float = 650.0,
) -> str:
    if type(n_train_half_games) is not int or n_train_half_games <= 0:
        raise ValueError("Model size requires actual complete train-split game evidence")
    maximum = model_tier(n_train_half_games).lower()
    ladder = ("large", "medium", "small")
    for tier in ladder[ladder.index(maximum) :]:
        estimate = forecasts.get(tier)
        if estimate is None or estimate.phase != "M2":
            continue
        try:
            BudgetPlan("M2", fixed_phase_cap_usd, budget_spent_usd, estimate)
        except BudgetExceededError:
            continue
        return tier
    raise BudgetExceededError("No data-eligible model has a measured forecast within the M2 cap")
