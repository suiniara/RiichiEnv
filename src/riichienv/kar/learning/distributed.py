from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as _torch_distributed

# PyTorch exposes backend symbols conditionally; the narrow adapter also contains
# the version-specific native Gloo factory used by Windows torchrun jobs.
dist: Any = _torch_distributed


class DistributedStageError(RuntimeError):
    pass


def launch_topology() -> tuple[int, int, int]:
    rank, world, local = (
        int(os.environ.get(key, default)) for key, default in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))
    )
    if world < 1 or not 0 <= rank < world or local < 0:
        raise ValueError("Invalid torchrun topology")
    return rank, world, local


def selected_backend(device: str, backend: str | None, gloo_bind_host: str | None) -> str:
    selected = backend or ("nccl" if device == "cuda" and dist.is_nccl_available() else "gloo")
    if selected == "gloo" and gloo_bind_host:
        return "kar_gloo_" + hashlib.sha256(gloo_bind_host.encode()).hexdigest()[:12]
    return selected


@dataclass
class StartupCoordinator:
    """Coordinate failures before CUDA/process-group creation using the shared run filesystem.

    Torchrun supplies a unique run ID (including --standalone). Other launchers
    must supply KAR_DISTRIBUTED_RUN_ID. Receipts are never reused or overwritten;
    the primary's release binds every participant's fresh nonce.
    """

    rank: int
    world_size: int
    local_rank: int
    directory: Path | None
    nonce: str
    timeout_seconds: float = 600.0
    _stage: int = 0

    @classmethod
    def open(cls, output: Path, *, timeout_seconds: float = 600.0) -> StartupCoordinator:
        rank, world, local = launch_topology()
        nonce = uuid.uuid4().hex
        if world == 1:
            return cls(rank, world, local, None, nonce, timeout_seconds)
        run_id = os.environ.get("TORCHELASTIC_RUN_ID")
        if not run_id or run_id == "none":
            run_id = os.environ.get("KAR_DISTRIBUTED_RUN_ID")
        if not run_id or run_id == "none":
            raise ValueError("Distributed startup requires a unique torchrun or KAR_DISTRIBUTED_RUN_ID")
        launch = [
            run_id,
            os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
            world,
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
            str(output.resolve()),
        ]
        key = hashlib.sha256(json.dumps(launch).encode()).hexdigest()
        directory = output.parent / ("." + output.name + ".launch-" + key)
        directory.mkdir(parents=True, exist_ok=True)
        result = cls(rank, world, local, directory, nonce, timeout_seconds)
        result._publish(f"rank-{rank}.json", {"nonce": nonce, "world_size": world})
        return result

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def _publish(self, name: str, record: dict[str, Any]) -> None:
        assert self.directory is not None
        target = self.directory / name
        temporary = self.directory / (name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # Atomic, exclusive publication: a stale or concurrent launch cannot replace a receipt.
                os.link(temporary, target)
            except FileExistsError as error:
                raise ValueError("Distributed startup receipt already exists; use a fresh launch ID") from error
        finally:
            temporary.unlink(missing_ok=True)

    def _read(self, name: str) -> dict[str, Any] | None:
        assert self.directory is not None
        path = self.directory / name
        if not path.exists():
            return None
        if path.stat().st_size > 1024**2:
            raise ValueError("Oversized distributed startup receipt")
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Invalid distributed startup receipt")
        return result

    def coordinate(self, error: Exception | None, stage: str) -> None:
        detail = f"{type(error).__name__}: {error}" if error is not None else None
        if self.world_size == 1:
            if error is not None:
                raise DistributedStageError(f"{stage} aborted on all ranks: {detail}") from error
            return
        number = self._stage
        self._stage += 1
        deadline = time.monotonic() + self.timeout_seconds
        try:
            leader = self._read("rank-0.json")
            while leader is None:
                failure = self._read(f"stage-{number}-failure-0.json")
                if failure is not None:
                    raise DistributedStageError(str(failure.get("error", "startup primary failed")))
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for distributed startup primary")
                time.sleep(0.05)
                leader = self._read("rank-0.json")
            # A fresh primary cannot count a surviving report from an older, partial launch.
            primary_nonce = leader["nonce"]
            self._publish(
                f"stage-{number}-rank-{self.rank}.json",
                {
                    "nonce": self.nonce,
                    "primary_nonce": primary_nonce,
                    "stage": stage,
                    "error": detail,
                },
            )
            while True:
                nonces = []
                for rank in range(self.world_size):
                    failure = self._read(f"stage-{number}-failure-{rank}.json")
                    if failure is not None:
                        raise DistributedStageError(str(failure.get("error", "startup peer failed")))
                    participant = self._read(f"rank-{rank}.json")
                    report = self._read(f"stage-{number}-rank-{rank}.json")
                    if participant is None or report is None:
                        continue
                    if (
                        participant.get("world_size") != self.world_size
                        or report.get("nonce") != participant.get("nonce")
                        or report.get("stage") != stage
                        or report.get("primary_nonce") != primary_nonce
                    ):
                        raise ValueError("Distributed startup participants/stages differ")
                    if report.get("error") is not None:
                        raise DistributedStageError(str(report["error"]))
                    nonces.append(participant["nonce"])
                release_name = f"stage-{number}-release.json"
                if self.primary and len(nonces) == self.world_size:
                    self._publish(release_name, {"stage": stage, "nonces": nonces})
                release = self._read(release_name)
                # The files are not an atomic snapshot: a peer report and the
                # release may appear after this scan already missed that report.
                # Re-scan before comparing a complete release with partial nonces.
                if release is not None and len(nonces) == self.world_size:
                    if (
                        release.get("stage") != stage
                        or release.get("nonces") != nonces
                        or nonces[self.rank] != self.nonce
                    ):
                        raise ValueError("Distributed startup release belongs to another launch")
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for distributed startup peers")
                time.sleep(0.05)
        except Exception as caught:
            # Publish local protocol/time-out errors too, so peers need not wait for their own deadline.
            try:
                self._publish(f"stage-{number}-failure-{self.rank}.json", {"error": str(caught)})
            except Exception:
                pass
            raise DistributedStageError(f"{stage} aborted on all ranks: {caught}") from (error or caught)


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    owns_group: bool
    _control: torch.Tensor

    @classmethod
    def open(cls, device: str, *, backend: str | None = None, gloo_bind_host: str | None = None) -> DistributedContext:
        rank, world, local = launch_topology()
        target = torch.device("cuda", local) if device == "cuda" else torch.device("cpu")
        if device == "cuda":
            torch.cuda.set_device(target)
        owned = False
        if world > 1 and not dist.is_initialized():
            selected = selected_backend(device, backend, gloo_bind_host)
            if selected.startswith("kar_gloo_") and gloo_bind_host:
                name = selected

                def factory(store, worker_rank, size, timeout):
                    options = dist.ProcessGroupGloo._Options()
                    options._timeout = timeout
                    options._devices = [dist.ProcessGroupGloo.create_device(hostname=gloo_bind_host)]
                    options._threads = 1
                    return dist.ProcessGroupGloo(store, worker_rank, size, options)

                if name not in dist.Backend.backend_list:
                    dist.Backend.register_backend(name, factory, devices=["cpu", "cuda"])
                selected = name
            dist.init_process_group(backend=selected, rank=rank, world_size=world, timeout=timedelta(minutes=10))
            owned = True
        if world > 1 and (dist.get_rank() != rank or dist.get_world_size() != world):
            raise ValueError("Existing distributed group differs from this launch")
        control_device = target if world > 1 and str(dist.get_backend()) == "nccl" else torch.device("cpu")
        return cls(rank, world, local, target, owned, torch.zeros(1, dtype=torch.int32, device=control_device))

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def coordinate(self, error: Exception | None, stage: str) -> None:
        self._control.fill_(1 if error is not None else 0)
        if self.world_size > 1:
            dist.all_reduce(self._control, op=dist.ReduceOp.MAX)
        if self._control.item():
            detail = f"{type(error).__name__}: {error}" if error is not None else "another rank failed"
            raise DistributedStageError(f"{stage} aborted on all ranks: {detail}") from error

    def gather(self, value: Any) -> list[Any]:
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def gather_to_primary(self, value: Any) -> list[Any] | None:
        if self.world_size == 1:
            return [value]
        values: list[Any] | None = [None] * self.world_size if self.primary else None
        dist.gather_object(value, values, dst=0)
        return values

    def broadcast_from_primary(self, value: Any) -> Any:
        if self.world_size == 1:
            return value
        values = [value if self.primary else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def global_count(self, local_count: int) -> int:
        count = torch.tensor([local_count], dtype=torch.int64, device=self._control.device)
        if self.world_size > 1:
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return int(count.item())

    @torch.no_grad()
    def synchronize_model(self, model: torch.nn.Module) -> None:
        if self.world_size == 1:
            return
        native = str(dist.get_backend()) == "nccl"
        for tensor in (*model.parameters(), *model.buffers()):
            data = tensor.detach()
            transport = data if native or data.device.type == "cpu" else data.cpu()
            dist.broadcast(transport, src=0)
            if transport.device != data.device:
                data.copy_(transport)

    def average_gradients(self, model: torch.nn.Module) -> None:
        error = None
        for parameter in model.parameters():
            if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                error = FloatingPointError("Missing/non-finite local gradient")
                break
        self.coordinate(error, "local gradient validation")
        if self.world_size > 1:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    gradient = parameter.grad
                    native = str(dist.get_backend()) == "nccl"
                    transport = gradient if native or gradient.device.type == "cpu" else gradient.detach().cpu()
                    dist.all_reduce(transport, op=dist.ReduceOp.SUM)
                    transport.div_(self.world_size)
                    if transport.device != gradient.device:
                        gradient.copy_(transport)
        error = None
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        ):
            error = FloatingPointError("Non-finite reduced gradient")
        self.coordinate(error, "reduced gradient validation")

    def close(self) -> None:
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
