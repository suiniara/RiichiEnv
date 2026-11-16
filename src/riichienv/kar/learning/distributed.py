from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as _torch_distributed

# PyTorch exposes backend symbols conditionally; the narrow adapter also contains
# the version-specific native Gloo factory used by Windows torchrun jobs.
dist: Any = _torch_distributed


class DistributedStageError(RuntimeError):
    pass


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
        rank, world, local = (
            int(os.environ.get(key, default))
            for key, default in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))
        )
        if world < 1 or not 0 <= rank < world:
            raise ValueError("Invalid torchrun topology")
        target = torch.device("cuda", local) if device == "cuda" else torch.device("cpu")
        if device == "cuda":
            torch.cuda.set_device(target)
        owned = False
        if world > 1 and not dist.is_initialized():
            selected = backend or ("nccl" if device == "cuda" and dist.is_nccl_available() else "gloo")
            if selected == "gloo" and gloo_bind_host:
                name = "kar_gloo_" + hashlib.sha256(gloo_bind_host.encode()).hexdigest()[:12]

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
