from __future__ import annotations

import hashlib
import json
import os
import random
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contracts import content_hash
from ..versioning import validate_artifact_contract
from .model import KARTransformer

PRODUCTION_CHECKPOINT_VERSION = "kar_production_exact_resume_v1"


def rank_rng_state(device: torch.device) -> dict[str, Any]:
    algorithm, keys, position, gaussian, cached = np.random.get_state()
    if not isinstance(keys, np.ndarray):
        raise ValueError("Expected NumPy RandomState key array")
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "keys": torch.as_tensor(keys, dtype=torch.int64),
            "position": position,
            "gaussian": gaussian,
            "cached": cached,
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
    }


def restore_rank_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state(
        (
            value["algorithm"],
            value["keys"].cpu().numpy().astype(np.uint32),
            value["position"],
            value["gaussian"],
            value["cached"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if (state["cuda"] is not None) != (device.type == "cuda"):
        raise ValueError("Exact CPU/CUDA resume cannot change runtime")
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def _generation(path: Path, name: str) -> Path:
    pattern = re.escape(path.stem) + r"\.g[0-9]{12}-[0-9a-f]{12}\.pt"
    if not isinstance(name, str) or re.fullmatch(pattern, name) is None:
        raise ValueError("Checkpoint pointer escapes its owned generation namespace")
    return path.parent / name


def save_production_checkpoint(
    path: Path,
    model: KARTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    contract: dict[str, Any],
    trainer_state: dict[str, Any],
    rank_rng: list[dict[str, Any]],
    runtime: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    validate_artifact_contract(contract, context="production checkpoint")
    if len(rank_rng) != runtime["world_size"]:
        raise ValueError("Every distributed rank needs its own RNG state")
    metadata = {
        "checkpoint_version": PRODUCTION_CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "model_contract_hash": model.config.contract_hash,
        "parameter_count": model.config.parameter_count,
        "release_contract": contract,
        "runtime": runtime,
        "step": step,
    }
    payload = {
        "metadata": metadata,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "trainer": trainer_state,
        "rank_rng": rank_rng,
        "scaler": None,
    }
    pointer = path.with_suffix(".json")
    previous = None
    if pointer.exists():
        previous = json.loads(pointer.read_text(encoding="utf-8"))
        old = _generation(path, previous["checkpoint_file"])
        if previous.get("checkpoint_sha256") != hashlib.sha256(old.read_bytes()).hexdigest():
            raise ValueError("Previous committed checkpoint was modified")
    generation = _generation(path, f"{path.stem}.g{step:012d}-{uuid.uuid4().hex[:12]}.pt")
    with generation.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    sidecar = {
        "metadata": metadata,
        "metadata_sha256": content_hash(metadata),
        "checkpoint_file": generation.name,
        "checkpoint_sha256": hashlib.sha256(generation.read_bytes()).hexdigest(),
        "previous_generation": {"file": previous["checkpoint_file"], "sha256": previous["checkpoint_sha256"]}
        if previous is not None
        else None,
    }
    temporary = pointer.with_name(pointer.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(sidecar, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(pointer)
    # Retain the current and preceding complete generation. An interrupted write
    # never replaces the last committed pointer; uncommitted orphan files stay intact.
    stale = previous.get("previous_generation") if previous is not None else None
    if stale is not None:
        target = _generation(path, stale["file"])
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == stale["sha256"]:
            target.unlink()
    return sidecar


def read_production_checkpoint(path: Path, *, contract: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    validate_artifact_contract(contract, context="requested production checkpoint")
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    metadata = sidecar["metadata"]
    validate_artifact_contract(metadata.get("release_contract", {}), context="stored production checkpoint")
    generation = _generation(path, sidecar["checkpoint_file"])
    if not generation.is_file() or generation.stat().st_size > 2 * 1024**3:
        raise ValueError("Missing/oversized checkpoint generation")
    if sidecar.get("checkpoint_sha256") != hashlib.sha256(generation.read_bytes()).hexdigest() or sidecar.get(
        "metadata_sha256"
    ) != content_hash(metadata):
        raise ValueError("Checkpoint integrity mismatch")
    if (
        metadata.get("checkpoint_version") != PRODUCTION_CHECKPOINT_VERSION
        or metadata.get("release_contract") != contract
        or metadata.get("runtime") != runtime
    ):
        raise ValueError("Data/schema/model/recipe/build/precision/topology mismatch; exact resume refused")
    payload = torch.load(generation, map_location="cpu", weights_only=True)
    if payload["metadata"] != metadata or len(payload["rank_rng"]) != runtime["world_size"]:
        raise ValueError("Payload/sidecar conflict")
    return payload
