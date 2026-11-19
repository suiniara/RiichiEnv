from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contracts import content_hash
from ..evaluation_authorization import exclusive_job
from ..versioning import validate_artifact_contract
from .model import KARTransformer

PRODUCTION_CHECKPOINT_VERSION = "kar_production_exact_resume_v1"
TRAINING_STARTUP_VERSION = "kar_training_startup_v1"


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_startup(directory: Path, record: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in record.items() if key != "startup_sha256"}
    signed = {**unsigned, "startup_sha256": content_hash(unsigned)}
    target = directory / "startup.json"
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(signed, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    _sync_directory(directory)
    return signed


def create_training_directory(directory: Path, *, identity: dict[str, Any], ledger_path: Path) -> None:
    """Publish a run identity before an output directory can contain any computation."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_job(directory.with_name("." + directory.name + ".startup.lock")):
        if directory.exists():
            raise ValueError("Refuse to overwrite a training run")
        previous = ledger_path.read_bytes() if ledger_path.exists() else None
        origin = (
            {"bytes": len(previous), "sha256": hashlib.sha256(previous).hexdigest()}
            if previous is not None
            else None
        )
        with tempfile.TemporaryDirectory(prefix="." + directory.name + ".initializing-", dir=directory.parent) as tmp:
            staging = Path(tmp)
            _write_startup(staging, {
                "version": TRAINING_STARTUP_VERSION,
                "identity": identity,
                "phase": "preparing",
                "ledger_origin": origin,
                "budget_ledger": None,
            })
            staging.rename(directory)
            _sync_directory(directory.parent)


def read_training_startup(
    directory: Path, *, identity: dict[str, Any], ledger_path: Path, checkpoint_exists: bool
) -> dict[str, Any] | None:
    """Only a recorded pre-checkpoint initialization may restart without weights."""
    path = directory / "startup.json"
    if not path.is_file():
        if checkpoint_exists:
            return None  # Older committed runs still use the complete checkpoint contract.
        raise ValueError("Resume requires a committed checkpoint or a recorded initialization; never restart silently")
    if path.stat().st_size > 1024**2:
        raise ValueError("Oversized training startup record")
    record = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in record.items() if key != "startup_sha256"}
    if (
        record.get("version") != TRAINING_STARTUP_VERSION
        or record.get("startup_sha256") != content_hash(unsigned)
        or record.get("identity") != identity
        or record.get("phase") not in {"preparing", "initializing", "checkpoint_committed"}
    ):
        raise ValueError("Training startup identity/phase differs; preserve the original run and ledger")
    if (
        (record["phase"] == "preparing" and record.get("budget_ledger") is not None)
        or (record["phase"] != "preparing" and not isinstance(record.get("budget_ledger"), dict))
    ):
        raise ValueError("An initialized run requires its original budget-ledger anchor")
    origin = record["ledger_origin"]
    if origin is not None:
        if (
            not isinstance(origin, dict) or type(origin.get("bytes")) is not int or origin["bytes"] < 0
            or not isinstance(origin.get("sha256"), str) or len(origin["sha256"]) != 64
        ):
            raise ValueError("Invalid shared-ledger origin evidence")
        with ledger_path.open("rb") as stream:
            prefix = stream.read(origin["bytes"])
        if len(prefix) != origin["bytes"] or hashlib.sha256(prefix).hexdigest() != origin["sha256"]:
            raise ValueError("The shared phase ledger predating this startup is missing or changed")
    if not checkpoint_exists:
        if record["phase"] == "checkpoint_committed":
            raise ValueError("A committed checkpoint is missing; initialization cannot replace trained weights")
        if (directory / "run_manifest.json").exists() or any(
            path.stat().st_size for path in directory.glob("updates-attempt-*.jsonl")
        ):
            raise ValueError("Training update evidence forbids initialization restart")
        for generation in directory.glob("checkpoint.g*.pt"):
            if not generation.name.startswith("checkpoint.g000000000000-"):
                raise ValueError("A post-initialization checkpoint generation forbids restart")
    return record


def advance_training_startup(
    directory: Path, record: dict[str, Any], *, phase: str, budget_ledger: dict[str, Any]
) -> dict[str, Any]:
    transitions = {
        "preparing": {"initializing"},
        "initializing": {"initializing", "checkpoint_committed"},
        "checkpoint_committed": {"checkpoint_committed"},
    }
    if phase not in transitions[record["phase"]]:
        raise ValueError("Training startup phases cannot be rewound")
    if phase == "checkpoint_committed" and not (directory / "checkpoint.json").is_file():
        raise ValueError("Only a committed checkpoint may close initialization")
    return _write_startup(directory, {**record, "phase": phase, "budget_ledger": budget_ledger})


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
