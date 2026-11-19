from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contracts import content_hash
from .config import MODEL_VERSION
from .export import learning_contract
from .model import KARTransformer
from .production_checkpoint import _generation, _publish_checkpoint

CHECKPOINT_VERSION = "kar_exact_resume_v2"


def implementation_identity() -> dict[str, Any]:
    package = Path(__file__).parent
    files = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in (
            "config.py",
            "model.py",
            "batch.py",
            "losses.py",
            "metrics.py",
            "train.py",
            "checkpoint.py",
            "production_checkpoint.py",
            "dataset.py",
        )
    }
    root = Path(__file__).parents[4]
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {
        "implementation_sha256": content_hash(files),
        "code_commit": result.stdout.strip() if result.returncode == 0 else None,
        "source_files": files,
    }


def _rng_state() -> dict[str, Any]:
    algorithm, keys, position, gaussian, cached = np.random.get_state()
    if not isinstance(keys, np.ndarray):
        raise ValueError("Expected the legacy NumPy RandomState key array")
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
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state(
        (
            numpy["algorithm"],
            numpy["keys"].numpy().astype(np.uint32),
            numpy["position"],
            numpy["gaussian"],
            numpy["cached"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_initialized():
            raise ValueError("An exact CUDA resume cannot silently become a CPU run")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: KARTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    run_contract: dict[str, Any],
    sampler_state: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_version": MODEL_VERSION,
        "model_config": asdict(model.config),
        "model_contract_hash": model.config.contract_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **learning_contract(),
        **implementation_identity(),
        "run_contract": run_contract,
        "step": step,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
    }
    payload = {
        "metadata": metadata,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler_state,
        "rng": _rng_state(),
    }
    return _publish_checkpoint(path, payload, metadata, step=step)


def load_checkpoint(
    path: Path,
    model: KARTransformer,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    *,
    run_contract: dict[str, Any],
    restore_rng: bool = True,
) -> dict[str, Any]:
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    metadata = sidecar["metadata"]
    if metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Legacy checkpoint format requires explicit migration; exact resume refused")
    generation = _generation(path, sidecar.get("checkpoint_file"))
    if not generation.is_file() or generation.stat().st_size > 2 * 1024**3:
        raise ValueError("Missing or oversized checkpoint generation")
    if sidecar.get("checkpoint_sha256") != hashlib.sha256(generation.read_bytes()).hexdigest() or sidecar.get(
        "metadata_sha256"
    ) != content_hash(metadata):
        raise ValueError("Checkpoint integrity metadata mismatch")
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_version": MODEL_VERSION,
        "model_config": asdict(model.config),
        "model_contract_hash": model.config.contract_hash,
        **learning_contract(),
        **implementation_identity(),
        "run_contract": run_contract,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Schema/data/model/recipe/runtime mismatch; exact resume refused")
    payload = torch.load(generation, map_location="cpu", weights_only=True)
    if payload["metadata"] != metadata:
        raise ValueError("Checkpoint payload metadata conflict")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        _restore_rng(payload["rng"])
    return {"step": metadata["step"], "sampler": payload["sampler"], "metadata": metadata}
