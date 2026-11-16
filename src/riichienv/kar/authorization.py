from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import content_hash
from .readiness import validate_readiness

AUTHORIZATION_VERSION = "kar_explicit_update_authorization_v1"


class ParameterUpdatesNotAuthorizedError(PermissionError):
    pass


@dataclass(frozen=True)
class UpdateAuthorization:
    document: Mapping[str, Any]


def update_implementation_sha256() -> str:
    root = Path(__file__).parent
    material = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.suffix in {".py", ".json"}
    }
    return content_hash(material)


def load_update_authorization(path: Path | None) -> UpdateAuthorization | None:
    if path is None:
        return None
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ParameterUpdatesNotAuthorizedError("Missing or oversized explicit authorization")
    return UpdateAuthorization(json.loads(path.read_text(encoding="utf-8")))


def require_parameter_updates(
    authorization: UpdateAuthorization | None,
    *,
    operation: str,
    dataset_hash: str | None = None,
    phase: str | None = None,
    recipe_sha256: str | None = None,
) -> None:
    """Workflow boundary for our update entrypoints, not a sandbox for arbitrary torch code."""
    if not isinstance(authorization, UpdateAuthorization):
        raise ParameterUpdatesNotAuthorizedError(
            "Parameter updates are disabled; data PASS is not user training permission"
        )
    doc = authorization.document
    if not isinstance(doc, Mapping):
        raise ParameterUpdatesNotAuthorizedError("Authorization must be an explicit structured document")
    try:
        unsigned = {key: value for key, value in doc.items() if key != "authorization_sha256"}
        if doc.get("version") != AUTHORIZATION_VERSION or doc.get("authorization_sha256") != content_hash(unsigned):
            raise ValueError("Authorization integrity/version mismatch")
        if (
            doc.get("explicit_user_approval") is not True
            or not isinstance(doc.get("approval_reference"), str)
            or not doc["approval_reference"].strip()
        ):
            raise ValueError("Missing explicit user approval reference")
        readiness = doc["readiness"]
        validate_readiness(readiness)
        if readiness.get("data_eligible") is not True or readiness.get("resource_ready") is not True:
            raise ValueError("Data eligibility and resource readiness must be separately established")
        if readiness.get("full_feature_ready") is not True:
            raise ValueError("Complete feature readiness is required, not only core or data readiness")
        if readiness.get("source_hash") != update_implementation_sha256():
            raise ValueError("Feature evidence belongs to another source implementation")
        operations = doc["allowed_operations"]
        if (
            not isinstance(operations, (list, tuple))
            or any(not isinstance(item, str) for item in operations)
            or operation not in operations
        ):
            raise ValueError("The requested operation was not authorized")
        if not isinstance(doc.get("dataset_manifest_sha256"), str) or not doc["dataset_manifest_sha256"]:
            raise ValueError("Authorization must bind a dataset")
        if phase is not None:
            phases = doc.get("allowed_phases")
            if not isinstance(phases, list) or phase not in phases:
                raise ValueError("This training phase was not explicitly authorized")
        if recipe_sha256 is not None:
            recipes = doc.get("allowed_recipe_sha256")
            if not isinstance(recipes, list) or recipe_sha256 not in recipes:
                raise ValueError("Training configuration differs from the approved frozen recipe")
        if dataset_hash is not None and dataset_hash != doc["dataset_manifest_sha256"]:
            raise ValueError("Authorization dataset differs")
    except (ValueError, KeyError, TypeError) as error:
        raise ParameterUpdatesNotAuthorizedError(str(error)) from error
