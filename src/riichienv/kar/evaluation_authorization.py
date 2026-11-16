from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .authorization import update_implementation_sha256
from .contracts import content_hash

EVALUATION_AUTHORIZATION_VERSION = "explicit_frozen_evaluation_authorization_v1"


def require_evaluation_authorization(document: dict[str, Any] | None, *, operation: str, plan_sha256: str) -> None:
    if document is None:
        raise PermissionError(
            "Evaluation is not implicitly started by writing code; explicit evaluation approval required"
        )
    unsigned = {key: value for key, value in document.items() if key != "authorization_sha256"}
    if document.get("version") != EVALUATION_AUTHORIZATION_VERSION or document.get(
        "authorization_sha256"
    ) != content_hash(unsigned):
        raise PermissionError("Evaluation authorization integrity mismatch")
    operations = document.get("allowed_operations")
    if not isinstance(operations, list) or any(not isinstance(value, str) for value in operations):
        raise PermissionError("Evaluation operations must be an explicit exact-name list")
    if (
        document.get("explicit_user_approval") is not True
        or not document.get("approval_reference")
        or operation not in document.get("allowed_operations", [])
    ):
        raise PermissionError("This evaluation operation was not approved")
    if document.get("plan_sha256") != plan_sha256 or document.get("source_sha256") != update_implementation_sha256():
        raise PermissionError("Evaluation plan or source differs from the approved frozen job")


@contextmanager
def exclusive_job(path: Path) -> Iterator[None]:
    """OS-released advisory lease; a crashed process does not leave an unrecoverable PID lock."""
    stream = path.open("a+b")
    try:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        import os  # noqa: PLC0415

        if os.name == "nt":
            import msvcrt  # noqa: PLC0415

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def read_evaluation_authorization(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 1024**2:
        raise PermissionError("Missing/oversized evaluation approval")
    return json.loads(path.read_text(encoding="utf-8"))
