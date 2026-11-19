from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from riichienv._riichienv import native_pending_kan_commits

from .contracts import canonical_bytes

KAN_TRANSACTION_VERSION = "public_kan_commit_evidence_v2"
COMMITTED_KAN_KINDS = frozenset({"ankan", "kakan", "daiminkan"})


def pending_kan_commits(event: Mapping[str, Any], *, actor: int, committed_kans: int, indicators: int) -> bool:
    """Public commitment evidence, including dora queued by an earlier kan.

    The initial indicator accounts for one slot. If already committed kans still
    have unrevealed indicators, a dora event cannot prove the newest declaration.
    An explicit native commit acknowledgement or the actor's replacement draw can.
    """
    return native_pending_kan_commits(
        canonical_bytes(dict(event)).decode("utf-8"), actor, committed_kans, indicators,
    )
