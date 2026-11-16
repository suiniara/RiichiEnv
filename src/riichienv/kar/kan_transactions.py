from __future__ import annotations

from collections.abc import Mapping
from typing import Any

KAN_TRANSACTION_VERSION = "public_kan_commit_evidence_v2"
COMMITTED_KAN_KINDS = frozenset({"ankan", "kakan", "daiminkan"})


def pending_kan_commits(event: Mapping[str, Any], *, actor: int, committed_kans: int, indicators: int) -> bool:
    """Public commitment evidence, including dora queued by an earlier kan.

    The initial indicator accounts for one slot. If already committed kans still
    have unrevealed indicators, a dora event cannot prove the newest declaration.
    An explicit native commit acknowledgement or the actor's replacement draw can.
    """
    typ = event["type"]
    if typ in {"tsumo", "kan_commit"}:
        if event.get("actor") != actor:
            raise ValueError("Kan commitment/replacement draw belongs to another actor")
        return True
    if typ != "dora":
        return False
    if committed_kans < 0 or indicators < 1 or indicators > committed_kans + 1:
        raise ValueError("Public kan/indicator accounting is inconsistent")
    return indicators == committed_kans + 1
