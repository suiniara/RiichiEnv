from __future__ import annotations

import hashlib

CALIBRATION_SPLIT_SALT = "kar-cal-fit-check-v1"
CALIBRATION_PARTITION_VERSION = "immutable_cal_fit_selection_cal_check_diagnostics_v1"


def calibration_is_fit(game_id: str, *, salt: str = CALIBRATION_SPLIT_SALT) -> bool:
    if not isinstance(game_id, str) or not game_id or not isinstance(salt, str) or not salt:
        raise ValueError("Calibration roles require an explicit stable game ID and hash salt")
    return int.from_bytes(hashlib.sha256((salt + "\0" + game_id).encode()).digest()[:8], "big") % 2 == 0
