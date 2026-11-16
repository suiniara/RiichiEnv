from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, TypedDict

from riichienv import GameRule, RiichiEnv

REFERENCE_SCHEMA_ID = "kar-v1.2.2"
SCHEMA_ID = "kar-riichienv-v1"
ENGINE_COMMIT = "0ee1236a283b15508f3ac28407027f340b81e1de"
SCHEMA_OVERLAY_SHA256 = "90286d088014eb72e9f99f1b3c35d0a7553bcf5fec99e8aeb3ba5f02a010716a"
RESPONSE_MASK_VERSION = "public_rule_response_mask_v2_mjsoul_nagashi"
GAME_MODE = "4p-red-half"
MAX_CANDIDATES = 32
KIND_NAMES = (
    "PAD",
    "BOS",
    "ROUND_STATE",
    "PLAYER_STATE",
    "HAND_COUNT",
    "MELD_STATE",
    "DORA_STATE",
    "DECISION_CONTEXT",
    "LEGAL_SET",
    "DECIDE",
    "ACTION",
    "ROUND_START",
    "DRAW_SELF",
    "DRAW_OTHER",
    "DISCARD",
    "CHI_ACCEPT",
    "PON_ACCEPT",
    "DAIMINKAN_ACCEPT",
    "ANKAN_DECLARE",
    "ANKAN_ACCEPT",
    "KAKAN_DECLARE",
    "KAKAN_ACCEPT",
    "RIICHI_DECLARE",
    "RIICHI_ACCEPT",
    "DORA_REVEAL",
    "SCORE_DELTA",
    "SELF_PASS",
    "WIN_RESOLVED",
    "ROUND_END",
    "PUBLIC_HAND_REVEAL",
)


class Response(IntEnum):
    NEXT_DRAW = 0
    CHI_P1 = 1
    PON_P1 = 2
    PON_P2 = 3
    PON_P3 = 4
    DAIMINKAN_P1 = 5
    DAIMINKAN_P2 = 6
    DAIMINKAN_P3 = 7
    RON_MASK_1 = 8
    RON_MASK_2 = 9
    RON_MASK_3 = 10
    RON_MASK_4 = 11
    RON_MASK_5 = 12
    RON_MASK_6 = 13
    RON_MASK_7 = 14
    EXHAUSTIVE_DRAW = 15
    ABORT_FOUR_WINDS = 16
    ABORT_FOUR_RIICHI = 17
    ABORT_FOUR_KANS = 18
    ABORT_TRIPLE_RON = 19
    NAGASHI_MANGAN = 20


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class Ruleset:
    # v2 makes the FAQ-backed nagashi interpretation explicit in every artifact
    # binding; own calls/kans do not disqualify (docs/kar-nagashi-rules.md).
    name: str = "mahjong_soul_4p_red_half_v2"
    engine_commit: str = ENGINE_COMMIT
    game_mode: str = GAME_MODE
    rank_tiebreak: str = "initial_seat_order_east_south_west_north"
    allows_ron_on_ankan_for_kokushi_musou: bool = True
    is_kokushi_musou_13machi_double: bool = True
    is_suuankou_tanki_double: bool = True
    is_junsei_chuurenpoutou_double: bool = True
    is_daisuushii_double: bool = True
    yakuman_pao_is_liability_only: bool = True
    sanchaho_is_draw: bool = False
    kuikae_forbidden: bool = True
    dealer_first_discard_is_tedashi: bool = True
    chi_allowed: bool = True
    four_winds_abort: bool = True
    four_riichi_abort: bool = True
    four_kans_abort: bool = True
    nagashi_mangan: bool = True

    @property
    def ruleset_hash(self) -> str:
        return content_hash(asdict(self))

    def engine_rule(self) -> GameRule:
        rule = GameRule.default_mjsoul()
        for name in (
            "allows_ron_on_ankan_for_kokushi_musou",
            "is_kokushi_musou_13machi_double",
            "is_suuankou_tanki_double",
            "is_junsei_chuurenpoutou_double",
            "is_daisuushii_double",
            "yakuman_pao_is_liability_only",
            "sanchaho_is_draw",
            "kuikae_forbidden",
            "dealer_first_discard_is_tedashi",
        ):
            setattr(rule, name, getattr(self, name))
        return rule

    def create_env(self, seed: int | None = None) -> RiichiEnv:
        if not all(
            (self.chi_allowed, self.four_winds_abort, self.four_riichi_abort, self.four_kans_abort, self.nagashi_mangan)
        ):
            raise ValueError("The frozen engine does not expose overrides for these public rule switches")
        if self.game_mode != GAME_MODE:
            raise ValueError("KAR M0 requires four-player red hanchan")
        return RiichiEnv(game_mode=self.game_mode, seed=seed, rule=self.engine_rule())


class CohortContract(TypedDict):
    cohort_id: str
    category: int
    mode: int
    mode_ids: list[int]
    seats: list[int]
    golden_games_excluded_from_training: bool


MJSOUL_RULESET = Ruleset()
PRIMARY_COHORT: CohortContract = {
    "cohort_id": "mjsoul_jade_4p_hanchan_v1",
    "category": 2,
    "mode": 2,
    "mode_ids": [12],
    "seats": [0, 1, 2, 3],
    "golden_games_excluded_from_training": True,
}


def cohort_hash() -> str:
    return content_hash(PRIMARY_COHORT)


def matches_primary_cohort(metadata: dict[str, Any]) -> bool:
    return (
        metadata.get("category") == PRIMARY_COHORT["category"]
        and metadata.get("mode") == PRIMARY_COHORT["mode"]
        and metadata.get("mode_id") in PRIMARY_COHORT["mode_ids"]
        and sorted(metadata.get("account_seats", [])) == PRIMARY_COHORT["seats"]
    )


def model_tier(n_train_half_games: int) -> str:
    if isinstance(n_train_half_games, bool) or not isinstance(n_train_half_games, int) or n_train_half_games < 0:
        raise ValueError("N_train_half_games must be a nonnegative integer")
    if n_train_half_games < 30_000:
        return "small"
    if n_train_half_games < 50_000:
        return "medium"
    return "large"


def game_split(game_id: str, salt: str) -> str:
    if not game_id or not salt:
        raise ValueError("game_id and split salt are required")
    bucket = int.from_bytes(hashlib.sha256(canonical_bytes([salt, game_id])).digest()[:8], "big") % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def missing_parent_mappings() -> list[dict[str, Any]]:
    return [
        {
            "kind_id": kind_id,
            "kind": kind,
            "missing": ["legacy_field_applicability", "legacy_stage_values", "legacy_flags", "legacy_numeric_presence"],
            "available_override": kind in {"HAND_COUNT", "ROUND_STATE"},
        }
        for kind_id, kind in enumerate(KIND_NAMES)
        if kind not in {"PAD", "BOS", "DECIDE", "ACTION"}
    ]
