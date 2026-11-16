import pytest

from riichienv import GameRule
from riichienv.kar.contracts import (
    MJSOUL_RULESET,
    Response,
    canonical_bytes,
    game_split,
    missing_parent_mappings,
    model_tier,
)


def test_mjsoul_rules_are_explicit_and_match_engine():
    rule = MJSOUL_RULESET.engine_rule()
    default = GameRule.default_mjsoul()
    assert rule.sanchaho_is_draw is False
    assert rule.allows_ron_on_ankan_for_kokushi_musou is True
    for name in ("dealer_first_discard_is_tedashi", "kuikae_forbidden", "is_suuankou_tanki_double"):
        assert getattr(rule, name) == getattr(default, name)
    assert len(MJSOUL_RULESET.ruleset_hash) == 64
    assert list(Response) == list(range(21))


@pytest.mark.parametrize(
    "count,tier", [(0, "small"), (29999, "small"), (30000, "medium"), (49999, "medium"), (50000, "large")]
)
def test_model_tier_uses_training_games(count, tier):
    assert model_tier(count) == tier


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_bad_training_count_is_rejected(count):
    with pytest.raises(ValueError):
        model_tier(count)


def test_canonical_bytes_and_game_split_are_deterministic():
    assert canonical_bytes({"b": 1, "a": 2}) == canonical_bytes({"a": 2, "b": 1})
    assert game_split("same-game", "salt") == game_split("same-game", "salt")
    assert {game_split(str(i), "salt") for i in range(1000)} == {"train", "validation", "test"}
    with pytest.raises(ValueError):
        canonical_bytes({"value": float("nan")})
    assert any(row["kind"] == "PLAYER_STATE" for row in missing_parent_mappings())
