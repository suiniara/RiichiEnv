"""Reject retired training modes while retaining the PPO and GRP examples."""

from pathlib import Path

import pytest

pytest.importorskip("pydantic", minversion="2.0", reason="Install the optional ML configuration dependencies")
pytest.importorskip("yaml", reason="Install the optional ML configuration dependencies")

import yaml
from pydantic import ValidationError
from riichienv_ml.config import load_config

EXAMPLES = Path(__file__).resolve().parents[2] / "riichienv-ml" / "src" / "riichienv_ml" / "configs"


def _write_config(tmp_path, document):
    path = tmp_path / "training.yml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("section", ["bc", "cql"])
def test_retired_training_sections_are_rejected(tmp_path, section):
    path = _write_config(tmp_path, {section: {"game": {"n_players": 4}, "data_glob": "replays/*.jsonl.gz"}})
    with pytest.raises(ValidationError) as error:
        load_config(path)
    assert any(item["loc"] == (section,) and item["type"] == "extra_forbidden" for item in error.value.errors())


def test_ppo_does_not_silently_accept_the_retired_dqn_algorithm(tmp_path):
    path = _write_config(tmp_path, {"ppo": {"algorithm": "dqn"}})
    with pytest.raises(ValidationError) as error:
        load_config(path)
    assert any(item["loc"] == ("ppo", "algorithm") and item["type"] == "literal_error" for item in error.value.errors())


@pytest.mark.parametrize(
    "field,value",
    [("alpha_cql_init", 1.0), ("capacity", 1000000), ("exploration", "boltzmann"), ("target_update_freq", 2000)],
)
def test_ppo_rejects_retired_dqn_and_cql_options(tmp_path, field, value):
    path = _write_config(tmp_path, {"ppo": {"algorithm": "ppo", field: value}})
    with pytest.raises(ValidationError) as error:
        load_config(path)
    assert any(item["loc"] == ("ppo", field) and item["type"] == "extra_forbidden" for item in error.value.errors())


@pytest.mark.parametrize(
    "example,section,players",
    [
        ("3p/grp.yml", "grp", 3),
        ("4p/grp.yml", "grp", 4),
        ("3p/ppo.yml", "ppo", 3),
        ("4p/ppo.yml", "ppo", 4),
        ("4p/ppo_v2.yml", "ppo", 4),
    ],
)
def test_retained_training_examples_load_with_consistent_game_settings(example, section, players):
    config = getattr(load_config(str(EXAMPLES / example)), section)
    assert config.game.n_players == players
    assert len(config.game.starting_scores) == players
    if section == "ppo":
        assert config.algorithm == "ppo"
        assert config.model.tile_dim == config.game.tile_dim
        assert config.model.num_actions == config.game.num_actions
        assert len(config.pts_weight) == players
