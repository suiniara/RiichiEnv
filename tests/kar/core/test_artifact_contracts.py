import math
from dataclasses import replace

import pytest

from riichienv.kar.calibration import calibration_split_from_manifest, choose_validation_tau
from riichienv.kar.contracts import content_hash, game_split
from riichienv.kar.policy import fixed_tau_grid
from riichienv.kar.runtime_contracts import runtime_contract
from riichienv.kar.support import support_from_train_games

SALT = "kar-riichienv-v1-games"


def _selection(split, count):
    games = []
    index = 0
    while len(games) < count:
        identity = f"synthetic-contract-fixture-{index}"
        if game_split(identity, SALT) == split:
            games.append({"game_id": identity, "source_role": "main_cohort_candidate"})
        index += 1
    current = runtime_contract()
    record = {
        "scope": dict(current),
        "games": games,
    }
    record["scope"]["split_salt"] = SALT
    return {**record, "manifest_sha256": content_hash(games)}


def test_independent_train_support_cannot_be_filled_by_validation_or_repeated_decisions():
    document = _selection("train", 500)
    identities = [row["game_id"] for row in document["games"]]
    observed = {game: [3, 3, 3] for game in identities[:499]}
    support = support_from_train_games(document, observed, source_reference="synthetic contract fixture")
    assert support.train_game_counts[3] == 499 and not support.supported[3]
    foreign = _selection("validation", 1)["games"][0]["game_id"]
    with pytest.raises(ValueError):
        support_from_train_games(document, {**observed, foreign: [3]}, source_reference="invalid")
    support = support_from_train_games(
        document, {**observed, identities[499]: [3]}, source_reference="synthetic fixture"
    )
    assert support.train_game_counts[3] == 500 and support.supported[3]


def test_validation_manifest_factory_rejects_train_split_and_hash_tampering():
    document = _selection("validation", 20)
    split = calibration_split_from_manifest(document)
    assert split.validation_manifest_sha256 == document["manifest_sha256"]
    with pytest.raises(ValueError):
        calibration_split_from_manifest(_selection("train", 20))
    document["games"][0]["game_id"] = "tampered"
    with pytest.raises(ValueError):
        calibration_split_from_manifest(document)


def test_artifact_metrics_are_deeply_read_only_and_unbound_fixture_cannot_promote_to_production(
    binding, event_support, calibration_fixture
):
    artifact, _, _ = calibration_fixture
    with pytest.raises(TypeError):
        artifact.fit_metrics["action"]["count"] = 99
    with pytest.raises(ValueError):
        replace(artifact, diagnostic_fixture=False).validate(binding, event_support)


def test_tau_grid_and_selection_use_only_validation_fit_and_keep_infinity_baseline(binding, calibration_fixture):
    artifact, _, _ = calibration_fixture
    split = artifact.split
    grid = (math.inf, 1.0, 0.5, 0.25)
    game = split.fit_games[0]
    outcomes = {tau: {game: 0.1 if tau == 0.5 else 0.0} for tau in grid}
    report = choose_validation_tau(outcomes, split=split, binding=binding, grid=grid, diagnostic_fixture=True)
    assert report["selected_tau"] == "0.5" and report["evidence_status"] == "empirical_only_no_error_certificate"
    with pytest.raises(ValueError):
        choose_validation_tau(
            {tau: {split.check_games[0]: 0.0} for tau in grid}, split=split, binding=binding, diagnostic_fixture=True
        )
    assert fixed_tau_grid(grid) == grid
    for invalid in ((1.0, 0.5), (0.5, 0.5, math.inf), (0.0, math.inf), (math.nan, math.inf)):
        with pytest.raises(ValueError):
            fixed_tau_grid(invalid)
    with pytest.raises(ValueError):
        fixed_tau_grid(grid, max_prior_amplification=2.0)
