"""New semantic/quality identities reject legacy bytes rather than restamping artifacts."""
import copy
from dataclasses import replace

import pytest

from riichienv.kar.calibration import CalibrationArtifact, CalibrationRow, calibrate_static_rows, calibration_split
from riichienv.kar.contracts import MJSOUL_RULESET, content_hash
from riichienv.kar.event_semantics import PUBLIC_CLOCK_VERSION
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.runtime_contracts import RuntimeBinding
from riichienv.kar.support import EventSupport
from riichienv.kar.tokenize import TokenHistoryCache, TokenizationUnavailableError, encode_prefix
from riichienv.kar.versioning import artifact_contract, validate_artifact_contract

LEGACY_FIELDS = {
    "artifact_contract_version": "kar_riichienv_contract_v3",
    "input_pipeline_version": "legal_set_pool_private_self_pass_v1",
    "response_mask_version": "public_rule_response_mask_v2_mjsoul_nagashi",
    "public_clock_version": "source_record_clock_v0",
    "terminal_public_version": "terminal_without_abort_reveal_v0",
    "branch_quality_version": "overall_only_v0",
}


@pytest.mark.parametrize("field,legacy", LEGACY_FIELDS.items())
@pytest.mark.parametrize("missing", [True, False])
def test_semantic_contract_rejects_missing_and_explicit_legacy_identity(field, legacy, missing):
    current = artifact_contract()
    previous = dict(current)
    if missing:
        previous.pop(field)
    else:
        previous[field] = legacy
    with pytest.raises(ValueError, match="version contract differs"):
        validate_artifact_contract(previous)
    assert current["schema_id"] == "kar-riichienv-v1"
    assert current["schema_hash"] == "fcc02a0bef0a7e95ca219c59ae199c4d1f5129372fac91323d3791bc30c31673"
    assert current["model_tier_basis"] == "N_train_half_games"


@pytest.mark.parametrize("clock_version", [None, "source_record_clock_v0"])
def test_legacy_snapshot_cannot_become_current_by_reusing_schema_or_cache(clock_version):
    owner = ObserverLedger(0, (0, 1, 2, 3))
    for event in MJSOUL_RULESET.create_env(seed=42).mjai_log:
        owner.apply_event(event)
    current = owner.snapshot()
    assert current.public_clock_version == PUBLIC_CLOCK_VERSION
    cache = TokenHistoryCache()
    expected = encode_prefix(current, (4, 8), history_cache=cache).to_bytes()
    old = replace(current, public_clock_version=clock_version)
    with pytest.raises(TokenizationUnavailableError, match="semantic clock"):
        encode_prefix(old, (4, 8), history_cache=cache)
    assert encode_prefix(current, (4, 8), history_cache=cache).to_bytes() == expected


def fixtures():
    binding = RuntimeBinding.current("a" * 64, "b" * 64)
    support = EventSupport(binding.train_manifest_sha256, (500,) * 21, "synthetic version rejection only")
    split = calibration_split([f"version-validation-{index}" for index in range(20)])
    def row(game):
        return CalibrationRow(
            game, game + "/decision", (4, 8), (0.0,) * 546, 4,
            response_logits=(0.0,) * 21, response_allowed=(True,) * 21,
            rank_logits=((0.0,) * 4,) * 21,
            response_target=0, response_valid=True, rank_target=0, value_valid=True,
        )

    artifact = calibrate_static_rows(
        [row(split.fit_games[0])], [row(split.check_games[0])],
        split=split, binding=binding, support=support, temperatures=(1.0,),
        source_reference="synthetic,not actual calibration", diagnostic_fixture=True,
    )
    return binding, support, artifact


@pytest.mark.parametrize("field,legacy", LEGACY_FIELDS.items())
def test_checksum_valid_old_support_and_calibration_contracts_are_rejected(field, legacy):
    binding, support, artifact = fixtures()
    assert EventSupport.from_record(support.to_record(), binding) == support
    loaded = CalibrationArtifact.from_record(artifact.to_record(), binding, support, allow_diagnostic=True)
    assert loaded.to_record() == artifact.to_record()
    for kind, source in (("support", support.to_record()), ("calibration", artifact.to_record())):
        forged = copy.deepcopy(source)
        forged["contract"][field] = legacy
        hash_key = kind + "_sha256"
        forged[hash_key] = content_hash({key: value for key, value in forged.items() if key != hash_key})
        with pytest.raises(ValueError, match="contract mismatch"):
            if kind == "support":
                EventSupport.from_record(forged, binding)
            else:
                CalibrationArtifact.from_record(forged, binding, support, allow_diagnostic=True)


@pytest.mark.parametrize("kind,old_version", [("support", "train_independent_games_v1"),
                                              ("calibration", "heldout_temperatures_log_gate_v3")])
def test_old_outer_artifact_version_is_rejected_even_with_a_valid_checksum(kind, old_version):
    binding, support, artifact = fixtures()
    source = support.to_record() if kind == "support" else artifact.to_record()
    source["version"] = old_version
    key = kind + "_sha256"
    source[key] = content_hash({name: value for name, value in source.items() if name != key})
    with pytest.raises(ValueError, match="version/contract mismatch"):
        if kind == "support":
            EventSupport.from_record(source, binding)
        else:
            CalibrationArtifact.from_record(source, binding, support, allow_diagnostic=True)
