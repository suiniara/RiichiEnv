"""Same actual random Small model on source-aligned inputs; forward only."""
import json

import pytest
import torch

from riichienv.kar.actions import legal_candidates
from riichienv.kar.learning.batch import collate_inputs
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.inference import model_weights_sha256
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.tokenize import append_action, encode_prefix
from tests.kar.test_input_semantics import opening_pair
from tests.kar.test_response_mask_boundaries import four_riichi


@pytest.mark.parametrize("scenario", ["opening", "fourth-riichi"])
@torch.no_grad()
def test_source_aligned_inputs_have_identical_small_fp32_readouts(scenario, record_property):
    if scenario == "opening":
        env, raw, offline = opening_pair()
        live = LiveEventProjection()
        live.observe_native(raw)
        actor = 0
        ids = tuple(item.global_id for item in legal_candidates(env.get_observation(actor), drawn_tile=env.drawn_tile))
        action = ids[0]
    else:
        env, live, chosen = four_riichi()
        actor = 3
        ids = tuple(item.global_id for item in legal_candidates(env.get_observation(actor), drawn_tile=env.drawn_tile))
        action = chosen.global_id
        # This comparison changes source envelopes, not the midgame decoder.
        offline = [{**event, "source_record_index": 0, "source_record_sha256": "f" * 64} for event in live.events]
    record_property("comparison_source", "RecordNewRound normalization vs native" if scenario == "opening"
                    else "native midgame vs same live events with different source metadata")
    owner = ObserverLedger(actor, (0, 1, 2, 3))
    for event in offline:
        owner.apply_event(event)
    samples = []
    for view in (owner.snapshot(), live.snapshot(actor)):
        prefix = encode_prefix(view, ids)
        sequence = append_action(prefix, action)
        samples.append({"tokens": sequence.to_bytes()[108:], "candidates": ids,
                        "decide_index": prefix.decide_index, "action_index": len(sequence.rows) - 1,
                        "decision_row_id": "source-parity"})
    left, right = (collate_inputs([sample]) for sample in samples)
    assert left.context_ids == right.context_ids
    assert torch.equal(left.candidates, right.candidates)
    assert all(torch.equal(left.fields[key], right.fields[key]) for key in left.fields)
    torch.manual_seed(20261118)
    model = KARTransformer(ModelConfig.from_preset("small")).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 5239435
    before = model_weights_sha256(model)
    a, b = model(left), model(right)
    differences = []
    for key in ("policy_logits", "response_logits", "rank_logits"):
        first, second = getattr(a, key), getattr(b, key)
        assert first.dtype == second.dtype == torch.float32
        assert not first.requires_grad and not second.requires_grad
        differences.append(float((first - second).abs().max()))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert model_weights_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    record_property("model_weights_sha256_before_and_after", before)
    record_property("model_parameter_count", 5239435)
    record_property("head_max_abs_differences", json.dumps(differences))
