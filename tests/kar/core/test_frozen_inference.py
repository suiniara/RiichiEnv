# ruff: noqa: E402
import hashlib
import random
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
from riichienv import Phase
from riichienv.kar.actions import execute_candidate, execute_joint_responses
from riichienv.kar.agent import KARAgent
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.learning.batch import collate_inputs
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.inference import FrozenModelPredictor, model_weights_sha256
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.live_events import LiveEventProjection
from riichienv.kar.runtime import RuntimePolicy
from riichienv.kar.tokenize import append_action, encode_prefix


@pytest.fixture
def frozen_predictor():
    torch.manual_seed(27)
    model = KARTransformer(
        ModelConfig(layers=2, width=32, heads=4, feedforward=64, preset="functional-miniature")
    ).eval()
    return FrozenModelPredictor(model, hashlib.sha256(b"functional-manifest-fixture").hexdigest())


def test_frozen_raw_policy_and_all_candidate_scores_match_dense_without_updates(
    frozen_predictor, public_view, record_property
):
    snapshot, ids = public_view
    predictor = frozen_predictor
    before = model_weights_sha256(predictor.model)
    policy = predictor.policy(snapshot, ids, "context")
    selected = tuple(action for action in ids if action < 148)
    scores = predictor.candidates(snapshot, ids, selected, "context")
    prefix = encode_prefix(snapshot, ids)
    rows = []
    for action in selected:
        conditional = append_action(prefix, action)
        rows.append(
            {
                "tokens": conditional.to_bytes()[108:],
                "candidates": ids,
                "decide_index": prefix.decide_index,
                "action_index": prefix.decide_index + 1,
                "decision_row_id": "context",
            }
        )
    with torch.no_grad():
        dense = predictor.model(collate_inputs(rows))
    for index, action in enumerate(selected):
        torch.testing.assert_close(torch.tensor(policy), dense.policy_logits[index], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(torch.tensor(scores[action][0]), dense.response_logits[index], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(torch.tensor(scores[action][1]), dense.rank_logits[index], atol=2e-6, rtol=2e-5)
    after = model_weights_sha256(predictor.model)
    assert before == after
    record_property("model_weights_sha256_before", before)
    record_property("model_weights_sha256_after", after)
    record_property("backward_or_update_executed", False)
    assert all(parameter.grad is None for parameter in predictor.model.parameters())


def test_frozen_fp32_ignores_ambient_autocast_for_prefix_and_suffix(frozen_predictor, public_view, record_property):
    view, ids = public_view
    predictor = frozen_predictor
    selected = tuple(action for action in ids if action < 148)
    expected_policy = predictor.policy(view, ids, "fp32-reference")
    expected = predictor.candidates(view, ids, selected, "fp32-reference")
    recipe = predictor.binding.inference_recipe_sha256
    weight_sha = model_weights_sha256(predictor.model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        suffix_in_bf16 = predictor.candidates(view, ids, selected, "fp32-reference")
        actual_policy = predictor.policy(view, ids, "ambient-bf16-prefix")
        same_outer = predictor.candidates(view, ids, selected, "ambient-bf16-prefix")
        assert torch.is_autocast_enabled("cpu")
    suffix_in_fp32 = predictor.candidates(view, ids, selected, "ambient-bf16-prefix")
    record_property("max_policy_delta", max(abs(left - right) for left, right in zip(actual_policy, expected_policy)))
    for label, scores in (("bf16_outer_suffix", suffix_in_bf16), ("bf16_outer_both", same_outer),
                          ("fp32_outer_suffix", suffix_in_fp32)):
        for head, name in enumerate(("response", "rank")):
            delta = max(float((torch.tensor(scores[action][head]) - torch.tensor(expected[action][head])).abs().max())
                        for action in selected)
            record_property(f"{label}/{name}", delta)
    assert actual_policy == expected_policy and suffix_in_bf16 == same_outer == suffix_in_fp32 == expected
    assert predictor.binding.inference_recipe_sha256 == recipe
    assert model_weights_sha256(predictor.model) == weight_sha


@pytest.mark.parametrize("prefix_bf16", [True, False])
def test_raw_kv_cache_rejects_cross_precision_suffix_even_with_same_model_dtype(
    frozen_predictor, public_view, prefix_bf16,
):
    snapshot, ids = public_view
    prefix = encode_prefix(snapshot, ids)
    action = next(action for action in ids if action < 148)
    row = {"candidates": ids, "decision_row_id": "raw-autocast-cache"}
    before = collate_inputs([{**row, "tokens": prefix.to_bytes()[108:], "decide_index": prefix.decide_index}])
    suffix = collate_inputs([{**row, "tokens": append_action(prefix, action).to_bytes()[-139:],
                             "decide_index": 0, "action_index": 0}])
    model = frozen_predictor.model
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=prefix_bf16):
        cache = model.encode_prefix(before)
        response, rank = model.score_actions(cache, suffix)
    assert torch.isfinite(response).all() and torch.isfinite(rank).all()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=not prefix_bf16), pytest.raises(
        ValueError, match="cache autocast precision differs",
    ):
        model.score_actions(cache, suffix)


def test_all_sixty_two_state_slots_are_permutation_equivariant(frozen_predictor, public_view):
    snapshot, ids = public_view
    prefix = encode_prefix(snapshot, ids)
    batch = collate_inputs(
        [
            {
                "tokens": prefix.to_bytes()[108:],
                "candidates": ids,
                "decide_index": prefix.decide_index,
                "decision_row_id": "permutation",
            }
        ]
    )
    indices = list(range(batch.fields["kind"].shape[1]))
    state = list(range(prefix.history_rows + 1, prefix.history_rows + 63))
    shuffled = state.copy()
    random.Random(51).shuffle(shuffled)
    assert len(state) == 62 and state != shuffled
    for dest, source in zip(state, shuffled):
        indices[dest] = source
    permuted = replace(batch, fields={key: value[:, indices].clone() for key, value in batch.fields.items()})
    with torch.no_grad():
        normal = frozen_predictor.model(batch).policy_logits
        swapped = frozen_predictor.model(permuted).policy_logits
    torch.testing.assert_close(normal, swapped, atol=2e-6, rtol=2e-5)


def test_foreign_observer_context_and_training_mode_reject_cache(frozen_predictor, public_view):
    snapshot, ids = public_view
    discard = tuple(action for action in ids if action < 148)
    frozen_predictor.policy(snapshot, ids, "same")
    with pytest.raises(ValueError):
        frozen_predictor.candidates(snapshot, ids, discard, "other")
    with pytest.raises(ValueError):
        frozen_predictor.candidates(replace(snapshot, observer_player_id=1), ids, discard, "same")
    before = model_weights_sha256(frozen_predictor.model)
    frozen_predictor.model.train()
    with pytest.raises(ValueError):
        frozen_predictor.policy(snapshot, ids, "same")
    assert before == model_weights_sha256(frozen_predictor.model)


def test_feature_recipe_cannot_reuse_calibration_or_cached_prefix_from_another_mode(frozen_predictor, public_view):
    other = FrozenModelPredictor(
        frozen_predictor.model, frozen_predictor.binding.train_manifest_sha256, use_rule_features=False
    )
    assert frozen_predictor.binding.model_weights_sha256 == other.binding.model_weights_sha256
    assert frozen_predictor.binding != other.binding
    view, ids = public_view
    frozen_predictor.policy(view, ids, "recipe-bound")
    frozen_predictor.use_rule_features = False
    with pytest.raises(ValueError):
        frozen_predictor.policy(view, ids, "recipe-bound")


def test_frozen_policy_executes_six_real_agent_windows_without_changing_weights(frozen_predictor, record_property):
    env = MJSOUL_RULESET.create_env(seed=17)
    observations = env.reset()
    projection = LiveEventProjection()
    projection.observe_native(list(env.rule_log))
    predictors = {
        seat: FrozenModelPredictor(frozen_predictor.model, frozen_predictor.binding.train_manifest_sha256)
        for seat in range(4)
    }
    agents = {seat: KARAgent(seat, RuntimePolicy(predictor, seed=17 + seat)) for seat, predictor in predictors.items()}
    before_weights = model_weights_sha256(frozen_predictor.model)
    for _ in range(6):
        choices = {
            seat: agents[seat].choose(observation, projection.snapshot(seat), projection.round_instance)[0]
            for seat, observation in observations.items()
            if observation.legal_actions()
        }
        before = len(env.rule_log)
        if env.phase == Phase.WaitResponse:
            observations = dict(execute_joint_responses(env, choices))
        else:
            observations = dict(execute_candidate(env, env.current_player, choices[env.current_player]))
        projection.observe_native(list(env.rule_log[before:]))
    assert sum(agent.decisions for agent in agents.values()) >= 6
    assert model_weights_sha256(frozen_predictor.model) == before_weights
    assert all(predictor.candidate_calls == 0 for predictor in predictors.values())
    record_property("real_agent_windows", 6)
    record_property("unchanged_model_sha256", before_weights)


def test_frozen_model_is_connected_to_full_prior_sampler_without_auxiliary_work(frozen_predictor, public_view):
    snapshot, ids = public_view
    before = model_weights_sha256(frozen_predictor.model)
    result = RuntimePolicy(frozen_predictor, seed=7).decide(snapshot, ids, "runtime")
    assert result.action_id in ids and result.evidence_status == "unfitted_prior_only"
    assert frozen_predictor.candidate_calls == 0 and before == model_weights_sha256(frozen_predictor.model)
