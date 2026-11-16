import math
import random

import pytest

from riichienv.kar.policy import one_sided_bounds, restricted_kl, sample_complete, tau_minimum
from riichienv.kar.probability import legal_prior, masked_softmax
from riichienv.kar.support import EventSupport, candidate_value


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan, True])
def test_bad_probability_temperatures_are_rejected(bad):
    with pytest.raises(ValueError):
        masked_softmax([0.0, 1.0], [True, True], bad)


@pytest.mark.parametrize(
    "logits,mask",
    [([0.0, math.nan], [True, True]), ([0.0, 1.0], [False, False]), ([0.0], [True, True]), ([0.0, 1.0], [True, 1])],
)
def test_empty_bad_masks_and_nonfinite_logits_rejected(logits, mask):
    with pytest.raises(ValueError):
        masked_softmax(logits, mask)


def test_masked_probabilities_and_stable_global_id_identity():
    logits = [-1000.0] * 546
    logits[4] = 1000.0
    logits[219] = 999.0
    prior, logs = legal_prior(logits, [219, 4])
    assert prior[4] + prior[219] == pytest.approx(1.0)
    assert all(prior[index] == 0 for index in range(546) if index not in (4, 219))
    assert logs[5] == -math.inf
    assert prior == legal_prior(logits, [4, 219])[0]


def test_unsupported_penalty_happens_after_public_mask(binding):
    support = EventSupport(binding.train_manifest_sha256, (500,) * 20 + (0,), "static fixture")
    response = [0.0] * 21
    response[20] = 20
    ranks = ((20.0, 0.0, 0.0, 0.0),) * 21
    masked = candidate_value(4, response, ranks, (True,) * 20 + (False,), support)
    assert masked.unsupported_mass == 0 and masked.conservative_value == pytest.approx(masked.marginal_value)
    allowed = candidate_value(4, response, ranks, (True,) * 21, support)
    assert allowed.unsupported_mass > 0.99 and allowed.conservative_value < -1.49
    none = EventSupport(binding.train_manifest_sha256, (499,) * 21, "static fixture")
    assert candidate_value(4, response, ranks, (True,) * 21, none).conservative_value == pytest.approx(-1.5)


def test_event_support_round_trip_and_foreign_data_rejection(binding, event_support):
    doc = event_support.to_record()
    assert EventSupport.from_record(doc, binding) == event_support
    doc["train_game_counts"][0] = 499
    with pytest.raises(ValueError):
        EventSupport.from_record(doc, binding)


def test_non_discard_cannot_acquire_fake_response_value(event_support):
    with pytest.raises(ValueError):
        candidate_value(221, [0.0] * 21, ((0.0,) * 4,) * 21, [True] * 21, event_support)


@pytest.mark.parametrize("tau", [math.inf, 0.1, 0.5, 2.0])
def test_restricted_update_preserves_all_outside_probabilities_and_mass(tau):
    prior = [0.0] * 546
    for index, p in ((4, 0.1), (8, 0.2), (12, 0.15), (219, 0.25), (221, 0.3)):
        prior[index] = p
    update = restricted_kl(prior, [4, 8], [1.0, -1.0], tau)
    assert sum(update.probabilities) == pytest.approx(1.0)
    assert sum(update.probabilities[index] for index in (4, 8)) == pytest.approx(0.3)
    assert all(update.probabilities[index] == prior[index] for index in range(546) if index not in (4, 8))
    if math.isinf(tau):
        assert update.probabilities == tuple(prior)


def test_candidate_order_changes_neither_action_identity_nor_distribution():
    prior = [0.0] * 546
    prior[4] = 0.4
    prior[8] = 0.6
    assert (
        restricted_kl(prior, [4, 8], [1.0, -1.0], 0.5).probabilities
        == restricted_kl(prior, [8, 4], [-1.0, 1.0], 0.5).probabilities
    )


def test_zero_prior_stays_zero_and_empty_singleton_support_falls_back():
    prior = [0.0] * 546
    prior[4] = 1.0
    assert restricted_kl(prior, [4, 8], [-1.0, 1.0], 0.1).probabilities[8] == 0
    for ids, values in (([], []), ([4], [1.0]), ([8, 12], [0.0, 0.0])):
        assert restricted_kl(prior, ids, values, 0.1).probabilities == tuple(prior)


def test_rmax_tau_bound_is_enforced():
    prior = [0.0] * 546
    prior[4] = prior[8] = 0.5
    minimum = tau_minimum(3.0)
    with pytest.raises(ValueError):
        restricted_kl(prior, [4, 8], [1.5, -1.5], minimum / 2, max_prior_amplification=3.0)
    result = restricted_kl(prior, [4, 8], [1.5, -1.5], minimum, max_prior_amplification=3.0)
    assert max(new / old for new, old in zip(result.qtau, result.q0)) <= 3.0 + 1e-12


def test_original_wrong_qtau_rho_counterexample_is_rejected_by_e1_e2():
    prior = [0.0] * 546
    prior[4] = prior[8] = 0.5
    update = restricted_kl(prior, [4, 8], [-1.5, 0.0], 0.1)
    e1, e2 = one_sided_bounds(update, [1.0, 0.0], [0.0, 0.0], 0.0, 0.1)
    real = sum((new - old) * value for new, old, value in zip(update.qtau, update.q0, [1.5, 0.0]))
    wrong = 0.1 * update.kl - 3 * update.qtau[0]
    assert wrong > 0 and real < 0 and e1 <= real + 1e-12 and e2 == pytest.approx(real)


def test_random_finite_sets_obey_kl_optimality_and_one_sided_bounds():
    rng = random.Random(103)
    for _ in range(150):
        count = rng.randrange(2, 16)
        ids = list(range(0, count * 4, 4))
        weights = [rng.random() + 0.01 for _ in ids]
        total = sum(weights)
        prior = [0.0] * 546
        for index, w in zip(ids, weights):
            prior[index] = w / total
        rho = [rng.random() for _ in ids]
        truth = [rng.uniform(-1.5, 1.5) for _ in ids]
        # Construct a valid one-sided pessimistic-error instance with b=0.
        scores = [q - (q + 1.5) * r for q, r in zip(truth, rho)]
        tau = rng.uniform(0.1, 3.0)
        result = restricted_kl(prior, ids, scores, tau)
        e1, e2 = one_sided_bounds(result, rho, [0.0] * count, 0.0, tau)
        real = sum((new - old) * q for new, old, q in zip(result.qtau, result.q0, truth))
        assert result.model_gain + 1e-12 >= tau * result.kl
        assert real + 1e-11 >= e2 >= e1 - 1e-11


def test_sampling_is_reproducible_and_argmax_is_only_explicit_diagnostic():
    prior = [0.0] * 546
    prior[4] = 0.25
    prior[219] = 0.75
    first, second = random.Random(4), random.Random(4)
    assert [sample_complete(prior, [4, 219], first) for _ in range(100)] == [
        sample_complete(prior, [219, 4], second) for _ in range(100)
    ]
    assert sample_complete(prior, [4, 219], first, diagnostic_argmax=True) == 219
