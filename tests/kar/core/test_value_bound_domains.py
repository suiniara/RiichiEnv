"""Exact mathematical counterexamples, not empirical error certificates."""
from fractions import Fraction


def test_nonempty_support_error_bound_does_not_bound_all_branch_nominal_value(record_property):
    public_allowed = (0, 1)
    supported = {0}
    predicted_probability = true_probability = (Fraction(1, 2),) * 2
    true_value = (Fraction(0), Fraction(-3, 2))
    predicted_value = (Fraction(0), Fraction(3, 2))
    epsilon_all = max(abs(predicted_value[event] - true_value[event]) for event in public_allowed)
    epsilon_sup = max(abs(predicted_value[event] - true_value[event]) for event in supported)
    delta_e = sum(abs(p - q) for p, q in zip(predicted_probability, true_probability)) / 2
    q_true = sum(true_probability[event] * true_value[event] for event in public_allowed)
    q_hat = sum(predicted_probability[event] * predicted_value[event] for event in public_allowed)
    q_tilde = sum(predicted_probability[event] * (
        predicted_value[event] if event in supported else Fraction(-3, 2)
    ) for event in public_allowed)
    rho_u = sum(predicted_probability[event] for event in public_allowed if event not in supported)
    b_sup = epsilon_sup + 3 * delta_e

    assert q_true == q_tilde == Fraction(-3, 4) and q_hat == Fraction(3, 4)
    assert epsilon_sup == delta_e == 0 and epsilon_all == 3
    assert abs(q_hat - q_true) == Fraction(3, 2) > epsilon_sup + 3 * delta_e
    assert abs(q_hat - q_true) <= epsilon_all + 3 * delta_e
    assert rho_u == Fraction(1, 2) and -b_sup - 3 * rho_u <= q_tilde - q_true <= b_sup
    for key, value in {"q_true": q_true, "q_hat": q_hat, "q_tilde": q_tilde, "delta_e": delta_e,
                       "epsilon_all": epsilon_all, "epsilon_sup": epsilon_sup, "rho_u": rho_u,
                       "conservative_lower": -b_sup - 3 * rho_u, "conservative_upper": b_sup}.items():
        record_property(key, str(value))
