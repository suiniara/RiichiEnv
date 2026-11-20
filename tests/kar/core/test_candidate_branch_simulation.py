from riichienv import Phase

from riichienv.kar.candidate_branches import simulate_candidate_branches
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.decision import ReplayOpportunity
from riichienv.kar.observation import ObserverLedger


def _own_action_opportunity(seed=73):
    env = MJSOUL_RULESET.create_env(seed=seed)
    env.reset()
    assert env.phase == Phase.WaitAct
    seat = int(env.current_player)
    ledger = ObserverLedger(seat, (0, 1, 2, 3))
    for event in env.rule_log:
        ledger.apply_event(event)
    observation = env.get_observations([seat])[seat]
    opportunity = ReplayOpportunity.observe(ledger, observation)
    return env, opportunity


def test_candidate_probe_branches_every_unambiguous_legal_discard_without_mutating_source():
    env, opportunity = _own_action_opportunity()
    before_log = tuple(env.rule_log)
    before_hands = tuple(tuple(hand) for hand in env.hands)
    before_scores = tuple(env.scores())
    expected = {
        item.global_id
        for item in opportunity.mapped
        if 0 <= item.global_id < 148 and not item.native_execution_ambiguous
    }
    rows = simulate_candidate_branches(
        env,
        opportunity,
        source_game_id="synthetic-source-game",
        cutoff_event_id=0,
        master_seed=17,
        replicates=1,
        max_decisions=1,
    )
    assert {row["candidate_action_id"] for row in rows} == expected
    assert all(row["source_state_identity"] for row in rows)
    assert all(row["continuation_policy"] == "uniform_native_legal_v1" for row in rows)
    assert all(row["status"] in {"INCOMPLETE", "PASS"} for row in rows)
    assert tuple(env.rule_log) == before_log
    assert tuple(tuple(hand) for hand in env.hands) == before_hands
    assert tuple(env.scores()) == before_scores


def test_candidate_probe_seed_and_source_identity_are_reproducible():
    env, opportunity = _own_action_opportunity()
    arguments = {
        "source_game_id": "synthetic-source-game",
        "cutoff_event_id": 0,
        "master_seed": 991,
        "replicates": 1,
        "max_decisions": 1,
    }
    first = simulate_candidate_branches(env, opportunity, **arguments)
    second = simulate_candidate_branches(env, opportunity, **arguments)
    assert [
        (row["candidate_action_id"], row["seed"], row["source_state_identity"], row["status"])
        for row in first
    ] == [
        (row["candidate_action_id"], row["seed"], row["source_state_identity"], row["status"])
        for row in second
    ]
