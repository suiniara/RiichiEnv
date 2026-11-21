import sqlite3
from types import SimpleNamespace

from riichienv import Phase

from riichienv.kar.candidate_branches import simulate_candidate_branches
from riichienv.kar.contracts import MJSOUL_RULESET
from riichienv.kar.decision import ReplayOpportunity
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.learning.export import GameWriter
from riichienv.kar.tokenize import append_action, encode_prefix


def test_baseline_discard_score_preserves_connected_tiles_and_dora():
    from riichienv.kar.candidate_branches import _discard_efficiency_score

    hand = [0, 1, 2, 4, 5, 6, 8, 9, 10, 108, 112, 116, 120, 124]
    isolated_honor = _discard_efficiency_score(hand, 108, set())
    connected_tile = _discard_efficiency_score(hand, 4, set())
    dora_tile = _discard_efficiency_score(hand, 4, {1})
    assert isolated_honor > connected_tile
    assert dora_tile < connected_tile


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
    expected = {item.global_id for item in opportunity.mapped if 0 <= item.global_id < 148}
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
    canonical_prefix = encode_prefix(opportunity.snapshot, opportunity.candidates)
    for row in rows:
        candidate_id = row["candidate_action_id"]
        canonical = append_action(canonical_prefix, candidate_id).to_bytes()[108:]
        assert bytes.fromhex(row["tokens_hex"]) == canonical
        assert row["action_index"] == row["decide_index"] + 1
    assert all(row["source_state_identity"] for row in rows)
    assert len({row["wall_seed"] for row in rows}) == 1
    assert all(row["wall_conditioning"] == "native_unknown_tiles_sampled_once_per_source_replicate_shared_across_candidates" for row in rows)
    assert all(row["continuation_policy"] == "native_riichi_baseline_v2" for row in rows)
    assert all(row["status"] in {"BLOCKED", "INCOMPLETE", "PASS"} for row in rows)
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



def test_candidate_branch_selection_covers_all_game_timeline_strata():
    game = SimpleNamespace(game_id="stratified-game", events=[{"type": "noop"} for _ in range(90)])
    writer = GameWriter(sqlite3.connect(":memory:"), game, branch_probe_decisions=6, branch_seed=23)
    class Cloneable:
        def clone(self):
            return Cloneable()
    opportunity = SimpleNamespace(snapshot=SimpleNamespace(decision_phase="own_act", observer_player_id=0), candidates=(0, 1))
    for index in range(30):
        writer.on_opportunity(Cloneable(), opportunity, game.game_id, index * 3)
        writer.ordinals[-1, 0] += 1
    selected = writer._selected_branch_opportunities()
    strata = [(3 * item[3]) // len(game.events) for item in selected]
    assert len(selected) == 6
    assert all(strata.count(index) >= 2 for index in range(3))
    assert writer.branch_opportunities_seen == 30
