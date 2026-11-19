"""Frozen sampler digest/ordering compatibility, with no parameter updates."""
import random
from types import SimpleNamespace

import pytest

from riichienv.kar.contracts import content_hash
from riichienv.kar.learning import sampling
from riichienv.kar.learning.sampling import GlobalEpochSampler


def _old_order(lengths, seed, epoch, window):
    indices = list(range(len(lengths)))
    random.Random(seed + 1000003 * epoch).shuffle(indices)
    return tuple(index for start in range(0, len(indices), window)
                 for index in sorted(indices[start:start + window], key=lengths.__getitem__))


def test_epoch_and_restore_hash_each_immutable_sequence_once(monkeypatch, record_property):
    lengths = [300 + i % 11 for i in range(103)]
    cfg = {"seed": 17, "effective_batch": 16, "microbatch": 3, "rank": 2, "world_size": 4, "bucket_window": 25}
    calls = []

    def observed(sequence):
        calls.append(len(sequence))
        return content_hash(sequence)

    monkeypatch.setattr(sampling, "content_hash", observed)
    first = GlobalEpochSampler(lengths, **cfg)
    expected_lengths = content_hash(lengths)
    assert calls == [103, 103]
    for epoch in range(2):
        expected = _old_order(lengths, cfg["seed"], epoch, cfg["bucket_window"])
        assert first.order == expected
        before = len(calls)
        for _ in range(6):
            state = first.state_dict()
            assert state["lengths_sha256"] == expected_lengths
            assert state["order_sha256"] == content_hash(expected)
        assert len(calls) == before
        while first.cursor < len(lengths) - 7:
            first.commit(first.peek())
        tail = first.peek()
        assert tail.global_indices == expected[-7:] and tail.local_indices == expected[-7:][2::4]
        if epoch == 0:
            first.commit(tail)
            first.next_epoch()
            assert len(calls) == before + 1
    state = first.state_dict()
    resumed = GlobalEpochSampler(lengths, **cfg)
    calls.clear()
    resumed.load_state_dict(state)
    assert calls == [103] and resumed.state_dict() == state and resumed.peek() == tail
    assert calls == [103]  # state_dict() and peek() didn't rehash after load
    resumed.commit(resumed.peek())
    assert resumed.exhausted
    record_property("checkpoint_hash_calls_per_state_dict", 0)
    record_property("order_entries", len(resumed.order))


@pytest.mark.parametrize("invalid", [
    {"lengths_sha256": "0" * 64}, {"order_sha256": "0" * 64}, {"cursor": 3}, {"cursor": -1},
    {"cursor": 104}, {"cursor": True}, {"epoch": -1},
])
def test_bad_digest_or_cursor_never_replaces_committed_sampler_state(invalid):
    cfg = {"seed": 9, "effective_batch": 16, "microbatch": 2}
    sampler = GlobalEpochSampler([300] * 103, **cfg)
    sampler.commit(sampler.peek())
    previous_group, previous_state = sampler.peek(), sampler.state_dict()
    with pytest.raises(ValueError, match="resume|Resume cursor|Corrupt sampler"):
        sampler.load_state_dict({**previous_state, **invalid})
    assert sampler.state_dict() == previous_state and sampler.peek() == previous_group


def test_cached_digest_cannot_be_staled_by_mutating_constructor_list_or_reassigning_order():
    lengths = [300 + index for index in range(5)]
    sampler = GlobalEpochSampler(lengths, seed=2, effective_batch=2, microbatch=1)
    before = sampler.state_dict()
    lengths[0] += 10
    assert sampler.lengths == (300, 301, 302, 303, 304) and sampler.state_dict() == before
    with pytest.raises(AttributeError):
        sampler.lengths = tuple(lengths)
    with pytest.raises(AttributeError):
        sampler.order = tuple(reversed(sampler.order))
    with pytest.raises(ValueError, match="configuration/data differs"):
        GlobalEpochSampler(lengths, seed=2, effective_batch=2, microbatch=1).load_state_dict(before)


def test_actual_checkpoint_restores_same_next_group_and_still_checks_source_identity(tmp_path):
    torch = pytest.importorskip("torch")
    from riichienv.kar.learning.config import ModelConfig  # noqa: PLC0415
    from riichienv.kar.learning.inference import model_weights_sha256  # noqa: PLC0415
    from riichienv.kar.learning.model import KARTransformer  # noqa: PLC0415
    from riichienv.kar.learning.production_checkpoint import (  # noqa: PLC0415
        rank_rng_state,
        read_production_checkpoint,
        save_production_checkpoint,
    )
    from riichienv.kar.release import implementation_release_identity  # noqa: PLC0415
    from riichienv.kar.versioning import artifact_contract  # noqa: PLC0415

    lengths = [300 + i % 10 for i in range(23)]
    config = {"seed": 12, "effective_batch": 8, "microbatch": 2}
    original = GlobalEpochSampler(lengths, **config)
    while not original.exhausted:
        original.commit(original.peek())
    original.next_epoch()
    original.commit(original.peek())  # sampler bookkeeping only, no optimizer step
    model = KARTransformer(ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="checkpoint-sampler"))
    before = model_weights_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = SimpleNamespace(state_dict=lambda: {"synthetic-no-updates": True})
    implementation = implementation_release_identity()
    contract = {**artifact_contract(), "implementation": implementation}
    runtime = {"world_size": 1, "device": "cpu"}
    path = tmp_path / "checkpoint.pt"
    save_production_checkpoint(path, model, optimizer, scheduler, contract=contract, runtime=runtime,
                               trainer_state={"sampler": original.state_dict()},
                               rank_rng=[rank_rng_state(torch.device("cpu"))], step=0)
    saved = read_production_checkpoint(path, contract=contract, runtime=runtime)
    resumed = GlobalEpochSampler(lengths, **config)
    resumed.load_state_dict(saved["trainer"]["sampler"])
    assert resumed.state_dict() == original.state_dict() and resumed.peek() == original.peek()
    assert resumed.order == _old_order(lengths, config["seed"], 1, 128)
    assert not optimizer.state and model_weights_sha256(model) == before
    different = {**contract, "implementation": {**implementation, "source_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="exact resume refused"):
        read_production_checkpoint(path, contract=different, runtime=runtime)
