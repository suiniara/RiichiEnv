# ruff: noqa: E402 -- optional Torch must be checked before ML imports
import os
import socket
from multiprocessing import get_context
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.nn.parallel import DistributedDataParallel

from riichienv.kar.actions import legal_candidates
from riichienv.kar.authorization import UpdateAuthorization, update_implementation_sha256
from riichienv.kar.budget import BudgetPlan, CostForecast
from riichienv.kar.contracts import MJSOUL_RULESET, content_hash
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.distributed import DistributedContext
from riichienv.kar.learning.inference import model_weights_sha256
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.learning.production_checkpoint import (
    rank_rng_state,
    read_production_checkpoint,
    save_production_checkpoint,
)
from riichienv.kar.learning.sampling import GlobalEpochSampler
from riichienv.kar.learning.train import create_optimizer, create_scheduler
from riichienv.kar.learning.trainer import production_update
from riichienv.kar.learning.training_config import ProductionTrainConfig
from riichienv.kar.observation import ObserverLedger
from riichienv.kar.readiness import CORE_REQUIREMENTS, FULL_REMAINING_REQUIREMENTS, feature_readiness
from riichienv.kar.response_mask import public_rule_response_mask
from riichienv.kar.tokenize import append_action, encode_prefix

# These tests are source only in the code-first phase. Running them requires the
# user's separate explicit parameter-update test opt-in. They never authorize real data.
pytestmark = pytest.mark.parameter_updates
DATA_SHA = "f" * 64


class SyntheticDataset:
    def __init__(self):
        env = MJSOUL_RULESET.create_env(seed=17)
        observations = env.reset()
        ledger = ObserverLedger(0, (0, 1, 2, 3))
        for event in env.mjai_log:
            ledger.apply_event(event)
        snapshot = ledger.snapshot()
        ids = tuple(value.global_id for value in legal_candidates(observations[0]))
        action = next(value for value in ids if value < 148)
        prefix = encode_prefix(snapshot, ids)
        conditional = append_action(prefix, action)
        self.row = {
            "tokens": conditional.to_bytes()[108:],
            "candidates": ids,
            "decide_index": prefix.decide_index,
            "action_index": prefix.decide_index + 1,
            "action_target": action,
            "action_valid": True,
            "response_target": 0,
            "response_applicable": True,
            "response_observed": True,
            "final_rank": 0,
            "value_valid": True,
            "response_allowed": public_rule_response_mask(snapshot, action),
        }
        self.manifest = {"manifest_sha256": DATA_SHA, "rho_E": 1.0, "rho_V": 1.0}

    def __getitem__(self, index):
        return {**self.row, "game_id": f"synthetic-game-{index}", "decision_row_id": f"synthetic-row-{index}"}

    def __len__(self):
        return 3


def _authorization():
    checks = {key: True for key in (*CORE_REQUIREMENTS, *FULL_REMAINING_REQUIREMENTS)}
    readiness = feature_readiness(
        checks,
        evidence={key: "synthetic opt-in unit test only" for key in checks},
        source_hash=update_implementation_sha256(),
        data_eligible=True,
        resource_ready=True,
    )
    doc = {
        "version": "kar_explicit_update_authorization_v1",
        "explicit_user_approval": True,
        "approval_reference": "explicit --allow-parameter-updates for synthetic contract tests only",
        "readiness": readiness,
        "allowed_operations": ["production_update"],
        "dataset_manifest_sha256": DATA_SHA,
        "allowed_phases": ["M1"],
        "allowed_recipe_sha256": [config().recipe_sha256],
    }
    return UpdateAuthorization({**doc, "authorization_sha256": content_hash(doc)})


def config():
    estimate = CostForecast(
        "M1", 100, 100.0, 0.0, 0.0, "synthetic unit-test timing,not a production measurement", "a" * 64
    )
    return ProductionTrainConfig(
        "M1",
        "small",
        3,
        3,
        1,
        BudgetPlan("M1", 100.0, 0.0, estimate),
        device="cpu",
        precision="fp32",
        microbatch_size=1,
    )


def components():
    torch.manual_seed(23)
    raw = KARTransformer(ModelConfig(layers=1, width=16, heads=2, feedforward=32, preset="synthetic-update-test"))
    optimizer = create_optimizer(raw)
    scheduler = create_scheduler(optimizer, 2)
    return raw, optimizer, scheduler


def _worker(rank, world, port, output):
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)
    )
    torch.set_num_threads(1)
    context = DistributedContext.open("cpu", backend="gloo", gloo_bind_host="127.0.0.1")
    try:
        data = SyntheticDataset()
        raw, optimizer, scheduler = components()
        context.synchronize_model(raw)
        model = DistributedDataParallel(raw, broadcast_buffers=False, init_sync=False)
        group = GlobalEpochSampler(
            [65] * 3, seed=1, effective_batch=128, microbatch=1, rank=rank, world_size=world
        ).peek()
        result = production_update(model, raw, optimizer, scheduler, data, group, context, config(), _authorization())
        torch.save({"model": raw.state_dict(), "result": result}, Path(output) / f"rank-{rank}.pt")
    finally:
        context.close()


def test_real_production_global_sum_with_empty_microbatch_matches_single_rank(tmp_path, monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    workers = [get_context("spawn").Process(target=_worker, args=(rank, 2, port, str(tmp_path))) for rank in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(45)
        assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)
    context = DistributedContext.open("cpu")
    try:
        data = SyntheticDataset()
        raw, optimizer, scheduler = components()
        group = GlobalEpochSampler([65] * 3, seed=1, effective_batch=128, microbatch=1).peek()
        production_update(raw, raw, optimizer, scheduler, data, group, context, config(), _authorization())
        for rank in range(2):
            saved = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
            assert saved["result"]["global_examples"] == 3
            for name, value in raw.state_dict().items():
                torch.testing.assert_close(saved["model"][name], value, atol=3e-6, rtol=3e-5)
    finally:
        context.close()


def test_checkpoint_pointer_keeps_last_committed_generation_and_rejects_runtime_change(tmp_path):
    raw, optimizer, scheduler = components()
    device = torch.device("cpu")
    path = tmp_path / "checkpoint.pt"
    runtime = {"world_size": 1, "device": "cpu"}
    contract = {"synthetic": True}
    before = model_weights_sha256(raw)
    first = save_production_checkpoint(
        path,
        raw,
        optimizer,
        scheduler,
        contract=contract,
        trainer_state={"sampler": {"cursor": 0}},
        rank_rng=[rank_rng_state(device)],
        runtime=runtime,
        step=0,
    )
    second = save_production_checkpoint(
        path,
        raw,
        optimizer,
        scheduler,
        contract=contract,
        trainer_state={"sampler": {"cursor": 3}},
        rank_rng=[rank_rng_state(device)],
        runtime=runtime,
        step=1,
    )
    assert (tmp_path / first["checkpoint_file"]).exists() and (tmp_path / second["checkpoint_file"]).exists()
    # An orphan partial write does not modify the committed pointer.
    (tmp_path / "checkpoint.g000000000999-aaaaaaaaaaaa.pt").write_bytes(b"partial")
    payload = read_production_checkpoint(path, contract=contract, runtime=runtime)
    assert payload["metadata"]["step"] == 1 and model_weights_sha256(raw) == before
    with pytest.raises(ValueError):
        read_production_checkpoint(path, contract=contract, runtime={**runtime, "world_size": 2})
