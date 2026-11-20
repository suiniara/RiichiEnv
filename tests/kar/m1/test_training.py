import copy
import json
import random
import socket
from dataclasses import asdict, replace
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from riichienv.kar.authorization import UpdateAuthorization, update_implementation_sha256
from riichienv.kar.contracts import content_hash
from riichienv.kar.learning.batch import ModelBatch
from riichienv.kar.learning.checkpoint import load_checkpoint, save_checkpoint
from riichienv.kar.learning.config import ModelConfig
from riichienv.kar.learning.dataset import TrainingBatch, collate_examples
from riichienv.kar.learning.losses import Labels, loss_sums
from riichienv.kar.learning.model import KARTransformer
from riichienv.kar.learning.train import create_optimizer, create_scheduler, optimizer_update, parameter_groups
from riichienv.kar.readiness import CORE_REQUIREMENTS, FULL_REMAINING_REQUIREMENTS, feature_readiness

pytestmark = pytest.mark.parameter_updates


def _legacy_test_authorization(dataset_sha="e" * 64):
    # Constructed only inside tests that the user explicitly opted into. The
    # synthetic data binding cannot authorize any production corpus by default.
    checks = {key: True for key in (*CORE_REQUIREMENTS, *FULL_REMAINING_REQUIREMENTS)}
    ready = feature_readiness(
        checks,
        evidence={key: "synthetic opt-in test evidence" for key in checks},
        source_hash=update_implementation_sha256(),
        data_eligible=True,
        resource_ready=True,
    )
    doc = {
        "version": "kar_explicit_update_authorization_v1",
        "explicit_user_approval": True,
        "approval_reference": "explicit parameter-update pytest opt-in",
        "readiness": ready,
        "allowed_operations": ["optimizer_update", "train_cpu"],
        "dataset_manifest_sha256": dataset_sha,
    }
    return UpdateAuthorization({**doc, "authorization_sha256": content_hash(doc)})


def _training_batch(make_sample):
    rows = []
    for seed in (42, 0, 8):
        row = make_sample(seed)
        rows.append(
            {
                **row,
                "action_valid": True,
                "response_target": 0,
                "response_applicable": True,
                "response_observed": True,
                "final_rank": 1,
                "value_valid": True,
                "response_allowed": [True] * 21,
            }
        )
    batch = collate_examples(rows)
    active = torch.tensor([True, False, True])
    labels = replace(
        batch.labels,
        response_applicable=active,
        response_observed=active,
        value_valid=active,
        response_target=torch.tensor([0, 255, 0]),
        final_rank_target=torch.tensor([1, 255, 1]),
    )
    return replace(batch, labels=labels)


def _slice_batch(batch, start, end):
    inputs = replace(
        batch.inputs,
        fields={key: value[start:end] for key, value in batch.inputs.fields.items()},
        candidates=batch.inputs.candidates[start:end],
        decide_index=batch.inputs.decide_index[start:end],
        action_index=batch.inputs.action_index[start:end],
        context_ids=batch.inputs.context_ids[start:end],
    )
    return TrainingBatch(
        inputs,
        Labels(**{key: value[start:end] for key, value in batch.labels.__dict__.items()}),
        batch.game_ids[start:end],
    )


def test_scheduler_warmup_is_bounded_by_short_training_horizon(miniature):
    from riichienv.kar.learning.train import create_scheduler

    optimizer = create_optimizer(miniature, learning_rate=3e-4)
    short = create_scheduler(optimizer, total_updates=16)
    # 3% of a short horizon rounds to one update; it must not become 100.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    optimizer.step()
    short.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)

    optimizer = create_optimizer(miniature, learning_rate=3e-4)
    long = create_scheduler(optimizer, total_updates=1000)
    # The same proportional rule retains a 30-update warmup for long runs.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 / 30)
    for _ in range(29):
        optimizer.step()
        long.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_adamw_groups_are_exact_and_disjoint(miniature):
    groups = parameter_groups(miniature)
    assignment = {id(parameter): group["weight_decay"] for group in groups for parameter in group["params"]}
    assert len(assignment) == len(list(miniature.parameters()))
    for name, parameter in miniature.named_parameters():
        expected = 0.05 if name == "value_head.weight" else 0.01 if parameter.ndim >= 2 else 0
        assert assignment[id(parameter)] == expected


def test_uneven_gradient_accumulation_matches_full_update(miniature, make_sample):
    batch = _training_batch(make_sample)
    other = copy.deepcopy(miniature)
    optimizer = create_optimizer(miniature)
    second = create_optimizer(other)
    optimizer_update(
        miniature,
        optimizer,
        [batch],
        global_examples=3,
        rho_e=2 / 3,
        rho_v=2 / 3,
        authorization=_legacy_test_authorization(),
    )
    optimizer_update(
        other,
        second,
        [_slice_batch(batch, 0, 1), _slice_batch(batch, 1, 3)],
        global_examples=3,
        rho_e=2 / 3,
        rho_v=2 / 3,
        authorization=_legacy_test_authorization(),
    )
    for first, last in zip(miniature.parameters(), other.parameters()):
        torch.testing.assert_close(first, last, atol=4e-6, rtol=2e-4)


def test_checkpoint_restores_rng_optimizer_sampler_and_next_update(tmp_path, miniature, make_sample):
    batch = _training_batch(make_sample)
    optimizer = create_optimizer(miniature)
    scheduler = create_scheduler(optimizer, 500)
    optimizer_update(
        miniature,
        optimizer,
        [batch],
        global_examples=3,
        rho_e=2 / 3,
        rho_v=2 / 3,
        scheduler=scheduler,
        authorization=_legacy_test_authorization(),
    )
    path = tmp_path / "state.pt"
    contract = {"data": "frozen", "seed": 7}
    save_checkpoint(
        path,
        miniature,
        optimizer,
        scheduler,
        run_contract=contract,
        sampler_state={"cursor": 3, "order": torch.arange(8)},
        step=1,
    )
    random_expected = (random.random(), np.random.rand(), torch.rand(3))
    optimizer_update(
        miniature,
        optimizer,
        [batch],
        global_examples=3,
        rho_e=2 / 3,
        rho_v=2 / 3,
        scheduler=scheduler,
        authorization=_legacy_test_authorization(),
    )
    expected = {key: value.clone() for key, value in miniature.state_dict().items()}
    restored = KARTransformer(miniature.config)
    new_optimizer = create_optimizer(restored)
    new_scheduler = create_scheduler(new_optimizer, 500)
    info = load_checkpoint(path, restored, new_optimizer, new_scheduler, run_contract=contract)
    assert info["step"] == 1 and info["sampler"]["cursor"] == 3
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == random_expected[:2]
    torch.testing.assert_close(actual[2], random_expected[2], rtol=0, atol=0)
    optimizer_update(
        restored,
        new_optimizer,
        [batch],
        global_examples=3,
        rho_e=2 / 3,
        rho_v=2 / 3,
        scheduler=new_scheduler,
        authorization=_legacy_test_authorization(),
    )
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert scheduler.state_dict() == new_scheduler.state_dict()


def test_checkpoint_rejects_different_data_or_recipe(tmp_path, miniature):
    optimizer = create_optimizer(miniature)
    scheduler = create_scheduler(optimizer, 200)
    path = tmp_path / "state.pt"
    save_checkpoint(path, miniature, optimizer, scheduler, run_contract={"data": "first"}, sampler_state={}, step=0)
    with pytest.raises(ValueError, match="resume refused"):
        load_checkpoint(path, miniature, run_contract={"data": "different"})
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar["metadata"]["schema_hash"] = "0" * 64
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_checkpoint(path, miniature, run_contract={"data": "first"})


def test_real_trainer_interrupt_resume_matches_clean_run(tmp_path, monkeypatch):
    from riichienv.kar.learning import train as training  # noqa: PLC0415

    dataset = Path(__file__).parents[4] / "kar-m1-data" / "cpu-probe-v2"
    if not (dataset / "manifest.json").is_file():
        pytest.skip("Private CPU dataset is unavailable")
    config = training.TrainConfig(
        preset="tiny",
        max_updates=3,
        effective_batch_size=2,
        microbatch_size=1,
        checkpoint_interval=1,
        validation_examples=2,
    )
    original = training.save_checkpoint

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        if kwargs["step"] == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(training, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.train_cpu(
            dataset,
            tmp_path / "resumed",
            config,
            authorization=_legacy_test_authorization(
                json.loads((dataset / "manifest.json").read_text())["manifest_sha256"]
            ),
        )
    monkeypatch.setattr(training, "save_checkpoint", original)
    resumed = training.train_cpu(
        dataset,
        tmp_path / "resumed",
        config,
        resume=True,
        authorization=_legacy_test_authorization(
            json.loads((dataset / "manifest.json").read_text())["manifest_sha256"]
        ),
    )
    clean = training.train_cpu(
        dataset,
        tmp_path / "clean",
        config,
        authorization=_legacy_test_authorization(
            json.loads((dataset / "manifest.json").read_text())["manifest_sha256"]
        ),
    )
    assert resumed["sample_order_sha256"] == clean["sample_order_sha256"]
    assert resumed["train_history"] == clean["train_history"]
    assert resumed["validation_before"] == clean["validation_before"]
    assert resumed["validation_after"] == clean["validation_after"]
    left_pointer = json.loads((tmp_path / "resumed" / "checkpoint.json").read_text())
    right_pointer = json.loads((tmp_path / "clean" / "checkpoint.json").read_text())
    left = torch.load(tmp_path / "resumed" / left_pointer["checkpoint_file"], weights_only=True)
    right = torch.load(tmp_path / "clean" / right_pointer["checkpoint_file"], weights_only=True)
    for name, value in left["model"].items():
        torch.testing.assert_close(value, right["model"][name], rtol=0, atol=0)


def _loopback_gloo_backend(store, rank, size, timeout):
    options = dist.ProcessGroupGloo._Options()
    options._timeout = timeout
    options._devices = [dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")]
    options._threads = 1
    return dist.ProcessGroupGloo(store, rank, size, options)


def _gloo_worker(rank, source, output, rendezvous):
    torch.set_num_threads(1)
    payload = torch.load(source, weights_only=True)
    model = KARTransformer(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    stage = Path(output) / f"stage-{rank}.txt"
    stage.write_text("before_process_group", encoding="utf-8")
    store = dist.TCPStore("127.0.0.1", rendezvous, 2, rank == 0, timeout=timedelta(seconds=20), use_libuv=False)
    stage.write_text("store_ready", encoding="utf-8")
    # This factory constructs the actual native Gloo group with explicit IPv4
    # options; the built-in PyTorch 2.15 factory discards Gloo pg_options.
    dist.Backend.register_backend("kar_loopback_gloo", _loopback_gloo_backend, devices=["cpu"])
    dist.init_process_group("kar_loopback_gloo", store=store, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    stage.write_text("process_group_ready", encoding="utf-8")
    try:
        indices = torch.arange(rank, 3, 2)
        fields = {key: value[indices] for key, value in payload["fields"].items()}
        inputs = ModelBatch(
            fields,
            payload["candidates"][indices],
            payload["decide_index"][indices],
            payload["action_index"][indices],
            tuple(payload["ids"][index] for index in indices),
        )
        labels = Labels(**{key: value[indices] for key, value in payload["labels"].items()})
        wrapped = DistributedDataParallel(model)
        stage.write_text("ddp_ready", encoding="utf-8")
        prediction = wrapped(inputs)
        stage.write_text("forward_ready", encoding="utf-8")
        objective = loss_sums(prediction, labels, inputs.legal_mask).objective(
            global_examples=3, world_size=2, rho_e=2 / 3, rho_v=2 / 3
        )
        objective.backward()
        gradient = torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
        torch.save(gradient, Path(output) / f"gradient-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_real_gloo_ddp_with_uneven_valid_counts_matches_global_gradient(tmp_path, miniature, make_sample):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo backend unavailable; DDP acceptance is not verified")
    batch = _training_batch(make_sample)
    source = tmp_path / "ddp-source.pt"
    torch.save(
        {
            "config": asdict(miniature.config),
            "model": miniature.state_dict(),
            "fields": batch.inputs.fields,
            "candidates": batch.inputs.candidates,
            "decide_index": batch.inputs.decide_index,
            "action_index": batch.inputs.action_index,
            "ids": list(batch.inputs.context_ids),
            "labels": batch.labels.__dict__,
        },
        source,
    )
    loss_sums(miniature(batch.inputs), batch.labels, batch.inputs.legal_mask).objective(
        global_examples=3, world_size=1, rho_e=2 / 3, rho_v=2 / 3
    ).backward()
    expected = torch.cat([parameter.grad.flatten() for parameter in miniature.parameters()])
    context = get_context("spawn")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = [
        context.Process(target=_gloo_worker, args=(rank, str(source), str(tmp_path), port)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=45)
        stages = [
            (tmp_path / f"stage-{rank}.txt").read_text(encoding="utf-8")
            if (tmp_path / f"stage-{rank}.txt").is_file()
            else "not_entered"
            for rank in range(2)
        ]
        assert not any(process.is_alive() for process in processes), f"DDP workers did not finish: {stages}"
        assert all(process.exitcode == 0 for process in processes), stages
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        actual = torch.load(tmp_path / f"gradient-{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-4)
