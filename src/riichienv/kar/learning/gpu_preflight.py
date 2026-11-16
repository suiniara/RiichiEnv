from __future__ import annotations

import argparse
import copy
import json
import platform
import socket
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..authorization import UpdateAuthorization, load_update_authorization, require_parameter_updates
from ..contracts import content_hash
from ..schema import decode_token
from ..tokenize import TokenSequence, append_action
from .batch import collate_inputs
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig
from .dataset import DecisionDataset, collate_examples
from .losses import loss_sums
from .model import KARTransformer
from .train import create_optimizer, create_scheduler


def representative_indices(lengths: list[int]) -> list[int]:
    if not lengths:
        raise ValueError("No preflight training rows")
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    return list(dict.fromkeys(order[round((len(order) - 1) * quantile)] for quantile in (0, 0.5, 0.95, 0.99, 1)))


def _finite_gradients(model: KARTransformer) -> None:
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise FloatingPointError("Missing/non-finite GPU gradients")


@torch.no_grad()
def _candidate_check(model: KARTransformer, sample: dict[str, Any], device: str) -> dict[str, Any]:
    model.eval()
    prefix_count = sample["decide_index"] + 1
    raw = sample["tokens"][: prefix_count * 139]
    rows = tuple(decode_token(raw[index : index + 139]) for index in range(0, len(raw), 139))
    sequence = TokenSequence(
        rows, tuple(sample["candidates"]), prefix_count - 64, sample["decide_index"], rows[-1]["positions"]
    )
    prefix = collate_inputs([{**sample, "tokens": raw, "action_index": -1}]).to(device)
    dense_samples, branch_samples = [], []
    for action in sample["candidates"]:
        conditional = append_action(sequence, action)
        dense_samples.append({**sample, "tokens": conditional.to_bytes()[108:]})
        branch_samples.append({**sample, "tokens": conditional.to_bytes()[-139:], "action_index": 0, "decide_index": 0})
    cache = model.encode_prefix(prefix)
    branch = collate_inputs(branch_samples).to(device)
    response, ranks = model.score_actions(cache, branch)
    # Compare each dense branch independently to keep VRAM bounded by microbatch=1.
    maximum = 0.0
    for index, sample_row in enumerate(dense_samples):
        dense = model(collate_inputs([sample_row]).to(device))
        torch.testing.assert_close(response[index], dense.response_logits[0], rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(ranks[index], dense.rank_logits[0], rtol=3e-4, atol=3e-4)
        maximum = max(maximum, float((response[index] - dense.response_logits[0]).abs().max()))
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        model.score_actions(cache, branch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "status": "PASS",
        "candidate_count": len(branch_samples),
        "prefix_tokens": prefix_count,
        "dense_cache_max_absolute_response_difference": maximum,
        "candidate_suffixes_per_second": 3 * len(branch_samples) / elapsed,
    }


def run_preflight(  # noqa: PLR0915
    dataset_dir: Path,
    *,
    max_reserved_gib: float = 4.5,
    scratch_dir: Path | None = None,
    authorization: UpdateAuthorization | None = None,
) -> dict[str, Any]:
    require_parameter_updates(authorization, operation="run_preflight")
    if not 1 <= max_reserved_gib <= 4.5:
        raise ValueError("Remote laptop preflight cap must remain in [1,4.5] GiB")
    data = DecisionDataset(dataset_dir, "train", history_cache_items=8)
    require_parameter_updates(authorization, operation="run_preflight", dataset_hash=data.manifest["manifest_sha256"])
    rho_e, rho_v = data.manifest["rho_E"], data.manifest["rho_V"]
    torch.set_num_threads(1)
    torch.manual_seed(17)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; never substitute the controller CPU/GPU")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < 4 * 1024**3:
        raise RuntimeError("Insufficient free VRAM after CUDA initialization")
    cap = min(max_reserved_gib * 1024**3, free - 1.5 * 1024**3)
    if cap < 1024**3:
        raise RuntimeError("Cannot preserve the minimum other-process VRAM reserve")
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = representative_indices([length + 65 for _, length in data.indices])
    config = ModelConfig.from_preset("small")
    cpu = KARTransformer(config).eval()
    gpu = copy.deepcopy(cpu).to(device).eval()
    cases = []
    bf16 = torch.cuda.is_bf16_supported()
    for ordinal, index in enumerate(selected):
        sample = data[index]
        batch = collate_examples([sample])
        moved = batch.to(device)
        cpu.zero_grad(set_to_none=True)
        gpu.zero_grad(set_to_none=True)
        cpu_outputs = cpu(batch.inputs)
        gpu_outputs = gpu(moved.inputs)
        differences = {}
        for key in ("policy_logits", "response_logits", "rank_logits"):
            first, second = getattr(cpu_outputs, key), getattr(gpu_outputs, key).cpu()
            torch.testing.assert_close(first, second, rtol=3e-4, atol=3e-4)
            differences[key] = float((first - second).abs().max())
        objective = loss_sums(gpu_outputs, moved.labels, moved.inputs.legal_mask).objective(
            global_examples=1, world_size=1, rho_e=rho_e, rho_v=rho_v
        )
        objective.backward()
        _finite_gradients(gpu)
        gradient_difference = None
        if ordinal == 0:
            reference = loss_sums(cpu_outputs, batch.labels, batch.inputs.legal_mask).objective(
                global_examples=1, world_size=1, rho_e=rho_e, rho_v=rho_v
            )
            reference.backward()
            largest = 0.0
            for first, second in zip(cpu.parameters(), gpu.parameters()):
                torch.testing.assert_close(first.grad, second.grad.cpu(), rtol=3e-3, atol=3e-5)
                largest = max(largest, float((first.grad - second.grad.cpu()).abs().max()))
            gradient_difference = largest
        case: dict[str, Any] = {
            "status": "PASS",
            "row_id": sample["decision_row_id"],
            "sequence_tokens": len(sample["tokens"]) // 139,
            "FP32_objective": float(objective.detach()),
            "FP32_logits_difference": differences,
            "CPU_GPU_gradient_max_absolute_difference": gradient_difference,
        }
        if bf16:
            gpu.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = gpu(moved.inputs)
                loss = loss_sums(outputs, moved.labels, moved.inputs.legal_mask).objective(
                    global_examples=1, world_size=1, rho_e=rho_e, rho_v=rho_v
                )
            loss.backward()
            _finite_gradients(gpu)
            if any(
                getattr(outputs, key).dtype != torch.float32
                for key in ("policy_logits", "response_logits", "rank_logits")
            ):
                raise ValueError("Mixed-precision output/loss heads must remain FP32")
            case["BF16_objective"] = float(loss.detach())
            case["BF16_forward_backward"] = "PASS"
        cases.append(case)
        del gpu_outputs, cpu_outputs, moved, batch
    candidate = _candidate_check(gpu, data[selected[0]], str(device))
    # Exercise an actual CUDA optimizer step and reject a stale prefix cache.
    sample = data[selected[0]]
    prefix = collate_inputs([{**sample, "tokens": sample["tokens"][:-139], "action_index": -1}]).to(device)
    old_cache = gpu.encode_prefix(prefix)
    gpu.train()
    batch = collate_examples([sample]).to(device)
    optimizer = create_optimizer(gpu)
    optimizer.zero_grad(set_to_none=True)
    objective = loss_sums(gpu(batch.inputs), batch.labels, batch.inputs.legal_mask).objective(
        global_examples=1, world_size=1, rho_e=rho_e, rho_v=rho_v
    )
    objective.backward()
    _finite_gradients(gpu)
    torch.nn.utils.clip_grad_norm_(gpu.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    gpu.eval()
    action = collate_inputs([{**sample, "tokens": sample["tokens"][-139:], "decide_index": 0, "action_index": 0}]).to(
        device
    )
    try:
        gpu.score_actions(old_cache, action)
    except ValueError:
        stale_rejected = True
    else:
        raise RuntimeError("CUDA update did not invalidate the prefix cache")
    rng = torch.cuda.get_rng_state(device)
    expected = torch.rand(16, device=device)
    torch.cuda.set_rng_state(rng, device)
    torch.testing.assert_close(torch.rand(16, device=device), expected, rtol=0, atol=0)
    scheduler = create_scheduler(optimizer, 100)
    contract = {"phase": "remote_gpu_preflight", "dataset": data.manifest["manifest_sha256"], "device": str(device)}
    with tempfile.TemporaryDirectory(prefix="cuda-checkpoint-", dir=scratch_dir) as temporary:
        checkpoint = Path(temporary) / "state.pt"
        saved_parameters = {key: value.detach().cpu().clone() for key, value in gpu.state_dict().items()}
        save_checkpoint(
            checkpoint,
            gpu,
            optimizer,
            scheduler,
            run_contract=contract,
            sampler_state={"cursor": 1, "order": torch.tensor(selected)},
            step=1,
        )
        expected_rng = torch.rand(16, device=device)
        restored = load_checkpoint(checkpoint, gpu, optimizer, scheduler, run_contract=contract)
        if restored["step"] != 1 or restored["sampler"]["cursor"] != 1:
            raise RuntimeError("CUDA checkpoint cursor mismatch")
        for key, value in gpu.state_dict().items():
            torch.testing.assert_close(value.cpu(), saved_parameters[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(16, device=device), expected_rng, rtol=0, atol=0)
    torch.cuda.synchronize()
    return {
        "status": "PASS",
        "scope": "remote_small_sample_GPU_preflight_not_full_M1_probe",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "CUDA_runtime": torch.version.cuda,
        "GPU": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "compiled_architectures": torch.cuda.get_arch_list(),
        "GPU_total_GiB": total / 1024**3,
        "GPU_free_after_init_GiB": free / 1024**3,
        "configured_allocator_cap_GiB": cap / 1024**3,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved(device) / 1024**3,
        "parameter_count": sum(parameter.numel() for parameter in gpu.parameters()),
        "config": asdict(config),
        "dataset_manifest_sha256": data.manifest["manifest_sha256"],
        "dataset_train_games": data.manifest["N_train_half_games"],
        "representative_cases": cases,
        "BF16_supported": bf16,
        "candidate_check": candidate,
        "CUDA_optimizer_step": "PASS",
        "stale_cache_rejected": stale_rejected,
        "CUDA_RNG_round_trip": "PASS",
        "CUDA_checkpoint_round_trip": "PASS",
        "full_epoch_started": False,
        "cloud_spent_usd": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded, target-machine Small CUDA preflight only")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-reserved-gib", type=float, default=4.5)
    parser.add_argument("--update-authorization", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite an existing preflight result")
    result: dict[str, Any]
    start = time.perf_counter()
    try:
        result = run_preflight(
            args.dataset,
            max_reserved_gib=args.max_reserved_gib,
            scratch_dir=args.output.parent,
            authorization=load_update_authorization(args.update_authorization),
        )
        code = 0
    except Exception as error:
        result = {
            "status": "FAIL",
            "scope": "remote_small_sample_GPU_preflight_not_full_M1_probe",
            "host": socket.gethostname(),
            "error_type": type(error).__name__,
            "detail": str(error),
            "full_epoch_started": False,
        }
        code = 1
    result["elapsed_seconds"] = time.perf_counter() - start
    result["report_sha256"] = content_hash(result)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
