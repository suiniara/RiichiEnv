"""Stage-1 End-to-End Real Benchmark for KAR V2.4 (Rust Encoding + PyTorch Throughput).

Measures real numbers on actual hardware (zero theoretical guesses):
1. Exact parameter counts for B (Spatial-Only), C1 (V2-Base), C1-Flat, C2 (V2-Large), OracleCriticV2.
2. Live `RiichiEnv` -> Rust V2 feature encoding latency (`us/step`, Actor-only vs Actor+Oracle).
3. PyTorch forward (inference) & forward+backward (training) throughput (`samples/sec`) and peak VRAM.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

import riichienv
from riichienv import _riichienv as native
from riichienv.kar.v2.model import (
    CAND_DYN_DIM,
    CAND_PAD_ID,
    EVENT_BYTES,
    GLOBAL_FEATS_DIM,
    MAX_CANDIDATES,
    ORACLE_PLAYER_DIM,
    ORACLE_TILE_DIM,
    PLAYER_FEATS_DIM,
    TILE_DIM,
    TILE_FEATS_DIM,
    CriticModelConfig,
    KARV2Actor,
    OracleCriticV2,
    V2ModelConfig,
    oracle_critic_loss,
    ppo_actor_loss,
)


def count_params(module: torch.nn.Module, *, exclude_belief: bool = False) -> int:
    total = 0
    for name, p in module.named_parameters():
        if exclude_belief and name.startswith("belief_heads."):
            continue
        total += p.numel()
    return total


def benchmark_rust_encoding(num_steps: int = 2000) -> dict[str, float]:
    """Measure real release-mode Rust V2 encoding latency over live RiichiEnv games."""
    env = riichienv.RiichiEnv(seed=2026)
    obs_map = env.reset()

    actor_times: list[float] = []
    bc_belief_times: list[float] = []
    full_times: list[float] = []
    steps_done = 0

    while steps_done < num_steps:
        if env.done():
            obs_map = env.reset()
        actions = {}
        for pid, obs in obs_map.items():
            t0 = time.perf_counter()
            _ = native.v2_encode_env_step(env, pid, "tenhou", False, False, False)
            t1 = time.perf_counter()
            _ = native.v2_encode_env_step(env, pid, "tenhou", False, False, True)
            t2 = time.perf_counter()
            _ = native.v2_encode_env_step(env, pid, "tenhou", False, True, False)
            t3 = time.perf_counter()
            actor_times.append((t1 - t0) * 1e6)
            bc_belief_times.append((t2 - t1) * 1e6)
            full_times.append((t3 - t2) * 1e6)
            steps_done += 1

            legal = obs.legal_actions()
            chosen = next((a for a in legal if a.action_type == riichienv.ActionType.PASS), None)
            if chosen is None:
                chosen = next(
                    (a for a in legal if a.action_type == riichienv.ActionType.DISCARD),
                    legal[0],
                )
            actions[pid] = chosen
            if steps_done >= num_steps:
                break
        obs_map = env.step(actions)

    actor_times.sort()
    bc_belief_times.sort()
    full_times.sort()
    n = len(actor_times)
    return {
        "steps_measured": float(n),
        "actor_only_mean_us": sum(actor_times) / n,
        "actor_only_p95_us": actor_times[int(0.95 * (n - 1))],
        "actor_only_steps_per_sec_1core": 1e6 / (sum(actor_times) / n),
        "actor_plus_belief_mean_us": sum(bc_belief_times) / n,
        "actor_plus_belief_p95_us": bc_belief_times[int(0.95 * (n - 1))],
        "actor_plus_belief_steps_per_sec_1core": 1e6 / (sum(bc_belief_times) / n),
        "actor_plus_oracle_mean_us": sum(full_times) / n,
        "actor_plus_oracle_p95_us": full_times[int(0.95 * (n - 1))],
        "actor_plus_oracle_steps_per_sec_1core": 1e6 / (sum(full_times) / n),
    }


def _make_synth_batch(batch_size: int, seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    cand_ids = torch.full((batch_size, MAX_CANDIDATES), CAND_PAD_ID, dtype=torch.long, device=device)
    cand_mask = torch.zeros((batch_size, MAX_CANDIDATES), dtype=torch.bool, device=device)
    cand_mode = torch.zeros((batch_size, MAX_CANDIDATES), dtype=torch.long, device=device)
    for k in range(14):
        cand_ids[:, k] = k * 4
        cand_mask[:, k] = True
        cand_mode[:, k] = 1

    events = torch.zeros((batch_size, seq_len, EVENT_BYTES), dtype=torch.uint8, device=device)
    events[:, 0, 0] = 1
    events[:, 0, 1] = 4
    events[:, 0, 2] = 4
    events[:, 0, 3] = 5
    events[:, 1:, 0] = 3
    events[:, 1:, 1] = 1
    events[:, 1:, 2] = 4
    events[:, 1:, 3] = 10
    events[:, :, 4] = 37
    events[:, :, 5] = 37

    return {
        "tile_feats": torch.randn(batch_size, TILE_DIM, TILE_FEATS_DIM, device=device),
        "oracle_tile": torch.randn(batch_size, TILE_DIM, ORACLE_TILE_DIM, device=device),
        "events": events,
        "event_len": torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
        "global_feats": torch.randn(batch_size, GLOBAL_FEATS_DIM, device=device),
        "player_feats": torch.randn(batch_size, 4, PLAYER_FEATS_DIM, device=device),
        "oracle_player": torch.randn(batch_size, 3, ORACLE_PLAYER_DIM, device=device),
        "cand_ids": cand_ids,
        "cand_mask": cand_mask,
        "cand_mode": cand_mode,
        "cand_dyn": torch.randn(batch_size, MAX_CANDIDATES, CAND_DYN_DIM, device=device),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench_actor(
    cfg: V2ModelConfig,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    amp_dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, float | int]:
    use_amp = device.type == "cuda"
    batch_size = batch["tile_feats"].shape[0]
    model = KARV2Actor(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    actor_kwargs = {
        "tile_feats": batch["tile_feats"],
        "events": batch["events"],
        "event_len": batch["event_len"],
        "global_feats": batch["global_feats"],
        "player_feats": batch["player_feats"],
        "cand_ids": batch["cand_ids"],
        "cand_mask": batch["cand_mask"],
        "cand_mode": batch["cand_mode"],
        "cand_dyn": batch["cand_dyn"],
    }

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                _ = model(**actor_kwargs)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                _ = model(**actor_kwargs)
        _sync(device)
        infer_s = (time.perf_counter() - t0) / iters

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    chosen_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    chosen_gid = batch["cand_ids"][:, 0]
    old_lp = torch.full((batch_size,), -2.5, device=device)
    adv = torch.randn(batch_size, device=device)
    learner_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    def _step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(**actor_kwargs, return_belief=True)
        loss = ppo_actor_loss(
            logits=out["logits"],
            cand_ids=batch["cand_ids"],
            cand_mask=batch["cand_mask"],
            chosen_cand_idx=chosen_idx,
            chosen_global_id=chosen_gid,
            old_log_prob=old_lp,
            advantages=adv,
            is_learner_seat=learner_mask,
        )["loss"]
        loss.backward()
        opt.step()

    for _ in range(warmup):
        _step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _step()
    _sync(device)
    train_s = (time.perf_counter() - t0) / iters
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "params_total": count_params(model),
        "params_inference_only": count_params(model, exclude_belief=True),
        "infer_samples_per_sec": batch_size / infer_s,
        "train_samples_per_sec": batch_size / train_s,
        "peak_vram_mb": peak_mb,
    }


def _bench_critic(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    amp_dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, float | int]:
    use_amp = device.type == "cuda"
    batch_size = batch["tile_feats"].shape[0]
    critic = OracleCriticV2(CriticModelConfig()).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    grp_probs = torch.full((batch_size, 4), 0.25, device=device)
    payoffs = torch.tensor([0.90, 0.45, 0.0, -1.35], device=device)
    ranks = torch.randint(0, 4, (batch_size,), device=device)
    targets = torch.randn(batch_size, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def _step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            co = critic(
                tile_feats=batch["tile_feats"],
                oracle_tile=batch["oracle_tile"],
                events=batch["events"],
                event_len=batch["event_len"],
                global_feats=batch["global_feats"],
                player_feats=batch["player_feats"],
                oracle_player=batch["oracle_player"],
                grp_prior_probs=grp_probs,
                rank_payoffs=payoffs,
            )
        cl = oracle_critic_loss(
            rank_logits=co["rank_logits"],
            v_shaped=co["v_shaped"],
            final_rank_idx=ranks,
            shaped_target=targets,
        )["loss"]
        cl.backward()
        opt.step()

    for _ in range(warmup):
        _step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _step()
    _sync(device)
    train_s = (time.perf_counter() - t0) / iters
    return {
        "params_total": count_params(critic),
        "train_samples_per_sec": batch_size / train_s,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0),
    }


def benchmark_models(
    *,
    batch_size: int = 64,
    seq_len: int = 64,
    warmup: int = 3,
    iters: int = 10,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    batch = _make_synth_batch(batch_size, seq_len, device)

    presets = {
        "B_spatial_only": V2ModelConfig.spatial_only(),
        "C1_v2_base": V2ModelConfig.v2_base(),
        "C1_flat546": V2ModelConfig.v2_base_flat(),
        "C2_v2_large": V2ModelConfig.v2_large(),
    }
    models_out: dict[str, Any] = {
        name: _bench_actor(cfg, batch, device, amp_dtype, warmup, iters) for name, cfg in presets.items()
    }
    models_out["OracleCriticV2_independent"] = _bench_critic(batch, device, amp_dtype, warmup, iters)
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "amp": str(amp_dtype) if use_amp else "fp32",
        "models": models_out,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KAR V2 Rust encoding and PyTorch throughput benchmark.")
    parser.add_argument("--rust-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args(argv)

    rust_bench = benchmark_rust_encoding(num_steps=args.rust_steps)
    model_bench = benchmark_models(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(json.dumps({"rust_encoding": rust_bench, "pytorch_models": model_bench}, indent=2))


if __name__ == "__main__":
    main()
