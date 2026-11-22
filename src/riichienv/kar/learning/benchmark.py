from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ..schema import decode_token
from ..tokenize import TokenSequence, encode_action_token
from .batch import collate_inputs
from .config import ModelConfig
from .dataset import DecisionDataset
from .model import KARTransformer


def peak_process_memory_bytes() -> int | None:
    if sys.platform == "win32":
        from ctypes import wintypes  # noqa: PLC0415

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    try:
        import resource  # noqa: PLC0415

        size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(size if sys.platform == "darwin" else size * 1024)
    except ImportError:
        return None


@torch.no_grad()
def benchmark_candidates(dataset_dir: Path, *, preset: str = "small", repetitions: int = 5) -> dict[str, Any]:
    if not 1 <= repetitions <= 100:
        raise ValueError("Use a bounded inference benchmark")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    dataset = DecisionDataset(dataset_dir, "validation")
    row = None
    discard_candidates: list[int] = []
    for index in range(len(dataset)):
        candidate_row = dataset[index]
        discards = [action for action in candidate_row["candidates"] if 0 <= action < 148]
        if discards:
            row = candidate_row
            discard_candidates = discards
            break
    if row is None:
        raise ValueError("Validation dataset has no decision with legal discard candidates")
    count = row["decide_index"] + 1
    prefix_bytes = row["tokens"][: count * 139]
    tokens = tuple(decode_token(prefix_bytes[index : index + 139]) for index in range(0, len(prefix_bytes), 139))
    prefix = TokenSequence(tokens, tuple(row["candidates"]), count - 64, row["decide_index"], tokens[-1]["positions"])
    prefix_input = collate_inputs([{**row, "tokens": prefix_bytes, "action_index": -1}])
    branches = [
        {**row, "tokens": encode_action_token(prefix, action), "decide_index": 0, "action_index": 0}
        for action in discard_candidates
    ]
    action_input = collate_inputs(branches)
    model = KARTransformer(ModelConfig.from_preset(preset)).eval()
    cache = model.encode_prefix(prefix_input)
    model.score_actions(cache, action_input)
    start = time.perf_counter()
    for _ in range(repetitions):
        cache = model.encode_prefix(prefix_input)
    prefix_seconds = (time.perf_counter() - start) / repetitions
    start = time.perf_counter()
    for _ in range(repetitions):
        response, ranks = model.score_actions(cache, action_input)
    branch_seconds = time.perf_counter() - start
    return {
        "device": "cpu",
        "preset": preset,
        "repetitions": repetitions,
        "candidates": len(branches),
        "prefix_tokens": count,
        "prefix_seconds": prefix_seconds,
        "candidate_suffixes_per_second": repetitions * len(branches) / branch_seconds,
        "peak_process_memory_bytes": peak_process_memory_bytes(),
        "gpu_memory_measured": False,
        "simulation_matches_per_second": None,
        "response_shape": list(response.shape),
        "rank_shape": list(ranks.shape),
        "scope": "one_validation_decision_inference_throughput_not_playing_strength",
        "weights": "random_initialization_architecture_throughput_only",
    }
