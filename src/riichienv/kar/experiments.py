from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import MJSOUL_RULESET, content_hash

M3_VERSION = "balanced_paired_seed_blocks_v2"
M3_PAIRS = 10000
M3_MATCHES = 20000


def independent_seed(master: int, *parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(str([master, *parts]).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class PairedSeedBlock:
    pair_index: int
    subject_seat: int
    wall_seed: int
    baseline_subject_rng_seed: int
    candidate_subject_rng_seed: int
    opponent_rng_seeds: tuple[int, int, int]
    opponent_release_ids: tuple[str, str, str]
    initial_scores: tuple[int, int, int, int] = (25000, 25000, 25000, 25000)
    initial_seat_order: tuple[int, int, int, int] = (0, 1, 2, 3)

    def to_record(self) -> dict[str, Any]:
        value = json.loads(json.dumps(asdict(self)))
        return {**value, "block_sha256": content_hash(value)}


def paired_seed_manifest(
    *,
    master_seed: int,
    subject_release_sha256: str,
    opponent_release_ids: tuple[str, str, str],
    partition: str = "test",
    inference_device: str = "cpu",
    statistics_seed: int = 17,
    bootstrap_replicates: int = 10000,
    max_decisions: int = 10000,
    benchmark_sha256: str | None = None,
) -> dict[str, Any]:
    if (
        type(master_seed) is not int
        or master_seed < 0
        or len(subject_release_sha256) != 64
        or len(opponent_release_ids) != 3
        or any(len(value) != 64 for value in opponent_release_ids)
    ):
        raise ValueError("M3 needs explicit immutable subject/opponent releases and seed")
    if partition not in {"validation", "test"}:
        raise ValueError("Unknown independent evaluation partition")
    if (
        inference_device not in {"cpu", "cuda"}
        or bootstrap_replicates < 1
        or max_decisions < 1
        or type(statistics_seed) is not int
    ):
        raise ValueError("Preregister the inference backend,complete-match cap and cluster statistics recipe")
    if benchmark_sha256 is not None and (
        not isinstance(benchmark_sha256, str)
        or len(benchmark_sha256) != 64
        or any(char not in "0123456789abcdef" for char in benchmark_sha256)
    ):
        raise ValueError("Freeze the exact completed benchmark SHA-256, not a throughput estimate")
    blocks = []
    for index in range(M3_PAIRS):
        blocks.append(
            PairedSeedBlock(
                index,
                index % 4,
                independent_seed(master_seed, "wall", index),
                independent_seed(master_seed, "subject", "baseline", index),
                independent_seed(master_seed, "subject", "candidate", index),
                (
                    independent_seed(master_seed, "opponent", index, 0),
                    independent_seed(master_seed, "opponent", index, 1),
                    independent_seed(master_seed, "opponent", index, 2),
                ),
                opponent_release_ids,
            ).to_record()
        )
    record = {
        "version": M3_VERSION,
        "partition": partition,
        "master_seed": master_seed,
        "pair_count": M3_PAIRS,
        "match_count": M3_MATCHES,
        "subject_release_sha256": subject_release_sha256,
        "opponent_release_ids": list(opponent_release_ids),
        "ruleset_hash": MJSOUL_RULESET.ruleset_hash,
        "seat_counts": [M3_PAIRS // 4] * 4,
        "blocks": blocks,
        "RNG_contract": (
            "independent subject arms; registered opponent streams; different actions naturally fork trajectories"
        ),
        "counterfactual_state_equality_claimed": False,
        "execution": {"device": inference_device, "max_decisions": max_decisions},
        "forecast_evidence": {"benchmark_sha256": benchmark_sha256},
        "execution_ready": benchmark_sha256 is not None,
        "statistics": {
            "seed": statistics_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "confidence": 0.95,
            "unit": "paired_seed_block",
        },
    }
    return {**record, "paired_seed_manifest_sha256": content_hash(record)}


def validate_paired_manifest(document: dict[str, Any], *, require_benchmark: bool = False) -> None:
    unsigned = {key: value for key, value in document.items() if key != "paired_seed_manifest_sha256"}
    if document.get("version") != M3_VERSION or document.get("paired_seed_manifest_sha256") != content_hash(unsigned):
        raise ValueError("Paired seed manifest integrity mismatch")
    expected = paired_seed_manifest(
        master_seed=document["master_seed"],
        subject_release_sha256=document["subject_release_sha256"],
        opponent_release_ids=tuple(document["opponent_release_ids"]),
        partition=document["partition"],
        inference_device=document["execution"]["device"],
        statistics_seed=document["statistics"]["seed"],
        bootstrap_replicates=document["statistics"]["bootstrap_replicates"],
        max_decisions=document["execution"]["max_decisions"],
        benchmark_sha256=document["forecast_evidence"]["benchmark_sha256"],
    )
    if expected != document:
        raise ValueError("M3 seed count,seat balance,rules,RNG or opponent identity changed")
    if require_benchmark and not document["execution_ready"]:
        raise ValueError("A draft paired manifest cannot start M3 without measured policy-pair throughput")
