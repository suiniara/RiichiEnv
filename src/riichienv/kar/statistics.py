from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .contracts import Response
from .runtime_contracts import UTILITIES

STATISTICS_VERSION = "whole_game_cluster_statistics_v2"


def _new_gate() -> dict[str, Any]:
    return {
        "eligible": 0,
        "excluded": 0,
        "fewer_than_two": 0,
        "support_mass_sum": 0.0,
        "support_mass_histogram": [0] * 101,
        "support_size_histogram": [0] * 33,
    }


def _add_gate(value: dict[str, Any], *, retained: bool, support_size: int, mass: float) -> None:
    if type(retained) is not bool or type(support_size) is not int or not 0 <= support_size <= 32:
        raise ValueError("Gate statistics require the complete distinct candidate support set")
    if isinstance(mass, bool) or not math.isfinite(mass) or not 0 <= mass <= 1 + 1e-6:
        raise ValueError("Gate mass must be a finite full-prior probability")
    value["eligible"] += 1
    value["excluded"] += not retained
    value["fewer_than_two"] += support_size < 2
    value["support_mass_sum"] += mass
    value["support_mass_histogram"][min(100, int(mass * 100))] += 1
    value["support_size_histogram"][support_size] += 1


def _merge_gate(value: dict[str, Any], other: dict[str, Any]) -> None:
    for key in ("eligible", "excluded", "fewer_than_two", "support_mass_sum"):
        value[key] += other[key]
    for key in ("support_mass_histogram", "support_size_histogram"):
        if len(value[key]) != len(other[key]):
            raise ValueError("Gate histogram contracts differ")
        for index, count in enumerate(other[key]):
            value[key][index] += count


def _gate_report(value: dict[str, Any]) -> dict[str, Any]:
    count = value["eligible"]
    return {
        **value,
        "recorded_action_exclusion_fraction": value["excluded"] / count if count else None,
        "fewer_than_two_fraction": value["fewer_than_two"] / count if count else None,
        "mean_support_mass": value["support_mass_sum"] / count if count else None,
        "support_mass_bin_width": 0.01,
        "support_mass_endpoint_one_bin": 100,
    }


def percentile(values: list[float], probability: float) -> float | None:
    if not 0 <= probability <= 1:
        raise ValueError("Invalid quantile probability")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


@dataclass
class GameClusters:
    totals: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def add(self, game: str, key: str, value: float, weight: float = 1.0) -> None:
        if not game or not math.isfinite(value) or not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid clustered observation")
        total = self.totals.setdefault(game, {}).setdefault(key, [0.0, 0.0])
        total[0] += value * weight
        total[1] += weight

    def merge(self, other: dict[str, dict[str, list[float]]]) -> None:
        for game, metrics in other.items():
            for key, (summed, weight) in metrics.items():
                total = self.totals.setdefault(game, {}).setdefault(key, [0.0, 0.0])
                total[0] += summed
                total[1] += weight

    def interval(self, key: str, *, seed: int = 17, replicates: int = 2000, confidence: float = 0.95) -> dict[str, Any]:
        if replicates < 1 or not 0 < confidence < 1:
            raise ValueError("Invalid cluster-bootstrap recipe")
        games = [value[key] for _, value in sorted(self.totals.items()) if key in value and value[key][1] > 0]
        total = math.fsum(value[0] for value in games)
        weight = math.fsum(value[1] for value in games)
        point = total / weight if weight else None
        if len(games) < 2:
            return {
                "mean": point,
                "lower": None,
                "upper": None,
                "independent_games": len(games),
                "status": "insufficient_independent_games",
            }
        rng = random.Random(seed)
        samples = []
        for _ in range(replicates):
            chosen = [games[rng.randrange(len(games))] for _ in games]
            denominator = math.fsum(value[1] for value in chosen)
            samples.append(math.fsum(value[0] for value in chosen) / denominator)
        tail = (1 - confidence) / 2
        return {
            "mean": point,
            "lower": percentile(samples, tail),
            "upper": percentile(samples, 1 - tail),
            "independent_games": len(games),
            "bootstrap_replicates": replicates,
            "seed": seed,
            "confidence": confidence,
            "resampling_unit": "whole_game_or_paired_seed_block",
            "status": "empirical_interval_not_error_certificate",
        }


@dataclass
class ReliabilityBins:
    bins: int = 10
    count: list[int] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.bins < 1:
            raise ValueError("Positive reliability bin count required")
        if not self.count:
            self.count = [0] * self.bins
            self.predicted = [0.0] * self.bins
            self.observed = [0.0] * self.bins

    def add(self, probability: float, outcome: float) -> None:
        if not math.isfinite(probability) or not 0 <= probability <= 1 or not 0 <= outcome <= 1:
            raise ValueError("Invalid reliability observation")
        index = min(self.bins - 1, int(probability * self.bins))
        self.count[index] += 1
        self.predicted[index] += probability
        self.observed[index] += outcome

    def merge(self, other: dict[str, Any]) -> None:
        if other["bins"] != self.bins:
            raise ValueError("Reliability bin contracts differ")
        for index in range(self.bins):
            self.count[index] += other["count"][index]
            self.predicted[index] += other["predicted"][index]
            self.observed[index] += other["observed"][index]

    def state_dict(self) -> dict[str, Any]:
        return {
            "bins": self.bins,
            "count": list(self.count),
            "predicted": list(self.predicted),
            "observed": list(self.observed),
        }

    def report(self) -> dict[str, Any]:
        total = sum(self.count)
        rows = []
        ece = 0.0
        for index, count in enumerate(self.count):
            predicted = self.predicted[index] / count if count else None
            observed = self.observed[index] / count if count else None
            if count:
                assert predicted is not None and observed is not None
                ece += count * abs(predicted - observed) / total
            rows.append(
                {
                    "lower": index / self.bins,
                    "upper": (index + 1) / self.bins,
                    "count": count,
                    "predicted": predicted,
                    "observed": observed,
                }
            )
        return {"count": total, "ece": ece if total else None, "bins": rows, "evidence_status": "empirical_only"}


class LatencyDistribution:
    """Mergeable bounded-memory latency distribution; quantiles are explicit bin bounds."""

    upper_bounds = tuple(1e-6 * 2**index for index in range(31))

    def __init__(self) -> None:
        self.counts = [0] * (len(self.upper_bounds) + 1)
        self.count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def add(self, seconds: float) -> None:
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Latency must be a measured finite nonnegative duration")
        import bisect  # noqa: PLC0415

        self.counts[bisect.bisect_left(self.upper_bounds, seconds)] += 1
        self.count += 1
        self.total += seconds
        self.minimum = seconds if self.minimum is None else min(self.minimum, seconds)
        self.maximum = seconds if self.maximum is None else max(self.maximum, seconds)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": "log_latency_bins_v1",
            "counts": list(self.counts),
            "count": self.count,
            "total_seconds": self.total,
            "minimum_seconds": self.minimum,
            "maximum_seconds": self.maximum,
        }

    def merge(self, state: dict[str, Any]) -> None:
        if (
            state["version"] != "log_latency_bins_v1"
            or len(state["counts"]) != len(self.counts)
            or sum(state["counts"]) != state["count"]
        ):
            raise ValueError("Latency distribution contract mismatch")
        for index, count in enumerate(state["counts"]):
            if type(count) is not int or count < 0:
                raise ValueError("Invalid latency bin count")
            self.counts[index] += count
        self.count += state["count"]
        self.total += state["total_seconds"]
        if state["count"]:
            low, high = state["minimum_seconds"], state["maximum_seconds"]
            if low is None or high is None or not 0 <= low <= high or not math.isfinite(high):
                raise ValueError("Invalid latency extrema")
            self.minimum = low if self.minimum is None else min(self.minimum, low)
            self.maximum = high if self.maximum is None else max(self.maximum, high)

    def quantile_bounds(self, probability: float) -> dict[str, float | None] | None:
        if not 0 < probability <= 1:
            raise ValueError("Invalid latency quantile")
        if not self.count:
            return None
        threshold, accumulated = math.ceil(probability * self.count), 0
        for index, count in enumerate(self.counts):
            accumulated += count
            if accumulated >= threshold:
                return {
                    "lower_seconds": self.upper_bounds[index - 1] if index else 0.0,
                    "upper_seconds": self.upper_bounds[index] if index < len(self.upper_bounds) else None,
                }
        raise ValueError("Latency count differs from its histogram")

    def report(self) -> dict[str, Any]:
        return {
            **self.state_dict(),
            "mean_seconds": self.total / self.count if self.count else None,
            "p50_seconds_bounds": self.quantile_bounds(0.5),
            "p95_seconds_bounds": self.quantile_bounds(0.95),
            "quantile_method": "bounded_histogram_ranges_not_exact_point_quantiles",
        }


class ClassificationMetrics:
    def __init__(self, classes: int, *, per_class_reliability: bool = False):
        self.classes = classes
        self.count = 0
        self.nll = 0.0
        self.brier = 0.0
        self.correct = 0
        self.support = [0] * classes
        self.hits = [0] * classes
        self.predicted_counts = [0] * classes
        self.probability_sums = [0.0] * classes
        self.nll_by_target = [0.0] * classes
        self.brier_by_target = [0.0] * classes
        self.confidence = ReliabilityBins()
        self.reliability = [ReliabilityBins() for _ in range(classes)] if per_class_reliability else []

    def add(
        self, probabilities: list[float], target: int, *, log_probability: float | None = None
    ) -> tuple[float, float]:
        if len(probabilities) != self.classes or type(target) is not int or not 0 <= target < self.classes:
            raise ValueError("Invalid conditional metric target")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities) or not math.isclose(
            sum(probabilities), 1.0, abs_tol=2e-6
        ):
            raise ValueError("Metrics require actual normalized probabilities")
        nll = -log_probability if log_probability is not None else -math.log(probabilities[target])
        if not math.isfinite(nll) or nll < -1e-6:
            raise ValueError("Use stable true-label log probabilities, not a clipped probability")
        brier = math.fsum((value - int(index == target)) ** 2 for index, value in enumerate(probabilities))
        prediction = max(range(self.classes), key=lambda index: probabilities[index])
        self.count += 1
        self.nll += nll
        self.brier += brier
        self.correct += prediction == target
        self.support[target] += 1
        self.nll_by_target[target] += nll
        self.brier_by_target[target] += brier
        self.hits[target] += prediction == target
        self.predicted_counts[prediction] += 1
        self.confidence.add(probabilities[prediction], float(prediction == target))
        for index, value in enumerate(probabilities):
            self.probability_sums[index] += value
            if self.reliability:
                self.reliability[index].add(value, float(target == index))
        return nll, brier

    def state_dict(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "count": self.count,
            "nll": self.nll,
            "brier": self.brier,
            "correct": self.correct,
            "support": list(self.support),
            "hits": list(self.hits),
            "predicted_counts": list(self.predicted_counts),
            "probability_sums": list(self.probability_sums),
            "nll_by_target": list(self.nll_by_target),
            "brier_by_target": list(self.brier_by_target),
            "confidence": self.confidence.state_dict(),
            "reliability": [value.state_dict() for value in self.reliability],
        }

    def merge(self, other: dict[str, Any]) -> None:
        if other["classes"] != self.classes or len(other["reliability"]) != len(self.reliability):
            raise ValueError("Metric class contracts differ")
        for key in ("count", "nll", "brier", "correct"):
            setattr(self, key, getattr(self, key) + other[key])
        for key in ("support", "hits", "predicted_counts", "probability_sums", "nll_by_target", "brier_by_target"):
            values = getattr(self, key)
            for index, value in enumerate(other[key]):
                values[index] += value
        self.confidence.merge(other["confidence"])
        for own, value in zip(self.reliability, other["reliability"]):
            own.merge(value)

    def report(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "nll": self.nll / self.count if self.count else None,
            "brier": self.brier / self.count if self.count else None,
            "top1": self.correct / self.count if self.count else None,
            "class_counts": list(self.support),
            "class_nll": [value / count if count else None for value, count in zip(self.nll_by_target, self.support)],
            "class_brier": [
                value / count if count else None for value, count in zip(self.brier_by_target, self.support)
            ],
            "class_recall": [hit / count if count else None for hit, count in zip(self.hits, self.support)],
            "mean_predicted_distribution": [
                value / self.count if self.count else None for value in self.probability_sums
            ],
            "actual_distribution": [value / self.count if self.count else None for value in self.support],
            "confidence_reliability": self.confidence.report(),
            "per_class_reliability": [value.report() for value in self.reliability],
        }


class FullDiagnostics:
    def __init__(self):
        self.heads = {
            "action": ClassificationMetrics(546),
            "response": ClassificationMetrics(21, per_class_reliability=True),
            "rank": ClassificationMetrics(4, per_class_reliability=True),
        }
        self.rank_by_event = {str(event): ClassificationMetrics(4, per_class_reliability=True) for event in range(21)}
        self.slices: dict[str, dict[str, ClassificationMetrics]] = {}
        self.clusters = GameClusters()
        self.utility_bins = [{"count": 0, "predicted": 0.0, "observed": 0.0} for _ in range(10)]
        self.gate = _new_gate()
        self.gate_slices: dict[str, dict[str, Any]] = {}
        self.unsupported_histogram = [0] * 101

    def add(
        self,
        *,
        game: str,
        head: str,
        probabilities: list[float],
        target: int,
        log_probability: float,
        slices: tuple[str, ...] = (),
        event: int | None = None,
    ) -> None:
        nll, brier = self.heads[head].add(probabilities, target, log_probability=log_probability)
        self.clusters.add(game, head + "_nll", nll)
        self.clusters.add(game, head + "_brier", brier)
        if head == "response":
            self.clusters.add(game, f"response_event_{target}_nll", nll)
            self.clusters.add(game, f"response_event_{target}_brier", brier)
        for name in slices:
            if name not in self.slices:
                self.slices[name] = {
                    key: ClassificationMetrics(classes)
                    for key, classes in (("action", 546), ("response", 21), ("rank", 4))
                }
            entry = self.slices[name]
            entry[head].add(probabilities, target, log_probability=log_probability)
        if head == "rank":
            if event is None or not 0 <= event < 21:
                raise ValueError("Observed rank diagnostics require their true response branch")
            self.rank_by_event[str(event)].add(probabilities, target, log_probability=log_probability)
            expected = math.fsum(value * utility for value, utility in zip(probabilities, UTILITIES))
            actual = UTILITIES[target]
            index = min(9, max(0, int((expected + 1.5) / 3 * 10)))
            row = self.utility_bins[index]
            row["count"] += 1
            row["predicted"] += expected
            row["observed"] += actual
            self.clusters.add(game, f"utility_error_bin_{index}", actual - expected)
            self.clusters.add(game, "utility_error", actual - expected)
            for key, value in (
                ("nll", nll),
                ("brier", brier),
                ("predicted_utility", expected),
                ("actual_utility", actual),
                ("utility_error", actual - expected),
            ):
                self.clusters.add(game, f"rank_event_{event}_{key}", value)

    def add_gate(self, *, retained: bool, support_size: int, mass: float, slices: tuple[str, ...] = ()) -> None:
        _add_gate(self.gate, retained=retained, support_size=support_size, mass=mass)
        for name in slices:
            if name not in self.gate_slices:
                self.gate_slices[name] = _new_gate()
            _add_gate(self.gate_slices[name], retained=retained, support_size=support_size, mass=mass)

    def add_unsupported(self, mass: float) -> None:
        if not math.isfinite(mass) or not 0 <= mass <= 1 + 1e-6:
            raise ValueError("Invalid post-mask unsupported mass")
        self.unsupported_histogram[min(100, int(mass * 100))] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "heads": {key: value.state_dict() for key, value in self.heads.items()},
            "rank_by_event": {key: value.state_dict() for key, value in self.rank_by_event.items()},
            "slices": {
                name: {key: value.state_dict() for key, value in heads.items()} for name, heads in self.slices.items()
            },
            "clusters": self.clusters.totals,
            "utility_bins": self.utility_bins,
            "gate": self.gate,
            "gate_slices": self.gate_slices,
            "unsupported_histogram": self.unsupported_histogram,
        }

    def merge(self, state: dict[str, Any]) -> None:
        for key, value in state["heads"].items():
            self.heads[key].merge(value)
        for key, value in state["rank_by_event"].items():
            self.rank_by_event[key].merge(value)
        for name, heads in state["slices"].items():
            if name not in self.slices:
                self.slices[name] = {
                    key: ClassificationMetrics(classes)
                    for key, classes in (("action", 546), ("response", 21), ("rank", 4))
                }
            entry = self.slices[name]
            for key, value in heads.items():
                entry[key].merge(value)
        self.clusters.merge(state["clusters"])
        for own, value in zip(self.utility_bins, state["utility_bins"]):
            for key in own:
                own[key] += value[key]
        _merge_gate(self.gate, state["gate"])
        for name, gate in state["gate_slices"].items():
            if name not in self.gate_slices:
                self.gate_slices[name] = _new_gate()
            _merge_gate(self.gate_slices[name], gate)
        for index, value in enumerate(state["unsupported_histogram"]):
            self.unsupported_histogram[index] += value

    def report(self, *, bootstrap_seed: int = 17, bootstrap_replicates: int = 2000) -> dict[str, Any]:
        calibration = []
        for index, row in enumerate(self.utility_bins):
            count = row["count"]
            calibration.append(
                {
                    "bin": index,
                    "count": count,
                    "predicted_mean": row["predicted"] / count if count else None,
                    "actual_mean": row["observed"] / count if count else None,
                    "cluster_error_interval": self.clusters.interval(
                        f"utility_error_bin_{index}", seed=bootstrap_seed, replicates=bootstrap_replicates
                    ),
                }
            )
        families: dict[str, list[int]] = {}
        for event in Response:
            name = event.name
            group = (
                "RON_MASK"
                if name.startswith("RON_MASK")
                else "CHI"
                if name.startswith("CHI")
                else "PON"
                if name.startswith("PON")
                else "DAIMINKAN"
                if name.startswith("DAIMINKAN")
                else "NEXT_DRAW"
                if name == "NEXT_DRAW"
                else "TERMINAL"
            )
            families.setdefault(group, []).append(int(event))
        response = self.heads["response"]
        grouped = {
            name: {
                "event_ids": ids,
                "count": sum(response.support[event] for event in ids),
                "recall": sum(response.hits[event] for event in ids) / sum(response.support[event] for event in ids)
                if sum(response.support[event] for event in ids)
                else None,
            }
            for name, ids in families.items()
        }
        result = {
            "statistics_version": STATISTICS_VERSION,
            "response_event_names": [event.name for event in Response],
            "response_groups": grouped,
            "heads": {key: value.report() for key, value in self.heads.items()},
            "rank_by_response": {key: value.report() for key, value in self.rank_by_event.items()},
            "by_slice": {
                name: {key: value.report() for key, value in heads.items()} for name, heads in self.slices.items()
            },
            "game_clustered_intervals": {
                head: self.clusters.interval(head + "_nll", seed=bootstrap_seed, replicates=bootstrap_replicates)
                for head in self.heads
            },
            "game_clustered_brier_intervals": {
                head: self.clusters.interval(head + "_brier", seed=bootstrap_seed, replicates=bootstrap_replicates)
                for head in self.heads
            },
            "response_class_intervals": {
                str(event): {
                    metric: self.clusters.interval(
                        f"response_event_{event}_{metric}", seed=bootstrap_seed, replicates=bootstrap_replicates
                    )
                    for metric in ("nll", "brier")
                }
                for event in range(21)
            },
            "rank_branch_intervals": {
                str(event): {
                    metric: self.clusters.interval(
                        f"rank_event_{event}_{metric}", seed=bootstrap_seed, replicates=bootstrap_replicates
                    )
                    for metric in ("nll", "brier", "predicted_utility", "actual_utility", "utility_error")
                }
                for event in range(21)
            },
            "branch_calibration_claimed": False,
            "utility_calibration": calibration,
            "epsilon_m_empirical_proxy": max(
                (abs(value["predicted_mean"] - value["actual_mean"]) for value in calibration if value["count"]),
                default=None,
            ),
            "gate": _gate_report(self.gate),
            "gate_by_slice": {name: _gate_report(value) for name, value in self.gate_slices.items()},
            "unsupported_mass_histogram": {"bin_width": 0.01, "counts": list(self.unsupported_histogram)},
            "evidence_status": "empirical_only_not_uniform_per_state_error_bounds",
        }
        return result
