"""Causal, reliability-aware fusion over two completed windows.

All code is a reference implementation of the reconstructed methods, not a
reproduction of a measured DRL result. The core uses only the Python standard
library. Source order is workload, operating state, external performance.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

SOURCE_NAMES = ("workload", "operating", "performance")


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalized_jsd(first: list[str], second: list[str]) -> float:
    """Normalized Jensen--Shannon divergence: 0 for equality, 1 if disjoint."""
    if not first or not second:
        raise ValueError("JSD requires two nonempty samples")
    a, b = Counter(first), Counter(second)
    divergence = 0.0
    for key in sorted(a.keys() | b.keys()):
        p, q = a[key] / len(first), b[key] / len(second)
        mixture = (p + q) / 2.0
        if p:
            divergence += 0.5 * p * math.log(p / mixture)
        if q:
            divergence += 0.5 * q * math.log(q / mixture)
    return clip(divergence / math.log(2.0))


def moving_block_sample(rows: list[dict], block_size: int, rng: random.Random) -> list[dict]:
    """Circular moving-block bootstrap, retaining within-block temporal order."""
    result = []
    while len(result) < len(rows):
        start = rng.randrange(len(rows))
        result.extend(rows[(start + j) % len(rows)] for j in range(min(block_size, len(rows))))
    return result[:len(rows)]


@dataclass
class PredictiveCalibration:
    """Affine predictor fitted only when the next window's target is observed."""
    ridge: float = 0.1
    forgetting: float = 0.99
    error_rate: float = 0.15
    error: float = 0.05
    a00: float = field(init=False)
    a01: float = 0.0
    a11: float = field(init=False)
    b0: float = 0.0
    b1: float = field(init=False)

    def __post_init__(self) -> None:
        self.a00 = self.a11 = self.ridge
        self.b1 = self.ridge  # prior prediction = source score

    def coefficients(self) -> tuple[float, float]:
        determinant = self.a00 * self.a11 - self.a01 * self.a01
        if determinant <= 1e-15:
            return 0.0, 1.0
        return ((self.b0 * self.a11 - self.b1 * self.a01) / determinant,
                (self.b1 * self.a00 - self.b0 * self.a01) / determinant)

    def predict(self, score: float) -> float:
        intercept, slope = self.coefficients()
        return clip(intercept + slope * score)

    def update(self, previous_score: float, stored_prediction: float, observed_target: float) -> None:
        self.error = ((1.0 - self.error_rate) * self.error
                      + self.error_rate * (stored_prediction - observed_target) ** 2)
        self.a00 = self.forgetting * self.a00 + 1.0
        self.a01 = self.forgetting * self.a01 + previous_score
        self.a11 = self.forgetting * self.a11 + previous_score ** 2
        self.b0 = self.forgetting * self.b0 + observed_target
        self.b1 = self.forgetting * self.b1 + previous_score * observed_target


@dataclass
class FusionSnapshot:
    completed_slot: int = 0
    window_index: int = 0
    scores: list[float | None] = field(default_factory=lambda: [None] * 3)
    masks: list[bool] = field(default_factory=lambda: [False] * 3)
    weights: list[float] = field(default_factory=lambda: [0.0] * 3)
    reliability: list[float] = field(default_factory=lambda: [0.0] * 3)
    effective_samples: list[int] = field(default_factory=lambda: [0] * 3)
    ages: list[int | None] = field(default_factory=lambda: [None] * 3)
    noise_variances: list[float | None] = field(default_factory=lambda: [None] * 3)
    predictive_errors: list[float] = field(default_factory=lambda: [0.05] * 3)
    next_target_predictions: list[float | None] = field(default_factory=lambda: [None] * 3)
    gamma: float = 0.0
    disagreement: float = 0.0
    uncertainty: float = 1.0
    coverage: float = 0.0
    fallback: bool = True
    fallback_reason: str = "no_completed_pair"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["source_names"] = list(SOURCE_NAMES)
        result["available_from_slot"] = self.completed_slot + 1
        return result


class EvidenceFusion:
    def __init__(self, config: dict, reference: dict) -> None:
        self.config = config
        self.reference = reference
        calibration = config["predictive_calibration"]
        self.predictors = [PredictiveCalibration(**calibration) for _ in SOURCE_NAMES]
        self.previous_rows: list[dict] | None = None
        self.snapshot = FusionSnapshot(completed_slot=reference["fit_through_slot"])
        self.last_valid_gamma = 0.0

    def _valid(self, source: int, row: dict) -> bool:
        if source == 0:
            return row.get("regime") not in (None, "")
        if source == 1:
            return all(finite(row.get(name)) for name in self.reference["state_columns"])
        return (finite(row.get("environment_reward")) and finite(row.get("observed_training_cost"))
                and all(row.get(name) not in (None, "") for name in
                        ("logged_training", "logged_inference", "policy_version")))

    def _performance(self, previous: list[dict], current: list[dict]) -> tuple[float | None, int, int | None]:
        def groups(rows: list[dict]) -> dict:
            grouped = defaultdict(list)
            for row in rows:
                if self._valid(2, row):
                    context = tuple(str(row[name]) for name in
                                    ("logged_training", "logged_inference", "policy_version"))
                    adjusted = (row["environment_reward"]
                                + self.config["observed_cost_reward_scale"] * row["observed_training_cost"])
                    grouped[context].append((adjusted, row["slot"]))
            return grouped

        a, b = groups(previous), groups(current)
        n_total, before, after, latest = 0, 0.0, 0.0, None
        for context in sorted(a.keys() & b.keys()):
            n = min(len(a[context]), len(b[context]))
            if n < self.config["minimum_context_samples"]:
                continue
            n_total += n
            before += n * statistics.fmean(value for value, _ in a[context])
            after += n * statistics.fmean(value for value, _ in b[context])
            context_latest = max(slot for _, slot in b[context])
            latest = context_latest if latest is None else max(latest, context_latest)
        if n_total < self.config["minimum_source_samples"]:
            return None, n_total, latest
        before, after = before / n_total, after / n_total
        score = clip(max(0.0, before - after) / (abs(before) + self.config["epsilon_reward"]))
        return score, n_total, latest

    def _evidence(self, source: int, previous: list[dict], current: list[dict]) -> tuple[float | None, int, int | None]:
        if source == 2:
            return self._performance(previous, current)
        a = [row for row in previous if self._valid(source, row)]
        b = [row for row in current if self._valid(source, row)]
        n = min(len(a), len(b))
        latest = max((row["slot"] for row in b), default=None)
        if n < self.config["minimum_source_samples"]:
            return None, n, latest
        if source == 0:
            return normalized_jsd([str(row["regime"]) for row in a],
                                  [str(row["regime"]) for row in b]), n, latest
        components = []
        for name in self.reference["state_columns"]:
            scale = self.reference["state_scales"][name]
            before = statistics.fmean((row[name] - scale["mean"]) / scale["std"] for row in a)
            after = statistics.fmean((row[name] - scale["mean"]) / scale["std"] for row in b)
            components.append(min(1.0, abs(after - before)))
        return statistics.fmean(components), n, latest

    def complete_window(self, current: list[dict], window_index: int) -> FusionSnapshot:
        if not current or len(current) != self.config["window_size"]:
            raise ValueError("Only a complete, nonempty window can update fusion")
        if any(b["slot"] != a["slot"] + 1 for a, b in zip(current, current[1:])):
            raise ValueError("Window rows must be contiguous and ordered")
        end = current[-1]["slot"]
        if end <= self.snapshot.completed_slot:
            raise ValueError("Completed windows must arrive in increasing order")
        if self.previous_rows is None:
            self.previous_rows = list(current)
            self.snapshot = FusionSnapshot(completed_slot=end, window_index=window_index)
            return self.snapshot

        previous = self.previous_rows
        evidence = [self._evidence(s, previous, current) for s in range(3)]
        scores = [item[0] for item in evidence]
        masks = [score is not None for score in scores]

        # Previous predictions are scored only against this newly observed target.
        old = self.snapshot
        target = scores[2]
        if target is not None:
            for source, predictor in enumerate(self.predictors):
                if old.masks[source] and old.next_target_predictions[source] is not None:
                    predictor.update(old.scores[source], old.next_target_predictions[source], target)

        bootstrap_scores: list[list[float]] = [[], [], []]
        rng = random.Random(self.config["seed"] + 104729 * window_index)
        for _ in range(self.config["bootstrap_replicates"]):
            boot_a = moving_block_sample(previous, self.config["bootstrap_block_size"], rng)
            boot_b = moving_block_sample(current, self.config["bootstrap_block_size"], rng)
            for source in range(3):
                if masks[source]:
                    score, _, _ = self._evidence(source, boot_a, boot_b)
                    if score is not None:
                        bootstrap_scores[source].append(score)

        noise: list[float | None] = []
        reliabilities = []
        ages = []
        for source in range(3):
            values = bootstrap_scores[source]
            # Unsupported variance is conservative, never interpreted as zero noise.
            variance = statistics.pvariance(values) if len(values) >= 2 else None
            noise.append(variance)
            score, n, latest = evidence[source]
            age = end - latest if latest is not None else None
            ages.append(age)
            if score is None:
                reliabilities.append(0.0)
                continue
            if variance is None:
                variance = self.config["unsupported_noise_variance"]
            reliability = (n / (n + self.config["support_scale"])
                           * math.exp(-age / self.config["freshness_scale"])
                           * math.exp(-self.predictors[source].error / self.config["error_scale"])
                           / (variance + self.config["epsilon_noise"]))
            reliabilities.append(reliability)

        denominator = sum(reliabilities)
        coverage = sum(masks) / 3.0
        if denominator > 0.0:
            weights = [r / denominator for r in reliabilities]
            gamma = sum(w * score for w, score in zip(weights, scores) if score is not None)
            disagreement = sum(w * (score - gamma) ** 2 for w, score in zip(weights, scores)
                               if score is not None)
            uncertainty = clip(4.0 * disagreement + self.config["missing_source_penalty"] * (1.0 - coverage))
            fallback, reason = False, ""
            self.last_valid_gamma = gamma
        else:
            weights = [0.0] * 3
            gamma, disagreement, uncertainty = self.last_valid_gamma, 0.0, 1.0
            fallback, reason = True, "no_valid_source"

        self.snapshot = FusionSnapshot(
            completed_slot=end, window_index=window_index, scores=scores, masks=masks,
            weights=weights, reliability=reliabilities,
            effective_samples=[item[1] for item in evidence], ages=ages,
            noise_variances=noise, predictive_errors=[p.error for p in self.predictors],
            next_target_predictions=[self.predictors[s].predict(scores[s]) if masks[s] else None
                                     for s in range(3)],
            gamma=gamma, disagreement=disagreement, uncertainty=uncertainty,
            coverage=coverage, fallback=fallback, fallback_reason=reason)
        self.previous_rows = list(current)
        return self.snapshot
