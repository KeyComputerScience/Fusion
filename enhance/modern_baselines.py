"""Causal scalar forecast aggregation baselines, with no external dependencies.

All predictors consume the SAME next-target predictions supplied by an
EvidenceFusion snapshot. Outcomes update saved forecasts only after arrival.
EWA/BOA use squared forecast loss; mask-renormalisation is an operational
sleeping-expert adaptation, not a claim that the original regret proof applies.

DS implements the actual binary Dempster rule, including ignorance and conflict.
Its continuous forecasts are used as explicitly constructed evidence support,
not claimed to be calibrated event-class posteriors. Its nonlinear combination
has no linear source weights. See ../baseline_review.md for sources/limitations.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any


def _get(snapshot: Any, name: str, default=None):
    return snapshot.get(name, default) if isinstance(snapshot, dict) else getattr(snapshot, name, default)


def _bounded(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, result)) if math.isfinite(result) else None


def _normalise(values, masks):
    result = [float(v) if m else 0.0 for v, m in zip(values, masks)]
    total = sum(result)
    if total <= 0.0:
        count = sum(masks)
        return [float(m) / count if count else 0.0 for m in masks]
    return [v / total for v in result]


def dempster_binary(supports, discounts, masks=None, conflict_tolerance=1e-12):
    """Combine m(change)=r*p, m(stable)=r*(1-p), m(Theta)=1-r.

    Accumulate unnormalised conjunctive masses and normalise ONCE. ``conflict``
    is the full empty-set mass, not the largest intermediate pairwise conflict.
    At total conflict the rule is undefined; return probability=None to abstain.
    """
    if len(supports) != len(discounts):
        raise ValueError("supports and discounts must have equal length")
    masks = [True] * len(supports) if masks is None else list(masks)
    if len(masks) != len(supports):
        raise ValueError("masks must have the same length as supports")
    change, stable, ignorance, conflict = 0.0, 0.0, 1.0, 0.0
    count = 0
    for p, r, present in zip(supports, discounts, masks):
        if not present:
            continue
        p, r = _bounded(p), _bounded(r)
        if p is None or r is None:
            continue
        c, s, u = r * p, r * (1.0 - p), 1.0 - r
        new_conflict = conflict + change * s + stable * c
        change, stable, ignorance = (
            change * (c + u) + ignorance * c,
            stable * (s + u) + ignorance * s,
            ignorance * u,
        )
        conflict = new_conflict
        count += 1
    denominator = change + stable + ignorance
    if count == 0 or denominator <= conflict_tolerance:
        return dict(probability=None, mass_change=None, mass_stable=None,
                    mass_ignorance=None, conflict=min(1.0, conflict), fallback=True)
    change, stable, ignorance = (v / denominator for v in (change, stable, ignorance))
    return dict(probability=change + 0.5 * ignorance, mass_change=change,
                mass_stable=stable, mass_ignorance=ignorance,
                conflict=min(1.0, conflict), fallback=False)


@dataclass
class BaselineForecast:
    method: str
    prediction: float | None
    raw_score: float | None
    source_predictions: list[float | None]
    raw_scores: list[float | None]
    masks: list[bool]
    weights: list[float] | None
    predictive_mse: list[float]
    discounts: list[float] | None = None
    masses: dict | None = None
    raw_masses: dict | None = None
    window_index: int = 0
    completed_slot: int = 0

    def to_dict(self):
        return asdict(self)


class FusionBaseline:
    """Standalone interface: predict(snapshot), then observe(saved_forecast, y).

    Missing target -> no updates. EF/FF predict the same calibrated target as
    IMSE/EWA/BOA. ``raw_score`` is separate current-drift diagnostic only.
    """
    ALIASES = {
        "equal": "EF", "ef": "EF", "fixed": "FF", "ff": "FF",
        "imse": "IMSE", "precision": "IMSE", "calibrated_precision": "IMSE",
        "ewa": "EWA", "hedge": "EWA", "boa": "BOA",
        "ds": "DS", "dst": "DS", "dsf": "DS", "dempster": "DS",
    }

    def __init__(self, method, config=None):
        self.method = self.ALIASES.get(str(method).lower(), str(method).upper())
        if self.method not in set(self.ALIASES.values()):
            raise ValueError(f"Unknown baseline: {method}")
        self.config = dict(config or {})
        self.eta = float(self.config.get("learning_rate", 0.5))
        self.error_rate = float(self.config.get("error_rate", 0.15))
        self.epsilon_mse = float(self.config.get("epsilon_mse", 0.0025))
        self.ds_scale = float(self.config.get("ds_consistency_scale", 0.20))
        if not 0.0 < self.eta <= 0.5:
            raise ValueError("learning_rate must be in (0, 0.5] for bounded-loss BOA")
        if not 0.0 < self.error_rate <= 1.0 or self.epsilon_mse <= 0.0 or self.ds_scale <= 0.0:
            raise ValueError("invalid error_rate, epsilon_mse, or ds_consistency_scale")
        self.mse = None
        self.log_weights = None
        self.fixed_weights = self.config.get("fixed_weights")
        self.update_count = 0

    def _initialise(self, snapshot, size):
        if self.mse is not None:
            if len(self.mse) != size:
                raise ValueError("Source count cannot change; use masks for missing sources")
            return
        initial = _get(snapshot, "predictive_errors", [0.05] * size)
        self.mse = [max(float(v), 0.0) if v is not None and math.isfinite(float(v)) else 0.05
                    for v in initial]
        if len(self.mse) != size:
            raise ValueError("predictive_errors has wrong source count")
        self.log_weights = [0.0] * size
        if self.fixed_weights is None:
            self.fixed_weights = [1.0 / size] * size
        else:
            self.fixed_weights = [float(v) for v in self.fixed_weights]
        if len(self.fixed_weights) != size or any(v < 0.0 or not math.isfinite(v) for v in self.fixed_weights):
            raise ValueError("fixed_weights must be finite, nonnegative, and match source count")

    def predict(self, snapshot):
        raw = list(_get(snapshot, "scores", []))
        predictions = list(_get(snapshot, "next_target_predictions", [None] * len(raw)))
        masks = list(_get(snapshot, "masks", [False] * len(raw)))
        if not raw or len(raw) != len(predictions) or len(raw) != len(masks):
            raise ValueError("scores, masks, and next_target_predictions must have equal nonzero length")
        raw, predictions = ([_bounded(v) for v in values] for values in (raw, predictions))
        masks = [bool(m) and p is not None for m, p in zip(masks, predictions)]
        self._initialise(snapshot, len(raw))
        weights, discounts, masses, raw_masses = None, None, None, None
        if self.method == "DS":
            discounts = [math.exp(-value / self.ds_scale) for value in self.mse]
            masses = dempster_binary(predictions, discounts, masks)
            raw_masses = dempster_binary(raw, discounts, masks)
            prediction, raw_score = masses["probability"], raw_masses["probability"]
        else:
            if self.method == "EF":
                values = [1.0] * len(raw)
            elif self.method == "FF":
                values = self.fixed_weights
            elif self.method == "IMSE":
                values = [1.0 / (v + self.epsilon_mse) for v in self.mse]
            else:
                maximum = max((v for v, m in zip(self.log_weights, masks) if m), default=0.0)
                values = [math.exp(v - maximum) if m else 0.0 for v, m in zip(self.log_weights, masks)]
            weights = _normalise(values, masks)
            prediction = sum(w * p for w, p in zip(weights, predictions) if p is not None) if any(masks) else None
            raw_score = sum(w * p for w, p in zip(weights, raw) if p is not None) if any(masks) else None
        return BaselineForecast(self.method, prediction, raw_score, predictions, raw, masks,
                                weights, list(self.mse), discounts, masses, raw_masses,
                                int(_get(snapshot, "window_index", 0)),
                                int(_get(snapshot, "completed_slot", 0)))

    def observe(self, forecast, target):
        """Score a SAVED, pre-outcome forecast; never recompute it after fitting.

        For BOA use centred expert losses, as in Figure 1 of Wintenberger.
        Sleeping experts receive no loss update; returning experts retain state.
        """
        target = _bounded(target)
        if target is None or forecast.prediction is None or not any(forecast.masks):
            return False
        losses = [(p - target) ** 2 if m and p is not None else None
                  for p, m in zip(forecast.source_predictions, forecast.masks)]
        for index, loss in enumerate(losses):
            if loss is not None:
                self.mse[index] = (1.0 - self.error_rate) * self.mse[index] + self.error_rate * loss
        if self.method in ("EWA", "BOA"):
            average_expert_loss = sum(w * loss for w, loss in zip(forecast.weights, losses) if loss is not None)
            for index, loss in enumerate(losses):
                if loss is None:
                    continue
                if self.method == "EWA":
                    self.log_weights[index] -= self.eta * loss
                else:
                    excess = loss - average_expert_loss
                    self.log_weights[index] -= self.eta * excess + self.eta ** 2 * excess ** 2
            # A common shift preserves probabilities and prevents magnitude growth.
            shift = max(self.log_weights)
            self.log_weights = [v - shift for v in self.log_weights]
        self.update_count += 1
        return True


class ReplaySnapshot:
    """FusionSnapshot-compatible object with explicit nonlinear-DS metadata."""
    def __init__(self, base, forecast):
        self.__dict__.update(copy.deepcopy(vars(base)))
        self.method = forecast.method
        self.weights = forecast.weights
        self.gamma = forecast.raw_score if forecast.raw_score is not None else 0.0
        self.forecast_score = forecast.prediction
        self.forecast_probability = forecast.prediction
        self.next_target_predictions = list(forecast.source_predictions)
        self.masks = list(forecast.masks)
        self.coverage = sum(self.masks) / len(self.masks)
        self.predictive_errors = list(forecast.predictive_mse)
        self.reliability = [1.0 / (v + 0.0025) if m else 0.0
                            for v, m in zip(forecast.predictive_mse, self.masks)]
        self.nonlinear_combination = forecast.weights is None
        self.raw_score_calibrated = False
        self.dempster_masses = forecast.masses
        self.dempster_raw_masses = forecast.raw_masses
        self.fallback = forecast.prediction is None
        self.fallback_reason = "no_valid_forecast_or_total_conflict" if self.fallback else ""
        if forecast.weights is not None and forecast.prediction is not None:
            self.disagreement = sum(w * (p - forecast.prediction) ** 2
                                    for w, p in zip(forecast.weights, forecast.source_predictions) if p is not None)
        else:
            self.disagreement = 0.0
        # Only a diagnostic placeholder. Root supplies the unified uncertainty wrapper.
        self.uncertainty = 1.0 if self.fallback else min(1.0, 4.0 * self.disagreement + 0.5 * (1.0 - self.coverage))
        self.weighting_mode = forecast.method

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        result["source_names"] = ["workload", "operating", "performance"]
        result["available_from_slot"] = self.completed_slot + 1
        result["maximum_weight"] = max(self.weights) if self.weights is not None else None
        return result


class ReplayBaseline:
    """Drop-in replay adapter: ReplayBaseline(mode, fusion_config, reference).

    The core extractor is shared. Every new visible service target updates the
    previous forecast before the current predictions are made. Full outages
    leave aggregation skill states unchanged. Use standalone FusionBaseline
    if a separate clean/evaluation target ledger is required.
    """
    def __init__(self, mode, config, reference):
        try:
            from core.fusion import EvidenceFusion
        except ImportError:
            from fusion import EvidenceFusion
        fusion_config = copy.deepcopy(config.get("fusion", config))
        self.extractor = EvidenceFusion(fusion_config, reference)
        settings = copy.deepcopy(fusion_config.get("modern_baselines", {}))
        if "fixed_weights" not in settings and "fixed_weights" in fusion_config:
            settings["fixed_weights"] = fusion_config["fixed_weights"]
        self.baseline = FusionBaseline(mode, settings)
        self.previous_forecast = None

    def complete_window(self, rows, window_index):
        base = self.extractor.complete_window(rows, window_index)
        if self.previous_forecast is not None and self.previous_forecast.window_index + 1 == window_index:
            self.baseline.observe(self.previous_forecast, base.scores[2])
        forecast = self.baseline.predict(base)
        self.previous_forecast = forecast
        return ReplaySnapshot(base, forecast)
