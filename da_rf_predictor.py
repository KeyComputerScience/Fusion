"""Causal, delayed-label common-target predictors for DA-RF.

NumPy only. Stored feature vectors and forecasts are copied at issuance; an
arriving label never replaces a historical prediction with a refitted one.
This module predicts operational loss, not causal recovery of an update.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class RidgeConfig:
    ridge: float = 1e-3
    forgetting: float = 0.98
    max_pending: int = 4096

    def __post_init__(self):
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be finite and strictly positive")
        if not np.isfinite(self.forgetting) or not 0 < self.forgetting <= 1:
            raise ValueError("forgetting must lie in (0, 1]")
        if self.max_pending < 1:
            raise ValueError("max_pending must be positive")


class DelayedRidgeForecaster:
    """Exponentially weighted ridge branches with immutable issuance records.

    At clock k, an origin-r label contributes forgetting**(k-r) x_r x_r.T.
    Late labels may arrive in any origin order. Event/arrival time is monotone;
    adding an older record does not give it the weight of a new observation.
    Missing-source rows are skipped and are never imputed as target zero.
    """
    def __init__(self, n_sources: int, feature_dim: int,
                 config: RidgeConfig | None = None):
        if n_sources < 1 or feature_dim < 1:
            raise ValueError("positive source and feature dimensions required")
        self.n_sources, self.feature_dim = n_sources, feature_dim
        self.config = config or RidgeConfig()
        self._gram = np.zeros((n_sources, feature_dim, feature_dim))
        self._rhs = np.zeros((n_sources, feature_dim))
        self._clock: int | None = None
        self._last_issue: int | None = None
        self._pending: dict[int, dict] = {}
        self._observed: set[int] = set()
        self._counts = np.zeros(n_sources, dtype=int)
        self._prefix_fitted = False

    def _validate_features(self, features, available):
        x = np.asarray(features, dtype=float)
        m = np.asarray(available, dtype=bool)
        if x.shape != (self.n_sources, self.feature_dim):
            raise ValueError("features must have shape (n_sources, feature_dim)")
        if m.shape != (self.n_sources,):
            raise ValueError("available must have shape (n_sources,)")
        if not np.isfinite(x[m]).all():
            raise ValueError("available feature rows must be finite")
        return x.copy(), m.copy()

    def _advance(self, window: int):
        if not isinstance(window, (int, np.integer)) or window < 0:
            raise ValueError("window must be a nonnegative integer")
        if self._clock is not None and window < self._clock:
            raise ValueError("event time cannot move backwards")
        if self._clock is not None:
            decay = self.config.forgetting ** (window - self._clock)
            self._gram *= decay
            self._rhs *= decay
        self._clock = int(window)

    @property
    def coefficients(self):
        ridge = self.config.ridge * np.eye(self.feature_dim)
        return np.stack([np.linalg.solve(g + ridge, b)
                         for g, b in zip(self._gram, self._rhs)])

    def fit_prefix(self, features, targets, available=None):
        """Fit chronological historical cases; call once before live issuance.

        No clock or test labels are inferred from file order. The live caller
        chooses window identifiers starting at zero after this separate prefix.
        """
        if self._last_issue is not None or self._prefix_fitted:
            raise ValueError("prefix fitting is allowed once, before live issuance")
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 3 or x.shape[1:] != (self.n_sources, self.feature_dim):
            raise ValueError("prefix features must have shape (K, S, d)")
        if y.shape != (len(x),) or not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
            raise ValueError("prefix targets must be finite values in [0, 1]")
        mask = (np.ones(x.shape[:2], dtype=bool) if available is None
                else np.asarray(available, dtype=bool))
        if mask.shape != x.shape[:2] or not np.isfinite(x[mask]).all():
            raise ValueError("prefix masks or available features are invalid")
        for r, (xr, yr, mr) in enumerate(zip(x, y, mask)):
            weight = self.config.forgetting ** (len(x) - 1 - r)
            for s in np.flatnonzero(mr):
                self._gram[s] += weight * np.outer(xr[s], xr[s])
                self._rhs[s] += weight * xr[s] * yr
                self._counts[s] += 1
        self._prefix_fitted = True
        return self

    def issue(self, window: int, features, available, target_window=None):
        if self._last_issue is not None and window <= self._last_issue:
            raise ValueError("forecast issuance windows must strictly increase")
        if len(self._pending) >= self.config.max_pending:
            raise RuntimeError("pending archive full; resolve/discard labels explicitly")
        x, mask = self._validate_features(features, available)
        target_window = window + 1 if target_window is None else target_window
        if not isinstance(target_window, (int, np.integer)) or target_window <= window:
            raise ValueError("the target window must follow forecast issuance")
        self._advance(window)
        pred = np.full(self.n_sources, np.nan)
        pred[mask] = np.clip(np.sum(self.coefficients[mask] * x[mask], axis=1), 0, 1)
        self._pending[int(window)] = {
            "features": x, "available": mask, "prediction": pred.copy(),
            "target_window": int(target_window), "issued_window": int(window),
        }
        self._last_issue = int(window)
        return pred.copy()

    def observe_label(self, forecast_window: int, target: float, arrival_window: int):
        if forecast_window in self._observed:
            raise ValueError("duplicate label for an already scored forecast")
        if forecast_window not in self._pending:
            raise KeyError("unknown or explicitly discarded forecast identifier")
        record = self._pending[forecast_window]
        if arrival_window < record["target_window"]:
            raise ValueError("label arrived before its target window completed")
        if not np.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("target must be finite and lie in [0, 1]")
        self._advance(arrival_window)
        weight = self.config.forgetting ** (arrival_window - forecast_window)
        for s in np.flatnonzero(record["available"]):
            xr = record["features"][s]
            self._gram[s] += weight * np.outer(xr, xr)
            self._rhs[s] += weight * xr * target
            self._counts[s] += 1
        self._pending.pop(forecast_window)
        self._observed.add(forecast_window)
        error = record["prediction"] - target
        return {
            "forecast_window": int(forecast_window),
            "arrival_window": int(arrival_window), "origin_weight": float(weight),
            "stored_prediction": [float(v) if np.isfinite(v) else None
                                  for v in record["prediction"]],
            "residual": [float(v) if np.isfinite(v) else None for v in error],
            "available": record["available"].tolist(),
        }

    def discard_pending(self, forecast_window: int):
        """Explicitly declare a permanently unavailable label; never train on it."""
        if forecast_window not in self._pending:
            raise KeyError("unknown forecast identifier")
        self._pending.pop(forecast_window)
        return {"forecast_window": int(forecast_window), "label_unavailable": True}

    def pending_snapshot(self, forecast_window: int):
        r = self._pending[forecast_window]
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in r.items()}

    def state_dict(self):
        """Audit summary, not a resumable checkpoint."""
        return {"config": asdict(self.config), "clock": self._clock,
                "last_issue": self._last_issue,
                "observed_counts": self._counts.tolist(),
                "pending_ids": sorted(self._pending),
                "coefficients": self.coefficients.tolist()}
