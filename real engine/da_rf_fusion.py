"""Causal decision-aware reliability fusion, using only NumPy and the standard library.

The matrix learned here is a rolling, mask-conditioned *archive* moment.  Its
transfer to a future context distribution is an empirical question.  The buffer
is an empirical regularizer, not a statistical confidence certificate.  No
method in this module claims that a service policy has actually recovered.

Forecast window k predicts the target for window k + 1.  Labels are scored
against the originally saved forecasts, including their issuance-time masks.
Missing residuals remain NaN in storage and never enter a matrix outer product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence
import copy
import math

import numpy as np


_MODES = frozenset({
    "decision_full", "decision_diagonal", "unweighted_full",
    "unweighted_diagonal", "trace_matched", "no_risk", "original_c",
})


@dataclass(frozen=True)
class FusionConfig:
    tau: float = 0.1
    lam: float = 1.0
    inertia: float = 0.05
    max_weight: float = 0.6
    quality_min: float = 1e-6
    support_scale: float = 20.0
    freshness_tau: float = 4.0
    variance_epsilon: float = 1e-6
    variance_fallback: float = 1.0
    min_count: float = 1.0
    max_observation_age: float = 8.0
    decay: float = 0.95
    context_bandwidth: float = 1.0
    prior_weight: float = 1.0
    prior_risk: float = 0.05
    prior_unweighted_risk: float = 0.05
    archive_length: int = 512
    exact_mask: bool = False
    require_complete_enabled_vectors: bool = False
    max_pending_forecasts: int = 4096
    delta_base: float = 0.01
    delta_prior: float = 0.1
    delta_count: float = 0.05
    delta_age: float = 0.01
    max_risk_age: float = 8.0
    disagreement_weight: float = 0.25
    missing_weight: float = 0.25
    risk_weight: float = 0.25
    age_weight: float = 0.25
    risk_reward_scale: float = 1.0
    solver_tolerance: float = 1e-8
    solver_max_iterations: int = 100
    force_solver_fallback: bool = False
    moment_mode: str = "decision_full"
    mse_temperature: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            "tau", "quality_min", "freshness_tau", "variance_epsilon",
            "variance_fallback", "context_bandwidth", "prior_weight",
            "max_risk_age", "risk_reward_scale", "solver_tolerance",
            "mse_temperature",
        )
        nonnegative = (
            "lam", "inertia", "support_scale", "min_count",
            "max_observation_age", "prior_risk", "prior_unweighted_risk",
            "delta_base", "delta_prior", "delta_count", "delta_age",
            "disagreement_weight", "missing_weight", "risk_weight", "age_weight",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not (math.isfinite(self.max_weight) and 0 < self.max_weight <= 1):
            raise ValueError("max_weight must belong to (0, 1]")
        if not (math.isfinite(self.decay) and 0 < self.decay <= 1):
            raise ValueError("decay must belong to (0, 1]")
        if not isinstance(self.archive_length, int) or self.archive_length < 1:
            raise ValueError("archive_length must be a positive integer")
        if not isinstance(self.max_pending_forecasts, int) or self.max_pending_forecasts < 1:
            raise ValueError("max_pending_forecasts must be a positive integer")
        if not isinstance(self.solver_max_iterations, int) or self.solver_max_iterations < 0:
            raise ValueError("solver_max_iterations must be a nonnegative integer")
        if self.moment_mode not in _MODES:
            raise ValueError(f"moment_mode must be one of {sorted(_MODES)}")


@dataclass(frozen=True)
class QualityObservation:
    valid: bool
    count: float
    age: float
    variance: float | None
    diagnostic: float | None = None


@dataclass(frozen=True)
class FusionSnapshot:
    window: int
    weights: np.ndarray
    forecast: float | None
    uncertainty: float
    coverage: float
    diagnostic: float | None = None
    risk: float = 0.0
    solver_fallback: bool = False
    all_missing: bool = False
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverResult:
    weights: np.ndarray
    objective: float
    primal_residual: float
    kkt_residual: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class _ForecastRecord:
    window: int
    predictions: np.ndarray
    mask: np.ndarray
    context: np.ndarray
    chi: float


@dataclass(frozen=True)
class _ResidualRecord:
    forecast_window: int
    arrival_window: int
    target: float
    mask: np.ndarray
    residuals: np.ndarray
    context: np.ndarray
    chi: float


def _readonly(values: Any, dtype: Any = float) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _vector(values: Any, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or (length is not None and array.size != length):
        raise ValueError(f"{name} must be a one-dimensional vector" +
                         (f" of length {length}" if length is not None else ""))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.array(array, copy=True)


def _psd_matrix(values: Any, length: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (length, length) or not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must be a finite square matrix matching q")
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError("matrix must be symmetric")
    matrix = (matrix + matrix.T) / 2
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-10 * scale:
        raise ValueError("matrix must be positive semidefinite")
    return matrix


def _normalized_positive(values: np.ndarray) -> np.ndarray:
    if values.size == 0 or np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("quality must be nonempty, positive, and finite")
    logs = np.log(values)
    rates = np.exp(np.maximum(logs - np.max(logs), -690.0))
    return rates / np.sum(rates)


def _capped_quality(pi: np.ndarray, cap: float) -> np.ndarray:
    """KL projection: weights min(cap, t*pi), with exact mass redistribution."""
    count = pi.size
    if abs(cap * count - 1.0) <= 1e-12:
        return np.full(count, 1.0 / count)
    weights = np.zeros(count)
    free = np.ones(count, dtype=bool)
    remaining = 1.0
    for _ in range(count + 1):
        indices = np.flatnonzero(free)
        if indices.size == 0:
            break
        proposed = remaining * pi[indices] / np.sum(pi[indices])
        hit = proposed > cap
        if not np.any(hit):
            weights[indices] = proposed
            break
        fixed = indices[hit]
        weights[fixed] = cap
        free[fixed] = False
        remaining = 1.0 - float(np.sum(weights[~free]))
    return weights


def _objective(w: np.ndarray, pi: np.ndarray, matrix: np.ndarray,
               anchor: np.ndarray, tau: float, lam: float, inertia: float) -> float:
    positive = w > 0
    entropy = float(np.sum(w[positive] * (np.log(w[positive]) - np.log(pi[positive]))))
    return float(tau * entropy + lam / 2 * (w @ matrix @ w) +
                 inertia / 2 * np.sum((w - anchor) ** 2))


def _gradient(w: np.ndarray, pi: np.ndarray, matrix: np.ndarray,
              anchor: np.ndarray, tau: float, lam: float, inertia: float) -> np.ndarray:
    return tau * (np.log(w) - np.log(pi) + 1) + lam * (matrix @ w) + inertia * (w - anchor)


def _certificate(w: np.ndarray, pi: np.ndarray, matrix: np.ndarray,
                 anchor: np.ndarray, cap: float, tau: float, lam: float,
                 inertia: float, tolerance: float) -> tuple[float, float]:
    primal = max(abs(float(np.sum(w)) - 1.0), float(np.max(np.maximum(-w, 0))),
                 float(np.max(np.maximum(w - cap, 0))))
    if np.any(w <= 0):
        return primal, math.inf
    gradient = _gradient(w, pi, matrix, anchor, tau, lam, inertia)
    upper = w >= cap - max(5 * tolerance, 1e-12)
    free = ~upper
    if np.any(free):
        # This minimizes the free-coordinate infinity-norm stationarity residual.
        nu = -float((np.max(gradient[free]) + np.min(gradient[free])) / 2)
    else:
        nu = -float(np.max(gradient))
    multiplier = np.zeros_like(w)
    multiplier[upper] = np.maximum(-(gradient[upper] + nu), 0)
    stationarity = float(np.max(np.abs(gradient + nu + multiplier)))
    complementarity = float(np.max(np.abs(multiplier * (cap - w))))
    return primal, max(stationarity, complementarity)


def _free_newton(pi: np.ndarray, matrix: np.ndarray, anchor: np.ndarray,
                 cap: float, upper: np.ndarray, tau: float, lam: float,
                 inertia: float, tolerance: float,
                 max_iterations: int) -> tuple[np.ndarray | None, int]:
    free = np.flatnonzero(~upper)
    remaining = 1.0 - float(np.sum(upper)) * cap
    if free.size == 0 or remaining <= 0 or remaining >= free.size * cap + 1e-10:
        return None, 0
    weights = np.zeros(pi.size)
    weights[upper] = cap
    weights[free] = remaining / free.size
    ones = np.ones(free.size)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        gradient = _gradient(weights, pi, matrix, anchor, tau, lam, inertia)[free]
        if float(np.max(gradient) - np.min(gradient)) <= tolerance:
            return weights, iterations
        hessian = np.diag(tau / weights[free] + inertia) + lam * matrix[np.ix_(free, free)]
        try:
            inv_gradient = np.linalg.solve(hessian, gradient)
            inv_ones = np.linalg.solve(hessian, ones)
        except np.linalg.LinAlgError:
            return None, iterations
        direction = -inv_gradient + inv_ones * (np.sum(inv_gradient) / np.sum(inv_ones))
        direction[-1] -= np.sum(direction)  # preserve the equality to roundoff
        if not np.all(np.isfinite(direction)):
            return None, iterations
        descent = float(gradient @ direction)
        if descent >= 0:
            return None, iterations
        negative = direction < 0
        step = 1.0
        if np.any(negative):
            step = min(step, float(np.min(-0.99 * weights[free][negative] / direction[negative])))
        current_value = _objective(weights, pi, matrix, anchor, tau, lam, inertia)
        accepted = False
        for _ in range(60):
            candidate = weights.copy()
            candidate[free] += step * direction
            if (np.all(candidate[free] > 0) and
                    _objective(candidate, pi, matrix, anchor, tau, lam, inertia)
                    <= current_value + 1e-4 * step * descent + 1e-15 * max(1, abs(current_value))):
                weights = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return None, iterations
    gradient = _gradient(weights, pi, matrix, anchor, tau, lam, inertia)[free]
    if float(np.max(gradient) - np.min(gradient)) <= 5 * tolerance:
        return weights, iterations
    return None, iterations


def solve_capped_fusion(q: Any, matrix: Any, anchor: Any, cap: float,
                        tau: float, lam: float, inertia: float, *,
                        tolerance: float = 1e-8, max_iterations: int = 100,
                        max_enumerated_sources: int = 8) -> SolverResult:
    """Solve capped KL + joint quadratic risk + history inertia.

    Up to eight sources use cap-set enumeration and equality-constrained Newton.
    Larger problems use composite entropic mirror steps and the same explicit KKT
    check.  A failed solve returns the capped-quality fallback with converged=False;
    its nonzero KKT residual is retained.  No SciPy dependency is required.
    """
    quality = _vector(q, "q")
    pi = _normalized_positive(quality)
    size = quality.size
    history = _vector(anchor, "anchor", size)
    risk = _psd_matrix(matrix, size)
    for value, name in ((tau, "tau"), (tolerance, "tolerance")):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for value, name in ((lam, "lam"), (inertia, "inertia")):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if not math.isfinite(cap) or cap <= 0 or cap > 1 or cap * size < 1 - 1e-12:
        raise ValueError("cap must be in [1/n, 1]")
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    fallback = _capped_quality(pi, cap)

    def package(weights: np.ndarray, converged: bool, iterations: int) -> SolverResult:
        primal, kkt = _certificate(weights, pi, risk, history, cap, tau, lam, inertia, tolerance)
        return SolverResult(_readonly(weights),
                            _objective(weights, pi, risk, history, tau, lam, inertia),
                            primal, kkt, converged, iterations)

    if size == 1 or abs(size * cap - 1) <= 1e-12 or (lam == 0 and inertia == 0):
        return package(fallback, True, 0)
    total_iterations = 0
    if size <= max_enumerated_sources:
        for upper_count in range(size):
            if upper_count * cap >= 1:
                break
            for fixed in combinations(range(size), upper_count):
                upper = np.zeros(size, dtype=bool)
                upper[list(fixed)] = True
                weights, iterations = _free_newton(pi, risk, history, cap, upper,
                                                    tau, lam, inertia, tolerance,
                                                    max_iterations)
                total_iterations += iterations
                if weights is None or np.any(weights > cap + tolerance):
                    continue
                result = package(weights, False, total_iterations)
                if result.primal_residual <= tolerance and result.kkt_residual <= 5 * tolerance:
                    return package(weights, True, total_iterations)
    else:
        # Proximal entropy step for the smooth quadratic component.  The smooth
        # Hessian norm gives a conservative step; the cap projection is exact.
        weights = fallback.copy()
        lipschitz = lam * float(np.linalg.norm(risk, ord=2)) + inertia
        step = 1.0 / max(lipschitz, tau, 1e-12)
        for iteration in range(1, max_iterations + 1):
            smooth_gradient = lam * (risk @ weights) + inertia * (weights - history)
            log_rates = (np.log(weights) + step * tau * np.log(pi) -
                         step * smooth_gradient) / (1 + step * tau)
            rates = np.exp(np.maximum(log_rates - np.max(log_rates), -690.0))
            weights = _capped_quality(rates / np.sum(rates), cap)
            result = package(weights, False, iteration)
            if result.primal_residual <= tolerance and result.kkt_residual <= 5 * tolerance:
                return package(weights, True, iteration)
        total_iterations = max_iterations
    return package(fallback, False, total_iterations)


def sensitivity_bound(q: Any, q_tilde: Any, matrix: Any, matrix_tilde: Any,
                      anchor: Any, anchor_tilde: Any, cap: float, tau: float,
                      lam: float, inertia: float) -> float:
    """Fixed-mask/cap L2 bound; common log-quality shifts are removed."""
    quality = _vector(q, "q")
    other_quality = _vector(q_tilde, "q_tilde", quality.size)
    _normalized_positive(quality)
    _normalized_positive(other_quality)
    risk = _psd_matrix(matrix, quality.size)
    other_risk = _psd_matrix(matrix_tilde, quality.size)
    history = _vector(anchor, "anchor", quality.size)
    other_history = _vector(anchor_tilde, "anchor_tilde", quality.size)
    if (not math.isfinite(cap) or cap <= 0 or cap > 1 or
            cap * quality.size < 1 - 1e-12):
        raise ValueError("cap must be in [1/n, 1]")
    if (not math.isfinite(tau) or tau <= 0 or not math.isfinite(lam) or lam < 0 or
            not math.isfinite(inertia) or inertia < 0):
        raise ValueError("invalid regularization coefficients")
    log_change = np.log(quality) - np.log(other_quality)
    log_change -= np.mean(log_change)
    numerator = (tau * np.linalg.norm(log_change) +
                 lam * math.sqrt(cap) * np.linalg.norm(risk - other_risk, ord=2) +
                 inertia * np.linalg.norm(history - other_history))
    return float(numerator / (tau / cap + inertia))


class DecisionAwareFusion:
    """Delayed, mask-conditioned decision-moment fusion.

    The ingestion clock never moves backward.  Old forecast labels may arrive
    in any order at the current/later clock, including several labels at the
    same arrival window.  issue() uses strictly increasing forecast windows.
    """

    def __init__(self, n_sources: int, context_dim: int,
                 config: FusionConfig | None = None) -> None:
        if not isinstance(n_sources, int) or n_sources < 1:
            raise ValueError("n_sources must be a positive integer")
        if not isinstance(context_dim, int) or context_dim < 0:
            raise ValueError("context_dim must be a nonnegative integer")
        self.n_sources = n_sources
        self.context_dim = context_dim
        self.config = config if config is not None else FusionConfig()
        if not isinstance(self.config, FusionConfig):
            raise TypeError("config must be a FusionConfig")
        self._forecasts: dict[int, _ForecastRecord] = {}
        self._labelled: set[int] = set()
        self._discarded: set[int] = set()
        self._archive: list[_ResidualRecord] = []
        self._last_issue = -1
        self._clock = -1
        self._last_weights = np.zeros(n_sources)
        self._last_matrices: dict[str, Any] | None = None

    @staticmethod
    def _window(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(value)

    def _quality(self, observations: Sequence[QualityObservation],
                 predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
        if len(observations) != self.n_sources:
            raise ValueError("quality must have one observation per source")
        config = self.config
        values = np.zeros(self.n_sources)
        available = np.zeros(self.n_sources, dtype=bool)
        variance_fallbacks: list[int] = []
        for source, observation in enumerate(observations):
            if not isinstance(observation, QualityObservation):
                raise TypeError("quality entries must be QualityObservation objects")
            if not observation.valid or not np.isfinite(predictions[source]):
                continue
            count, age = float(observation.count), float(observation.age)
            if not (math.isfinite(count) and count >= 0 and math.isfinite(age) and age >= 0):
                raise ValueError("valid observation count and age must be finite and nonnegative")
            if count <= 0 or count < config.min_count or age > config.max_observation_age:
                continue
            if observation.variance is None:
                variance = config.variance_fallback
                variance_fallbacks.append(source)
            else:
                variance = float(observation.variance)
                if not math.isfinite(variance) or variance < 0:
                    raise ValueError("variance must be nonnegative and finite, or None")
            if observation.diagnostic is not None:
                diagnostic = float(observation.diagnostic)
                if not math.isfinite(diagnostic) or not 0 <= diagnostic <= 1:
                    raise ValueError("diagnostic must belong to [0, 1], or be None")
            support = count / (count + config.support_scale) if count + config.support_scale else 0.0
            quality = support * math.exp(-age / config.freshness_tau) / (variance + config.variance_epsilon)
            values[source] = max(config.quality_min, quality)
            available[source] = True
        return values, available, variance_fallbacks

    def _moments(self, window: int, context: np.ndarray,
                 active: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        config = self.config
        indices = np.flatnonzero(active)
        count = indices.size
        weighted = config.prior_weight * config.prior_risk * np.eye(count)
        unweighted = config.prior_weight * config.prior_unweighted_risk * np.eye(count)
        mass = 0.0
        mass_squared = 0.0
        used: list[_ResidualRecord] = []
        exact_count = 0
        eligible_count = 0
        for record in self._archive:
            if config.require_complete_enabled_vectors and not np.all(record.mask):
                continue
            if not np.all(record.mask[indices]):
                continue
            exact = bool(np.array_equal(record.mask, active))
            if config.exact_mask and not exact:
                continue
            eligible_count += 1
            with np.errstate(over="ignore", under="ignore", invalid="raise"):
                difference = context - record.context
                distance = float(difference @ difference)
                coefficient = (config.decay ** (window - record.forecast_window) *
                               math.exp(-distance / (2 * config.context_bandwidth ** 2)))
            if coefficient <= 0:
                continue
            residuals = record.residuals[indices]
            if not np.all(np.isfinite(residuals)):
                raise RuntimeError("an eligible mask contains an unobserved residual")
            outer = np.outer(residuals, residuals)
            weighted += coefficient * record.chi * outer
            unweighted += coefficient * outer
            mass += coefficient
            mass_squared += coefficient ** 2
            used.append(record)
            exact_count += int(exact)
        denominator = config.prior_weight + mass
        weighted /= denominator
        unweighted /= denominator
        prior_fraction = config.prior_weight / denominator
        weight_count = mass ** 2 / mass_squared if mass_squared > 0 else 0.0
        # Age concerns the original target window, not the label arrival time.
        raw_age = (window - max(r.forecast_window + 1 for r in used)
                   if used else config.max_risk_age)
        risk_age = max(0.0, float(raw_age))
        capped_age = min(risk_age, config.max_risk_age)
        delta = (config.delta_base + config.delta_prior * prior_fraction +
                 config.delta_count / math.sqrt(1 + len(used)) +
                 config.delta_age * capped_age / config.max_risk_age)
        audit: dict[str, Any] = {
            "support_count": len(used), "eligible_count": eligible_count,
            "exact_mask_count": exact_count, "superset_count": len(used) - exact_count,
            "risk_age": risk_age, "prior_fraction": prior_fraction,
            "kernel_mass": mass, "effective_weight_count": weight_count,
            "effective_weight_count_interpretation": "kernel concentration diagnostic, not independent sample size",
            "delta": delta, "mask_policy": "exact" if config.exact_mask else "contains",
            "complete_enabled_control": config.require_complete_enabled_vectors,
            "moment_target": "local pooled completed archive",
            "confidence_guarantee": False,
            "used_forecast_windows": [record.forecast_window for record in used],
        }
        return weighted, unweighted, delta, audit

    def issue(self, window: int, predictions: Any, context: Any,
              quality: Sequence[QualityObservation], slopes: Any) -> FusionSnapshot:
        window = self._window(window, "window")
        if window <= self._last_issue or window < self._clock:
            raise ValueError("issue windows must strictly increase and cannot precede the ingestion clock")
        if len(self._forecasts) >= self.config.max_pending_forecasts:
            raise ValueError("pending forecast limit reached; explicitly discard an unavailable target before issuing")
        forecasts = np.asarray(predictions, dtype=float)
        if forecasts.shape != (self.n_sources,) or np.any(np.isinf(forecasts)):
            raise ValueError("predictions must have one value per source; NaN marks unavailable forecasts")
        finite = np.isfinite(forecasts)
        if np.any((forecasts[finite] < 0) | (forecasts[finite] > 1)):
            raise ValueError("finite common-target predictions must belong to [0, 1]")
        forecasts = np.array(forecasts, copy=True)
        context_vector = _vector(context, "context", self.context_dim)
        candidate_slopes = _vector(slopes, "slopes")
        if candidate_slopes.size == 0:
            raise ValueError("slopes must contain at least one candidate")
        span = float(np.max(candidate_slopes)) - float(np.min(candidate_slopes))
        chi = span * span
        if not math.isfinite(chi):
            raise ValueError("squared slope span must be finite")
        qualities, active, variance_fallbacks = self._quality(quality, forecasts)
        indices = np.flatnonzero(active)
        full_weights = np.zeros(self.n_sources)
        audit: dict[str, Any] = {
            "moment_mode": self.config.moment_mode, "active_sources": indices.tolist(),
            "forecast_mask": active.tolist(), "quality": qualities.tolist(),
            "variance_fallback_sources": variance_fallbacks,
            "slope_span": span, "chi": chi, "all_missing": indices.size == 0,
        }
        matrices: dict[str, Any]
        if indices.size == 0:
            audit.update({"support_count": 0, "eligible_count": 0, "exact_mask_count": 0,
                          "superset_count": 0, "risk_age": self.config.max_risk_age,
                          "prior_fraction": 1.0, "delta": 0.0,
                          "solver_primal_residual": 0.0, "solver_kkt_residual": 0.0,
                          "solver_converged": True, "solver_iterations": 0,
                          "abstention": "no valid common-target forecast"})
            empty = np.empty((0, 0))
            matrices = {"active_sources": indices.copy(), "Mhat": empty.copy(),
                        "R": empty.copy(), "S": empty.copy(), "optimizer_matrix": empty.copy(),
                        "delta": 0.0, "audit": copy.deepcopy(audit)}
            snapshot = FusionSnapshot(window, _readonly(full_weights), None, 1.0, 0.0,
                                      None, 0.0, False, True, copy.deepcopy(audit))
        else:
            weighted, unweighted, delta, moment_audit = self._moments(window, context_vector, active)
            audit.update(moment_audit)
            canonical_risk = weighted + delta * np.eye(indices.size)
            optimizer_risk = weighted.copy()
            optimizer_quality = qualities[indices].copy()
            lam, inertia = self.config.lam, self.config.inertia
            mode = self.config.moment_mode
            if mode == "decision_diagonal":
                optimizer_risk = np.diag(np.diag(weighted))
            elif mode == "unweighted_full":
                optimizer_risk = unweighted.copy()
            elif mode == "unweighted_diagonal":
                optimizer_risk = np.diag(np.diag(unweighted))
            elif mode == "trace_matched":
                source_trace, target_trace = float(np.trace(unweighted)), float(np.trace(weighted))
                if source_trace > 0:
                    ratio = target_trace / source_trace
                    optimizer_risk = unweighted * ratio
                    audit["trace_scale"] = ratio
                    audit["trace_matched_fallback"] = False
                else:
                    optimizer_risk = self.config.prior_risk * np.eye(indices.size)
                    audit["trace_scale"] = None
                    audit["trace_matched_fallback"] = True
            elif mode == "no_risk":
                lam = 0.0
            elif mode == "original_c":
                log_quality = (np.log(optimizer_quality) -
                               np.diag(unweighted) / self.config.mse_temperature)
                optimizer_quality = np.exp(np.maximum(log_quality - np.max(log_quality), -690.0))
                lam, inertia = 0.0, 0.0
                audit["baseline_scope"] = "target-aligned Original RF-C, not version-faithful Original RF"
            optimizer_risk += delta * np.eye(indices.size)
            retained = self._last_weights[indices]
            anchor = (retained / np.sum(retained) if np.sum(retained) > 0
                      else np.full(indices.size, 1 / indices.size))
            cap = max(self.config.max_weight, 1 / indices.size)
            if self.config.force_solver_fallback and indices.size > 1:
                pi = _normalized_positive(optimizer_quality)
                weights = _capped_quality(pi, cap)
                primal, kkt = _certificate(weights, pi, optimizer_risk, anchor, cap,
                                           self.config.tau, lam, inertia,
                                           self.config.solver_tolerance)
                result = SolverResult(_readonly(weights),
                                      _objective(weights, pi, optimizer_risk, anchor,
                                                 self.config.tau, lam, inertia),
                                      primal, kkt, False, 0)
                audit["fallback_reason"] = "explicitly forced numerical fallback test"
            else:
                result = solve_capped_fusion(optimizer_quality, optimizer_risk, anchor, cap,
                                            self.config.tau, lam, inertia,
                                            tolerance=self.config.solver_tolerance,
                                            max_iterations=self.config.solver_max_iterations)
                if not result.converged:
                    audit["fallback_reason"] = "optimizer did not obtain a primal/KKT certificate"
            full_weights[indices] = result.weights
            fused = float(result.weights @ forecasts[indices])
            disagreement = float(result.weights @ ((forecasts[indices] - fused) ** 2))
            risk = max(0.0, float(result.weights @ canonical_risk @ result.weights))
            prediction_risk = max(0.0, float(result.weights @ unweighted @ result.weights))
            coverage = indices.size / self.n_sources
            uncertainty = min(1.0,
                              4 * self.config.disagreement_weight * disagreement +
                              self.config.missing_weight * (1 - coverage) +
                              self.config.risk_weight * risk / self.config.risk_reward_scale ** 2 +
                              self.config.age_weight * min(audit["risk_age"], self.config.max_risk_age) /
                              self.config.max_risk_age)
            diagnostic_mass, diagnostic_sum = 0.0, 0.0
            for source in indices:
                if quality[source].diagnostic is not None:
                    diagnostic_mass += full_weights[source]
                    diagnostic_sum += full_weights[source] * float(quality[source].diagnostic)
            diagnostic = diagnostic_sum / diagnostic_mass if diagnostic_mass > 0 else None
            audit.update({"cap": cap, "anchor": anchor.tolist(),
                          "optimizer_quality": optimizer_quality.tolist(),
                          "disagreement": disagreement, "prediction_risk": prediction_risk,
                          "decision_risk": risk, "diagnostic_mass": diagnostic_mass,
                          "solver_objective": result.objective,
                          "solver_primal_residual": result.primal_residual,
                          "solver_kkt_residual": result.kkt_residual,
                          "solver_converged": result.converged,
                          "solver_iterations": result.iterations,
                          "uncertainty_matrix": "full decision moment shared across optimizer controls"})
            matrices = {"active_sources": indices.copy(), "Mhat": weighted.copy(),
                        "R": unweighted.copy(), "S": canonical_risk.copy(),
                        "optimizer_matrix": optimizer_risk.copy(), "delta": delta,
                        "audit": copy.deepcopy(audit)}
            snapshot = FusionSnapshot(window, _readonly(full_weights), fused, uncertainty,
                                      coverage, diagnostic, risk, not result.converged,
                                      False, copy.deepcopy(audit))
        # Commit only after all validation and numerical construction have succeeded.
        self._forecasts[window] = _ForecastRecord(window, _readonly(forecasts),
                                                 _readonly(active, bool),
                                                 _readonly(context_vector), chi)
        self._last_issue = window
        self._clock = window
        self._last_weights = full_weights.copy()
        self._last_matrices = matrices
        return snapshot

    def observe_label(self, forecast_window: int, target: float,
                      arrival_window: int) -> dict[str, Any]:
        origin = self._window(forecast_window, "forecast_window")
        arrival = self._window(arrival_window, "arrival_window")
        if origin in self._labelled:
            raise ValueError("a label for this forecast_window has already been observed")
        if origin in self._discarded:
            raise ValueError("this pending target was explicitly discarded and cannot be rescored")
        if origin not in self._forecasts:
            raise ValueError("no forecast was issued for forecast_window")
        if arrival <= origin:
            raise ValueError("a next-window label must arrive after forecast issuance")
        if arrival < self._clock:
            raise ValueError("label arrival cannot precede the ingestion clock")
        target = float(target)
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("target must belong to [0, 1]")
        issued = self._forecasts[origin]
        errors = np.full(self.n_sources, np.nan)
        errors[issued.mask] = issued.predictions[issued.mask] - target
        record = _ResidualRecord(origin, arrival, target, issued.mask,
                                 _readonly(errors), issued.context, issued.chi)
        self._archive.append(record)
        # Retain recent forecast origins, rather than letting a very old late label
        # evict newer observations solely because it arrived last.
        self._archive.sort(key=lambda item: item.forecast_window)
        self._archive = self._archive[-self.config.archive_length:]
        self._labelled.add(origin)
        del self._forecasts[origin]
        self._clock = arrival
        return {"forecast_window": origin, "arrival_window": arrival, "target_window": origin + 1,
                "observed_sources": int(np.sum(issued.mask)), "mask": issued.mask.tolist(),
                "chi": issued.chi, "archive_count": len(self._archive),
                "used_original_forecast": True}

    def get_last_matrices(self) -> dict[str, Any]:
        """Return independent copies of the most recent active matrices and audit."""
        if self._last_matrices is None:
            raise ValueError("no fusion snapshot has been issued")
        return copy.deepcopy(self._last_matrices)

    def archive_summary(self) -> list[dict[str, Any]]:
        """Copied archive diagnostics; NaN explicitly denotes missing residuals."""
        return [{"forecast_window": record.forecast_window,
                 "arrival_window": record.arrival_window, "target": record.target,
                 "mask": record.mask.copy(), "residuals": record.residuals.copy(),
                 "context": record.context.copy(), "chi": record.chi}
                for record in self._archive]

    def pending_windows(self) -> tuple[int, ...]:
        return tuple(sorted(self._forecasts))

    def discard_pending(self, window: int, *, reason: str = "target unavailable") -> dict[str, Any]:
        """Explicitly release an unlabelled record; it cannot later be rescored.

        No pending record is silently evicted when the memory limit is reached.
        This operation does not add an observation or reset any risk-age counter.
        """
        origin = self._window(window, "window")
        if origin not in self._forecasts:
            raise ValueError("window is not a pending forecast")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("discard reason must be a nonempty string")
        del self._forecasts[origin]
        self._discarded.add(origin)
        return {"forecast_window": origin, "discarded": True, "reason": reason,
                "pending_count": len(self._forecasts), "observation_added": False}


__all__ = ["FusionConfig", "QualityObservation", "FusionSnapshot", "SolverResult",
           "DecisionAwareFusion", "solve_capped_fusion", "sensitivity_bound"]
