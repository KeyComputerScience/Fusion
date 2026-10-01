"""Causal quality, residual-risk and inertia fusion for the revision experiments.

This module operates on completed ``FusionSnapshot`` objects from core/fusion.py.
It leaves the underlying source estimators unchanged.  Raw source changes are
retained for the drift score; disagreement is computed on forecasts of one common
next-window service target.  Observed prediction risk is not measurement quality.

The solver minimizes tau*KL(w||pi)+lambda/2*w'Rw+kappa/2*||w-w_prev||^2.
R is updated only from complete vectors of STORED forecasts after their common
target has arrived.  Complete-case updating avoids non-PSD pairwise covariance
estimates and never interprets a missing residual as zero prediction error.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize


def _capped_reference(reference: np.ndarray, cap: float) -> np.ndarray:
    """Water-filling KL projection used as a feasible optimization start."""
    result = np.zeros_like(reference)
    remaining = np.ones(len(reference), dtype=bool)
    mass = 1.0
    while np.any(remaining):
        indices = np.flatnonzero(remaining)
        proposal = mass * reference[indices] / reference[indices].sum()
        overloaded = proposal > cap + 1e-14
        if not np.any(overloaded):
            result[indices] = proposal
            break
        saturated = indices[overloaded]
        result[saturated] = cap
        remaining[saturated] = False
        mass -= cap * len(saturated)
    return result


def solve_weights(reference, valid, *, cap=0.7, residual_matrix=None,
                  previous=None, entropy_temperature=1.0, residual_penalty=1.0,
                  inertia=0.1):
    """Solve the strictly convex masked problem; return weights and diagnostics.

    Positive reference values are required on active sources. The numerical lower
    bound is 1e-12 on active coordinates, substantially below reporting precision.
    A failed solve raises an exception rather than silently using heuristic weights.
    """
    pi, mask = np.asarray(reference, float), np.asarray(valid, bool)
    if pi.ndim != 1 or mask.shape != pi.shape:
        raise ValueError("Reference and source mask must be equal-length vectors")
    if not 0 < cap <= 1 or entropy_temperature <= 0 or residual_penalty < 0 or inertia < 0:
        raise ValueError("Require 0<cap<=1, tau>0, lambda>=0 and kappa>=0")
    dimension = len(pi)
    matrix = np.zeros((dimension, dimension)) if residual_matrix is None else np.asarray(residual_matrix, float)
    prior = np.zeros(dimension) if previous is None else np.asarray(previous, float)
    if matrix.shape != (dimension, dimension) or prior.shape != pi.shape:
        raise ValueError("Residual matrix and previous weights have inconsistent dimensions")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(prior)):
        raise ValueError("Solver state must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-12) or np.linalg.eigvalsh(matrix).min() < -1e-10:
        raise ValueError("Residual-risk matrix must be symmetric positive semidefinite")
    active = np.flatnonzero(mask)
    weights = np.zeros(dimension)
    metadata = {"solver": "abstain", "success": True, "active_sources": int(len(active)),
                "effective_cap": 1.0, "objective": None, "kkt_residual": 0.0}
    if len(active) == 0:
        return weights, metadata
    if not np.all(np.isfinite(pi[active])) or np.any(pi[active] <= 0):
        raise ValueError("Active reference masses must be finite and positive")
    p = pi[active] / pi[active].sum()
    effective_cap = max(float(cap), 1.0 / len(active))
    risk = matrix[np.ix_(active, active)]
    prev = prior[active]
    tau, lam, kappa = float(entropy_temperature), float(residual_penalty), float(inertia)

    def objective(value):
        positive = np.maximum(value, 1e-300)
        return float(tau * np.sum(value * np.log(positive / p))
                     + 0.5 * lam * value @ risk @ value
                     + 0.5 * kappa * np.sum((value - prev) ** 2))

    def gradient(value):
        return (tau * (np.log(np.maximum(value, 1e-300) / p) + 1)
                + lam * risk @ value + kappa * (value - prev))

    if len(active) == 1 or abs(effective_cap * len(active) - 1.0) < 1e-13:
        answer = np.full(len(active), 1.0 / len(active))
        solver_name = "singleton" if len(active) == 1 else "singleton_feasible_set"
        iterations = 0
    elif lam == 0 and kappa == 0:
        answer = _capped_reference(p, effective_cap)
        solver_name, iterations = "capped_kl_water_filling", 0
    else:
        initial = _capped_reference(p, effective_cap)
        # Bound clipping cannot disturb feasibility at ordinary reference scales.
        initial = np.maximum(initial, 1e-12)
        initial /= initial.sum()
        fitted = minimize(objective, initial, jac=gradient, method="SLSQP",
                          bounds=[(1e-12, effective_cap)] * len(active),
                          constraints={"type": "eq", "fun": lambda a: float(a.sum() - 1),
                                       "jac": lambda a: np.ones_like(a)},
                          options={"ftol": 1e-12, "maxiter": 300, "disp": False})
        if not fitted.success:
            raise RuntimeError(f"Convex fusion solver failed: {fitted.message}")
        answer = fitted.x
        solver_name, iterations = "convex_slsqp", int(fitted.nit)
    if abs(answer.sum() - 1) > 1e-9 or answer.min() < -1e-10 or answer.max() > effective_cap + 1e-9:
        raise RuntimeError("Convex fusion solver returned infeasible weights")
    grad = gradient(answer)
    interior = (answer > 1e-10) & (answer < effective_cap - 1e-9)
    # Equality multipliers are identified by free coordinates. At upper bounds
    # the objective derivative must be no larger than that free derivative.
    if np.any(interior):
        multiplier = float(grad[interior].mean())
        stationarity = float(np.max(np.abs(grad[interior] - multiplier)))
        upper = answer >= effective_cap - 1e-9
        lower = answer <= 1e-10
        violations = [stationarity]
        if np.any(upper):
            violations.append(float(np.max(np.maximum(grad[upper] - multiplier, 0))))
        if np.any(lower):
            violations.append(float(np.max(np.maximum(multiplier - grad[lower], 0))))
        kkt = max(violations)
        if kkt > 2e-5:
            raise RuntimeError(f"Convex fusion solver failed stationarity check ({kkt:g})")
    else:
        kkt = 0.0  # A singleton feasible set needs no free-coordinate check.
    weights[active] = answer
    metadata.update(solver=solver_name, effective_cap=effective_cap,
                    objective=objective(answer), kkt_residual=kkt, iterations=iterations)
    return weights, metadata


def sensitivity_certificate(before, after, *, score_kind="forecast"):
    """Realized perturbation certificate for this convex objective.

    The masks, cap and objective coefficients must be identical. The inequality
    accounts for reference-quality, residual-matrix and inertia-state changes;
    it is not the original fixed-reliability bound and is not valid across outages.
    This is a deterministic numerical audit, not a service-performance guarantee.
    """
    if score_kind not in ("forecast", "raw"):
        raise ValueError("Score kind must be forecast or raw")
    if (before.masks != after.masks or not any(before.masks)
            or abs(before.effective_weight_cap - after.effective_weight_cap) > 1e-12
            or before.revised_fusion_parameters != after.revised_fusion_parameters):
        return {"applicable": False, "reason": "mask_cap_or_objective_changed"}
    mask = np.asarray(before.masks, bool)
    a, b = np.asarray(before.weights), np.asarray(after.weights)
    old_q, new_q = np.asarray(before.quality_reference), np.asarray(after.quality_reference)
    old_r, new_r = np.asarray(before.residual_second_moment), np.asarray(after.residual_second_moment)
    old_v, new_v = np.asarray(before.previous_forecast_weights), np.asarray(after.previous_forecast_weights)
    cfg = before.revised_fusion_parameters
    tau, lam, kappa = cfg["entropy_temperature"], cfg["residual_penalty"], cfg["inertia"]
    log_change = np.log(new_q[mask]) - np.log(old_q[mask])
    force = tau * log_change + kappa * (new_v - old_v)[mask] - lam * ((new_r - old_r) @ b)[mask]
    oscillation = float(force.max() - force.min())
    weight_bound = min(2.0, oscillation / (2 * tau))
    old_values = before.next_target_predictions if score_kind == "forecast" else before.scores
    new_values = after.next_target_predictions if score_kind == "forecast" else after.scores
    x = np.asarray([old_values[s] for s in np.flatnonzero(mask)], float)
    y = np.asarray([new_values[s] for s in np.flatnonzero(mask)], float)
    change_bound = min(1.0, before.effective_weight_cap * float(np.abs(y - x).sum()) + weight_bound / 2)
    observed = abs(float(b[mask] @ y - a[mask] @ x))
    return {"applicable": True, "score_kind": score_kind,
            "weight_l1_observed": float(np.abs(b - a).sum()), "weight_l1_bound": weight_bound,
            "score_change_observed": observed, "score_change_bound": change_bound,
            "force_oscillation": oscillation,
            "numerical_tolerance": 2e-5}


class RevisedFusion:
    """Augment causal source estimators and replace their fusion decision state."""

    def __init__(self, config: dict, *, entropy_temperature=1.0,
                 residual_penalty=1.0, inertia=0.1, residual_decay=0.9,
                 use_forecast_disagreement=True):
        self.config = copy.deepcopy(config.get("fusion", config))
        if not 0 <= residual_decay < 1:
            raise ValueError("Residual decay must lie in [0,1)")
        self.entropy_temperature = entropy_temperature
        self.residual_penalty = residual_penalty
        self.inertia = inertia
        self.residual_decay = residual_decay
        self.use_forecast_disagreement = use_forecast_disagreement
        self.residual_matrix = None
        self.previous_weights = None
        self.pending = None
        self.residual_updates = 0
        self.skipped_incomplete = 0
        self.skipped_context = 0
        self.last_completed_slot = None

    @staticmethod
    def _context(context):
        if context is None:
            return False, True, None
        if isinstance(context, dict):
            return True, bool(context.get("stable", False)), context.get("key")
        return True, True, context

    def update(self, snapshot: Any, control_context=None):
        """Return a copied snapshot usable by the existing service coordinator.

        Call once per completed window. Optional context is either a stable
        identifier or {"stable": bool, "key": identifier}; changing/mixed contexts
        prevent delayed residual updates. None means unconditioned prediction
        risk, appropriate to an explicitly fixed-policy replay, not causal quality.
        """
        if self.last_completed_slot is not None and snapshot.completed_slot <= self.last_completed_slot:
            raise ValueError("Completed snapshots must arrive once in chronological order")
        output = copy.deepcopy(snapshot)
        dimension = len(snapshot.scores)
        if self.residual_matrix is None:
            self.residual_matrix = np.zeros((dimension, dimension))
            self.previous_weights = np.zeros(dimension)
        if self.residual_matrix.shape != (dimension, dimension):
            raise ValueError("Number of sources cannot change during a run")
        enabled = list(self.config.get("enabled_sources", range(dimension)))
        if not enabled or len(set(enabled)) != len(enabled) or any(s < 0 or s >= dimension for s in enabled):
            raise ValueError("Enabled source list must contain unique valid indices")
        conditioned, stable, context_key = self._context(control_context)
        update_kind = "no_previous_forecast"
        target = snapshot.scores[2] if dimension >= 3 else None
        if self.pending is not None:
            context_ok = ((not conditioned and not self.pending["conditioned"])
                          or (conditioned and self.pending["conditioned"] and stable
                              and self.pending["stable"] and context_key == self.pending["context"]))
            complete = (target is not None and math.isfinite(target)
                        and all(self.pending["masks"][s] and self.pending["predictions"][s] is not None
                                for s in enabled))
            if not context_ok:
                self.skipped_context += 1
                update_kind = "skipped_control_context"
            elif not complete:
                self.skipped_incomplete += 1
                update_kind = "skipped_incomplete_residual_vector"
            else:
                residual = np.zeros(dimension)
                residual[enabled] = [self.pending["predictions"][s] - target for s in enabled]
                self.residual_matrix = (self.residual_decay * self.residual_matrix
                                        + (1 - self.residual_decay) * np.outer(residual, residual))
                self.residual_updates += 1
                update_kind = "arrived_target_complete_vector"

        predictions = list(snapshot.next_target_predictions)
        mask = np.array([s in enabled and bool(snapshot.masks[s]) and predictions[s] is not None
                         and math.isfinite(predictions[s]) and snapshot.scores[s] is not None
                         and math.isfinite(snapshot.scores[s]) for s in range(dimension)], dtype=bool)
        if any(not 0 <= predictions[s] <= 1 or not 0 <= snapshot.scores[s] <= 1 for s in np.flatnonzero(mask)):
            raise ValueError("Source changes and forecasts must lie in [0,1]")
        reference = np.zeros(dimension)
        log_quality = np.full(dimension, -np.inf)
        for source in np.flatnonzero(mask):
            n = float(snapshot.effective_samples[source])
            age = float(snapshot.ages[source])
            variance = snapshot.noise_variances[source]
            variance = float(self.config.get("unsupported_noise_variance", 0.25) if variance is None else variance)
            if not math.isfinite(n + age + variance) or n <= 0 or age < 0 or variance < 0:
                raise ValueError("Available source support, age and variability must be valid")
            log_mass = 0.0
            if self.config.get("use_support", True):
                log_mass += math.log(n / (n + self.config.get("support_scale", 20.0)))
            if self.config.get("use_freshness", True):
                log_mass -= age / self.config.get("freshness_scale", 50.0)
            if self.config.get("use_noise", True):
                log_mass -= math.log(variance + self.config.get("epsilon_noise", 0.0025))
            log_quality[source] = log_mass
        if np.any(mask):
            # A documented 1e-12 floor protects numerical reference positivity;
            # outcome consistency is deliberately excluded from this quality mass.
            reference[mask] = np.maximum(np.exp(np.maximum(log_quality[mask] - log_quality[mask].max(), -700)), 1e-12)
            reference /= reference.sum()
        weights, solver = solve_weights(
            reference, mask, cap=self.config.get("maximum_source_weight", 0.7),
            residual_matrix=self.residual_matrix, previous=self.previous_weights,
            entropy_temperature=self.entropy_temperature,
            residual_penalty=self.residual_penalty, inertia=self.inertia)
        original = {name: copy.deepcopy(getattr(snapshot, name))
                    for name in ("gamma", "weights", "reliability", "disagreement", "uncertainty", "coverage")}
        output.weights = weights.tolist()
        output.masks = mask.tolist()
        output.reliability = reference.tolist()
        output.coverage = float(mask.sum() / len(enabled))
        if mask.sum():
            forecast = float(sum(weights[s] * predictions[s] for s in np.flatnonzero(mask)))
            raw_gamma = float(sum(weights[s] * snapshot.scores[s] for s in np.flatnonzero(mask)))
            forecast_disagreement = float(sum(weights[s] * (predictions[s] - forecast) ** 2 for s in np.flatnonzero(mask)))
            raw_disagreement = float(sum(weights[s] * (snapshot.scores[s] - raw_gamma) ** 2 for s in np.flatnonzero(mask)))
            used_disagreement = forecast_disagreement if self.use_forecast_disagreement else raw_disagreement
            uncertainty = min(1.0, (4 * used_disagreement if self.config.get("use_disagreement", True) else 0)
                              + self.config.get("missing_source_penalty", 0.5) * (1 - output.coverage))
            output.gamma, output.disagreement, output.uncertainty = raw_gamma, used_disagreement, uncertainty
            output.fallback, output.fallback_reason = False, ""
        else:
            forecast, forecast_disagreement, raw_disagreement = None, 0.0, 0.0
            output.gamma = snapshot.gamma  # Historical diagnostic only; fallback prohibits training.
            output.disagreement, output.uncertainty = 0.0, 1.0
            output.coverage, output.fallback, output.fallback_reason = 0.0, True, "no_valid_forecast_source"
        output.effective_weight_cap = solver["effective_cap"]
        for name, value in original.items():
            setattr(output, "original_" + name, value)
        output.forecast_score = forecast
        output.forecast_weights = weights.tolist()
        output.forecast_disagreement = forecast_disagreement
        output.raw_disagreement = raw_disagreement
        output.forecast_risk = float(weights @ self.residual_matrix @ weights)
        output.forecast_valid = bool(mask.sum())
        output.quality_reference = reference.tolist()
        output.residual_second_moment = self.residual_matrix.tolist()
        output.residual_updates = self.residual_updates
        output.residual_update_kind = update_kind
        output.skipped_incomplete_residual_updates = self.skipped_incomplete
        output.skipped_control_residual_updates = self.skipped_context
        output.action_conditioned = conditioned
        output.control_context_stable = stable
        output.forecast_solver = solver
        output.revised_fusion_parameters = {
            "entropy_temperature": self.entropy_temperature, "residual_penalty": self.residual_penalty,
            "inertia": self.inertia, "residual_decay": self.residual_decay,
            "use_forecast_disagreement": self.use_forecast_disagreement}
        output.previous_forecast_weights = self.previous_weights.tolist()
        self.previous_weights = weights.copy()
        self.pending = {"predictions": predictions, "masks": mask.tolist(),
                        "conditioned": conditioned, "stable": stable, "context": copy.deepcopy(context_key)}
        self.last_completed_slot = snapshot.completed_slot
        return output

    @staticmethod
    def extras(snapshot):
        """JSON-safe additional fields; base dataclass.to_dict omits dynamic attrs."""
        names = ("forecast_score", "forecast_weights", "forecast_disagreement", "raw_disagreement",
                 "forecast_risk", "forecast_valid", "quality_reference", "residual_second_moment",
                 "residual_updates", "residual_update_kind", "skipped_incomplete_residual_updates",
                 "skipped_control_residual_updates", "action_conditioned", "control_context_stable",
                 "forecast_solver", "revised_fusion_parameters", "previous_forecast_weights", "original_gamma", "original_weights",
                 "original_reliability", "original_disagreement", "original_uncertainty", "original_coverage")
        return {name: copy.deepcopy(getattr(snapshot, name)) for name in names}
