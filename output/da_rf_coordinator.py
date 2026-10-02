"""Finite-profile DA-RF coordination with explicit conditional claim boundaries.

Only NumPy and the standard library are required. ``FusionSnapshot`` is provided
by the sibling ``da_rf_fusion`` module. Candidate values are caller-supplied
full-horizon estimates: this module does not invent counterfactual recovery,
perform gradient updates, or infer a simultaneous error bound from a flag.

An accepted decision is a launch instruction. Probe cost is reserved at selection
time and reported separately from measured execution expenditure. The caller
must execute the instruction, maintain worker eligibility/occupied capacities,
and reconcile actual costs externally. Repeating the same slot with the same
decision inputs is idempotent; changing its inputs is an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Iterable

import numpy as np

from da_rf_fusion import FusionSnapshot


def _finite(value: float, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _jsonable(value: Any) -> Any:
    """Convert an audit into JSON data, rejecting NaN/Inf instead of hiding them."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite(value, "audit float")
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"Unsupported audit value: {type(value).__name__}")


@dataclass(frozen=True)
class Candidate:
    """One planned update/inference pair, expressed in common horizon units.

``resource_demand`` is the total pair demand and must match capacity's shape.
``gross_baseline`` is W, ``gain`` is the paired full-horizon gross difference G,
and ``cost`` is the full-horizon direct training charge before alpha_cost.
An empirical error allowance can support conditional admission but cannot alone
support ``certificate_valid``. That flag asserts external coverage by a valid
simultaneous error bound. No-update candidates have exactly zero G/cost/steps.
"""

    name: str
    update: bool
    gross_baseline: float
    gain: float
    error_allowance: float
    cost: float
    resource_demand: np.ndarray
    gradient_steps: int
    delay: int
    supported: bool
    certificate_valid: bool = False
    worker_eligible: bool = True
    support_count: int = 0
    probe_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Candidate name must be nonempty")
        for flag in ("update", "supported", "certificate_valid", "worker_eligible",
                     "probe_eligible"):
            if not isinstance(getattr(self, flag), (bool, np.bool_)):
                raise ValueError(f"{flag} must be boolean")
            object.__setattr__(self, flag, bool(getattr(self, flag)))
        for name in ("gross_baseline", "gain"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in ("error_allowance", "cost"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0))
        for name in ("gradient_steps", "delay", "support_count"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        demand = np.array(self.resource_demand, dtype=float, copy=True)
        if demand.size == 0 or not np.isfinite(demand).all() or (demand < 0).any():
            raise ValueError("resource_demand must contain finite nonnegative demands")
        demand.setflags(write=False)
        object.__setattr__(self, "resource_demand", demand)
        if not self.update:
            if self.gain != 0 or self.cost != 0 or self.gradient_steps != 0:
                raise ValueError("No-update candidates require gain=cost=gradient_steps=0")
            if self.probe_eligible:
                raise ValueError("A no-update candidate cannot be a probe")
        elif self.gradient_steps < 1 or self.delay < 1:
            raise ValueError("Update candidates require positive gradient_steps and delay")
        if self.certificate_valid and not self.supported:
            raise ValueError("An unsupported candidate cannot have a recovery certificate")


@dataclass(frozen=True)
class CoordinatorConfig:
    threshold_on: float = 0.15
    threshold_off: float = 0.08
    min_coverage: float = 2 / 3
    alpha_cost: float = 0.12
    omega_intensity: float = 0.04
    lambda_uncertainty: float = 0.08
    cost_reference: float = 1.0
    margin: float = 0.0
    probe_name: str | None = None
    probe_budget: float = 0.0
    probe_cooldown: int = 50
    probe_loss_windows: int = 2
    observed_loss_max_age: int = 1
    snapshot_max_age: int = 1
    max_probe_gradient_steps: int = 32
    simultaneous_certificate: bool = False
    feasibility_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        for name in ("threshold_on", "threshold_off", "min_coverage"):
            object.__setattr__(self, name, _finite(getattr(self, name), name,
                                                 minimum=0, maximum=1))
        if not self.threshold_off < self.threshold_on:
            raise ValueError("Require threshold_off < threshold_on")
        for name in ("alpha_cost", "omega_intensity", "lambda_uncertainty", "margin",
                     "probe_budget", "feasibility_tolerance"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0))
        object.__setattr__(self, "cost_reference",
                           _finite(self.cost_reference, "cost_reference", minimum=0))
        if self.cost_reference == 0:
            raise ValueError("cost_reference must be positive")
        for name in ("probe_cooldown", "observed_loss_max_age", "snapshot_max_age"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ("probe_loss_windows", "max_probe_gradient_steps"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        if self.probe_name is not None and (not isinstance(self.probe_name, str)
                                           or not self.probe_name.strip()):
            raise ValueError("probe_name must be None or a nonempty name")
        if not isinstance(self.simultaneous_certificate, (bool, np.bool_)):
            raise ValueError("simultaneous_certificate must be boolean")


@dataclass(frozen=True)
class Decision:
    candidate_name: str | None
    is_update: bool
    is_probe: bool
    certificate_status: str
    reason: str
    scores: dict[str, float]
    admission_margins: dict[str, float]
    loss_flag: bool
    probe_budget_used: float
    audit: dict[str, Any] = field(default_factory=dict)

    def audit_dict(self) -> dict[str, Any]:
        return _jsonable({
            "candidate_name": self.candidate_name,
            "is_update": self.is_update,
            "is_probe": self.is_probe,
            "certificate_status": self.certificate_status,
            "reason": self.reason,
            "scores": self.scores,
            "admission_margins": self.admission_margins,
            "loss_flag": self.loss_flag,
            "probe_budget_used": self.probe_budget_used,
            "audit": self.audit,
        })

    def audit_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.audit_dict(), allow_nan=False, sort_keys=True, indent=indent)


class PersistentRecoveryCoordinator:
    """Stateful hysteresis, exact profile enumeration, and budgeted probe selection.

    Observation/snapshot ages and completed-window identities are measured in windows;
    cooldown and ``slot`` are measured in slots. A carried label retains its
    original identity and cannot supply another persistent-loss observation.
    ``select`` does not update W/G/support from a factual probe outcome. Those
    labels require the caller's controlled comparison protocol.
    """

    def __init__(self, config: CoordinatorConfig | None = None) -> None:
        self.config = config or CoordinatorConfig()
        self.loss_flag = False
        self.probe_budget_used = 0.0
        self.probe_reservations: list[dict[str, Any]] = []
        self._observed_losses: dict[int, float] = {}
        self._last_probe_slot: int | None = None
        self._last_slot: int | None = None
        self._last_fingerprint: tuple[Any, ...] | None = None
        self._last_decision: Decision | None = None

    def select(self, *, slot: int, snapshot: FusionSnapshot,
               candidates: Iterable[Candidate], capacity: np.ndarray,
               observed_loss: float | None = None, observed_loss_age: int = 0,
               completed_window: int | None = None,
               observed_loss_window: int | None = None) -> Decision:
        cfg = self.config
        slot = _integer(slot, "slot")
        age = _integer(observed_loss_age, "observed_loss_age")
        window = _integer(snapshot.window if completed_window is None else completed_window,
                          "completed_window")
        snapshot_window = _integer(snapshot.window, "snapshot.window")
        if window < snapshot_window:
            raise ValueError("completed_window cannot precede snapshot.window")
        snapshot_age = window - snapshot_window
        stale_snapshot = snapshot_age > cfg.snapshot_max_age
        forecast = None if snapshot.forecast is None else _finite(
            snapshot.forecast, "snapshot.forecast", minimum=0, maximum=1)
        uncertainty = _finite(snapshot.uncertainty, "snapshot.uncertainty", minimum=0, maximum=1)
        coverage = _finite(snapshot.coverage, "snapshot.coverage", minimum=0, maximum=1)
        all_missing = bool(snapshot.all_missing)
        solver_fallback = bool(snapshot.solver_fallback)
        obs = None if observed_loss is None else _finite(
            observed_loss, "observed_loss", minimum=0, maximum=1)
        origin = window - age if observed_loss_window is None else _integer(
            observed_loss_window, "observed_loss_window")
        if observed_loss_window is not None and origin != window - age:
            raise ValueError("observed_loss_window must equal completed_window - observed_loss_age")
        if obs is not None and origin < 0:
            raise ValueError("A loss observation cannot precede completed window zero")
        fresh_obs = obs is not None and age <= cfg.observed_loss_max_age
        pool = tuple(candidates)
        if any(not isinstance(candidate, Candidate) for candidate in pool):
            raise TypeError("Every candidate must be a Candidate instance")
        if len({candidate.name for candidate in pool}) != len(pool):
            raise ValueError("Candidate names must be unique")
        available = np.array(capacity, dtype=float, copy=True)
        if available.size == 0 or not np.isfinite(available).all() or (available < 0).any():
            raise ValueError("capacity must contain finite nonnegative values")
        for candidate in pool:
            if candidate.resource_demand.shape != available.shape:
                raise ValueError(f"Resource shape mismatch for candidate {candidate.name}")
        fingerprint = (
            window, snapshot_window, forecast, uncertainty, coverage, all_missing, solver_fallback,
            obs, age, origin, available.shape, available.tobytes(),
            tuple((c.name, c.update, c.gross_baseline, c.gain, c.error_allowance, c.cost,
                   c.resource_demand.tobytes(), c.gradient_steps, c.delay, c.supported,
                   c.certificate_valid, c.worker_eligible, c.support_count, c.probe_eligible)
                  for c in pool),
        )
        if self._last_slot is not None:
            if slot < self._last_slot:
                raise ValueError("Decision slots must be nondecreasing")
            if slot == self._last_slot:
                if fingerprint != self._last_fingerprint:
                    raise ValueError("A repeated slot must have identical decision inputs")
                assert self._last_decision is not None
                return self._last_decision
        if fresh_obs and origin in self._observed_losses and self._observed_losses[origin] != obs:
            raise ValueError("A completed loss observation cannot change after publication")

        intensity = {c.name: min(1.0, c.cost / cfg.cost_reference) for c in pool}
        scores = {
            c.name: _finite(c.gross_baseline + c.gain - c.error_allowance
                            - cfg.alpha_cost * c.cost
                            - cfg.omega_intensity * intensity[c.name] ** 2
                            - cfg.lambda_uncertainty * uncertainty * intensity[c.name],
                            f"score[{c.name}]")
            for c in pool
        }
        feasible = {c.name: bool(np.all(c.resource_demand <= available + cfg.feasibility_tolerance))
                    for c in pool}
        no_update = [c for c in pool if not c.update and feasible[c.name]]
        baseline_upper = None if not no_update else max(
            _finite(c.gross_baseline + c.error_allowance, f"upper[{c.name}]") for c in no_update)
        margins = {} if baseline_upper is None else {
            c.name: _finite(scores[c.name] - baseline_upper - cfg.margin,
                            f"admission_margin[{c.name}]") for c in pool if c.update
        }

        # Validity is evaluated before a loss enters hysteresis or persistence.
        terms = ([] if all_missing or stale_snapshot or forecast is None else [forecast])
        if fresh_obs and not all_missing:
            terms.append(obs)
        h = max(terms) if terms else None
        if h is not None:
            if h >= cfg.threshold_on:
                self.loss_flag = True
            elif h < cfg.threshold_off:
                self.loss_flag = False
        # Absence of evidence leaves the old flag intact, but blocks admission.
        streak = 0
        if fresh_obs and not all_missing:
            self._observed_losses[origin] = obs
            cursor = origin
            while self._observed_losses.get(cursor, -1.0) >= cfg.threshold_on:
                streak += 1
                cursor -= 1
            retain = max(8, cfg.probe_loss_windows + cfg.observed_loss_max_age + 2)
            cutoff = max(self._observed_losses) - retain
            self._observed_losses = {k: v for k, v in self._observed_losses.items() if k >= cutoff}

        eligible: list[Candidate] = []
        rejections: dict[str, list[str]] = {}
        for candidate in pool:
            if not candidate.update:
                continue
            why: list[str] = []
            if all_missing:
                why.append("all_missing")
            if stale_snapshot:
                why.append("stale_fusion_snapshot")
            if h is None:
                why.append("no_valid_loss_indicator")
            elif not self.loss_flag:
                why.append("loss_gate_off")
            if coverage < cfg.min_coverage:
                why.append("insufficient_coverage")
            if solver_fallback:
                why.append("fusion_solver_fallback")
            if not feasible[candidate.name]:
                why.append("resource_infeasible")
            if not candidate.worker_eligible:
                why.append("worker_ineligible")
            if not candidate.supported:
                why.append("unsupported_recovery")
            if why:
                rejections[candidate.name] = why
            else:
                eligible.append(candidate)

        base_audit: dict[str, Any] = {
            "slot": slot, "completed_window": window, "observed_loss_window":
            origin if obs is not None else None, "observed_loss_age": age,
            "fresh_observed_loss": fresh_obs, "persistent_observed_windows": streak,
            "loss_indicator": h, "baseline_upper": baseline_upper,
            "uncertainty": uncertainty, "coverage": coverage,
            "all_missing": all_missing, "solver_fallback": solver_fallback,
            "stale_fusion_snapshot": stale_snapshot,
            "effective_evidence_abstention": all_missing or stale_snapshot,
            "intensities": intensity, "resource_feasible": feasible,
            "pre_admission_rejections": rejections,
            "probe_budget_basis": "planned_reserved_direct_cost",
            "simultaneous_certificate_asserted": bool(cfg.simultaneous_certificate),
            "snapshot_window": snapshot_window, "snapshot_age": snapshot_age,
        }

        def finish(candidate: Candidate | None, *, probe: bool, status: str,
                   reason: str, extra: dict[str, Any] | None = None) -> Decision:
            audit = dict(base_audit)
            if extra:
                audit.update(extra)
            decision = Decision(
                candidate_name=None if candidate is None else candidate.name,
                is_update=False if candidate is None else candidate.update,
                is_probe=probe, certificate_status=status, reason=reason,
                scores=dict(scores), admission_margins=dict(margins),
                loss_flag=self.loss_flag, probe_budget_used=self.probe_budget_used,
                audit=_jsonable(audit),
            )
            self._last_slot, self._last_fingerprint, self._last_decision = slot, fingerprint, decision
            return decision

        if not no_update:
            return finish(None, probe=False, status="not_applicable",
                          reason="service_deferral_no_feasible_no_update")
        fallback = min(no_update, key=lambda c: (-scores[c.name], c.name))
        if all_missing:
            return finish(fallback, probe=False, status="not_applicable",
                          reason="all_missing_no_update")
        if stale_snapshot:
            return finish(fallback, probe=False, status="not_applicable",
                          reason="stale_fusion_snapshot")
        admitted = [c for c in eligible if margins[c.name] > 0]
        base_audit["admitted_updates"] = [c.name for c in admitted]
        if admitted:
            chosen = min(admitted, key=lambda c: (-scores[c.name], c.name))
            comparison_pool = no_update + eligible
            certified = bool(cfg.simultaneous_certificate and all(
                c.certificate_valid for c in comparison_pool))
            return finish(chosen, probe=False,
                          status="certified" if certified else "conditional",
                          reason="supported_update_clears_value_margin",
                          extra={"certificate_comparison_pool": [c.name for c in comparison_pool],
                                 "certificate_is_external_bound_assertion": certified})

        # A probe is an explicitly budgeted exception, never a value certificate.
        probe_reasons: list[str] = []
        if not fresh_obs or obs < cfg.threshold_on:
            probe_reasons.append("no_fresh_high_observed_loss")
        if streak < cfg.probe_loss_windows:
            probe_reasons.append("insufficient_distinct_loss_windows")
        if cfg.probe_budget <= 0:
            probe_reasons.append("probe_budget_disabled")
        if self._last_probe_slot is not None and slot - self._last_probe_slot < cfg.probe_cooldown:
            probe_reasons.append("probe_cooldown")
        probe_candidates = []
        if not probe_reasons:
            for candidate in pool:
                permitted = candidate.name == cfg.probe_name if cfg.probe_name is not None else candidate.probe_eligible
                if (candidate.update and permitted and candidate.worker_eligible
                        and feasible[candidate.name]
                        and candidate.gradient_steps <= cfg.max_probe_gradient_steps
                        and candidate.cost <= cfg.probe_budget - self.probe_budget_used):
                    probe_candidates.append(candidate)
        if probe_candidates:
            # At equal probe cost, retain the best predicted no-update inference
            # value; remaining ties use work/delay/name deterministically.
            chosen = min(probe_candidates,
                         key=lambda c: (c.cost, -c.gross_baseline, c.gradient_steps,
                                        c.delay, c.name))
            self.probe_budget_used += chosen.cost
            self._last_probe_slot = slot
            reservation = {"slot": slot, "candidate": chosen.name, "cost": chosen.cost,
                           "gradient_steps": chosen.gradient_steps, "loss_window": origin}
            self.probe_reservations.append(reservation)
            return finish(chosen, probe=True, status="probe_not_certified",
                          reason="budgeted_probe_after_persistent_observed_loss",
                          extra={"probe_reservation": reservation})
        if not probe_reasons:
            probe_reasons.append("no_permitted_feasible_probe_within_budget")
        return finish(fallback, probe=False, status="not_applicable",
                      reason="no_update_no_admitted_exploitation_or_probe",
                      extra={"probe_rejections": probe_reasons})


__all__ = ["Candidate", "CoordinatorConfig", "Decision", "PersistentRecoveryCoordinator"]
