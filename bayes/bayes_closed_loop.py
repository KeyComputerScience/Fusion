#!/usr/bin/env python3
"""Controlled Bayesian-Markov residual-fusion service experiment.

No trace, sensor or hardware claim: source instruments are constructed around a
synthetic reference-loss process.  Archive labels are *observed empirical* audit
errors, never the simulator's latent flip probability.  The Gaussian HMM is fit
on a disjoint prefix and held fixed.  Delayed evidence is replayed at its original
issue time; filtering is exact under that fitted model.  Online covariance
moments are a finite-archive PSD plug-in, not an exact NIW parameter posterior.
Actual logistic SGD runs on worker copies and model deployment is delayed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs"))
from da_rf_fusion import solve_capped_fusion


@dataclass(frozen=True)
class Config:
    slots: int = 560
    jobs: int = 16
    audit_jobs: int = 64
    delay_min: int = 2
    delay_max: int = 6
    lag: int = 128
    archive: int = 160
    beta: float = 0.985
    kernel_other: float = 0.55
    quality_power: float = 1.0
    tau: float = 0.003
    risk_scale: float = 1.0
    history_weight: float = 0.0
    max_weight: float = 0.8
    train_steps: int = 32
    train_lr: float = 0.65
    train_slots: int = 3
    deployment_delay: int = 5
    train_capacity_loss: int = 5
    step_cost: float = 1.5
    horizon: int = 24
    cooldown: int = 28
    max_updates: int = 9
    train_examples: int = 256
    prior_mass: float = 8.0
    value_margin: float = 12.0
    source_sigma: float = 0.22
    persistent_gate: int = 2
    drift: bool = True


TRUE_P = np.full((3, 3), 0.04)
np.fill_diagonal(TRUE_P, 0.92)
TRUE_STAKES = np.array([4.0, 0.7, 0.7])
TRUE_CONTEXT = np.full((3, 3), 0.24)
np.fill_diagonal(TRUE_CONTEXT, 0.52)
METHODS = ("bm_joint", "context_joint", "bm_diagonal", "context_diagonal", "bm_context_only_belief",
           "original_c", "observed_loss", "periodic", "random", "frozen")


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sigmoid(v):
    return 1 / (1 + np.exp(-np.clip(v, -35, 35)))


def sgd(weight, x, y, cfg):
    """Actual gradients; no desired sign or latent relation is passed in."""
    w = float(weight)
    for _ in range(cfg.train_steps):
        gradient = np.mean((sigmoid(w * x) - y) * x) + 0.003 * w
        w = float(np.clip(w - cfg.train_lr * gradient, -6, 6))
    return w


def make_covariance(sigma, rho=0.86):
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        matrix = np.eye(3) * sigma ** 2
        matrix[a, b] = matrix[b, a] = rho * sigma ** 2
        out.append(matrix)
    return np.array(out)


def hmm_smooth(log_likelihood, transition, initial):
    """Exact forward/backward recursion conditional on supplied fitted params."""
    n = len(log_likelihood)
    alpha = np.empty((n, 3))
    likelihood = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
    prior = np.asarray(initial)
    for t in range(n):
        alpha[t] = prior * likelihood[t]
        alpha[t] /= max(alpha[t].sum(), 1e-300)
        prior = alpha[t] @ transition
    backward = np.ones((n, 3))
    for t in range(n - 2, -1, -1):
        backward[t] = transition @ (likelihood[t + 1] * backward[t + 1])
        backward[t] /= max(backward[t].sum(), 1e-300)
    gamma = alpha * backward
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, alpha, likelihood, backward


def gaussian_log_probability(residual, covariance):
    d = len(residual)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Emission covariance must be positive definite")
    return -0.5 * (d * np.log(2 * np.pi) + logdet + residual @ np.linalg.solve(covariance, residual))


def fit_prefix_hmm(cfg, seed=92001, windows=900, iterations=18):
    """Unsupervised offline EM; latent prefix states never enter fitting."""
    rng = np.random.default_rng(seed)
    true_cov = make_covariance(cfg.source_sigma)
    context, residual, stake = [], [], []
    z = int(rng.integers(3))
    for t in range(windows):
        z = int(rng.choice(3, p=TRUE_P[z]))
        c = int(rng.choice(3, p=TRUE_CONTEXT[z]))
        p = float(rng.uniform(0.1, 0.9))
        observed_loss = float(rng.binomial(cfg.audit_jobs, p) / cfg.audit_jobs)
        error = rng.multivariate_normal(np.zeros(3), true_cov[z]) + p - observed_loss
        context.append(c)
        residual.append(error)
        stake.append(float(TRUE_STAKES[z]))
    context, residual, stake = np.array(context), np.array(residual), np.array(stake)
    # Initialization is a declared covariance-family assumption, not test tuning.
    cov = make_covariance(cfg.source_sigma, rho=0.5) + 0.002 * np.ones((3, 3, 3))
    transition = np.full((3, 3), 0.075)
    np.fill_diagonal(transition, 0.85)
    emissions = np.full((3, 3), 0.25)
    np.fill_diagonal(emissions, 0.5)
    initial = np.full(3, 1 / 3)
    likelihood_history = []
    for iteration in range(iterations):
        log_lik = np.stack([[gaussian_log_probability(row, cov[z]) for z in range(3)]
                            for row in residual]) + np.log(emissions[:, context].T)
        gamma, alpha, likelihood, backward = hmm_smooth(log_lik, transition, initial)
        counts = np.zeros((3, 3))
        for t in range(windows - 1):
            pair = alpha[t, :, None] * transition * (likelihood[t + 1] * backward[t + 1])[None, :]
            counts += pair / pair.sum()
        transition = (counts + 1) / (counts.sum(axis=1, keepdims=True) + 3)
        cov = np.einsum("tz,ti,tj->zij", gamma, residual, residual) / gamma.sum(axis=0)[:, None, None]
        cov += np.eye(3)[None, :, :] * 1e-5
        for z in range(3):
            for c in range(3):
                emissions[z, c] = gamma[context == c, z].sum() + 1
        emissions /= emissions.sum(axis=1, keepdims=True)
        initial = gamma[0]
        likelihood_history.append(float(log_lik.max(axis=1).sum()))
    # Stake estimates use only observed prefix rewards and inferred states.
    mean_stake = (gamma.T @ stake) / gamma.sum(axis=0)
    return {"seed": seed, "windows": windows, "em_iterations": iterations,
            "transition": transition, "emissions": emissions, "covariance": cov,
            "initial": np.full(3, 1 / 3), "mean_stake": mean_stake,
            "prefix_context": context, "prefix_residual": residual,
            "prefix_stake": stake, "prefix_gamma": gamma,
            "emission_parameters_frozen_online": True,
            "online_covariance_is_adaptive_psd_plugin_not_parameter_posterior": True}


def fit_recovery_model(cfg, hmm, seed=92002, repeats=35):
    """Disjoint full-horizon paired interventions with real worker SGD.

    Values are observed service-return differences.  Simulated continuation
    regimes are sampled from the *prefix-fitted* HMM, not test-state truth.  The
    regression uses sign and reference loss.  Its slopes are empirical value
    sensitivities; they are not causal probability or guaranteed error bounds.
    """
    rng = np.random.default_rng(seed)
    horizon = cfg.horizon
    p_grid = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    rows = []
    coefficients = np.zeros((3, 2, 2))
    validation = []
    for z0 in range(3):
        for s, sign in enumerate((1, -1)):
            points, gains = [], []
            for p in p_grid:
                for rep in range(repeats):
                    train_x = rng.normal(size=cfg.train_examples)
                    train_y = (train_x >= 0).astype(float)
                    train_y = np.logical_xor(train_y, rng.random(len(train_x)) < p).astype(float)
                    before = sign * 4.0
                    after = sgd(before, train_x, train_y, cfg)
                    x = rng.normal(size=(horizon, cfg.jobs))
                    y = np.logical_xor(x >= 0, rng.random(x.shape) < p)
                    z = z0
                    gain = -cfg.train_steps * cfg.step_cost
                    for t in range(horizon):
                        # Slot zero is the known current component h. Future
                        # transitions occur only after pricing the current slot.
                        stake = float(hmm["mean_stake"][z])
                        # Both branches share physical jobs, labels and regime.
                        base = np.sum((before * x[t] >= 0) == y[t]) * stake
                        used_weight = after if t >= cfg.deployment_delay else before
                        served = cfg.jobs - cfg.train_capacity_loss if t < cfg.train_slots else cfg.jobs
                        active = np.sum((used_weight * x[t, :served] >= 0) == y[t, :served]) * stake
                        gain += active - base
                        z = int(rng.choice(3, p=hmm["transition"][z]))
                    # Same-orientation updates have no classification benefit.
                    # A hinge model respects this observed prefix mechanism.
                    hinge = max(0.0, p - 0.5) if sign > 0 else max(0.0, 0.5 - p)
                    points.append([1.0, hinge])
                    gains.append(gain)
                    rows.append({"z": z0, "model_sign": sign, "p": float(p),
                                 "repeat": rep, "paired_gain": gain,
                                 "updated_sign": float(np.sign(after))})
            # Train on 4/5 rollouts; fifth is never fit (group-level split).
            points, gains = np.asarray(points), np.asarray(gains)
            train = np.arange(len(gains)) % 5 != 4
            coeff = np.linalg.solve(points[train].T @ points[train] + 1e-6 * np.eye(2),
                                    points[train].T @ gains[train])
            coefficients[z0, s] = coeff
            errors = points[~train] @ coeff - gains[~train]
            validation.append({"z": z0, "model_sign": sign,
                               "mae": float(np.mean(np.abs(errors))),
                               "max_abs_error": float(np.max(np.abs(errors)))})
    return {"seed": seed, "rollouts": rows, "coefficients": coefficients,
            "validation": validation, "horizon": horizon,
            "exact_service_oracle": False,
            "feature": "[1,max(0,sign*(reference_loss-0.5))]",
            "slope_interpretation": "maximum conditional-current-component full-horizon gain sensitivity; squared expected slope is a consequential surrogate, not expected pathwise squared slope",
            "state_continuations_sample_prefix_fitted_hmm": True,
            "unverified_bound": "heldout errors are empirical, not simultaneous coverage"}


def generate_world(cfg, seed):
    rng = np.random.default_rng(seed)
    n = cfg.slots
    x = rng.normal(size=(n, cfg.jobs))
    # Five alternating gradual relation changes, with random phase/jitter.
    phase = int(rng.integers(2, 13))
    relation = np.zeros(n)
    p = np.full(n, 0.2)
    transitions = []
    for t in range(n):
        block = max(0, (t - phase) // 95)
        target = 0.2 if not cfg.drift or block % 2 == 0 else 0.8
        if t:
            p[t] = p[t - 1] + np.clip(target - p[t - 1], -0.045, 0.045)
        relation[t] = block % 2 if cfg.drift else 0
        if t > 0 and relation[t] != relation[t - 1]:
            transitions.append(t)
    y = np.logical_xor(x >= 0, rng.random(x.shape) < p[:, None])
    audit_flips = rng.random((n, cfg.audit_jobs)) < p[:, None]
    observed_loss = audit_flips.mean(axis=1)
    audit_delay = rng.integers(cfg.delay_min, cfg.delay_max + 1, n)
    label_delay = 3
    z = np.empty(n, dtype=int)
    context = np.empty(n, dtype=int)
    stake = np.empty(n)
    z[0] = int(rng.integers(3))
    source = np.empty((n, 3))
    quality = np.empty((n, 3))
    masks = np.ones((n, 3), dtype=bool)
    scale = np.ones((n, 3))
    population_cov = np.empty((n, 3, 3))
    for t in range(n):
        if t:
            z[t] = int(rng.choice(3, p=TRUE_P[z[t - 1]]))
        context[t] = int(rng.choice(3, p=TRUE_CONTEXT[z[t]]))
        stake[t] = TRUE_STAKES[z[t]]
        rho = 0.68 if t < n // 2 else 0.9
        cov = make_covariance(cfg.source_sigma, rho)[z[t]]
        # Visible physical instrument quality shock; no forecast errors in q.
        if 210 <= t < 280:
            scale[t, 1] = 2.6
        cov = scale[t, :, None] * cov * scale[t, None, :]
        population_cov[t] = cov + p[t] * (1 - p[t]) / cfg.audit_jobs * np.ones((3, 3))
        source[t] = p[t] + rng.multivariate_normal(np.zeros(3), cov)
        replicates = rng.normal(size=(8, 3)) * cfg.source_sigma * scale[t]
        quality[t] = 1 / (np.var(replicates, axis=0, ddof=1) + 0.02)
        if rng.random() < 0.16:
            masks[t, int(rng.integers(3))] = False
        if 150 <= t < 170:
            masks[t, 0] = False
        if 360 <= t < 380:
            masks[t, 2] = False
        if t in (205, 206, 405):
            masks[t] = False
    source[~masks] = np.nan
    return {"seed": seed, "x": x, "y": y, "p_truth_scoring_only": p,
            "audit_loss": observed_loss, "audit_delay": audit_delay,
            "training_label_delay": label_delay, "z_truth_scoring_only": z,
            "context": context, "stake": stake, "source": source,
            "mask": masks, "quality": quality, "scale": scale,
            "population_cov_truth_scoring_only": population_cov,
            "relation_transition_scoring_only": transitions}


class DelayedFilter:
    def __init__(self, hmm, cfg):
        self.hmm, self.cfg = hmm, cfg
        self.context = []
        self.residual = {}
        self.arrival = {}
        self.start = 0
        self.initial = np.asarray(hmm["initial"]).copy()
        self.last_gamma = np.empty((0, 3))
        self.last_alpha = np.empty((0, 3))
        self.discarded_labels = 0
        self.audit = []
        self.cache = {}
        for bit in range(1, 8):
            active = np.array([i for i in range(3) if bit & (1 << i)])
            covs = hmm["covariance"][:, active[:, None], active]
            inverse = np.linalg.inv(covs)
            logdet = np.linalg.slogdet(covs)[1]
            self.cache[bit] = (active, inverse, logdet)

    def issue(self, t, context, labels):
        self.context.append(int(context))
        # A label arriving now can alter only its original likelihood factor.
        for original, residual, mask, scale, arrival in labels:
            if arrival > t or original >= t:
                raise AssertionError("Future or current outcome leaked")
            if original < self.start:
                self.discarded_labels += 1
                continue
            self.residual[original] = (residual.copy(), mask.copy(), scale.copy())
            self.arrival[original] = int(arrival)
            self.audit.append((int(t), int(original), int(arrival)))
        # Marginalize old factors into the correctly filtered boundary message.
        desired_start = max(0, t - self.cfg.lag + 1)
        if desired_start > self.start:
            removed = desired_start - self.start
            self.initial = self.last_alpha[removed - 1] @ self.hmm["transition"]
            self.start = desired_start
        n = t - self.start + 1
        log_lik = np.log(self.hmm["emissions"][:, self.context[self.start:t + 1]].T)
        for original, (error, mask, scale) in self.residual.items():
            if original < self.start:
                continue
            bit = sum(1 << i for i in np.flatnonzero(mask))
            if bit == 0:
                continue
            active, inverse, logdet = self.cache[bit]
            standardized = error[active] / scale[active]
            quadratic = np.einsum("i,zij,j->z", standardized, inverse, standardized)
            log_lik[original - self.start] += -0.5 * (len(active) * np.log(2 * np.pi) + logdet + quadratic)
        gamma, alpha, _, _ = hmm_smooth(log_lik, self.hmm["transition"], self.initial)
        self.last_gamma, self.last_alpha = gamma, alpha
        # When records leave the replay buffer, their last smoothed state is
        # frozen for empirical archive use; filtering boundary remains exact.
        return alpha[-1].copy(), gamma


def learn_moments(t, active, world, state, hmm, cfg):
    eligible = np.array([r for r in state.residual
                         if r >= max(0, t - cfg.archive) and np.all(world["mask"][r, active])], dtype=int)
    prior = hmm["covariance"][:, active[:, None], active].copy()
    moments = prior.copy()
    supports = np.zeros(3)
    if len(eligible):
        # Records outside replay are omitted deliberately: no stale smoothed
        # responsibility is presented as current conditioned state evidence.
        eligible = eligible[eligible >= state.start]
    if len(eligible):
        errors = world["source"][eligible][:, active] - world["audit_loss"][eligible, None]
        kernels = np.where(world["context"][eligible] == world["context"][t], 1.0, cfg.kernel_other)
        ages = cfg.beta ** (t - eligible)
        responsibilities = state.last_gamma[eligible - state.start]
        a = kernels[:, None] * ages[:, None] * responsibilities
        supports = a.sum(axis=0)
        numerator = cfg.prior_mass * prior + np.einsum("rz,ri,rj->zij", a, errors, errors)
        moments = numerator / (cfg.prior_mass + supports)[:, None, None]
    return moments, supports, eligible


def fusion_sequences(world, hmm, calibration, cfg):
    n = cfg.slots
    state = DelayedFilter(hmm, cfg)
    methods = ("bm_joint", "context_joint", "bm_diagonal", "context_diagonal", "original_c", "bm_context_only_belief")
    weights = {m: np.zeros((n, 2, 3)) for m in methods}
    predictions = {m: np.full((n, 2), np.nan) for m in methods}
    moment_errors, posterior_brier, supports, min_eigs, solver_failures = [], [], [], [], 0
    moment_updates, errors_by_slot = [], []
    posteriors = np.zeros((n, 3))
    context_only = np.zeros((n, 3))
    last = {m: np.full((2, 3), 1 / 3) for m in methods}
    slopes = calibration["coefficients"][:, :, 1]
    arrived = {}
    for r in range(n):
        arrival = r + 1 + int(world["audit_delay"][r])
        arrived.setdefault(arrival, []).append(r)
    for t in range(n):
        labels = []
        for r in arrived.get(t, []):
            error = world["source"][r] - world["audit_loss"][r]
            labels.append((r, error, world["mask"][r], world["scale"][r], t))
        posterior, _ = state.issue(t, world["context"][t], labels)
        posteriors[t] = posterior
        context_only[t] = hmm["initial"] * hmm["emissions"][:, world["context"][t]]
        context_only[t] /= context_only[t].sum()
        truth = np.eye(3)[world["z_truth_scoring_only"][t]]
        posterior_brier.append(float(np.sum((posterior - truth) ** 2)))
        active = np.flatnonzero(world["mask"][t])
        if not len(active):
            continue
        rhat, support, eligible = learn_moments(t, active, world, state, hmm, cfg)
        supports.append(support.sum())
        r = np.einsum("z,zij->ij", posterior, rhat)
        prior_r = np.einsum("z,zij->ij", posterior, hmm["covariance"][:, active[:, None], active])
        moment_updates.append(float(np.linalg.norm(r - prior_r, ord="fro")))
        actual = world["population_cov_truth_scoring_only"][t][np.ix_(active, active)]
        moment_errors.append(float(np.linalg.norm(r - actual, ord="fro")))
        errors_by_slot.append((t, moment_errors[-1]))
        q = world["quality"][t, active] ** cfg.quality_power
        q = q / q.sum()
        for model_sign in range(2):
            m = np.einsum("z,z,zij->ij", posterior, slopes[:, model_sign] ** 2, rhat)
            # Normalize trace to R's trace to isolate matrix shape; B² scale is
            # not permitted to create a stronger risk coefficient in this test.
            m *= np.trace(r) / max(np.trace(m), 1e-12)
            min_eigs.append(float(np.linalg.eigvalsh(m).min()))
            matrices = {"bm_joint": m, "context_joint": r,
                        "bm_diagonal": np.diag(np.diag(m)),
                        "context_diagonal": np.diag(np.diag(r))}
            ablated_m = np.einsum("z,z,zij->ij", context_only[t], slopes[:, model_sign] ** 2, rhat)
            ablated_m *= np.trace(r) / max(np.trace(ablated_m), 1e-12)
            matrices["bm_context_only_belief"] = ablated_m
            for method in methods:
                if method == "original_c":
                    # Aligned component control: scalar source-MSE quality;
                    # not a reproduction of the disputed original execution.
                    modified_q = q * np.exp(-np.diag(r) / 0.045)
                    modified_q /= modified_q.sum()
                    matrix = np.zeros_like(r)
                else:
                    modified_q, matrix = q, matrices[method]
                anchor = last[method][model_sign, active]
                anchor = anchor / anchor.sum() if anchor.sum() else np.full(len(active), 1 / len(active))
                result = solve_capped_fusion(modified_q, matrix, anchor, max(cfg.max_weight, 1 / len(active)),
                                             cfg.tau, cfg.risk_scale, cfg.history_weight,
                                             tolerance=1e-8, max_iterations=50)
                if not result.converged:
                    solver_failures += 1
                weights[method][t, model_sign, active] = result.weights
                last[method][model_sign] = weights[method][t, model_sign]
                predictions[method][t, model_sign] = float(np.clip(result.weights @ world["source"][t, active], 0, 1))
    return {"weights": weights, "predictions": predictions, "posterior": posteriors,
            "context_only_posterior": context_only,
            "audit": state.audit, "discarded_labels": state.discarded_labels,
            "moment_frobenius_error": float(np.mean(moment_errors)),
            "moment_change_from_prefix": float(np.mean(moment_updates)),
            "moment_error_first50": float(np.mean([e for t, e in errors_by_slot if t < 50])),
            "moment_error_last100": float(np.mean([e for t, e in errors_by_slot if t >= cfg.slots - 100])),
            "posterior_brier": float(np.mean(posterior_brier)),
            "mean_effective_support": float(np.mean(supports)),
            "minimum_eigenvalue": float(np.min(min_eigs)), "solver_failures": solver_failures,
            "final_hmm": state}


def make_schedule(cfg, seed, method, count=None, eligible=None):
    available = np.arange(cfg.train_examples // cfg.jobs + 4, cfg.slots - cfg.deployment_delay)
    if eligible is not None:
        available = available[eligible[available]]
    if count is None:
        count = cfg.max_updates
    if method == "periodic":
        proposed = np.linspace(30, cfg.slots - 18, count, dtype=int)
        if eligible is None:
            return set(proposed.tolist())
        selected = []
        for t in proposed:
            candidates = available[np.argsort(np.abs(available - t))]
            for candidate in candidates:
                if all(abs(int(candidate) - k) >= cfg.cooldown for k in selected):
                    selected.append(int(candidate))
                    break
        return set(selected)
    rng = np.random.default_rng(seed + 810000)
    # A scheduled start respects the common cooldown in advance.
    selected = []
    for t in rng.permutation(available):
        if all(abs(int(t) - k) >= cfg.cooldown for k in selected):
            selected.append(int(t))
            if len(selected) == count:
                break
    return set(selected)


def run_arm(world, shared, calibration, cfg, method, matched_count=None):
    n = cfg.slots
    weight, pending, updates, last_update = 4.0, None, 0, -10**9
    deployment_times, update_times, oldsign = [], [], 1
    model_weights, served_accuracy, loss, net, costs, served_correct, occupied = [], [], [], [], [], [], []
    actions = np.zeros(n, dtype=bool)
    value_estimates = np.full(n, np.nan)
    reasons = {"budget": 0, "busy": 0, "missing": 0, "insufficient_labels": 0}
    schedule = make_schedule(cfg, world["seed"], "periodic" if "periodic" in method else "random", matched_count,
                             world["mask"].any(axis=1) if matched_count is not None else None) if ("periodic" in method or "random" in method) else None
    coefficients = calibration["coefficients"]
    latest_observed_audit = None
    value_votes = []
    launch_audit = []
    for t in range(n):
        if pending is not None and pending["deploy"] == t:
            weight = pending["weight"]
            deployment_times.append(t)
            pending = None
        completed = np.flatnonzero(np.arange(t) + 1 + world["audit_delay"][:t] <= t)
        if len(completed):
            latest_observed_audit = float(world["audit_loss"][completed[-1]])
        sign_index = 0 if weight >= 0 else 1
        if method in shared["predictions"]:
            p_hat = shared["predictions"][method][t, sign_index]
        elif method == "observed_loss":
            p_hat = latest_observed_audit if latest_observed_audit is not None else np.nan
        else:
            p_hat = np.nan
        expected_value = np.nan
        if np.isfinite(p_hat):
            hinge = max(0.0, p_hat - 0.5) if sign_index == 0 else max(0.0, 0.5 - p_hat)
            posterior = shared["context_only_posterior"][t] if method == "bm_context_only_belief" else shared["posterior"][t]
            expected_value = float(posterior @ (coefficients[:, sign_index, 0] + coefficients[:, sign_index, 1] * hinge))
            value_estimates[t] = expected_value
        value_votes.append(bool(np.isfinite(expected_value) and expected_value > cfg.value_margin))
        if method == "frozen":
            trigger = False
        elif schedule is not None:
            trigger = t in schedule
        else:
            trigger = value_votes[-1] and sum(value_votes[-3:]) >= cfg.persistent_gate
        if trigger:
            if updates >= (matched_count if matched_count is not None else cfg.max_updates):
                reasons["budget"] += 1
            elif pending is not None or t - last_update < cfg.cooldown:
                reasons["busy"] += 1
            elif not np.any(world["mask"][t]):
                reasons["missing"] += 1
            elif t + cfg.deployment_delay >= n:
                reasons["busy"] += 1
            else:
                end = t - world["training_label_delay"]
                if end < cfg.train_examples // cfg.jobs:
                    reasons["insufficient_labels"] += 1
                else:
                    x = world["x"][:end].ravel()[-cfg.train_examples:]
                    y = world["y"][:end].ravel()[-cfg.train_examples:].astype(float)
                    copied = sgd(weight, x, y, cfg)
                    pending = {"weight": copied, "deploy": t + cfg.deployment_delay,
                               "train_end": t + cfg.train_slots, "start": t}
                    updates += 1
                    last_update = t
                    update_times.append(t)
                    actions[t] = True
                    launch_audit.append({"launch_slot": t, "max_training_origin": end - 1,
                                         "max_training_arrival": end - 1 + 1 + world["training_label_delay"],
                                         "deployment_slot": t + cfg.deployment_delay,
                                         "actual_steps": cfg.train_steps, "copied_weight": copied})
        busy = pending is not None and t < pending["train_end"]
        served = cfg.jobs - cfg.train_capacity_loss if busy else cfg.jobs
        prediction = weight * world["x"][t] >= 0
        correct = prediction == world["y"][t]
        cost = cfg.train_steps * cfg.step_cost if actions[t] else 0.0
        reward = float(np.sum(correct[:served]) * world["stake"][t] - cost)
        model_weights.append(weight)
        served_accuracy.append(float(np.mean(correct)))
        loss.append(float(1 - np.mean(correct)))
        net.append(reward)
        costs.append(cost)
        served_correct.append(int(np.sum(correct[:served])))
        occupied.append(cfg.jobs - served)
    # Retrospective recovery: sign restored after each completed relation flip.
    recovery, recovered = [], 0
    warray = np.asarray(model_weights)
    for transition in world["relation_transition_scoring_only"]:
        end = min(n, transition + 70)
        target_negative = world["p_truth_scoring_only"][min(transition + 20, n - 1)] > 0.5
        times = np.flatnonzero((warray[transition:end] < 0) == target_negative)
        if len(times):
            recovery.append(int(times[0]))
            recovered += 1
        else:
            recovery.append(70)
    return {"method": method, "net_return": float(np.sum(net)),
            "mean_accuracy": float(np.mean(served_accuracy)),
            "tail_accuracy": float(np.mean(served_accuracy[-100:])),
            "training_steps": updates * cfg.train_steps, "training_cost": float(np.sum(costs)),
            "updates": updates, "deployments": len(deployment_times),
            "training_occupancy_job_loss": int(np.sum(occupied)),
            "served_correct_jobs": int(np.sum(served_correct)),
            "accuracy_interpretation": "potential classification accuracy over all jobs before occupancy drops",
            "recovery_interpretation": "model-sign restoration after synthetic relation transition, capped at70slots",
            "mean_recovery_delay": float(np.mean(recovery)) if recovery else None,
            "recovered_transitions": recovered, "transitions": len(recovery),
            "actions": actions, "update_times": update_times, "deployment_times": deployment_times,
            "model_weights": warray, "per_slot_net": net,
            "per_slot_accuracy": served_accuracy, "value_estimates": value_estimates,
            "block_reasons": reasons, "launch_audit": launch_audit}


def run_seed(seed, cfg, hmm, calibration, include_matched=True):
    world = generate_world(cfg, seed)
    shared = fusion_sequences(world, hmm, calibration, cfg)
    arms = {m: run_arm(world, shared, calibration, cfg, m) for m in METHODS}
    if include_matched:
        for m in ("periodic_matched", "random_matched"):
            arms[m] = run_arm(world, shared, calibration, cfg, m, arms["bm_joint"]["updates"])
            assert arms[m]["updates"] == arms["bm_joint"]["updates"]
            assert arms[m]["training_steps"] == arms["bm_joint"]["training_steps"]
            assert arms[m]["training_cost"] == arms["bm_joint"]["training_cost"]
    for arm in arms.values():
        assert arm["training_cost"] == arm["training_steps"] * cfg.step_cost
        assert arm["training_occupancy_job_loss"] == arm["updates"] * cfg.train_slots * cfg.train_capacity_loss
        assert all(a["max_training_arrival"] <= a["launch_slot"] and
                   a["deployment_slot"] == a["launch_slot"] + cfg.deployment_delay for a in arm["launch_audit"])
    # Actual action disagreements; local advantage uses a fork of the complete
    # worker update under the same future physical stream.  Fork labels never
    # feed any policy.  Model-state differences are recorded, not suppressed.
    disagreement = {}
    for control in ("context_joint", "bm_diagonal", "original_c"):
        a, b = arms["bm_joint"], arms[control]
        changes = np.flatnonzero(a["actions"] != b["actions"])
        beneficial, harmful, neutral = 0, 0, 0
        differences = []
        for t in changes:
            end = t - world["training_label_delay"]
            if end < cfg.train_examples // cfg.jobs:
                continue
            current = a["model_weights"][t]
            xtrain = world["x"][:end].ravel()[-cfg.train_examples:]
            ytrain = world["y"][:end].ravel()[-cfg.train_examples:].astype(float)
            after = sgd(current, xtrain, ytrain, cfg)
            true_gain = -cfg.train_steps * cfg.step_cost
            for h in range(min(cfg.horizon, cfg.slots - t)):
                x, y, stake = world["x"][t + h], world["y"][t + h], world["stake"][t + h]
                before_return = np.sum((current * x >= 0) == y) * stake
                copied = after if h >= cfg.deployment_delay else current
                served = cfg.jobs - cfg.train_capacity_loss if h < cfg.train_slots else cfg.jobs
                after_return = np.sum((copied * x[:served] >= 0) == y[:served]) * stake
                true_gain += after_return - before_return
            direction = 1 if a["actions"][t] else -1
            difference = direction * true_gain
            differences.append(float(difference))
            beneficial += difference > 1e-8
            harmful += difference < -1e-8
            neutral += abs(difference) <= 1e-8
        disagreement[control] = {"different_update_slots": len(changes),
                                 "beneficial_local_fork": int(beneficial),
                                 "harmful_local_fork": int(harmful),
                                 "neutral_local_fork": int(neutral),
                                 "sum_local_fork_advantage": float(np.sum(differences)),
                                 "not_additive_to_long_run_return": True}
    diagnostics = {k: shared[k] for k in ("moment_frobenius_error", "posterior_brier",
                                         "mean_effective_support", "minimum_eigenvalue",
                                         "solver_failures", "discarded_labels",
                                         "moment_change_from_prefix", "moment_error_first50", "moment_error_last100")}
    diagnostics.update({"delay_audit_records": len(shared["audit"]),
                        "max_delay_used": max(t - r for t, r, arrival in shared["audit"]),
                        "all_missing_slots": int(np.sum(~world["mask"].any(axis=1))),
                        "partial_missing_slots": int(np.sum(world["mask"].sum(axis=1) == 2))})
    model_signs = (arms["bm_joint"]["model_weights"] < 0).astype(int)
    primary_weights = shared["weights"]["bm_joint"][np.arange(cfg.slots), model_signs]
    return {"seed": seed, "arms": arms, "diagnostics": diagnostics,
            "executed_primary_weights": primary_weights,
            "action_disagreements": disagreement,
            "world_hash": hashlib.sha256(world["x"].tobytes() + world["y"].tobytes() + world["source"].tobytes()).hexdigest()}


def summarize(rows):
    methods = list(rows[0]["arms"])
    summary = {}
    scalar_metrics = ("net_return", "mean_accuracy", "tail_accuracy", "training_steps", "training_cost",
                      "updates", "deployments", "mean_recovery_delay", "recovered_transitions")
    for method in methods:
        summary[method] = {}
        for metric in scalar_metrics:
            values = np.array([r["arms"][method][metric] for r in rows], dtype=float)
            summary[method][metric] = {"mean": float(values.mean()),
                                       "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    contrasts = {}
    main = np.array([r["arms"]["bm_joint"]["net_return"] for r in rows])
    for method in methods:
        if method == "bm_joint":
            continue
        values = main - np.array([r["arms"][method]["net_return"] for r in rows])
        se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        # Descriptive paired normal intervals, not distribution-free coverage.
        contrasts[method] = {"mean_gain": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0,
                             "paired_95_normal_interval": [float(values.mean() - 1.96 * se), float(values.mean() + 1.96 * se)],
                             "positive_seeds": int(np.sum(values > 0)), "seed_count": len(values),
                             "per_seed": values.tolist()}
    return {"methods": summary, "paired_net_return": contrasts}


def run_payload(payload):
    """Pure per-seed worker; no parameter selection or shared mutations."""
    seed, cfg, hmm, calibration = payload
    return run_seed(seed, cfg, hmm, calibration, include_matched=False)


def causal_tests(cfg, hmm, calibration):
    # Late factor replay must equal full-history replay while within the lag.
    short = replace(cfg, slots=180, lag=40)
    world = generate_world(short, 91999)
    filter_a = DelayedFilter(hmm, short)
    filter_b = DelayedFilter(hmm, short)
    differences = []
    all_labels = {}
    for t in range(short.slots):
        labels = []
        for r in range(t):
            if r + 1 + world["audit_delay"][r] == t:
                labels.append((r, world["source"][r] - world["audit_loss"][r],
                               world["mask"][r], world["scale"][r], t))
                all_labels[r] = labels[-1]
        posterior, gamma = filter_a.issue(t, world["context"][t], labels)
        # Independent batch recursion from saved issue-time factors.
        loglik = np.log(hmm["emissions"][:, world["context"][:t + 1]].T)
        for r, error, mask, scale, arrival in all_labels.values():
            active = np.flatnonzero(mask)
            if not len(active):
                continue
            for z in range(3):
                loglik[r, z] += gaussian_log_probability(error[active] / scale[active],
                                                        hmm["covariance"][z][np.ix_(active, active)])
        _, alpha, _, _ = hmm_smooth(loglik, hmm["transition"], hmm["initial"])
        differences.append(float(np.max(np.abs(posterior - alpha[-1]))))
    assert max(differences) < 1e-12
    # Future labels cannot change a decision-time posterior or matrix.
    assert all(t >= arrival and t > r for t, r, arrival in filter_a.audit)
    # Worker SGD can recover a genuine sign reversal; no target sign in SGD.
    rng = np.random.default_rng(91998)
    x = rng.normal(size=cfg.train_examples)
    y = (x < 0).astype(float)
    trained = sgd(4, x, y, cfg)
    assert trained < 0
    # Changing all future observations and responses must leave earlier weights,
    # posteriors and complete-loop actions unchanged.
    short = replace(cfg, slots=75, lag=40)
    world_a = generate_world(short, 91997)
    world_b = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in world_a.items()}
    cutoff = 35
    world_b["source"][cutoff:] = 0.99
    world_b["audit_loss"][cutoff:] = 0.0
    world_b["y"][cutoff:] = ~world_b["y"][cutoff:]
    world_b["context"][cutoff:] = 2
    world_b["z_truth_scoring_only"][cutoff:] = 1
    world_b["p_truth_scoring_only"][cutoff:] = 0.0
    world_b["stake"][cutoff:] = 100.0
    shared_a = fusion_sequences(world_a, hmm, calibration, short)
    shared_b = fusion_sequences(world_b, hmm, calibration, short)
    np.testing.assert_array_equal(shared_a["posterior"][:cutoff], shared_b["posterior"][:cutoff])
    for method in ("bm_joint", "context_joint", "observed_loss"):
        arm_a = run_arm(world_a, shared_a, calibration, short, method)
        arm_b = run_arm(world_b, shared_b, calibration, short, method)
        np.testing.assert_array_equal(arm_a["actions"][:cutoff], arm_b["actions"][:cutoff])
        np.testing.assert_array_equal(arm_a["model_weights"][:cutoff], arm_b["model_weights"][:cutoff])
    return {"late_factor_replay_max_abs_error": max(differences),
            "issuance_arrival_order_passed": True, "actual_sgd_sign_recovery_passed": True,
            "true_test_regime_never_passed_to_filter": True,
            "online_archive_uses_observed_empirical_audit_loss": True,
            "checkpoint_trim_test_crossed_lag": True,
            "future_data_mutation_action_and_posterior_invariance_passed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=93001)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "bayes_results.json")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--no-drift", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cfg = Config(drift=not args.no_drift)
    start = time.time()
    hmm = fit_prefix_hmm(cfg)
    calibration = fit_recovery_model(cfg, hmm)
    tests = causal_tests(cfg, hmm, calibration)
    rows = []
    for i in range(args.seeds):
        row = run_seed(args.seed_start + i, cfg, hmm, calibration)
        rows.append(row)
        gains = {m: round(row["arms"]["bm_joint"]["net_return"] - row["arms"][m]["net_return"], 2)
                 for m in ("context_joint", "bm_diagonal", "observed_loss", "frozen")}
        print(f"seed={row['seed']} elapsed={time.time()-start:.1f}s gains={gains}", flush=True)
    sensitivity = {}
    if args.sensitivity:
        # Every sweep uses the same independent seed pairing, with no selection
        # of the best test setting. Calibration/HMM parameters are held fixed.
        sweep_cfgs = {"quality_power": [0.0, 0.5, 1.0, 2.0],
                      "risk_scale": [0.0, 0.5, 1.0, 2.0],
                      "history_weight": [0.0, 0.01, 0.05],
                      "beta": [0.90, 0.97, 0.985, 1.0]}
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
          for field, values in sweep_cfgs.items():
            sensitivity[field] = {}
            for value in values:
                altered = replace(cfg, **{field: value})
                if altered == cfg:
                    # Reuse exact default executions rather than retest them.
                    altered_rows = [{**r, "arms": {m: a for m, a in r["arms"].items() if not m.endswith("_matched")}}
                                    for r in rows[:min(10, args.seeds)]]
                else:
                    payloads = [(args.seed_start + i, altered, hmm, calibration) for i in range(min(10, args.seeds))]
                    altered_rows = list(pool.map(run_payload, payloads))
                comparison = []
                for primary, altered_row in zip(rows, altered_rows):
                    assert primary["seed"] == altered_row["seed"] and primary["world_hash"] == altered_row["world_hash"]
                    base_arm, alt_arm = primary["arms"]["bm_joint"], altered_row["arms"]["bm_joint"]
                    weight_difference = np.linalg.norm(np.asarray(altered_row["executed_primary_weights"]) -
                                                       np.asarray(primary["executed_primary_weights"]), axis=1)
                    comparison.append({"seed": primary["seed"], "net_return": alt_arm["net_return"],
                                       "net_change_from_default": alt_arm["net_return"] - base_arm["net_return"],
                                       "changed_update_slots": int(np.sum(alt_arm["actions"] != base_arm["actions"])),
                                       "changed_action_fraction": float(np.mean(alt_arm["actions"] != base_arm["actions"])),
                                       "mean_executed_weight_l2_change": float(np.mean(weight_difference)),
                                       "max_executed_weight_l2_change": float(np.max(weight_difference)),
                                       "updates": alt_arm["updates"], "steps": alt_arm["training_steps"],
                                       "cost": alt_arm["training_cost"], "recovery_sign_delay": alt_arm["mean_recovery_delay"]})
                sensitivity[field][str(value)] = {"summary": summarize(altered_rows), "action_sensitivity": comparison}
                partial = {"config": asdict(cfg), "completed_sensitivity": sensitivity,
                           "calibration": {k: v for k, v in calibration.items() if k != "rollouts"},
                           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "is_partial": True}
                args.output.with_suffix(".partial.json").write_text(json.dumps(safe_json(partial), indent=2), encoding="utf-8")
                print(f"sensitivity {field}={value} done elapsed={time.time()-start:.1f}s", flush=True)
    result = {"scope": "controlled synthetic actual-SGD service loop; not real sensors, traces or hardware",
              "config": asdict(cfg), "pilot": args.pilot,
              "fit_prefix_seeds": [92001, 92002], "evaluation_seeds": [r["seed"] for r in rows],
              "no_test_parameter_selection": True,
              "protocol_frozen_before_formal_seeds": True,
              "timing_correction_after_formal_run": "Paired prefix price slot0 under initial component, transition after slot. Corrected all main/negative/sensitivity runs; old formal main and negative runs retained as before_timefix audits. No policy hyperparameter tuning.",
              "pilot_selection_history": {"pilot_seeds": [92501, 92502, 92503],
                                          "initial_linear_model": "early gain extrapolation exhausted update budget; retained in bayes_results_pilot.json",
                                          "final_change": "sign-specific hinge gain model and common2-of3value gate; retained in bayes_results_pilot_hinge.json",
                                          "final_pilot_weighted_increment": "no action/return gain vs same-context full in all3pilot seeds",
                                          "no_formal_seed_tuning": True},
              "hmm": {k: v for k, v in hmm.items() if not k.startswith("prefix_")},
              "calibration": {k: v for k, v in calibration.items() if k != "rollouts"},
              "calibration_rollout_count": len(calibration["rollouts"]),
              "causal_tests": tests, "summary": summarize(rows), "per_seed": rows,
              "sensitivity": sensitivity,
              "fairness": {"same_physical_jobs_labels_stakes": True,
                            "same_forecasts_masks_context_kernel_delayed_archive": True,
                            "same_hmm_and_value_model": True,
                            "same_primary_max_budget_steps_cooldown": True,
                            "count_matched_controls_are_ex_post_non_deployable_benchmarks": True,
                            "resource_occupancy_and_step_fee_charged_once": True},
              "limitations": ["designed residual instruments, not natural source telemetry",
                              "controlled workload and resource model, no queue/hardware measurements",
                              "Gaussian fixed-model HMM exactness is model conditional",
                              "online covariance is finite-archive PSD adaptive plug-in",
                              "calibration continuation uses fitted synthetic HMM, not externally validated simulator",
                              "known future-stake uncertainty is not removed by fused-loss weighting",
                              "local paired-fork advantages cannot be summed as long-run causal effects",
                              "95% normal paired intervals are descriptive with limited seeds"],
              "runtime_seconds": time.time() - start,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fusion_solver_sha256": hashlib.sha256((ROOT / "outputs" / "da_rf_fusion.py").read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest(),
              "python": platform.python_version(), "numpy": np.__version__}
    args.output.write_text(json.dumps(safe_json(result), indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
