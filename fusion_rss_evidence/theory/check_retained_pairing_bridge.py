#!/usr/bin/env python3
"""Independent algebra checks; synthetic diagnostic, not RSS performance.

Uses NumPy and the standard library only. Writes beside this script.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261004
N = 128
Q = 1.2815515655
NU = 0.01


def density(x, means, covariance, weights):
    inv = np.linalg.inv(covariance)
    residual = x[:, None, :] - means[None, :, :]
    quad = np.einsum("nki,ij,nkj->nk", residual, inv, residual)
    constant = (2 * math.pi) ** (-means.shape[1] / 2)
    constant /= math.sqrt(np.linalg.det(covariance))
    return constant * (np.exp(-0.5 * quad) @ weights)


def source_marginal(x, means, covariance, weights, s):
    variance = covariance[s, s]
    standardized = (x[:, None] - means[None, :, s]) / math.sqrt(variance)
    return np.exp(-0.5 * standardized**2) @ weights / math.sqrt(2 * math.pi * variance)


def source_moments(means, covariance, weights):
    mean = weights @ means
    centered = means - mean
    return mean, covariance + (centered.T * weights) @ centered


def posterior(likelihood, u, quad_weights):
    evidence = float(quad_weights @ likelihood)
    probability_density = likelihood / evidence
    mean = float(quad_weights @ (probability_density * u))
    variance = float(quad_weights @ (probability_density * (u - mean) ** 2))
    return evidence, probability_density, mean, variance


def score(mean, variance, q=Q, nu=NU):
    return N * (mean - q * math.sqrt(variance + nu**2)) - 5


def bridge_checks():
    a = 3 / 32
    patterns = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
    all_means = a * patterns
    correction = 0.00009 * np.array([[1, 0.55, 0.25], [0.55, 1, 0.30], [0.25, 0.30, 1]])
    full_covariance = 0.015**2 * np.eye(3) + correction
    diagonal_covariance = np.diag(np.diag(full_covariance))
    product_weights = np.full(8, 1 / 8)
    u, qw = np.polynomial.legendre.leggauss(512)
    taus = np.linspace(0, 1, 21)
    source_grid = np.linspace(-0.4, 0.4, 257)
    errors = {name: 0.0 for name in (
        "posterior_density", "conditional_mean", "conditional_variance",
        "paid_score", "score_derivative", "source_marginal", "source_mean", "source_covariance",
        "posterior_TV_scaling",
    )}
    cases = []
    for parity in (-1, 1):
        joint_means = all_means[np.prod(patterns, axis=1) == parity]
        joint_weights = np.full(4, 1 / 4)
        joint_mean, joint_covariance = source_moments(joint_means, full_covariance, joint_weights)
        gaussian_means = joint_mean[None, :]
        gaussian_weights = np.ones(1)
        bridge_endpoints = (
            ("exact_marginal_to_joint", all_means, diagonal_covariance, product_weights,
             joint_means, full_covariance, joint_weights, True, False),
            ("exact_marginal_to_joint_diagonal_P", all_means, diagonal_covariance, product_weights,
             joint_means, diagonal_covariance, joint_weights, True, False),
            ("full_Gaussian_to_joint", gaussian_means, joint_covariance, gaussian_weights,
             joint_means, full_covariance, joint_weights, False, True),
        )
        for (name, means0, cov0, weights0, means1, cov1, weights1,
             preserve_marginals, preserve_moments) in bridge_endpoints:
            if preserve_marginals:
                for s in range(3):
                    marginal0 = source_marginal(source_grid, means0, cov0, weights0, s)
                    marginal1 = source_marginal(source_grid, means1, cov1, weights1, s)
                    for tau in taus:
                        marginal_tau = (1 - tau) * marginal0 + tau * marginal1
                        errors["source_marginal"] = max(errors["source_marginal"],
                            float(np.max(np.abs(marginal_tau - marginal0))))
            if preserve_moments:
                mean0, sigma0 = source_moments(means0, cov0, weights0)
                mean1, sigma1 = source_moments(means1, cov1, weights1)
                for tau in taus:
                    mean_tau = (1 - tau) * mean0 + tau * mean1
                    second_tau = ((1 - tau) * (sigma0 + np.outer(mean0, mean0))
                                  + tau * (sigma1 + np.outer(mean1, mean1)))
                    sigma_tau = second_tau - np.outer(mean_tau, mean_tau)
                    errors["source_mean"] = max(errors["source_mean"], float(np.max(abs(mean_tau - mean1))))
                    errors["source_covariance"] = max(errors["source_covariance"], float(np.max(abs(sigma_tau - sigma1))))
            for h in (np.full(3, 5 / N), np.array([0.07, 0.04, 0.025]), np.array([-0.11, 0.08, 0.03])):
                line = h[None, :] - u[:, None]
                likelihood0 = density(line, means0, cov0, weights0)
                likelihood1 = density(line, means1, cov1, weights1)
                e0, p0, mu0, v0 = posterior(likelihood0, u, qw)
                e1, p1, mu1, v1 = posterior(likelihood1, u, qw)
                d = mu1 - mu0
                endpoint_tv = float(qw @ abs(p1 - p0) / 2)
                line_l1 = float(qw @ abs(likelihood1 - likelihood0))
                normalization_bound = (line_l1 + abs(e1 - e0)) / (2 * max(e1, e0))
                assert endpoint_tv <= normalization_bound + 1e-12
                records = []
                for tau in taus:
                    et, pt, mut, vt = posterior((1 - tau) * likelihood0 + tau * likelihood1, u, qw)
                    rho = tau * e1 / ((1 - tau) * e0 + tau * e1)
                    mu_formula = mu0 + rho * d
                    variance_formula = v0 + rho * (v1 - v0) + rho * (1 - rho) * d * d
                    score_formula = score(mu0, v0) + N * rho * d - Q * N * (
                        math.sqrt(variance_formula + NU**2) - math.sqrt(v0 + NU**2))
                    errors["posterior_density"] = max(errors["posterior_density"],
                        float(np.max(abs(pt - ((1 - rho) * p0 + rho * p1)))))
                    errors["conditional_mean"] = max(errors["conditional_mean"], abs(mut - mu_formula))
                    errors["conditional_variance"] = max(errors["conditional_variance"], abs(vt - variance_formula))
                    errors["paid_score"] = max(errors["paid_score"], abs(score(mut, vt) - score_formula))
                    errors["posterior_TV_scaling"] = max(errors["posterior_TV_scaling"],
                        abs(float(qw @ abs(pt - p0) / 2) - rho * endpoint_tv))
                    records.append({"tau": float(tau), "rho": float(rho), "evidence": et,
                                    "mean": mut, "variance": vt, "paid_score": score(mut, vt)})
                    if 0.01 < tau < 0.99:
                        epsilon = 1e-5
                        scores = []
                        for shifted in (tau - epsilon, tau + epsilon):
                            _, _, mus, vs = posterior((1 - shifted) * likelihood0 + shifted * likelihood1, u, qw)
                            scores.append(score(mus, vs))
                        numerical_derivative = (scores[1] - scores[0]) / (2 * epsilon)
                        derivative_rho = e0 * e1 / ((1 - tau) * e0 + tau * e1) ** 2
                        analytic_derivative = N * derivative_rho * (
                            d - Q * (v1 - v0 + (1 - 2 * rho) * d * d) /
                            (2 * math.sqrt(variance_formula + NU**2)))
                        errors["score_derivative"] = max(errors["score_derivative"],
                            abs(numerical_derivative - analytic_derivative))
                delta = float(qw @ abs(p1 - p0) / 2)
                b = min(1, 4 * delta)
                bound = N * (2 * delta + Q * (math.sqrt(NU**2 + b) - NU))
                cases.append({"bridge": name, "parity": parity, "h": h.tolist(),
                              "evidence0": e0, "evidence1": e1, "fitted_posterior_tv": delta,
                              "score0": score(mu0, v0), "score1": score(mu1, v1),
                              "score_change_bound": bound,
                              "gate_changed": (score(mu0, v0) > 0) != (score(mu1, v1) > 0),
                              "tau_records": records})
    tolerances = {"score_derivative": 5e-5, "posterior_density": 1e-10}
    for name, error in errors.items():
        assert error < tolerances.get(name, 1e-10), (name, error)
    for case in cases:
        assert abs(case["score1"] - case["score0"]) <= case["score_change_bound"] + 1e-10
    return {"max_absolute_identity_errors": errors, "cases": cases,
            "gate_changes_in_synthetic_cases": sum(c["gate_changed"] for c in cases)}


def finite_probability_checks():
    rng = np.random.default_rng(SEED)
    support = np.linspace(-1, 1, 41)
    cases = []
    for alpha in (0.01, 0.1, 1.0):
        p = rng.dirichlet(np.full(len(support), alpha), size=4000)
        raw_q = rng.dirichlet(np.full(len(support), alpha), size=4000)
        fractions = rng.uniform(0, 1, size=(4000, 1))
        q = (1 - fractions) * p + fractions * raw_q
        cases.append((p, q))
    # Extremes verify the factor 2 in the mean bound and near-factor 4 in variance.
    p = np.zeros((101, len(support)))
    p[:, 0] = 1
    q = p.copy()
    fractions = np.linspace(0, 1, len(p))
    q[:, 0] = 1 - fractions
    q[:, -1] = fractions
    cases.append((p, q))
    result = {"case_count": 0, "max_mean_bound_ratio": 0.0,
              "max_variance_bound_ratio": 0.0, "max_score_bound_ratio": 0.0,
              "smallest_variance_bound_slack": math.inf,
              "smallest_score_bound_slack": math.inf}
    for p, q in cases:
        delta = abs(q - p).sum(axis=1) / 2
        mean0, mean1 = p @ support, q @ support
        v0 = (p * (support[None, :] - mean0[:, None]) ** 2).sum(axis=1)
        v1 = (q * (support[None, :] - mean1[:, None]) ** 2).sum(axis=1)
        b = np.minimum(1, 4 * delta)
        difference_mean, difference_var = abs(mean1 - mean0), abs(v1 - v0)
        result["case_count"] += len(p)
        assert np.all(difference_mean <= 2 * delta + 1e-12)
        assert np.all(difference_var <= b + 1e-12)
        nonzero = delta > 0
        result["max_mean_bound_ratio"] = max(result["max_mean_bound_ratio"],
            float(np.max(difference_mean[nonzero] / (2 * delta[nonzero]))))
        result["max_variance_bound_ratio"] = max(result["max_variance_bound_ratio"],
            float(np.max(difference_var[nonzero] / b[nonzero])))
        result["smallest_variance_bound_slack"] = min(result["smallest_variance_bound_slack"],
            float(np.min(b - difference_var)))
        for nu in (0.01, 0.02, 0.1):
            for coefficient in (0.0, Q, 2.326347874):
                s0 = N * (mean0 - coefficient * np.sqrt(v0 + nu**2)) - 5
                s1 = N * (mean1 - coefficient * np.sqrt(v1 + nu**2)) - 5
                bound = N * (2 * delta + coefficient * (np.sqrt(nu**2 + b) - nu))
                difference_score = abs(s1 - s0)
                assert np.all(difference_score <= bound + 1e-9)
                result["max_score_bound_ratio"] = max(result["max_score_bound_ratio"],
                    float(np.max(difference_score[nonzero] / bound[nonzero])))
                result["smallest_score_bound_slack"] = min(result["smallest_score_bound_slack"],
                    float(np.min(bound - difference_score)))
    return result


def main():
    report = {"purpose": "Independent mathematical identity and bound checks; synthetic, not RSS evidence",
              "seed": SEED, "parameters": {"N": N, "q": Q, "nu": NU},
              "bridge_checks": bridge_checks(), "finite_probability_checks": finite_probability_checks(),
              "limits": ["Numerical agreement checks implementation of the displayed identities; it is not their proof.",
                         "The report's fitted-posterior TV is an endpoint sensitivity, not a bound to the physical conditional law.",
                         "Gate changes do not alone establish realized paid-return improvement.",
                         "The common execution-state condition is local; sequential policies can consume different reserve."]}
    (HERE / "retained_pairing_bridge_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    errors = report["bridge_checks"]["max_absolute_identity_errors"]
    lines = ["# Retained pairing bridge: independent checks", "",
             "These synthetic checks validate the algebra used by the new TeX fragment. They do not rerun or change scientific results.", "",
             f"Seed: {SEED}. Target support: [-1, 1]. N={N}, q={Q}, nu={NU}. Gaussian quadrature: 512 nodes.", "",
             "| Check | Maximum absolute error |", "| --- | ---: |"]
    lines += [f"| {name} | {value:.12g} |" for name, value in errors.items()]
    finite = report["finite_probability_checks"]
    lines += ["", f"All {finite['case_count']} finite-distribution pairs passed the mean and variance bounds; the paid-score bound passed at three nu and three q values.",
              f"Mean, variance and score maximum bound ratios: {finite['max_mean_bound_ratio']:.12g}, {finite['max_variance_bound_ratio']:.12g}, {finite['max_score_bound_ratio']:.12g}.", "",
              "The checks include exact-marginal→joint, exact-marginal→joint with diagonal P, and full moment-matched Gaussian→joint bridges. The first two keep coordinate densities fixed. The last keeps the full residual mean and covariance fixed. Direct normalized line quadrature independently checks the posterior bridge and its moments.", "",
              "The bridge weights are evidence weighted: rho generally differs from tau. Its variance includes rho(1-rho)(mu1-mu0)^2. Retained pairing is not guaranteed to produce a monotone score path.", "",
              "## Scope and estimation limit", "",
              "The source densities in this diagnostic are declared fitted working laws. Their computable posterior TV measures the response to a specific information reduction. No finite estimated value here certifies closeness to the unknown physical law. A true-law certificate requires an independently justified posterior-TV or density-estimation bound; the score inequality then propagates that bound to the gate.", "",
              "A bridge-induced admission difference becomes a paid benefit only after checking the candidate/reference return and common feasible execution state. Sequential reserve differences must be handled by complete replay accounting."]
    (HERE / "retained_pairing_bridge_checks.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"max_absolute_identity_errors": errors,
                      "finite_probability_checks": finite,
                      "gate_changes_in_synthetic_cases": report["bridge_checks"]["gate_changes_in_synthetic_cases"]}, indent=2))


if __name__ == "__main__":
    main()
