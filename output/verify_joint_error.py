#!/usr/bin/env python3
"""Known-moment synthetic verification, not an online edge-service experiment.

Checks matched marginal error, correlation-aware fusion, executed threshold
decisions, and fixed-feasible-set sensitivity. Requires numpy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from itertools import combinations


def regularized_weights(rho, sigma, tau, lam, inertia):
    # Symmetry reduces the exact three-source convex problem to w3 = y.
    if rho == 0:
        return np.full(3, 1 / 3)
    lo, hi = 1 / 3, (1 + rho) / (3 + rho)
    for _ in range(100):
        y = (lo + hi) / 2
        derivative = (lam * sigma**2 / 2 * ((3 + rho) * y - (1 + rho))
                      + tau * np.log(2 * y / (1 - y))
                      + 1.5 * inertia * (y - 1 / 3))
        if derivative > 0:
            hi = y
        else:
            lo = y
    y = (lo + hi) / 2
    return np.array([(1 - y) / 2, (1 - y) / 2, y])


def solve_general(q, matrix, anchor, cap=0.7, tau=0.2, lam=0.7, inertia=0.3):
    pi = q / q.sum()
    def objective(w):
        wp = np.maximum(w, 1e-14)
        return (tau * np.sum(w * np.log(wp / pi))
                + lam / 2 * w @ matrix @ w
                + inertia / 2 * np.sum((w - anchor)**2))
    def gradient(w):
        return (tau * (np.log(np.maximum(w, 1e-14) / pi) + 1)
                + lam * matrix @ w + inertia * (w - anchor))
    # Entropy makes every available coordinate positive. Enumerating the upper-
    # bound active sets is inexpensive for three sources. Newton solves each
    # equality-constrained free-coordinate subproblem; no SciPy is needed.
    candidates = []
    n = len(q)
    for count in range(n):
        for active in combinations(range(n), count):
            mass = 1 - count * cap
            free = [i for i in range(n) if i not in active]
            if mass <= 0 or mass > len(free) * cap + 1e-12:
                continue
            w = np.full(n, cap)
            w[free] = mass / len(free)
            for _ in range(200):
                g = gradient(w)[free]
                if np.ptp(g) < 1e-11:
                    break
                hessian = tau * np.diag(1 / w) + lam * matrix + inertia * np.eye(n)
                h = hessian[np.ix_(free, free)]
                hg = np.linalg.solve(h, g)
                h1 = np.linalg.solve(h, np.ones(len(free)))
                direction = -hg + h1 * (hg.sum() / h1.sum())
                if np.linalg.norm(direction) < 1e-12:
                    break
                step = 1.0
                negative = direction < 0
                if negative.any():
                    step = min(step, 0.99 * np.min(-w[free][negative] / direction[negative]))
                old = objective(w)
                slope = g @ direction
                for _ in range(60):
                    trial = w.copy()
                    trial[free] += step * direction
                    if (trial.min() > 0 and
                        objective(trial) <= old + 1e-4 * step * slope + 1e-15):
                        break
                    step /= 2
                w = trial
            if (np.ptp(gradient(w)[free]) < 2e-7
                    and w.max() <= cap + 1e-10
                    and abs(w.sum() - 1) < 1e-10):
                candidates.append(w)
    if not candidates:
        raise RuntimeError("No converged feasible active-set solution")
    return min(candidates, key=objective)


def sensitivity_check():
    rng = np.random.default_rng(71203)
    ratios = []
    for _ in range(100):
        q = np.exp(rng.normal(0, 0.8, 3))
        q2 = q * np.exp(rng.normal(0, 0.04, 3))
        errors = rng.uniform(-0.25, 0.25, (64, 3))
        matrix = errors.T @ errors / len(errors)
        v = rng.normal(size=3)
        matrix2 = matrix + 0.006 * np.outer(v, v)
        anchor = rng.dirichlet(np.ones(3))
        anchor2 = 0.95 * anchor + 0.05 * rng.dirichlet(np.ones(3))
        w = solve_general(q, matrix, anchor)
        w2 = solve_general(q2, matrix2, anchor2)
        log_delta = np.log(q) - np.log(q2)
        quality_delta = np.linalg.norm(log_delta - log_delta.mean())
        bound = (0.2 * quality_delta
                 + 0.7 * np.sqrt(0.7) * np.linalg.norm(matrix - matrix2, 2)
                 + 0.3 * np.linalg.norm(anchor - anchor2)) / (0.2 / 0.7 + 0.3)
        actual = np.linalg.norm(w - w2)
        if actual > bound + 2e-6:
            raise AssertionError((actual, bound))
        ratios.append(actual / bound if bound else 0)
    return {"cases": 100, "largest_observed_change_to_bound_ratio": max(ratios),
            "numerical_tolerance": 2e-6,
            "scope": "fixed mask, cap, and regularization; finite numerical audit"}


def three_regime_check(out_dir, seeds=20, samples=100000):
    """Execute the fixed-pooled three-regime decision-weighting mechanism.

    All rules receive the known population moments at issuance. This tests
    threshold actions with decision-weighted cross-errors, not online moment
    estimation, a regime-conditioned fusion rule, or neural policy recovery.
    """
    sigma, theta, half_width = 0.15, 0.5, 0.25
    tau, lam, inertia, delta, cap = 0.001, 1.0, 0.001, 0.0, 0.7
    correlations = [0.0, 0.25, 0.5, 0.75, 0.9]
    uniform = np.full(3, 1 / 3)
    rows, paired = [], []
    worst_marginal_error = 0.0
    worst_diagonal_error = 0.0
    worst_trace_error = 0.0
    worst_solver_error = 0.0
    worst_empirical_bound_ratio = 0.0
    for rho in correlations:
        r12 = sigma**2 * np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]])
        pooled = sigma**2 * ((1 - rho / 3) * np.eye(3)
                            + (rho / 3) * np.ones((3, 3)))
        matched = pooled / 3
        weighted = r12 / 3
        diagonal_error = float(np.max(np.abs(np.diag(weighted) - np.diag(matched))))
        trace_error = float(abs(np.trace(weighted) - np.trace(matched)))
        worst_diagonal_error = max(worst_diagonal_error, diagonal_error)
        worst_trace_error = max(worst_trace_error, trace_error)
        assert diagonal_error <= 1e-15 and trace_error <= 1e-15
        methods = {
            "pooled_R": uniform.copy(),
            "trace_matched_R_over_3": uniform.copy(),
            "decision_weighted_regularized":
                regularized_weights(rho, sigma, tau, lam / 3, inertia),
            "decision_weighted_unregularized_reference":
                np.array([1, 1, 1 + rho]) / (3 + rho),
        }
        # Independent active-set Newton checks the symmetry-reduced solution.
        for name, matrix in [("pooled_R", pooled),
                             ("trace_matched_R_over_3", matched),
                             ("decision_weighted_regularized", weighted)]:
            general = solve_general(np.ones(3), matrix, uniform, cap=cap,
                                    tau=tau, lam=lam, inertia=inertia)
            solver_error = float(np.max(np.abs(general - methods[name])))
            worst_solver_error = max(worst_solver_error, solver_error)
            assert solver_error < 2e-7
        if rho > 0:
            w = methods["decision_weighted_regularized"]
            assert 1 / (3 + rho) < w[0] < 1 / 3
            assert float(w @ weighted @ w) < float(uniform @ weighted @ uniform)
        values = {name: [] for name in methods}
        for seed in range(seeds):
            # Pair all rules, and reuse the underlying random streams over rho.
            rng = np.random.default_rng(62000 + seed)
            target = rng.uniform(theta - half_width, theta + half_width, samples)
            regime = rng.integers(0, 3, samples)  # 12, 13, 23 with probability 1/3.
            first = rng.choice([-1.0, 1.0], samples)
            multiplier_uniform = rng.random(samples)
            multiplier = np.where(multiplier_uniform < (1 + rho) / 2, 1.0, -1.0)
            independent = rng.choice([-1.0, 1.0], samples)
            errors = np.empty((samples, 3))
            for code, (a, b, remaining) in enumerate([(0, 1, 2), (0, 2, 1), (1, 2, 0)]):
                mask = regime == code
                errors[mask, a] = first[mask]
                errors[mask, b] = multiplier[mask] * first[mask]
                errors[mask, remaining] = independent[mask]
            errors *= sigma
            marginal_error = float(np.max(np.abs(errors**2 - sigma**2)))
            worst_marginal_error = max(worst_marginal_error, marginal_error)
            assert marginal_error <= 1e-15
            assert np.all((target[:, None] + errors >= 0)
                          & (target[:, None] + errors <= 1))
            stakes = (regime == 0).astype(float)
            oracle_utility = stakes * np.maximum(target - theta, 0)
            baseline_action = (target + errors @ uniform >= theta).astype(float)
            baseline_utility = stakes * baseline_action * (target - theta)
            seed_metrics = {}
            for name, w in methods.items():
                assert w.min() >= 0 and w.max() <= cap + 1e-12
                assert abs(w.sum() - 1) <= 1e-12
                error = errors @ w
                action = (target + error >= theta).astype(float)
                utility = stakes * action * (target - theta)
                regret = oracle_utility - utility
                changed = action != baseline_action
                utility_change = utility - baseline_utility
                beneficial = changed & (utility_change > 0)
                harmful = changed & (utility_change < 0)
                neutral = changed & (utility_change == 0)
                assert np.array_equal(beneficial | harmful | neutral, changed)
                assert not np.any((beneficial | harmful) & (stakes == 0))
                changed_count = int(changed.sum())
                assert regret.min() >= -1e-15
                # B^2=B for the declared binary-stakes construction.
                empirical_risk = float(np.mean(stakes * error**2))
                empirical_regret = float(np.mean(regret))
                empirical_bound = float(np.sqrt(empirical_risk))
                ratio = empirical_regret / empirical_bound if empirical_bound else 0.0
                worst_empirical_bound_ratio = max(worst_empirical_bound_ratio, ratio)
                assert np.all(regret <= stakes * np.abs(error) + 1e-15)
                assert empirical_regret <= empirical_bound + 1e-15
                metrics = {
                    "utility": float(np.mean(utility)),
                    "regret": empirical_regret,
                    "decision_risk": empirical_risk,
                    "forecast_mse": float(np.mean(error**2)),
                    "action_rate": float(np.mean(action)),
                    "action_changed_fraction": float(np.mean(changed)),
                    "beneficial_action_change_fraction": float(np.mean(beneficial)),
                    "harmful_action_change_fraction": float(np.mean(harmful)),
                    "neutral_action_change_fraction": float(np.mean(neutral)),
                    "beneficial_fraction_among_changed":
                        float(beneficial.sum() / changed_count) if changed_count else 0.0,
                    "harmful_fraction_among_changed":
                        float(harmful.sum() / changed_count) if changed_count else 0.0,
                    "neutral_fraction_among_changed":
                        float(neutral.sum() / changed_count) if changed_count else 0.0,
                }
                for label, stratum in [("B1", stakes == 1), ("B0", stakes == 0)]:
                    assert np.any(stratum)
                    metrics[f"action_changed_fraction_{label}"] = float(np.mean(changed[stratum]))
                    for change_name, change_mask in [("beneficial", beneficial),
                                                    ("harmful", harmful), ("neutral", neutral)]:
                        metrics[f"{change_name}_action_change_fraction_{label}"] = \
                            float(np.mean(change_mask[stratum]))
                seed_metrics[name] = metrics
                values[name].append(metrics)
            base = seed_metrics["pooled_R"]
            for name, metrics in seed_metrics.items():
                paired.append({
                    "rho": rho, "seed": seed, "rng_seed": 62000 + seed,
                    "method": name, "samples": samples,
                    "regime_12_count": int(np.sum(regime == 0)),
                    "regime_13_count": int(np.sum(regime == 1)),
                    "regime_23_count": int(np.sum(regime == 2)),
                    "utility": metrics["utility"], "regret": metrics["regret"],
                    "decision_risk": metrics["decision_risk"],
                    "forecast_mse": metrics["forecast_mse"],
                    "action_rate": metrics["action_rate"],
                    "paired_utility_gain_vs_pooled": metrics["utility"] - base["utility"],
                    "paired_regret_reduction_vs_pooled": base["regret"] - metrics["regret"],
                    "same_per_trial_marginal_maxerror": marginal_error,
                    **{key: value for key, value in metrics.items()
                       if "fraction" in key},
                })
        for name, w in methods.items():
            population_risk = float(w @ weighted @ w)
            population_regret = population_risk / (4 * half_width)
            population_baseline_regret = float(uniform @ weighted @ uniform) / (4 * half_width)
            a = values[name]
            paired_gain = np.array([a[i]["utility"] - values["pooled_R"][i]["utility"]
                                    for i in range(seeds)])
            row = {
                "rho": rho, "method": name, "seeds": seeds, "samples_per_seed": samples,
                "w1": float(w[0]), "w2": float(w[1]), "w3": float(w[2]),
                "population_forecast_mse": float(w @ pooled @ w),
                "population_decision_weighted_risk": population_risk,
                "analytic_population_regret": population_regret,
                "analytic_population_utility": half_width / 12 - population_regret,
                "analytic_utility_gain_vs_pooled": population_baseline_regret - population_regret,
                "population_regret_bound": float(np.sqrt(population_risk)),
                "same_diagonal_maxerror": diagonal_error,
                "same_trace_maxerror": trace_error,
                "paired_utility_gain_mean": float(paired_gain.mean()),
                "paired_utility_gain_seed_sd": float(paired_gain.std(ddof=1)),
            }
            for metric in a[0]:
                array = np.array([item[metric] for item in a])
                row[f"empirical_{metric}_mean"] = float(array.mean())
                row[f"empirical_{metric}_seed_sd"] = float(array.std(ddof=1))
            rows.append(row)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [("decision_weighting_verification.csv", rows),
             ("decision_weighting_paired.csv", paired)]
    for filename, data in files:
        with (out_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    script = Path(__file__).resolve()
    specification = script.parent.parent / "work" / "decision_risk_fragment.tex"
    audit = {
        "scope": "known-moment fixed-pooled three-regime threshold-decision verification",
        "verification_version": "three-regime-v2-action-audit",
        "not_evaluated": ["online moment estimation", "regime-conditioned fusion",
                          "edge queue service", "neural retraining", "DA-RF recovery"],
        "seeds": seeds, "rng_seeds": list(range(62000, 62000 + seeds)),
        "samples_per_seed_per_correlation": samples, "correlations": correlations,
        "regimes": ["12", "13", "23"], "regime_probability": [1 / 3] * 3,
        "B": [1, 0, 0], "sigma": sigma, "theta": theta, "half_width": half_width,
        "tau": tau, "lambda": lam, "inertia": inertia, "delta": delta, "cap": cap,
        "quality_reference": [1 / 3] * 3, "anchor": [1 / 3] * 3,
        "model_error_allowance": 0,
        "moment_source": "exact population moments, supplied before every threshold action",
        "action_audit_reference": "threshold actions of fixed pooled_R uniform weights",
        "action_audit_denominators": {
            "action_changed_fraction": "all trials",
            "beneficial_harmful_neutral_action_change_fraction": "all trials",
            "fractions_with_B1_suffix": "all trials with stakes B=1",
            "fractions_with_B0_suffix": "all trials with stakes B=0",
            "fractions_among_changed": "all changed actions; zero when no action changes",
        },
        "maximum_per_trial_marginal_squared_error_difference": worst_marginal_error,
        "maximum_population_diagonal_difference": worst_diagonal_error,
        "maximum_population_trace_difference": worst_trace_error,
        "maximum_symmetry_solver_vs_general_solver_error": worst_solver_error,
        "largest_empirical_regret_to_empirical_moment_bound_ratio": worst_empirical_bound_ratio,
        "numpy_version": np.__version__,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "specification_sha256_at_run":
            hashlib.sha256(specification.read_bytes()).hexdigest() if specification.exists() else None,
        "output_sha256": {filename: hashlib.sha256((out_dir / filename).read_bytes()).hexdigest()
                          for filename, _ in files},
        "results": rows,
    }
    (out_dir / "decision_weighting_audit.json").write_text(json.dumps(audit, indent=2))
    for row in rows:
        if row["method"] in ["pooled_R", "decision_weighted_regularized"]:
            print(f"three-regime rho={row['rho']:.2f} {row['method']:30s} "
                  f"utility={row['empirical_utility_mean']:.8f} "
                  f"regret={row['empirical_regret_mean']:.8f} "
                  f"paired_gain={row['paired_utility_gain_mean']:.8f} "
                  f"bound={row['population_regret_bound']:.8f}")
    print("Three-regime audit:", {key: audit[key] for key in [
        "maximum_per_trial_marginal_squared_error_difference",
        "maximum_population_diagonal_difference", "maximum_population_trace_difference",
        "maximum_symmetry_solver_vs_general_solver_error",
        "largest_empirical_regret_to_empirical_moment_bound_ratio", "script_sha256"]})
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--three-regime-only", action="store_true",
                        help="Run only the independent three-regime decision-weighting check")
    args = parser.parse_args()
    if args.seeds < 2 or args.samples < 1:
        parser.error("--seeds must be at least 2 and --samples must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.three_regime_only:
        three_regime_check(args.out_dir, args.seeds, args.samples)
        return
    sigma, theta, half_width = 0.15, 0.5, 0.25
    tau, lam, inertia = 0.001, 1.0, 0.001
    correlations = [0.0, 0.25, 0.5, 0.75, 0.9]
    rows, paired = [], []
    for rho in correlations:
        matrix = sigma**2 * np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]])
        methods = {"diagonal": np.full(3, 1 / 3),
                   "full_regularized": regularized_weights(rho, sigma, tau, lam, inertia),
                   "full_unregularized_oracle": np.array([1, 1, 1 + rho]) / (3 + rho)}
        values = {name: [] for name in methods}
        for seed in range(args.seeds):
            # Identical random streams across rho, and paired inputs across methods.
            rng = np.random.default_rng(51000 + seed)
            target = rng.uniform(theta - half_width, theta + half_width, args.samples)
            first = rng.choice([-1.0, 1.0], args.samples)
            multiplier = np.where(rng.random(args.samples) < (1 + rho) / 2, 1.0, -1.0)
            third = rng.choice([-1.0, 1.0], args.samples)
            errors = sigma * np.column_stack([first, multiplier * first, third])
            assert np.allclose(errors**2, sigma**2)
            assert np.all((target[:, None] + errors >= 0)
                          & (target[:, None] + errors <= 1))
            oracle_utility = np.maximum(target - theta, 0)
            for name, w in methods.items():
                error = errors @ w
                action = target + error >= theta
                utility = action * (target - theta)
                regret = oracle_utility - utility
                assert regret.min() >= -1e-15
                values[name].append([np.mean(error**2), np.mean(regret), np.mean(utility)])
            paired.append({"rho": rho, "seed": seed,
                           "utility_gain_full_minus_diagonal":
                           values["full_regularized"][-1][2] - values["diagonal"][-1][2]})
        for name, w in methods.items():
            a = np.asarray(values[name])
            risk = float(w @ matrix @ w)
            if rho > 0 and name == "full_regularized":
                assert risk < float(methods["diagonal"] @ matrix @ methods["diagonal"])
            rows.append({"rho": rho, "method": name,
                         "w1": w[0], "w2": w[1], "w3": w[2],
                         "population_mse": risk,
                         "empirical_mse_mean": a[:, 0].mean(),
                         "empirical_mse_seed_sd": a[:, 0].std(ddof=1),
                         "population_decision_regret": risk / (4 * half_width),
                         "empirical_regret_mean": a[:, 1].mean(),
                         "empirical_regret_seed_sd": a[:, 1].std(ddof=1),
                         "empirical_utility_mean": a[:, 2].mean(),
                         "empirical_utility_seed_sd": a[:, 2].std(ddof=1)})
    for filename, data in [("joint_error_verification.csv", rows),
                           ("joint_error_paired_utility.csv", paired)]:
        with (args.out_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    audit = {"scope": "known-moment synthetic threshold-decision verification",
             "not_evaluated": ["online moment estimation", "edge queue service",
                               "neural retraining", "DA-RF recovery sentinel"],
             "seeds": args.seeds, "samples_per_seed_per_correlation": args.samples,
             "correlations": correlations, "sigma": sigma, "theta": theta,
             "half_width": half_width, "tau": tau, "lambda": lam, "inertia": inertia,
             "cap": 0.7, "numpy_version": np.__version__,
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "sensitivity": sensitivity_check()}
    (args.out_dir / "joint_error_audit.json").write_text(json.dumps(audit, indent=2))
    for row in rows:
        if row["method"] in ["diagonal", "full_regularized"]:
            print(f"rho={row['rho']:.2f} {row['method']:18s} "
                  f"mse={row['empirical_mse_mean']:.7f} "
                  f"regret={row['empirical_regret_mean']:.7f} "
                  f"utility={row['empirical_utility_mean']:.7f}")
    print("Sensitivity audit:", audit["sensitivity"])
    three_regime_check(args.out_dir, args.seeds, args.samples)


if __name__ == "__main__":
    main()
