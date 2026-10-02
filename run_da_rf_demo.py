#!/usr/bin/env python3
"""Execute separate synthetic pipeline diagnostics; never an edge-service claim.

The first streams causal forecasts through delayed labels and missing sources.
The second performs real NumPy logistic-gradient updates on worker copies and
deploys them only after a simulated delay. It checks recoverability separately
from DA-RF; no DA-RF recovery or queue-service improvement is asserted.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import numpy as np


def json_safe(value):
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def streaming_fusion_demo(windows=120):
    from da_rf_fusion import DecisionAwareFusion, FusionConfig, QualityObservation
    from da_rf_predictor import DelayedRidgeForecaster, RidgeConfig
    from da_rf_evidence import block_bootstrap_variance

    rng_prefix = np.random.default_rng(81007)
    prefix_size = 160
    latent = np.empty(prefix_size + 1)
    latent[0] = 0.2
    for k in range(prefix_size):
        latent[k + 1] = np.clip(0.8 * latent[k] + 0.2 * 0.2
                                + rng_prefix.normal(0, 0.025), 0, 1)
    prefix_features = np.ones((prefix_size, 3, 2))
    for s, sd in enumerate([0.02, 0.06, 0.10]):
        lag = 2 if s == 2 else 0
        signal = latent[np.maximum(np.arange(prefix_size) - lag, 0)]
        prefix_features[:, s, 1] = signal + rng_prefix.normal(0, sd, prefix_size)
    predictor_config = RidgeConfig(ridge=0.01, forgetting=0.98)
    predictor = DelayedRidgeForecaster(3, 2, predictor_config)
    predictor.fit_prefix(prefix_features, latent[1:])
    fusion_config = FusionConfig(tau=0.02, lam=1.0, inertia=0.02,
                                 max_weight=0.6, context_bandwidth=1.0)
    fusion = DecisionAwareFusion(3, 2, fusion_config)
    rng = np.random.default_rng(81008)
    rng_bootstrap = np.random.default_rng(81009)
    current = 0.2
    state_history = []
    measurements = [[] for _ in range(3)]
    diagnostic_history = [[] for _ in range(3)]
    labels_due = []
    issued = {}
    snapshots = []
    label_audits = []
    target_log = {}

    def ingest(clock):
        due = sorted([item for item in labels_due if item[0] <= clock])
        for arrival, origin, target in due:
            predictor_audit = predictor.observe_label(origin, target, arrival)
            fusion_audit = fusion.observe_label(origin, target, arrival)
            label_audits.append({"forecast_window": origin, "target_window": origin + 1,
                                 "arrival_window": arrival, "target": target,
                                 "predictor": predictor_audit, "fusion": fusion_audit})
            labels_due.remove((arrival, origin, target))

    # The current physical outcome is generated sequentially. The target for
    # forecast k is created in physical window k+1, then withheld for its delay.
    for window in range(windows + 1):
        mean = 0.2 if window < 40 else (0.65 if window < 80 else 0.3)
        current = float(np.clip(0.8 * current + 0.2 * mean + rng.normal(0, 0.025), 0, 1))
        state_history.append(current)
        if window > 0:
            origin = window - 1
            delay = 2 + origin % 3
            labels_due.append((window + delay, origin, current))
            target_log[origin] = {"target": current, "arrival_window": window + delay}
        ingest(window)
        if window == windows:
            break
        available = np.ones(3, dtype=bool)
        if 30 <= window < 40:
            available[0] = False
        if 65 <= window < 73:
            available[1] = False
        if 35 <= window < 44:
            available[2] = False
        if 90 <= window < 92:
            available[:] = False
        features = np.ones((3, 2))
        quality = []
        quality_audits = []
        for s, sd in enumerate([0.02, 0.06, 0.10]):
            lag = 2 if s == 2 else 0
            observed = state_history[max(0, window - lag)]
            observed += rng.normal(0, 0.35 if s == 1 and 50 <= window < 62 else sd)
            if available[s]:
                previous = measurements[s][-1] if measurements[s] else np.nan
                measurements[s].append(float(observed))
                diagnostic = (float(np.clip(abs(observed - previous) / 0.3, 0, 1))
                              if np.isfinite(previous) else np.nan)
                diagnostic_history[s].append(diagnostic)
                features[s, 1] = observed
            else:
                measurements[s].append(np.nan)
                diagnostic_history[s].append(np.nan)
                features[s] = np.nan
                diagnostic = np.nan
            recent = np.array(diagnostic_history[s][-10:])
            count = int(np.isfinite(recent).sum())
            variance, bootstrap_audit = block_bootstrap_variance(
                recent, rng=rng_bootstrap, replicates=200, fallback_variance=0.25)
            quality.append(QualityObservation(bool(available[s]), count, lag, variance,
                                               diagnostic if np.isfinite(diagnostic) else None))
            quality_audits.append({"source": s, "available_measurement": bool(available[s]),
                                  "valid_diagnostic_count": count,
                                  "chronological_diagnostic_window_size": len(recent),
                                  "diagnostic": diagnostic, "variance": variance,
                                  "bootstrap": bootstrap_audit})
        predictions = predictor.issue(window, features, available)
        # Context and stakes depend only on currently observed physical fields.
        context = np.array([np.cos(window / 13), np.sin(window / 11)])
        slopes = np.array([0.0, 0.4 + 0.2 * (1 + context[0])])
        snapshot = fusion.issue(window, predictions, context, quality, slopes)
        issued[window] = float(snapshot.forecast) if snapshot.forecast is not None else np.nan
        snapshots.append({"window": window, "weights": snapshot.weights,
                          "forecast": snapshot.forecast, "uncertainty": snapshot.uncertainty,
                          "coverage": snapshot.coverage, "all_missing": snapshot.all_missing,
                          "solver_fallback": snapshot.solver_fallback,
                          "diagnostic": snapshot.diagnostic, "risk": snapshot.risk,
                          "available": available, "quality_audits": quality_audits,
                          "audit": snapshot.audit})
    while labels_due:
        ingest(min(item[0] for item in labels_due))
    scored = [(issued[k], row["target"]) for k, row in target_log.items()
              if np.isfinite(issued[k])]
    errors = np.array([pred - target for pred, target in scored])
    for row in snapshots:
        row.update({"target_for_retrospective_scoring": target_log[row["window"]]["target"],
                    "label_arrival_window": target_log[row["window"]]["arrival_window"]})
    return {
        "scope": "causal streaming fusion diagnostic with independent prefix-fitted source predictors",
        "not_evaluated": ["queue serving", "neural policy recovery", "DA-RF service superiority"],
        "seed_prefix": 81007, "seed_stream": 81008, "seed_bootstrap": 81009,
        "forecast_target": "next physical window's normalized outcome, observed after delay",
        "quality_variance": "200-replicate moving-block bootstrap of the last ten chronological adjacent-measurement diagnostic values, preserving NaN gaps; fallback 0.25",
        "support_count": "finite diagnostic values in the same chronological quality window",
        "predictor_config": predictor_config, "fusion_config": fusion_config,
        "issued_windows": windows, "label_records": len(label_audits),
        "scored_forecasts": len(scored), "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors**2)),
        "all_missing_windows": sum(bool(s["all_missing"]) for s in snapshots),
        "partial_missing_windows": sum(0 < s["coverage"] < 1 for s in snapshots),
        "solver_fallback_windows": sum(bool(s["solver_fallback"]) for s in snapshots),
        "snapshots": snapshots, "label_audits": label_audits,
        "last_issuance_matrices": fusion.get_last_matrices(),
        "archive_summary": fusion.archive_summary(),
    }


def permanent_outage_moment_demo(windows=400, seed=83001):
    """Estimate an active-subset moment despite one permanently missing source.

    The comparator's complete-enabled requirement admits zero residual records.
    Its isotropic prior, uniform quality, and uniform anchor therefore keep the
    active weights uniform. This is a synthetic archive-mechanism diagnostic,
    not a reproduced external algorithm or a policy-recovery experiment.
    """
    from da_rf_fusion import DecisionAwareFusion, FusionConfig, QualityObservation
    config = FusionConfig(tau=0.001, lam=1.0, inertia=0,
                          max_weight=0.7, prior_weight=1e-6, decay=1.0,
                          delta_base=0, delta_prior=0, delta_count=0, delta_age=0)
    fusion = DecisionAwareFusion(4, 1, config)
    complete_config = replace(config, require_complete_enabled_vectors=True)
    complete_reference = DecisionAwareFusion(4, 1, complete_config)
    rng = np.random.default_rng(seed)
    sigma, rho = 0.15, 0.9
    reference = np.full(3, 1 / 3)
    population = sigma**2 * np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]])
    records = []
    for window in range(windows):
        if window > 0:
            fusion.observe_label(window - 1, 0.5, window)
            complete_reference.observe_label(window - 1, 0.5, window)
        signs = rng.choice([-1.0, 1.0], 2)
        multiplier = 1.0 if rng.random() < (1 + rho) / 2 else -1.0
        errors = sigma * np.array([signs[0], multiplier * signs[0], signs[1]])
        assert np.all(errors**2 == sigma**2)
        predictions = np.r_[0.5 + errors, np.nan]
        quality = [QualityObservation(True, 50, 0, 0.02)] * 3
        quality += [QualityObservation(False, 0, 0, None)]
        snapshot = fusion.issue(window, predictions, [0.0], quality, [0.0, 1.0])
        reference_snapshot = complete_reference.issue(window, predictions, [0.0], quality, [0.0, 1.0])
        np.testing.assert_allclose(reference_snapshot.weights[:3], reference, atol=1e-10)
        weights = snapshot.weights[:3]
        matrices = fusion.get_last_matrices()
        minimum_eigenvalue = float(np.linalg.eigvalsh(matrices["Mhat"]).min())
        assert minimum_eigenvalue >= -1e-12
        assert snapshot.weights[3] == 0
        records.append({"seed": seed, "window": window,
                        "active_archive_records": snapshot.audit["support_count"],
                        "complete_enabled_support_records": reference_snapshot.audit["support_count"],
                        "w1": weights[0], "w2": weights[1], "w3": weights[2], "w4": 0.0,
                        "minimum_eigenvalue": minimum_eigenvalue,
                        "population_mse_active_subset": float(weights @ population @ weights),
                        "population_mse_complete_enabled_reference": float(reference @ population @ reference),
                        "executed_forecast_squared_error": float((weights @ errors)**2),
                        "reference_forecast_squared_error": float((reference @ errors)**2),
                        "solver_fallback": snapshot.solver_fallback})
    fusion.observe_label(windows - 1, 0.5, windows)
    complete_reference.observe_label(windows - 1, 0.5, windows)
    tail = records[-100:]
    return ({"scope": "permanently missing fourth-source archive-learning diagnostic",
             "not_evaluated": ["online service recovery", "external algorithm superiority"],
             "seed": seed, "enabled_sources": 4, "permanently_missing_source": 3,
             "active_sources": [0, 1, 2], "sigma": sigma, "rho": rho,
             "config": config, "complete_enabled_config": complete_config,
             "issued_windows": windows,
             "complete_enabled_support_records": records[-1]["complete_enabled_support_records"],
             "final_active_archive_records_at_last_issuance": records[-1]["active_archive_records"],
             "last_issued_weights": [records[-1][f"w{i}"] for i in range(1, 5)],
             "minimum_recorded_moment_eigenvalue": min(r["minimum_eigenvalue"] for r in records),
             "tail_100_mean_population_mse_active_subset": float(np.mean([r["population_mse_active_subset"] for r in tail])),
             "population_mse_complete_enabled_reference": records[-1]["population_mse_complete_enabled_reference"],
             "tail_100_empirical_mse_active_subset": float(np.mean([r["executed_forecast_squared_error"] for r in tail])),
             "tail_100_empirical_mse_complete_enabled_reference": float(np.mean([r["reference_forecast_squared_error"] for r in tail])),
             "solver_fallback_windows": sum(r["solver_fallback"] for r in records)}, records)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def logistic_update(theta, x, y, steps, learning_rate):
    result = np.asarray(theta, dtype=float).copy()
    for _ in range(steps):
        gradient = x.T @ (sigmoid(x @ result) - y) / len(y)
        result -= learning_rate * gradient
    return result


def gradient_recoverability_demo(horizon=1200, shift_slot=600):
    # A separate, explicitly synthetic diagnostic. The schedule and loss
    # sentinel are comparators; neither is relabeled as a DA-RF controller.
    rng_prefix = np.random.default_rng(82001)
    truth = np.array([1.0, -0.5])
    prefix_x = rng_prefix.normal(size=(1000, 2))
    prefix_y = (prefix_x @ truth >= 0).astype(float)
    initial = logistic_update(np.zeros(2), prefix_x, prefix_y, 160, 0.2)
    rng = np.random.default_rng(82002)
    x = rng.normal(size=(horizon, 2))
    y = (x @ truth >= 0).astype(float)
    y[shift_slot:] = 1 - y[shift_slot:]
    profile = {"gradient_steps_per_worker": 32, "learning_rate": 0.25,
               "deployment_delay_slots": 8, "label_delay_slots": 3,
               "batch_size": 96, "minimum_new_labels": 64,
               "maximum_workers": 12, "cost_per_gradient_step": 0.001,
               "periodic_interval_slots": 96, "observed_loss_threshold": 0.35}
    arms, records = {}, []
    for arm in ["frozen", "periodic_worker", "observed_loss_worker"]:
        theta = initial.copy()
        active = None
        predictions = np.empty(horizon)
        observed_losses = []
        jobs, deployments, steps = 0, 0, 0
        last_consumed_labels = 0
        version = 0
        launches = []
        deployment_log = []
        for slot in range(horizon):
            deployed_now = False
            if active is not None and slot >= active["complete_slot"]:
                theta = active["parameters"].copy()
                deployment_log.append({"launch_slot": active["launch_slot"],
                                       "deployment_slot": slot,
                                       "gradient_steps": active["gradient_steps"]})
                active = None
                deployments += 1
                version += 1
                deployed_now = True
            predictions[slot] = sigmoid(x[slot] @ theta)
            arrived_origin = slot - profile["label_delay_slots"]
            if arrived_origin >= 0:
                observed_losses.append(float((predictions[arrived_origin] >= 0.5)
                                             != bool(y[arrived_origin])))
            arrived_count = max(0, arrived_origin + 1)
            eligible = (active is None and jobs < profile["maximum_workers"]
                        and arrived_count - last_consumed_labels >= profile["minimum_new_labels"])
            periodic = arm == "periodic_worker" and slot > 0 and slot % profile["periodic_interval_slots"] == 0
            loss_trigger = (arm == "observed_loss_worker" and len(observed_losses) >= 64
                            and np.mean(observed_losses[-64:]) >= profile["observed_loss_threshold"])
            launched = eligible and (periodic or loss_trigger)
            if launched:
                stop = arrived_count
                start = max(0, stop - profile["batch_size"])
                assert stop - 1 <= slot - profile["label_delay_slots"]
                private_theta = logistic_update(theta, x[start:stop], y[start:stop],
                                                profile["gradient_steps_per_worker"],
                                                profile["learning_rate"])
                active = {"parameters": private_theta,
                          "launch_slot": slot,
                          "complete_slot": slot + profile["deployment_delay_slots"],
                          "gradient_steps": profile["gradient_steps_per_worker"]}
                # The worker changed its private copy; theta remains deployed.
                jobs += 1
                steps += profile["gradient_steps_per_worker"]
                last_consumed_labels = arrived_count
                launches.append({"slot": slot, "last_training_label_origin": stop - 1,
                                 "complete_slot": active["complete_slot"],
                                 "training_rows": stop - start})
            records.append({"arm": arm, "slot": slot, "prediction": predictions[slot],
                            "label_for_retrospective_scoring": y[slot], "version": version,
                            "theta_0": theta[0], "theta_1": theta[1],
                            "worker_launched": bool(launched), "deployed_now": deployed_now,
                            "active_complete_slot": active["complete_slot"] if active else None,
                            "gradient_steps_so_far": steps})
        correct = (predictions >= 0.5) == y
        arms[arm] = {
            "pre_shift_accuracy": float(np.mean(correct[:shift_slot])),
            "post_shift_accuracy": float(np.mean(correct[shift_slot:])),
            "tail_256_accuracy": float(np.mean(correct[-256:])),
            "workers_launched": jobs, "completed_deployments": deployments,
            "actual_gradient_steps": steps, "simulated_gradient_cost": steps * 0.001,
            "launches": launches, "deployments": deployment_log,
        }
    return ({"scope": "independent logistic-gradient drift-recoverability diagnostic",
             "not_evaluated": ["DA-RF recovery", "DQN", "PPO", "edge queue service",
                               "equal realized expenditure between strategies"],
             "common_inputs": True, "initial_parameters": initial.tolist(),
             "seed_prefix": 82001, "seed_evaluation": 82002,
             "horizon": horizon, "shift_slot_for_retrospective_reporting": shift_slot,
             "drift_generator": "binary relation flips; no onset signal is supplied to either trigger",
             "profile": profile, "arms": arms,
             "deployment_model": "actual gradients computed at launch on a private copy; deployment waits for simulated delay",
             "cost_note": "same worker profile and maximum budget; actual expenditure can differ",
             "resource_note": "no edge-resource allocation or hardware training-time model"}, records)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--gradient-only", action="store_true",
                        help="Run only the independent actual-gradient recoverability diagnostic")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gradient, records = gradient_recoverability_demo()
    streaming = None if args.gradient_only else streaming_fusion_demo()
    permanent, permanent_records = None, None
    if not args.gradient_only:
        runs, permanent_records = [], []
        for seed in [83001, 83002, 83003]:
            run, rows = permanent_outage_moment_demo(seed=seed)
            runs.append(run)
            permanent_records.extend(rows)
        population_gains = np.array([r["population_mse_complete_enabled_reference"]
                                     - r["tail_100_mean_population_mse_active_subset"] for r in runs])
        empirical_gains = np.array([r["tail_100_empirical_mse_complete_enabled_reference"]
                                    - r["tail_100_empirical_mse_active_subset"] for r in runs])
        permanent = {
            "scope": "three-seed active-subset versus complete-enabled archive mechanism diagnostic",
            "seeds": [83001, 83002, 83003], "windows_per_seed": 400,
            "population_mse_reduction_tail_100_mean": float(population_gains.mean()),
            "population_mse_reduction_tail_100_seed_sd": float(population_gains.std(ddof=1)),
            "empirical_mse_reduction_tail_100_mean": float(empirical_gains.mean()),
            "empirical_mse_reduction_tail_100_seed_sd": float(empirical_gains.std(ddof=1)),
            "runs": runs,
            "not_evaluated": ["online service recovery", "external algorithm superiority",
                              "a population confidence guarantee for learned moments"]}
    script_dir = Path(__file__).resolve().parent
    files = ["run_da_rf_demo.py", "da_rf_fusion.py", "da_rf_predictor.py", "da_rf_coordinator.py"]
    result = {"version": "da-rf-diagnostics-v2-bootstrap-quality", "python_version": platform.python_version(),
              "numpy_version": np.__version__,
              "scope": "separate synthetic diagnostics, not a combined policy or service-benefit experiment",
              "script_sha256": {name: hashlib.sha256((script_dir / name).read_bytes()).hexdigest()
                                for name in files if (script_dir / name).exists()},
              "streaming_fusion": streaming, "permanent_outage_moment": permanent,
              "gradient_recoverability": gradient}
    write_csv(args.out_dir / "da_rf_gradient_recovery.csv", records)
    if streaming is not None:
        flat = []
        for row in streaming["snapshots"]:
            flat.append({"window": row["window"], "forecast": row["forecast"],
                         "target_for_retrospective_scoring": row["target_for_retrospective_scoring"],
                         "label_arrival_window": row["label_arrival_window"],
                         "coverage": row["coverage"], "uncertainty": row["uncertainty"],
                         "all_missing": row["all_missing"], "solver_fallback": row["solver_fallback"],
                         "w1": row["weights"][0], "w2": row["weights"][1], "w3": row["weights"][2]})
        write_csv(args.out_dir / "da_rf_streaming_fusion.csv", flat)
        write_csv(args.out_dir / "da_rf_permanent_outage.csv", permanent_records)
    (args.out_dir / "da_rf_demo_results.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
    for name, arm in gradient["arms"].items():
        print(f"{name}: post accuracy={arm['post_shift_accuracy']:.4f}, "
              f"tail accuracy={arm['tail_256_accuracy']:.4f}, updates={arm['actual_gradient_steps']}")
    if streaming is not None:
        print(f"Streaming fusion: windows={streaming['issued_windows']}, "
              f"labels={streaming['label_records']}, MAE={streaming['mae']:.6f}, "
              f"all-missing={streaming['all_missing_windows']}")
        print(f"Permanent fourth-source outage: population MSE reduction="
              f"{permanent['population_mse_reduction_tail_100_mean']:.6f} "
              f"+/- {permanent['population_mse_reduction_tail_100_seed_sd']:.6f}")


if __name__ == "__main__":
    main()
