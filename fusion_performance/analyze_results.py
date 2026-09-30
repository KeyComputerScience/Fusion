"""Validate executed logs and summarize paired seeds without substituting returns."""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np

from data_pipeline import ROOT, read_json, sha256, write_json
from run_experiments import write_csv

LABELS = {"fusion": "Reliability fusion", "no_rt": "No retraining", "periodic": "Periodic",
          "reactive": "Workload trigger", "dpp": "Deficit heuristic", "equal": "Equal weights",
          "fixed": "Fixed weights", "workload": "Workload only", "operating": "Operating only",
          "performance": "Performance only", "no_predictive": "Without predictive calibration",
          "uncapped": "Without weight cap", "no_disagreement": "Without disagreement",
          "no_delayed": "Without delayed value"}


def describe(values):
    values = [float(x) for x in values if x is not None]
    return {"mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None}


def paired_statistics(values, reference):
    difference = np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)
    n = len(difference)
    means = np.asarray([difference[list(indices)].mean() for indices in itertools.product(range(n), repeat=n)])
    low, high = np.quantile(means, [0.025, 0.975])
    # Conditional sign enumeration also handles ties in absolute differences.
    nonzero = difference[np.abs(difference) > 1e-12]
    ranks = np.asarray([1 + sum(abs(y) < abs(x) for y in nonzero)
                        + (sum(abs(y) == abs(x) for y in nonzero) - 1) / 2 for x in nonzero])
    observed = abs(float(np.sum(np.sign(nonzero) * ranks)))
    alternatives = [abs(float(np.sum(np.asarray(signs) * ranks)))
                    for signs in itertools.product((-1, 1), repeat=len(nonzero))]
    pvalue = sum(x >= observed - 1e-12 for x in alternatives) / len(alternatives)
    return {"paired_difference": describe(difference), "bootstrap_ci95": [float(low), float(high)],
            "signed_rank_p_two_sided": pvalue, "nonzero_pairs": len(nonzero),
            "seed_pairs": n, "bootstrap_resamples": len(means),
            "interpretation": "exploratory; five paired seeds; unadjusted conditional signed-rank test"}


def validate_logs(results, rows, experiment, manifest):
    slots, events, failures = 0, 0, []
    total = experiment["evaluation_slots"]
    prefix = experiment["calibration_slots"]
    windows = total // manifest["core_configuration"]["fusion"]["window_size"]
    sources = {}
    for entry in manifest["inputs"]:
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise AssertionError("Exogenous file hash differs from the run manifest")
    for record in rows:
        key = f"{record['backbone']}_{record['method']}_seed_{record['seed']}"
        directory = results / "runs" / key
        checkpoints = results / "calibration" / f"{record['backbone']}_seed_{record['seed']}"
        source_key = record["seed"]
        if source_key in sources and sources[source_key] != record["input_sha256"]:
            failures.append(f"{key}: unpaired exogenous input")
        sources[source_key] = record["input_sha256"]
        if sha256(checkpoints / "initial_policy.pt") != record["calibration_sha256"]:
            failures.append(f"{key}: calibration checkpoint hash mismatch")
        if record["resource_violations"] != 0 or record["complete_windows"] != windows:
            failures.append(f"{key}: resource or window count mismatch")
        if record["arrival_count"] != record["completed_count"] + record["deadline_count"] + record["queue_end"]:
            failures.append(f"{key}: task count mismatch")
        gained_slots, cost, rewards, count = [], 0., 0., 0
        with gzip.open(directory / "slots.csv.gz", "rt", newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                count += 1
                slot = int(raw["slot"])
                if slot != prefix + count or int(raw["fusion_completed_slot"]) >= slot:
                    failures.append(f"{key}: feedback chronology mismatch")
                if raw["inference_profile"] != raw["logged_inference"]:
                    failures.append(f"{key}: selected inference not executed")
                if int(raw["application_action"]) not in (0, 1, 2):
                    failures.append(f"{key}: service action outside three-action space")
                queue = min(1., float(raw["queue_after"]) / (10 * experiment["queue_scale"]))
                weights = experiment["reward_weights"]
                expected = (weights["completion"] * float(raw["completion_ratio"])
                            - weights["deadline"] * float(raw["deadline_loss"])
                            - weights["queue"] * queue
                            - weights["training"] * float(raw["observed_training_cost"]))
                reward = float(raw["environment_reward"])
                if not math.isclose(reward, expected, abs_tol=1e-12):
                    failures.append(f"{key}: external reward mismatch")
                after = float(raw["rho"]) * float(raw["recovery_state"]) + float(raw["deployed_gain_used"])
                if not math.isclose(after, float(raw["recovery_after_feedback"]), abs_tol=1e-12):
                    failures.append(f"{key}: recovery transition mismatch")
                if float(raw["deployed_gain_used"]) > 0:
                    gained_slots.append(slot)
                cost += float(raw["observed_training_cost"])
                rewards += reward
        deployment_events = read_json(directory / "deployments.json")
        if gained_slots != [e["deployment_slot"] for e in deployment_events]:
            failures.append(f"{key}: gain outside an actual deployment")
        if any(not e["parameters_changed"] or e["delay"] < e["declared_delay"] for e in deployment_events):
            failures.append(f"{key}: unchanged or early model deployment")
        if count != total or not math.isclose(rewards, record["return"], abs_tol=1e-8):
            failures.append(f"{key}: horizon or return mismatch")
        if not math.isclose(cost, record["training_cost"], abs_tol=1e-10):
            failures.append(f"{key}: cost mismatch")
        slots += count
        events += len(deployment_events)
    if failures:
        raise AssertionError("\n".join(failures[:20]))
    return {"runs": len(rows), "slots_checked": slots, "deployments_checked": events,
            "resource_violations": sum(r["resource_violations"] for r in rows),
            "checks": ["paired exogenous hashes", "initial checkpoint hashes", "task conservation",
                       "inference selection execution", "completed-window chronology", "external reward identity",
                       "recovery identity", "actual parameter-changing deployment", "horizon and window counts"],
            "all_passed": True}


def common_forecasts(results, diagnostic_rows, experiment):
    """Within a scenario use the intersection of eligible targets across variants."""
    output = []
    window_size = read_json(ROOT / "core/config.json")["fusion"]["window_size"]
    for seed in experiment["seeds"]:
        clean = read_json(results / "fusion" / f"clean_fusion_seed_{seed}.json")
        targets = {w["window_index"]: w["scores"][2] for w in clean}
        for scenario in ["clean"] + experiment["stress_scenarios"]:
            compared = [r for r in diagnostic_rows if r["seed"] == seed and r["scenario"] == scenario]
            predictions, observed = {}, {}
            for row in compared:
                windows = read_json(results / "fusion" / f"{scenario}_{row['method']}_seed_{seed}.json")
                values = {}
                for window in windows:
                    index = window["window_index"]
                    ps, ws = window["next_target_predictions"], window["weights"]
                    if (index < 2 or window["fallback"] or targets.get(index) is None
                            or targets.get(index + 1) is None or any(p is None and a > 0 for p, a in zip(ps, ws))):
                        continue
                    values[index] = sum(p * w for p, w in zip(ps, ws) if p is not None)
                predictions[row["method"]] = values
            common = sorted(set.intersection(*(set(values) for values in predictions.values())))
            shift_targets = {math.ceil(t / window_size) for t in experiment["drift_onsets"]}
            changing = [index for index in common if index + 1 in shift_targets]
            for row in compared:
                values = predictions[row["method"]]
                mae = np.mean([abs(values[k] - targets[k + 1]) for k in common]) if common else None
                persistence = np.mean([abs(targets[k] - targets[k + 1]) for k in common]) if common else None
                shift_error = np.mean([abs(values[k] - targets[k + 1]) for k in changing]) if changing else None
                shift_persistence = np.mean([abs(targets[k] - targets[k + 1]) for k in changing]) if changing else None
                output.append({**row, "common_forecast_mae": float(mae) if mae is not None else None,
                               "common_persistence_mae": float(persistence) if persistence is not None else None,
                               "common_forecast_pairs": len(common), "common_forecast_windows": common,
                               "change_target_mae": float(shift_error) if shift_error is not None else None,
                               "change_target_persistence_mae": float(shift_persistence) if shift_persistence is not None else None,
                               "change_target_pairs": len(changing)})
    return output


def summarize(results):
    experiment = read_json(ROOT / "experiment_config.json")
    manifest = read_json(results / "run_manifest.json")
    if manifest["configuration"] != experiment or manifest["core_configuration"] != read_json(ROOT / "core/config.json"):
        raise ValueError("Current configurations differ from the executed experiment")
    rows = read_json(results / "seed_results.json")
    expected = len(experiment["seeds"]) * (len(experiment["backbones"]) * len(experiment["main_methods"]) + len(experiment["ablation_methods"]))
    if manifest["data_kind"] != "controlled_synthetic_closed_loop" or len(rows) != expected:
        raise ValueError("Manuscript tables require the complete declared controlled experiment")
    if any(r["data_kind"] != "controlled_synthetic" or r["evaluation_slots"] != experiment["evaluation_slots"] for r in rows):
        raise ValueError("Do not mix real-trace or smoke outputs with controlled experiment tables")
    validation = validate_logs(results, rows, experiment, manifest)
    metrics = ["return", "completion_fraction", "deadline_fraction", "p95_latency_slots", "training_jobs",
               "deployed_jobs", "gradient_updates", "training_cost", "coordinator_mean_ms", "runtime_seconds"]
    main, ablation, pairs = [], [], []
    lookup = {(r["backbone"], r["method"], r["seed"]): r for r in rows}
    for backbone in experiment["backbones"]:
        methods = experiment["main_methods"] + (experiment["ablation_methods"] if backbone == "dqn" else [])
        for method in methods:
            group = [lookup[backbone, method, seed] for seed in experiment["seeds"]]
            item = {"backbone": backbone, "method": method, "label": LABELS[method],
                    **{metric: describe([r[metric] for r in group]) for metric in metrics}}
            if method in experiment["main_methods"]:
                main.append(item)
            if backbone == "dqn" and method in ["fusion"] + experiment["ablation_methods"]:
                ablation.append(item)
            if method != "no_rt":
                baseline = [lookup[backbone, "no_rt", seed]["return"] for seed in experiment["seeds"]]
                pairs.append({"backbone": backbone, "method": method, "reference": "no_rt",
                              **paired_statistics([r["return"] for r in group], baseline)})
            if method != "fusion":
                baseline = [lookup[backbone, "fusion", seed]["return"] for seed in experiment["seeds"]]
                pairs.append({"backbone": backbone, "method": method, "reference": "fusion",
                              **paired_statistics([r["return"] for r in group], baseline)})
    raw = read_json(results / "fusion_seed_results.json")
    diagnostics = common_forecasts(results, raw, experiment)
    fusion_metrics = ["roc_auc", "average_precision", "nominal_window_alarm_fraction", "detected_changes",
                      "detection_delay_slots", "forecast_mae", "persistence_mae", "common_forecast_mae",
                      "common_persistence_mae", "common_forecast_pairs", "fallback_windows", "coverage_mean",
                      "maximum_source_weight", "change_target_mae", "change_target_persistence_mae", "change_target_pairs"]
    fusion_rows = []
    for scenario, method in dict.fromkeys((r["scenario"], r["method"]) for r in diagnostics):
        group = [r for r in diagnostics if (r["scenario"], r["method"]) == (scenario, method)]
        fusion_rows.append({"scenario": scenario, "method": method, "label": LABELS[method],
                            **{metric: describe([r[metric] for r in group]) for metric in fusion_metrics},
                            "total_detected": sum(r["detected_changes"] for r in group),
                            "total_changes": sum(r["change_count"] for r in group)})
    audit = read_json(results / "solver_audit.json")
    feasible = [r for r in audit if r["proxy_gap"] is not None]
    solver = {"states": len(audit), "feasible_states": len(feasible), "pairs": audit[0]["profile_pairs"],
              "maximum_proxy_gap": max(r["proxy_gap"] for r in feasible),
              "same_pair_fraction": sum(r["same_pair"] for r in feasible) / len(feasible),
              "coordinate_convergence_fraction": sum(r["coordinate_converged"] for r in feasible) / len(feasible)}
    for metric in ["proxy_gap", "exact_ms", "ao_ms", "exact_pairs", "ao_pairs"]:
        seed_means = [float(np.mean([r[metric] for r in feasible if r["seed"] == seed])) for seed in experiment["seeds"]]
        solver[metric] = describe(seed_means)
    solver["total_time_ratio"] = sum(r["exact_ms"] for r in feasible) / sum(r["ao_ms"] for r in feasible)
    report = {"validation": validation, "main": main, "ablation": ablation, "paired_comparisons": pairs,
              "fusion_diagnostics": fusion_rows, "solver": solver, "runtime": manifest,
              "statistics": {"replicates": "five paired seeds", "spread": "sample standard deviation",
                             "ci": "paired seed percentile bootstrap; all 3125 index resamples",
                             "p": "two-sided conditional sign enumeration of signed ranks; zeros excluded",
                             "smallest_p_with_five_nonzero_pairs": 0.0625,
                             "limitations": "No confirmatory p<0.05 claim; no adjustment for exploratory comparisons."}}
    write_json(results / "analysis.json", report)
    write_json(results / "validation.json", validation)
    write_json(results / "fusion_common_forecasts.json", diagnostics)
    write_csv(results / "main_summary.csv", [
        {"backbone": r["backbone"], "method": r["method"], **{f"{metric}_{stat}": r[metric][stat] for metric in metrics for stat in ("mean", "sd")}}
        for r in main])
    write_csv(results / "paired_comparisons.csv", [
        {"backbone": r["backbone"], "method": r["method"], "reference": r["reference"],
         "mean_difference": r["paired_difference"]["mean"], "ci95_low": r["bootstrap_ci95"][0],
         "ci95_high": r["bootstrap_ci95"][1], "p_two_sided": r["signed_rank_p_two_sided"]} for r in pairs])
    print(json.dumps({"validated_runs": len(rows), "validated_slots": validation["slots_checked"],
                      "resource_violations": validation["resource_violations"], "solver_states": solver["states"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    summarize(parser.parse_args().results)
