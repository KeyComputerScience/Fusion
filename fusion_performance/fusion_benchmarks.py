"""Fusion-only comparisons share executed feedback; solver comparisons share states."""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from closed_loop import ROOT, method_config
from data_pipeline import read_json, write_json
from run_experiments import write_csv
from fusion import EvidenceFusion, FusionSnapshot
from coordinator import Coordinator


def load_feedback(path):
    from logio import FLOAT_COLUMNS
    rows = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = {key: (None if value == "" else value) for key, value in raw.items()}
            row["slot"] = int(row["slot"])
            for key in FLOAT_COLUMNS:
                if key in row:
                    row[key] = float(row[key]) if row[key] is not None else None
            rows.append(row)
    return rows


def corrupt(rows, scenario, experiment, seed, reference):
    rows = copy.deepcopy(rows)
    rng = random.Random(seed + 17011)
    origin = experiment["calibration_slots"]
    lo, hi = experiment["fault_interval"]
    for row in rows:
        if not lo <= row["slot"] - origin <= hi:
            continue
        if scenario in ("performance_outage", "all_outage"):
            row["environment_reward"] = row["observed_training_cost"] = None
        if scenario == "all_outage":
            row["regime"] = None
            for feature in reference["state_columns"]:
                row[feature] = None
        if scenario == "operating_noise":
            for feature in reference["state_columns"]:
                row[feature] += rng.gauss(0, 3 * reference["state_scales"][feature]["std"])
        if scenario == "conflict":
            row["regime"] = "light"
    return rows


def fuse(rows, core, reference, method, seed):
    config = method_config(core, method, seed)
    module = EvidenceFusion(config["fusion"], reference)
    size, windows = config["fusion"]["window_size"], []
    for index, start in enumerate(range(0, len(rows), size), 1):
        windows.append(module.complete_window(rows[start:start + size], index).to_dict())
    return windows


def diagnostics(windows, clean_targets, experiment, core, method, seed, scenario):
    size = core["fusion"]["window_size"]
    change_windows = {math.ceil(onset / size) for onset in experiment["drift_onsets"]}
    valid = [w for w in windows if w["window_index"] >= 2]
    truth = [int(w["window_index"] in change_windows) for w in valid]
    scores = [w["gamma"] if not w["fallback"] else 0.0 for w in valid]
    alarms = [w["gamma"] >= core["coordination"]["trigger"]["on"] and not w["fallback"] for w in valid]
    errors, persistence_errors = [], []
    for window in valid:
        target = clean_targets.get(window["window_index"] + 1)
        observed = clean_targets.get(window["window_index"])
        predictions = window["next_target_predictions"]
        if target is None or observed is None or window["fallback"] or any(p is None and a > 0 for p, a in zip(predictions, window["weights"])):
            continue
        prediction = sum(a * p for a, p in zip(window["weights"], predictions) if p is not None)
        errors.append(abs(prediction - target))
        persistence_errors.append(abs(observed - target))
    detected, delays = 0, []
    for onset in experiment["drift_onsets"]:
        choices = [w["available_from_slot"] - experiment["calibration_slots"] for w in valid
                   if w["gamma"] >= core["coordination"]["trigger"]["on"] and not w["fallback"]
                   and w["completed_slot"] - experiment["calibration_slots"] >= onset
                   and w["available_from_slot"] - experiment["calibration_slots"] <= onset + 2 * size]
        if choices:
            detected += 1
            delays.append(min(choices) - onset)
    false = sum(alarm and not label for alarm, label in zip(alarms, truth))
    return {"method": method, "seed": seed, "scenario": scenario,
            "roc_auc": float(roc_auc_score(truth, scores)), "average_precision": float(average_precision_score(truth, scores)),
            "nominal_window_alarm_fraction": false / max(sum(not t for t in truth), 1),
            "detected_changes": detected, "change_count": len(experiment["drift_onsets"]),
            "detection_delay_slots": float(np.mean(delays)) if delays else None,
            "forecast_mae": float(np.mean(errors)) if errors else None,
            "forecast_pairs": len(errors), "persistence_mae": float(np.mean(persistence_errors)) if persistence_errors else None,
            "fallback_windows": sum(w["fallback"] for w in valid),
            "coverage_mean": float(np.mean([w["coverage"] for w in valid])),
            "maximum_source_weight": max(max(w["weights"]) for w in valid),
            "reference_feedback": "dqn_no_rt executed log; identical for all fusion variants"}


def expanded_profiles(profiles, training_count, inference_count):
    result = {"base_training_profile": "tr0", "training": [], "inference": []}
    source = profiles["training"]
    for index in range(training_count):
        x = index * (len(source) - 1) / (training_count - 1)
        a, b, ratio = int(math.floor(x)), int(math.ceil(x)), x % 1
        profile = {"id": f"tr{index}", "updates": max(0, round(x * 2))}
        for key in ("gain", "cost", "intensity"):
            profile[key] = (1 - ratio) * source[a][key] + ratio * source[b][key]
        profile["resources"] = [(1 - ratio) * va + ratio * vb for va, vb in zip(source[a]["resources"], source[b]["resources"])]
        profile["deployment_delay"] = max(1, round((1 - ratio) * source[a]["deployment_delay"] + ratio * source[b]["deployment_delay"]))
        result["training"].append(profile)
    source = profiles["inference"]
    for index in range(inference_count):
        x = index * (len(source) - 1) / (inference_count - 1)
        a, b, ratio = int(math.floor(x)), int(math.ceil(x)), x % 1
        result["inference"].append({"id": f"inf{index}", "beta": (1 - ratio) * source[a]["beta"] + ratio * source[b]["beta"],
            "resources": [(1 - ratio) * va + ratio * vb for va, vb in zip(source[a]["resources"], source[b]["resources"])],
            "quality_prior": {key: (1 - ratio) * source[a]["quality_prior"][key] + ratio * source[b]["quality_prior"][key] for key in source[a]["quality_prior"]}})
    return result


def solver_audit(experiment, core, profiles, output):
    expanded = expanded_profiles(profiles, experiment["solver_audit"]["training_profiles"], experiment["solver_audit"]["inference_profiles"])
    rows = []
    for seed in experiment["seeds"]:
        rng = random.Random(seed + 41071)
        for index in range(experiment["solver_audit"]["states_per_seed"]):
            snapshot = FusionSnapshot(gamma=rng.uniform(0.15, 0.7), uncertainty=rng.uniform(0, 0.7), coverage=1.0, fallback=False)
            config = copy.deepcopy(core["coordination"])
            config["initial_recovery"] = rng.uniform(0.01, 4)
            capacities = [[rng.uniform(0.4, 1.2) for _ in range(3)] for _ in range(config["nodes"])]
            training_load, inference_load = rng.uniform(0.1, 0.8), rng.uniform(0.3, 1.0)
            decisions, times = {}, {}
            for solver in ("exact", "ao"):
                engine = Coordinator(config, expanded)
                stamp = time.perf_counter()
                decisions[solver] = engine.choose(index + 1, training_load, inference_load, capacities, snapshot, 16, solver)
                times[solver] = (time.perf_counter() - stamp) * 1000
            exact, ao = decisions["exact"], decisions["ao"]
            if exact.status != ao.status or (exact.objective is not None and ao.objective > exact.objective + 1e-12):
                raise AssertionError("AO feasibility or exact comparison failed")
            rows.append({"seed": seed, "state": index, "profile_pairs": len(expanded["training"]) * len(expanded["inference"]),
                         "exact_objective": exact.objective, "ao_objective": ao.objective,
                         "proxy_gap": exact.objective - ao.objective if exact.objective is not None else None,
                         "exact_ms": times["exact"], "ao_ms": times["ao"],
                         "exact_pairs": exact.evaluated_pairs, "ao_pairs": ao.evaluated_pairs,
                         "same_pair": (exact.training_profile, exact.inference_profile) == (ao.training_profile, ao.inference_profile),
                         "coordinate_converged": ao.coordinate_converged,
                         "scope": "controlled synthetic states, interpolated profiles, current proxy only"})
    write_csv(output / "solver_audit.csv", rows)
    write_json(output / "expanded_profiles.json", expanded)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    experiment, core = read_json(ROOT / "experiment_config.json"), read_json(ROOT / "core/config.json")
    # Keep the stress perturbation on the first injected change; declared before any evaluation.
    methods = ["fusion", "equal", "fixed", "workload", "operating", "performance", "no_predictive", "uncapped", "no_disagreement"]
    summaries = []
    for seed in experiment["seeds"]:
        path = args.results / "runs" / f"dqn_no_rt_seed_{seed}" / "slots.csv.gz"
        rows = load_feedback(path)
        if any(row["provenance"] != "controlled_synthetic" for row in rows):
            raise ValueError("Known synthetic shift labels cannot score imported trace streams")
        reference = read_json(args.results / "calibration" / f"dqn_seed_{seed}" / "reference.json")
        clean = fuse(rows, core, reference, "fusion", seed)
        clean_targets = {w["window_index"]: w["scores"][2] for w in clean}
        for scenario in ["clean"] + experiment["stress_scenarios"]:
            observations = rows if scenario == "clean" else corrupt(rows, scenario, experiment, seed, reference)
            compared = methods if scenario == "clean" else ["fusion", "equal", "uncapped", "workload"]
            for method in compared:
                windows = fuse(observations, core, reference, method, seed)
                summaries.append(diagnostics(windows, clean_targets, experiment, core, method, seed, scenario))
                write_json(args.results / "fusion" / f"{scenario}_{method}_seed_{seed}.json", windows)
        print(f"Fusion-only benchmark seed={seed} complete", flush=True)
    write_csv(args.results / "fusion_seed_results.csv", summaries)
    write_json(args.results / "fusion_seed_results.json", summaries)
    audit = solver_audit(experiment, core, read_json(ROOT / "core/profiles.json"), args.results)
    write_json(args.results / "solver_audit.json", audit)
    print(f"Solver audit complete: {len(audit)} paired states", flush=True)


if __name__ == "__main__":
    main()
