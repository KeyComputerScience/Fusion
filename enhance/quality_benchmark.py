"""Frozen observer-quality replay on genuinely executed, synthetic NoRT logs.

No service outcomes are generated here. Clean future targets are used only by
the offline scorer. Interventions alter the fusion-visible copy of slot logs.
The package core and this runner use the standard library; no Torch is imported.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "core"))
from fusion import EvidenceFusion, capped_weights
from logio import FLOAT_COLUMNS


DESIGN = {
    "version": "quality-replay-1.0",
    "scope": "controlled_synthetic_observer_quality_replay_on_executed_DQN_NoRT_logs",
    "seeds": [10, 20, 30, 40, 50],
    "calibration_slots": 1200,
    "evaluation_slots": 3000,
    "window_size": 50,
    "event_onsets": [901, 1801],
    "event_label_origin": "known exogenous synthetic phase switches, never used by online methods",
    "event_detection_tolerance_slots": 100,
    "fault_intervals_relative_to_evaluation": [[401, 500], [901, 1000], [1401, 1500], [1801, 1900]],
    "scenarios": ["clean", "service_outage", "operating_noise", "workload_conflict", "simultaneous_loss"],
    "operating_noise_standard_deviations": 3.0,
    "workload_conflict_reports_by_interval": ["cpu_heavy", "light", "light", "cpu_heavy"],
    "corruption_rng_offset": 17011,
    "alarm_on": 0.15,
    "alarm_off": 0.08,
    "coverage_minimum": 2.0 / 3.0,
    "uncertainty_maximum": 0.8,
    "forecast_horizon": "one completed window ahead",
    "target": "clean context-matched service degradation from NoRT executed logs",
    "common_target_policy": "intersection of valid forecasts across every reported method within each seed/scenario",
    "nominal_alarm_denominator": "completed windows k>=2 other than the two actual event windows",
    "unit": "abstract service slot; these existing simulations do not imply 60-second hardware slots",
    "prospective_status": "extension designed after audit of manuscript and archived results; not external preregistration",
    "keep_negative_results": True,
    "revision_parameters": {"entropy_temperature": 1.0, "residual_penalty": 1.0, "inertia": 0.1, "residual_decay": 0.9},
    "modern_baseline_parameters": {"learning_rate": 0.5, "error_rate": 0.15, "epsilon_mse": 0.0025, "ds_consistency_scale": 0.2},
    "fixed_baseline_calibration": "inverse mean squared error of stored one-window-ahead source forecasts within calibration prefix, frozen for evaluation",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refuse to generate an empty results table")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def load_feedback(path, expected_slots=None, origin=None):
    rows = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = {key: (None if value == "" else value) for key, value in raw.items()}
            row["slot"] = int(row["slot"])
            for key in FLOAT_COLUMNS:
                if row.get(key) is not None:
                    row[key] = float(row[key])
                    if not math.isfinite(row[key]):
                        raise ValueError(f"Nonfinite feedback {key} at {row['slot']}")
            rows.append(row)
    expected_slots = DESIGN["evaluation_slots"] if expected_slots is None else expected_slots
    origin = DESIGN["calibration_slots"] + 1 if origin is None else origin
    if len(rows) != expected_slots:
        raise ValueError("The benchmark requires the declared number of executed slots")
    expected = list(range(origin, origin + len(rows)))
    if [r["slot"] for r in rows] != expected:
        raise ValueError("Evaluation feedback must be contiguous and start after the calibration prefix")
    if any(r.get("provenance") != "controlled_synthetic" for r in rows):
        raise ValueError("Do not apply known synthetic event labels to another provenance")
    return rows


def fit_fixed_weights(prefix, configuration, reference, seed):
    historical_reference = copy.deepcopy(reference)
    historical_reference["fit_through_slot"] = 0
    module = EvidenceFusion({**configuration, "seed": seed + 20260930}, historical_reference)
    pending, losses = None, [[], [], []]
    for index, start in enumerate(range(0, len(prefix), DESIGN["window_size"]), 1):
        current = module.complete_window(prefix[start:start + DESIGN["window_size"]], index)
        target = current.scores[2]
        if target is not None and pending is not None:
            for source, estimate in enumerate(pending.next_target_predictions):
                if pending.masks[source] and estimate is not None:
                    losses[source].append((estimate - target) ** 2)
        pending = current
    if any(not values for values in losses):
        raise ValueError("FF historical calibration requires support for every source")
    errors = [statistics.fmean(values) for values in losses]
    precision = [1.0 / (value + DESIGN["modern_baseline_parameters"]["epsilon_mse"]) for value in errors]
    return {"weights": [value / sum(precision) for value in precision], "historical_mse": errors,
            "historical_forecast_pairs": [len(values) for values in losses], "evaluation_targets_used": False}


def corrupt_feedback(clean, scenario, seed, reference):
    """Corrupt only observer fields; never change physical rewards in clean."""
    rows = copy.deepcopy(clean)
    rng = random.Random(seed + DESIGN["corruption_rng_offset"])
    interventions = []
    for row in rows:
        relative = row["slot"] - DESIGN["calibration_slots"]
        fault_id = next((index for index, (low, high) in enumerate(DESIGN["fault_intervals_relative_to_evaluation"])
                         if low <= relative <= high), None)
        if fault_id is None or scenario == "clean":
            continue
        changed = {}
        if scenario in ("service_outage", "simultaneous_loss"):
            for feature in ("environment_reward", "observed_training_cost"):
                changed[feature] = {"before": row[feature], "after": None}
                row[feature] = None
        if scenario == "simultaneous_loss":
            changed["regime"] = {"before": row["regime"], "after": None}
            row["regime"] = None
            for feature in reference["state_columns"]:
                changed[feature] = {"before": row[feature], "after": None}
                row[feature] = None
        if scenario == "operating_noise":
            for feature in reference["state_columns"]:
                before = row[feature]
                after = before + rng.gauss(0.0, DESIGN["operating_noise_standard_deviations"] * reference["state_scales"][feature]["std"])
                changed[feature] = {"before": before, "after": after}
                row[feature] = after
        if scenario == "workload_conflict":
            after = DESIGN["workload_conflict_reports_by_interval"][fault_id]
            changed["regime"] = {"before": row["regime"], "after": after}
            row["regime"] = after
        interventions.append({"slot": row["slot"], "evaluation_slot": relative, "fault_interval": fault_id, "changed": changed})
    # All untouched fields, including physical completion/deadline counts, agree.
    allowed = {"environment_reward", "observed_training_cost", "regime", *reference["state_columns"]}
    for original, damaged in zip(clean, rows):
        if any(original[key] != damaged[key] for key in original if key not in allowed):
            raise AssertionError("An observer corruption changed physical execution")
    return rows, interventions


def pack_snapshot(snapshot, revised=None):
    item = snapshot.to_dict()
    if revised is not None:
        item.update(revised.extras(snapshot))
    forecast = item.get("forecast_score", item.get("forecast_probability"))
    if forecast is None and item.get("weights") is not None:
        terms = [(weight, estimate) for weight, estimate in zip(item["weights"], item["next_target_predictions"])
                 if weight > 0]
        if terms and all(estimate is not None for _, estimate in terms):
            forecast = sum(weight * estimate for weight, estimate in terms)
    item["forecast_score"] = forecast
    item["forecast_valid"] = bool(forecast is not None and not item.get("fallback", False))
    return item


def remove_consistency_factor(snapshot, configuration):
    """Ablate only qE while retaining the same causal affine calibrators."""
    result = copy.deepcopy(snapshot)
    if result.fallback:
        return result
    reliability = [r * math.exp(error / configuration["error_scale"])
                   for r, error in zip(result.reliability, result.predictive_errors)]
    weights = capped_weights(reliability, configuration["maximum_source_weight"])
    result.reliability, result.weights = reliability, weights
    result.gamma = sum(weight * score for weight, score in zip(weights, result.scores) if score is not None)
    result.disagreement = sum(weight * (score - result.gamma) ** 2 for weight, score in zip(weights, result.scores) if score is not None)
    result.uncertainty = min(1.0, 4.0 * result.disagreement + configuration["missing_source_penalty"] * (1.0 - result.coverage))
    return result


def method_specs():
    variants = [
        {"name": "original_reliability", "kind": "original"},
        {"name": "original_without_qN", "kind": "original", "config": {"use_support": False}},
        {"name": "original_without_qA", "kind": "original", "config": {"use_freshness": False}},
        {"name": "original_without_qV", "kind": "original", "config": {"use_noise": False}},
        {"name": "original_without_qE", "kind": "original", "remove_qE_only": True},
        {"name": "original_without_cap", "kind": "original", "config": {"maximum_source_weight": 1.0}},
        {"name": "original_without_disagreement", "kind": "original", "config": {"use_disagreement": False}},
        {"name": "original_without_missing_penalty", "kind": "original", "config": {"missing_source_penalty": 0.0}},
        {"name": "original_without_U_gate", "kind": "original", "disable_U_gate": True},
        {"name": "revised_residual_fusion", "kind": "revised"},
        {"name": "revised_without_residual_consistency", "kind": "revised", "revision": {"residual_penalty": 0.0}},
        {"name": "revised_without_inertia", "kind": "revised", "revision": {"inertia": 0.0}},
        {"name": "revised_without_qN", "kind": "revised", "config": {"use_support": False}},
        {"name": "revised_without_qA", "kind": "revised", "config": {"use_freshness": False}},
        {"name": "revised_without_qV", "kind": "revised", "config": {"use_noise": False}},
        {"name": "revised_without_cap", "kind": "revised", "config": {"maximum_source_weight": 1.0}},
        {"name": "revised_without_disagreement", "kind": "revised", "config": {"use_disagreement": False}},
        {"name": "revised_with_raw_disagreement", "kind": "revised", "revision": {"use_forecast_disagreement": False}},
        {"name": "revised_without_missing_penalty", "kind": "revised", "config": {"missing_source_penalty": 0.0}},
        {"name": "revised_without_U_gate", "kind": "revised", "disable_U_gate": True},
    ]
    variants.extend({"name": "baseline_" + mode, "kind": "baseline", "mode": mode}
                    for mode in ("EF", "FF", "IMSE", "EWA", "BOA", "DS"))
    variants.extend({"name": "baseline_" + mode + "_capped", "kind": "baseline", "mode": mode, "apply_cap": 0.7}
                    for mode in ("IMSE", "BOA"))
    return variants


def run_method(rows, configuration, reference, spec, seed):
    config = copy.deepcopy(configuration)
    config.update(spec.get("config", {}))
    config["seed"] = seed + 20260930
    reviser = None
    if spec["kind"] == "baseline":
        from modern_baselines import ReplayBaseline
        module = ReplayBaseline(spec["mode"], config, reference)
    else:
        module = EvidenceFusion(config, reference)
        if spec["kind"] == "revised":
            from revised_fusion import RevisedFusion
            settings = {**DESIGN["revision_parameters"], **spec.get("revision", {})}
            reviser = RevisedFusion(config, **settings)
    windows, hysteresis = [], False
    size = config["window_size"]
    for index, start in enumerate(range(0, len(rows), size), 1):
        raw = module.complete_window(rows[start:start + size], index)
        if spec.get("apply_cap") and not raw.fallback:
            raw.weights = capped_weights(raw.weights, spec["apply_cap"])
            raw.gamma = sum(weight * score for weight, score in zip(raw.weights, raw.scores) if score is not None)
            raw.forecast_score = sum(weight * prediction for weight, prediction in zip(raw.weights, raw.next_target_predictions) if prediction is not None)
            raw.forecast_probability = raw.forecast_score
            raw.disagreement = sum(weight * (prediction - raw.forecast_score) ** 2
                                   for weight, prediction in zip(raw.weights, raw.next_target_predictions) if prediction is not None)
            raw.uncertainty = min(1.0, 4 * raw.disagreement + config["missing_source_penalty"] * (1.0 - raw.coverage))
            raw.effective_weight_cap = max(spec["apply_cap"], 1.0 / sum(raw.masks))
        if spec.get("remove_qE_only"):
            raw = remove_consistency_factor(raw, config)
        if reviser is not None:
            raw = reviser.update(raw)
        item = pack_snapshot(raw, reviser)
        if not item["fallback"]:
            if item["gamma"] >= DESIGN["alarm_on"]:
                hysteresis = True
            elif item["gamma"] <= DESIGN["alarm_off"]:
                hysteresis = False
        # Missing evidence never counts as an affirmative admission.
        admissible = (hysteresis and not item["fallback"]
                      and item["coverage"] >= DESIGN["coverage_minimum"]
                      and (spec.get("disable_U_gate", False) or item["uncertainty"] <= DESIGN["uncertainty_maximum"]))
        item.update(method=spec["name"], method_kind=spec["kind"], seed=seed,
                    evaluation_completed_slot=item["completed_slot"] - DESIGN["calibration_slots"],
                    raw_window_alarm=not item["fallback"] and item["gamma"] >= DESIGN["alarm_on"],
                    hysteresis_active=hysteresis, adaptation_admissible=bool(admissible),
                    disable_U_gate=spec.get("disable_U_gate", False))
        windows.append(item)
    return windows


def group_auc(truth, scores):
    positive = [score for label, score in zip(truth, scores) if label]
    negative = [score for label, score in zip(truth, scores) if not label]
    if not positive or not negative:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def group_average_precision(truth, scores):
    total = sum(truth)
    if not total:
        return None
    found, seen, value = 0, 0, 0.0
    for threshold in sorted(set(scores), reverse=True):
        labels = [label for label, score in zip(truth, scores) if score == threshold]
        added = sum(labels)
        found += added
        seen += len(labels)
        value += added / total * found / seen
    return value


def detect_events(windows, key):
    delays, missed = [], []
    for onset in DESIGN["event_onsets"]:
        hits = [w["available_from_slot"] - DESIGN["calibration_slots"] - onset
                for w in windows if w["window_index"] >= 2 and w[key]
                and w["evaluation_completed_slot"] >= onset
                and 0 <= w["available_from_slot"] - DESIGN["calibration_slots"] - onset <= DESIGN["event_detection_tolerance_slots"]]
        if hits:
            delays.append(min(hits))
        else:
            missed.append(onset)
    return len(delays), statistics.fmean(delays) if delays else None, missed


def summarize(windows, clean_targets, common_indices, scenario, seed):
    valid = [item for item in windows if item["window_index"] >= 2]
    event_windows = {math.ceil(onset / DESIGN["window_size"]) for onset in DESIGN["event_onsets"]}
    nominal = [item for item in valid if item["window_index"] not in event_windows]
    available = [item for item in valid if item["forecast_valid"] and clean_targets.get(item["window_index"] + 1) is not None]
    common = [item for item in available if item["window_index"] in common_indices]
    scores = [item["gamma"] if not item["fallback"] else 0.0 for item in valid]
    truth = [int(item["window_index"] in event_windows) for item in valid]
    raw_count, raw_delay, raw_missed = detect_events(valid, "raw_window_alarm")
    gate_count, gate_delay, gate_missed = detect_events(valid, "adaptation_admissible")
    weights = [max(item["weights"]) for item in valid if item.get("weights")]
    errors = [abs(item["forecast_score"] - clean_targets[item["window_index"] + 1]) for item in common]
    unrestricted = [abs(item["forecast_score"] - clean_targets[item["window_index"] + 1]) for item in available]
    persistence = [abs(clean_targets[item["window_index"]] - clean_targets[item["window_index"] + 1])
                   for item in common if clean_targets.get(item["window_index"]) is not None]
    return {"scenario": scenario, "seed": seed, "method": windows[0]["method"],
            "method_kind": windows[0]["method_kind"], "nominal_windows": len(nominal),
            "raw_alarm_fraction": sum(item["raw_window_alarm"] for item in nominal) / len(nominal),
            "admission_fraction": sum(item["adaptation_admissible"] for item in nominal) / len(nominal),
            "raw_false_alarm_windows": sum(item["raw_window_alarm"] for item in nominal),
            "admission_false_alarm_windows": sum(item["adaptation_admissible"] for item in nominal),
            "raw_events_detected": raw_count, "admitted_events_detected": gate_count,
            "event_count": len(DESIGN["event_onsets"]),
            "raw_delay_slots": raw_delay, "admitted_delay_slots": gate_delay,
            "raw_missed_events": raw_missed, "admitted_missed_events": gate_missed,
            "forecast_mae_common": statistics.fmean(errors) if errors else None,
            "forecast_pairs_common": len(errors),
            "forecast_mae_available": statistics.fmean(unrestricted) if unrestricted else None,
            "forecast_pairs_available": len(unrestricted),
            "persistence_mae_common": statistics.fmean(persistence) if persistence else None,
            "maximum_linear_weight": max(weights) if weights else None,
            "mean_coverage": statistics.fmean(item["coverage"] for item in valid),
            "mean_uncertainty": statistics.fmean(item["uncertainty"] for item in valid),
            "fallback_windows": sum(item["fallback"] for item in valid),
            "roc_auc_raw_support": group_auc(truth, scores),
            "average_precision_raw_support": group_average_precision(truth, scores),
            "nonlinear_combination": any(item.get("nonlinear_combination", False) for item in valid),
            "unit": "abstract slots", "service_effect_scope": "observer replay only; no counterfactual deployments or service returns"}


def aggregate(seed_rows):
    result = []
    grouping = sorted({(row["scenario"], row["method"]) for row in seed_rows})
    numeric = ["raw_alarm_fraction", "admission_fraction", "raw_false_alarm_windows", "admission_false_alarm_windows",
               "raw_delay_slots", "admitted_delay_slots", "forecast_mae_common", "forecast_mae_available",
               "forecast_pairs_common", "forecast_pairs_available", "persistence_mae_common", "maximum_linear_weight",
               "mean_coverage", "mean_uncertainty", "fallback_windows", "roc_auc_raw_support", "average_precision_raw_support"]
    for scenario, method in grouping:
        selected = [row for row in seed_rows if row["scenario"] == scenario and row["method"] == method]
        output = {"scenario": scenario, "method": method, "seed_count": len(selected),
                  "raw_events_detected_total": sum(row["raw_events_detected"] for row in selected),
                  "admitted_events_detected_total": sum(row["admitted_events_detected"] for row in selected),
                  "event_trials_total": sum(row["event_count"] for row in selected),
                  "distinct_event_types": 2, "nonlinear_combination": selected[0]["nonlinear_combination"]}
        for key in numeric:
            values = [row[key] for row in selected if row[key] is not None]
            mean = statistics.fmean(values) if values else None
            sd = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
            radius = 2.776 * sd / math.sqrt(5) if len(values) == 5 else None
            output.update({key + "_mean": mean, key + "_sd": sd,
                           key + "_ci95_low": mean - radius if radius is not None else None,
                           key + "_ci95_high": mean + radius if radius is not None else None,
                           key + "_n": len(values)})
        result.append(output)
    return result


def paired_comparisons(seed_rows):
    output = []
    rng = random.Random(918271)
    for scenario in DESIGN["scenarios"]:
        methods = sorted({row["method"] for row in seed_rows})
        for comparator in methods:
            if comparator == "revised_residual_fusion":
                continue
            for metric in ("forecast_mae_common", "raw_alarm_fraction", "admission_fraction"):
                differences = []
                for seed in DESIGN["seeds"]:
                    a = next(row for row in seed_rows if row["scenario"] == scenario and row["seed"] == seed and row["method"] == "revised_residual_fusion")
                    b = next(row for row in seed_rows if row["scenario"] == scenario and row["seed"] == seed and row["method"] == comparator)
                    if a[metric] is not None and b[metric] is not None:
                        differences.append(a[metric] - b[metric])
                if not differences:
                    continue
                boot = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(10000))
                output.append({"scenario": scenario, "reference": "revised_residual_fusion", "comparator": comparator,
                               "metric": metric, "direction": "negative favors revised for error/alarm fractions",
                               "paired_n": len(differences), "difference_mean": statistics.fmean(differences),
                               "difference_sd": statistics.stdev(differences) if len(differences) > 1 else 0.0,
                               "paired_bootstrap95_low": boot[249], "paired_bootstrap95_high": boot[9749],
                               "scope": "five seeds of one synthetic schedule; no population-level trace generalization"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-root", type=Path, default=HERE.parent / "results/original/runs")
    parser.add_argument("--calibration-root", type=Path, default=HERE.parent / "results/original/calibration")
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/quality")
    args = parser.parse_args()
    started = time.perf_counter()
    core = json.loads((HERE / "core/config.json").read_text())
    if core["fusion"]["window_size"] != DESIGN["window_size"]:
        raise ValueError("Window size differs from frozen design")
    specs = method_specs()
    frozen = {**DESIGN, "methods": specs, "core_configuration": core}
    args.output.mkdir(parents=True, exist_ok=True)
    design_path = args.output / "frozen_design.json"
    if design_path.exists() and json.loads(design_path.read_text()) != frozen:
        raise ValueError("Existing frozen design differs; use a new output directory for a changed study")
    write_json(design_path, frozen)
    # Resolve inputs before any method has access to held-out targets.
    inputs, seed_results = [], []
    for seed in DESIGN["seeds"]:
        log_path = args.feedback_root / f"dqn_no_rt_seed_{seed}" / "slots.csv.gz"
        reference_path = args.calibration_root / f"dqn_seed_{seed}" / "reference.json"
        prefix_path = args.calibration_root / f"dqn_seed_{seed}" / "prefix.csv.gz"
        if not log_path.is_file() or not reference_path.is_file() or not prefix_path.is_file():
            raise FileNotFoundError("Executed NoRT slot logs and their calibration reference are required; archived summaries cannot substitute")
        clean = load_feedback(log_path)
        reference = json.loads(reference_path.read_text())
        prefix = load_feedback(prefix_path, expected_slots=DESIGN["calibration_slots"], origin=1)
        fixed = fit_fixed_weights(prefix, core["fusion"], reference, seed)
        seed_configuration = copy.deepcopy(core["fusion"])
        seed_configuration["modern_baselines"] = {**DESIGN["modern_baseline_parameters"], "fixed_weights": fixed["weights"]}
        write_json(args.output / "fixed_calibration" / f"seed_{seed}.json", fixed)
        inputs.append({"seed": seed, "feedback": str(log_path.resolve()), "feedback_sha256": sha256(log_path),
                       "reference": str(reference_path.resolve()), "reference_sha256": sha256(reference_path),
                       "calibration_prefix": str(prefix_path.resolve()), "calibration_prefix_sha256": sha256(prefix_path),
                       "fixed_weights_calibrated_on_prefix": fixed})
        # This independent clean extractor supplies targets only to the scorer.
        clean_module = EvidenceFusion({**core["fusion"], "seed": seed + 20260930}, reference)
        clean_targets = {}
        for index, start in enumerate(range(0, len(clean), DESIGN["window_size"]), 1):
            target_snapshot = clean_module.complete_window(clean[start:start + DESIGN["window_size"]], index)
            clean_targets[index] = target_snapshot.scores[2]
        for scenario in DESIGN["scenarios"]:
            observed, interventions = corrupt_feedback(clean, scenario, seed, reference)
            write_json(args.output / "interventions" / f"{scenario}_seed_{seed}.json", interventions)
            methods = {}
            for spec in specs:
                methods[spec["name"]] = run_method(observed, seed_configuration, reference, spec, seed)
            index_sets = [{w["window_index"] for w in windows if w["window_index"] >= 2
                           and w["forecast_valid"] and clean_targets.get(w["window_index"] + 1) is not None}
                          for windows in methods.values()]
            common = set.intersection(*index_sets)
            write_json(args.output / "targets" / f"{scenario}_seed_{seed}.json",
                       {"clean_targets_for_offline_scoring_only": clean_targets, "common_forecast_origin_windows": sorted(common)})
            for name, windows in methods.items():
                for window in windows:
                    window["scenario"] = scenario
                    target = clean_targets.get(window["window_index"] + 1)
                    window["offline_clean_next_target"] = target
                    window["in_common_forecast_set"] = window["window_index"] in common
                    window["offline_absolute_error"] = (abs(window["forecast_score"] - target)
                                                        if window["forecast_valid"] and target is not None else None)
                write_json(args.output / "windows" / f"{scenario}_{name}_seed_{seed}.json", windows)
                write_csv(args.output / "windows" / f"{scenario}_{name}_seed_{seed}.csv", windows)
                seed_results.append(summarize(windows, clean_targets, common, scenario, seed))
            print(f"seed={seed} scenario={scenario} methods={len(specs)} common_forecast_pairs={len(common)}", flush=True)
    write_csv(args.output / "seed_results.csv", seed_results)
    write_json(args.output / "seed_results.json", seed_results)
    summary = aggregate(seed_results)
    write_csv(args.output / "summary.csv", summary)
    write_json(args.output / "summary.json", summary)
    write_csv(args.output / "paired_comparisons.csv", paired_comparisons(seed_results))
    sources = [HERE / "quality_benchmark.py", HERE / "modern_baselines.py", HERE / "revised_fusion.py", HERE / "core/fusion.py", HERE / "core/config.json"]
    write_json(args.output / "run_manifest.json", {"design_sha256": sha256(design_path), "inputs": inputs,
               "implementation_hashes": {str(path.resolve()): sha256(path) for path in sources},
               "python": sys.version, "method_seed_scenario_runs": len(seed_results),
               "completed_windows_total": len(seed_results) * DESIGN["evaluation_slots"] // DESIGN["window_size"],
               "elapsed_seconds": time.perf_counter() - started,
               "no_future_targets_provided_to_methods": True,
               "no_counterfactual_service_effects_claimed": True, "synthetic_distinct_events": 2,
               "distinct_events_not_24": True})
    print(f"Complete {len(seed_results)} replay runs. Negative and null results retained.", flush=True)


if __name__ == "__main__":
    main()
