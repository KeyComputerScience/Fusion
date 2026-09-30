"""Ablation/stress diagnostics on a frozen-prefix log, never policy-return claims."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics

from calibrate_reference import fit_reference
from coordinator import Coordinator
from fusion import finite
from logio import read_csv, read_json, write_csv, write_json
from pipeline import run_replay


def patched_config(config, patch):
    result = copy.deepcopy(config)
    for group, settings in patch.items():
        if not isinstance(settings, dict):
            result[group] = settings
        else:
            result[group].update(copy.deepcopy(settings))
    return result


def inject_stress(rows, case, config, reference, seed):
    result = copy.deepcopy(rows)
    rng = random.Random(seed)
    names = config["state_columns"]
    for row in result:
        relative = row["slot"] - reference["fit_through_slot"]
        if not case["start"] <= relative <= case["end"]:
            continue
        if case["kind"] == "outage":
            for source in case["sources"]:
                if source == 0:
                    row["regime"] = None
                elif source == 1:
                    for name in names:
                        row[name] = None
                elif source == 2:
                    row["environment_reward"] = row["observed_training_cost"] = None
        elif case["kind"] == "state_noise":
            for name in names:
                if finite(row.get(name)):
                    sigma = reference["state_scales"][name]["std"] * case["std_multiplier"]
                    value = max(0.0, row[name] + rng.gauss(0.0, sigma))
                    row[name] = min(1.0, value) if name in ("utilization","cpu_fraction") else value
        elif case["kind"] == "conflict":
            row["regime"] = "injected_unseen_regime"
            for name in names:
                row[name] = reference["state_scales"][name]["mean"]
            if finite(row.get("observed_training_cost")):
                row["environment_reward"] = (case["adjusted_reward"]
                    - config["fusion"]["observed_cost_reward_scale"] * row["observed_training_cost"])
        else:
            raise ValueError("Unknown stress kind")
        row["provenance"] = str(row.get("provenance","unspecified")) + "+measurement_stress"
    # This check concerns actual data read by prefix-only normalization.
    cutoff = reference["fit_through_slot"]
    if fit_reference(result, config, cutoff) != fit_reference(rows, config, cutoff):
        raise AssertionError("A stress case changed the calibration prefix")
    return result


def diagnose(rows, config, profiles, reference, decisions, windows):
    evaluator = Coordinator(config["coordination"], profiles)
    lookup = {row["slot"]:row for row in rows}
    violations = stale_or_future = weight_violations = 0
    for decision in decisions:
        if decision["fusion_completed_slot"] >= decision["slot"]:
            stale_or_future += 1
        if decision["status"] == "feasible":
            row = lookup[decision["slot"]]
            feasible = evaluator.feasible(evaluator.training_by_id[decision["training_profile"]],
                evaluator.inference_by_id[decision["inference_profile"]],
                row["training_load"], row["inference_load"], row["capacities"])
            violations += int(not feasible)
    for window in windows:
        weights = window["weights"]
        if window["fallback"]:
            invalid = any(weights) or window["uncertainty"] != 1 or window["coverage"] != 0
        else:
            invalid = (not math.isclose(sum(weights), 1, abs_tol=1e-12)
                or any(w < 0 or w > window["effective_weight_cap"]+1e-12 for w in weights)
                or any(w != 0 for w,m in zip(weights,window["masks"]) if not m)
                or not 0 <= window["gamma"] <= 1+1e-12
                or not 0 <= window["disagreement"] <= 0.25+1e-12)
        weight_violations += int(invalid)
    predictive_errors, score_alignment = [], []
    for previous, current in zip(windows, windows[1:]):
        target = current["scores"][2]
        predictions = previous["next_target_predictions"]
        if previous["fallback"] or not finite(target):
            continue
        if all(finite(p) for p,w in zip(predictions, previous["weights"]) if w > 0):
            predicted = sum(w*p for w,p in zip(previous["weights"],predictions) if w > 0)
            predictive_errors.append(predicted-target)
            score_alignment.append(previous["gamma"]-target)
    feasible = [d for d in decisions if d["status"] == "feasible"]
    base = profiles["base_training_profile"]
    return {
        "slots":len(decisions), "complete_windows":len(windows),
        "fallback_windows":sum(w["fallback"] for w in windows),
        "infeasible_slots":len(decisions)-len(feasible),
        "resource_violations":violations, "future_information_violations":stale_or_future,
        "weight_or_mask_violations":weight_violations,
        "training_recommendation_fraction":sum(d["training_profile"] != base for d in feasible)/len(feasible) if feasible else None,
        "mean_disagreement":statistics.fmean(w["disagreement"] for w in windows) if windows else None,
        "maximum_source_weight":max((max(w["weights"]) for w in windows),default=0),
        "next_degradation_pairs":len(predictive_errors),
        "weighted_next_prediction_mae":statistics.fmean(abs(e) for e in predictive_errors) if predictive_errors else None,
        "weighted_next_prediction_rmse":math.sqrt(statistics.fmean(e*e for e in predictive_errors)) if predictive_errors else None,
        "raw_gamma_next_degradation_mae":statistics.fmean(abs(e) for e in score_alignment) if score_alignment else None,
        "causal_policy_return_evaluated":False,
        "calibration_prefix_sha256":reference["calibration_prefix_sha256"]
    }


def run_suite(rows, config, profiles, reference, ablations, stresses, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    tasks = [("ablation", variant["name"], rows, patched_config(config, variant["patch"]))
             for variant in ablations["variants"]]
    for case in stresses["cases"]:
        perturbed = inject_stress(rows, case, config, reference, stresses["seed"])
        tasks.append(("stress",case["name"],perturbed,copy.deepcopy(config)))
    for category, name, input_rows, selected_config in tasks:
        choices, windows, replay_summary = run_replay(input_rows, selected_config, profiles, reference)
        audit = diagnose(input_rows, selected_config, profiles, reference, choices, windows)
        record = {"category":category, "name":name, **audit}
        case_dir = directory / category / name
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(case_dir / "config.json", selected_config)
        write_json(case_dir / "diagnostics.json", {**record, "provenance":replay_summary["provenance"]})
        # Raw windows retain physical masks and pre-observation predictions.
        (case_dir / "fusion_windows.jsonl").write_text(
            "".join(json.dumps(w,ensure_ascii=False,allow_nan=False)+"\n" for w in windows),encoding="utf-8")
        summaries.append(record)
        print(f"{category}/{name}: violations={audit['resource_violations']}, windows={len(windows)}",flush=True)
    write_csv(directory / "diagnostics.csv", summaries)
    write_json(directory / "diagnostics.json", {
        "scope":"offline measurement and recommendation diagnostics",
        "claims_empirical_DRL_improvement":False,
        "warning":"Prediction alignment is to an observed performance score, not a ground-truth drift label. No confidence intervals or improvement significance are inferred from one synthetic log.",
        "results":summaries})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="example_stream.csv")
    parser.add_argument("--config",default="config.json")
    parser.add_argument("--profiles",default="profiles.json")
    parser.add_argument("--reference",default="reference.json")
    parser.add_argument("--ablations",default="ablation_plan.json")
    parser.add_argument("--stresses",default="stress_plan.json")
    parser.add_argument("--output",default="experiment_results")
    args = parser.parse_args()
    run_suite(read_csv(args.input),read_json(args.config),read_json(args.profiles),read_json(args.reference),
              read_json(args.ablations),read_json(args.stresses),args.output)


if __name__ == "__main__":
    main()
