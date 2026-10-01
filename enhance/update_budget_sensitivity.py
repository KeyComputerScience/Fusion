"""Explicitly post-hoc update-budget diagnostic; preserve every dose and outcome.

The original eight-step results and calibrations are read without modification.
This adds 32/128 real worker steps with 32/128 service-slot deployment delays.
It is a sensitivity analysis proposed AFTER observing poor eight-step recovery,
not an initially preregistered favorable test or a replacement of queue results.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from policies import ServiceAgent
from update_necessity import CONFIG, digest, describe, paired_statistics, run_evaluation, write_csv, write_json


HERE = Path(__file__).resolve().parent
EVENT_FIELDS = ["launch_slot", "deployment_slot", "parent_version", "declared_delay_slots", "actual_gradient_updates",
                "parent_parameter_sha256", "worker_parameter_sha256", "parameters_changed", "trigger_basis",
                "past_observed_accuracy", "reference_accuracy", "status", "deployed_version", "deployment_parameter_sha256"]


def restore(path, backbone, seed, cfg):
    checkpoint = torch.load(path, map_location="cpu")
    agent = ServiceAgent(backbone, copy.deepcopy(cfg["agent"]), seed, dimension=3)
    agent.network.load_state_dict(checkpoint["network"])
    agent.target.load_state_dict(checkpoint["target"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    agent.replay, agent.rollout = copy.deepcopy(checkpoint["replay"]), copy.deepcopy(checkpoint["rollout"])
    agent.rng.setstate(checkpoint["rng_state"])
    agent.version, agent.updates = checkpoint["version"], checkpoint["updates"]
    agent.new_transitions = checkpoint["new_transitions"]
    return agent


def validate_run(rows, events, summary, cfg, contexts):
    if len(rows) != cfg["evaluation_samples"] or [r["context"] for r in rows] != list(contexts):
        raise AssertionError("Evaluation contexts differ from the frozen common stream")
    if sum(r["correct"] for r in rows) != summary["correct_samples"]:
        raise AssertionError("Accuracy conservation failed")
    if sum(r["actual_gradient_updates_launched"] for r in rows) != summary["actual_gradient_updates"]:
        raise AssertionError("Actual update conservation failed")
    if summary["normalized_training_cost"] != summary["actual_gradient_updates"] / 1000:
        raise AssertionError("Actual-gradient cost convention changed")
    previous_version, parameter_hash = 0, summary["initial_parameter_sha256"]
    for row in rows:
        if row["served_model_version"] != previous_version:
            raise AssertionError("Serving model changed before completed deployment")
        previous_version = row["model_version_after_feedback"]
    for event in events:
        if event["actual_gradient_updates"] != cfg["job_gradient_steps"]:
            raise AssertionError("Worker did not execute the declared update budget")
        if event["parent_parameter_sha256"] != parameter_hash:
            raise AssertionError("Worker parent digest differs from the deployed model")
        if event["status"] == "deployed":
            if event["deployment_slot"] - event["launch_slot"] + 1 != cfg["deployment_delay_slots"]:
                raise AssertionError("A larger training budget received a free deployment delay")
            if event["deployment_parameter_sha256"] != event["worker_parameter_sha256"]:
                raise AssertionError("Worker parameters were not actually deployed")
            parameter_hash = event["deployment_parameter_sha256"]
    if parameter_hash != summary["final_parameter_sha256"]:
        raise AssertionError("Final deployment digest does not match model state")
    if summary["scenario"] == "no_shift" and summary["method"] == "observed_loss" and events:
        raise AssertionError("Observed-loss negative control launched unexpected updates")


def analysis_for(all_rows, cfg):
    groups, pairs = [], []
    metrics = ["accuracy", "post_boundary_accuracy", "last_512_accuracy", "deployments", "training_jobs_started", "actual_gradient_updates", "normalized_training_cost"]
    for budget, backbone, scenario, method in itertools.product([8, 32, 128], cfg["backbones"], cfg["scenarios"], cfg["methods"]):
        selected = sorted((row for row in all_rows if row["budget_updates"] == budget
                           and (row["backbone"], row["scenario"], row["method"]) == (backbone, scenario, method)), key=lambda r: r["seed"])
        if len(selected) != 5:
            raise AssertionError("Every budget/scenario/backbone/method must retain all five seeds")
        groups.append({"budget_updates": budget, "deployment_delay_slots": budget,
                       "backbone": backbone, "scenario": scenario, "method": method,
                       "source": "shared_original_no_rt" if method == "no_rt" else ("original_8_step" if budget == 8 else "post_hoc_budget_extension"),
                       **{metric: describe([row[metric] for row in selected]) for metric in metrics}})
    for budget, backbone, scenario, method in itertools.product([8, 32, 128], cfg["backbones"], cfg["scenarios"], ["periodic", "observed_loss"]):
        selected = {(row["method"], row["seed"]): row for row in all_rows if row["budget_updates"] == budget
                    and (row["backbone"], row["scenario"]) == (backbone, scenario)}
        for metric in ["accuracy", "post_boundary_accuracy", "last_512_accuracy"]:
            differences = [selected[method, seed][metric] - selected["no_rt", seed][metric] for seed in cfg["seeds"]]
            pairs.append({"budget_updates": budget, "backbone": backbone, "scenario": scenario,
                          "method_minus_no_rt": method, "metric": metric, **paired_statistics(differences)})
    return {"groups": groups, "paired_comparisons": pairs,
            "status": "post_hoc diagnostic dose-response after original incomplete recovery; all doses retained",
            "scope": "synthetic contextual bandit only; no claim of trace/queue/fusion superiority"}


def write_report(path, analysis):
    lines = ["# Update budget and deployment-delay sensitivity", "",
             "This sensitivity analysis was explicitly proposed **after** observing poor recovery with eight updates per worker. It is not an initially preregistered result. All 8/32/128-update budgets are retained; none is selected or omitted according to its outcome. The original eight-update runs, their configuration and their hashes are unchanged.", "",
             "Only the worker's actual gradient-step budget and corresponding service-slot deployment delay change: 8/8, 32/32, and 128/128. The cost remains executed gradient updates divided by 1,000. Calibration policies, random contexts, hidden shift, reward, serving exploration, trigger and scheduling thresholds are reused. No-retraining is the same original baseline and is not rerun. The bandit has no queue resource model: linear deployment delay and update counts expose this diagnostic's training tradeoff but are not hardware latency or energy measurements.", "",
             "| Backbone | Scenario | Method | Updates/delay per job | Post-boundary accuracy % | Final 512 accuracy % | Deployments | Actual gradient updates | Normalized cost |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["groups"]:
        if row["method"] == "no_rt" and row["budget_updates"] != 8:
            continue  # Display the shared baseline once; retain it in JSON and CSV.
        lines.append(f"| {row['backbone'].upper()} | {row['scenario']} | {row['method']} | {row['budget_updates'] if row['method']!='no_rt' else 0} | {100*row['post_boundary_accuracy']['mean']:.2f} | {100*row['last_512_accuracy']['mean']:.2f} | {row['deployments']['mean']:.1f} | {row['actual_gradient_updates']['mean']:.1f} | {row['normalized_training_cost']['mean']:.3f} |")
    lines += ["", "Paired post-boundary differences against the shared original no-retraining baseline:", "",
              "| Backbone | Scenario | Method | Budget | Mean change, percentage points | 95% paired mean bootstrap interval | Exact two-sided signed-rank p |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in analysis["paired_comparisons"]:
        if row["metric"] != "post_boundary_accuracy":
            continue
        interval = row["paired_mean_bootstrap_ci95"]
        lines.append(f"| {row['backbone'].upper()} | {row['scenario']} | {row['method_minus_no_rt']} | {row['budget_updates']} | {100*row['difference']['mean']:+.2f} | [{100*interval[0]:+.2f}, {100*interval[1]:+.2f}] | {row['exact_signed_rank_p_two_sided']:.4f} |")
    lines += ["", "These descriptive comparisons are a post-hoc diagnostic across several budgets and outcomes. Five paired seeds cannot support an exact two-sided p<0.05 claim. Stronger recovery at one dose, if present, does not establish that updating improves the original queue service or that quality fusion beats a direct service-loss trigger. Raw files include all failures, negative controls, pending workers and parameter digests."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=HERE.parent / "results/update_necessity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    original = args.original.resolve()
    output = args.output.resolve() if args.output else original / "budget_sensitivity"
    if (output / "manifest.json").exists():
        raise SystemExit("Budget sensitivity directory already has a manifest; do not overwrite it")
    original_cfg = json.loads((original / "frozen_config.json").read_text())
    if original_cfg != CONFIG:
        raise ValueError("Original protocol differs from the source's frozen configuration")
    original_manifest = json.loads((original / "manifest.json").read_text())
    if digest(HERE / "policies.py") != original_manifest["source_hashes"]["policies.py"]:
        raise ValueError("ServiceAgent source changed since the original diagnostic")
    if digest(HERE / "update_necessity.py") != original_manifest["source_hashes"]["update_necessity.py"]:
        raise ValueError("Original diagnostic source changed")
    sensitivity_cfg = {"protocol": "post_hoc_training_budget_sensitivity_v1",
                       "post_hoc": True, "reason": "original eight-update adaptation yielded incomplete recovery",
                       "initially_preregistered": False, "all_budgets_retained": [8, 32, 128],
                       "new_budgets": [32, 128], "deployment_delay_equals_gradient_updates": True,
                       "methods": ["periodic", "observed_loss"], "no_rt_reused_not_rerun": True,
                       "original_protocol": original_cfg,
                       "original_config_sha256": digest(original / "frozen_config.json"),
                       "original_results_sha256": digest(original / "seed_results.json"),
                       "tuning_policy": "hold all other parameters fixed, retain all budgets and failures"}
    write_json(output / "declared_extension_config.json", sensitivity_cfg)
    manifest = {"status": "running", "post_hoc": True, "initially_preregistered": False,
                "source_hashes": {name: digest(HERE / name) for name in ["update_budget_sensitivity.py", "update_necessity.py", "policies.py"]},
                "configuration_sha256": digest(output / "declared_extension_config.json"),
                "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "expected_new_runs": 80, "completed_new_runs": 0, "shared_calibrations": [],
                "original_config_sha256": digest(original / "frozen_config.json"), "all_budgets_retained": [8, 32, 128]}
    write_json(output / "manifest.json", manifest)
    original_rows = json.loads((original / "seed_results.json").read_text())
    all_rows = []
    for budget in [8, 32, 128]:
        for row in original_rows:
            if budget == 8 or row["method"] == "no_rt":
                copied = copy.deepcopy(row)
                copied.update(budget_updates=budget, deployment_delay_slots=budget,
                              result_source="shared_original_no_rt" if row["method"] == "no_rt" else "original_8_step")
                all_rows.append(copied)
    started = time.perf_counter()
    new_rows, total_raw_rows = [], 0
    for seed in original_cfg["seeds"]:
        input_path = original / "inputs" / f"contexts_seed_{seed}.csv"
        contexts = np.array([int(row["context"]) for row in csv.DictReader(input_path.open())][original_cfg["calibration_samples"]:])
        for backbone in original_cfg["backbones"]:
            calibration = original / "calibration" / f"{backbone}_seed_{seed}"
            reference = json.loads((calibration / "summary.json").read_text())["reference_tail_accuracy"]
            initial = restore(calibration / "initial_policy.pt", backbone, seed, original_cfg)
            expected_hash = json.loads((calibration / "summary.json").read_text())["initial_policy_sha256"]
            if initial.parameter_digest() != expected_hash:
                raise AssertionError("Restored calibration policy differs from original")
            manifest["shared_calibrations"].append({"seed": seed, "backbone": backbone,
                                                    "checkpoint_sha256": digest(calibration / "initial_policy.pt"),
                                                    "deployed_parameter_sha256": expected_hash,
                                                    "context_stream_sha256": digest(input_path)})
            for budget, scenario, method in itertools.product([32, 128], original_cfg["scenarios"], ["periodic", "observed_loss"]):
                cfg = copy.deepcopy(original_cfg)
                cfg.update(job_gradient_steps=budget, deployment_delay_slots=budget)
                rows, events, summary = run_evaluation(contexts, initial, reference, method, scenario, cfg)
                summary.update(seed=seed, budget_updates=budget, deployment_delay_slots=budget,
                               post_hoc=True, initially_preregistered=False, result_source="post_hoc_budget_extension",
                               context_stream_sha256=digest(input_path))
                validate_run(rows, events, summary, cfg, contexts)
                directory = output / "runs" / f"budget_{budget}_{backbone}_{scenario}_{method}_seed_{seed}"
                write_csv(directory / "slots.csv", rows)
                write_csv(directory / "deployments.csv", events, EVENT_FIELDS)
                write_json(directory / "summary.json", summary)
                new_rows.append(summary)
                all_rows.append(summary)
                total_raw_rows += len(rows)
                manifest["completed_new_runs"] = len(new_rows)
                write_json(output / "manifest.json", manifest)
                print(f"budget={budget} seed={seed} {backbone} {scenario} {method}: post={summary['post_boundary_accuracy']:.4f} final={summary['last_512_accuracy']:.4f} deploys={summary['deployments']} updates={summary['actual_gradient_updates']}", flush=True)
    analysis = analysis_for(all_rows, original_cfg)
    write_csv(output / "new_seed_results.csv", new_rows)
    write_json(output / "new_seed_results.json", new_rows)
    write_json(output / "all_budget_seed_results.json", all_rows)
    write_json(output / "analysis.json", analysis)
    dose_rows = [{"budget_updates": row["budget_updates"], "backbone": row["backbone"], "scenario": row["scenario"],
                  "method": row["method"], "source": row["source"],
                  **{metric + "_mean": row[metric]["mean"] for metric in ["accuracy", "post_boundary_accuracy", "last_512_accuracy", "deployments", "actual_gradient_updates", "normalized_training_cost"]}}
                 for row in analysis["groups"]]
    write_csv(output / "dose_response.csv", dose_rows)
    write_report(output / "report.md", analysis)
    write_json(output / "validation.json", {"all_passed": True, "new_runs": len(new_rows), "new_raw_evaluation_rows": total_raw_rows,
               "checked": ["shared exact calibration parameter hashes", "common frozen context draws", "actual worker gradient budgets",
                           "linear budget-dependent deployment delay", "factual accuracy and cost conservation", "served policy version timeline",
                           "real parameter deployment digests", "all negative control and failure outcomes retained"]})
    artifacts = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name not in ["artifact_hashes.json", "manifest.json"]}
    write_json(output / "artifact_hashes.json", artifacts)
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                    artifact_hashes_sha256=digest(output / "artifact_hashes.json"), all_validation_passed=True)
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
