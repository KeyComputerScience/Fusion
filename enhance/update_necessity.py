"""A separately labeled contextual-bandit concept-drift update diagnostic.

This is NOT Alibaba or the queue service. Contexts retain the same distribution;
only the hidden context-to-reward relation changes at a preregistered time. The
existing ServiceAgent DQN/PPO, worker and deploy methods perform actual updates.
No target label, phase, drift time or counterfactual reward enters the agent or
the observed-loss admission rule. All outcomes, including failures, are retained.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from policies import ServiceAgent


HERE = Path(__file__).resolve().parent
CONFIG = {
    "protocol": "frozen_contextual_bandit_update_diagnostic_v1",
    "provenance": "controlled_synthetic_contextual_bandit_not_trace_or_queue_service",
    "seeds": [10, 20, 30, 40, 50],
    "backbones": ["dqn", "ppo"],
    "methods": ["no_rt", "periodic", "observed_loss"],
    "scenarios": ["relation_shift", "no_shift"],
    "contexts": 3,
    "actions": 3,
    "context_distribution": "iid discrete Uniform{0,1,2}; same exogenous draws for all methods",
    "state": "one_hot_context_only; no slot, phase, target or drift time",
    "calibration_samples": 4096,
    "evaluation_samples": 3072,
    "relation_shift_first_slot": 1025,
    "target_pre_shift": "context",
    "target_post_shift": "(context+1) mod 3",
    "no_shift_target": "context throughout",
    "reward_correct": 1.0,
    "reward_incorrect": -1.0,
    "episode_length": 1,
    "calibration_train_interval": 128,
    "calibration_gradient_steps": 8,
    "calibration_dqn_epsilon_start": 0.5,
    "calibration_dqn_epsilon_end": 0.1,
    "evaluation_dqn_epsilon": 0.1,
    "ppo_sampling": "existing stochastic ServiceAgent.act with no added epsilon mixture",
    "periodic_interval": 256,
    "observed_accuracy_window": 64,
    "observed_accuracy_drop_threshold": 0.20,
    "reference_accuracy_tail_samples": 512,
    "minimum_new_feedback": 128,
    "minimum_launch_interval": 128,
    "job_gradient_steps": 8,
    "deployment_delay_slots": 8,
    "training_cost_definition": "actually executed worker gradient updates / 1000; dimensionless bookkeeping, not energy/hardware latency",
    "cost_normalization_updates": 1000,
    "agent": {
        "hidden": [64, 64], "discount": 0.0, "learning_rate": 0.0008,
        "batch_size": 128, "replay_capacity": 512, "target_update": 64,
        "ppo_clip": 0.2, "ppo_gae": 0.95, "ppo_entropy": 0.05,
        "gradient_clip": 1.0, "threads": 1,
    },
    "statistics": "paired seed differences; enumerate all 5^5 paired mean bootstrap resamples; exact sign-flip signed-rank two-sided p",
    "tuning_policy": "Parameters frozen before running; retain every method/backbone/scenario, including no benefit or calibration failure",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def state(context):
    return [float(context == index) for index in range(3)]


def environment_response(context, action, scenario, evaluation_slot, cfg):
    shifted = scenario == "relation_shift" and evaluation_slot >= cfg["relation_shift_first_slot"]
    target = (context + int(shifted)) % 3
    correct = int(action == target)
    reward = cfg["reward_correct"] if correct else cfg["reward_incorrect"]
    return reward, correct, target, int(shifted)


def calibrate(contexts, backbone, seed, cfg):
    agent = ServiceAgent(backbone, copy.deepcopy(cfg["agent"]), seed, dimension=3)
    records = []
    for index, context in enumerate(contexts, 1):
        fraction = index / len(contexts)
        epsilon = (cfg["calibration_dqn_epsilon_start"]
                   + fraction * (cfg["calibration_dqn_epsilon_end"] - cfg["calibration_dqn_epsilon_start"]))
        observation = state(int(context))
        action, logged, value = agent.act(observation, epsilon if backbone == "dqn" else 0)
        correct = int(action == int(context))
        reward = cfg["reward_correct"] if correct else cfg["reward_incorrect"]
        agent.record(observation, action, reward, observation, True, logged, value)
        steps = 0
        if index % cfg["calibration_train_interval"] == 0:
            steps = agent.train(cfg["calibration_gradient_steps"])
        records.append({"sample": index, "context": int(context), "action": action,
                        "observed_reward": reward, "correct": correct,
                        "gradient_updates_after_feedback": steps, "dqn_epsilon": epsilon if backbone == "dqn" else 0})
    reference = statistics.fmean(r["correct"] for r in records[-cfg["reference_accuracy_tail_samples"]:])
    # Clear in-flight PPO samples; retain the bounded factual DQN replay buffer.
    agent.rollout, agent.new_transitions = [], 0
    return agent, records, reference


def should_launch(method, slot, correct_history, reference_accuracy, agent, job, last_launch, cfg):
    if method == "no_rt" or job is not None:
        return False, "disabled_or_worker_active", None
    ready = (agent.new_transitions >= cfg["minimum_new_feedback"]
             and (agent.kind != "ppo" or len(agent.rollout) >= cfg["minimum_new_feedback"])
             and slot - last_launch >= cfg["minimum_launch_interval"])
    if not ready:
        return False, "insufficient_factual_feedback_or_cooldown", None
    window = cfg["observed_accuracy_window"]
    accuracy = statistics.fmean(correct_history[-window:]) if len(correct_history) >= window else None
    if method == "periodic":
        return slot % cfg["periodic_interval"] == 0, "periodic_clock", accuracy
    if method == "observed_loss":
        return (accuracy is not None and accuracy < reference_accuracy - cfg["observed_accuracy_drop_threshold"]), "past_observed_accuracy", accuracy
    raise ValueError(f"Unknown method {method}")


def run_evaluation(contexts, calibrated, reference_accuracy, method, scenario, cfg):
    agent = copy.deepcopy(calibrated)
    initial_hash = agent.parameter_digest()
    records, events, correct_history = [], [], []
    job, last_launch = None, -cfg["minimum_launch_interval"]
    gradient_updates = 0
    for slot, context in enumerate(contexts, 1):
        launch, trigger, observed_accuracy = should_launch(
            method, slot, correct_history, reference_accuracy, agent, job, last_launch, cfg)
        started_steps = 0
        if launch:
            parent_hash = agent.parameter_digest()
            worker, started_steps = agent.worker(cfg["job_gradient_steps"])
            if started_steps <= 0:
                raise AssertionError("Admitted worker did not perform an eligible actual update")
            gradient_updates += started_steps
            event = {"launch_slot": slot, "deployment_slot": None, "parent_version": agent.version,
                     "declared_delay_slots": cfg["deployment_delay_slots"],
                     "actual_gradient_updates": started_steps,
                     "parent_parameter_sha256": parent_hash, "worker_parameter_sha256": worker.parameter_digest(),
                     "parameters_changed": parent_hash != worker.parameter_digest(),
                     "trigger_basis": trigger, "past_observed_accuracy": observed_accuracy,
                     "reference_accuracy": reference_accuracy, "status": "pending"}
            events.append(event)
            job = {"worker": worker, "remaining": cfg["deployment_delay_slots"], "event": event,
                   "parent_version": agent.version}
            last_launch = slot
        deployed_version = agent.version
        observation = state(int(context))
        action, logged, value = agent.act(observation, cfg["evaluation_dqn_epsilon"] if agent.kind == "dqn" else 0)
        # The hidden relation and target are accessed only by this environment.
        reward, correct, target, shifted = environment_response(int(context), action, scenario, slot, cfg)
        agent.record(observation, action, reward, observation, True, logged, value)
        correct_history.append(correct)
        deployed = False
        if job is not None:
            job["remaining"] -= 1
            if job["remaining"] == 0:
                if job["parent_version"] != agent.version:
                    raise AssertionError("Worker attempted to deploy over a different policy version")
                agent.deploy(job["worker"])
                job["event"].update(deployment_slot=slot, status="deployed", deployed_version=agent.version,
                                      deployment_parameter_sha256=agent.parameter_digest())
                if slot - job["event"]["launch_slot"] + 1 != cfg["deployment_delay_slots"]:
                    raise AssertionError("A worker deployed before its declared service-slot delay")
                job, deployed = None, True
        records.append({"evaluation_slot": slot, "context": int(context), "action": action,
                        "observed_reward": reward, "correct": correct,
                        "environment_target_audit_only": target, "environment_shift_audit_only": shifted,
                        "served_model_version": deployed_version, "new_worker": int(launch),
                        "actual_gradient_updates_launched": started_steps, "deployed_after_feedback": int(deployed),
                        "model_version_after_feedback": agent.version,
                        "past_observed_accuracy_for_trigger": observed_accuracy,
                        "reference_accuracy": reference_accuracy,
                        "trigger_basis": trigger, "log_probability": logged,
                        "state_value": value, "worker_active_after_feedback": int(job is not None)})
    onset = cfg["relation_shift_first_slot"]
    before = [r["correct"] for r in records if r["evaluation_slot"] < onset]
    after = [r["correct"] for r in records if r["evaluation_slot"] >= onset]
    tail = [r["correct"] for r in records[-512:]]
    summary = {"backbone": agent.kind, "method": method, "scenario": scenario,
               "evaluation_samples": len(records), "correct_samples": sum(correct_history),
               "accuracy": statistics.fmean(correct_history), "pre_boundary_accuracy": statistics.fmean(before),
               "post_boundary_accuracy": statistics.fmean(after), "last_512_accuracy": statistics.fmean(tail),
               "reward_sum": sum(r["observed_reward"] for r in records),
               "training_jobs_started": len(events), "deployments": sum(e["status"] == "deployed" for e in events),
               "actual_gradient_updates": gradient_updates,
               "normalized_training_cost": gradient_updates / cfg["cost_normalization_updates"],
               "calibration_gradient_updates_excluded": calibrated.updates,
               "reference_accuracy": reference_accuracy,
               "initial_parameter_sha256": initial_hash, "final_parameter_sha256": agent.parameter_digest(),
               "state_has_phase_or_target": False, "trigger_has_phase_or_target": False,
               "common_exogenous_contexts": True}
    return records, events, summary


def describe(values):
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0,
            "minimum": min(values), "maximum": max(values)}


def paired_statistics(differences):
    differences = list(differences)
    bootstrap = [statistics.fmean(sample) for sample in itertools.product(differences, repeat=len(differences))]
    interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    nonzero = [value for value in differences if abs(value) > 1e-12]
    if not nonzero:
        pvalue = 1.0
    else:
        ranks = []
        absolute = [abs(value) for value in nonzero]
        for value in absolute:
            less = sum(other < value for other in absolute)
            ties = sum(abs(other - value) < 1e-12 for other in absolute)
            ranks.append(less + (ties + 1) / 2)
        observed = abs(sum(math.copysign(rank, difference) for rank, difference in zip(ranks, nonzero)))
        outcomes = [abs(sum(sign * rank for sign, rank in zip(signs, ranks)))
                    for signs in itertools.product((-1, 1), repeat=len(nonzero))]
        pvalue = sum(value >= observed - 1e-12 for value in outcomes) / len(outcomes)
    return {"difference": describe(differences), "seed_differences": differences,
            "paired_mean_bootstrap_ci95": interval, "exact_signed_rank_p_two_sided": pvalue,
            "seed_pairs": len(differences), "bootstrap_mean_resamples": len(bootstrap)}


def analyze(summaries, cfg):
    groups, paired = [], []
    metrics = ["accuracy", "post_boundary_accuracy", "last_512_accuracy", "reward_sum", "deployments", "actual_gradient_updates", "normalized_training_cost"]
    for backbone, scenario, method in itertools.product(cfg["backbones"], cfg["scenarios"], cfg["methods"]):
        selected = sorted((r for r in summaries if (r["backbone"], r["scenario"], r["method"]) == (backbone, scenario, method)), key=lambda r: r["seed"])
        groups.append({"backbone": backbone, "scenario": scenario, "method": method,
                       **{metric: describe([r[metric] for r in selected]) for metric in metrics}})
    for backbone, scenario, method in itertools.product(cfg["backbones"], cfg["scenarios"], ["periodic", "observed_loss"]):
        selected = {(r["method"], r["seed"]): r for r in summaries if (r["backbone"], r["scenario"]) == (backbone, scenario)}
        for metric in ["accuracy", "post_boundary_accuracy", "last_512_accuracy", "reward_sum"]:
            differences = [selected[method, seed][metric] - selected["no_rt", seed][metric] for seed in cfg["seeds"]]
            paired.append({"backbone": backbone, "scenario": scenario, "method_minus_no_rt": method,
                           "metric": metric, **paired_statistics(differences)})
    return {"groups": groups, "paired_comparisons": paired,
            "interpretation": "A separate controlled concept-drift diagnostic. It cannot replace negative queue results, establish trace validity, or isolate an information-fusion advantage."}


def report(path, analysis, cfg):
    lines = ["# Separate update-necessity diagnostic", "",
             "This experiment tests a narrow, explicitly synthetic failure mode: an unobserved context-to-action relation changes while the observed context distribution stays fixed. It is a contextual bandit, not Alibaba, a queue service, or a reproduction of Table 6. The original queue results must remain a separate panel.", "",
             "The DQN and PPO policies, private training workers, gradient steps, and delayed deployments are the existing executable ServiceAgent implementation. Agents observe one-hot context and factual action rewards only. Scheduling uses either a clock or past observed service accuracy; neither receives phase labels, target actions, or the known drift time.", "",
             "The fixed five seeds, 4,096-sample calibration, 3,072-sample evaluation, and drift beginning at slot 1,025 were declared before execution. A no-shift negative control uses the same context draws. Costs below are actual worker gradient steps divided by 1,000; they are dimensionless bookkeeping rather than hardware or energy measurements.", "",
             "| Backbone | Scenario | Method | Accuracy % (mean ± SD) | Post-boundary accuracy % | Final 512 accuracy % | Deployments | Actual gradient updates | Normalized cost |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["groups"]:
        acc, post, tail = row["accuracy"], row["post_boundary_accuracy"], row["last_512_accuracy"]
        lines.append(f"| {row['backbone'].upper()} | {row['scenario']} | {row['method']} | {100*acc['mean']:.2f} ± {100*acc['sd']:.2f} | {100*post['mean']:.2f} | {100*tail['mean']:.2f} | {row['deployments']['mean']:.1f} | {row['actual_gradient_updates']['mean']:.1f} | {row['normalized_training_cost']['mean']:.3f} |")
    lines += ["", "Paired post-boundary differences against no retraining:", "",
              "| Backbone | Scenario | Method | Accuracy change (percentage points) | 95% paired mean bootstrap interval | Exact two-sided signed-rank p |",
              "|---|---|---|---:|---:|---:|"]
    for row in analysis["paired_comparisons"]:
        if row["metric"] == "post_boundary_accuracy":
            interval = row["paired_mean_bootstrap_ci95"]
            lines.append(f"| {row['backbone'].upper()} | {row['scenario']} | {row['method_minus_no_rt']} | {100*row['difference']['mean']:+.2f} | [{100*interval[0]:+.2f}, {100*interval[1]:+.2f}] | {row['exact_signed_rank_p_two_sided']:.4f} |")
    lines += ["", "All individual seed runs and raw slot/deployment/calibration CSV files are retained. With only five seed pairs, an exact two-sided signed-rank test cannot attain p below 0.0625 when all differences are nonzero. Bootstrap intervals describe this fixed protocol, not all possible drift environments.", "",
              "Successful recovery here would show that updating can be necessary for this particular hidden relation shift. It would not show that retraining is beneficial in the original queue workload, that fusion beats a direct service-loss trigger, or that the proposed reliability model is novel. An observed failure to recover is retained, including any PPO exploration or finite-update limitation.", "",
              "Reproduce from the revision code directory with the same environment:", "", "```sh", "python update_necessity.py --output ../results/update_necessity", "```", "",
              "The manifest records code/config/context hashes, numerical library versions, deployment timing checks, and exact run counts."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/update_necessity")
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        raise SystemExit("This frozen experiment directory already has a manifest; choose a new directory rather than overwrite results.")
    cfg = copy.deepcopy(CONFIG)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen_config.json", cfg)
    hashes = {"update_necessity.py": digest(__file__), "policies.py": digest(HERE / "policies.py"),
              "frozen_config.json": digest(output / "frozen_config.json")}
    manifest = {"status": "running", "configuration_sha256": hashes["frozen_config.json"], "source_hashes": hashes,
                "frozen_before_execution": True, "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "device": "cpu", "threads": 1, "inputs": [],
                "expected_runs": 60, "completed_runs": 0,
                "not_alibaba_or_queue": True, "parameters_tuned_after_results": False}
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    summaries = []
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed + 20261001)
        contexts = rng.integers(0, 3, size=cfg["calibration_samples"] + cfg["evaluation_samples"])
        inputs_path = output / "inputs" / f"contexts_seed_{seed}.csv"
        write_csv(inputs_path, [{"sample": index + 1, "context": int(context),
                                 "part": "calibration" if index < cfg["calibration_samples"] else "evaluation"}
                                for index, context in enumerate(contexts)])
        manifest["inputs"].append({"seed": seed, "relative_path": str(inputs_path.relative_to(output)), "sha256": digest(inputs_path)})
        for backbone in cfg["backbones"]:
            initial, calibration_rows, reference = calibrate(contexts[:cfg["calibration_samples"]], backbone, seed, cfg)
            calibration_dir = output / "calibration" / f"{backbone}_seed_{seed}"
            write_csv(calibration_dir / "slots.csv", calibration_rows)
            torch.save(initial.checkpoint(), calibration_dir / "initial_policy.pt")
            write_json(calibration_dir / "summary.json", {"reference_tail_accuracy": reference,
                       "actual_calibration_gradient_updates": initial.updates,
                       "initial_policy_sha256": initial.parameter_digest(),
                       "calibration_accuracy": statistics.fmean(r["correct"] for r in calibration_rows),
                       "calibration_sufficient_claim": "tail observed accuracy reported; no convergence claim"})
            for scenario, method in itertools.product(cfg["scenarios"], cfg["methods"]):
                records, events, summary = run_evaluation(contexts[cfg["calibration_samples"]:], initial, reference, method, scenario, cfg)
                summary["seed"] = seed
                summary["context_stream_sha256"] = digest(inputs_path)
                directory = output / "runs" / f"{backbone}_{scenario}_{method}_seed_{seed}"
                write_csv(directory / "slots.csv", records)
                write_csv(directory / "deployments.csv", events,
                          fields=["launch_slot", "deployment_slot", "parent_version", "declared_delay_slots", "actual_gradient_updates",
                                  "parent_parameter_sha256", "worker_parameter_sha256", "parameters_changed", "trigger_basis",
                                  "past_observed_accuracy", "reference_accuracy", "status", "deployed_version", "deployment_parameter_sha256"])
                write_json(directory / "summary.json", summary)
                summaries.append(summary)
                manifest["completed_runs"] = len(summaries)
                write_json(output / "manifest.json", manifest)
                print(f"seed={seed} {backbone} {scenario} {method}: accuracy={summary['accuracy']:.4f} post={summary['post_boundary_accuracy']:.4f} deployments={summary['deployments']} updates={summary['actual_gradient_updates']}", flush=True)
    write_csv(output / "seed_results.csv", summaries)
    write_json(output / "seed_results.json", summaries)
    analysis = analyze(summaries, cfg)
    write_json(output / "analysis.json", analysis)
    report(output / "report.md", analysis, cfg)
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                    output_hashes={name: digest(output / name) for name in ["seed_results.csv", "seed_results.json", "analysis.json", "report.md"]})
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
