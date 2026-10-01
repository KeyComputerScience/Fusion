"""Frozen TWO-SOURCE contextual-bandit fusion-trigger adaptation.

This post-hoc diagnostic reuses the original real DQN/PPO calibration and
workers. It is synthetic, has no resource telemetry, and is not a queue/trace
experiment. Monitoring outages mask BOTH loss-trigger and fusion feedback;
the learner still receives its factual action reward. Hidden onset/targets are
used only by the environment and offline scoring. Negative results are retained.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "core"))
from fusion import EvidenceFusion
from revised_fusion import RevisedFusion
from update_necessity import CONFIG as ORIGINAL_CONFIG
from update_necessity import calibrate, digest, state, write_csv, write_json
from update_budget_sensitivity import restore


CONFIG = copy.deepcopy(ORIGINAL_CONFIG)
CONFIG.update(
    protocol="post_hoc_frozen_two_source_bandit_fusion_trigger_v1",
    methods=["no_rt", "periodic", "observed_loss", "equal", "risk_rf",
             "service_only", "workload_only", "no_variability",
             "no_disagreement", "no_residual"],
    scenarios=["relation_shift", "no_shift", "relation_shift_service_outage",
               "relation_shift_all_source_loss", "no_shift_workload_conflict"],
    job_gradient_steps=32, deployment_delay_slots=32, max_jobs=12,
    hidden_onset_rng_offset=913748, hidden_onset_minimum=768,
    hidden_onset_maximum=1280, observed_accuracy_minimum_valid=32,
    monitor_feedback_delay_slots=0,
    monitor_timing="reward observed at slot end; decisions use only earlier slots",
    fault_first_slot=769, fault_last_slot=1280,
    workload_conflict="monitor category collapses to category 0; actual contexts unchanged",
    monitor_outage="learner retains factual reward; monitoring channel reward is missing",
    enabled_sources=[0, 2], operating_source_present=False,
    trigger_on=0.15, trigger_off=0.08, trigger_minimum_coverage=2/3,
    trigger_maximum_uncertainty=0.8,
    risk_rf={"entropy_temperature": 1.0, "residual_penalty": 1.0,
             "inertia": 0.1, "residual_decay": 0.9},
    initially_preregistered=False, post_hoc=True,
    tuning_policy="freeze this extension before test; retain every outcome; no threshold tuning",
)
CONFIG.pop("relation_shift_first_slot")
CONFIG["statistics"] = "separate analysis; paired fixed seed descriptive comparisons"


def hidden_onset(seed, cfg=CONFIG):
    """Independent of context/action draws; caller passes it only to env/scorer."""
    return int(np.random.default_rng(seed + cfg["hidden_onset_rng_offset"]).integers(
        cfg["hidden_onset_minimum"], cfg["hidden_onset_maximum"] + 1))


def environment_response(context, action, scenario, slot, onset, cfg):
    shifted = scenario.startswith("relation_shift") and slot >= onset
    target = (context + int(shifted)) % 3
    correct = int(action == target)
    reward = cfg["reward_correct"] if correct else cfg["reward_incorrect"]
    return float(reward), correct, target, int(shifted)


def monitor_row(slot, context, reward, served_version, worker_active, scenario, cfg):
    """No hidden onset, target, correctness, or policy state enters this function."""
    faulty = cfg["fault_first_slot"] <= slot <= cfg["fault_last_slot"]
    service_valid = not (faulty and scenario in (
        "relation_shift_service_outage", "relation_shift_all_source_loss"))
    workload_valid = not (faulty and scenario == "relation_shift_all_source_loss")
    regime = str(context) if workload_valid else None
    if faulty and scenario == "no_shift_workload_conflict":
        regime = "0"
    return {"slot": cfg["calibration_samples"] + slot,
            "regime": regime, "environment_reward": reward if service_valid else None,
            "observed_training_cost": 0.0 if service_valid else None,
            "logged_training": "worker_active" if worker_active else "none",
            "logged_inference": "bandit", "policy_version": str(served_version),
            "source0_visible": workload_valid, "source2_visible": service_valid,
            "monitor_available_from_slot": slot + 1}


class BanditFusion:
    """Original source estimators; two-source, declared decision adaptation."""
    def __init__(self, method, seed, cfg):
        fusion_cfg = json.loads((HERE / "core/config.json").read_text())["fusion"]
        fusion_cfg.update(enabled_sources=[0, 2], observed_cost_reward_scale=0.0,
                          seed=seed + 20260930)
        if method == "service_only":
            fusion_cfg["enabled_sources"] = [2]
        elif method == "workload_only":
            fusion_cfg["enabled_sources"] = [0]
        elif method == "no_variability":
            fusion_cfg["use_noise"] = False
        elif method == "no_disagreement":
            fusion_cfg["use_disagreement"] = False
        elif method == "equal":
            fusion_cfg["weighting_mode"] = "equal"
        # A deliberately missing dummy prevents vacuous all([]) from inventing
        # a valid operating source. It has no telemetry or numeric observations.
        reference = {"fit_through_slot": cfg["calibration_samples"],
                     "state_columns": ["absent_operating_telemetry"],
                     "state_scales": {"absent_operating_telemetry": {"mean": 0., "std": 1.}},
                     "provenance": ["two_source_contextual_bandit_no_resource_channel"]}
        self.base = EvidenceFusion(fusion_cfg, reference)
        parameters = copy.deepcopy(cfg["risk_rf"])
        if method == "no_residual":
            parameters["residual_penalty"] = 0.0
        self.extension = RevisedFusion(fusion_cfg, **parameters)
        self.method, self.cfg = method, cfg
        self.snapshot = self.base.snapshot
        self.buffer, self.windows, self.trigger_active = [], [], False

    def observe(self, row):
        self.buffer.append(row)
        if len(self.buffer) != self.base.config["window_size"]:
            return
        snap = self.base.complete_window(self.buffer, len(self.windows) + 1)
        contexts = {tuple(str(r[k]) for k in (
            "logged_training", "logged_inference", "policy_version")) for r in self.buffer}
        control = {"stable": len(contexts) == 1,
                   "key": next(iter(contexts)) if len(contexts) == 1 else None}
        if self.method == "equal":
            # Equal weighting shares the common-target disagreement and missing
            # coverage wrapper, isolating the weighting/learned-risk difference.
            valid = [s for s in (0, 2) if snap.masks[s]]
            forecast = (sum(snap.weights[s] * snap.next_target_predictions[s] for s in valid)
                        if valid else None)
            disagreement = (sum(snap.weights[s] * (snap.next_target_predictions[s] - forecast)**2
                                for s in valid) if valid else 0.)
            snap.forecast_score, snap.forecast_disagreement = forecast, disagreement
            snap.disagreement = disagreement
            if valid:
                snap.uncertainty = min(1., 4 * disagreement + .5 * (1 - snap.coverage))
            extras = {"forecast_score": forecast, "forecast_disagreement": disagreement,
                      "equal_wrapper": "common-target disagreement; raw gamma admission"}
        else:
            snap = self.extension.update(snap, control)
            extras = self.extension.extras(snap)
        if snap.masks[1] or snap.scores[1] is not None:
            raise AssertionError("Absent operating telemetry became valid")
        self.snapshot = snap
        self.windows.append({**snap.to_dict(), **extras,
                             "evaluation_completed_slot": row["slot"] - self.cfg["calibration_samples"],
                             "actual_available_from_evaluation_slot": row["monitor_available_from_slot"],
                             "control_context_stable": control["stable"],
                             "enabled_sources": self.base.config["enabled_sources"]})
        self.buffer = []

    def allows(self):
        snap, cfg = self.snapshot, self.cfg
        if snap.gamma >= cfg["trigger_on"]:
            self.trigger_active = True
        elif snap.gamma < cfg["trigger_off"]:
            self.trigger_active = False
        return (self.trigger_active and not snap.fallback
                and snap.coverage >= cfg["trigger_minimum_coverage"]
                and snap.uncertainty <= cfg["trigger_maximum_uncertainty"])


def observed_accuracy(monitor_history, cfg):
    """Last 64 EVENT slots, minimum32 visible; missing feedback is not removed from time."""
    recent = monitor_history[-cfg["observed_accuracy_window"]:]
    if len(recent) < cfg["observed_accuracy_window"]:
        return None
    rewards = [r["environment_reward"] for r in recent if r["source2_visible"]]
    if len(rewards) < cfg["observed_accuracy_minimum_valid"]:
        return None
    return statistics.fmean((reward + 1.) / 2. for reward in rewards)


def launch_rule(method, slot, monitor_history, reference, agent, job, last_launch, jobs, fusion, cfg):
    accuracy = observed_accuracy(monitor_history, cfg)
    # Hysteresis tracks every available decision snapshot, including cooldown
    # and active-worker periods; eligibility must not hide an off crossing.
    fusion_allowed = fusion.allows()
    if method == "no_rt" or job is not None or jobs >= cfg["max_jobs"]:
        return False, "disabled_worker_or_budget_ceiling", accuracy
    if (agent.new_transitions < cfg["minimum_new_feedback"]
            or (agent.kind == "ppo" and len(agent.rollout) < cfg["minimum_new_feedback"])
            or slot - last_launch < cfg["minimum_launch_interval"]):
        return False, "factual_feedback_or_cooldown", accuracy
    if method == "periodic":
        return slot % cfg["periodic_interval"] == 0, "periodic_clock", accuracy
    if method == "observed_loss":
        return (accuracy is not None and accuracy < reference - cfg["observed_accuracy_drop_threshold"]), "past_visible_service_accuracy", accuracy
    return fusion_allowed, "raw_change_fusion_hysteresis", accuracy


def run_evaluation(contexts, calibrated, reference, method, scenario, seed, cfg):
    agent, fusion = copy.deepcopy(calibrated), BanditFusion(method, seed, cfg)
    initial_hash, onset = agent.parameter_digest(), hidden_onset(seed, cfg)
    rows, events, history = [], [], []
    job, last_launch, total_updates = None, -cfg["minimum_launch_interval"], 0
    for slot, context in enumerate(contexts, 1):
        if history and history[-1]["monitor_available_from_slot"] > slot:
            raise AssertionError("Admission read feedback before arrival")
        launch, basis, accuracy = launch_rule(method, slot, history, reference, agent,
            job, last_launch, len(events), fusion, cfg)
        steps = 0
        if launch:
            parent_hash = agent.parameter_digest()
            worker, steps = agent.worker(cfg["job_gradient_steps"])
            if steps != cfg["job_gradient_steps"]:
                raise AssertionError("Worker did not execute its declared gradient budget")
            total_updates += steps
            event = {"launch_slot": slot, "deployment_slot": None,
                     "parent_version": agent.version, "status": "pending",
                     "actual_gradient_updates": steps, "declared_delay_slots": cfg["deployment_delay_slots"],
                     "parent_parameter_sha256": parent_hash, "worker_parameter_sha256": worker.parameter_digest(),
                     "parameters_changed": parent_hash != worker.parameter_digest(),
                     "trigger_basis": basis, "past_visible_accuracy": accuracy,
                     "gamma_at_launch": fusion.snapshot.gamma, "uncertainty_at_launch": fusion.snapshot.uncertainty,
                     "coverage_at_launch": fusion.snapshot.coverage}
            events.append(event)
            job = {"worker": worker, "remaining": cfg["deployment_delay_slots"], "event": event,
                   "parent_version": agent.version}
            last_launch = slot
        served_version, worker_active = agent.version, job is not None
        snapshot_for_decision = fusion.snapshot
        observation = state(int(context))
        action, logged, value = agent.act(observation, cfg["evaluation_dqn_epsilon"] if agent.kind == "dqn" else 0)
        reward, correct, target, shifted = environment_response(int(context), action, scenario, slot, onset, cfg)
        agent.record(observation, action, reward, observation, True, logged, value)
        visible = monitor_row(slot, int(context), reward, served_version, worker_active, scenario, cfg)
        history.append(visible)
        fusion.observe(visible)
        deployed = False
        if job is not None:
            job["remaining"] -= 1
            if job["remaining"] == 0:
                if job["parent_version"] != agent.version:
                    raise AssertionError("Deploying over a different parent policy")
                agent.deploy(job["worker"])
                job["event"].update(deployment_slot=slot, status="deployed", deployed_version=agent.version,
                                      deployment_parameter_sha256=agent.parameter_digest())
                if slot - job["event"]["launch_slot"] + 1 != cfg["deployment_delay_slots"]:
                    raise AssertionError("Worker deployed before declared service delay")
                job, deployed = None, True
        rows.append({"evaluation_slot": slot, "context": int(context), "action": action,
                     "actual_reward": reward, "correct_audit_only": correct,
                     "environment_target_audit_only": target, "environment_shift_audit_only": shifted,
                     "monitor_reward": visible["environment_reward"], "monitor_regime": visible["regime"],
                     "monitor_service_visible": visible["source2_visible"],
                     "monitor_workload_visible": visible["source0_visible"],
                     "served_model_version": served_version, "model_version_after_feedback": agent.version,
                     "actual_worker_active_during_service": worker_active,
                     "new_worker": launch, "actual_gradient_updates_launched": steps,
                     "deployed_after_feedback": deployed, "trigger_basis": basis,
                     "past_visible_accuracy_for_trigger": accuracy,
                     "fusion_gamma_for_trigger": snapshot_for_decision.gamma,
                     "fusion_coverage_for_trigger": snapshot_for_decision.coverage,
                     "fusion_uncertainty_for_trigger": snapshot_for_decision.uncertainty,
                     "fusion_event_completed_slot_for_trigger": snapshot_for_decision.completed_slot})
    before = [r["correct_audit_only"] for r in rows if r["evaluation_slot"] < onset]
    after = [r["correct_audit_only"] for r in rows if r["evaluation_slot"] >= onset]
    summary = {"method": method, "backbone": agent.kind, "scenario": scenario, "seed": seed,
               "evaluation_samples": len(rows), "accuracy": statistics.fmean(r["correct_audit_only"] for r in rows),
               "pre_boundary_accuracy": statistics.fmean(before) if before else None,
               "post_boundary_accuracy": statistics.fmean(after) if after else None,
               "last_512_accuracy": statistics.fmean(r["correct_audit_only"] for r in rows[-512:]),
               "reward_sum": sum(r["actual_reward"] for r in rows), "actual_gradient_updates": total_updates,
               "training_jobs_started": len(events), "deployments": sum(e["status"] == "deployed" for e in events),
               "job_gradient_steps": cfg["job_gradient_steps"], "deployment_delay_slots": cfg["deployment_delay_slots"],
               "max_jobs": cfg["max_jobs"], "budget_ceiling_updates": cfg["max_jobs"] * cfg["job_gradient_steps"],
               "normalized_training_cost": total_updates / cfg["cost_normalization_updates"],
               "reference_accuracy": reference, "initial_parameter_sha256": initial_hash,
               "final_parameter_sha256": agent.parameter_digest(), "audit_only_hidden_onset": onset,
               "state_has_phase_or_target": False, "trigger_has_phase_or_target": False,
               "two_source_adaptation": True, "operating_channel_present": False,
               "visible_service_samples": sum(r["monitor_service_visible"] for r in rows),
               "audit_preboundary_jobs": sum(e["launch_slot"] < onset for e in events),
               "audit_postboundary_jobs": sum(e["launch_slot"] >= onset for e in events),
               "audit_first_postboundary_launch_delay": next(
                   (e["launch_slot"] - onset for e in events if e["launch_slot"] >= onset), None),
               "audit_missed_postboundary_launch": (not any(e["launch_slot"] >= onset for e in events)
                   if scenario.startswith("relation_shift") else None),
               "negative_control_jobs": len(events) if scenario.startswith("no_shift") else None,
               "post_hoc": True, "initially_preregistered": False}
    validate_run(rows, events, fusion.windows, summary, cfg)
    return rows, events, fusion.windows, summary


def validate_run(rows, events, windows, summary, cfg):
    if sum(r["actual_gradient_updates_launched"] for r in rows) != summary["actual_gradient_updates"]:
        raise AssertionError("Gradient conservation failed")
    if len(events) > cfg["max_jobs"] or summary["actual_gradient_updates"] > summary["budget_ceiling_updates"]:
        raise AssertionError("Training ceiling exceeded")
    version = 0
    for r in rows:
        if r["served_model_version"] != version:
            raise AssertionError("Served version changed before completed deployment")
        version = r["model_version_after_feedback"]
        if r["fusion_event_completed_slot_for_trigger"] >= cfg["calibration_samples"] + r["evaluation_slot"]:
            raise AssertionError("Fusion used current/future event feedback")
    for event in events:
        if event["status"] == "deployed" and event["deployment_parameter_sha256"] != event["worker_parameter_sha256"]:
            raise AssertionError("Worker parameters not actually deployed")
    if any(w["masks"][1] or w["scores"][1] is not None for w in windows):
        raise AssertionError("Invented operating source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=HERE.parent / "results/update_necessity")
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/fusion_trigger_study")
    parser.add_argument("--max-jobs", type=int, choices=[4, 12], default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=CONFIG["seeds"])
    for name in ("methods", "scenarios", "backbones"):
        parser.add_argument("--" + name, nargs="+", choices=CONFIG[name], default=CONFIG[name])
    args = parser.parse_args()
    original, output = args.original.resolve(), args.output.resolve()
    if (output / "manifest.json").exists():
        raise SystemExit("Existing frozen results: choose a new output directory")
    cfg = copy.deepcopy(CONFIG)
    cfg.update(max_jobs=args.max_jobs, seeds=args.seeds, methods=args.methods,
               scenarios=args.scenarios, backbones=args.backbones)
    if (original / "frozen_config.json").exists():
        if json.loads((original / "frozen_config.json").read_text()) != ORIGINAL_CONFIG:
            raise ValueError("Original calibration protocol differs from its unchanged source")
    write_json(output / "frozen_config.json", cfg)
    manifest = {"status": "running", "post_hoc": True, "frozen_before_execution": True,
                "parameters_tuned_after_results": False, "configuration_sha256": digest(output / "frozen_config.json"),
                "source_hashes": {name: digest(HERE / name) for name in (
                    "fusion_trigger_study.py", "policies.py", "update_necessity.py", "update_budget_sensitivity.py",
                    "revised_fusion.py", "core/fusion.py", "core/config.json")},
                "expected_runs": len(cfg["seeds"]) * len(cfg["backbones"]) * len(cfg["scenarios"]) * len(cfg["methods"]),
                "completed_runs": 0, "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "device": "cpu", "two_source_adaptation": True}
    write_json(output / "manifest.json", manifest)
    started, summaries = time.perf_counter(), []
    for seed in cfg["seeds"]:
        path = original / "inputs" / f"contexts_seed_{seed}.csv"
        if path.exists():
            contexts = np.array([int(r["context"]) for r in csv.DictReader(path.open())])
        else:
            contexts = np.random.default_rng(seed + 20261001).integers(0, 3,
                size=cfg["calibration_samples"] + cfg["evaluation_samples"])
        if len(contexts) != cfg["calibration_samples"] + cfg["evaluation_samples"]:
            raise ValueError("Context stream does not have the frozen calibration/evaluation length")
        write_csv(output / "inputs" / f"contexts_seed_{seed}.csv",
                  [{"sample": n, "context": int(c)} for n, c in enumerate(contexts, 1)])
        context_hash = digest(output / "inputs" / f"contexts_seed_{seed}.csv")
        for backbone in cfg["backbones"]:
            directory = original / "calibration" / f"{backbone}_seed_{seed}"
            checkpoint = directory / "initial_policy.pt"
            if checkpoint.exists():
                agent = restore(checkpoint, backbone, seed, ORIGINAL_CONFIG)
                calibration = json.loads((directory / "summary.json").read_text())
                reference = calibration["reference_tail_accuracy"]
                if agent.parameter_digest() != calibration["initial_policy_sha256"]:
                    raise AssertionError("Calibration checkpoint hash changed")
                calibration["checkpoint_sha256"] = digest(checkpoint)
                local = output / "calibration" / f"{backbone}_seed_{seed}"
                local.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checkpoint, local / "initial_policy.pt")
            else:
                agent, calibration_rows, reference = calibrate(contexts[:cfg["calibration_samples"]], backbone, seed, ORIGINAL_CONFIG)
                local = output / "calibration" / f"{backbone}_seed_{seed}"
                write_csv(local / "slots.csv", calibration_rows)
                torch.save(agent.checkpoint(), local / "initial_policy.pt")
                calibration = {"reference_tail_accuracy": reference,
                               "initial_policy_sha256": agent.parameter_digest(),
                               "actual_calibration_gradient_updates": agent.updates,
                               "checkpoint_sha256": digest(local / "initial_policy.pt")}
            write_json(output / "calibration" / f"{backbone}_seed_{seed}" / "summary.json", calibration)
            for scenario, method in itertools.product(cfg["scenarios"], cfg["methods"]):
                rows, events, windows, summary = run_evaluation(
                    contexts[cfg["calibration_samples"]:], agent, reference, method, scenario, seed, cfg)
                summary.update(context_stream_sha256=context_hash,
                               input_stream_sha256=context_hash,
                               calibration_checkpoint_sha256=calibration["checkpoint_sha256"])
                run = output / "runs" / f"{backbone}_{scenario}_{method}_seed_{seed}"
                write_csv(run / "slots.csv", rows)
                write_json(run / "deployments.json", events)
                write_json(run / "windows.json", windows)
                write_json(run / "summary.json", summary)
                summaries.append(summary)
                write_json(output / "seed_results.json", summaries)
                write_csv(output / "seed_results.csv", summaries)
                manifest["completed_runs"] = len(summaries)
                write_json(output / "manifest.json", manifest)
                print(f"seed={seed} {backbone} {scenario} {method}: post={summary['post_boundary_accuracy']:.4f} tail={summary['last_512_accuracy']:.4f} jobs={len(events)} updates={summary['actual_gradient_updates']}", flush=True)
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                    seed_results_sha256=digest(output / "seed_results.json"), all_validation_passed=True)
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
