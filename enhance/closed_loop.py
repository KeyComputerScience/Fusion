"""Paired closed-loop runs: choose, execute, observe, deploy, then reuse feedback."""
from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
from calibrate_reference import fit_reference
from coordinator import Coordinator, Decision
from fusion import FusionSnapshot
from pipeline import FusionPipeline
from backend import EdgeService
from policies import ServiceAgent
from data_pipeline import read_json, write_json


def write_records(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".part")
    with gzip.open(staging, "wt", newline="", encoding="utf-8") as stream:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    with staging.open("rb") as stream:
        os.fsync(stream.fileno())
    staging.replace(path)


def method_config(core, method, seed):
    config = copy.deepcopy(core)
    fusion = config["fusion"]
    fusion["seed"] = seed + 20260930
    if method in ("equal", "fixed"):
        fusion["weighting_mode"] = method
    if method in ("workload", "operating", "performance", "reactive"):
        fusion["enabled_sources"] = [{"workload": 0, "operating": 1, "performance": 2, "reactive": 0}[method]]
    if method == "no_predictive":
        fusion["enable_predictive_calibration"] = False
    if method == "uncapped":
        fusion["maximum_source_weight"] = 1.0
    if method == "no_disagreement":
        fusion["use_disagreement"] = False
    if method == "no_delayed":
        config["coordination"]["enable_delayed_value"] = False
    return config


class ExperimentCoordinator(Coordinator):
    def __init__(self, config, profiles, experiment, method):
        super().__init__(config, profiles)
        self.experiment, self.method = experiment, method
        self.relative_slot, self.deficit = 0, 0.0

    def _allowed(self, snapshot):
        return False if self.method == "no_rt" else super()._allowed(snapshot)

    def choose(self, slot, training_load, inference_load, capacities, snapshot, remaining_slots,
               solver="auto", future_retentions=None, pending_gain_forecast=None):
        if self.method not in ("periodic", "dpp"):
            return super().choose(slot, training_load, inference_load, capacities, snapshot,
                                  remaining_slots, solver, future_retentions, pending_gain_forecast)
        rho = self.retention(snapshot)
        if self.method == "periodic":
            scheduled = self.relative_slot % self.experiment["periodic_interval"] == 1 and training_load > 0
            i_set = [self.training_by_id[self.experiment["periodic_profile"]]] if scheduled else [self.training_by_id[self.base]]
        else:
            scheduled, i_set = training_load > 0, self.training
        chosen, optimum, evaluated = None, None, 0
        for profile_i in i_set:
            for profile_j in self.inference:
                evaluated += 1
                if not self.feasible(profile_i, profile_j, training_load, inference_load, capacities):
                    continue
                current = math.exp(-self.config["xi"] * snapshot.gamma) * self.service_quality(profile_j) * (-math.expm1(-self.H))
                value = current
                if self.method == "dpp":
                    dpp = self.experiment["dpp"]
                    value = (dpp["quality_scale"] * current + self.deficit * training_load * profile_i["gain"]
                             - dpp["cost_scale"] * self.config["lambda_cost"] * training_load * profile_i["cost"])
                if optimum is None or value > optimum:
                    chosen, optimum = (profile_i, profile_j), value
        if chosen is None and self.method == "periodic" and scheduled:
            scheduled = False
            for profile_j in self.inference:
                evaluated += 1
                profile_i = self.training_by_id[self.base]
                if self.feasible(profile_i, profile_j, training_load, inference_load, capacities):
                    value = self.service_quality(profile_j)
                    if optimum is None or value > optimum:
                        chosen, optimum = (profile_i, profile_j), value
        return Decision(slot=slot, training_profile=chosen[0]["id"] if chosen else None,
                        inference_profile=chosen[1]["id"] if chosen else None, objective=optimum,
                        solver="periodic_feasible" if self.method == "periodic" else "dpp_enumeration",
                        status="feasible" if chosen else "infeasible_admission_required",
                        training_allowed=scheduled, rho=rho, recovery_state=self.H,
                        fusion_completed_slot=snapshot.completed_slot, gamma=snapshot.gamma,
                        uncertainty=snapshot.uncertainty, coverage=snapshot.coverage,
                        evaluated_pairs=evaluated, trajectory=[optimum] if chosen else [],
                        deployment_delay=chosen[0]["deployment_delay"] if chosen else 0)

    def observe(self, feedback, rho_used):
        output = super().observe(feedback, rho_used)
        target = self.experiment["dpp"]["completion_target"]
        self.deficit = max(0.0, self.deficit + target - feedback["completion_ratio"])
        return output


def finish_transition(agent, pending, state, done=False):
    if pending is None:
        return
    old_state, action, reward, logged, value, version = pending
    if agent.kind == "dqn" or version == agent.version:
        agent.record(old_state, action, reward, state, done, logged, value)


def calibrate(data, experiment, core, profiles, seed, backbone):
    agent = ServiceAgent(backbone, experiment["agent"], seed, experiment["state_dimension"])
    environment = EdgeService(data, experiment, profiles)
    pending, rows, metrics = None, [], {p["id"]: [] for p in profiles["inference"]}
    snapshot = FusionSnapshot(uncertainty=1.0, fallback=True)
    checker = Coordinator(core["coordination"], profiles)
    for slot in range(1, experiment["calibration_slots"] + 1):
        observation, _ = environment.begin(slot, agent, calibration=True)
        index = (slot - 1) % len(profiles["inference"])
        while index > 0 and not checker.feasible(
                profiles["training"][0], profiles["inference"][index], 0, observation["inference_load"], observation["capacities"]):
            index -= 1
        feasible = checker.feasible(profiles["training"][0], profiles["inference"][index],
                                    0, observation["inference_load"], observation["capacities"])
        profile_id = profiles["inference"][index]["id"] if feasible else None
        state = environment.state(observation, snapshot, 1.0, profile_id, 0)
        finish_transition(agent, pending, state)
        epsilon = experiment["agent"]["epsilon_start"] + (experiment["agent"]["epsilon_end"] - experiment["agent"]["epsilon_start"]) * slot / experiment["calibration_slots"]
        decision = Decision(slot, "tr0" if feasible else None, profile_id, 0, "calibration",
                            "feasible" if feasible else "infeasible_admission_required", False, 1.0, 1.0,
                            0, 0, 1, 0, 1, [0])
        feedback, diagnostics, _ = environment.execute(decision, agent, state, epsilon)
        rows.append(feedback)
        if feasible:
            metrics[profile_id].append({"completion": feedback["completion_ratio"],
                                       "deadline_compliance": math.exp(-feedback["deadline_loss"]),
                                       "availability": feedback["availability"], "throughput": feedback["throughput_ratio"]})
        pending = (state, diagnostics["application_action"], feedback["environment_reward"],
                   diagnostics["log_probability"], diagnostics["state_value"], agent.version)
        if backbone == "dqn" and len(agent.replay) >= experiment["agent"]["warmup"] and slot % 4 == 0:
            agent.train(1)
        elif backbone == "ppo" and len(agent.rollout) >= experiment["agent"]["batch_size"]:
            agent.train(4)
    finish_transition(agent, pending, state, done=True)
    reference = fit_reference(rows, core, experiment["calibration_slots"])
    calibrated = copy.deepcopy(profiles)
    for profile in calibrated["inference"]:
        if not metrics[profile["id"]]:
            raise ValueError("Every inference profile requires calibration support")
        profile["quality_prior"] = {name: float(np.mean([r[name] for r in metrics[profile["id"]]]))
                                    for name in profile["quality_prior"]}
    agent.new_transitions, agent.rollout = 0, []
    return agent, reference, calibrated, rows


def run_closed(data, experiment, core, profiles, calibrated_agent, reference, method, seed):
    started_at = time.perf_counter()
    config = method_config(core, method, seed)
    agent = copy.deepcopy(calibrated_agent)
    environment = EdgeService(data, experiment, profiles)
    pipeline = FusionPipeline(config, profiles, reference, collect_history=True)
    pipeline.coordinator = ExperimentCoordinator(config["coordination"], profiles, experiment, method)
    pending, records, timings = None, [], []
    origin = pipeline.origin
    for slot in range(origin, origin + experiment["evaluation_slots"]):
        relative = slot - origin + 1
        observation, pending_gains = environment.begin(slot, agent)
        pipeline.coordinator.relative_slot = relative
        stamp = time.perf_counter()
        decision = pipeline.decide(slot, observation["training_load"], observation["inference_load"],
                                   observation["capacities"], pending_gain_forecast=pending_gains)
        elapsed = (time.perf_counter() - stamp) * 1000
        timings.append(elapsed)
        snapshot = pipeline.fusion.snapshot
        state = environment.state(observation, snapshot, pipeline.coordinator.H, decision.inference_profile, relative)
        finish_transition(agent, pending, state)
        feedback, diagnostics, deployment = environment.execute(decision, agent, state,
                                                               experiment["agent"]["epsilon_end"] if agent.kind == "dqn" else 0)
        pending = (state, diagnostics["application_action"], feedback["environment_reward"],
                   diagnostics["log_probability"], diagnostics["state_value"], agent.version)
        outcomes = pipeline.observe(feedback)
        environment.apply_deployment(agent, deployment)
        record = {**feedback, **decision.to_dict(), **outcomes, **diagnostics,
                  "evaluation_slot": relative, "coordinator_ms": elapsed,
                  "source_scores": snapshot.scores, "source_weights": snapshot.weights,
                  "source_masks": snapshot.masks, "predictions": snapshot.next_target_predictions,
                  "policy_version_after": agent.version, "backbone": agent.kind, "method": method, "seed": seed}
        records.append(record)
    finish_transition(agent, pending, state, done=True)
    if environment.arrived != environment.completed + environment.expired + environment.rejected + int(environment.queue_counts().sum()):
        raise AssertionError("Task conservation failed")
    if any(r["logged_inference"] != r["inference_profile"] for r in records):
        raise AssertionError("A recommended inference profile was not executed")
    if any(e["delay"] < e["declared_delay"] for e in environment.events):
        raise AssertionError("Deployment occurred before declared completion")
    summary = {"method": method, "backbone": agent.kind, "seed": seed,
               "data_kind": str(data["provenance"]), "mode": "executed_closed_loop",
               "return": sum(r["environment_reward"] for r in records),
               "completion_fraction": environment.completed / max(environment.arrived, 1),
               "deadline_fraction": (environment.expired + environment.rejected) / max(environment.arrived, 1),
               "arrival_count": environment.arrived, "completed_count": environment.completed,
               "deadline_count": environment.expired + environment.rejected,
               "queue_end": int(environment.queue_counts().sum()),
               "p95_latency_slots": float(np.percentile(environment.latencies, 95)) if environment.latencies else None,
               "training_jobs": environment.started_jobs, "deployed_jobs": environment.deployed_jobs,
               "gradient_updates": environment.training_steps, "training_cost": environment.total_cost,
               "paused_training_slots": environment.paused_slots,
               "resource_violations": environment.resource_violations,
               "infeasible_slots": sum(r["status"] != "feasible" for r in records),
               "coordinator_mean_ms": float(np.mean(timings)), "coordinator_p95_ms": float(np.percentile(timings, 95)),
               "complete_windows": len(pipeline.window_results),
               "parameters_changed_events": sum(e["parameters_changed"] for e in environment.events),
               "runtime_seconds": time.perf_counter() - started_at,
               "initial_parameter_sha256": calibrated_agent.parameter_digest(), "final_parameter_sha256": agent.parameter_digest()}
    return records, pipeline.window_results, environment.events, summary
