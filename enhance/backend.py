"""Queue service, actual policy updates, occupied resources and delayed deployments."""
from __future__ import annotations

from collections import deque
import math
import time

import numpy as np


class EdgeService:
    def __init__(self, data, configuration, profiles):
        self.data, self.config, self.profiles = data, configuration, profiles
        self.nodes = data["arrivals"].shape[1]
        self.queues = [[deque() for _ in range(3)] for _ in range(self.nodes)]
        self.training_by_id = {p["id"]: p for p in profiles["training"]}
        self.inference_by_id = {p["id"]: p for p in profiles["inference"]}
        self.job = None
        self.utilization = 0.0
        self.events, self.latencies = [], []
        self.arrived = self.completed = self.expired = self.rejected = 0
        self.started_jobs = self.deployed_jobs = self.training_steps = self.paused_slots = 0
        self.resource_violations = 0
        self.total_cost = 0.0

    def queue_counts(self):
        return np.asarray([[len(q) for q in node] for node in self.queues])

    def begin(self, slot, agent, calibration=False):
        self.slot = slot
        arrivals = self.data["arrivals"][slot - 1]
        base = self.data["capacities"][slot - 1].copy()
        self.slot_rejected = 0
        for node in range(self.nodes):
            limit = min(20, max(0, int(base[node, 0] * 0.5 / 0.002)))
            for kind in range(3):
                for _ in range(int(arrivals[node, kind])):
                    self.arrived += 1
                    if sum(len(q) for q in self.queues[node]) >= limit:
                        self.rejected += 1
                        self.slot_rejected += 1
                    else:
                        self.queues[node][kind].append([
                            slot, slot + self.config["task_deadlines"][kind], self.config["task_work"][kind]])
        counts = self.queue_counts()
        queue_memory = np.zeros_like(base)
        queue_memory[:, 0] = counts.sum(axis=1) * 0.002
        self.base, self.queue_memory = base, queue_memory
        self.running = self.job is not None and np.all(self.job["resources"] + queue_memory <= base + 1e-12)
        occupied = self.job["resources"] if self.running else np.zeros_like(base)
        capacities = np.maximum(0.0, base - queue_memory - occupied)
        mean_arrivals = float(arrivals.sum() / self.nodes)
        all_count = max(int(arrivals.sum()), 1)
        composition = arrivals.sum(axis=0) / all_count
        regime = "cpu_heavy" if composition[2] >= 0.4 else ("light" if composition[2] < 0.2 else "mixed")
        ready = self.job is None and agent.new_transitions >= self.config["minimum_new_transitions"]
        if agent.kind == "ppo":
            ready = ready and len(agent.rollout) >= self.config["minimum_new_transitions"]
        observation = {
            "slot": slot, "training_load": self.config["training_load"] if ready and not calibration else 0.0,
            "inference_load": float(np.clip(0.65 + 0.12 * mean_arrivals, 0.5, 1.1)),
            "capacities": capacities.tolist(), "queue": float(counts.sum() / (self.nodes * self.config["queue_scale"])),
            "arrival_rate": mean_arrivals, "utilization": self.utilization,
            "cpu_fraction": float(composition[2]), "bandwidth": float(base[:, 2].mean()),
            "regime": regime, "provenance": str(self.data["provenance"])}
        self.observation = observation
        pending = {}
        if self.job is not None:
            lag = self.job["remaining"] + int(not self.running)
            pending[lag] = self.job["gain"]
        return observation, pending

    def state(self, observation, snapshot, recovery, inference_id, relative):
        counts = self.queue_counts().sum(axis=0)
        proportions = counts / max(int(counts.sum()), 1)
        capacities = np.asarray(observation["capacities"])
        level = int(inference_id[3:]) / 5 if inference_id is not None else 0.0
        return [min(1.0, observation["queue"]), min(1.0, observation["arrival_rate"] / 4),
                *proportions.tolist(), snapshot.gamma, snapshot.uncertainty, snapshot.coverage,
                recovery / (1 + recovery), *np.clip(capacities.min(axis=0), 0, 1).tolist(),
                max(0.0, min(1.0, relative / self.config["evaluation_slots"])), level]

    def execute(self, decision, agent, state, epsilon=0.0):
        slot, observation = self.slot, self.observation
        served_version = agent.version
        started, update_wall_ms = False, 0.0
        if (decision.status == "feasible" and decision.training_profile != "tr0"
                and observation["training_load"] > 0):
            if self.job is not None:
                raise AssertionError("A training job is already active")
            profile = self.training_by_id[decision.training_profile]
            started_at = time.perf_counter()
            worker, steps = agent.worker(profile["updates"])
            update_wall_ms = (time.perf_counter() - started_at) * 1000
            if not steps:
                raise AssertionError("A selected training job has no eligible updates")
            load = observation["training_load"]
            resources = np.tile(profile["resources"], (self.nodes, 1)) * load
            self.job = {"profile": profile["id"], "remaining": profile["deployment_delay"],
                        "initial_delay": profile["deployment_delay"], "load": load,
                        "gain": load * profile["gain"], "worker": worker,
                        "resources": resources, "cost_per_slot": load * profile["cost"] / profile["deployment_delay"],
                        "launch_slot": slot, "steps": steps, "parent_version": agent.version}
            self.running, started = True, True
            self.started_jobs += 1
            self.training_steps += steps
        action, log_probability, value = agent.act(state, epsilon)
        demand = np.zeros_like(self.base)
        cost, training_id = 0.0, "tr0"
        if self.job is not None and self.running:
            demand += self.job["resources"]
            cost, training_id = self.job["cost_per_slot"], self.job["profile"]
        elif self.job is not None:
            self.paused_slots += 1
        before = int(self.queue_counts().sum()) + self.slot_rejected
        completed, missed, service_work = 0, self.slot_rejected, 0.0
        if decision.status == "feasible":
            profile = self.inference_by_id[decision.inference_profile]
            demand += np.tile(profile["resources"], (self.nodes, 1)) * observation["inference_load"]
            for node in range(self.nodes):
                budget = profile["resources"][1] * observation["inference_load"] * self.config["work_per_compute_unit"]
                order = [action] + [kind for kind in range(3) if kind != action]
                for kind in order:
                    queue = self.queues[node][kind]
                    while queue and budget > 1e-12:
                        task = queue[0]
                        used = min(budget, task[2])
                        budget -= used
                        task[2] -= used
                        service_work += used
                        if task[2] <= 1e-12:
                            queue.popleft()
                            completed += 1
                            self.latencies.append(slot - task[0] + 1)
                        else:
                            break
        self.resource_violations += int(np.any(demand + self.queue_memory > self.base + 1e-10))
        for queues in self.queues:
            for queue in queues:
                while queue and queue[0][1] <= slot:
                    queue.popleft()
                    missed += 1
        self.completed += completed
        self.expired += missed - self.slot_rejected
        self.total_cost += cost
        completion = completed / max(before, 1)
        deadline_loss = missed / max(before, 1)
        queue_level = min(1.0, float(self.queue_counts().sum()) / (self.nodes * self.config["queue_scale"]))
        weights = self.config["reward_weights"]
        reward = (weights["completion"] * completion - weights["deadline"] * deadline_loss
                  - weights["queue"] * queue_level - weights["training"] * cost)
        self.utilization = float(np.mean(demand[:, 1] / self.base[:, 1]))
        deploy = None
        if self.job is not None and self.running:
            self.job["remaining"] -= 1
            if self.job["remaining"] == 0:
                deploy, self.job = self.job, None
                self.deployed_jobs += 1
        feedback = {**observation,
                    "logged_training": training_id if decision.status == "feasible" else (training_id if training_id != "tr0" else None),
                    "logged_inference": decision.inference_profile, "policy_version": f"{agent.kind}:v{served_version}",
                    "execution_status": "executed" if decision.status == "feasible" else "no_dispatch",
                    "environment_reward": reward, "observed_training_cost": cost,
                    "completion_ratio": completion, "deadline_loss": deadline_loss,
                    "availability": 1.0 if decision.status == "feasible" else 0.0,
                    "throughput_ratio": min(1.0, completed / max(int(self.data["arrivals"][slot - 1].sum()), 1)),
                    "deployed_gain": None if deploy else 0.0,
                    "completed_training_profile": deploy["profile"] if deploy else None,
                    "completed_training_load": deploy["load"] if deploy else None}
        diagnostics = {"new_training": started, "application_action": action,
                       "log_probability": log_probability, "state_value": value,
                       "served_policy_version": served_version, "completed": completed,
                       "missed": missed, "arrivals": int(self.data["arrivals"][slot - 1].sum()),
                       "queue_after": int(self.queue_counts().sum()), "service_work": service_work,
                       "training_wall_ms": update_wall_ms, "running_job": self.job is not None}
        return feedback, diagnostics, deploy

    def apply_deployment(self, agent, job):
        if job is None:
            return
        if job["parent_version"] != agent.version:
            raise AssertionError("Stale training output cannot replace the deployed policy")
        before = agent.parameter_digest()
        agent.deploy(job["worker"])
        after = agent.parameter_digest()
        self.events.append({"launch_slot": job["launch_slot"], "deployment_slot": self.slot,
                            "profile": job["profile"], "delay": self.slot - job["launch_slot"] + 1,
                            "declared_delay": job["initial_delay"], "gradient_updates": job["steps"],
                            "gain_kind": "profile_estimate_on_completed_training",
                            "gain": job["gain"], "model_version": agent.version,
                            "parameters_changed": before != after,
                            "parameter_sha256_before": before, "parameter_sha256_after": after})
