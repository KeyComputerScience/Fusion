"""Check deployment causality, reward independence, paired inputs and trace units."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import EdgeService
from closed_loop import calibrate, method_config, run_closed
from coordinator import Coordinator, Decision
from data_pipeline import generate, import_alibaba, read_json
from fusion import EvidenceFusion, FusionSnapshot
from policies import ServiceAgent


def parameters():
    return (read_json(ROOT / "experiment_config.json"), read_json(ROOT / "core/config.json"),
            read_json(ROOT / "core/profiles.json"))


class ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.experiment, self.core, self.profiles = parameters()

    def test_tiny_objectives_retain_inference_argmax(self):
        config = copy.deepcopy(self.core["coordination"])
        config["initial_recovery"] = 1e-30
        engine = Coordinator(config, self.profiles)
        snapshot = FusionSnapshot(gamma=0, uncertainty=0, coverage=1, fallback=False)
        decision = engine.choose(1, 0, 0.5, [[1, 1, 1]] * 10, snapshot, 16)
        self.assertEqual(decision.inference_profile, "inf5")
        self.assertGreater(decision.objective, 0)

    def test_private_training_cannot_change_deployed_parameters(self):
        for kind in ("dqn", "ppo"):
            agent = ServiceAgent(kind, self.experiment["agent"], 10)
            for t in range(128):
                state = [t / 128] * 14
                action, logged, value = agent.act(state, 0)
                agent.record(state, action, 0.4, state, False, logged, value)
            before, random_state = agent.parameter_digest(), agent.rng.getstate()
            worker, updates = agent.worker(2)
            self.assertEqual(updates, 2)
            self.assertEqual(before, agent.parameter_digest())
            self.assertNotEqual(before, worker.parameter_digest())
            agent.deploy(worker)
            self.assertEqual(agent.version, 1)
            self.assertEqual(worker.parameter_digest(), agent.parameter_digest())
            self.assertEqual(random_state, agent.rng.getstate())

    def test_training_resources_pause_and_delay_deployment(self):
        experiment = copy.deepcopy(self.experiment)
        experiment.update(calibration_slots=0, evaluation_slots=4)
        data = {"arrivals": np.zeros((4, 10, 3), int), "capacities": np.ones((4, 10, 3)),
                "provenance": np.array("controlled_synthetic")}
        data["capacities"][1, :, 1] = 0.02
        agent = ServiceAgent("dqn", experiment["agent"], 10)
        for _ in range(128):
            agent.record([0.] * 14, 0, 0.3, [0.] * 14, False)
        env = EdgeService(data, experiment, self.profiles)
        before = agent.parameter_digest()
        deployments = []
        for slot in range(1, 5):
            obs, _ = env.begin(slot, agent)
            tr = "tr2" if slot == 1 else "tr0"
            feasible = slot != 2
            decision = Decision(slot, tr if feasible else None, "inf0" if feasible else None,
                                0.0, "test", "feasible" if feasible else "infeasible_admission_required",
                                True, 1., 1., 0, 0., 0., 1., 1, [])
            state = env.state(obs, FusionSnapshot(), 1., decision.inference_profile, slot)
            _, _, job = env.execute(decision, agent, state)
            if slot < 3:
                self.assertEqual(before, agent.parameter_digest())
                self.assertIsNone(job)
            env.apply_deployment(agent, job)
            if job:
                deployments.append(slot)
        self.assertEqual(deployments, [3])
        self.assertEqual(env.paused_slots, 1)
        self.assertEqual(env.events[0]["delay"], 3)
        self.assertEqual(env.resource_violations, 0)

    def test_external_reward_does_not_use_recovery_or_drift(self):
        data = generate(self.experiment, 10, 10)
        agent = ServiceAgent("dqn", self.experiment["agent"], 10)
        # Same physical execution, different analytical variables.
        rewards = []
        for recovery, gamma in [(0., 0.), (100., 1.)]:
            env = EdgeService(data, self.experiment, self.profiles)
            env.begin(1, agent)
            decision = Decision(1, "tr0", "inf5", 999., "test", "feasible", False,
                                1., recovery, 0, gamma, 1., 0., 1, [])
            feedback, _, _ = env.execute(decision, copy.deepcopy(agent), [0.] * 14)
            weights = self.experiment["reward_weights"]
            queue = float(env.queue_counts().sum()) / (10 * self.experiment["queue_scale"])
            expected = feedback["completion_ratio"] - 0.5 * feedback["deadline_loss"] - 0.05 * min(1., queue)
            self.assertAlmostEqual(feedback["environment_reward"], expected)
            rewards.append(feedback["environment_reward"])
        self.assertEqual(*rewards)

    def test_exogenous_arrays_are_seed_deterministic(self):
        a, b = generate(self.experiment, 10, 20), generate(self.experiment, 10, 20)
        np.testing.assert_array_equal(a["arrivals"], b["arrivals"])
        np.testing.assert_array_equal(a["capacities"], b["capacities"])
        self.assertFalse(np.array_equal(a["arrivals"], generate(self.experiment, 10, 30)["arrivals"]))
        self.assertEqual(a["rates"][1200 + 900], 2.2)
        self.assertEqual(a["rates"][1200 + 899], 1.15)

    def test_trace_units_prefix_selection_and_nonfinite_rejection(self):
        experiment = copy.deepcopy(self.experiment)
        experiment.update(calibration_slots=2, evaluation_slots=3)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            usage, tasks, output = directory / "usage.csv", directory / "tasks.csv", directory / "trace.npz"
            with usage.open("w", newline="") as stream:
                csv.writer(stream).writerows([
                    ["a", 0, 20, 40], ["a", 60, 40, 60],
                    ["b", 0, 10, 20], ["b", 120, 10, 20], ["b", 180, 10, 20], ["b", 240, 10, 20],
                    ["a", 120, "nan", 60]])
            with tasks.open("w", newline="") as stream:
                csv.writer(stream).writerows([
                    ["t1", 2, "j", 1, "Terminated", 0, 1, 100, 50],
                    ["t2", 2, "j", 1, "Terminated", 60, 61, 100, 50],
                    ["bad", 1, "j", 1, "Terminated", 120, 121, "nan", 50]])
            import_alibaba(usage, tasks, experiment, 1, 60, output)
            imported = dict(np.load(output, allow_pickle=False))
            meta = read_json(output.with_suffix(".json"))
            self.assertEqual(meta["machines"], ["a"])
            self.assertAlmostEqual(imported["capacities"][0, 0, 0], 0.6)
            self.assertAlmostEqual(imported["capacities"][1, 0, 1], 0.6)
            self.assertAlmostEqual(meta["task_memory_mean"], 0.5)
            self.assertEqual(meta["invalid_task_rows"], 1)
            self.assertEqual(meta["invalid_usage_rows"], 1)
            np.testing.assert_array_equal(imported["capacities"][1], imported["capacities"][4])

    def test_complete_window_and_outage_are_explicit(self):
        config = copy.deepcopy(self.core["fusion"])
        config.update(window_size=4, minimum_source_samples=2, minimum_context_samples=1,
                      bootstrap_replicates=4, bootstrap_block_size=2)
        reference = {"fit_through_slot": 0, "state_columns": self.core["state_columns"],
                     "state_scales": {key: {"mean": 0., "std": 1.} for key in self.core["state_columns"]}}
        def row(slot, present=True):
            return {"slot": slot, "regime": "light" if present else None,
                    **{key: 0. if present else None for key in reference["state_columns"]},
                    "environment_reward": 1. if present else None, "observed_training_cost": 0.,
                    "logged_training": "tr0", "logged_inference": "inf5", "policy_version": "v0"}
        fusion = EvidenceFusion(config, reference)
        with self.assertRaises(ValueError):
            fusion.complete_window([row(1)], 1)
        fusion.complete_window([row(t) for t in range(1, 5)], 1)
        clean = fusion.complete_window([row(t) for t in range(5, 9)], 2)
        self.assertEqual(clean.to_dict()["available_from_slot"], 9)
        outage = fusion.complete_window([row(t, False) for t in range(9, 13)], 3)
        self.assertTrue(outage.fallback)
        self.assertEqual(outage.uncertainty, 1.)
        self.assertEqual(outage.coverage, 0.)

    def test_short_closed_loop_is_reproducible_and_causal(self):
        experiment = copy.deepcopy(self.experiment)
        experiment.update(calibration_slots=256, evaluation_slots=300, drift_onsets=[101, 201])
        experiment["agent"]["warmup"] = 128
        core = copy.deepcopy(self.core)
        core["evaluation_horizon"] = 300
        data = generate(experiment, 10, 10)
        agent, reference, profiles, _ = calibrate(data, experiment, core, self.profiles, 10, "dqn")
        outcomes = [run_closed(data, experiment, core, profiles, agent, reference, "fusion", 10) for _ in range(2)]
        self.assertEqual(outcomes[0][3]["return"], outcomes[1][3]["return"])
        self.assertEqual(outcomes[0][3]["final_parameter_sha256"], outcomes[1][3]["final_parameter_sha256"])
        records, windows, events, summary = outcomes[0]
        self.assertEqual(summary["resource_violations"], 0)
        self.assertEqual(summary["parameters_changed_events"], summary["deployed_jobs"])
        self.assertTrue(all(r["fusion_completed_slot"] < r["slot"] for r in records))
        self.assertEqual(summary["complete_windows"], 6)
        self.assertTrue(all(e["delay"] >= e["declared_delay"] for e in events))
        self.assertEqual(summary["arrival_count"], summary["completed_count"] + summary["deadline_count"] + summary["queue_end"])
        self.assertTrue(all(r["application_action"] in (0, 1, 2) for r in records))


if __name__ == "__main__":
    unittest.main()
