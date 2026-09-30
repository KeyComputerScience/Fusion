"""Functional invariants, causality and coupled-constraint regression checks."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibrate_reference import fit_reference
from coordinator import Coordinator
from fusion import EvidenceFusion, FusionSnapshot, normalized_jsd
from generate_demo import generate
from logio import validate_config
from pipeline import FusionPipeline, run_replay


def config_and_profiles():
    return (json.loads((ROOT / "config.json").read_text()),
            json.loads((ROOT / "profiles.json").read_text()))


def decoded_demo(config, profiles, calibration_slots=40):
    rows = generate(config, profiles, calibration_slots)
    for row in rows:
        row["capacities"] = json.loads(row.pop("capacities_json"))
        row["completed_training_profile"] = row["completed_training_load"] = None
    return rows


def evidence_row(slot, regime="a", level=0.0, reward=1.0):
    return {"slot": slot, "regime": regime, "queue": level, "arrival_rate": level,
            "utilization": level, "cpu_fraction": level, "bandwidth": level,
            "environment_reward": reward, "observed_training_cost": 0.0,
            "logged_training": "tr0", "logged_inference": "inf0", "policy_version": "v1"}


class TestEvidence(unittest.TestCase):
    def setUp(self):
        config, _ = config_and_profiles()
        self.config = config["fusion"]
        self.config.update(window_size=4, minimum_source_samples=2, minimum_context_samples=1,
                           bootstrap_replicates=8, bootstrap_block_size=2)
        self.reference = {"fit_through_slot": 0, "state_columns": config["state_columns"],
                          "state_scales": {name: {"mean": 0.0, "std": 1.0}
                                           for name in config["state_columns"]}}

    def test_jsd_extremes_and_symmetry(self):
        self.assertEqual(normalized_jsd(["a", "b"], ["b", "a"]), 0.0)
        self.assertAlmostEqual(normalized_jsd(["a"] * 10, ["b"] * 10), 1.0)
        self.assertAlmostEqual(normalized_jsd(["a", "a", "b"], ["a", "c"]),
                               normalized_jsd(["a", "c"], ["a", "a", "b"]))

    def test_masking_bounds_and_hold_on_total_outage(self):
        engine = EvidenceFusion(self.config, self.reference)
        engine.complete_window([evidence_row(t) for t in range(1, 5)], 1)
        valid = engine.complete_window([evidence_row(t, "b", 1.0, 0.4) for t in range(5, 9)], 2)
        self.assertTrue(all(valid.masks))
        self.assertAlmostEqual(sum(valid.weights), 1.0)
        self.assertTrue(0 <= valid.disagreement <= 0.25)
        outage = [evidence_row(t, None, None, None) for t in range(9, 13)]
        fallback = engine.complete_window(outage, 3)
        self.assertTrue(fallback.fallback)
        self.assertEqual(fallback.gamma, valid.gamma)
        self.assertEqual((fallback.coverage, fallback.uncertainty), (0.0, 1.0))
        self.assertEqual(fallback.weights, [0.0, 0.0, 0.0])

    def test_one_source_does_not_imply_full_confidence(self):
        engine = EvidenceFusion(self.config, self.reference)
        before = [evidence_row(t, "a", None, None) for t in range(1, 5)]
        after = [evidence_row(t, "b", None, None) for t in range(5, 9)]
        engine.complete_window(before, 1)
        result = engine.complete_window(after, 2)
        self.assertEqual(result.masks, [True, False, False])
        self.assertEqual(result.weights, [1.0, 0.0, 0.0])
        self.assertEqual(result.disagreement, 0.0)
        self.assertGreater(result.uncertainty, 0.0)
        self.assertAlmostEqual(result.coverage, 1 / 3)

    def test_performance_context_matching_and_missing_overlap(self):
        engine = EvidenceFusion(self.config, self.reference)
        before = [evidence_row(t, reward=1.0) for t in range(1, 5)]
        after = [evidence_row(t, reward=0.5) for t in range(5, 9)]
        engine.complete_window(before, 1)
        result = engine.complete_window(after, 2)
        self.assertAlmostEqual(result.scores[2], 0.5 / (1 + self.config["epsilon_reward"]))
        for row in after:
            row["policy_version"] = "new_version"
        score, count, _ = engine._performance(before, after)
        self.assertIsNone(score)
        self.assertEqual(count, 0)

    def test_unsupported_bootstrap_variance_is_conservative(self):
        self.config["bootstrap_replicates"] = 0  # direct low-level robustness check
        engine = EvidenceFusion(self.config, self.reference)
        engine.complete_window([evidence_row(t) for t in range(1, 5)], 1)
        result = engine.complete_window([evidence_row(t, "b") for t in range(5, 9)], 2)
        self.assertEqual(result.noise_variances, [None, None, None])
        self.assertTrue(all(math.isfinite(v) for v in result.reliability))


class TestCoordination(unittest.TestCase):
    def setUp(self):
        config, profiles = config_and_profiles()
        self.config = config["coordination"]
        self.config.update(nodes=1, initial_recovery=1.0, eta_gamma=0.0, eta_uncertainty=0.0, xi=0.0)
        self.snapshot = FusionSnapshot(completed_slot=0, gamma=0.6, uncertainty=0.0,
                                       coverage=1.0, fallback=False)
        self.profiles = {
            "base_training_profile": "tr0",
            "training": [
                {"id":"tr0", "gain":0.0, "cost":0.0, "intensity":0.0, "deployment_delay":1, "resources":[0,0,0]},
                {"id":"tr1", "gain":0.3, "cost":0.1, "intensity":0.2, "deployment_delay":1, "resources":[0,0.35,0]},
                {"id":"tr2", "gain":0.9, "cost":0.3, "intensity":0.6, "deployment_delay":1, "resources":[0,0.6,0]}],
            "inference": [
                {"id":"low", "beta":0.6, "resources":[0,0.4,0], "quality_prior":{name:1.0 for name in self.config["quality_weights"]}},
                {"id":"high", "beta":1.0, "resources":[0,0.7,0], "quality_prior":{name:1.0 for name in self.config["quality_weights"]}}]}

    def test_exact_coupled_constraint_solution(self):
        engine = Coordinator(self.config, self.profiles)
        result = engine.choose(1, 1.0, 1.0, [[1,1,1]], self.snapshot, 1, "exact")
        self.assertEqual((result.training_profile, result.inference_profile), ("tr2", "low"))
        expected = 0.6 * (1 - math.exp(-1)) + 0.9 * math.exp(-1) - 0.12 * 0.3 - 0.04 * 0.6 ** 2
        self.assertAlmostEqual(result.objective, expected)
        result_tight = engine.choose(2, 1.0, 1.0, [[1,0.95,1]], self.snapshot, 1, "exact")
        self.assertEqual((result_tight.training_profile, result_tight.inference_profile), ("tr0", "high"))

    def test_ao_feasibility_and_monotone_sweeps(self):
        engine = Coordinator(self.config, self.profiles)
        result = engine.choose(1, 1.0, 1.0, [[1,1,1]], self.snapshot, 4, "ao")
        self.assertEqual(result.status, "feasible")
        self.assertTrue(result.coordinate_converged)
        self.assertTrue(all(after >= before - 1e-12 for before, after in zip(result.trajectory, result.trajectory[1:])))
        self.assertTrue(engine.feasible(engine.training_by_id[result.training_profile],
                                        engine.inference_by_id[result.inference_profile], 1, 1, [[1,1,1]]))

    def test_final_slot_gives_no_delayed_training_credit(self):
        result = Coordinator(self.config, self.profiles).choose(10, 1, 1, [[1,1,1]], self.snapshot, 0)
        self.assertEqual(result.training_profile, "tr0")

    def test_infeasible_request_never_returns_executable_pair(self):
        result = Coordinator(self.config, self.profiles).choose(1, 1, 1, [[1,0.1,1]], self.snapshot, 8)
        self.assertEqual(result.status, "infeasible_admission_required")
        self.assertIsNone(result.training_profile)
        self.assertIsNone(result.inference_profile)
        with self.assertRaises(ValueError):
            Coordinator(self.config, self.profiles).choose(1, 1, 1, [[1,-1,1]], self.snapshot, 8)

    def test_recommendation_is_not_a_deployed_gain(self):
        engine = Coordinator(self.config, self.profiles)
        recommendation = engine.choose(1, 1, 1, [[1,1,1]], self.snapshot, 1)
        self.assertEqual(recommendation.training_profile, "tr2")
        feedback = {"logged_training":"tr0", "logged_inference":"high", "deployed_gain":0.0}
        engine.observe(feedback, rho_used=0.9)
        self.assertAlmostEqual(engine.H, 0.9)
        feedback["deployed_gain"] = 0.2
        engine.observe(feedback, rho_used=0.9)
        self.assertAlmostEqual(engine.H, 1.01)


class TestStreamingCausality(unittest.TestCase):
    def setUp(self):
        self.config, self.profiles = config_and_profiles()
        self.config["evaluation_horizon"] = 120
        self.config["fusion"].update(window_size=10, minimum_source_samples=4,
                                      bootstrap_replicates=8, bootstrap_block_size=2)
        self.rows = decoded_demo(self.config, self.profiles)
        self.reference = fit_reference(self.rows, self.config, 40)

    def test_suffix_change_cannot_change_prefix_decisions_or_reference(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            if row["slot"] > 100:
                row["regime"] = "future_only_regime"
                row["environment_reward"] = -10.0
                row["deployed_gain"] = 2.0
                for name in self.config["state_columns"]:
                    row[name] = 100.0
        self.assertEqual(self.reference, fit_reference(changed, self.config, 40))
        original_decisions, original_windows, _ = run_replay(self.rows, self.config, self.profiles, self.reference)
        changed_decisions, changed_windows, _ = run_replay(changed, self.config, self.profiles, self.reference)
        self.assertEqual(original_decisions[:60], changed_decisions[:60])
        self.assertEqual(original_windows[:6], changed_windows[:6])
        self.assertNotEqual(original_windows[-1], changed_windows[-1])

    def test_window_feedback_becomes_available_only_next_slot(self):
        pipeline = FusionPipeline(self.config, self.profiles, self.reference)
        for row in self.rows[40:60]:
            choice = pipeline.decide(row["slot"], row["training_load"], row["inference_load"], row["capacities"])
            self.assertLess(choice.fusion_completed_slot, choice.slot)
            if row["slot"] == 60:
                self.assertEqual(choice.fusion_completed_slot, 50)
            pipeline.observe(row)
        row = self.rows[60]
        choice = pipeline.decide(row["slot"], row["training_load"], row["inference_load"], row["capacities"])
        self.assertEqual(choice.fusion_completed_slot, 60)
        with self.assertRaises(RuntimeError):
            pipeline.decide(row["slot"] + 1, 1, 1, row["capacities"])

    def test_partial_window_does_not_update_fusion(self):
        _, windows, summary = run_replay(self.rows[:57], self.config, self.profiles, self.reference)
        self.assertEqual(len(windows), 1)
        self.assertEqual(summary["unfinished_window_samples"], 7)

    def test_invalid_config_is_rejected(self):
        invalid = copy.deepcopy(self.config)
        invalid["fusion"]["epsilon_noise"] = 0
        with self.assertRaises(ValueError):
            validate_config(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
