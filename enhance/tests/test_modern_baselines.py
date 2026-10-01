"""Analytic and causal checks for actual aggregation rules, not result rankings."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modern_baselines import FusionBaseline, ReplayBaseline, dempster_binary


def snapshot(p=(0.1, 0.5, 0.9), masks=(True, True, True)):
    return SimpleNamespace(scores=[0.2, 0.4, 0.8], next_target_predictions=list(p),
                           masks=list(masks), predictive_errors=[0.05, 0.05, 0.05],
                           completed_slot=100, window_index=2)


class BinaryDempsterTests(unittest.TestCase):
    def test_one_source_discount_and_ignorance(self):
        result = dempster_binary([0.8], [0.5])
        self.assertAlmostEqual(result["mass_change"], 0.4)
        self.assertAlmostEqual(result["mass_stable"], 0.1)
        self.assertAlmostEqual(result["mass_ignorance"], 0.5)
        self.assertAlmostEqual(result["probability"], 0.65)

    def test_two_sources_closed_form(self):
        result = dempster_binary([0.8, 0.6], [1.0, 1.0])
        self.assertAlmostEqual(result["conflict"], 0.44)
        self.assertAlmostEqual(result["probability"], 0.48 / 0.56)
        self.assertAlmostEqual(sum(result[key] for key in ("mass_change", "mass_stable", "mass_ignorance")), 1.0)

    def test_total_conflict_abstains(self):
        result = dempster_binary([1.0, 0.0], [1.0, 1.0])
        self.assertIsNone(result["probability"])
        self.assertTrue(result["fallback"])

    def test_ignorance_is_neutral_and_order_does_not_matter(self):
        a = dempster_binary([0.2, 0.8, 0.4], [0.7, 0.8, 0.0])
        b = dempster_binary([0.8, 0.2], [0.8, 0.7])
        for key in ("probability", "conflict", "mass_change", "mass_stable", "mass_ignorance"):
            self.assertAlmostEqual(a[key], b[key])


class ForecastAggregationTests(unittest.TestCase):
    def test_equal_and_fixed_dual_scales(self):
        equal = FusionBaseline("EF").predict(snapshot())
        self.assertAlmostEqual(equal.prediction, 0.5)
        self.assertAlmostEqual(equal.raw_score, 1.4 / 3)
        fixed = FusionBaseline("FF", {"fixed_weights": [0.8, 0.1, 0.1]}).predict(snapshot())
        self.assertAlmostEqual(fixed.prediction, 0.22)
        self.assertAlmostEqual(fixed.raw_score, 0.28)

    def test_precision_uses_delayed_squared_calibrated_loss(self):
        learner = FusionBaseline("IMSE", {"error_rate": 1.0})
        previous = learner.predict(snapshot())
        learner.observe(previous, 0.1)
        current = learner.predict(snapshot())
        self.assertGreater(current.weights[0], current.weights[1])
        self.assertGreater(current.weights[1], current.weights[2])

    def test_no_outcome_means_no_state_update(self):
        learner = FusionBaseline("BOA")
        previous = learner.predict(snapshot())
        old_mse, old_weights = list(learner.mse), list(learner.log_weights)
        self.assertFalse(learner.observe(previous, None))
        self.assertEqual(old_mse, learner.mse)
        self.assertEqual(old_weights, learner.log_weights)
        self.assertEqual(learner.update_count, 0)

    def test_ewa_closed_form_update(self):
        learner = FusionBaseline("EWA", {"learning_rate": 0.5})
        previous = learner.predict(snapshot())
        learner.observe(previous, 0.1)
        current = learner.predict(snapshot())
        expected = [math.exp(-0.5 * loss) for loss in (0.0, 0.16, 0.64)]
        expected = [w / sum(expected) for w in expected]
        for got, want in zip(current.weights, expected):
            self.assertAlmostEqual(got, want)

    def test_boa_second_order_update_uses_saved_forecast(self):
        learner = FusionBaseline("BOA", {"learning_rate": 0.5})
        previous = learner.predict(snapshot())
        learner.observe(previous, 0.1)
        current = learner.predict(snapshot(p=(0.9, 0.9, 0.9)))
        losses = [0.0, 0.16, 0.64]
        excess = [v - sum(losses) / 3 for v in losses]
        expected = [math.exp(-0.5 * e - 0.25 * e * e) for e in excess]
        expected = [v / sum(expected) for v in expected]
        for got, want in zip(current.weights, expected):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(previous.prediction, 0.5)

    def test_sleeping_expert_receives_no_loss(self):
        learner = FusionBaseline("EWA")
        previous = learner.predict(snapshot(masks=(True, False, True)))
        before = learner.mse[1]
        learner.observe(previous, 0.1)
        self.assertEqual(learner.mse[1], before)
        self.assertEqual(previous.weights[1], 0.0)

    def test_all_missing_abstains(self):
        for method in ("EF", "FF", "IMSE", "EWA", "BOA", "DS"):
            forecast = FusionBaseline(method).predict(snapshot(masks=(False, False, False)))
            self.assertIsNone(forecast.prediction)

    def test_ds_does_not_invent_linear_influence(self):
        forecast = FusionBaseline("DS").predict(snapshot())
        self.assertIsNone(forecast.weights)
        self.assertIsNotNone(forecast.masses)
        self.assertTrue(0.0 <= forecast.prediction <= 1.0)

    def test_replay_adapter_scores_previous_forecast_on_new_visible_target(self):
        config = dict(window_size=10, predictive_calibration=dict(ridge=0.1, forgetting=0.99,
                      error_rate=0.15, error=0.05), minimum_source_samples=2,
                      minimum_context_samples=1, observed_cost_reward_scale=0.12,
                      epsilon_reward=1e-6, bootstrap_replicates=3, bootstrap_block_size=2,
                      seed=10, unsupported_noise_variance=0.25, support_scale=20,
                      freshness_scale=50, error_scale=0.2, epsilon_noise=0.0025,
                      missing_source_penalty=0.5, maximum_source_weight=0.7)
        reference = dict(fit_through_slot=0, state_columns=["cpu_fraction"],
                         state_scales={"cpu_fraction": dict(mean=0.5, std=0.1)})
        replay = ReplayBaseline("BOA", config, reference)
        def rows(start, reward):
            return [dict(slot=t, regime="light", cpu_fraction=0.5,
                         environment_reward=reward, observed_training_cost=0.0,
                         logged_training="tr0", logged_inference="in0", policy_version="v0")
                    for t in range(start, start + 10)]
        first = replay.complete_window(rows(1, 1.0), 1)
        second = replay.complete_window(rows(11, 0.8), 2)
        self.assertTrue(first.fallback)
        self.assertFalse(second.fallback)
        self.assertEqual(replay.baseline.update_count, 0)
        replay.complete_window(rows(21, 0.7), 3)
        self.assertEqual(replay.baseline.update_count, 1)
        replay.complete_window(rows(31, None), 4)
        self.assertEqual(replay.baseline.update_count, 1)


if __name__ == "__main__":
    unittest.main()
