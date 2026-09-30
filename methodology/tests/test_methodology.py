"""Mathematical boundary cases and solver claims added by the core rewrite."""
import copy
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coordinator import Coordinator
from fusion import capped_weights
from pipeline import FusionPipeline
from proxy import RecoveryForecast, geometric_sum
from test_core import config_and_profiles, decoded_demo
import test_core as core_fixture
from calibrate_reference import fit_reference


class TestProjection(unittest.TestCase):
    def test_high_precision_source_cannot_capture_all_mass(self):
        weights = capped_weights([100.0, 1.0, 1.0], 0.7)
        for actual, expected in zip(weights, [0.7, 0.15, 0.15]):
            self.assertAlmostEqual(actual, expected)

    def test_cap_relaxes_only_to_make_available_simplex_feasible(self):
        self.assertEqual(capped_weights([100, 0, 1], 0.3), [0.5, 0.0, 0.5])
        self.assertEqual(capped_weights([0, 2, 0], 0.7), [0.0, 1.0, 0.0])
        self.assertEqual(capped_weights([0, 0, 0], 0.7), [0.0, 0.0, 0.0])

    def test_projection_constraints_and_conditional_influence(self):
        rng = random.Random(37)
        for _ in range(100):
            reliability = [10 ** rng.uniform(-5, 5) for _ in range(3)]
            weights = capped_weights(reliability, 0.6)
            self.assertAlmostEqual(sum(weights), 1.0)
            self.assertTrue(all(0 <= w <= 0.6 + 1e-13 for w in weights))
            before = [rng.random() for _ in range(3)]
            after = list(before)
            after[1] = 1.0 - before[1]
            delta = abs(sum(w*(b-a) for w, a, b in zip(weights, before, after)))
            self.assertLessEqual(delta, 0.6 * abs(after[1]-before[1]) + 1e-13)


class TestDelayedBounds(unittest.TestCase):
    def test_current_state_tangent_is_not_a_future_increment_bound(self):
        forecast = RecoveryForecast.construct(2.0, 0.5, 1, 1.0)
        gain = 0.1
        increment = forecast.exact_increment(gain, 1)
        self.assertGreater(increment, math.exp(-2.0) * gain)
        self.assertLessEqual(increment, forecast.coefficient(1) * gain)

    def test_nonconstant_forecast_pending_jobs_and_delay(self):
        forecast = RecoveryForecast.construct(2.0, 0.9, 4, 0.8,
                                             [0.5, 0.8, 0.7, 0.6], {2:0.3, 4:0.2})
        self.assertAlmostEqual(forecast.baseline[2], 1.1)
        self.assertEqual(list(forecast.factors(2)), [(2,1.0), (3,0.7), (4,0.42)])
        for gain in (0, 0.01, 0.2, 2, 20):
            actual = forecast.exact_increment(gain, 2)
            tangent = forecast.coefficient(2) * gain
            coarse = forecast.coefficient(2, "geometric_upper") * gain
            self.assertTrue(0 <= actual <= tangent + 1e-12 <= coarse + 2e-12)

    def test_horizon_and_deployment_boundaries(self):
        forecast = RecoveryForecast.construct(1.0, 0.9, 4, 1.0)
        self.assertAlmostEqual(forecast.coefficient(4), math.exp(-0.9**4))
        self.assertEqual(forecast.coefficient(5), 0.0)
        self.assertEqual(forecast.exact_increment(1, 5), 0.0)
        self.assertEqual(RecoveryForecast.construct(1, 0.9, 0, 1).coefficient(1), 0.0)

    def test_geometric_zero_one_and_near_one_limits(self):
        self.assertEqual(geometric_sum(0, 0), 0.0)
        self.assertEqual(geometric_sum(0, 4), 1.0)
        self.assertEqual(geometric_sum(1, 4), 4.0)
        self.assertAlmostEqual(geometric_sum(1-1e-12, 1000),
                               sum((1-1e-12)**h for h in range(1000)), places=9)
        with self.assertRaises(ValueError):
            geometric_sum(0.9, 2.5)

    def test_conditional_bound_grid(self):
        for recovery in (0, 0.1, 1, 10):
            for rho in (0, 0.5, 0.999999, 1):
                forecast = RecoveryForecast.construct(recovery, rho, 16, 0.7)
                for delay in (1, 4, 16, 17):
                    for gain in (0.001, 0.1, 1, 10):
                        actual = forecast.exact_increment(gain, delay)
                        bound = forecast.coefficient(delay) * gain
                        self.assertLessEqual(actual, bound + 1e-12)


class TestPruningAndAO(unittest.TestCase):
    def fixture(self):
        # Reuse the known coupled-resource fixture, without importing its tests.
        fixture = core_fixture.TestCoordination()
        fixture.setUp()
        return fixture.config, fixture.profiles, fixture.snapshot

    def test_pruning_preserves_known_optima_at_different_loads(self):
        config, profiles, snapshot = self.fixture()
        inferior = copy.deepcopy(profiles["training"][1])
        inferior.update(id="dominated", cost=0.5, resources=[0,0.5,0])
        profiles["training"].append(inferior)
        for load in (0.1, 0.7, 1.0, 1.4):
            ordinary = Coordinator(config, profiles).choose(1, load, 1, [[1,1,1]], snapshot, 4, "exact")
            pruned_config = dict(config, dominance_pruning=True)
            pruned = Coordinator(pruned_config, profiles).choose(1, load, 1, [[1,1,1]], snapshot, 4, "exact")
            self.assertAlmostEqual(pruned.objective, ordinary.objective)
            self.assertGreaterEqual(pruned.pruned_training_profiles, 1)

    def test_pruned_warm_pair_does_not_reenter_ao(self):
        config, profiles, snapshot = self.fixture()
        inferior = copy.deepcopy(profiles["inference"][0])
        inferior.update(id="dominated_inf", beta=0.3)
        profiles["inference"].append(inferior)
        config["dominance_pruning"] = True
        engine = Coordinator(config, profiles)
        engine.warm_start = ("tr0", "dominated_inf")
        choice = engine.choose(1, 1, 1, [[1,1,1]], snapshot, 4, "ao")
        self.assertNotEqual(choice.inference_profile, "dominated_inf")
        self.assertGreaterEqual(choice.pruned_inference_profiles, 1)

    def test_ao_can_have_a_positive_global_gap(self):
        config, _, snapshot = self.fixture()
        config.update(omega_intensity=0.0, lambda_cost=0.0)
        profiles = {"base_training_profile":"tr0", "training":[], "inference":[]}
        for index, demand, value in ((0,0,0), (1,0.4,0.5), (2,0.8,0.6)):
            profiles["training"].append({"id":f"tr{index}", "gain":value/math.exp(-1),
                "cost":0, "intensity":0, "deployment_delay":1, "resources":[0,demand,0]})
        for name, demand, beta in (("high",0.9,1.0), ("mid",0.55,0.4), ("low",0.2,0.1)):
            profiles["inference"].append({"id":name, "beta":beta, "resources":[0,demand,0],
                "quality_prior":{metric:1.0 for metric in config["quality_weights"]}})
        exact = Coordinator(config, profiles).choose(1,1,1,[[1,1,1]],snapshot,1,"exact")
        ao = Coordinator(config, profiles).choose(1,1,1,[[1,1,1]],snapshot,1,"ao")
        self.assertEqual((exact.training_profile, exact.inference_profile), ("tr1","mid"))
        self.assertTrue(ao.coordinate_converged)
        self.assertGreater(exact.objective-ao.objective, 0.08)

    def test_live_history_is_bounded_by_default(self):
        config, profiles = config_and_profiles()
        config["evaluation_horizon"] = 40
        config["fusion"].update(window_size=10, minimum_source_samples=4,
                               bootstrap_replicates=8, bootstrap_block_size=2)
        rows = decoded_demo(config, profiles)
        pipeline = FusionPipeline(config, profiles, fit_reference(rows, config, 40))
        for row in rows[40:]:
            pipeline.decide(row["slot"], row["training_load"], row["inference_load"], row["capacities"])
            pipeline.observe(row)
        self.assertEqual(pipeline.window_results, [])
        self.assertLessEqual(len(pipeline.fusion.previous_rows), 10)

    def test_no_dispatch_skips_quality_but_records_pending_deployment(self):
        config, profiles, _ = self.fixture()
        engine = Coordinator(config, profiles)
        quality = copy.deepcopy(engine.quality)
        engine.observe({"logged_inference":None, "logged_training":None,
                        "execution_status":"no_dispatch", "deployed_gain":0.2,
                        "completion_ratio":0.0},rho_used=0.9)
        self.assertAlmostEqual(engine.H, 1.1)
        self.assertEqual(engine.quality, quality)


if __name__ == "__main__":
    unittest.main(verbosity=2)
