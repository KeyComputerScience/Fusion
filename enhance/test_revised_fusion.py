"""Behavioral checks for chronology, PSD risk, masking and convex optimization."""
import copy
import math
from types import SimpleNamespace
import unittest

import numpy as np

from revised_fusion import RevisedFusion, sensitivity_certificate, solve_weights


def snapshot(slot=50, scores=(0.1, 0.6, 0.2), predictions=(0.2, 0.3, 0.4), masks=(True, True, True)):
    return SimpleNamespace(
        completed_slot=slot, scores=list(scores), next_target_predictions=list(predictions),
        masks=list(masks), weights=[1 / 3] * 3, reliability=[1.0] * 3,
        effective_samples=[50] * 3, ages=[0] * 3, noise_variances=[0.01] * 3,
        gamma=sum(x for x in scores if x is not None) / sum(x is not None for x in scores),
        disagreement=0.1, uncertainty=0.4, coverage=sum(masks) / 3,
        fallback=False, fallback_reason="", effective_weight_cap=0.7)


class RevisedFusionTests(unittest.TestCase):
    def test_arrived_target_scores_stored_forecast_not_current_forecast(self):
        engine = RevisedFusion({}, residual_decay=0.9)
        first = snapshot(predictions=(0.1, 0.3, 0.5))
        engine.update(first)
        second = snapshot(100, scores=(0.8, 0.6, 0.2), predictions=(0.9, 0.9, 0.9))
        output = engine.update(second)
        residual = np.array([-0.1, 0.1, 0.3])
        np.testing.assert_allclose(engine.residual_matrix, 0.1 * np.outer(residual, residual), atol=1e-14)
        self.assertEqual(output.residual_updates, 1)
        alternative = RevisedFusion({}, residual_decay=0.9)
        alternative.update(first)
        second.next_target_predictions = [0.0, 0.0, 0.0]
        alternative.update(second)
        np.testing.assert_array_equal(alternative.residual_matrix, engine.residual_matrix)

    def test_no_target_or_incomplete_vector_does_not_invent_zero_error(self):
        engine = RevisedFusion({})
        engine.update(snapshot(predictions=(0.1, None, 0.4), masks=(True, False, True)))
        output = engine.update(snapshot(100))
        self.assertEqual(output.residual_updates, 0)
        self.assertEqual(output.residual_update_kind, "skipped_incomplete_residual_vector")
        engine.update(snapshot(150, scores=(0.2, 0.4, None), predictions=(0.2, 0.4, None), masks=(True, True, False)))
        self.assertEqual(engine.residual_updates, 0)

    def test_stable_control_context_filter(self):
        engine = RevisedFusion({})
        engine.update(snapshot(), {"stable": True, "key": ("tr0", "inf1", 1)})
        different = engine.update(snapshot(100), {"stable": True, "key": ("tr0", "inf1", 2)})
        self.assertEqual(different.residual_updates, 0)
        matching = engine.update(snapshot(150), {"stable": True, "key": ("tr0", "inf1", 2)})
        self.assertEqual(matching.residual_updates, 1)
        mixed = engine.update(snapshot(200), {"stable": False, "key": None})
        self.assertEqual(mixed.residual_updates, 1)
        self.assertTrue(mixed.action_conditioned)
        self.assertFalse(mixed.control_context_stable)

    def test_risk_stays_psd_under_delayed_outer_product_updates(self):
        engine = RevisedFusion({}, residual_decay=0.7)
        for k in range(1, 8):
            engine.update(snapshot(50 * k, scores=(0.1, 0.4, (k % 3) / 3),
                                   predictions=(0.2, 0.7, 0.3)))
            self.assertGreaterEqual(np.linalg.eigvalsh(engine.residual_matrix).min(), -1e-14)

    def test_raw_changes_are_distinct_from_common_forecast_disagreement(self):
        raw = snapshot(scores=(0.0, 1.0, 0.0), predictions=(0.2, 0.2, 0.2))
        output = RevisedFusion({}).update(raw)
        self.assertAlmostEqual(output.forecast_score, 0.2)
        self.assertAlmostEqual(output.disagreement, 0.0)
        self.assertAlmostEqual(output.uncertainty, 0.0)
        self.assertGreater(output.raw_disagreement, 0.0)
        self.assertNotAlmostEqual(output.gamma, output.forecast_score)
        self.assertEqual(output.original_gamma, raw.gamma)
        self.assertEqual(raw.disagreement, 0.1)  # Input snapshot was not mutated.

    def test_all_missing_abstains_and_singleton_has_cap_one(self):
        engine = RevisedFusion({})
        none = engine.update(snapshot(predictions=(None, None, None), masks=(False, False, False)))
        self.assertTrue(none.fallback)
        self.assertFalse(none.forecast_valid)
        self.assertIsNone(none.forecast_score)
        self.assertEqual(none.weights, [0, 0, 0])
        self.assertEqual((none.coverage, none.uncertainty), (0, 1))
        one = engine.update(snapshot(100, predictions=(0.2, None, None), masks=(True, False, False)))
        self.assertEqual(one.weights, [1, 0, 0])
        self.assertEqual(one.effective_weight_cap, 1)
        self.assertAlmostEqual(one.coverage, 1 / 3)

    def test_no_predictive_consistency_is_inserted_into_measurement_quality(self):
        raw_a, raw_b = snapshot(), snapshot()
        raw_a.predictive_errors = [0, 0, 0]
        raw_b.predictive_errors = [1, 0, 0]
        first, second = RevisedFusion({}).update(raw_a), RevisedFusion({}).update(raw_b)
        np.testing.assert_array_equal(first.quality_reference, second.quality_reference)
        np.testing.assert_array_equal(first.weights, second.weights)

    def test_chronology_and_json_diagnostics(self):
        engine = RevisedFusion({})
        output = engine.update(snapshot())
        self.assertIn("original_weights", engine.extras(output))
        self.assertIn("residual_second_moment", engine.extras(output))
        with self.assertRaises(ValueError):
            engine.update(snapshot())

    def test_capped_kl_limit_has_known_solution(self):
        weights, diagnostics = solve_weights([0.9, 0.08, 0.02], [True] * 3,
                                            residual_penalty=0, inertia=0)
        np.testing.assert_allclose(weights, [0.7, 0.24, 0.06], atol=1e-12)
        self.assertEqual(diagnostics["solver"], "capped_kl_water_filling")

    def test_large_residual_risk_downweights_bad_predictor(self):
        matrix = np.diag([1, 0, 0])
        weights, _ = solve_weights([1 / 3] * 3, [True] * 3, residual_matrix=matrix,
                                   residual_penalty=8, inertia=0)
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(weights[1], weights[2], places=6)
        masked, _ = solve_weights([0.9, 0.05, 0.05], [False, True, True], residual_matrix=matrix)
        self.assertEqual(masked[0], 0)
        self.assertAlmostEqual(masked.sum(), 1)

    def test_convex_solution_beats_feasible_grid_and_is_not_diagonal_variance(self):
        matrix = np.array([[0.8, 0.2, -0.1], [0.2, 0.3, 0.05], [-0.1, 0.05, 0.4]])
        self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0)
        reference = np.array([0.6, 0.25, 0.15])
        previous = np.array([0.2, 0.4, 0.4])
        weights, metadata = solve_weights(reference, [True] * 3, residual_matrix=matrix,
                                          previous=previous, residual_penalty=2, inertia=0.7)

        def objective(a):
            a = np.asarray(a)
            nonzero = a > 0
            return (np.sum(a[nonzero] * np.log(a[nonzero] / reference[nonzero]))
                    + a @ matrix @ a + 0.35 * np.sum((a - previous) ** 2))

        grid = []
        for first in np.arange(0, 0.701, 0.02):
            for second in np.arange(0, 0.701, 0.02):
                third = 1 - first - second
                if -1e-12 <= third <= 0.7:
                    grid.append(objective([first, second, max(0, third)]))
        self.assertLessEqual(objective(weights), min(grid) + 1e-8)
        diagonal, _ = solve_weights(reference, [True] * 3, residual_matrix=np.diag(np.diag(matrix)),
                                    previous=previous, residual_penalty=2, inertia=0.7)
        self.assertGreater(np.linalg.norm(weights - diagonal), 0.001)
        self.assertLess(metadata["kkt_residual"], 2e-5)

    def test_invalid_risk_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_weights([1 / 3] * 3, [True] * 3, residual_matrix=np.diag([-1, 1, 1]))

    def test_adaptive_certificate_accounts_for_risk_and_inertia_changes(self):
        engine = RevisedFusion({}, residual_penalty=3, inertia=0.8)
        first = engine.update(snapshot())
        incoming = snapshot(100, scores=(0.3, 0.6, 0.8), predictions=(0.25, 0.45, 0.55))
        incoming.noise_variances = [0.05, 0.01, 0.08]
        second = engine.update(incoming)
        for kind in ("forecast", "raw"):
            certificate = sensitivity_certificate(first, second, score_kind=kind)
            self.assertTrue(certificate["applicable"])
            self.assertLessEqual(certificate["weight_l1_observed"], certificate["weight_l1_bound"] + 2e-5)
            self.assertLessEqual(certificate["score_change_observed"], certificate["score_change_bound"] + 2e-5)
        absent = engine.update(snapshot(150, predictions=(0.2, None, 0.4), masks=(True, False, True)))
        self.assertFalse(sensitivity_certificate(second, absent)["applicable"])


if __name__ == "__main__":
    unittest.main()
