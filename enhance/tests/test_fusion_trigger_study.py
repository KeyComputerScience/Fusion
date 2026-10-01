"""Causality, outage fairness, source identity, and real deployment checks."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fusion_trigger_study as f


class TriggerProtocolTests(unittest.TestCase):
    def test_shared_service_visibility_and_clock_window(self):
        cfg = copy.deepcopy(f.CONFIG)
        clean = [f.monitor_row(s, s % 3, 1., 0, False, "no_shift", cfg)
                 for s in range(800, 864)]
        outage = [f.monitor_row(s, s % 3, 1., 0, False,
                               "relation_shift_service_outage", cfg)
                  for s in range(800, 864)]
        self.assertEqual(f.observed_accuracy(clean, cfg), 1.)
        self.assertIsNone(f.observed_accuracy(outage, cfg))
        self.assertTrue(all(r["environment_reward"] is None for r in outage))
        self.assertTrue(all(r["regime"] is not None for r in outage))
        # Outages consume event time instead of allowing old clean feedback to
        # masquerade as the current 64-slot observation window.
        self.assertIsNone(f.observed_accuracy(clean + outage, cfg))

    def test_only_actual_two_sources_and_fail_closed_coverage(self):
        cfg = copy.deepcopy(f.CONFIG)
        fusion = f.BanditFusion("risk_rf", 10, cfg)
        for slot in range(800, 900):
            fusion.observe(f.monitor_row(slot, slot % 3, 1., 0, False,
                                         "relation_shift_service_outage", cfg))
        self.assertEqual(fusion.snapshot.masks, [True, False, False])
        self.assertEqual(fusion.snapshot.coverage, .5)
        self.assertIsNone(fusion.snapshot.scores[1])
        fusion.snapshot.gamma = 1.
        self.assertFalse(fusion.allows())
        missing = f.BanditFusion("risk_rf", 10, cfg)
        for slot in range(800, 900):
            missing.observe(f.monitor_row(slot, slot % 3, 1., 0, False,
                                          "relation_shift_all_source_loss", cfg))
        self.assertTrue(missing.snapshot.fallback)
        self.assertEqual(missing.snapshot.coverage, 0.)
        self.assertFalse(missing.allows())

    def test_admission_does_not_substitute_forecast_or_persistent_loss(self):
        fusion = f.BanditFusion("risk_rf", 10, f.CONFIG)
        snap = fusion.snapshot
        snap.fallback, snap.coverage, snap.uncertainty = False, 1., 0.
        snap.forecast_score, snap.gamma = .95, 0.
        self.assertFalse(fusion.allows())
        snap.forecast_score, snap.gamma = 0., .2
        self.assertTrue(fusion.allows())

    def test_seeded_onset_has_no_periodic_alignment_requirement(self):
        values = [f.hidden_onset(seed) for seed in f.CONFIG["seeds"]]
        self.assertTrue(all(768 <= value <= 1280 for value in values))
        self.assertGreater(len(set(values)), 1)
        self.assertTrue(any(value % 256 != 1 for value in values))
        self.assertEqual(values, [f.hidden_onset(seed) for seed in f.CONFIG["seeds"]])

    def test_actual_worker_budget_and_deployment_with_masked_monitor(self):
        cfg = copy.deepcopy(f.CONFIG)
        cfg.update(calibration_samples=512, evaluation_samples=512, max_jobs=1,
                   hidden_onset_minimum=256, hidden_onset_maximum=300,
                   fault_first_slot=100, fault_last_slot=300)
        calibration_contexts = np.random.default_rng(22).integers(0, 3, 512)
        agent, _, reference = f.calibrate(calibration_contexts, "dqn", 10, cfg)
        contexts = np.random.default_rng(23).integers(0, 3, 512)
        rows, events, _, summary = f.run_evaluation(
            contexts, agent, reference, "periodic", "relation_shift_service_outage", 10, cfg)
        self.assertEqual(summary["actual_gradient_updates"], 32)
        self.assertEqual(summary["training_jobs_started"], 1)
        self.assertEqual(events[0]["launch_slot"], 256)
        self.assertEqual(events[0]["deployment_slot"], 287)
        self.assertTrue(events[0]["parameters_changed"])
        self.assertEqual(events[0]["worker_parameter_sha256"], events[0]["deployment_parameter_sha256"])
        self.assertEqual(rows[286]["served_model_version"], 0)
        self.assertEqual(rows[287]["served_model_version"], 1)
        # Factual rewards still exist during a monitoring outage, without being
        # exposed to the observed-loss or fusion admission interfaces.
        self.assertTrue(all(r["actual_reward"] in (-1., 1.) for r in rows[99:300]))
        self.assertTrue(all(r["monitor_reward"] is None for r in rows[99:300]))


if __name__ == "__main__":
    unittest.main()
