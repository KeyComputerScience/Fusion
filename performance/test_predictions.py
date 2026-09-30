"""Checks for the exported numerical data and manuscript algorithm."""
import math
from pathlib import Path
import re
import unittest

from export_predictions import ROOT, export


class TestExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.rows = export()

    def test_snapshot_timing(self):
        self.assertEqual(len(self.rows), 1200)
        for case in self.data["checkpoints"]:
            self.assertLess(case["fusion"]["completed_slot"], case["slot"])
            self.assertEqual(case["baseline_recovery"][0], case["decision"]["recovery_state"])

    def test_values_and_delays(self):
        for case in self.data["checkpoints"]:
            for profile in case["training_terms"]:
                self.assertLessEqual(profile["exact_proxy_increment"], profile["tangent_credit"] + 1e-12)
                for h in range(min(profile["delay"], len(case["baseline_recovery"]))):
                    self.assertEqual(profile["conditional_recovery"][h], case["baseline_recovery"][h])
            if case["lookahead"] == 0:
                self.assertTrue(all(t["V"] == 0.0 for t in case["training_terms"]))

    def test_gain_uses_logged_feedback(self):
        for row in self.rows:
            self.assertTrue(math.isclose(row["recovery_after_feedback"],
                                        row["rho"] * row["H"] + row["deployed_gain_used"], abs_tol=1e-12))

    def test_missing_sources(self):
        cases = {c["slot"]: c for c in self.data["checkpoints"]}
        self.assertAlmostEqual(cases[1151]["fusion"]["coverage"], 2 / 3)
        self.assertTrue(cases[1251]["fusion"]["fallback"])
        self.assertFalse(cases[1251]["decision"]["training_allowed"])

    def test_algorithm_wording(self):
        for name in ("algorithm_revised.tex", "prediction_equations.tex", "prediction_case.tex"):
            content = (ROOT / name).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"measure\w*|protect\w*|previous\w*", content, flags=re.I))
        content = (ROOT / "algorithm_revised.tex").read_text(encoding="utf-8")
        self.assertIn("HandleInfeasibleAndObserve", content)
        self.assertNotIn("deployment-event queue", content)

    def test_posthoc_target_not_in_decision_inputs(self):
        for case in self.data["checkpoints"]:
            check = case["posthoc_prediction_check"]
            self.assertFalse(check["used_in_slot_decision"])
            self.assertNotIn("target", case["inputs"])
            if check["target_available_from_slot"] is not None:
                self.assertGreater(check["target_available_from_slot"], case["slot"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
