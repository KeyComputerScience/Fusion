"""Executable DA-RF regression tests using unittest and NumPy only.

Run from any directory: python /absolute/path/test_da_rf.py
These tests validate code contracts and declared synthetic mechanisms, not
counterfactual model-recovery calibration or an edge-service benefit.
"""
from __future__ import annotations

from dataclasses import replace
import json
import unittest
import numpy as np

from da_rf_fusion import (DecisionAwareFusion, FusionConfig, FusionSnapshot,
                          QualityObservation, solve_capped_fusion, sensitivity_bound)
from da_rf_predictor import DelayedRidgeForecaster, RidgeConfig
from da_rf_coordinator import Candidate, CoordinatorConfig, PersistentRecoveryCoordinator
from da_rf_evidence import persistent_loss, matched_service_change, block_bootstrap_variance
from da_rf_calibration import PairedRecoverySample, PairedRecoveryCalibrator
from run_da_rf_demo import gradient_recoverability_demo, logistic_update, permanent_outage_moment_demo


def observations(mask, variances=None):
    variances = [0.02] * len(mask) if variances is None else variances
    return [QualityObservation(bool(m), 50 if m else 0, 0,
                               variances[i] if m else None, 0.2 if m else None)
            for i, m in enumerate(mask)]


def fixture_snapshot(window=1, forecast=0.6, uncertainty=0.1,
                     coverage=1.0, all_missing=False, solver_fallback=False):
    return FusionSnapshot(window=window,
                          weights=np.zeros(3) if all_missing else np.full(3, 1 / 3),
                          forecast=None if all_missing else forecast,
                          uncertainty=uncertainty, coverage=coverage,
                          all_missing=all_missing, solver_fallback=solver_fallback)


def base_candidate(**overrides):
    values = dict(name="no_update", update=False, gross_baseline=1.0, gain=0.0,
                  error_allowance=0.01, cost=0.0, resource_demand=np.array([0.1]),
                  gradient_steps=0, delay=0, supported=True)
    values.update(overrides)
    return Candidate(**values)


def update_candidate(**overrides):
    values = dict(name="update", update=True, gross_baseline=1.0, gain=0.5,
                  error_allowance=0.01, cost=0.2, resource_demand=np.array([0.3]),
                  gradient_steps=8, delay=2, supported=True, support_count=20)
    values.update(overrides)
    return Candidate(**values)


class CalibrationTests(unittest.TestCase):
    @staticmethod
    def rows(prefix):
        rows=[]
        for i,z in enumerate([-1.0,-0.5,0.5,1.0]):
            for j,y in enumerate([0.1,0.4,0.7,0.9]):
                base=2-0.3*y+0.1*z
                gain=0.5*y+0.2*z*y-0.05
                origin=f'{prefix}-{i}-{j}'
                rows.extend([
                    PairedRecoverySample('no','infer',False,np.array([z]),y,base,base,origin),
                    PairedRecoverySample('update','infer',True,np.array([z]),y,base,base+gain,origin)])
        return rows

    def test_independent_affine_model_recovers_slope_and_shared_baseline(self):
        model=PairedRecoveryCalibrator(1,ridge=1e-9,min_support=3).fit(
            self.rows('train'),self.rows('held'))
        c,b,audit=model.local_coefficients('update',[0.4])
        self.assertAlmostEqual(c,1.99,places=7)
        self.assertAlmostEqual(b,0.28,places=7)
        estimate=model.predict('update',[0.4],0.5)
        reference=model.predict('no',[0.4],0.5)
        self.assertAlmostEqual(estimate.baseline,1.89,places=7)
        self.assertAlmostEqual(estimate.gain,0.24,places=7)
        self.assertEqual(estimate.baseline,reference.baseline)
        self.assertEqual(reference.gain,0)
        self.assertFalse(estimate.certificate_valid)
        self.assertEqual(estimate.calibration_state,'supported')

    def test_counterfactual_calibration_does_not_invent_unknown_candidate(self):
        model=PairedRecoveryCalibrator(1).fit(self.rows('train'),self.rows('held'))
        unknown=model.predict('unmeasured',[0],0.5)
        self.assertIsNone(unknown.gain)
        self.assertIsNone(unknown.error_allowance)
        self.assertEqual(unknown.calibration_state,'unsupported')
        self.assertFalse(unknown.certificate_valid)
        with self.assertRaises(KeyError):
            model.local_coefficients('unmeasured',[0])

    def test_same_fork_cannot_leak_across_calibration_and_validation(self):
        with self.assertRaises(ValueError):
            PairedRecoveryCalibrator(1).fit(self.rows('same'),self.rows('same'))

    def test_missing_validation_blocks_supported_slope_and_refitting(self):
        model=PairedRecoveryCalibrator(1).fit(self.rows('train'),[])
        estimate=model.predict('update',[0],0.5)
        self.assertNotEqual(estimate.calibration_state,'supported')
        self.assertIsNone(estimate.error_allowance)
        with self.assertRaises(ValueError):
            model.slopes(['no','update'],[0])
        with self.assertRaises(ValueError):
            model.fit(self.rows('new'),self.rows('held'))

    def test_mismatched_paired_baseline_and_no_update_gain_rejected(self):
        rows=self.rows('train')
        r=rows[1]
        rows[1]=replace(r,no_update_gross=r.no_update_gross+0.1)
        with self.assertRaises(ValueError):
            PairedRecoveryCalibrator(1).fit(rows,self.rows('held'))
        with self.assertRaises(ValueError):
            PairedRecoverySample('no','infer',False,np.array([0.0]),0.5,1.0,1.1,'id')

    def test_constant_local_loss_cannot_support_an_identified_response_slope(self):
        train=[replace(r,loss_state=0.5) for r in self.rows('train')]
        held=[replace(r,loss_state=0.5) for r in self.rows('held')]
        model=PairedRecoveryCalibrator(1).fit(train,held)
        with self.assertRaises(ValueError):
            model.local_coefficients('update',[0])


class PredictorTests(unittest.TestCase):
    def test_late_original_features_and_issued_predictions_are_copied(self):
        model = DelayedRidgeForecaster(1, 1, RidgeConfig(ridge=1.0, forgetting=0.5))
        features = np.array([[2.0]])
        forecast = model.issue(0, features, [True])
        features[:] = 999
        forecast[:] = 999
        model.issue(1, [[1.0]], [True])
        audit = model.observe_label(1, 0.8, 2)
        self.assertEqual(audit["stored_prediction"], [0.0])
        late = model.observe_label(0, 0.4, 4)
        # At clock 4: origin 1 has weight .5^3, origin 0 has .5^4.
        gram = 0.5**3 * 1.0**2 + 0.5**4 * 2.0**2
        rhs = 0.5**3 * 1.0 * 0.8 + 0.5**4 * 2.0 * 0.4
        np.testing.assert_allclose(model.coefficients, [[rhs / (1.0 + gram)]], atol=1e-14)
        self.assertEqual(late["stored_prediction"], [0.0])
        self.assertEqual(late["origin_weight"], 0.5**4)
        self.assertEqual(late["residual"], [-0.4])

    def test_duplicate_early_and_backwards_labels_fail_without_adding_samples(self):
        model = DelayedRidgeForecaster(1, 1)
        model.issue(0, [[1]], [True], target_window=3)
        with self.assertRaises(ValueError):
            model.observe_label(0, 0.5, 2)
        self.assertEqual(model.state_dict()["observed_counts"], [0])
        model.observe_label(0, 0.5, 3)
        with self.assertRaises(ValueError):
            model.observe_label(0, 0.5, 4)
        self.assertEqual(model.state_dict()["observed_counts"], [1])
        with self.assertRaises(ValueError):
            model.issue(2, [[1]], [True])

    def test_missing_branch_does_not_train_on_zero(self):
        model = DelayedRidgeForecaster(2, 1)
        pred = model.issue(0, [[1], [np.nan]], [True, False])
        self.assertTrue(np.isnan(pred[1]))
        model.observe_label(0, 0.7, 1)
        self.assertEqual(model.state_dict()["observed_counts"], [1, 0])
        self.assertEqual(model.coefficients[1, 0], 0.0)

    def test_prefix_only_once_before_live_issue(self):
        model = DelayedRidgeForecaster(1, 1)
        model.fit_prefix(np.ones((2, 1, 1)), [0.2, 0.4])
        with self.assertRaises(ValueError):
            model.fit_prefix(np.ones((2, 1, 1)), [0.2, 0.4])
        model.issue(0, [[1]], [True])
        with self.assertRaises(ValueError):
            model.fit_prefix(np.ones((2, 1, 1)), [0.2, 0.4])

    def test_pending_snapshot_is_not_a_mutable_training_handle(self):
        model = DelayedRidgeForecaster(1, 1)
        model.issue(0, [[2]], [True])
        record = model.pending_snapshot(0)
        record["features"][:] = 999
        model.observe_label(0, 0.5, 1)
        weight = model.config.forgetting
        expected = weight * 2 * 0.5 / (model.config.ridge + weight * 4)
        self.assertAlmostEqual(model.coefficients[0, 0], expected, places=12)


class EvidenceTests(unittest.TestCase):
    def test_persistent_loss_uses_identical_valid_slot_set_and_minimum_support(self):
        reference = [1.0, np.nan, 0.0]
        actual = [0.5, 0.0, np.nan]
        target, audit = persistent_loss(reference, actual, min_slots=1)
        self.assertEqual(target, 0.5)
        self.assertEqual(audit["valid_slots"], 1)
        unavailable, audit = persistent_loss(reference, actual, min_slots=2)
        self.assertIsNone(unavailable)
        self.assertEqual(audit["valid_slots"], 1)
        self.assertFalse(audit["target_available"])

    def test_service_evidence_requires_common_execution_context(self):
        previous = {("train0", "inf1", "version0"): (10, 0.8)}
        current = {("train0", "inf1", "version1"): (10, 0.3)}
        self.assertIsNone(matched_service_change(previous, current))

    def test_empty_bootstrap_evidence_uses_variance_fallback_not_precise_zero(self):
        variance, audit = block_bootstrap_variance([np.nan, np.nan],
                                                   rng=np.random.default_rng(991),
                                                   fallback_variance=0.25)
        self.assertEqual(variance, 0.25)
        self.assertEqual(audit["valid_replicates"], 0)
        self.assertTrue(audit["fallback"])


class FusionTests(unittest.TestCase):
    def test_joint_quality_risk_history_perturbations_respect_checked_bound(self):
        rng=np.random.default_rng(74013)
        for trial in range(25):
            n=2+trial%4
            q=np.exp(rng.normal(0,0.8,n))
            changed_q=q*np.exp(rng.normal(0,0.05,n))
            errors=rng.normal(0,0.2,(40,n))
            matrix=errors.T@errors/len(errors)
            v=rng.normal(0,1,n)
            changed_matrix=matrix+0.002*np.outer(v,v)
            anchor=rng.dirichlet(np.ones(n))
            changed_anchor=.9*anchor+.1*rng.dirichlet(np.ones(n))
            config=dict(cap=.7,tau=.2,lam=.7,inertia=.3)
            first=solve_capped_fusion(q,matrix,anchor,**config)
            second=solve_capped_fusion(changed_q,changed_matrix,changed_anchor,**config)
            self.assertTrue(first.converged and second.converged)
            self.assertLessEqual(max(first.kkt_residual,second.kkt_residual),5e-8)
            bound=sensitivity_bound(q,changed_q,matrix,changed_matrix,
                                    anchor,changed_anchor,**config)
            self.assertLessEqual(np.linalg.norm(first.weights-second.weights),bound+2e-7)

    def issue(self, fusion, window, predictions, mask=None, slopes=None):
        mask = [True] * len(predictions) if mask is None else mask
        slopes = [0.0, 1.0] if slopes is None else slopes
        return fusion.issue(window, np.array(predictions, dtype=float), [0.0],
                            observations(mask), slopes)

    def test_uniform_limiting_case_and_kkt(self):
        fusion = DecisionAwareFusion(3, 1, FusionConfig(lam=0.0, inertia=0.0, max_weight=0.7))
        snapshot = self.issue(fusion, 0, [0.2, 0.4, 0.6])
        np.testing.assert_allclose(snapshot.weights, np.full(3, 1 / 3), atol=1e-8)
        self.assertFalse(snapshot.solver_fallback)
        self.assertLessEqual(snapshot.audit["solver_primal_residual"], 1e-7)
        self.assertLessEqual(snapshot.audit["solver_kkt_residual"], 1e-6)

    def test_full_cross_error_changes_weights_with_identical_diagonal_mse(self):
        sigma, rho, tau, lam, inertia = 0.15, 0.9, 0.001, 1.0, 0.001
        matrix = sigma**2 * np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]])
        uniform = np.full(3, 1 / 3)
        full = solve_capped_fusion(np.ones(3), matrix, uniform, 0.7, tau, lam, inertia)
        diagonal = solve_capped_fusion(np.ones(3), np.diag(np.diag(matrix)),
                                       uniform, 0.7, tau, lam, inertia)
        self.assertTrue(full.converged and diagonal.converged)
        np.testing.assert_allclose(diagonal.weights, uniform, atol=1e-9)
        self.assertAlmostEqual(full.weights[0], full.weights[1], places=9)
        self.assertGreater(full.weights[2], 1 / 3)
        self.assertLess(full.weights @ matrix @ full.weights, uniform @ matrix @ uniform)
        # Independent one-variable derivative of the symmetry-reduced objective.
        third = full.weights[2]
        derivative = (lam * sigma**2 / 2 * ((3 + rho) * third - (1 + rho))
                      + tau * np.log(2 * third / (1 - third))
                      + 1.5 * inertia * (third - 1 / 3))
        self.assertLess(abs(derivative), 1e-8)

    def test_common_quality_scale_cannot_change_the_optimizer(self):
        quality = np.array([1.0, 2.0, 5.0])
        matrix = np.array([[0.1, 0.02, 0.0], [0.02, 0.2, 0.0], [0.0, 0.0, 0.05]])
        uniform = np.full(3, 1 / 3)
        first = solve_capped_fusion(quality, matrix, uniform, 0.7, 0.1, 1.0, 0.05)
        scaled = solve_capped_fusion(1000 * quality, matrix, uniform, 0.7, 0.1, 1.0, 0.05)
        np.testing.assert_allclose(first.weights, scaled.weights, atol=1e-9)

    def test_precision_cap_has_analytic_solution(self):
        fusion = DecisionAwareFusion(3, 1, FusionConfig(lam=0.0, inertia=0.0, max_weight=0.7))
        quality = observations([True] * 3, [0.000001, 0.1, 0.1])
        snapshot = fusion.issue(0, [0.2, 0.4, 0.6], [0], quality, [0, 1])
        np.testing.assert_allclose(snapshot.weights, [0.7, 0.15, 0.15], atol=1e-6)
        self.assertLessEqual(snapshot.audit["solver_kkt_residual"], 1e-6)

    def test_single_source_cap_relaxes_and_missing_is_not_zero_evidence(self):
        fusion = DecisionAwareFusion(3, 1, FusionConfig(max_weight=0.6))
        snapshot = self.issue(fusion, 0, [np.nan, 0.4, np.nan], [False, True, False])
        np.testing.assert_array_equal(snapshot.weights, [0, 1, 0])
        self.assertAlmostEqual(snapshot.coverage, 1 / 3)
        self.assertAlmostEqual(snapshot.forecast, 0.4)

    def test_partial_sources_produce_psd_active_moments(self):
        fusion = DecisionAwareFusion(3, 1)
        self.issue(fusion, 0, [0.2, np.nan, 0.6], [True, False, True])
        fusion.observe_label(0, 0.4, 1)
        snapshot = self.issue(fusion, 1, [0.3, np.nan, 0.6], [True, False, True])
        self.assertEqual(snapshot.weights[1], 0)
        matrices = fusion.get_last_matrices()
        self.assertEqual(list(matrices["active_sources"]), [0, 2])
        for key in ["Mhat", "R", "S"]:
            matrix = matrices[key]
            self.assertEqual(matrix.shape, (2, 2))
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-12)

    def test_missing_fourth_source_does_not_block_three_source_learning(self):
        fusion = DecisionAwareFusion(4, 1)
        self.issue(fusion, 0, [0.35, 0.35, 0.65, np.nan], [True, True, True, False])
        fusion.observe_label(0, 0.5, 1)
        snapshot = self.issue(fusion, 1, [0.35, 0.35, 0.35, np.nan], [True, True, True, False])
        self.assertAlmostEqual(snapshot.coverage, 0.75)
        self.assertGreater(snapshot.audit["support_count"], 0)
        self.assertEqual(list(fusion.get_last_matrices()["active_sources"]), [0, 1, 2])
        archive = fusion.archive_summary()
        self.assertFalse(archive[0]["mask"][3])
        self.assertTrue(np.isnan(archive[0]["residuals"][3]))

    def test_complete_enabled_control_does_not_pad_partial_vectors(self):
        fusion = DecisionAwareFusion(4, 1, FusionConfig(require_complete_enabled_vectors=True))
        self.issue(fusion, 0, [0.35, 0.35, 0.65, np.nan], [True, True, True, False])
        fusion.observe_label(0, 0.5, 1)
        snapshot = self.issue(fusion, 1, [0.35, 0.35, 0.35, np.nan], [True, True, True, False])
        self.assertEqual(snapshot.audit["support_count"], 0)
        np.testing.assert_allclose(snapshot.weights[:3], np.full(3, 1 / 3), atol=1e-8)

    def test_pending_limit_requires_explicit_discard(self):
        fusion = DecisionAwareFusion(3, 1, FusionConfig(max_pending_forecasts=1))
        self.issue(fusion, 0, [0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            self.issue(fusion, 1, [0.2, 0.3, 0.4])
        fusion.discard_pending(0, reason="target will not be observed")
        self.issue(fusion, 1, [0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            fusion.observe_label(0, 0.3, 2)

    def test_missing_variance_uses_logged_conservative_fallback(self):
        fusion = DecisionAwareFusion(3, 1)
        quality = observations([True] * 3, [None, 0.02, 0.02])
        snapshot = fusion.issue(0, [0.2, 0.4, 0.6], [0], quality, [0, 1])
        self.assertEqual(snapshot.audit["variance_fallback_sources"], [0])
        self.assertLess(snapshot.weights[0], snapshot.weights[1])

    def test_late_labels_score_original_copy_and_original_stakes(self):
        fusion = DecisionAwareFusion(3, 1)
        predictions = np.array([0.2, 0.4, 0.6])
        slopes = np.array([0, 1.0])
        first = fusion.issue(0, predictions, [0], observations([True] * 3), slopes)
        saved_weights = first.weights.copy()
        predictions[:] = 0.99
        slopes[:] = 99
        self.issue(fusion, 1, [0.9, 0.9, 0.9])
        fusion.observe_label(1, 0.8, 2)
        fusion.observe_label(0, 0.3, 3)
        record = next(r for r in fusion.archive_summary() if r["forecast_window"] == 0)
        np.testing.assert_allclose(record["residuals"], [-0.1, 0.1, 0.3], atol=1e-14)
        self.assertAlmostEqual(record["chi"], 1.0)
        np.testing.assert_array_equal(first.weights, saved_weights)

    def test_duplicate_and_early_labels_are_rejected(self):
        fusion = DecisionAwareFusion(3, 1)
        self.issue(fusion, 0, [0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            fusion.observe_label(0, 0.3, 0)
        fusion.observe_label(0, 0.3, 1)
        with self.assertRaises(ValueError):
            fusion.observe_label(0, 0.3, 2)
        self.assertEqual(len(fusion.archive_summary()), 1)

    def test_all_missing_abstains_and_marks_fallback(self):
        fusion = DecisionAwareFusion(3, 1)
        snapshot = self.issue(fusion, 0, [np.nan] * 3, [False] * 3)
        self.assertTrue(snapshot.all_missing)
        self.assertEqual(snapshot.coverage, 0)
        self.assertEqual(snapshot.uncertainty, 1)
        np.testing.assert_array_equal(snapshot.weights, [0, 0, 0])
        self.assertTrue(snapshot.forecast is None or not np.isfinite(snapshot.forecast))

    def test_forced_solver_failure_is_explicit_and_feasible(self):
        fusion = DecisionAwareFusion(3, 1, FusionConfig(force_solver_fallback=True))
        snapshot = self.issue(fusion, 0, [0.2, 0.4, 0.6])
        self.assertTrue(snapshot.solver_fallback)
        self.assertAlmostEqual(snapshot.weights.sum(), 1)
        self.assertLessEqual(snapshot.weights.max(), 0.6 + 1e-12)


class CoordinatorTests(unittest.TestCase):
    def select(self, coordinator, snapshot=None, candidates=None, slot=1, **kwargs):
        return coordinator.select(slot=slot, snapshot=snapshot or fixture_snapshot(),
                                  candidates=candidates or [base_candidate(), update_candidate()],
                                  capacity=np.array([1.0]), **kwargs)

    def test_candidate_dependent_uncertainty_changes_admission(self):
        config = CoordinatorConfig(alpha_cost=0.1, omega_intensity=0,
                                   lambda_uncertainty=0.5, cost_reference=1)
        candidates = [base_candidate(), update_candidate(gain=0.11)]
        low = self.select(PersistentRecoveryCoordinator(config), fixture_snapshot(uncertainty=0.1), candidates)
        high = self.select(PersistentRecoveryCoordinator(config), fixture_snapshot(uncertainty=0.9), candidates)
        self.assertEqual(low.candidate_name, "update")
        self.assertEqual(high.candidate_name, "no_update")
        self.assertGreater(low.admission_margins["update"], 0)
        self.assertLess(high.admission_margins["update"], 0)

    def test_unsupported_recovery_does_not_become_a_certificate(self):
        candidate = update_candidate(gain=10, supported=False, support_count=0)
        decision = self.select(PersistentRecoveryCoordinator(), candidates=[base_candidate(), candidate])
        self.assertFalse(decision.is_update)
        self.assertIn("unsupported_recovery", decision.audit["pre_admission_rejections"]["update"])

    def test_conditional_vs_explicit_external_certificate(self):
        candidates = [base_candidate(certificate_valid=True), update_candidate(certificate_valid=True)]
        conditional = self.select(PersistentRecoveryCoordinator(), candidates=candidates)
        certified = self.select(PersistentRecoveryCoordinator(CoordinatorConfig(simultaneous_certificate=True)),
                                candidates=candidates)
        self.assertEqual(conditional.certificate_status, "conditional")
        self.assertEqual(certified.certificate_status, "certified")
        self.assertTrue(json.loads(certified.audit_json())["audit"]["certificate_is_external_bound_assertion"])

    def test_resource_infeasible_update_is_not_executed(self):
        decision = self.select(PersistentRecoveryCoordinator(), candidates=[
            base_candidate(), update_candidate(gain=10, resource_demand=np.array([2.0]))])
        self.assertEqual(decision.candidate_name, "no_update")
        self.assertFalse(decision.audit["resource_feasible"]["update"])

    def test_no_feasible_no_update_reports_service_deferral(self):
        decision = self.select(PersistentRecoveryCoordinator(), candidates=[
            base_candidate(resource_demand=np.array([2.0])), update_candidate()])
        self.assertIsNone(decision.candidate_name)
        self.assertFalse(decision.is_update)
        self.assertEqual(decision.reason, "service_deferral_no_feasible_no_update")

    def test_all_missing_blocks_exploitation_and_probe(self):
        config = CoordinatorConfig(probe_budget=1, probe_name="update", probe_loss_windows=1)
        decision = self.select(PersistentRecoveryCoordinator(config),
                               fixture_snapshot(all_missing=True, coverage=0, uncertainty=1),
                               observed_loss=0.8)
        self.assertEqual(decision.candidate_name, "no_update")
        self.assertFalse(decision.is_probe)
        self.assertEqual(decision.probe_budget_used, 0)

    def test_stale_carried_loss_is_not_a_second_persistent_window(self):
        config = CoordinatorConfig(probe_budget=0.2, probe_name="update", probe_cooldown=20)
        coordinator = PersistentRecoveryCoordinator(config)
        candidates = [base_candidate(), update_candidate(supported=False)]
        first = self.select(coordinator, fixture_snapshot(window=1), candidates,
                            slot=1, observed_loss=0.6, observed_loss_age=0)
        carried = self.select(coordinator, fixture_snapshot(window=2), candidates,
                              slot=2, observed_loss=0.6, observed_loss_age=1)
        self.assertFalse(first.is_probe)
        self.assertFalse(carried.is_probe)
        fresh = self.select(coordinator, fixture_snapshot(window=2), candidates,
                            slot=3, observed_loss=0.6, observed_loss_age=0)
        self.assertTrue(fresh.is_probe)
        self.assertEqual(fresh.certificate_status, "probe_not_certified")
        self.assertAlmostEqual(fresh.probe_budget_used, 0.2)
        repeated = self.select(coordinator, fixture_snapshot(window=2), candidates,
                               slot=3, observed_loss=0.6, observed_loss_age=0)
        self.assertIs(fresh, repeated)
        self.assertEqual(len(coordinator.probe_reservations), 1)
        exhausted = self.select(coordinator, fixture_snapshot(window=3), candidates,
                                slot=30, observed_loss=0.6)
        self.assertFalse(exhausted.is_probe)

    def test_probe_gradient_cap_is_enforced(self):
        coordinator = PersistentRecoveryCoordinator(CoordinatorConfig(
            probe_budget=1, probe_name="update", probe_loss_windows=1, max_probe_gradient_steps=4))
        decision = self.select(coordinator, candidates=[base_candidate(), update_candidate(supported=False)],
                               observed_loss=0.8)
        self.assertFalse(decision.is_probe)

    def test_stale_fusion_snapshot_blocks_update_and_probe(self):
        config = CoordinatorConfig(snapshot_max_age=1, probe_budget=1,
                                   probe_name="update", probe_loss_windows=1)
        coordinator = PersistentRecoveryCoordinator(config)
        decision = self.select(coordinator, fixture_snapshot(window=1), slot=4,
                               observed_loss=0.8, completed_window=4)
        self.assertEqual(decision.candidate_name, "no_update")
        self.assertFalse(decision.is_probe)
        self.assertIn("stale_fusion_snapshot", decision.audit["pre_admission_rejections"]["update"])

    def test_future_snapshot_is_rejected(self):
        with self.assertRaises(ValueError):
            self.select(PersistentRecoveryCoordinator(), fixture_snapshot(window=3),
                        slot=3, completed_window=2)

    def test_same_slot_changed_inputs_and_published_loss_mutation_are_rejected(self):
        coordinator = PersistentRecoveryCoordinator()
        self.select(coordinator, slot=1, observed_loss=0.6)
        with self.assertRaises(ValueError):
            self.select(coordinator, slot=1, observed_loss=0.7)
        with self.assertRaises(ValueError):
            self.select(coordinator, fixture_snapshot(window=1), slot=2, observed_loss=0.7)

    def test_candidate_resource_array_is_copied_and_immutable(self):
        array = np.array([0.3])
        candidate = update_candidate(resource_demand=array)
        array[:] = 999
        self.assertEqual(candidate.resource_demand[0], 0.3)
        with self.assertRaises(ValueError):
            candidate.resource_demand[0] = 1.0


class GradientWorkerTests(unittest.TestCase):
    def test_numpy_gradient_updates_private_copy(self):
        theta = np.zeros(2)
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        updated = logistic_update(theta, x, np.array([1.0, 1.0]), 1, 0.2)
        np.testing.assert_array_equal(theta, [0.0, 0.0])
        np.testing.assert_allclose(updated, [0.05, 0.05], atol=1e-14)

    def test_actual_updates_deploy_only_after_delay_and_use_completed_labels(self):
        result, records = gradient_recoverability_demo()
        profile = result["profile"]
        for name, arm in result["arms"].items():
            self.assertLessEqual(arm["actual_gradient_steps"],
                                 profile["maximum_workers"] * profile["gradient_steps_per_worker"])
            for launch in arm["launches"]:
                self.assertLessEqual(launch["last_training_label_origin"],
                                     launch["slot"] - profile["label_delay_slots"])
            for deployment in arm["deployments"]:
                self.assertEqual(deployment["deployment_slot"] - deployment["launch_slot"],
                                 profile["deployment_delay_slots"])
            history = [r for r in records if r["arm"] == name]
            for before, after in zip(history, history[1:]):
                if not after["deployed_now"]:
                    self.assertEqual(before["theta_0"], after["theta_0"])
                    self.assertEqual(before["theta_1"], after["theta_1"])
                    self.assertEqual(before["version"], after["version"])
        self.assertLess(result["arms"]["frozen"]["tail_256_accuracy"], 0.05)
        self.assertGreater(result["arms"]["periodic_worker"]["tail_256_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
