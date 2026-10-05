"""Finite-support proof audit for paired consequential fusion risk.

This constructed example is not an update-return experiment. Both fixed
models predict an incorrect class when the true label is class three.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'outputs'))
from da_rf_fusion import solve_capped_fusion

u = np.array([1., -1., 0.]) / np.sqrt(2.)
v = np.array([1., 1., -2.]) / np.sqrt(6.)
uniform = np.ones(3) / 3.
direction = np.array([-1., -1., 2.])
T = np.eye(3) - np.ones((3, 3)) / 3.
bits = np.array(list(itertools.product((-1., 1.), repeat=6)))
sigma = .05
results = []


def normalize(M):
    tangent_trace = float(np.trace(T @ M @ T))
    return (M / tangent_trace if tangent_trace >= 1e-6 else M), tangent_trace


for rho in (.15, .68, .95):
    a = np.stack((bits[:, 0], rho * bits[:, 0] + np.sqrt(1 - rho ** 2) * bits[:, 1], bits[:, 2]), axis=1)
    c = np.stack((bits[:, 3], -rho * bits[:, 3] + np.sqrt(1 - rho ** 2) * bits[:, 4], bits[:, 5]), axis=1)
    p = 1 / 3 + sigma * (np.einsum('is,c->isc', a, u) + np.einsum('is,c->isc', c, v))
    error = p - np.array([0., 0., 1.])[None, None, :]
    centered_error = error - error.mean(0)[None, :, :]
    R = np.einsum('isc,itc->st', centered_error, centered_error) / len(bits)
    z = error @ np.array([1., -1., 0.])
    centered_z = z - z.mean(0)
    consequential = centered_z.T @ centered_z / len(bits)
    A = np.eye(3)
    A[0, 1] = A[1, 0] = rho
    R_theory = 2 * sigma ** 2 * np.eye(3)
    consequential_theory = 2 * sigma ** 2 * A
    empirical_directional = float(direction @ consequential @ uniform)
    expected_directional = -4 * sigma ** 2 * rho / 3
    M, tangent_trace = normalize(consequential)
    context_M, _ = normalize(R)
    diagonal_M, _ = normalize(np.diag(np.diag(consequential)))
    shape_errors = dict(
        source_class_covariance=float(np.max(np.abs(R - R_theory))),
        action_covariance=float(np.max(np.abs(consequential - consequential_theory))),
        directional_identity=abs(empirical_directional - expected_directional),
        probability_sum=float(np.max(np.abs(p.sum(2) - 1.))),
        expected_action_bias=float(np.max(np.abs(z.mean(0))))
    )
    assert max(shape_errors.values()) < 1e-12
    assert p.min() > 0 and p.max() < 1
    assert empirical_directional < 0
    for tau, lam, cap in ((.003, 1., .8), (.05, .25, .5), (.5, 1., .34)):
        full = solve_capped_fusion(np.ones(3), M, uniform, cap, tau, lam, 0., tolerance=1e-10)
        context = solve_capped_fusion(np.ones(3), context_M, uniform, cap, tau, lam, 0., tolerance=1e-10)
        diagonal = solve_capped_fusion(np.ones(3), diagonal_M, uniform, cap, tau, lam, 0., tolerance=1e-10)
        risk_uniform = float(uniform @ consequential @ uniform)
        risk_full = float(full.weights @ consequential @ full.weights)
        objective_uniform = float(lam / 2 * uniform @ M @ uniform)
        strict_risk_reduction = risk_uniform - risk_full
        strict_objective_reduction = objective_uniform - full.objective
        assert full.converged and context.converged and diagonal.converged
        assert max(np.max(np.abs(context.weights - uniform)), np.max(np.abs(diagonal.weights - uniform))) < 1e-9
        assert strict_risk_reduction > 0 and strict_objective_reduction > 0
        results.append(dict(rho=rho, sigma=sigma, tau=tau, lam=lam, cap=cap,
                            finite_support_size=len(bits), minimum_probability=float(p.min()),
                            maximum_probability=float(p.max()), shape_errors=shape_errors,
                            raw_directional_value=empirical_directional,
                            expected_raw_directional_value=expected_directional,
                            tangent_trace=tangent_trace,
                            expected_tangent_trace=4 * sigma ** 2 * (1 - rho / 3),
                            full_weights=full.weights.tolist(), context_weights=context.weights.tolist(),
                            diagonal_weights=diagonal.weights.tolist(),
                            centered_action_risk_uniform=risk_uniform,
                            centered_action_risk_full=risk_full,
                            strict_raw_risk_reduction=strict_risk_reduction,
                            relative_raw_risk_reduction=strict_risk_reduction / risk_uniform,
                            strict_regularized_objective_reduction=strict_objective_reduction,
                            full_kkt_residual=full.kkt_residual))

summary = dict(
    status='passed', scope='Constructed finite-support fusion prediction-risk proof audit; not real-sensor performance or net-return evidence',
    bounded_support='Independent Rademacher X1,X2,X3,Y1,Y2,Y3; enumerate all 64 equally probable outcomes',
    theoretical_conditions=dict(sources=3, classes=3, rho='0<rho<1',
                                sigma='0<sigma<1/12 is sufficient, not necessary',
                                quality_prior='uniform', source_cap='cap>1/3',
                                regularization='tau>0, lambda>0'),
    proof_summary=[
        'Each source forecast remains on the probability simplex because class contrasts u,v sum to zero and bounded perturbations are smaller than 1/3.',
        'Centered unprojected source-class covariance is 2*sigma^2 I, so contextual and action-diagonal influence are uniform.',
        'Centered consequential covariance is 2*sigma^2 A. Direction (-1,-1,2) has derivative -4*sigma^2*rho/3 at uniform before positive scaling.',
        'cap>1/3 permits a positive step in this direction; the uniform KL term has zero tangent derivative. The unique full optimum strictly decreases its regularized objective.',
        'Because KL is minimized at uniform and is nonnegative, the full optimum also strictly decreases raw centered consequential prediction risk.',
        'Common-error-invariant scale is positive; its tangent trace is 4*sigma^2*(1-rho/3). A below-floor raw fallback preserves the same strict direction.'
    ],
    limitations=[
        'Truth is fixed to class three while candidate and deployed predict classes one and two, so both models have zero true correctness gain on every constructed request.',
        'Consequently this construction proves neither a useful deployment action nor positive cost-inclusive return.',
        'It is a mechanism witness for centered action-error risk, not a statistical or empirical generalization guarantee.'
    ],
    checks=results
)
path = ROOT / 'projection_gain_audit.json'
path.write_text(json.dumps(summary, indent=2))
print(json.dumps(dict(path=str(path), status=summary['status'], number_of_checks=len(results),
                     max_shape_error=max(max(r['shape_errors'].values()) for r in results),
                     minimum_strict_raw_risk_reduction=min(r['strict_raw_risk_reduction'] for r in results),
                     minimum_strict_objective_reduction=min(r['strict_regularized_objective_reduction'] for r in results),
                     maximum_solver_kkt=max(r['full_kkt_residual'] for r in results)), indent=2))
