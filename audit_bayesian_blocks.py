"""Synthetic algebra audit; this is not a real-sensor performance experiment."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'outputs' / 'fusion_repro'))
from block_bayesian_fusion import block_bayesian_moments, draw_block_posteriors
import real_air_invariant_fusion as air

rng = np.random.default_rng(114007)
sources, features, classes = 4, 5, 6
deployed = rng.normal(size=(features, classes))
candidate = rng.normal(size=(features, classes))


def block(rows, origin, mask, arrival):
    return dict(x=rng.normal(size=(rows, features)),
                p=rng.dirichlet(np.ones(classes), size=(sources, rows)),
                y=rng.integers(0, classes, size=rows),
                price=rng.choice([1., 3.], size=rows),
                context=rng.normal(size=2), origin=origin,
                mask=np.array(mask, dtype=bool), arrival=arrival)


prior = block(17, -1, [True] * 4, -1)
records = [block(5 + i, i + 2, [True] * 4, i + 4) for i in range(5)]
# One retained window cannot cover the selected source subset. Earlier
# records are outside the finite archive, as in the executed estimator.
records[3]['mask'][2] = False
cfg = dict(air.CFG, prior_mass=2., archive=3, beta=.97, context_bandwidth=2.)
active_mask = np.array([0, 2, 3])
context = np.array([.3, -.2])
k = 20
chunks = [(cfg['prior_mass'], prior)]
for r in records[-cfg['archive']:]:
    if not np.all(r['mask'][active_mask]):
        continue
    kernel = np.exp(-np.sum((context - r['context']) ** 2) /
                    cfg['context_bandwidth'] ** 2)
    chunks.append((cfg['beta'] ** (k - r['origin']) * kernel, r))

concentrations, means, seconds = [], [], []
for mass, r in chunks:
    error = r['p'][active_mask] - np.eye(classes)[r['y']][None, :, :]
    before = (r['x'] @ deployed).argmax(1)
    after = (r['x'] @ candidate).argmax(1)
    contrast = r['price'][:, None] * (np.eye(classes)[after] - np.eye(classes)[before])
    z = np.einsum('sic,ic->si', error, contrast)
    concentrations.append(mass)
    means.append(z.mean(1))
    seconds.append(z @ z.T / len(r['y']))

moments = block_bayesian_moments(concentrations, means, seconds)
legacy_covariance, _, legacy_mean = air.paired_moments(
    records, prior, context, k, active_mask, deployed, candidate,
    cfg, 'decision_full')

minimum_eigenvalues = {
    key: float(np.linalg.eigvalsh(moments[key]).min())
    for key in ('predictive_covariance', 'within_block_covariance',
                'between_block_covariance', 'posterior_mean_covariance',
                'expected_conditional_covariance')
}
errors = dict(
    paired_mean_max_abs=float(np.max(np.abs(moments['predictive_mean'] - legacy_mean))),
    paired_covariance_max_abs=float(np.max(np.abs(moments['predictive_covariance'] - legacy_covariance))),
    within_plus_between_max_abs=float(np.max(np.abs(
        moments['predictive_covariance'] - moments['within_block_covariance']
        - moments['between_block_covariance']))),
    total_covariance_max_abs=float(np.max(np.abs(
        moments['predictive_covariance'] - moments['posterior_mean_covariance']
        - moments['expected_conditional_covariance']))),
    incorrect_predictive_over_concentration_error=float(np.max(np.abs(
        moments['posterior_mean_covariance'] -
        moments['predictive_covariance'] / (moments['total_concentration'] + 1.))))
)
draws = draw_block_posteriors(concentrations, means, seconds,
                              draws=200000, seed=114008)
sample_parameter = np.cov(draws['mean'], rowvar=False, ddof=0)
mc_errors = dict(
    mean_max_abs=float(np.max(np.abs(draws['mean'].mean(0) - moments['predictive_mean']))),
    raw_second_max_abs=float(np.max(np.abs(draws['raw_second'].mean(0) - moments['predictive_raw_second']))),
    posterior_mean_covariance_max_abs=float(np.max(np.abs(sample_parameter - moments['posterior_mean_covariance']))),
    expected_conditional_covariance_max_abs=float(np.max(np.abs(draws['covariance'].mean(0) - moments['expected_conditional_covariance'])))
)
single = block_bayesian_moments([2.], [[.4]], [[[.5]]])
rejected = {}
for name, args in (
    ('zero_concentration', ([0.], [[0.]], [[[1.]]])),
    ('indefinite_block_covariance', ([1.], [[2.]], [[[1.]]])),
):
    try:
        block_bayesian_moments(*args)
    except ValueError:
        rejected[name] = True
    else:
        rejected[name] = False

assert max(errors[key] for key in errors if key.endswith('max_abs')) < 1e-12
assert min(minimum_eigenvalues.values()) >= -1e-12
assert errors['incorrect_predictive_over_concentration_error'] > 1e-3
assert max(mc_errors.values()) < .01
assert np.max(np.abs(single['posterior_mean_covariance'])) == 0.
assert all(rejected.values())

def jsonify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    return value

result = dict(
    status='passed', scope='Synthetic algebra and API audit, not real-sensor performance',
    construction='Dirichlet source probabilities, random features and six-class labels; current candidate/deployed remapping; finite archive and joint missing-source exclusion',
    seeds=dict(fixture=114007, dirichlet_audit=114008),
    posterior_draws=200000,
    archived_windows=len(records), retained_archive_limit=cfg['archive'],
    completed_eligible_blocks_including_prior=len(chunks), active_sources=active_mask.tolist(),
    concentrations=concentrations, analytic_moments=moments,
    minimum_eigenvalues=minimum_eigenvalues,
    deterministic_errors=errors, monte_carlo_errors=mc_errors,
    single_block_single_source=dict(posterior_mean_covariance=single['posterior_mean_covariance']),
    invalid_inputs_rejected=rejected,
    interpretation='Posterior predictive mean/covariance equal existing paired_moments to roundoff. Covariance of posterior mean uses between-block means, not all per-request predictive variation. Monte Carlo is audit-only and does not change the gate.'
)
path = ROOT / 'Bayesian_block_audit.json'
path.write_text(json.dumps(jsonify(result), indent=2))
print(json.dumps(dict(path=str(path), status=result['status'],
                     deterministic_errors=errors, monte_carlo_errors=mc_errors,
                     minimum_eigenvalues=minimum_eigenvalues), indent=2))
