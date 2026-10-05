"""Independent checks of the line posterior and conditional information example.

No physical data are read. Outputs are mechanism checks, not replay performance.
Run: python work/fusion_conditional_20261004/theory/verify_conditional_theory.py
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import math
import importlib.util
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.dont_write_bytecode = True
N = 128
FEE = 5
A = 3 / 32
Q = 1.2815515655
BANDWIDTH = 0.25
NORM_FLOOR = 0.01
ARCHIVE_ROWS = 32
BETA = 0.97
HORIZON = 4
SUPPORTED_DISAGREEMENT = 126 / 128
PATTERN_MASS = 1.5
TOTAL_MASS = 4 * PATTERN_MASS
PARITY = np.array([[1, 1, 1], [1, -1, -1],
                   [-1, 1, -1], [-1, -1, 1]], dtype=float) * A


def logsumexp(x, axis=None):
    mx = np.max(x, axis=axis, keepdims=True)
    result = mx + np.log(np.sum(np.exp(x - mx), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else float(result.item())


def make_law(sign):
    # 32 complete chronology-weighted rows, eight copies of each parity pattern.
    # Each pattern has effective eligible mass1.5 under the actual age/support kernel.
    r = (2 * 0.01 + TOTAL_MASS * A * A) / (2 + TOTAL_MASS)
    p = 1 / (100 + TOTAL_MASS / r)
    smoothing = BANDWIDTH ** 2 * 0.01
    means = np.concatenate([np.zeros((1, 3)), sign * PARITY])
    covs = np.stack([np.eye(3) * (r + p + smoothing)]
                    + [np.eye(3) * (p + smoothing)] * 4)
    alpha = np.array([2] + [PATTERN_MASS] * 4, dtype=float) / (2 + TOTAL_MASS)
    center = alpha @ means
    covariance = sum(alpha[j] * (covs[j] + np.outer(means[j] - center,
                                                   means[j] - center))
                     for j in range(5))
    return dict(mean=means, cov=covs, alpha=alpha, center=center,
                covariance=covariance, R=r, P=p)


def chronology_check():
    current_origin = 240
    assert current_origin % 13 >= 2
    potential_ages = HORIZON * np.arange(1, 49)
    # Only origins outside the actual 13-window/2-window forecast dropout have
    # full masks. Other origins have zero current contrast support and vanish.
    ages = np.array([age for age in potential_ages if (current_origin - age) % 13 >= 2][:ARCHIVE_ROWS])
    history = BETA ** ages
    group_mass_before_context = np.zeros(4)
    counts = np.zeros(4, dtype=int)
    assignment = []
    for weight in history:
        pattern = min((index for index in range(4) if counts[index] < 8),
                      key=lambda index: group_mass_before_context[index])
        assignment.append(pattern)
        counts[pattern] += 1
        group_mass_before_context[pattern] += weight
    assignment = np.array(assignment)
    attenuation = PATTERN_MASS / (SUPPORTED_DISAGREEMENT * group_mass_before_context)
    assert np.all((attenuation > 0) & (attenuation <= 1))
    context_radius = np.sqrt(-9 * np.log(attenuation))  # sigma_x=3
    kernel = history * SUPPORTED_DISAGREEMENT * attenuation[assignment]
    pattern_masses = np.array([kernel[assignment == index].sum() for index in range(4)])
    assert np.max(np.abs(pattern_masses - PATTERN_MASS)) < 1e-15
    assert kernel.min() > 1e-12
    return dict(archive_rows=ARCHIVE_ROWS, current_origin=current_origin,
                ages=ages.tolist(), historical_origins=(current_origin - ages).tolist(),
                pattern_assignment=assignment.tolist(), pattern_counts=counts.tolist(), beta=BETA,
                sigma_x=3., disagreement_support=SUPPORTED_DISAGREEMENT,
                unattenuated_pattern_masses=group_mass_before_context.tolist(),
                context_attenuation=attenuation.tolist(), context_radius=context_radius.tolist(),
                kernel_weights=kernel.tolist(), eligible_pattern_masses=pattern_masses.tolist(),
                eligible_total_mass=float(kernel.sum()),
                current_label_access=False,
                construction='historical context for pattern j equals context_radius[j]*e1; current context is zero; masks complete; equal source quality')


def execution_check(records, chronology):
    # Import only immutable mathematical/service helpers; no physical cache is read.
    path = ROOT.parent / 'run_conditional.py'
    spec = importlib.util.spec_from_file_location('theory_actual_conditional', path)
    actual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actual)
    engine, _, guard, _, cfg0, _, service, fork = actual.parent.binding()
    h = np.full(3, FEE / N)
    pre = dict(m=3, q=np.ones(3))
    cfg = dict(cfg0, prior_mass=2., prior_variance=.01, slice_bandwidth=BANDWIDTH,
               norm_floor=NORM_FLOOR, cap=.8)
    ids = np.arange(3)
    archives = []
    model = np.eye(2)
    reference_model = np.array([[0., 1.], [0., 1.]])
    # Every historical block has 126/128 current-action disagreement. One
    # agreement request is consumed by common probe work, leaving an even
    # number of disagreeing served positions and permitting exactly g=0.
    for age, weight, pattern in zip(chronology['ages'], chronology['kernel_weights'],
                                     chronology['pattern_assignment']):
        origin = chronology['current_origin'] - age
        x = np.tile([1., 0.], (N, 1))
        x[[0, 32]] = [0., 1.]
        common = 3 if origin % 8 == 0 else 1
        disagreement = (x @ model).argmax(1) != (x @ reference_model).argmax(1)
        eligible = np.flatnonzero(disagreement & (np.arange(N) >= common))
        assert len(eligible) % 2 == 0
        y = np.ones(N, dtype=int)
        y[eligible[:len(eligible) // 2]] = 0
        events = [dict(x=x[k*32:(k+1)*32], y=y[k*32:(k+1)*32],
                       candidate=model, reference=reference_model,
                       probe=k == 0, refresh=k == 0 and origin % 8 == 0)
                  for k in range(4)]
        row = dict(k=0, gain=0., posterior_or_block_sd=1., q_issued=Q)
        rebuilt = fork(events, row, cfg)
        assert rebuilt['gross'] == 0.
        z = PARITY[pattern]
        probability = np.repeat(np.array([[(1 + 32 * value / 31) / 2,
                                            (1 - 32 * value / 31) / 2] for value in z])[:, None, :], 32, axis=1)
        contrast = np.eye(2)[(x[:32] @ model).argmax(1)] - np.eye(2)[(x[:32] @ reference_model).argmax(1)]
        projected = np.einsum('src,rc->sr', probability, contrast).mean(axis=1)
        assert np.max(np.abs(projected - z)) < 1e-15
        assert 0 <= probability.min() <= probability.max() <= 1
        assert float(disagreement.mean()) == SUPPORTED_DISAGREEMENT
        archives.append(dict(ids=ids, residual=z, weight=weight))
    results = {}
    max_implementation_difference = 0.
    for name, sign, gross in [('plus', 1, -7), ('minus', -1, 17)]:
        law = make_law(sign)
        observations = [dict(block, residual=sign * block['residual']) for block in archives]
        atoms = actual.parent.atom_state(observations, np.eye(3) * law['R'], np.zeros(3), ids, cfg)
        mm = dict(paired_atoms=atoms, mu=np.zeros(3), P=np.eye(3) * law['P'])
        alpha, locations, covariances, _, _, _ = actual.law(mm, pre, ids, cfg)
        posterior = actual.joint_posterior(h, alpha, locations, covariances)
        expected = records[name]['paired']
        max_implementation_difference = max(max_implementation_difference,
                                             abs(posterior['mean'] - expected['mean']),
                                             abs(math.sqrt(posterior['variance']) - expected['sd']))
        mm.update(conditional={'conditional_joint': posterior}, eligible=[dict(maturity=0)],
                  current_source_rate=h.tolist(), current_disagreement=[0., 0., 0.],
                  line_gls_untruncated_mean=FEE/N)
        decision = dict(temporal=mm, ids=ids, N=N, truegross=gross)
        output = actual.conditional_forecast(decision, pre, cfg, 'conditional_joint', Q)
        F, S = output[2:4]
        poisoned = actual.conditional_forecast(dict(decision, truegross=1e50), pre, cfg, 'conditional_joint', Q)
        assert output[2:4] == poisoned[2:4]
        action = F - Q * S - FEE > 0
        assert action == (sign == -1)
        # At the current origin all 125 positions left after common work
        # disagree. The first two candidate-correct positions are dropped only
        # on admission, so gross−extra−3 equals the displayed complete D.
        labels = np.ones(N, dtype=int)
        positives = (125 + gross) // 2
        labels[3:3 + positives] = 0
        events = [dict(x=np.tile([1., 0.], (32, 1)), y=labels[k*32:(k+1)*32],
                       candidate=model, reference=reference_model,
                       probe=k == 0, refresh=k == 0) for k in range(4)]
        row = dict(k=0, action=bool(action), maturity=4, local_net=gross-5.,
                   gain=F, posterior_or_block_sd=S, q_issued=Q)
        rebuilt = fork(events, row, cfg)
        assert rebuilt['gross'] == gross and rebuilt['extra'] == 2.
        assert rebuilt['actual'] == gross - 5.
        baseline = service(events, pre, [], cfg)
        forced = service(events, pre, [dict(row, action=True)], cfg)
        assert forced['net'] - baseline['net'] == gross - 5.
        replay = guard.replay([row], len(events), 130., 'gross_loss', {0: 130.})
        served = service(events, pre, replay['rows'], cfg)
        prefix = guard.reconstruct_prefix_increment(events, replay['rows'], cfg)
        increment = served['net'] - baseline['net']
        assert increment == prefix['final'] == replay['final_settled_increment']
        assert increment == (12. if action else 0.)
        assert all(item['spent_loss'] + item['reserved'] <= 130. for item in replay['ledger'])
        assert prefix['minimum'] >= -130.
        results[name] = dict(gross=gross, lost_candidate_correctness=rebuilt['extra'],
                             counterfactual_complete_return=rebuilt['actual'],
                             issued_F=F, issued_S=S, issued_score=F-Q*S-5,
                             admitted=bool(action), guarded_service_increment=increment,
                             prefix_minimum=prefix['minimum'], fixed_reserve=130., budget=130.,
                             future_truth_poison_check=True, complete_identity_error=rebuilt['identity_error'])
    assert max_implementation_difference < 1e-12
    return dict(passed=True, historical_complete_fork_checks=len(archives),
                actual_implementation_vs_independent_posterior_error=max_implementation_difference,
                current_forks=results, no_physical_cache_read=True)


def logpdf(z, law, product=False, gaussian=False):
    z = np.atleast_2d(z)
    if gaussian:
        cov = law['covariance']
        residual = z - law['center']
        inv = np.linalg.inv(cov)
        return -0.5 * (3 * math.log(2 * math.pi) + np.linalg.slogdet(cov)[1]
                       + np.einsum('qs,st,qt->q', residual, inv, residual))
    if product:
        # Exact one-dimensional marginals of the same raw mixture law.
        diag = np.diagonal(law['cov'], axis1=1, axis2=2)
        residual = z[:, None, :] - law['mean'][None, :, :]
        components = (np.log(law['alpha'])[None, :, None]
                      - 0.5 * (math.log(2 * math.pi)
                               + np.log(diag)[None, :, :]
                               + residual ** 2 / diag[None, :, :]))
        return logsumexp(components, axis=1).sum(axis=1)
    inv = np.linalg.inv(law['cov'])
    residual = z[:, None, :] - law['mean'][None, :, :]
    components = (np.log(law['alpha'])[None, :]
                  - 0.5 * (3 * math.log(2 * math.pi)
                           + np.linalg.slogdet(law['cov'])[1][None, :]
                           + np.einsum('qjs,jst,qjt->qj', residual, inv, residual)))
    return logsumexp(components, axis=1)


def quadrature(h, law, nodes, **kwargs):
    u, weights = np.polynomial.legendre.leggauss(nodes)
    log_density = logpdf(h[None, :] - u[:, None], law, **kwargs)
    mass = weights * np.exp(log_density - log_density.max())
    mass /= mass.sum()
    mean = float(mass @ u)
    variance = float(mass @ ((u - mean) ** 2))
    issued_scale = N * math.sqrt(variance + NORM_FLOOR ** 2)
    return dict(mean=mean, sd=math.sqrt(variance),
                F=N * mean, S=issued_scale,
                score=N * mean - Q * issued_scale - FEE)


def analytic_paired(h, law):
    one = np.ones(3)
    inv = np.linalg.inv(law['cov'])
    residual = h - law['mean']
    precision = np.einsum('s,jst,t->j', one, inv, one)
    r = np.einsum('s,jst,jt->j', one, inv, residual)
    mu = r / precision
    sd = np.sqrt(1 / precision)
    perpendicular = np.einsum('js,jst,jt->j', residual, inv, residual) - r * r / precision
    lo = (-1 - mu) / sd
    hi = (1 - mu) / sd
    normal = lambda x: math.exp(-0.5 * float(x) ** 2) / math.sqrt(2 * math.pi)
    cdf = lambda x: 0.5 * (1 + math.erf(float(x) / math.sqrt(2)))
    phi_lo = np.array([normal(x) for x in lo])
    phi_hi = np.array([normal(x) for x in hi])
    trunc_mass = np.array([cdf(hh) - cdf(ll) for ll, hh in zip(lo, hi)])
    correction = (phi_lo - phi_hi) / trunc_mass
    component_mean = mu + sd * correction
    component_variance = sd * sd * (
        1 + (lo * phi_lo - hi * phi_hi) / trunc_mass - correction * correction)
    log_weights = (np.log(law['alpha'])
                   - 0.5 * np.linalg.slogdet(law['cov'])[1]
                   - 0.5 * perpendicular - 0.5 * np.log(precision)
                   + np.log(trunc_mass))
    rho = np.exp(log_weights - logsumexp(log_weights))
    mean = float(rho @ component_mean)
    variance = float(rho @ (component_variance + (component_mean - mean) ** 2))
    return dict(mean=mean, sd=math.sqrt(variance), component_weights=rho.tolist())


def exact_value_enumeration():
    # Uniform independent true target prior: integer gross g in [-64,64].
    # H=U1+Z, so every future target and current error are jointly realizable.
    patterns = (PARITY * N).round().astype(int)
    rows = []
    probability = Fraction(1, 2 * 4 * 129)
    for sign in (1, -1):
        for z in sign * patterns:
            for gross in range(-64, 65):
                forecast = tuple((gross + z).tolist())
                rows.append((sign, forecast, gross - FEE))
    reduced = defaultdict(list)
    rich = defaultdict(list)
    for sign, forecast, paid in rows:
        reduced[forecast].append(paid)
        rich[(sign, forecast)].append(paid)
    value = lambda groups: sum((probability * max(sum(returns), 0)
                               for returns in groups.values()), Fraction(0))
    v_reduced, v_rich = value(reduced), value(rich)
    gap = v_rich - v_reduced
    # Exact conditional Jensen gap using unnormalized probability masses.
    p_and_n = defaultdict(lambda: [Fraction(0), Fraction(0)])
    for (sign, forecast), returns in rich.items():
        mean = Fraction(sum(returns), len(returns))
        mass = probability * len(returns)
        p_and_n[forecast][0] += mass * max(mean, 0)
        p_and_n[forecast][1] += mass * max(-mean, 0)
    formula_gap = sum((min(p, n) for p, n in p_and_n.values()), Fraction(0))
    assert gap == formula_gap == Fraction(6, 43)
    shared = reduced[(FEE, FEE, FEE)]
    assert sorted(shared) == [-12, 12]
    assert max(max(abs(x) for x in h) for _, h, _ in rows) <= N
    return dict(number_of_joint_states=len(rows), reduced_forecast_states=len(reduced),
                retained_value=float(v_rich), reduced_value=float(v_reduced),
                exact_gap=str(gap), expected_gap=float(gap),
                same_forecast_complete_returns=sorted(shared),
                same_forecast_conditional_gap=6.0,
                target_prior='independent discrete uniform g=-64,...,64')


def check():
    plus, minus = make_law(1), make_law(-1)
    h = np.full(3, FEE / N)
    zero_mean = max(float(np.max(np.abs(x['center']))) for x in (plus, minus))
    covariance_error = float(np.max(np.abs(plus['covariance'] - minus['covariance'])))
    # Both exact source marginals and every pair law are unchanged by parity reversal.
    pair_marginals_equal = True
    for source in range(3):
        assert sorted(PARITY[:, source].tolist()) == sorted((-PARITY[:, source]).tolist())
    for s in range(3):
        for t in range(s + 1, 3):
            lhs = sorted(map(tuple, PARITY[:, [s, t]].tolist()))
            rhs = sorted(map(tuple, (-PARITY[:, [s, t]]).tolist()))
            pair_marginals_equal &= lhs == rhs
    assert pair_marginals_equal and zero_mean < 1e-15 and covariance_error < 1e-15
    records = {}
    max_quad_error = 0.0
    max_analytic_error = 0.0
    for name, law in [('plus', plus), ('minus', minus)]:
        records[name] = {}
        for kind in ('paired', 'product_marginal', 'full_gaussian'):
            kwargs = dict(product=kind == 'product_marginal', gaussian=kind == 'full_gaussian')
            result = quadrature(h, law, 1024, **kwargs)
            lower = quadrature(h, law, 512, **kwargs)
            max_quad_error = max(max_quad_error, *(abs(result[k] - lower[k])
                                                 for k in ('mean', 'sd', 'score')))
            records[name][kind] = result
        analytic = analytic_paired(h, law)
        max_analytic_error = max(max_analytic_error,
                                 abs(analytic['mean'] - records[name]['paired']['mean']),
                                 abs(analytic['sd'] - records[name]['paired']['sd']))
        records[name]['closed_form'] = analytic
    assert records['plus']['paired']['score'] < 0 < records['minus']['paired']['score']
    for kind in ('product_marginal', 'full_gaussian'):
        assert abs(records['plus'][kind]['score'] - records['minus'][kind]['score']) < 1e-11
        assert records['plus'][kind]['score'] < 0
    assert max_quad_error < 1e-9 and max_analytic_error < 1e-11
    # Arbitrary simplex source weighting must produce exactly the same target.
    targets = np.linspace(-1, 1, 2001)
    errors = h[None, :] - targets[:, None]
    simplex_error = 0.0
    for w in (np.ones(3) / 3, np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.2, 0.7])):
        simplex_error = max(simplex_error, float(np.max(np.abs(h @ w - errors @ w - targets))))
    assert simplex_error < 1e-15
    chronology = chronology_check()
    return dict(scope='constructed mechanism only; no physical streams read',
                parameters=dict(N=N, fee=FEE, a=A, q=Q, bandwidth=BANDWIDTH,
                                norm_floor=NORM_FLOOR,
                                archive_rows=ARCHIVE_ROWS, eligible_history_mass=TOTAL_MASS,
                                prior_mass=2, R=plus['R'], P=plus['P']),
                chronology=chronology,
                moment_matched_covariance=plus['covariance'].tolist(),
                pair_marginals_equal=pair_marginals_equal,
                mean_equality_error=zero_mean, covariance_equality_error=covariance_error,
                posterior=records,
                quadrature_512_vs_1024_max_error=max_quad_error,
                analytic_vs_quadrature_max_error=max_analytic_error,
                simplex_invariance_error=simplex_error,
                true_complete_return=dict(plus=-12, minus=12),
                execution=execution_check(records, chronology),
                exact_value_of_information=exact_value_enumeration())


if __name__ == '__main__':
    result = check()
    destination = ROOT / 'conditional_theory_report.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
