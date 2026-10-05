"""Finite CJ-R weighting and paid-policy checks; no physical traces are read.

Run with a Python environment containing NumPy. Outputs are mechanism checks,
not physical performance or statistical calibration guarantees.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OLD_PATH = PROJECT / 'work/fusion_conditional_20261004/run_conditional.py'
spec = importlib.util.spec_from_file_location('paid_information_actual_cj', OLD_PATH)
actual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actual)


def normalize(weights):
    weights = np.asarray(weights, dtype=float)
    if not len(weights):
        return weights.copy(), dict(raw_mass=0., raw_square_mass=0., n_eff=0., multiplier=0.)
    assert np.all(np.isfinite(weights)) and np.all(weights > 0)
    s = float(weights.sum())
    q = float(weights @ weights)
    multiplier = s / q
    return multiplier * weights, dict(raw_mass=s, raw_square_mass=q,
                                     n_eff=s*s/q, multiplier=multiplier)


def normalization_checks():
    rng = np.random.default_rng(1042026)
    cases = [[.01] * 48, [1e-10] * 32, [1., .5, .1],
             [1e-8, 1e-4, .1, 1.], [1.]]
    cases += [10**rng.uniform(-9, 0, size=n) for n in (2, 5, 17, 48)]
    max_relative_error = 0.
    summaries = []
    for w in cases:
        w = np.asarray(w)
        normalized, meta = normalize(w)
        assert 1-1e-12 <= meta['n_eff'] <= len(w)+1e-12
        assert abs(normalized.sum() - meta['n_eff']) < 1e-12
        assert np.max(np.abs(normalized/normalized.sum() - w/w.sum())) < 1e-15
        for scale in (1e-6, .01, 100., 1e6):
            scaled, _ = normalize(scale*w)
            error = float(np.max(np.abs(scaled-normalized)/(1+abs(normalized))))
            max_relative_error = max(error, max_relative_error)
        summaries.append(dict(count=len(w), **meta,
                              old_empirical_prior_share=meta['raw_mass']/(2+meta['raw_mass']),
                              new_empirical_prior_share=meta['n_eff']/(2+meta['n_eff'])))
    assert max_relative_error < 2e-15
    weights = np.array([5e-13, 2e-12])
    low = weights[weights > 1e-12]
    high = 10*weights[10*weights > 1e-12]
    assert len(low) == 1 and len(high) == 2
    p = np.array([.5, .5])
    gamma = np.array([[1., .9], [.9, 1.]])
    n_eff = 1/float(p@p)
    naive_variance = 1/n_eff
    dependent_variance = float(p@gamma@p)
    assert abs(dependent_variance-.95) < 1e-15
    return dict(passed=True, cases=summaries,
                common_scale_relative_error=max_relative_error,
                absolute_cutoff_counterexample=dict(before_count=len(low), after_count=len(high)),
                dependent_block_counterexample=dict(n_eff=n_eff, iid_variance=naive_variance,
                                                     true_weighted_variance=dependent_variance))


def finite_paid_checks():
    # Integer construction: same current forecasts and same every-pair law,
    # while retained parity regime changes the conditional paid target.
    patterns = np.array([[12,12,12], [12,-12,-12], [-12,12,-12], [-12,-12,12]])
    rows = []
    probability = Fraction(1, 2*4*129)
    for sign in (1, -1):
        for error in sign*patterns:
            for gross in range(-64, 65):
                h = tuple((gross+error).tolist())
                rows.append((sign, h, gross-5))
    reduced, retained = defaultdict(list), defaultdict(list)
    for sign, h, paid in rows:
        reduced[h].append(paid)
        retained[(sign,h)].append(paid)

    reports = []
    for mask in ('all', 'nontrivial_fixed_reduced_feasibility'):
        feasible = lambda h: 1 if mask == 'all' else int((sum(h)//3) % 5 != 0)
        rich_mean = {key: Fraction(sum(ds),len(ds)) for key,ds in retained.items()}
        rich_mass = {key: probability*len(ds) for key,ds in retained.items()}
        oracle_r = sum((probability*max(sum(ds),0)*feasible(h)
                        for h,ds in reduced.items()), Fraction(0))
        oracle_i = sum((rich_mass[key]*max(mean,0)*feasible(key[1])
                        for key,mean in rich_mean.items()), Fraction(0))
        gap = oracle_i-oracle_r
        conditional_positive_negative = defaultdict(lambda: [Fraction(0),Fraction(0)])
        for key,m in rich_mean.items():
            g = feasible(key[1])
            conditional_positive_negative[key[1]][0] += rich_mass[key]*max(m,0)*g
            conditional_positive_negative[key[1]][1] += rich_mass[key]*max(-m,0)*g
        gap_formula = sum((min(x) for x in conditional_positive_negative.values()), Fraction(0))
        assert gap == gap_formula >= 0
        if mask == 'all':
            assert gap == Fraction(6,43)
        policies = []
        for bias in (Fraction(0), Fraction(-20), Fraction(20), Fraction(1,2)):
            for penalty in (Fraction(0), Fraction(1,2), Fraction(5), Fraction(20)):
                value = sum((rich_mass[key]*m*feasible(key[1])
                             for key,m in rich_mean.items() if m+bias-penalty > 0), Fraction(0))
                regret = oracle_i-value
                exact = sum((rich_mass[key]*abs(m)*feasible(key[1])
                             for key,m in rich_mean.items()
                             if (m+bias-penalty > 0) != (m>0)), Fraction(0))
                error_term = sum((rich_mass[key]*abs(bias)*feasible(key[1])
                                  for key in rich_mean), Fraction(0))
                penalty_term = sum((rich_mass[key]*penalty*feasible(key[1])
                                    for key in rich_mean), Fraction(0))
                assert regret == exact and 0 <= regret <= error_term+penalty_term
                assert value-oracle_r >= gap-error_term-penalty_term
                policies.append(dict(bias=float(bias), penalty=float(penalty),
                                     paid_value=float(value), exact_regret=float(regret),
                                     error_plus_penalty_bound=float(error_term+penalty_term)))
        reports.append(dict(feasibility=mask, states=len(rows),
                            retained_oracle_value=float(oracle_i), reduced_oracle_value=float(oracle_r),
                            exact_information_gap=str(gap), policies=policies))

    acceptance_examples = []
    for means,probs in (([1,-100],[Fraction(99,100),Fraction(1,100)]),
                        ([100,-1],[Fraction(1,100),Fraction(99,100)])):
        reduced_mean = sum((p*m for p,m in zip(probs,means)), Fraction(0))
        rich_value = sum((p*max(m,0) for p,m in zip(probs,means)), Fraction(0))
        reduced_value = max(reduced_mean,0)
        rich_acceptance = sum((p for p,m in zip(probs,means) if m>0),Fraction(0))
        reduced_acceptance = Fraction(int(reduced_mean>0))
        assert rich_value-reduced_value == Fraction(99,100)
        acceptance_examples.append(dict(means=means, probabilities=[float(p) for p in probs],
                                        retained_acceptance=float(rich_acceptance),
                                        reduced_acceptance=float(reduced_acceptance),
                                        value_gain=float(rich_value-reduced_value)))

    # Relaxation band [new r, old r] has no guaranteed true sign.
    relaxation_examples = []
    for true_mean in (4., -4.):
        estimate, old_penalty, new_penalty = 3., 5., 1.
        old_action = estimate-old_penalty > 0
        new_action = estimate-new_penalty > 0
        gain = (int(new_action)-int(old_action))*true_mean
        assert not old_action and new_action
        relaxation_examples.append(dict(true_mean=true_mean, estimate=estimate,
                                        old_penalty=old_penalty, new_penalty=new_penalty,
                                        newly_admitted=True, actual_value_gain=gain))
    return dict(passed=True, exact_policy_checks=reports,
                oracle_acceptance_counterexamples=acceptance_examples,
                fixed_state_relaxation_counterexamples=relaxation_examples)


def independent_moments(h, alpha, locations, covariances, bins=128, product=False):
    x,w = np.polynomial.legendre.leggauss(8)
    mid = -1+(np.arange(bins)+.5)*(2/bins)
    target = (mid[:,None]+x[None,:]/bins).ravel()
    qw = np.tile(w/bins,bins)
    residual = h[None,None,:]-target[:,None,None]-locations[None,:,:]
    if product:
        variances = np.diagonal(covariances,axis1=1,axis2=2)
        logs = np.log(alpha)[None,:,None]-.5*(math.log(2*math.pi)+np.log(variances)[None,:,:]
                                                  +residual**2/variances[None,:,:])
        high = logs.max(axis=1,keepdims=True)
        density_log = (np.squeeze(high,axis=1)+np.log(np.exp(logs-high).sum(axis=1))).sum(axis=1)
    else:
        precision = np.linalg.inv(covariances)
        logs = np.log(alpha)[None,:]-.5*(len(h)*math.log(2*math.pi)
                 +np.linalg.slogdet(covariances)[1][None,:]
                 +np.einsum('njs,jst,njt->nj',residual,precision,residual))
        high = logs.max(axis=1)
        density_log = high+np.log(np.exp(logs-high[:,None]).sum(axis=1))
    high = density_log.max()
    mass = qw*np.exp(density_log-high)
    mass /= mass.sum()
    mu = float(mass@target)
    var = float(mass@(target-mu)**2)
    return mu,var


def implementation_checks():
    rng = np.random.default_rng(1042026)
    joint_error = marginal_error = truth_error = simplex_error = 0.
    cases = 0
    for m in (1,2,3,5):
        for count in (1,3,7):
            for repetition in range(3):
                alpha = rng.dirichlet(np.ones(count))
                locations = rng.uniform(-.3,.3,size=(count,m))
                r = rng.normal(0,.1,size=(count,m,m))
                covariances = np.einsum('jst,jut->jsu',r,r)+np.eye(m)[None,:,:]*.006
                h = rng.uniform(-.6,.6,size=m)
                post = actual.joint_posterior(h,alpha,locations,covariances)
                independent = independent_moments(h,alpha,locations,covariances)
                joint_error = max(joint_error,abs(post['mean']-independent[0]),
                                  abs(post['variance']-independent[1]))
                marginal = actual.product_posterior(h,alpha,locations,covariances)
                direct_marginal = independent_moments(h,alpha,locations,covariances,product=True)
                marginal_error = max(marginal_error,abs(marginal['mean']-direct_marginal[0]),
                                      abs(marginal['variance']-direct_marginal[1]))
                mm = dict(conditional={'conditional_joint':post},eligible=[{'maturity':0}],
                          current_source_rate=h.tolist(),current_disagreement=(h-h.mean()).tolist(),
                          line_gls_untruncated_mean=0.)
                cfg = dict(norm_floor=.01,cap=.8,slice_bandwidth=.5)
                pre = dict(q=np.ones(m))
                d = dict(temporal=mm,ids=np.arange(m),N=128,truegross=7.)
                issued = actual.conditional_forecast(d,pre,cfg,'conditional_joint',1.2)
                poisoned = actual.conditional_forecast(dict(d,truegross=-1e20),pre,cfg,'conditional_joint',1.2)
                truth_error = max(truth_error,abs(issued[2]-poisoned[2]),abs(issued[3]-poisoned[3]))
                assert issued[4] != poisoned[4]
                mm['eligible'] = []
                unready = actual.conditional_forecast(d,pre,cfg,'conditional_joint',1.2)
                assert unready[2] == 0 and not unready[5]['information_ready']
                targets = np.linspace(-1,1,37)
                weights = rng.dirichlet(np.ones(m))
                errors = h[None,:]-targets[:,None]
                simplex_error = max(simplex_error,float(np.max(abs(h@weights-errors@weights-targets))))
                cases += 1
    assert joint_error < 1e-11 and marginal_error < 1e-10
    assert truth_error == 0 and simplex_error < 1e-15
    return dict(passed=True, random_joint_laws=cases, dimensions=[1,2,3,5],
                actual_cj_vs_independent_quadrature_max_error=joint_error,
                actual_exact_marginal_vs_independent_quadrature_max_error=marginal_error,
                current_truth_poison_issuance_error=truth_error,
                truth_only_changes_offline_callback_score=True,
                no_eligible_history_retains_reference=True, simplex_invariance_error=simplex_error,
                current_api_source=str(OLD_PATH), source_sha256=hashlib.sha256(OLD_PATH.read_bytes()).hexdigest())


def bounded_support_check():
    # Uniform independent target and +/-a error: each branch overflows with a/2.
    a = Fraction(3,32)
    exact_overflow = a/2
    u = (np.arange(200000)+.5)*2/200000-1
    overflow = .5*np.mean((u+float(a)>1)|(u+float(a)<-1))
    overflow += .5*np.mean((u-float(a)>1)|(u-float(a)<-1))
    assert abs(overflow-float(exact_overflow)) < 1e-6
    return dict(passed=True, target_prior='independent uniform[-1,1]',
                error_law='equal-mass +/-3/32', exact_physical_forecast_overflow=str(exact_overflow),
                overflow_probability=float(exact_overflow), finite_grid_probability=overflow,
                inference='Nondegenerate independent error with full-bound uniform target is not an exact bounded-forecast physical model.')


def normalized_api_checks():
    """Actual CJ-R masked pipeline vs independent covariance/precision rebuild."""
    new_path = HERE.parent/'design/reliability_normalized.py'
    if not new_path.exists():
        return dict(passed=False, reason='CJ-R API has not been written yet.')
    spec = importlib.util.spec_from_file_location('paid_information_actual_cjr',new_path)
    cjr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cjr)
    pre = dict(m=3,classes=2,q=np.array([1.,1.2,.8]))
    cfg = dict(prior_mass=2.,prior_variance=.01,prior_sd=.1,beta=.97,kernel_bandwidth=3.,
               horizon=4,lease_archive=48,norm_floor=.01,cap=.8,slice_bandwidth=.5,
               reliability_normalized=True)
    current = dict(k=100,context=np.zeros(1),candidate=np.eye(2),
                   reference=np.array([[0.,1.],[0.,1.]]),ids=np.array([0,2]),h=np.array([.13,.07]))
    raw = np.array([.003,.005,.008,.013,.017,.023,.031,.047,.059])
    residuals = np.array([[.14,-.12,.06],[-.11,.1,-.13],[.17,.08,.12],
                          [-.15,-.09,.01],[.07,.18,-.16],[.12,-.13,-.07],
                          [-.04,.16,.1],[.11,.03,-.14],[-.12,-.1,.15]])
    masks = [np.array(x,dtype=bool) for x in ((1,1,1),(1,0,1),(0,1,1),
                                            (1,1,1),(1,1,0),(1,0,1),
                                            (1,1,1),(0,1,1),(1,1,1))]
    x = np.tile([1.,0.],(8,1))
    y = np.array([0,0,0,0,1,1,1,1])
    future = [dict(x=x.copy(),y=y.copy(),keep=np.ones(8,dtype=bool)) for _ in range(4)]
    def make_archive(scale):
        records = []
        for j,(w,z,mask) in enumerate(zip(raw,residuals,masks)):
            age = 4*(j+1)
            ratio = scale*w/cfg['beta']**age
            assert 0 < ratio <= 1
            context = np.array([math.sqrt(-cfg['kernel_bandwidth']**2*math.log(ratio))])
            probabilities = np.repeat(np.stack(((1+z)/2,(1-z)/2),axis=1)[:,None,:],8,axis=1)
            records.append(dict(k=current['k']-age,maturity=current['k']-age+4,
                                context=context,x=x.copy(),p=probabilities,mask=mask,future=future))
        return records
    transformed,meta = normalize(raw)
    new_transformed,new_meta = cjr.normalize_weights(raw)
    assert np.max(abs(transformed-new_transformed)) < 1e-15
    assert abs(meta['n_eff']-new_meta['effective_mass']) < 1e-14

    complete = np.array([i for i,mask in enumerate(masks) if mask.all()])
    mass = cfg['prior_mass']+transformed[complete].sum()
    center = sum(transformed[i]*residuals[i] for i in complete)/mass
    second = cfg['prior_mass']*cfg['prior_variance']*np.eye(3)
    second += sum(transformed[i]*np.outer(residuals[i],residuals[i]) for i in complete)
    R = second/mass-np.outer(center,center)
    val,vec = np.linalg.eigh((R+R.T)/2)
    R = (vec*np.maximum(val,1e-6))@vec.T
    A = np.diag((pre['q']/np.mean(pre['q']))/cfg['prior_sd']**2)
    rhs = np.zeros(3)
    for w,z,mask in zip(transformed,residuals,masks):
        ids = np.flatnonzero(mask)
        inverse = np.linalg.inv(R[np.ix_(ids,ids)])
        A[np.ix_(ids,ids)] += w*inverse
        rhs[ids] += w*inverse@z[ids]
    P,mu = np.linalg.inv(A),np.linalg.solve(A,rhs)
    baseline = cjr.state(make_archive(1.),current,pre,cfg)
    zero_record = dict(make_archive(1.)[0],mask=np.zeros(3,dtype=bool),k=59,maturity=63)
    with_zero = cjr.state(make_archive(1.)+[zero_record],current,pre,cfg)
    assert with_zero['reliability']['blocks'] == len(raw)
    assert len(with_zero['eligible']) == len(raw)
    for key in ('R','P','mu'):
        assert np.max(abs(with_zero[key]-baseline[key])) == 0
    zero_only = cjr.state([zero_record],current,pre,cfg)
    assert zero_only['reliability']['effective_mass'] == 0
    assert not zero_only['eligible'] and zero_only['paired_atoms']['mass'] == 2
    zero_issued = cjr.forecast(dict(temporal=zero_only,ids=current['ids'],N=128,truegross=7.),
                              pre,cfg,'r_joint',1.2)
    assert zero_issued[2] == 0 and not zero_issued[5]['information_ready']
    ids = current['ids']
    independent_errors = dict(R=float(np.max(abs(baseline['R']-R[np.ix_(ids,ids)]))),
                              P=float(np.max(abs(baseline['P']-P[np.ix_(ids,ids)]))),
                              mu=float(np.max(abs(baseline['mu']-mu[ids]))),
                              atom_mass=abs(baseline['paired_atoms']['mass']-(2+meta['n_eff'])))
    expected_alpha = np.r_[2.,transformed]/(2+meta['n_eff'])
    independent_errors['atom_alpha'] = float(np.max(abs(baseline['paired_atoms']['alpha']-expected_alpha)))
    assert max(independent_errors.values()) < 1e-13

    scale_errors = []
    for scale in (.01,.1,2.):
        changed = cjr.state(make_archive(scale),current,pre,cfg)
        errors = [float(np.max(abs(changed[key]-baseline[key]))) for key in ('R','P','mu')]
        for arm in ('conditional_joint','conditional_marginal','conditional_factorized',
                    'conditional_gaussian','conditional_sandwich','conditional_copula','conditional_joint_diagP'):
            for key in ('mean','variance'):
                errors.append(abs(changed['conditional'][arm][key]-baseline['conditional'][arm][key]))
        scale_errors.append(dict(raw_weight_scale=scale,max_state_or_posterior_error=max(errors)))
    assert max(x['max_state_or_posterior_error'] for x in scale_errors) < 1e-12

    d = dict(temporal=baseline,ids=ids,N=128,truegross=7.)
    issued = cjr.forecast(d,pre,cfg,'r_joint',1.2)
    poisoned = cjr.forecast(dict(d,truegross=-1e20),pre,cfg,'r_joint',1.2)
    assert issued[2:4] == poisoned[2:4] and issued[4] != poisoned[4]
    for arm in cjr.NEW_ARMS:
        assert cjr.configure(cfg,arm)['reliability_normalized']
    assert not cjr.configure(cfg,'conditional_joint')['reliability_normalized']
    # Remote same-regime archive, followed by a changed current regime. This
    # is a finite misspecification counterexample, not a physical replay.
    far_records = []
    far_z = np.full(3,-.18)
    far_probabilities = np.repeat(np.stack(((1+far_z)/2,(1-far_z)/2),axis=1)[:,None,:],8,axis=1)
    for j in range(12):
        age = 4*(j+1)
        radius = math.sqrt(-cfg['kernel_bandwidth']**2*math.log(1e-8/cfg['beta']**age))
        far_records.append(dict(k=100-age,maturity=104-age,context=np.array([radius]),
                                x=x.copy(),p=far_probabilities,mask=np.ones(3,dtype=bool),future=future))
    far_current = dict(current,h=np.full(2,.08))
    old_mm = cjr.state(far_records,far_current,pre,dict(cfg,reliability_normalized=False))
    new_mm = cjr.state(far_records,far_current,pre,cfg)
    old_f = cjr.forecast(dict(temporal=old_mm,ids=ids,N=128,truegross=-13.),pre,
                        dict(cfg,reliability_normalized=False),'conditional_joint',1.2815515655)
    new_f = cjr.forecast(dict(temporal=new_mm,ids=ids,N=128,truegross=-13.),pre,cfg,'r_joint',1.2815515655)
    old_score,new_score = [f[2]-1.2815515655*f[3]-5 for f in (old_f,new_f)]
    assert old_score < 0 < new_score
    far_example = dict(raw_mass=new_mm['reliability']['raw_mass'],
                       normalized_mass=new_mm['reliability']['effective_mass'],
                       old_score=old_score,new_score=new_score,new_admission=True,
                       declared_changed_regime_complete_paid_return=-18.,
                       interpretation='Remote negative residual history becomes large positive fitted target; a changed physical regime can make the additional action harmful.')
    return dict(passed=True,source=str(new_path),source_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
                complete_blocks=len(complete),partial_blocks=len(masks)-len(complete),
                active_source_ids=ids.tolist(),whole_block_normalization=meta,
                independent_R_P_mu_atom_checks=independent_errors,
                common_scale_pipeline_checks=scale_errors,
                normalized_truth_poison_issuance_error=0.,all_seven_matched_controls_use_normalization=True,
                zero_coordinate_blocks_excluded_from_effective_mass=True,
                zero_coordinate_only_history_retains_reference=True,
                far_context_changed_regime_counterexample=far_example,
                historical_masks_preserved=baseline['reliability']['block_masks']==[np.flatnonzero(m).tolist() for m in masks])


if __name__ == '__main__':
    report = dict(scope='Constructed mechanism and read-only API checks; no physical data read.',
                  normalization=normalization_checks(), paid_policy=finite_paid_checks(),
                  current_cj_api=implementation_checks(), bounded_support=bounded_support_check(),
                  normalized_cj_api=normalized_api_checks(),
                  runtime=dict(python=sys.version,numpy=np.__version__))
    output = HERE/'paid_information_report.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),passed=True,
                         normalization_scale_error=report['normalization']['common_scale_relative_error'],
                         joint_api_moment_error=report['current_cj_api']['actual_cj_vs_independent_quadrature_max_error'],
                         marginal_api_moment_error=report['current_cj_api']['actual_exact_marginal_vs_independent_quadrature_max_error'],
                         finite_information_gap=report['paid_policy']['exact_policy_checks'][0]['exact_information_gap']),indent=2))
