"""One new UNFROZEN horizon-conditional return law on used development.

All information controls see identical multivariate PHYSICAL source blocks.
Factorization preserves (E,entire source path) margins conditional on the
same common C/R descriptor, not merely individual coordinate margins.
"""
import math
import numpy as np
from horizon_descriptor import HorizonDescriptor, LAGS
from conditional_pairing import (psd, logsumexp, quadrature, DiscreteLaw,
                                  TruncatedMixture, RAW_ALPHA)

ARMS = ('paired', 'pair_factorized', 'full_gaussian',
        'joint_diagonal_kernel', 'unconditional')
SOURCE_WIDTH = 2*LAGS
PRIOR_MASS = 2.


def compact(descriptor):
    """Fixed map chosen before the development-law outcome run.

    Per lag retain soft directional evidence relative to the fused soft
    anchor, and the source's mean top-two probability margin. Common
    descriptor retains hard/soft anchors, C/R probability margins, both
    disagreement amounts, current availability and startup age.
    """
    classes = descriptor['schema']['classes']; m = descriptor['schema']['sources']
    full = descriptor['source_blocks'].reshape(m, LAGS, classes+4)
    mask = descriptor['source_coordinate_mask'].reshape(m, LAGS, classes+4)
    coords = [classes+1, classes+3]
    z = full[:, :, coords].reshape(m, SOURCE_WIDTH)
    observed = mask[:, :, coords].reshape(m, SOURCE_WIDTH)
    common = descriptor['common_descriptor']; tail = common[3*classes:]
    w = np.r_[descriptor['hard_anchor'],
              # Reconstruct the current fused soft anchor from the
              # snapshot is unnecessary: it is explicitly returned below.
              descriptor['soft_anchor'], tail]
    return z, observed, w


def immutable(d, recording):
    return dict(origin=int(d['k']), maturity=int(d['maturity']),
                anchor=float(d['anchor']),
                residual=float(d['truegross']/d['N']-d['anchor']),
                descriptor=np.asarray(d['descriptor']).copy(),
                observed=np.asarray(d['descriptor_observed']).copy(),
                common=np.asarray(d['horizon_common']).copy(),
                support=float(d['soft_support']),
                context=np.asarray(d['horizon_common']).copy(), recording=recording)


def library(stream):
    return [immutable(d, stream['recording']) for d in stream['decisions']
            if d['maturity'] <= stream['windows'] and d['soft_support'] > 0.]


def state(prior_records, arrived, current, pre):
    common_width = len(current['horizon_common']); m = pre['m']
    dimension = 1+common_width+m*SOURCE_WIDTH
    quality = np.asarray(pre['q']); scales = .01*quality.mean()/quality
    prior = np.diag(np.r_[.01, np.repeat(.01, common_width),
                          np.repeat(scales, SOURCE_WIDTH)])
    records = []
    context = np.asarray(current['horizon_common'])
    for persistent, collection in ((True, prior_records), (False, arrived[-48:])):
        for r in collection:
            weight = math.exp(-float(np.sum((context-r['context'])**2)))*r['support']
            if not persistent: weight *= .97**max(0, current['k']-r['origin'])
            if weight < 1e-12: continue
            source_ids = np.flatnonzero(np.asarray(r['observed']).ravel())+1+common_width
            ids = np.r_[np.arange(1+common_width), source_ids]
            full = np.r_[r['residual'], r['common'], np.asarray(r['descriptor']).ravel()]
            records.append(dict(weight=weight, ids=ids, values=full[ids],
                                persistent=persistent))
    complete = [r for r in records if len(r['ids']) == dimension]
    mass = PRIOR_MASS+sum(r['weight'] for r in complete)
    mu = sum((r['weight']*r['values'] for r in complete), np.zeros(dimension))/mass
    C = PRIOR_MASS*prior
    for r in complete: C += r['weight']*np.outer(r['values'], r['values'])
    C = psd(C/mass-np.outer(mu, mu))
    locations = [np.zeros(dimension)]; covariances = [prior.copy()]; weights = [PRIOR_MASS]
    for r in records:
        ix = r['ids']; cross = C[:, ix]
        solve = np.linalg.solve(C[np.ix_(ix, ix)], np.eye(len(ix)))
        location = mu+cross@solve@(r['values']-mu[ix])
        covariance = psd(C-cross@solve@cross.T, 0.)
        if np.max(abs(location[ix]-r['values'])) > 1e-7:
            raise AssertionError('Observed horizon block not preserved')
        locations.append(location); covariances.append(covariance); weights.append(r['weight'])
    weights = np.asarray(weights); weights /= weights.sum()
    locations = np.asarray(locations); covariances = np.asarray(covariances)
    mean = weights@locations
    moment = psd(np.einsum('j,js,jt->st', weights, locations-mean, locations-mean)
                 +np.einsum('j,jst->st', weights, covariances))
    return dict(weights=weights, locations=locations, covariances=covariances,
                mean=mean, moment=moment, prior=prior,
                eligible=bool(records) and len(current['ids'])>0,
                records=len(records), persistent_records=sum(r['persistent'] for r in records),
                common_width=common_width, full_dimension=dimension,
                complete_blocks=len(complete), partial_blocks=len(records)-len(complete))


def build_stream(events, pre, base, cfg, prior_records=(), recording=None):
    path = HorizonDescriptor(); issued = []
    for event in events:
        d = path.push(event, recording)
        # HorizonDescriptor also returns the current soft anchor, needed
        # by compact's common conditioning block.
        z, observed, common = compact(d)
        issued.append((d, z, observed, common))
    pending = []; arrived = []; decisions = []
    for old in base['decisions']:
        k = old['k']
        arrived.extend(r for r in pending if r['maturity'] <= k)
        pending = [r for r in pending if r['maturity'] > k]; arrived = arrived[-48:]
        query, z, observed, common = issued[k]
        d = dict(old, ids=query['active_sources'],
                 anchor=query['hard_anchor'], descriptor=z,
                 descriptor_observed=observed, horizon_common=common,
                 disagreement=query['hard_disagreement'], soft_support=query['state_weight_support'],
                 soft_disagreement=query['soft_disagreement'],
                 cal_context=np.asarray([query['hard_anchor'],query['hard_disagreement'],len(query['active_sources'])/pre['m']]))
        d['temporal'] = state(prior_records, arrived, d, pre)
        decisions.append(d); pending.append(immutable(d, recording))
    return dict(base, decisions=decisions, recording=recording)


def condition(mean, covariance, kept, observed, query):
    """Exact Gaussian block conditioning and observed log density."""
    kept = np.asarray(kept, int); observed = np.asarray(observed, int)
    if len(observed) == 0:
        return mean[kept], covariance[np.ix_(kept, kept)], 0.
    CC = covariance[np.ix_(observed, observed)]; cross = covariance[np.ix_(kept, observed)]
    delta = np.asarray(query)-mean[observed]
    solve = np.linalg.solve(CC, np.column_stack((delta, cross.T)))
    cm = mean[kept]+cross@solve[:, 0]
    cv = psd(covariance[np.ix_(kept, kept)]-cross@solve[:, 1:], 1e-12)
    log = -.5*(len(observed)*math.log(2*math.pi)+np.linalg.slogdet(CC)[1]+float(delta@solve[:, 0]))
    return cm, cv, float(log)


def conditioned_components(d, arm):
    s = d['temporal']; weights = s['weights']; neff = 1/float(weights@weights)
    cw = s['common_width']; source_ids = np.flatnonzero(d['descriptor_observed'].ravel())
    target = np.r_[0, source_ids+1+cw]
    common_ids = np.arange(1, 1+cw)
    # Scott's dimension counts E, common context and each OBSERVED path
    # coordinate. Coordinates are not recast as physical sources.
    scott = neff**(-2/(len(target)+cw+4))
    K = scott*(s['moment']+s['prior'])
    if arm == 'joint_diagonal_kernel': K = np.diag(np.diag(K))
    if arm == 'full_gaussian':
        means = [s['mean']]; covs = [psd(s['moment']+K)]; ws = [1.]
    else:
        means = s['locations']; covs = [psd(v+K) for v in s['covariances']]; ws = weights
    out_mean = []; out_cov = []; logs = []
    for mean, cov, weight in zip(means, covs, ws):
        cm, cv, log = condition(mean, cov, target, common_ids, d['horizon_common'])
        out_mean.append(cm); out_cov.append(cv); logs.append(math.log(weight)+log)
    logs = np.asarray(logs); logs -= logsumexp(logs)
    groups = []
    for source in range(d['descriptor'].shape[0]):
        members = np.flatnonzero((source_ids//SOURCE_WIDTH)==source)+1
        if len(members): groups.append(members)
    query = d['descriptor'].ravel()[source_ids]
    return np.asarray(out_mean), np.asarray(out_cov), logs, query, groups, neff


def scalar_log_terms(mean, covariance, query):
    """A Gaussian [E,Z] density as a quadratic in E, retaining full Z block."""
    inverse = np.linalg.solve(covariance, np.eye(len(mean)))
    delta = np.asarray(query)-mean[1:]; cross = float(inverse[0,1:]@delta)
    a = -.5*inverse[0,0]; b = mean[0]*inverse[0,0]-cross
    c = -.5*(mean[0]**2*inverse[0,0]-2*mean[0]*cross+float(delta@inverse[1:,1:]@delta))
    c -= .5*(len(mean)*math.log(2*math.pi)+np.linalg.slogdet(covariance)[1])
    return float(a), float(b), float(c)


def make_law(d, arm):
    means, covs, logs, query, groups, neff = conditioned_components(d, arm)
    bounds = (-1-d['anchor'], 1-d['anchor'])
    if arm == 'pair_factorized':
        terms = []
        for group in groups:
            ix = np.r_[0,group]
            terms.append(np.asarray([scalar_log_terms(mu[ix],C[np.ix_(ix,ix)],query[group-1]) for mu,C in zip(means,covs)]))
        marginal = np.asarray([scalar_log_terms(mu[:1], C[:1,:1], []) for mu,C in zip(means,covs)])
        def density(e, terms):
            return logsumexp(logs[:,None]+terms[:,0,None]*e[None,:]**2+terms[:,1,None]*e[None,:]+terms[:,2,None], axis=0)
        def evaluate(bins):
            e, qw = quadrature(bins,bounds)
            log = sum((density(e,t) for t in terms),np.zeros(len(e)))
            log -= (len(groups)-1)*density(e,marginal)
            log += np.log(qw)
            return DiscreteLaw(e,np.exp(log-logsumexp(log)))
        bins = max(32,int(math.ceil(1/math.sqrt(float(covs[:,0,0].min())))))
        old = evaluate(bins); error = float('inf')
        for _ in range(6):
            bins *= 2; law = evaluate(bins)
            error = max(abs(old.lower_tail(a)-law.lower_tail(a)) for a in (.1,.5,1.))
            if error < 1e-6: break
            old = law
        law.quadrature_error = error; law.quadrature_converged = error < 1e-6
    else:
        cm = []; cv = []; lw = []
        for mean,C,log in zip(means,covs,logs):
            if arm == 'unconditional':
                mu,var,more = mean[:1],C[:1,:1],0.
            else:
                mu,var,more = condition(mean,C,[0],np.arange(1,len(mean)),query)
            cm.append(float(mu[0])); cv.append(float(var[0,0])); lw.append(float(log+more))
        law = TruncatedMixture(cm,cv,lw,bounds)
        law.quadrature_error = 0.; law.quadrature_converged = True
    law.anchor = d['anchor']; law.neff = neff
    return law


def raw_run(stream, arm):
    rows = []; diagnostics = []
    for d in stream['decisions']:
        law = make_law(d,arm)
        score = float(d['N']*(law.lower_tail(RAW_ALPHA)+d['anchor'])-5.)
        ready = d['temporal']['eligible']
        rows.append(dict(k=d['k'],maturity=d['maturity'],action=bool(ready and score>0),
                         local_net=d['localnet'],truegross=d['truegross'],gate_score=score,
                         raw_gate_score=score,gain=score+5.,q_issued=0.,posterior_or_block_sd=1.,
                         standardized_score=score-(d['truegross']-5.),
                         lower_covered=bool(d['truegross']-5.>=score),information_ready=ready,
                         disagreement=d['disagreement'],anchor=d['anchor'],N=d['N'],
                         cal_context=d['cal_context'].tolist(),
                         horizon_common=d['horizon_common'].tolist(),
                         soft_support=d['soft_support'],source_descriptor=d['descriptor'].tolist(),
                         source_observed=d['descriptor_observed'].tolist(),active_source_ids=d['ids'].tolist(),
                         conditional_paid_mean=d['N']*(law.mean+d['anchor'])-5.,
                         conditional_paid_variance=d['N']**2*law.variance))
        diagnostics.append(dict(k=d['k'],neff=law.neff,quadrature_error=law.quadrature_error,
                                quadrature_converged=law.quadrature_converged,
                                complete_blocks=d['temporal']['complete_blocks'],
                                partial_blocks=d['temporal']['partial_blocks']))
    return dict(rows=rows,law_diagnostics=diagnostics,raw_alpha=RAW_ALPHA)
