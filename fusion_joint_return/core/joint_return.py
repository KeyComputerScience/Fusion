"""Immutable complete-lease target/source joint law, one conditional tail score.

This development operator replaces the independent U / error working model.
No issued decision reads its current future truth. Historical targets are the
ORIGINAL issued complete-lease contrasts, never current-model re-fits on rows
used by online training. Gaussian smoothing / masked completion are explicit
working models; their fitted probabilities are not coverage guarantees.
"""
import math
import numpy as np
from tail_fusion import TruncatedMixture


def psd(A, floor=1e-8):
    A=(A+A.T)/2;v,E=np.linalg.eigh(A)
    return (E*np.maximum(v,floor))@E.T


def state(arrived,current,pre,cfg):
    """Full [U,H] paired locations and partial-block Schur uncertainty."""
    m=pre['m'];dimension=m+1
    prior=np.diag(np.r_[cfg.get('return_prior_variance',.01),
        cfg['prior_variance']*np.mean(pre['q'])/pre['q']])
    observations=[]
    for r in arrived[-cfg['lease_archive']:]:
        a=cfg['beta']**(current['k']-r['k'])*math.exp(-float(np.sum((current['context']-r['context'])**2))/cfg['kernel_bandwidth']**2)
        # Support is a predictable property of the originally issued models.
        a*=r['issued_support']
        if a<1e-12:continue
        ids=np.r_[0,1+r['issued_ids']];v=np.r_[r['issued_u'],r['issued_h']]
        observations.append(dict(weight=a,ids=ids,value=v,origin=r['k'],maturity=r['maturity']))
    complete=[o for o in observations if len(o['ids'])==dimension]
    mass=cfg['prior_mass']+sum(o['weight'] for o in complete)
    mu=sum((o['weight']*o['value'] for o in complete),np.zeros(dimension))/mass
    C=cfg['prior_mass']*prior
    for o in complete:C+=o['weight']*np.outer(o['value'],o['value'])
    C=psd(C/mass-np.outer(mu,mu))
    locations=[np.zeros(dimension)];covariances=[prior.copy()];weights=[cfg['prior_mass']]
    for o in observations:
        ix=o['ids'];cross=C[:,ix];inverse=np.linalg.solve(C[np.ix_(ix,ix)],np.eye(len(ix)))
        loc=mu+cross@inverse@(o['value']-mu[ix])
        V=psd(C-cross@inverse@cross.T,0.)
        # Observed values stay exact: no posterior centering or pseudo-target.
        assert np.max(np.abs(loc[ix]-o['value']))<1e-7
        locations.append(loc);covariances.append(V);weights.append(o['weight'])
    a=np.array(weights);a/=a.sum();locations=np.array(locations);V=np.array(covariances)
    center=a@locations;moment=psd(np.einsum('j,js,jt->st',a,locations-center,locations-center)+np.einsum('j,jst->st',a,V))
    active=np.r_[0,1+current['ids']]
    return dict(eligible=[dict(origin=o['origin'],maturity=o['maturity'],weight=o['weight'],observed_components=len(o['ids'])-1) for o in observations],
        joint_return=dict(weights=a,locations=locations[:,active],covariances=V[:,active][:,:,active],
        moment=moment[np.ix_(active,active)],prior=prior[np.ix_(active,active)],mean=center[active]),
        target_mode='original_issued_complete_lease',complete_blocks=len(complete),partial_blocks=len(observations)-len(complete))


def augment(events,pre,base,cfg):
    pending=[];arrived=[];decisions=[]
    for d in base['decisions']:
        k=d['k'];e=events[k]
        arrived.extend(r for r in pending if r['maturity']<=k);pending=[r for r in pending if r['maturity']>k]
        arrived=arrived[-cfg['lease_archive']:]
        current=dict(d,context=e['context'].copy(),candidate=e['candidate'],reference=e['reference'])
        mm=state(arrived,current,pre,cfg);decisions.append(dict(current,temporal=mm))
        # Targets enter no state until maturity, matching the same fork contract.
        pending.append(dict(k=k,maturity=d['maturity'],context=e['context'].copy(),issued_ids=d['ids'].copy(),
            issued_h=d['h'].copy(),issued_u=d['truegross']/d['N'],issued_support=d['disagreement']))
    return dict(base,decisions=decisions)


def conditional_gaussian(h,mean,C,api):
    inv=np.linalg.solve(C[1:,1:],np.eye(len(h)));cross=C[0,1:]
    loc=float(mean[0]+cross@inv@(h-mean[1:]));var=max(1e-12,float(C[0,0]-cross@inv@cross))
    return loc,var


def make_law(d,pre,cfg,arm,api):
    a=d['temporal']['joint_return'];w=a['weights'];loc=a['locations'];V=a['covariances'];m=len(d['ids'])
    scale=np.diag(np.r_[cfg['b_return'],np.repeat(cfg['b_source'],m)])
    # One bandwidth covariance over [return,source forecasts] preserves target
    # coupling and source dependence. Both tunables get the same 3x3 budget.
    neff=1/float(w@w);scott=neff**(-2/(m+5))
    K=scott*scale@(a['moment']+a['prior'])@scale
    cov=np.array([psd(v+K) for v in V]);h=d['h'];dimension=m+1
    if arm=='return_gaussian':
        cm,cv=conditional_gaussian(h,a['mean'],a['moment']+K,api)
        law=TruncatedMixture([cm],[cv],[0.])
    elif arm=='return_joint':
        means=[];variances=[];logs=[]
        for weight,location,C in zip(w,loc,cov):
            cm,cv=conditional_gaussian(h,location,C,api);r=h-location[1:]
            inv=np.linalg.solve(C[1:,1:],np.eye(m));ld=np.linalg.slogdet(C[1:,1:])[1]
            logs.append(math.log(weight)-.5*(m*math.log(2*math.pi)+ld+float(r@inv@r)))
            means.append(cm);variances.append(cv)
        law=TruncatedMixture(means,variances,logs)
    elif arm=='return_factorized':
        # Preserve EXACT same U margin and all (U,H_s) pair marginals; discard
        # only higher-order conditional source dependence: prod f(U,H_s)/f(U)^(m-1).
        def evaluate(bins):
            u,qw=api.c.quadrature(bins);log=np.zeros(len(u))
            for s in range(m):
                ids=[0,1+s];logs=[]
                for weight,location,C in zip(w,loc,cov):
                    CC=C[np.ix_(ids,ids)];inverse=np.linalg.solve(CC,np.eye(2));r=np.column_stack((u-location[0],np.repeat(h[s]-location[1+s],len(u))))
                    logs.append(math.log(weight)-math.log(2*math.pi)-.5*np.linalg.slogdet(CC)[1]-.5*np.einsum('is,st,it->i',r,inverse,r))
                log+=api.c.logsumexp(np.array(logs).T,axis=1)
            umarg=np.array([math.log(weight)-.5*(math.log(2*math.pi*C[0,0])+(u-location[0])**2/C[0,0]) for weight,location,C in zip(w,loc,cov)]).T
            log-=(m-1)*api.c.logsumexp(umarg,axis=1);log+=np.log(qw)
            return api.DiscreteLaw(u,np.exp(log-api.c.logsumexp(log)))
        bins=max(32,int(math.ceil(1/math.sqrt(float(cov[:,0,0].min())))))
        old=evaluate(bins)
        for _ in range(6):
            bins*=2;new=evaluate(bins);error=max(abs(old.lower_tail(alpha)-new.lower_tail(alpha)) for alpha in (.0001,.01,.1,.5,1.))
            if error<1e-6:break
            old=new
        law=new;law.quadrature_error=error;law.quadrature_nodes=8*bins;law.quadrature_converged=error<1e-6
    else:raise ValueError(arm)
    law.component_count=len(w) if arm!='return_gaussian' else 1;law.active_dimension=m;law.neff=neff
    return law
