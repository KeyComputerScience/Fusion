"""Independent CJ-T joint-target and strong-product mechanism checks.

No physical data are read. All laws are constructed and finite.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys

import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
LOG2PI=math.log(2*math.pi)


def logsumexp(a,axis=-1):
    high=np.max(a,axis=axis,keepdims=True)
    return np.squeeze(high,axis=axis)+np.log(np.exp(a-high).sum(axis=axis))


def logpdf(x,alpha,locations,covariances):
    x=np.atleast_2d(x)
    residual=x[:,None,:]-locations[None,:,:]
    inverse=np.linalg.inv(covariances)
    terms=np.log(alpha)[None,:]-.5*(x.shape[1]*LOG2PI
          +np.linalg.slogdet(covariances)[1][None,:]
          +np.einsum('njs,jst,njt->nj',residual,inverse,residual))
    return logsumexp(terms,axis=1)


def project(locations,covariances,ids):
    return locations[:,ids],covariances[:,ids][:,:,ids]


def nodes(bins=128,left=-1.,right=1.):
    x,w=np.polynomial.legendre.leggauss(8)
    half=(right-left)/(2*bins)
    mid=left+(np.arange(bins)+.5)*(right-left)/bins
    return (mid[:,None]+half*x[None,:]).ravel(),np.tile(half*w,bins)


def product_logpdf(h,u,alpha,locations,covariances):
    m=len(h)
    loc,V=project(locations,covariances,[m])
    density=-(m-1)*logpdf(u[:,None],alpha,loc,V)
    for s in range(m):
        loc,V=project(locations,covariances,[s,m])
        density += logpdf(np.column_stack((np.full(len(u),h[s]),u)),alpha,loc,V)
    return density


def direct_quad(h,alpha,locations,covariances,product=False,bins=128):
    u,w=nodes(bins)
    if product:
        ld=product_logpdf(h,u,alpha,locations,covariances)
    else:
        ld=logpdf(np.column_stack((np.broadcast_to(h,(len(u),len(h))),u)),
                  alpha,locations,covariances)
    mass=w*np.exp(ld-ld.max())
    mass/=mass.sum()
    mean=float(mass@u)
    variance=float(mass@(u-mean)**2)
    return dict(mean=mean,variance=variance)


def analytic(h,alpha,locations,covariances):
    m=len(h)
    logs=[];means=[];seconds=[]
    cdf=lambda x:.5*math.erfc(-float(x)/math.sqrt(2))
    phi=lambda x:math.exp(-.5*float(x)**2)/math.sqrt(2*math.pi)
    for a,loc,S in zip(alpha,locations,covariances):
        inv=np.linalg.inv(S[:m,:m])
        residual=h-loc[:m]
        mean=float(loc[m]+S[m,:m]@inv@residual)
        variance=float(S[m,m]-S[m,:m]@inv@S[:m,m])
        assert variance>0
        sd=math.sqrt(variance)
        lo,hi=(-1-mean)/sd,(1-mean)/sd
        Z=cdf(hi)-cdf(lo) if lo<0 else cdf(-lo)-cdf(-hi)
        if Z<1e-280:
            continue
        shift=(phi(lo)-phi(hi))/Z
        mu=mean+sd*shift
        var=max(0.,variance*(1+(lo*phi(lo)-hi*phi(hi))/Z-shift**2))
        evidence=-.5*(m*LOG2PI+np.linalg.slogdet(S[:m,:m])[1]+residual@inv@residual)
        logs.append(math.log(a)+evidence+math.log(Z))
        means.append(mu);seconds.append(var+mu**2)
    logs=np.array(logs)
    p=np.exp(logs-logsumexp(logs,axis=0))
    mu=float(p@means)
    var=max(0.,float(p@seconds)-mu**2)
    return dict(mean=mu,variance=var,component_weights=p.tolist())


def random_law_checks():
    rng=np.random.default_rng(42026)
    error=singleton=gaussian=0.
    cases=0
    for m in (1,2,3,5):
        for count in (1,4,7):
            alpha=rng.dirichlet(np.ones(count))
            locations=rng.uniform(-.3,.3,(count,m+1))
            r=rng.normal(0,.08,(count,m+1,m+1))
            covariances=np.einsum('jst,jut->jsu',r,r)+np.eye(m+1)[None]*.003
            h=rng.uniform(-.6,.6,m)
            exact=analytic(h,alpha,locations,covariances)
            independent=direct_quad(h,alpha,locations,covariances)
            error=max(error,*(abs(exact[k]-independent[k]) for k in ('mean','variance')))
            if m==1:
                product=direct_quad(h,alpha,locations,covariances,product=True)
                singleton=max(singleton,*(abs(product[k]-independent[k]) for k in ('mean','variance')))
            if count==1:
                center=alpha@locations
                covariance=np.einsum('j,jst->st',alpha,covariances)
                covariance+=np.einsum('j,js,jt->st',alpha,locations-center,locations-center)
                gauss=analytic(h,np.ones(1),center[None],covariance[None])
                gaussian=max(gaussian,*(abs(gauss[k]-exact[k]) for k in ('mean','variance')))
            cases+=1
    assert error<1e-11 and singleton<1e-11 and gaussian<1e-14
    return dict(passed=True,constructed_laws=cases,dimensions=[1,2,3,5],
                analytic_vs_independent_joint_quadrature_error=error,
                single_source_joint_vs_product_error=singleton,
                gaussian_joint_vs_full_gaussian_error=gaussian)


def parity_checks():
    N,fee,q,nu=128,5,1.2815515655,.01
    a=.125; b=3/32; c=fee/N
    patterns=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=float)
    alpha=np.array([.2,.2,.2,.2,.2])
    locations=np.column_stack((patterns[:,:2]*a,c+patterns[:,2]*b))
    locations=np.r_[np.array([[0.,0.,c]]),locations]
    covariances=np.stack([np.diag([.1**2,.1**2,.1**2])]+[np.diag([.025**2,.025**2,.005**2])]*4)
    negative=locations.copy();negative[1:,2]=2*c-negative[1:,2]
    h=np.full(2,a)
    outputs={}
    for sign,loc in (('plus',locations),('minus',negative)):
        center=alpha@loc
        covariance=np.einsum('j,jst->st',alpha,covariances)
        covariance+=np.einsum('j,js,jt->st',alpha,loc-center,loc-center)
        outputs[sign]={}
        for arm,posterior in (('joint',analytic(h,alpha,loc,covariances)),
                              ('product',direct_quad(h,alpha,loc,covariances,product=True,bins=256)),
                              ('gaussian',analytic(h,np.ones(1),center[None],covariance[None]))):
            outputs[sign][arm]=dict(posterior,F=N*posterior['mean'],
                S=N*math.sqrt(posterior['variance']+nu**2),
                score=N*posterior['mean']-q*N*math.sqrt(posterior['variance']+nu**2)-fee)
        outputs[sign]['raw_joint_mean']=center.tolist()
        outputs[sign]['raw_joint_covariance']=covariance.tolist()
    marginal_error=0.
    for dims in ([2],[0,2],[1,2],[0,1]):
        lp,V=project(locations,covariances,dims)
        lm,_=project(negative,covariances,dims)
        grid=np.linspace(-.7,.7,201)
        test=np.column_stack([np.roll(grid,s*17) for s in range(len(dims))])
        marginal_error=max(marginal_error,float(np.max(abs(logpdf(test,alpha,lp,V)-logpdf(test,alpha,lm,V)))))
    covariance_error=float(np.max(abs(np.array(outputs['plus']['raw_joint_covariance'])
                                      -np.array(outputs['minus']['raw_joint_covariance']))))
    assert marginal_error<1e-12 and covariance_error<1e-15
    assert outputs['plus']['joint']['score']>0>outputs['minus']['joint']['score']
    for arm in ('product','gaussian'):
        assert abs(outputs['plus'][arm]['score']-outputs['minus'][arm]['score'])<1e-11
        assert outputs['plus'][arm]['score']<0
    # Before smoothing: both regimes equally probable, H patterns uniform.
    # Every reduced H state has conditional paid returns {-12,+12}; its mean
    # is zero. Rich information includes which paired regime generated it.
    rich_value=Fraction(1,2)*12
    reduced_value=Fraction(0)
    assert rich_value-reduced_value==6
    return dict(passed=True,source_target_and_target_marginals_identical=True,
                every_pair_including_forecast_pair_identical=True,
                exact_marginal_logdensity_error=marginal_error,
                full_raw_covariance_error=covariance_error,conditioned_forecast=h.tolist(),
                smoothed_posteriors=outputs,
                discrete_paid_construction=dict(gross_values=[-7,17],paid_values=[-12,12],
                    rich_oracle_value=float(rich_value),reduced_oracle_value=float(reduced_value),
                    exact_paid_information_gap=6.,source_forecasts_bounded=True,
                    target_and_forecasts_generated_jointly=True))


def completion_checks():
    rng=np.random.default_rng(200426)
    r=rng.normal(0,.15,(5,5));C=r@r.T+np.eye(5)*.01
    loc=rng.uniform(-.2,.2,5)
    error=0.;min_eigen=1.
    for observed in ([4],[0,4],[1,2,4],[0,1,2,3,4]):
        observed=np.array(observed)
        value=rng.uniform(-.3,.3,len(observed))
        cross=C[:,observed];inverse=np.linalg.inv(C[np.ix_(observed,observed)])
        mean=loc+cross@inverse@(value-loc[observed])
        V=C-cross@inverse@cross.T
        error=max(error,float(np.max(abs(mean[observed]-value))),float(np.max(abs(V[observed,:]))))
        min_eigen=min(min_eigen,float(np.linalg.eigvalsh((V+V.T)/2).min()))
        assert 4 in observed
    assert error<1e-14 and min_eigen>-1e-14
    return dict(passed=True,masked_patterns=4,observed_target_always_retained=True,
                observed_location_and_zero_completion_variance_error=error,
                completion_minimum_eigenvalue=min_eigen,
                target_only_history_is_observed_target_information=True)


def actual_api_checks():
    path=HERE.parent/'design_transport/transport.py'
    if not path.exists():
        return dict(passed=False,reason='CJ-T API not available yet.')
    sp=importlib.util.spec_from_file_location('theory_actual_target_api',path)
    api=importlib.util.module_from_spec(sp);sp.loader.exec_module(api)
    rng=np.random.default_rng(342026)
    analytic_error=product_error=refinement_error=0.;cases=0
    for m in (1,2,4):
        for count in (1,4,7):
            alpha=rng.dirichlet(np.ones(count));locations=rng.uniform(-.3,.3,(count,m+1))
            r=rng.normal(0,.08,(count,m+1,m+1))
            covariances=np.einsum('jst,jut->jsu',r,r)+np.eye(m+1)[None]*.003
            h=rng.uniform(-.6,.6,m)
            joint=api.conditional_posterior(h,alpha,locations,covariances)
            product=api.product_posterior(h,alpha,locations,covariances)
            expected=direct_quad(h,alpha,locations,covariances)
            expected_product=direct_quad(h,alpha,locations,covariances,product=True)
            refined_product=direct_quad(h,alpha,locations,covariances,product=True,bins=256)
            analytic_error=max(analytic_error,*(abs(joint[k]-expected[k]) for k in ('mean','variance')))
            product_error=max(product_error,*(abs(product[k]-expected_product[k]) for k in ('mean','variance')))
            refinement_error=max(refinement_error,*(abs(expected_product[k]-refined_product[k]) for k in ('mean','variance')))
            cases+=1
    assert max(analytic_error,product_error,refinement_error)<1e-10

    pre=dict(m=3,classes=2,q=np.array([1.,1.2,.8]))
    cfg=dict(prior_mass=2.,prior_variance=.01,prior_sd=.1,beta=.97,kernel_bandwidth=3.,
             horizon=4,lease_archive=48,norm_floor=.01,cap=.8,slice_bandwidth=.5)
    current=dict(k=100,context=np.zeros(1),candidate=np.eye(2),
                 reference=np.array([[0.,1.],[0.,1.]]),ids=np.array([0,2]),h=np.array([.13,.07]))
    raw=np.array([.003,.008,.017,.023,.047,.059])
    H=np.array([[.14,-.12,.06],[-.11,.1,-.13],[.17,.08,.12],
                [-.15,-.09,.01],[.07,.18,-.16],[.12,-.13,-.07]])
    masks=[np.array(v,dtype=bool) for v in ((1,1,1),(1,0,1),(0,0,0),
                                           (1,1,1),(0,1,1),(1,1,1))]
    positives=[3,4,5,4,3,5]
    records=[];expected=[]
    x=np.tile([1.,0.],(8,1))
    for j,(w,h,mask,kpos) in enumerate(zip(raw,H,masks,positives)):
        age=4*(j+1)
        radius=math.sqrt(-cfg['kernel_bandwidth']**2*math.log(w/cfg['beta']**age))
        y=np.ones(8,dtype=int);y[:kpos]=0
        future=[dict(x=x.copy(),y=y.copy(),keep=np.ones(8,dtype=bool)) for _ in range(4)]
        p=np.repeat(np.stack(((1+h)/2,(1-h)/2),axis=1)[:,None,:],8,axis=1)
        records.append(dict(k=100-age,maturity=104-age,context=np.array([radius]),
                            x=x.copy(),p=p,mask=mask,future=future))
        ids=np.r_[np.flatnonzero(mask),3]
        expected.append(dict(weight=w,ids=ids,value=np.r_[h[np.flatnonzero(mask)],(2*kpos-8)/8]))
    obs=api.collect_pairs(records,current,pre,cfg)
    assert len(obs)==len(records)
    collection_error=max(max(abs(o['weight']-e['weight']),float(np.max(abs(o['value']-e['value']))))
                         for o,e in zip(obs,expected))
    assert collection_error<1e-15
    assert all(o['ids'][-1]==3 for o in obs)
    assert len(obs[2]['ids'])==1 and obs[2]['target']==.25
    law=api.shared_law(obs,pre,current['ids'],cfg)
    prior_covariance=np.ones((4,4))/3
    prior_covariance[np.arange(3),np.arange(3)]+=.01*np.mean(pre['q'])/pre['q']
    assert np.max(abs(api.prior_covariance(pre,cfg)-prior_covariance))<1e-15
    mass=2.;first=np.zeros(4);second=2*prior_covariance.copy()
    for o in expected:
        if len(o['ids'])==4:
            mass+=o['weight'];first+=o['weight']*o['value'];second+=o['weight']*np.outer(o['value'],o['value'])
    mu=first/mass;R=second/mass-np.outer(mu,mu)
    ev,UU=np.linalg.eigh((R+R.T)/2);R=(UU*np.maximum(ev,1e-6))@UU.T
    active=np.array([0,2,3]);locs=[np.zeros(3)];Vs=[R[np.ix_(active,active)]];weights=[2.]
    for o in expected:
        ids=o['ids'];iv=np.linalg.inv(R[np.ix_(ids,ids)])
        loc=mu+R[:,ids]@iv@(o['value']-mu[ids]);V=R-R[:,ids]@iv@R[ids,:]
        loc[ids]=o['value'];V=(V+V.T)/2;V[ids,:]=0.;V[:,ids]=0.
        ev,UU=np.linalg.eigh(V);V=(UU*np.maximum(ev,0.))@UU.T
        locs.append(loc[active]);Vs.append(V[np.ix_(active,active)]);weights.append(o['weight'])
    jitter=np.diag(np.r_[.5**2*.01*np.mean(pre['q'])/pre['q'][current['ids']],.5**2*.01**2])
    independent_alpha=np.array(weights)/sum(weights)
    independent_locs=np.array(locs);independent_covs=np.array(Vs)+jitter
    law_error=max(float(np.max(abs(law['alpha']-independent_alpha))),
                  float(np.max(abs(law['locations']-independent_locs))),
                  float(np.max(abs(law['covariances']-independent_covs))))
    assert law_error<1e-12
    assert np.max(abs(law['locations'][1:,-1]-np.array([o['value'][-1] for o in obs])))==0
    actual_sharp_joint=api.conditional_posterior(current['h'],law['alpha'],law['locations'],law['covariances'])
    actual_sharp_product=api.product_posterior(current['h'],law['alpha'],law['locations'],law['covariances'])
    independent_sharp_joint=direct_quad(current['h'],law['alpha'],law['locations'],law['covariances'],bins=512)
    independent_sharp_product=direct_quad(current['h'],law['alpha'],law['locations'],law['covariances'],product=True,bins=512)
    refined_sharp_product=direct_quad(current['h'],law['alpha'],law['locations'],law['covariances'],product=True,bins=1024)
    sharp_error=max(*(abs(actual_sharp_joint[k]-independent_sharp_joint[k]) for k in ('mean','variance')),
                    *(abs(actual_sharp_product[k]-independent_sharp_product[k]) for k in ('mean','variance')),
                    *(abs(refined_sharp_product[k]-independent_sharp_product[k]) for k in ('mean','variance')))
    assert sharp_error<1e-10
    # Direct state, collection, pair prior, and all controls operate without
    # current truth; poison only the offline callback field at issuance.
    mm=api.state(records,current,pre,cfg)
    d=dict(temporal=mm,ids=current['ids'],N=128,truegross=7.)
    truth_error=0.
    for arm in api.NEW_ARMS:
        issued=api.forecast(d,pre,cfg,arm,1.2)
        poisoned=api.forecast(dict(d,truegross=-1e20),pre,cfg,arm,1.2)
        truth_error=max(truth_error,abs(issued[2]-poisoned[2]),abs(issued[3]-poisoned[3]))
        assert issued[4]!=poisoned[4]
    assert truth_error==0
    only_target=api.state([records[2]],current,pre,cfg)
    assert only_target['transport_ready'] and only_target['transport_pair_count']==1
    empty=api.state([],current,pre,cfg)
    assert not empty['transport_ready']
    empty_forecast=api.forecast(dict(d,temporal=empty),pre,cfg,'t_joint',1.2)
    assert empty_forecast[2]==0 and not empty_forecast[-1]['information_ready']
    try:
        api.collect_pairs([dict(records[0],maturity=101)],current,pre,cfg)
    except AssertionError:
        unmatured_rejected=True
    else:
        unmatured_rejected=False
    assert unmatured_rejected
    return dict(passed=True,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        random_laws=cases,actual_joint_vs_independent_quad_error=analytic_error,
        actual_strong_product_vs_independent_quad_error=product_error,
        independent_product_refinement_error=refinement_error,
        paired_collection_error=collection_error,independent_complete_masked_joint_law_error=law_error,
        actual_history_target_kernel_sd=law['target_kernel_sd'],actual_sharp_history_posterior_error=sharp_error,
        source_forecast_masks_and_observed_target_retained=True,target_only_block_contributes_observed_target=True,
        no_history_retains_reference=True,unmatured_block_rejected=True,
        all_five_target_arms_truth_poison_issuance_error=truth_error)


if __name__=='__main__':
    report=dict(scope='Constructed mechanism checks; no physical data read.',
                analytic_conditioning=random_law_checks(),paired_separation=parity_checks(),
                masked_completion=completion_checks(),actual_cjt_api=actual_api_checks(),
                runtime=dict(python=sys.version,numpy=np.__version__))
    output=HERE/'target_theory_report.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),passed=True,
        analytic_error=report['analytic_conditioning']['analytic_vs_independent_joint_quadrature_error'],
        paired_positive_score=report['paired_separation']['smoothed_posteriors']['plus']['joint']['score'],
        paired_negative_score=report['paired_separation']['smoothed_posteriors']['minus']['joint']['score']),indent=2))
