"""Strong non-Bayesian conditional-moment control.

It receives the paired conditional law's mean and variance AFTER full
joint-descriptor conditioning. It therefore retains the full joint
information needed for those conditional moments, unlike a global matrix.
Only conditional tail shape is reconstructed by bounded maximum entropy.
This comparator is not excluded by pair-summary information theorems.
"""
import numpy as np
from conditional_pairing import quadrature,logsumexp,DiscreteLaw,RAW_ALPHA

def fit_maxent(mean,variance,bins=128):
    if not -1<mean<1 or not 0<variance<1-mean*mean:raise ValueError('Moment pair outside strict bounded support interior')
    values,weights=quadrature(bins,(-1.,1.));X=np.column_stack((values,values**2));wanted=np.asarray([mean,variance+mean*mean])
    theta=np.asarray([mean/max(variance,1e-8),-.5/max(variance,1e-8)])
    def evaluate(t):
        log=X@t+np.log(weights);normalizer=float(logsumexp(log));p=np.exp(log-normalizer)
        moments=p@X;center=X-moments;H=np.einsum('i,ij,ik->jk',p,center,center)
        return normalizer-float(t@wanted),moments-wanted,H,p
    for iteration in range(100):
        value,gradient,H,p=evaluate(theta)
        if max(abs(gradient))<1e-11:break
        step=np.linalg.solve(H+1e-14*np.eye(2),gradient);rate=1.
        for _ in range(40):
            proposal=theta-rate*step
            if evaluate(proposal)[0]<=value-1e-4*rate*float(gradient@step)+1e-14:break
            rate*=.5
        theta=proposal
    _,gradient,_,p=evaluate(theta);error=float(max(abs(gradient)))
    if error>=1e-8:raise ValueError(f'Conditional maxentropy moment solver failed: mean={mean}, variance={variance}, error={error}')
    law=DiscreteLaw(values,p);law.moment_error=error;law.moment_nodes=len(values)
    return law

def raw_run(stream,paired_raw,cfg=None):
    rows=[];diagnostics=[]
    if len(stream['decisions'])!=len(paired_raw['rows']):raise ValueError('Unaligned paired conditional moments')
    for d,r in zip(stream['decisions'],paired_raw['rows']):
        if d['k']!=r['k']:raise ValueError('Unaligned conditional moment origin')
        mean=(r['conditional_paid_mean']+5.)/d['N'];variance=r['conditional_paid_variance']/d['N']**2
        law=fit_maxent(mean,variance)
        score=float(d['N']*law.lower_tail(RAW_ALPHA)-5.)
        row=dict(r,gate_score=score,raw_gate_score=score,gain=score+5.,action=bool(r['information_ready'] and score>0),lower_covered=bool(r['truegross']-5>=score),standardized_score=score-(r['truegross']-5),conditional_paid_mean=d['N']*law.mean-5.,conditional_paid_variance=d['N']**2*law.variance)
        rows.append(row);diagnostics.append(dict(k=d['k'],conditional_moment_error=law.moment_error,moment_nodes=law.moment_nodes))
    return dict(rows=rows,law_diagnostics=diagnostics,raw_alpha=RAW_ALPHA,control='Full paired conditional mean/variance followed by bounded maximum entropy')
