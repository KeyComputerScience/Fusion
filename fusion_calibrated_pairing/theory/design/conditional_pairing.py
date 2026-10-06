"""One prequential full-lease residual conditional law on common strong views.

The source probability views and fused anchor come from one fixed PDF
forecaster shared by every information control. No retrospective labels or
refitted historical model contrasts enter this law. A dedicated state-fit
library persists across physical chains; each chain adds only completely
matured immutable online tuples. Gaussian completion/smoothing are working
models, not coverage certificates.
"""
from pathlib import Path
import math,sys
sys.dont_write_bytecode=True
import numpy as np
PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'work/fusion_joint_tail_20261005'))
from tail_fusion import TruncatedMixture,DiscreteLaw
ARMS=('paired','pair_factorized','full_gaussian','joint_diagonal_kernel','unconditional')
RAW_ALPHA=.5

def psd(A,floor=1e-8):
    v,E=np.linalg.eigh((A+A.T)/2);return (E*np.maximum(v,floor))@E.T

def logsumexp(a,axis=None):
    a=np.asarray(a,float);m=np.max(a,axis=axis,keepdims=True)
    out=m+np.log(np.sum(np.exp(a-m),axis=axis,keepdims=True))
    return np.squeeze(out,axis=axis)

def quality(pre):return np.asarray(pre['q'],float)

def issued_descriptor(event,pre):
    """Only immutable probabilities/current candidate/reference, no outcomes."""
    pc=(event['x']@event['candidate']).argmax(1);pr=(event['x']@event['reference']).argmax(1)
    contrast=np.eye(pre['classes'])[pc]-np.eye(pre['classes'])[pr]
    ids=np.flatnonzero(event['mask'])
    h=np.mean(np.einsum('sic,ic->si',event['p'][ids],contrast),axis=1)
    anchor=float(np.mean(np.einsum('ic,ic->i',event['external_probability'],contrast))) if len(ids) else 0.
    support=float(np.mean(pc!=pr))
    return ids,h,anchor,support

def immutable_record(d,recording=None):
    return dict(origin=int(d['k']),maturity=int(d['maturity']),ids=np.asarray(d['ids'],int).copy(),
        residual=float(d['truegross']/d['N']-d['anchor']),descriptor=np.asarray(d['descriptor'],float).copy(),anchor=float(d['anchor']),
        support=float(d['disagreement']),context=np.asarray(d['cal_context'],float).copy(),recording=recording,
        target_mode='immutable_original_complete_lease',N=int(d['N']))

def prior_library(stream,complete_maturity_only=True):
    return [immutable_record(d,stream.get('recording')) for d in stream['decisions']
            if (not complete_maturity_only or d['maturity']<=stream['windows']) and d['disagreement']>0]

# Alias useful for physical runner.
library=prior_library

def state(prior_records,arrived,current,pre,cfg):
    m=pre['m'];dimension=m+1;q=quality(pre)
    prior=np.diag(np.r_[.01,.01*np.mean(q)/q])
    records=[]
    # The persistent library is one supervised reference population. It is
    # not assigned a fake local-origin age after every participant reset.
    # Each physical origin enters only once via a canonical delay schedule.
    context=np.asarray(current['cal_context'])
    for persistent,collection in ((True,prior_records),(False,arrived[-cfg.get('lease_archive',48):])):
        for row in collection:
            bw=cfg.get('state_context_bandwidth',1.)
            weight=math.exp(-float(np.sum((context-np.asarray(row['context']))**2))/bw**2)*row['support']
            if not persistent:weight*=cfg.get('beta',.97)**max(0,current['k']-row['origin'])
            if weight<1e-12:continue
            ix=np.r_[0,1+np.asarray(row['ids'],int)]
            value=np.r_[row['residual'],row['descriptor']]
            records.append(dict(weight=weight,ids=ix,value=value,persistent=persistent,origin=row['origin'],maturity=row['maturity']))
    complete=[r for r in records if len(r['ids'])==dimension]
    mass=cfg.get('prior_mass',2.)+sum(r['weight'] for r in complete)
    mu=sum((r['weight']*r['value'] for r in complete),np.zeros(dimension))/mass
    C=cfg.get('prior_mass',2.)*prior
    for row in complete:C+=row['weight']*np.outer(row['value'],row['value'])
    C=psd(C/mass-np.outer(mu,mu))
    loc=[np.zeros(dimension)];V=[prior.copy()];weights=[cfg.get('prior_mass',2.)]
    for row in records:
        ix=row['ids'];cross=C[:,ix];inv=np.linalg.solve(C[np.ix_(ix,ix)],np.eye(len(ix)))
        location=mu+cross@inv@(row['value']-mu[ix]);cov=psd(C-cross@inv@cross.T,0.)
        assert np.max(abs(location[ix]-row['value']))<1e-7
        loc.append(location);V.append(cov);weights.append(row['weight'])
    weights=np.asarray(weights);weights/=weights.sum();loc=np.asarray(loc);V=np.asarray(V)
    mean=weights@loc;moment=psd(np.einsum('j,js,jt->st',weights,loc-mean,loc-mean)+np.einsum('j,jst->st',weights,V))
    active=np.r_[0,1+current['ids']]
    return dict(eligible=bool(records) and len(current['ids'])>0,records=len(records),persistent_records=sum(r['persistent'] for r in records),online_records=sum(not r['persistent'] for r in records),
        weights=weights,locations=loc[:,active],covariances=V[:,active][:,:,active],moment=moment[np.ix_(active,active)],mean=mean[active],prior=prior[np.ix_(active,active)],
        complete_blocks=len(complete),partial_blocks=len(records)-len(complete),mass=mass,active_ids=current['ids'].tolist())

def build_stream(events,pre,base,cfg,prior_records=(),recording=None):
    pending=[];arrived=[];decisions=[]
    for d0 in base['decisions']:
        k=d0['k'];event=events[k]
        arrived.extend(row for row in pending if row['maturity']<=k);pending=[row for row in pending if row['maturity']>k]
        arrived=arrived[-cfg.get('lease_archive',48):]
        ids,h,anchor,support=issued_descriptor(event,pre)
        d=dict(d0,ids=ids,h=h,anchor=anchor,descriptor=h-anchor,disagreement=support,
            cal_context=np.asarray([anchor,support,len(ids)/pre['m']]),context=event['context'])
        d['temporal']=state(prior_records,arrived,d,pre,cfg)
        # Point forecasts were formed before reading the current future target.
        decisions.append(d);pending.append(immutable_record(d,recording))
    return dict(base,decisions=decisions,recording=recording)

def conditional_gaussian(h,mean,C):
    inverse=np.linalg.solve(C[1:,1:],np.eye(len(h)));cross=C[0,1:]
    mean=float(mean[0]+cross@inverse@(h-mean[1:]));variance=max(1e-12,float(C[0,0]-cross@inverse@cross))
    return mean,variance

def mixture_components(h,w,locations,covariances):
    means=[];variances=[];logs=[];m=len(h)
    for weight,loc,C in zip(w,locations,covariances):
        cm,cv=conditional_gaussian(h,loc,C);r=h-loc[1:];inv=np.linalg.solve(C[1:,1:],np.eye(m));ld=np.linalg.slogdet(C[1:,1:])[1]
        means.append(cm);variances.append(cv);logs.append(math.log(weight)-.5*(m*math.log(2*math.pi)+ld+float(r@inv@r)))
    return means,variances,logs

def quadrature(bins,bounds):
    nodes,weights=np.polynomial.legendre.leggauss(8)
    lo,hi=bounds;edges=np.linspace(lo,hi,bins+1);mid=(edges[1:]+edges[:-1])/2;half=(hi-lo)/(2*bins)
    return (mid[:,None]+half*nodes).ravel(),np.tile(half*weights,bins)

def make_law(d,arm,cfg):
    assert arm in ARMS;state=d['temporal'];w=state['weights'];loc=state['locations'];V=state['covariances'];m=len(d['ids'])
    neff=1/float(w@w);scott=neff**(-2/(m+5))
    K=scott*(state['moment']+state['prior'])
    if arm=='joint_diagonal_kernel':K=np.diag(np.diag(K))
    cov=np.asarray([psd(v+K) for v in V]);h=d['descriptor'];anchor=d['anchor'];bounds=(-1-anchor,1-anchor)
    if arm=='full_gaussian':
        cm,cv=conditional_gaussian(h,state['mean'],state['moment']+K)
        residual=TruncatedMixture([cm],[cv],[0.],bounds)
    elif arm in ('paired','joint_diagonal_kernel'):
        means,variances,logs=mixture_components(h,w,loc,cov)
        residual=TruncatedMixture(means,variances,logs,bounds)
    elif arm=='unconditional':
        residual=TruncatedMixture(loc[:,0],cov[:,0,0],np.log(w),bounds)
    else:
        # Strong reduction preserves exactly the same residual marginal and
        # EVERY (residual,physical-source descriptor) pair margin.
        def evaluate(bins):
            e,qw=quadrature(bins,bounds);log=np.zeros(len(e))
            for s in range(m):
                ids=[0,1+s];logs=[]
                for weight,location,C in zip(w,loc,cov):
                    CC=C[np.ix_(ids,ids)];inv=np.linalg.solve(CC,np.eye(2));delta=np.column_stack((e-location[0],np.repeat(h[s]-location[1+s],len(e))))
                    logs.append(math.log(weight)-math.log(2*math.pi)-.5*np.linalg.slogdet(CC)[1]-.5*np.einsum('is,st,it->i',delta,inv,delta))
                log+=logsumexp(np.asarray(logs).T,axis=1)
            margin=np.asarray([math.log(weight)-.5*(math.log(2*math.pi*C[0,0])+(e-location[0])**2/C[0,0]) for weight,location,C in zip(w,loc,cov)]).T
            log-=(m-1)*logsumexp(margin,axis=1);log+=np.log(qw)
            return DiscreteLaw(e,np.exp(log-logsumexp(log)))
        bins=max(32,int(math.ceil(1/math.sqrt(float(cov[:,0,0].min())))))
        old=evaluate(bins)
        for _ in range(6):
            bins*=2;new=evaluate(bins);error=max(abs(old.lower_tail(a)-new.lower_tail(a)) for a in (.1,.5,1.))
            if error<1e-6:break
            old=new
        residual=new;residual.quadrature_error=float(error);residual.quadrature_nodes=8*bins;residual.quadrature_converged=error<1e-6
    return ShiftedLaw(residual,anchor,neff,len(w),m)

class ShiftedLaw:
    def __init__(self,residual,anchor,neff,components,dimension):
        self.residual=residual;self.anchor=anchor;self.mean=residual.mean+anchor;self.variance=residual.variance
        self.neff=neff;self.component_count=components;self.active_dimension=dimension
        self.quadrature_error=getattr(residual,'quadrature_error',0.);self.quadrature_nodes=getattr(residual,'quadrature_nodes',0);self.quadrature_converged=getattr(residual,'quadrature_converged',True)
    def lower_tail(self,alpha):return self.anchor+self.residual.lower_tail(alpha)

def raw_run(stream,arm,cfg):
    rows=[];diagnostics=[]
    for d in stream['decisions']:
        law=make_law(d,arm,cfg);score=float(d['N']*law.lower_tail(RAW_ALPHA)-5.)
        ready=bool(d['temporal']['eligible']) and len(d['ids'])>0
        rows.append(dict(k=d['k'],action=bool(ready and score>0),maturity=d['maturity'],local_net=d['localnet'],truegross=d['truegross'],
            gate_score=score,raw_gate_score=score,gain=score+5.,posterior_or_block_sd=1.,q_issued=0.,standardized_score=score-(d['truegross']-5),
            lower_covered=bool(d['truegross']-5>=score),information_ready=ready,disagreement=d['disagreement'],anchor=d['anchor'],cal_context=d['cal_context'].tolist(),N=d['N'],
            source_descriptor=d['descriptor'].tolist(),active_source_ids=d['ids'].tolist(),conditional_paid_mean=d['N']*law.mean-5,conditional_paid_variance=d['N']**2*law.variance))
        diagnostics.append(dict(k=d['k'],quadrature_error=law.quadrature_error,quadrature_nodes=law.quadrature_nodes,quadrature_converged=law.quadrature_converged,neff=law.neff,components=law.component_count,persistent_records=d['temporal']['persistent_records'],online_records=d['temporal']['online_records']))
    return dict(rows=rows,law_diagnostics=diagnostics,raw_alpha=RAW_ALPHA)


def strong_prefix_quality(initial_models,audit_x,audit_y,pre,forward):
    """Quality only from a permanent model-excluded prefix error split.

    Caller must expose that fixed audit split; train rows are rejected as a
    substitute by protocol. All controls receive the returned shared pre.
    Preserve pre['q'] separately to avoid silently mixing linear/strong views.
    """
    assert len(audit_x)==len(audit_y)>0
    _,_,_,probabilities,_=forward([audit_x[:,ids] for ids in pre['features'][:-1]],initial_models)
    truth=np.eye(pre['classes'])[np.asarray(audit_y,int)]
    q=1/(np.mean(np.sum((probabilities-truth[None,:,:])**2,axis=2),axis=1)+.05)
    assert q.shape==(pre['m'],) and np.isfinite(q).all() and (q>0).all()
    return dict(pre,q=q,previous_linear_quality=np.asarray(pre['q']).copy(),quality_origin='fixed_model_excluded_prefix_error_split_of_common_nonlinear_PDF_sources')
