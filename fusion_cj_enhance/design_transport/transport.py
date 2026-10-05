"""CJ-T: paired historical forecast--complete-target conditional law.

Raw working density in (H,U), bounded U only at prediction. No current
target labels enter state construction. Immutable source code is imported.
"""
from pathlib import Path
import importlib.util,math,sys
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
FROZEN=PROJECT/'work/fusion_conditional_20261004'
sp=importlib.util.spec_from_file_location('cjt_frozen_copula',FROZEN/'copula_control.py')
cp=importlib.util.module_from_spec(sp);sp.loader.exec_module(cp)
c=cp.c;core=c.core;matrix=c.parent.parent.matrix
NEW_ARMS=('t_joint','t_product','t_gaussian','t_factorized','t_diagonal_completion')
ARMS=('conditional_joint',)+NEW_ARMS+('legacy_factorized','legacy_sandwich')

def collect_pairs(arrived,current,pre,cfg):
    out=[]
    for r in arrived[-cfg['lease_archive']:]:
        assert r['maturity']<=current['k']
        a=cfg['beta']**(current['k']-r['k'])*math.exp(-float(np.sum((current['context']-r['context'])**2))/cfg['kernel_bandwidth']**2)
        if a<1e-12:continue
        pa=(r['x']@current['reference']).argmax(1);pc=(r['x']@current['candidate']).argmax(1)
        contrast=np.eye(pre['classes'])[pc]-np.eye(pre['classes'])[pa];ids=np.flatnonzero(r['mask'])
        H=np.einsum('sic,ic->s',r['p'][ids],contrast)/len(r['x'])
        gross=0.;count=0;total=0
        for future in r['future']:
            p0=(future['x']@current['reference']).argmax(1);p1=(future['x']@current['candidate']).argmax(1);keep=future['keep']
            gross+=float(np.sum((p1[keep]==future['y'][keep]).astype(float)-(p0[keep]==future['y'][keep]).astype(float)))
            count+=int(np.sum(p1!=p0));total+=len(p0)
        a*=count/max(1,total)
        if a<1e-12:continue
        U=gross/(cfg['horizon']*len(r['x']))
        assert abs(U)<=1.+1e-12
        out.append(dict(weight=a,ids=np.r_[ids,pre['m']],value=np.r_[H,U],origin=r['k'],maturity=r['maturity'],
                        forecast_ids=ids.tolist(),target=U))
    return out

def prior_covariance(pre,cfg):
    m=pre['m'];tv=1/3;one=np.ones(m+1)
    # Gaussian prior has the neutral target's second moment; its truncated
    # density is not a uniform target prior.
    R=tv*np.outer(one,one)
    R[np.arange(m),np.arange(m)]+=cfg['prior_variance']*np.mean(pre['q'])/pre['q']
    return R

def shared_law(obs,pre,ids,cfg,diagonal_completion=False):
    m=pre['m'];mass=cfg['prior_mass'];first=np.zeros(m+1);second=mass*prior_covariance(pre,cfg)
    full=[x for x in obs if len(x['ids'])==m+1]
    for x in full:
        mass+=x['weight'];first+=x['weight']*x['value'];second+=x['weight']*np.outer(x['value'],x['value'])
    mu=first/mass;R=second/mass-np.outer(mu,mu);R=(R+R.T)/2
    ev,U=np.linalg.eigh(R);R=(U*np.maximum(ev,1e-6))@U.T
    if diagonal_completion:R=np.diag(np.diag(R))
    locations=[np.zeros(m+1)];covs=[R.copy()];weights=[cfg['prior_mass']]
    for x in obs:
        ix=x['ids'];inverse=np.linalg.solve(R[np.ix_(ix,ix)],np.eye(len(ix)))
        loc=mu+R[:,ix]@inverse@(x['value']-mu[ix]);V=R-R[:,ix]@inverse@R[ix,:]
        # Observed H and U remain exact; only missing H has Schur uncertainty.
        loc[ix]=x['value'];V=(V+V.T)/2;V[ix,:]=0.;V[:,ix]=0.
        ev,UU=np.linalg.eigh(V);V=(UU*np.maximum(ev,0.))@UU.T
        locations.append(loc);covs.append(V);weights.append(x['weight'])
    active=np.r_[ids,m];locations=np.asarray(locations)[:,active];covs=np.asarray([V[np.ix_(active,active)] for V in covs])
    jitter=np.diag(np.r_[cfg['slice_bandwidth']**2*cfg['prior_variance']*np.mean(pre['q'])/pre['q'][ids],
                           cfg['slice_bandwidth']**2*cfg['norm_floor']**2])
    covs+=jitter;alpha=np.asarray(weights);alpha/=alpha.sum()
    mean=alpha@locations;center=locations-mean
    C=np.einsum('j,js,jt->st',alpha,center,center)+np.einsum('j,jst->st',alpha,covs)
    return dict(alpha=alpha,locations=locations,covariances=covs,mean=mean,C=C,
        raw_mass=float(sum(x['weight'] for x in obs)),prior_mass=cfg['prior_mass'],complete_blocks=len(full),
        target_kernel_sd=cfg['slice_bandwidth']*cfg['norm_floor'],global_mean=mu,global_covariance=R)

def conditional_posterior(h,alpha,locations,covariances):
    means=[];variances=[];evidence=[];m=len(h)
    for loc,S in zip(locations,covariances):
        HH=S[:m,:m];delta=h-loc[:m];solve=np.linalg.solve(HH,delta);cross=S[m,:m]
        mean=float(loc[m]+cross@solve);var=float(S[m,m]-cross@np.linalg.solve(HH,S[:m,m]))
        assert var>0.;_,ld=np.linalg.slogdet(HH)
        means.append(mean);variances.append(var);evidence.append(-.5*(m*c.LOG2PI+ld+float(delta@solve)))
    return c.truncated_components(alpha,np.asarray(means),np.asarray(variances),np.asarray(evidence))

def product_posterior(h,alpha,locations,covariances):
    """Exact pair-marginal control: f_U product f(H_s|U)."""
    m=len(h);target_mu=locations[:,-1];target_var=covariances[:,-1,-1]
    if m==1:return conditional_posterior(h,alpha,locations,covariances)
    def integral(bins):
        u,qw=c.quadrature(bins);delta=u[:,None]-target_mu[None,:]
        logU=c.logsumexp(np.log(alpha)[None,:]-.5*(c.LOG2PI+np.log(target_var)[None,:]+delta*delta/target_var[None,:]),axis=1)
        logdensity=-(m-1)*logU
        for s in range(m):
            a=covariances[:,s,s];b=covariances[:,s,-1];v=target_var;det=a*v-b*b
            assert np.all(det>0)
            dh=h[s]-locations[:,s]
            quadratic=(v[None,:]*dh[None,:]**2-2*b[None,:]*dh[None,:]*delta+a[None,:]*delta**2)/det[None,:]
            logdensity+=c.logsumexp(np.log(alpha)[None,:]-.5*(2*c.LOG2PI+np.log(det)[None,:]+quadratic),axis=1)
        lp=logdensity+np.log(qw);prob=np.exp(lp-c.logsumexp(lp));mean=float(prob@u);var=float(prob@(u-mean)**2)
        return mean,var
    conditional_vars=[]
    for S in covariances:
        conditional_vars.append(float(S[-1,-1]-S[-1,:m]@np.linalg.solve(S[:m,:m],S[:m,-1])))
    minsd=math.sqrt(min(conditional_vars));bins=max(32,int(math.ceil(.25*math.sqrt(m)/minsd)))
    previous=integral(bins);err=1.
    for step in range(6):
        bins*=2;current=integral(bins);err=max(abs(current[0]-previous[0]),abs(current[1]-previous[1]))
        if err<1e-9:break
        previous=current
    assert err<1e-7,('target-pair product quadrature',err,bins)
    return dict(mean=current[0],variance=current[1],component_count=len(alpha),posterior_effective_components=None,
        quadrature_error=err,quadrature_nodes=8*bins,quadrature_refinements=step+1)

def posteriors(h,law,diagonal_law):
    a,l,V=law['alpha'],law['locations'],law['covariances']
    da,dl,dV=diagonal_law['alpha'],diagonal_law['locations'],diagonal_law['covariances']
    return dict(t_joint=conditional_posterior(h,a,l,V),t_product=product_posterior(h,a,l,V),
        t_gaussian=conditional_posterior(h,np.ones(1),law['mean'][None],law['C'][None]),
        t_factorized=c.truncated_components(a,l[:,-1],V[:,-1,-1],np.zeros(len(a))),
        t_diagonal_completion=conditional_posterior(h,da,dl,dV))

def state(arrived,current,pre,cfg):
    mm=cp.state(arrived,current,pre,cfg)
    obs=collect_pairs(arrived,current,pre,cfg);law=shared_law(obs,pre,current['ids'],cfg)
    diagonal=shared_law(obs,pre,current['ids'],cfg,True)
    mm.update(transport=posteriors(current['h'],law,diagonal),transport_ready=bool(obs),transport_law=law,
        transport_eligible=[dict(origin=x['origin'],maturity=x['maturity'],weight=x['weight'],forecast_ids=x['forecast_ids'],target_observed=True) for x in obs],
        transport_pair_count=len(obs),transport_raw_mass=law['raw_mass'],transport_complete_blocks=law['complete_blocks'])
    return mm

def forecast(d,pre,cfg,arm,q):
    if arm=='legacy_factorized':return c.parent.parent.factorized_forecast(d,pre,cfg,'factorized_information',q)
    if arm=='legacy_sandwich':return matrix.matrix_forecast(d,pre,cfg,'block_sandwich',q)
    if arm not in NEW_ARMS:return c.conditional_forecast(d,pre,cfg,arm,q)
    mm=d['temporal'];p=mm['transport'][arm];ready=mm['transport_ready']
    F=float(d['N']*p['mean']) if ready else 0.;S=float(d['N']*math.sqrt(p['variance']+cfg['norm_floor']**2))
    quality=pre['q'][d['ids']];weights=core.capped(quality/quality.sum(),max(cfg['cap'],1/len(quality)))
    return weights,dict(converged=True,kkt=0.,primal=0.,iterations=0),F,S,(F-d['truegross'])/S,dict(
        information_ready=ready,raw_optimized_gain=float(d['N']*p['mean']),posterior_norm=0.,entropy=0.,
        conditional_mean=p['mean'],conditional_variance=p['variance'],quadrature_error=p['quadrature_error'],
        posterior_effective_components=p['posterior_effective_components'],transport_pair_count=mm['transport_pair_count'],
        transport_raw_mass=mm['transport_raw_mass'],transport_complete_blocks=mm['transport_complete_blocks'],
        transport_eligible=mm['transport_eligible'],source_weights_redundant=True,
        target_kernel_sd=mm['transport_law']['target_kernel_sd'],joint_target_conditioning=True)

core.state=state;core.forecast=forecast
def configure(cfg,arm):return dict(cfg)
def binding():return c.parent.binding()
