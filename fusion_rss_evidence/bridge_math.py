"""Standalone NumPy target-line bridge mathematics; no physical labels or imports."""
import math
import numpy as np
TAUS=(0.,.25,.5,.75,1.)
FAMILIES=('product_to_joint','product_to_joint_diagP','gaussian_to_joint')
LOG2PI=math.log(2*math.pi)
QUAD_X,QUAD_W=np.polynomial.legendre.leggauss(8)
def logsumexp(a,axis=None):
    high=np.max(a,axis=axis,keepdims=True)
    out=high+np.log(np.sum(np.exp(a-high),axis=axis,keepdims=True))
    return float(out.ravel()[0]) if axis is None else np.squeeze(out,axis=axis)
def quadrature(bins):
    edges=np.linspace(-1.,1.,bins+1);mid=.5*(edges[:-1]+edges[1:]);half=1/bins
    return (mid[:,None]+half*QUAD_X).ravel(),np.tile(half*QUAD_W,bins)
def normal_parameters(h,alpha,locations,covariances):
    m=len(h);one=np.ones(m);means=[];variances=[];logs=[]
    for a,loc,V in zip(alpha,locations,covariances):
        inv=np.linalg.solve(V,np.eye(m));var=1/float(one@inv@one);rr=h-loc
        mean=var*float(one@inv@rr);_,ld=np.linalg.slogdet(V)
        offset=float(rr@inv@rr)-mean*mean/var
        means.append(mean);variances.append(var);logs.append(math.log(float(a))-.5*(m*LOG2PI+ld+offset))
    return np.array(means),np.array(variances),np.array(logs)
def normal_logline(target,parameters):
    mu,var,logs=parameters
    return logsumexp(logs[None,:]-.5*(target[:,None]-mu[None,:])**2/var[None,:],axis=1)
def product_logline(target,h,alpha,locations,covariances):
    variances=np.diagonal(covariances,axis1=1,axis2=2);out=np.zeros(len(target))
    for s in range(len(h)):
        rr=h[s]-target[:,None]-locations[None,:,s]
        lp=np.log(alpha)[None,:]-.5*(LOG2PI+np.log(variances[:,s])[None,:]+rr*rr/variances[:,s][None,:])
        out+=logsumexp(lp,axis=1)
    return out
def integral(target,qw,logdensity):
    lp=logdensity+np.log(qw);evidence=logsumexp(lp);prob=np.exp(lp-evidence)
    mean=float(prob@target);var=float(prob@(target-mean)**2)
    return dict(mean=mean,variance=var,log_line_evidence=evidence)
def mix_logline(left,right,tau):
    if tau==0:return left
    if tau==1:return right
    return np.logaddexp(math.log1p(-tau)+left,math.log(tau)+right)
def bridge_state(h,a,l,V,mu,C,P):
    """All 15 posteriors, line evidences and checked numerical identities."""
    diagonalP=V-P+np.diag(np.diag(P))
    params={
        'joint':normal_parameters(h,a,l,V),
        'joint_diagP':normal_parameters(h,a,l,diagonalP),
        'gaussian':normal_parameters(h,np.ones(1),mu[None],C[None])}
    variances=np.diagonal(V,axis1=1,axis2=2)
    minsd=math.sqrt(float(1/np.sum(1/variances.min(0))));bins=max(32,int(math.ceil(1/minsd)))
    def evaluate(bins):
        target,qw=quadrature(bins)
        logs={key:normal_logline(target,value) for key,value in params.items()}
        logs['product']=product_logline(target,h,a,l,V)
        endpoints={name:integral(target,qw,density) for name,density in logs.items()}
        post={}
        for family,left,right in (('product_to_joint','product','joint'),
                                  ('product_to_joint_diagP','product','joint_diagP'),
                                  ('gaussian_to_joint','gaussian','joint')):
            for tau in TAUS:
                p=integral(target,qw,mix_logline(logs[left],logs[right],tau))
                eff=0. if tau==0 else (1. if tau==1 else math.exp(math.log(tau)+endpoints[right]['log_line_evidence']-p['log_line_evidence']))
                mm=(1-eff)*endpoints[left]['mean']+eff*endpoints[right]['mean']
                vv=(1-eff)*(endpoints[left]['variance']+endpoints[left]['mean']**2)+eff*(endpoints[right]['variance']+endpoints[right]['mean']**2)-mm**2
                p.update(effective_joint_posterior_mass=eff,evidence_ratio_log=endpoints[right]['log_line_evidence']-endpoints[left]['log_line_evidence'],
                         mixture_moment_identity_error=max(abs(mm-p['mean']),abs(vv-p['variance'])))
                post[(family,tau)]=p
        return post,endpoints
    previous,_=evaluate(bins);err=1.
    for step in range(5):
        bins*=2;post,endpoints=evaluate(bins)
        err=max(abs(post[key][field]-previous[key][field]) for key in post for field in ('mean','variance','log_line_evidence'))
        if err<1e-9:break
        previous=post
    assert err<1e-7,err
    meanlaw=a@l;center=l-meanlaw
    covlaw=np.einsum('j,js,jt->st',a,center,center)+np.einsum('j,jst->st',a,V)
    meanerr=float(np.max(np.abs(meanlaw-mu)));coverr=float(np.max(np.abs(covlaw-C)))
    coorderr=float(np.max(np.abs(np.diagonal(diagonalP,axis1=1,axis2=2)-np.diagonal(V,axis1=1,axis2=2))))
    gaussian_bridge_errors=[];marginal_bridge_errors=[]
    for tau in TAUS:
        bm=(1-tau)*mu+tau*meanlaw
        bc=(1-tau)*(C+np.outer(mu,mu))+tau*(covlaw+np.outer(meanlaw,meanlaw))-np.outer(bm,bm)
        gaussian_bridge_errors.append(max(float(np.max(np.abs(bm-mu))),float(np.max(np.abs(bc-C)))))
        product_cov=np.diag(np.diag(C));prod_to_joint_cov=(1-tau)*product_cov+tau*covlaw
        marginal_bridge_errors.append(max(float(np.max(np.abs(bm-mu))),float(np.max(np.abs(np.diag(prod_to_joint_cov)-np.diag(C))))))
    assert max(meanerr,coverr,coorderr,max(gaussian_bridge_errors),max(marginal_bridge_errors))<1e-12
    return post,dict(quadrature_error=err,quadrature_nodes=8*bins,quadrature_refinements=step+1,
        joint_mean_identity_error=meanerr,joint_covariance_identity_error=coverr,
        diagP_coordinate_variance_identity_error=coorderr,
        gaussian_bridge_same_mean_covariance_error=max(gaussian_bridge_errors),
        product_bridge_same_coordinate_moments_error=max(marginal_bridge_errors),
        mixture_moment_identity_error=max(p['mixture_moment_identity_error'] for p in post.values()),endpoints=endpoints)

