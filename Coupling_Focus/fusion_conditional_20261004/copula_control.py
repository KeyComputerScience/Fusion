"""Same mixture marginals with a retained adaptive Gaussian dependence matrix.

This is a deliberately strong designed matrix control, not a reproduction
of a published vine-copula system. Its matrix is the normalized covariance
of the same complete-error law used by the conditional joint method.
"""
from pathlib import Path
import importlib.util
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('conditional_copula_parent', ROOT/'run_conditional.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
ARM = 'conditional_copula'

def ndtr(x):
    """Vectorized normal CDF; classical rational approximation (<8e-8)."""
    x = np.asarray(x, dtype=float)
    z = np.abs(x)
    t = 1/(1+.2316419*z)
    poly = t*(.319381530+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))))
    tail = np.exp(-z*z/2)/math.sqrt(2*math.pi)*poly
    return np.where(x>=0, 1-tail, tail)

def ndtri(p):
    """Acklam inverse-normal rational approximation with bounded tails."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1-1e-12)
    a = [-3.969683028665376e1,2.209460984245205e2,-2.759285104469687e2,
         1.383577518672690e2,-3.066479806614716e1,2.506628277459239]
    b = [-5.447609879822406e1,1.615858368580409e2,-1.556989798598866e2,
         6.680131188771972e1,-1.328068155288572e1]
    cc = [-7.784894002430293e-3,-3.223964580411365e-1,-2.400758277161838,
          -2.549732539343734,4.374664141464968,2.938163982698783]
    d = [7.784695709041462e-3,3.224671290700398e-1,2.445134137142996,3.754408661907416]
    out = np.empty_like(p)
    low = p<.02425; high = p>1-.02425; mid = ~(low|high)
    q = np.sqrt(-2*np.log(np.where(low,p,1-p)))
    v = (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    out[low] = v[low]; out[high] = -v[high]
    q = p-.5; r = q*q
    v = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    out[mid] = v[mid]
    return out

def dependence(C):
    scale = np.sqrt(np.diag(C))
    corr = C/np.outer(scale,scale)
    ev,U = np.linalg.eigh((corr+corr.T)/2)
    corr = (U*np.maximum(ev,1e-6))@U.T
    sd = np.sqrt(np.diag(corr)); corr /= np.outer(sd,sd)
    return corr

def moments(h,alpha,locations,covariances,C):
    variances = np.diagonal(covariances,axis1=1,axis2=2)
    corr = dependence(C)
    inverse = np.linalg.solve(corr,np.eye(len(h)))
    _,ld = np.linalg.slogdet(corr)
    correction = inverse-np.eye(len(h))
    minsd = math.sqrt(float(1/np.sum(1/variances.min(0))))
    bins = max(32,int(math.ceil(1/minsd)))
    def integral(bins):
        target,qw = c.quadrature(bins)
        logdensity = np.zeros(len(target)); latent = []
        for s in range(len(h)):
            r = h[s]-target[:,None]-locations[None,:,s]
            sd = np.sqrt(variances[:,s])[None,:]
            log = np.log(alpha)[None,:]-.5*(c.LOG2PI+np.log(variances[:,s])[None,:]+r*r/variances[:,s][None,:])
            logdensity += c.logsumexp(log,axis=1)
            probability = ndtr(r/sd)@alpha
            latent.append(ndtri(probability))
        eta = np.asarray(latent).T
        logdensity += -.5*ld-.5*np.einsum('is,st,it->i',eta,correction,eta)
        lp = logdensity+np.log(qw)
        prob = np.exp(lp-c.logsumexp(lp))
        mean = float(prob@target)
        variance = float(prob@(target-mean)**2)
        return mean,variance
    previous = integral(bins); error = 1.
    for step in range(5):
        bins *= 2
        current = integral(bins)
        error = max(abs(current[0]-previous[0]),abs(current[1]-previous[1]))
        if error<1e-9: break
        previous = current
    return dict(mean=current[0],variance=current[1],component_count=len(alpha),
                posterior_effective_components=None,quadrature_error=error,
                quadrature_nodes=8*bins,quadrature_refinements=step+1,
                retained_dependence_matrix=corr.tolist(),cdf_clip=1e-12)

original_state = c.conditional_state

def state(arrived,current,pre,cfg):
    mm = original_state(arrived,current,pre,cfg)
    alpha,loc,V,_,C,_ = c.law(mm,pre,current['ids'],cfg)
    post = moments(current['h'],alpha,loc,V,C)
    if post['quadrature_error']>=1e-7:
        raise ValueError('Copula quadrature did not converge; preserve failure before any action.')
    mm['conditional'][ARM] = post
    return mm

c.core.state = state
# The original forecast obtains the posterior by dictionary key; no formula
# or callback accounting is changed for any original comparison arm.
c.core.forecast = c.conditional_forecast
