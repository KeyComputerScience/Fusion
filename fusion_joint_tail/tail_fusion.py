"""Conditional joint complete-return tail functional. NumPy, no future labels.

The lower-tail functional is standard expected shortfall. The retained joint
error law and its decision-line conditioning supply the information. A law
error radius is an explicit input, never silently inferred from the posterior.
"""
import math
import numpy as np

LOG2PI = math.log(2 * math.pi)


def cdf(x):
    return .5 * math.erfc(-float(x) / math.sqrt(2))


def pdf(x):
    return math.exp(-.5 * float(x) ** 2) / math.sqrt(2 * math.pi)


def interval_mass(a, b):
    return cdf(b)-cdf(a) if a < 0 else cdf(-a)-cdf(-b)


class DiscreteLaw:
    def __init__(self, values, weights):
        values, weights = np.asarray(values, float), np.asarray(weights, float)
        assert values.ndim == weights.ndim == 1 and len(values) == len(weights)
        assert np.isfinite(values).all() and np.isfinite(weights).all()
        assert (weights >= 0).all() and weights.sum() > 0
        order = np.argsort(values, kind='stable')
        self.values, self.weights = values[order], weights[order]/weights.sum()
        self.mean = float(self.weights @ self.values)
        self.variance = float(self.weights @ (self.values-self.mean)**2)

    def lower_tail(self, alpha):
        assert 0 < alpha <= 1
        allocation = np.clip(alpha-np.r_[0., np.cumsum(self.weights)[:-1]], 0., self.weights)
        return float(allocation @ self.values / alpha)


class TruncatedMixture:
    """Scalar Gaussian mixture with component-wise truncation then reweighting."""
    def __init__(self, means, variances, log_masses, bounds=(-1., 1.)):
        self.lo, self.hi = map(float, bounds)
        assert self.lo < self.hi
        keep=[]
        for m,v,l in zip(means,variances,log_masses):
            assert np.isfinite([m,v,l]).all() and v > 0
            sd=math.sqrt(v); a=(self.lo-m)/sd; b=(self.hi-m)/sd
            mass=interval_mass(a,b)
            if mass > 1e-280:keep.append((float(m),sd,float(l)+math.log(mass),a,b,mass))
        if not keep:raise ValueError('No representable bounded line mass')
        a=np.asarray(keep);self.means=a[:,0];self.sd=a[:,1]
        log=a[:,2];self.weights=np.exp(log-log.max());self.weights/=self.weights.sum()
        self.a=a[:,3];self.b=a[:,4];self.mass=a[:,5]
        component_mean=self.means+self.sd*np.asarray([(pdf(x)-pdf(y))/z for x,y,z in zip(self.a,self.b,self.mass)])
        component_variance=self.sd**2*np.asarray([1+(x*pdf(x)-y*pdf(y))/z-((pdf(x)-pdf(y))/z)**2 for x,y,z in zip(self.a,self.b,self.mass)])
        self.mean=float(self.weights@component_mean)
        self.variance=max(0.,float(self.weights@(np.maximum(0.,component_variance)+component_mean**2))-self.mean**2)

    def partial(self, x):
        """Return P(U<=x), E[U 1(U<=x)] under the normalized fitted law."""
        if x <= self.lo:return 0.,0.
        if x >= self.hi:return 1.,self.mean
        z=(x-self.means)/self.sd
        mass=np.asarray([max(0.,interval_mass(a,t))/b for a,t,b in zip(self.a,z,self.mass)])
        first=self.means*mass+self.sd*np.asarray([(pdf(a)-pdf(t))/b for a,t,b in zip(self.a,z,self.mass)])
        return float(self.weights@mass),float(self.weights@first)

    def lower_tail(self, alpha):
        assert 0 < alpha <= 1
        if alpha == 1:return self.mean
        left,right=self.lo,self.hi
        for _ in range(55):
            middle=(left+right)/2
            if self.partial(middle)[0] < alpha:left=middle
            else:right=middle
        p,m=self.partial((left+right)/2)
        assert abs(p-alpha) < 1e-8
        return float(m/alpha)


def joint_line(h, weights, locations, covariances, bounds=(-1.,1.)):
    """Exact Gaussian component line integrals including determinant factors."""
    h=np.asarray(h,float);weights=np.asarray(weights,float)
    locations=np.asarray(locations,float);covariances=np.asarray(covariances,float)
    assert h.ndim==1 and locations.shape==(len(weights),len(h))
    assert covariances.shape==(len(weights),len(h),len(h))
    assert (weights>0).all() and np.isfinite(h).all()
    one=np.ones(len(h));means=[];variances=[];logs=[]
    for weight,location,covariance in zip(weights,locations,covariances):
        np.linalg.cholesky(covariance)
        precision=np.linalg.solve(covariance,np.eye(len(h)))
        a=float(one@precision@one);r=h-location;b=float(one@precision@r)
        variance=1/a;mean=b/a;_,ld=np.linalg.slogdet(covariance)
        log=math.log(weight)-.5*(len(h)*LOG2PI+ld+float(r@precision@r)-b*b/a)+.5*math.log(2*math.pi/a)
        means.append(mean);variances.append(variance);logs.append(log)
    return TruncatedMixture(means,variances,logs,bounds)


def paid_score(law, capacity, alpha, cost=5., w1_radius=0.):
    """W1 radius is in NORMALIZED target units, cost in service utility units."""
    assert capacity > 0 and cost >= 0 and w1_radius >= 0
    tail=law.lower_tail(alpha)
    return dict(score=float(capacity*(tail-w1_radius/alpha)-cost),
                lower_tail=float(tail),alpha=float(alpha),radius=float(w1_radius),
                mean=float(capacity*law.mean-cost),variance=float(capacity**2*law.variance))
