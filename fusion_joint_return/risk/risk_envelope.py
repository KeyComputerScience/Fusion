"""Time-uniform risk reports for predictable, paid lease admissions.

This module reports realized admission-prefix conditional risk. It does not
use the risk monitor as a promise about the next action under drift.
"""
import math
from dataclasses import dataclass,field

def conditional_risk_upper(total,n,error=.05,candidates=1,lambdas=(.25,.5,1.,2.,4.,8.)):
    if n==0:return 1.
    if not 0<=total<=n or not 0<error<1 or candidates<1:raise ValueError('Invalid risk report inputs')
    ls=tuple(float(x) for x in lambdas)
    if not ls or any(x<=0 for x in ls):raise ValueError('Positive predeclared lambda grid required')
    logpen=math.log(candidates*len(ls)/error)
    return min(1.,min((lam*total+logpen)/(-math.expm1(-lam)*n) for lam in ls))

def zero_failure_minimum(target=.1,error=.05,candidates=1,lambdas=(8.,)):
    logpen=math.log(candidates*len(lambdas)/error)
    return min(math.ceil(logpen/(-math.expm1(-lam)*target)) for lam in lambdas)

def chain_risk_upper(violations,admissions,max_leases,target=.1,error=.05,candidates=1):
    """Independent whole-chain calibration upper on E[V-target*N].

    Whole chains, not their requests or delay replicates, must be IID from
    the future environment sampling law for this reference to be valid.
    """
    if len(violations)!=len(admissions) or not violations:return None
    if any(not 0<=v<=a<=max_leases for v,a in zip(violations,admissions)):raise ValueError('Chain count support exceeded')
    m=len(violations)
    average=sum(v-target*a for v,a in zip(violations,admissions))/m
    upper=average+max_leases*math.sqrt(math.log(candidates/error)/(2*m))
    return dict(chains=m,mean_contrast=average,upper_contrast=upper,certified=upper<=0)

@dataclass
class AdmissionRiskMonitor:
    error:float=.05
    candidates:int=1
    lambdas:tuple=(.25,.5,1.,2.,4.,8.)
    issues:list=field(default_factory=list)
    callbacks:dict=field(default_factory=dict)
    processed:int=0
    totals:dict=field(default_factory=lambda:dict(violation=0.,harmful=0.,excess=0.,negative_loss=0.))

    def issue(self,origin,lower_score):
        if self.issues and origin<=self.issues[-1]['origin']:raise ValueError('Admission origins must increase')
        if not 0<float(lower_score)<=123.+1e-10:raise ValueError('Require paid positive score with declared support')
        self.issues.append(dict(origin=origin,lower_score=float(lower_score)))

    def callback(self,now,origin,maturity,complete_target,complete_return):
        if maturity>now:raise ValueError('Premature complete target')
        if origin not in {r['origin'] for r in self.issues}:raise ValueError('Only actual admitted origins are monitored')
        if origin in self.callbacks:raise ValueError('Duplicate callback')
        X=float(complete_target);D=float(complete_return)
        if X< -133.-1e-10 or D< -130.-1e-10:raise ValueError('Complete lease support exceeded')
        self.callbacks[origin]=(X,D)
        # Out-of-order arrivals cannot erase the holes in the complete
        # origin prefix used by the martingale theorem.
        while self.processed<len(self.issues):
            r=self.issues[self.processed];o=r['origin']
            if o not in self.callbacks:break
            X,D=self.callbacks[o];L=r['lower_score']
            self.totals['violation']+=float(X<L)
            self.totals['harmful']+=float(D<0)
            self.totals['excess']+=max(0.,L-X)/256.
            self.totals['negative_loss']+=max(0.,-D)/130.
            self.processed+=1
        return self.report()

    def report(self):
        # Allocate one overall error budget to all four reported endpoints.
        e=self.error/len(self.totals)
        bounds={key:conditional_risk_upper(v,self.processed,e,self.candidates,self.lambdas) for key,v in self.totals.items()}
        pending=len(self.issues)-self.processed
        all_prefix_bounds={key:min(1.,(value*self.processed+pending)/len(self.issues)) if self.issues else 1. for key,value in bounds.items()}
        return dict(issued=len(self.issues),completed_origin_prefix=self.processed,pending_origins=pending,totals=dict(self.totals),completed_prefix_average_conditional_risk_upper=bounds,issued_prefix_average_conditional_risk_upper=all_prefix_bounds)
