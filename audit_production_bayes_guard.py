"""Guard semantics audit; not a fusion performance experiment."""
from pathlib import Path
import hashlib,json
import numpy as np
import independent_bayes_fusion as engine
from production_bayes_guard import guarded_decision

ROOT=Path(__file__).resolve().parent
class ForbiddenMoment:
    def __array__(self,*args,**kwargs):raise AssertionError('empty-mask path accessed moments')


def main():
    checks=[];rng=np.random.default_rng(317);quality=np.array([.5,1.,2.,3.,4.])
    empty={'ids':np.array([],dtype=int),'S':ForbiddenMoment()}
    for mode in engine.MODES+('diagonal_posterior_gate',):
        r=guarded_decision(empty,quality,engine.BASE,mode,1.);assert not r['action'] and r['abstained'] and r['weights']==[]
        checks.append(dict(kind='empty_mask',mode=mode,passed=True))
    max_difference=0.;comparisons=0
    for n in (1,2,3,4,5):
        for generation in range(4):
            a=rng.normal(size=(n,n));S=a@a.T;b=rng.normal(size=(n,n));U=b@b.T/20
            d=dict(ids=np.arange(n),N=128.,h=rng.uniform(-.3,.3,n),mu=rng.uniform(-.1,.1,n),S=S,Us=U,U=U,B=U*5,truegross=1e15)
            for mode in engine.MODES+('diagonal_posterior_gate',):
                actual=guarded_decision(d,quality,engine.BASE,mode,.7)
                expected_mode='joint' if mode=='diagonal_posterior_gate' else mode
                w,cert=engine.convex_fuse(quality[:n],S,U,engine.BASE,expected_mode)
                gain=float(d['N']*w@(d['h']-d['mu']));variance=d['B'] if mode=='frequentist_gate' else U
                if mode=='diagonal_posterior_gate':variance=np.diag(np.diag(variance))
                sd=d['N']*np.sqrt(max(0.,float(w@variance@w))+1e-4)
                max_difference=max(max_difference,float(np.max(np.abs(np.array(actual['weights'])-w))),abs(actual['gain']-gain),abs(actual['sd']-sd));comparisons+=1
                if n==1:assert actual['weights']==[1.]
                changed=dict(d,truegross=-1e15,localnet=-1e12)
                assert guarded_decision(changed,quality,engine.BASE,mode,.7)==actual,'future target influenced action'
                infeasible=guarded_decision(d,quality,engine.BASE,mode,.7,feasible=False);assert not infeasible['action']
            checks.append(dict(kind='mask_size_and_future_isolation',sources=n,generation=generation,passed=True))
    assert max_difference<1e-10
    invalid=[{'ids':np.array([0,0])},{'ids':np.array([-1])},{'ids':np.array([5])},{'ids':np.array([[0]])}]
    for d in invalid:
        try:guarded_decision(d,quality,engine.BASE,'bayes_both',1.)
        except ValueError:pass
        else:raise AssertionError('malformed mask was accepted')
    report=dict(kind='production guard and future-information isolation audit; not performance evidence',all_passed=True,
                checks=checks,nonempty_comparisons=comparisons,max_frozen_nonempty_difference=max_difference,
                malformed_masks_rejected=len(invalid),engine_sha=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
                wrapper_sha=hashlib.sha256((ROOT/'production_bayes_guard.py').read_bytes()).hexdigest(),
                audit_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'production_guard_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
