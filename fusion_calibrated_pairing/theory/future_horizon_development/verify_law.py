"""Pure synthetic preflight; no physical efficacy or calibration claim."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'design'))
from verify_descriptor import event
import horizon_law as law
from horizon_descriptor import HorizonDescriptor


def run():
    path=HorizonDescriptor();events=[event() for _ in range(8)]
    for i,e in enumerate(events):
        if i%2:e['p']=e['p'][::-1].copy()
    base=dict(windows=8,decisions=[dict(k=0,maturity=4,N=128,truegross=16.,localnet=11.),
                                 dict(k=4,maturity=8,N=128,truegross=-8.,localnet=-13.)])
    pre=dict(m=2,q=np.asarray([1.,2.]));stream=law.build_stream(events,pre,base,{},recording='toy')
    prior=law.library(stream)
    assert len(prior)==2
    # JSON roundtrip must work for a persistent immutable tuple library.
    def serial(x):return x.tolist() if isinstance(x,np.ndarray) else x
    loaded=json.loads(json.dumps(prior,default=serial))
    stream=law.build_stream(events,pre,base,{},loaded,recording='toy-eval')
    outputs={arm:law.raw_run(stream,arm) for arm in law.ARMS}
    for arm,r in outputs.items():
        assert all(-133<=x['gate_score']<=123 for x in r['rows'])
        assert all(x['information_ready'] for x in r['rows'])
        assert all(x['quadrature_converged'] for x in r['law_diagnostics'])
    # Conditional block quadratic agrees with direct full Gaussian density.
    mu=np.array([.1,.2,-.1]);C=np.array([[.3,.02,-.04],[.02,.1,.01],[-.04,.01,.2]])
    query=np.array([.4,.1]);a,b,c=law.scalar_log_terms(mu,C,query)
    for e in (-.8,0.,.6):
        z=np.r_[e,query]-mu
        expected=-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(C)[1]+z@np.linalg.solve(C,z))
        assert abs(a*e*e+b*e+c-expected)<1e-12
    # No later immature complete target may affect a current fitted law.
    poisoned=copy.deepcopy(base);poisoned['decisions'][1]['truegross']=99.
    newer=law.build_stream(events,pre,poisoned,{},loaded,recording='toy-eval')
    for arm in law.ARMS:
        r=law.raw_run(newer,arm)
        assert r['rows'][0]['gate_score']==outputs[arm]['rows'][0]['gate_score']
        assert r['rows'][1]['gate_score']==outputs[arm]['rows'][1]['gate_score']
    report=dict(status='synthetic horizon-law preflight, not physical results',
                persistent_json_roundtrip=True,conditional_multivariate_block_quadratic=True,
                target_maturity_integrity=True,physical_support_all_arms=True,
                factor_blocks_per_physical_source=True,
                raw_scores={arm:[r['gate_score'] for r in result['rows']] for arm,result in outputs.items()})
    (ROOT/'law_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':run()
