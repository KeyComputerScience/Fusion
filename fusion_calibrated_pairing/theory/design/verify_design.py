from pathlib import Path
import json,math
import conditional_pairing as cp
np=cp.np

# One analytically specified law. Covariance kernel is a fixed .01I for
# this information-level witness, not a measured physical experiment.
def witness():
    laws=[]
    for sign in (1.,-1.):
        locations=[]
        for a in (-1.,1.):
            for b in (-1.,1.):locations.append([sign*.45*a*b,.45*a,.45*b])
        locations=np.asarray(locations);state=dict(weights=np.repeat(.25,4),locations=locations,covariances=np.repeat(np.zeros((3,3))[None],4,axis=0),moment=np.diag([.2025]*3),mean=np.zeros(3),prior=np.eye(3)*.01)
        # Keep the same full-moment smoothing in every law; the witness uses
        # the production Scott bandwidth and reports its observed scores.
        d=dict(ids=np.array([0,1]),descriptor=np.array([.45,.45]),anchor=.1,temporal=state)
        laws.append({arm:cp.make_law(d,arm,{}).lower_tail(.5) for arm in cp.ARMS})
    assert abs(laws[0]['full_gaussian']-laws[1]['full_gaussian'])<1e-12
    assert abs(laws[0]['pair_factorized']-laws[1]['pair_factorized'])<1e-9
    assert laws[0]['paired']>laws[1]['paired']
    return laws

# Validate U+anchor support transformation and exact affine invariance.
def shifts():
    base=cp.TruncatedMixture([.05],[.03],[0.],(-1.3,.7))
    shifted=cp.ShiftedLaw(base,.3,1.,1,2)
    direct=cp.TruncatedMixture([.35],[.03],[0.],(-1.,1.))
    return max(abs(shifted.lower_tail(a)-direct.lower_tail(a)) for a in (.01,.1,.5,1.))

def causal_check():
    cfg=dict(prior_mass=2.,beta=.97,lease_archive=48);pre=dict(m=2,q=np.ones(2),classes=2)
    n=4;events=[]
    for k in range(8):
        x=np.c_[np.linspace(-.5,.5,n),np.ones(n)];candidate=np.array([[1.,-1.],[.1,-.1]]);reference=np.array([[-1.,1.],[.1,-.1]])
        p=np.asarray([np.tile([.8,.2],(n,1)),np.tile([.7,.3],(n,1))])
        events.append(dict(x=x,candidate=candidate,reference=reference,mask=np.array([True,k%3!=0]),p=p,external_probability=np.tile([.75,.25],(n,1)),context=np.zeros(2)))
    decisions=[dict(k=k,maturity=k+3,N=16,truegross=float(2*k-4),localnet=float(2*k-7),ids=np.flatnonzero(events[k]['mask'])) for k in range(4)]
    base=dict(decisions=decisions,windows=8);s=cp.build_stream(events,pre,base,cfg)
    poison=dict(base,decisions=[dict(d,truegross=1e6,localnet=-1e6) for d in decisions])
    poisoned=cp.build_stream(events,pre,poison,cfg)
    errors=[]
    # Prior-free early issued laws are identical until the changed target's
    # scheduled maturity. This tests state construction, not future outcome.
    for d,p in zip(s['decisions'][:3],poisoned['decisions'][:3]):
        errors.append(abs(cp.make_law(d,'paired',cfg).lower_tail(.5)-cp.make_law(p,'paired',cfg).lower_tail(.5)))
    assert max(errors)==0.
    return dict(early_scores_unchanged=max(errors)==0.,first_callback_origin=0,first_callback_maturity=3)

if __name__=='__main__':
    out=dict(analytical_information_witness=witness(),affine_tail_error=shifts(),causality=causal_check(),status='Numerical algebra checks; not measured physical deployment results')
    assert out['affine_tail_error']<1e-12
    (Path(__file__).resolve().parent/'mathematical_checks.json').write_text(json.dumps(out,indent=2))
    print(out)
