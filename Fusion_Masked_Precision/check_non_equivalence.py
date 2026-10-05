"""Mathematical fixtures only; no physical evaluation or selected algorithm."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def optimize_2(P, R, gap=0., q=.1, tau=.05, kappa=1., floor=.01, cap=.8):
    def cost(x):
        w = np.array([x,1-x])
        return float(-gap*x + tau*np.sum(w*np.log(w/.5))
                     + kappa*np.sqrt(w@P@w+floor**2)
                     + q*np.sqrt(w@R@w+floor**2))
    def derivative(x):
        w = np.array([x,1-x]); d=np.array([1.,-1.])
        return float(-gap + tau*np.log(x/(1-x))
                     + kappa*(d@P@w)/np.sqrt(w@P@w+floor**2)
                     + q*(d@R@w)/np.sqrt(w@R@w+floor**2))
    lo,hi=1-cap,cap
    if derivative(lo)>=0: x=lo
    elif derivative(hi)<=0: x=hi
    else:
        for _ in range(100):
            mid=(lo+hi)/2
            if derivative(mid)<0:lo=mid
            else:hi=mid
        x=(lo+hi)/2
    return dict(weights=[x,1-x],cost=cost(x),derivative=derivative(x),
                posterior_variance=float(np.array([x,1-x])@P@np.array([x,1-x])))

R=np.eye(2)
mask_a=np.diag([.1,.5]);mask_b=np.diag([.5,.1])
out=dict(kind="MATHEMATICAL_COUNTEREXAMPLES_NOT_PHYSICAL_RESULTS",
         scalar_non_equivalence=dict(R=R.tolist(),P_a=mask_a.tolist(),P_b=mask_b.tolist(),
                                     full_a=optimize_2(mask_a,R),full_b=optimize_2(mask_b,R),
                                     scalar_any_c_weight=[.5,.5],total_mass_each=10))
R=np.array([[1.,.8],[.8,1.]])
P=np.linalg.solve(np.eye(2)+np.linalg.solve(R,np.eye(2)),np.eye(2))
full=optimize_2(P,R,gap=.1);diag=optimize_2(np.diag(np.diag(P)),R,gap=.1)
out['diagonal_non_equivalence']=dict(R=R.tolist(),P=P.tolist(),full=full,diagonal=diag,
                                   common_mean_shift_for_gate_crossing=(full['cost']+diag['cost'])/2,
                                   robust_value_gap=full['cost']-diag['cost'])
scaled_R=.01*np.eye(2)
out['prior_sd_point_one_scalar_example']=dict(
    full_a=optimize_2(.01*mask_a,scaled_R),
    full_b=optimize_2(.01*mask_b,scaled_R),scalar_any_c_weight=[.5,.5])
scaled_full=optimize_2(.01*P,.01*R,gap=.01)
scaled_diag=optimize_2(.01*np.diag(np.diag(P)),.01*R,gap=.01)
out['prior_sd_point_one_diagonal_example']=dict(full=scaled_full,diagonal=scaled_diag,
    common_mean_shift_for_paid_gate_crossing=(scaled_full['cost']+scaled_diag['cost'])/2+5/128,
    net_gate_full=128*(scaled_diag['cost']-scaled_full['cost'])/2,
    net_gate_diagonal=128*(scaled_full['cost']-scaled_diag['cost'])/2)
assert out['scalar_non_equivalence']['full_a']['weights'][0]>.5
assert out['scalar_non_equivalence']['full_b']['weights'][0]<.5
assert full['weights'][0]>diag['weights'][0]>.5
assert full['cost']>diag['cost']
assert max(abs(full['derivative']),abs(diag['derivative']))<1e-12
assert scaled_full['weights'][0]>scaled_diag['weights'][0]>.5
assert out['prior_sd_point_one_scalar_example']['full_a']['weights'][0]>.5
(ROOT/'non_equivalence_examples.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
