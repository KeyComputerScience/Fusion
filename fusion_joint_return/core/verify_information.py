"""Exact finite paired-return witness; analytical construction, not physical data."""
from pathlib import Path
import itertools,json
import numpy as np

def distribution(values):
    out={}
    for v in values:
        key=tuple(round(float(x),12) for x in np.atleast_1d(v))
        out[key]=out.get(key,0)+1/len(values)
    return out

H=.1*np.array(list(itertools.product((-1,1),repeat=3)),float)
U=.125*np.prod(H/.1,axis=1)
A=np.column_stack((U,H));B=np.column_stack((-U,H))
all_pairs={}
for i,j in itertools.combinations(range(4),2):
    same=distribution(A[:,[i,j]])==distribution(B[:,[i,j]])
    assert same;all_pairs[f'{i},{j}']=same
assert np.max(abs(A.mean(0)-B.mean(0)))<1e-15
assert np.max(abs(np.cov(A,rowvar=False,bias=True)-np.cov(B,rowvar=False,bias=True)))<1e-15
ZA=H-U[:,None];ZB=H+U[:,None]
for i,j in itertools.combinations(range(3),2):assert distribution(ZA[:,[i,j]])==distribution(ZB[:,[i,j]])
N=128;cost=5;positive=N*.125-cost;negative=-N*.125-cost
assert positive==11 and negative==-21
report=dict(status='Exact analytical law witness, NOT measured physical experiment',sources=3,capacity=N,conservative_cost=cost,
    laws='H iid±.1; modelA U=.125prod(signH); modelB U=−.125prod(signH)',
    identical_target_source_pair_marginals=all_pairs,identical_source_error_pair_marginals=True,
    full_joint_mean_covariance_error=float(np.max(abs(np.cov(A,rowvar=False,bias=True)-np.cov(B,rowvar=False,bias=True)))),
    query=[.1,.1,.1],conditional_U_A=.125,conditional_U_B=-.125,
    joint_score_A=positive,joint_score_B=negative,
    factorized_score_alpha_point1=negative,factorized_score_alpha1=-cost,
    true_joint_policy_expected_paid_return=positive/2,reduced_tail_policy_expected_paid_return=0,
    proof_scope='Exact joint information vs allbivariate marginals and allfulljointmoments under correct laws; no assertion allnonBayesianfulljoint methods fail; Gaussianconditionaldensity andKDE estimation are mature tools')
Path(__file__).with_name('information_witness.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
