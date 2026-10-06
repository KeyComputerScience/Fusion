"""Exact analytical information witness; this is not physical experiment data."""
from fractions import Fraction as F
from itertools import product,combinations
from collections import defaultdict
from pathlib import Path
import json,math
SIGNS=[s for s in product((-1,1),repeat=3) if len(set(s))>1]
ANCHOR=F(1,20);U0=F(1,8);H0=F(1,10);N=128

def law(direction):
 rows=[]
 for s in SIGNS:
  g=s[0]*s[1]-s[0]*s[2]
  pp=F(1,2)+direction*F(g,4)
  for sign,p in ((1,pp),(-1,1-pp)):
   if p:
    u=sign*U0;h=tuple(H0*v for v in s)
    rows.append(dict(S=s,U=u,H=h,R=u-ANCHOR,D=tuple(v-ANCHOR for v in h),p=p/6,X=N*u-5))
 return rows

def marginal(rows,ids):
 out=defaultdict(F)
 for r in rows:
  state=(r['R'],)+r['D'];out[tuple(state[i] for i in ids)]+=r['p']
 return dict(out)

A=law(1);B=law(-1)
checks=[]
for n in (1,2):
 for ids in combinations(range(4),n):
  assert marginal(A,ids)==marginal(B,ids)
  checks.append(list(ids))
assert sum(r['p'] for r in A)==sum(r['p'] for r in B)==1

def moments(rows):
 x=[(r['R'],)+r['D'] for r in rows]
 mu=[sum(r['p']*v[i] for r,v in zip(rows,x)) for i in range(4)]
 cov=[[sum(r['p']*(v[i]-mu[i])*(v[j]-mu[j]) for r,v in zip(rows,x)) for j in range(4)] for i in range(4)]
 return mu,cov
assert moments(A)==moments(B)

def low_tail(values,alpha):
 rem=alpha;v=F(0)
 for x,p in sorted(values):
  take=min(rem,p);v+=take*x;rem-=take
  if rem==0:break
 assert rem==0
 return v/alpha

scores={};returns={}
for name,rows in [('A',A),('B',B)]:
 value=F(0);ss=[]
 for s in SIGNS:
  conditional=[(r['U'],r['p']*6) for r in rows if r['S']==s]
  assert sum(p for _,p in conditional)==1
  L=N*low_tail(conditional,F(1,2))-5
  meanD=sum((N*u-5)*p for u,p in conditional)
  if L>0:value+=meanD/6
  ss.append(dict(signs=s,score=float(L),actual_conditional_mean=float(meanD)))
 scores[name]=ss;returns[name]=float(value)
 assert value==F(11,3)
for s in SIGNS:
 ea=sum(r['p']*6*r['X'] for r in A if r['S']==s)
 eb=sum(r['p']*6*r['X'] for r in B if r['S']==s)
 assert (ea+eb)/2==-5
result=dict(role='Exact analytical law separation, not real-source data or an executed learned-model performance result',sources=3,source_sign_contexts=len(SIGNS),common_calibration_context=[float(ANCHOR),1.,1.],all_state_univariate_and_bivariate_margins_identical=True,full_state_mean_covariance_identical=True,exact_margin_checks=checks,joint_expected_paid_return=returns,summary_factor_paid_lower_tail=-21.,global_gaussian_paid_lower_tail=-5-128*.125*math.sqrt(2/math.pi),minimax_summary_only_regret_lower_bound=float(F(11,3)),joint_action_scores=scores,scope='Identical population global summaries plus same current descriptor; excludes full-tuple adaptive localization and moments fitted after full joint conditioning',service='one common feasible lease with empty ledger and B>=130; X=D (interruption slack0)')
path=Path(__file__).with_name('context_information_witness.json');path.write_text(json.dumps(result,indent=2)+'\n')
print('Exact witness passed:',len(checks),'summary matches; value/regret=11/3; not physical evidence.')
