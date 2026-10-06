"""Exact conditional-tail witness; analytical law values, not measurements."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def lower_tail(law,alpha):
    remaining=alpha;total=F(0)
    for u,p in sorted(law):
        take=min(remaining,p);total+=take*u;remaining-=take
        if remaining==0:break
    assert remaining==0
    return total/alpha
def record(law,N=128,cost=5,alpha=F(1,2)):
    assert sum(p for u,p in law)==1
    mean=sum(u*p for u,p in law);second=sum(u*u*p for u,p in law)
    lt=lower_tail(law,alpha)
    return dict(atoms=[dict(U=str(u),probability=str(p)) for u,p in law],mean=str(mean),second_moment=str(second),variance=str(second-mean*mean),lower_tail=str(lt),paid_lower_tail_score=str(N*lt-cost),expected_paid_return=str(N*mean-cost),raw_tail_feasible=N*lt-cost>=0,positive_tail_admission=N*lt-cost>0)
def main():
    P=[(F(0),F(1,2)),(F(1,5),F(1,2))]
    Q=[(-F(1,5),F(1,10)),(F(2,15),F(9,10))]
    p,q=record(P),record(Q)
    assert p['mean']==q['mean']=='1/10'
    assert p['variance']==q['variance']=='1/100'
    assert p['second_moment']==q['second_moment']=='1/50'
    assert p['lower_tail']=='0' and q['lower_tail']=='1/15'
    assert p['paid_lower_tail_score']=='-5' and q['paid_lower_tail_score']=='53/15'
    assert p['expected_paid_return']==q['expected_paid_return']=='39/5'
    assert not p['positive_tail_admission'] and q['positive_tail_admission']
    out=dict(role='Exact analytical conditional-return scoring witness; no physical measurements or frozen algorithm changes',support=[-1,1],N=128,conservative_cost=5,alpha='1/2',common_score_correction='0',P=p,Q=q,
        equal_conditional_mean_variance=True,opposite_prescribed_tail_actions=True,
        summary_feasibility_scope='A moment-only decision receives identical current context and conditional mean/variance. Requiring nonnegative true lower-tail paid score for every compatible law excludes admission because P is compatible and its score is -5.',
        missed_Q_expected_return='39/5',
        no_unconstrained_mean_return_dominance=True,
        scope='Atoms give fractional gross returns at N128. This is a bounded-law/scoring functional witness, not an integer-correctness-count construction, physical performance evidence, or empirical calibration claim.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=ROOT/'conditional_tail_witness.json';path.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
