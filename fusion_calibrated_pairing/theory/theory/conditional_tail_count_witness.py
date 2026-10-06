"""Exact integer-count conditional-tail witness; not physical measurements."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def lower_tail(law,alpha):
    remaining=alpha;total=F(0)
    for G,p in sorted(law):
        take=min(remaining,p);total+=take*G;remaining-=take
        if remaining==0:break
    assert remaining==0
    return total/alpha
def record(law,N=128,cost=5,alpha=F(1,2)):
    assert sum(p for G,p in law)==1
    assert all(G.denominator==1 and -127<=G<=127 for G,p in law)
    mean=sum(G*p for G,p in law);second=sum(G*G*p for G,p in law)
    lt=lower_tail(law,alpha)
    return dict(atoms=[dict(G=int(G),U=str(G/N),probability=str(p),paid_return=int(G-cost)) for G,p in law],gross_mean=str(mean),gross_second_moment=str(second),gross_variance=str(second-mean*mean),U_mean=str(mean/N),U_variance=str((second-mean*mean)/(N*N)),gross_lower_tail=str(lt),U_lower_tail=str(lt/N),paid_lower_tail_score=str(lt-cost),expected_paid_return=str(mean-cost),raw_tail_feasible=lt-cost>=0,positive_tail_admission=lt-cost>0)
def main():
    P=[(F(0),F(1,2)),(F(16),F(1,2))]
    Q=[(F(-24),F(1,17)),(F(10),F(16,17))]
    p,q=record(P),record(Q)
    assert p['gross_mean']==q['gross_mean']=='8'
    assert p['gross_variance']==q['gross_variance']=='64'
    assert p['gross_second_moment']==q['gross_second_moment']=='128'
    assert p['U_mean']==q['U_mean']=='1/16'
    assert p['U_variance']==q['U_variance']=='1/256'
    assert p['gross_lower_tail']=='0' and q['gross_lower_tail']=='6'
    assert p['paid_lower_tail_score']=='-5' and q['paid_lower_tail_score']=='1'
    assert p['expected_paid_return']==q['expected_paid_return']=='3'
    assert not p['positive_tail_admission'] and q['positive_tail_admission']
    out=dict(role='Exact integer-count analytical conditional-return witness; no physical measurements or frozen algorithm changes',N=128,conservative_cost=5,alpha='1/2',common_score_correction='0',actual_interruption_slack='0',common_reserve=130,ledger_initial_loss=0,ledger_initial_pending=0,required_common_budget='at least130',P=p,Q=q,
        equal_conditional_mean_variance=True,opposite_prescribed_tail_actions=True,
        request_contrast_realisability='At most127 shared-reference-served positions. For every atom, reserve two candidate-correct/reference-incorrect positions for the admission interruption; use2 candidate-incorrect/reference-correct positions for G0,26 for G-24, and additional candidate-correct/reference-incorrect positions for G10 orG16. Remaining positions have zero contrast. Thus I=2 and D=G-5 exactly.',
        summary_feasibility_scope='A moment-only decision receives identical current context and conditional mean/variance. Requiring nonnegative true lower-tail paid score for every compatible law excludes admission because P is compatible and its score is-5.',
        missed_Q_expected_return='3',no_unconstrained_mean_return_dominance=True,
        scope='An exact conditional-tail feasibility witness, not an empirical performance total or calibration certificate. Under unrestricted expected-return maximization, admission in both laws has the same mean paid return3.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=ROOT/'conditional_tail_count_witness.json';path.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
