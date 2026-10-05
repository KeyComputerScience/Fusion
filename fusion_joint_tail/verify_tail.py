"""Executable same-summary separation and numerical checks, not field data."""
import itertools,json,math
from pathlib import Path
import numpy as np
from tail_fusion import DiscreteLaw,TruncatedMixture,joint_line,paid_score

def verify():
    N=128;c=13/N;small=3/N;large=12/N;alpha=.1
    plus=np.array([(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)],float)
    minus=-plus
    A=np.r_[small*plus,large*minus];B=np.r_[small*minus,large*plus]
    weights=np.r_[np.full(4,.8/4),np.full(4,.2/4)]
    def marginal(atoms,ids):
        return sorted((tuple(row),round(float(weights[np.all(atoms[:,ids]==row,axis=1)].sum()),14)) for row in np.unique(atoms[:,ids],axis=0))
    for size in (1,2):
        for ids in itertools.combinations(range(3),size):assert marginal(A,list(ids))==marginal(B,list(ids))
    meanA=weights@A;meanB=weights@B
    covA=np.einsum('j,js,jt->st',weights,A,A);covB=np.einsum('j,js,jt->st',weights,B,B)
    assert np.max(abs(meanA-meanB))<1e-15 and np.max(abs(covA-covB))<1e-15
    lawA=DiscreteLaw([5,20],[.8,.2]);lawB=DiscreteLaw([11,-4],[.8,.2])
    pooled=DiscreteLaw([5,20,11,-4],[.4,.1,.4,.1])
    assert lawA.mean==lawB.mean==8 and abs(lawA.variance-lawB.variance)<1e-13
    assert lawA.lower_tail(alpha)==5 and lawB.lower_tail(alpha)==-4 and pooled.lower_tail(alpha)==-4
    eps=1e-4;V=np.repeat((eps**2*np.eye(3))[None],8,axis=0)
    fits=[joint_line(np.full(3,c),weights,atoms,V) for atoms in (A,B)]
    outputs=[paid_score(p,N,alpha) for p in fits]
    assert outputs[0]['score']>4.98 and outputs[1]['score']<-3.99
    assert max(abs(o['mean']-8) for o in outputs)<1e-8
    assert abs(outputs[0]['variance']-outputs[1]['variance'])<1e-8
    # Common moment Gaussian is the same in both regimes and rejects both.
    gaussian=TruncatedMixture([c],[36/N**2],[0.])
    g=paid_score(gaussian,N,alpha);assert g['score']<0
    raw_gaussian=paid_score(joint_line(np.full(3,c),np.ones(1),np.zeros((1,3)),covA[None]+V[:1]),N,alpha)
    # Translation is a W1-distance-exact check of the Lipschitz constant.
    shifts=[]
    for shift in (-2.,-.3,.1,3.):
        moved=DiscreteLaw(lawA.values+shift,lawA.weights)
        error=abs(moved.lower_tail(alpha)-lawA.lower_tail(alpha))
        assert error<=abs(shift)/alpha+1e-12;shifts.append(error)
    # Same mean/sd gate with q=1.28155 proposes both: it cannot distinguish.
    old=8-1.2815515655446004*6
    assert old>0
    return dict(passed=True,scope='constructed realizable joint-error experiment; not a physical test',
        capacity=N,alpha=alpha,conditional_paid_mean=[lawA.mean,lawB.mean],conditional_paid_variance=[lawA.variance,lawB.variance],
        exact_lower_tail=[5.,-4.],reduced_pooled_lower_tail=-4.,smoothed_scores=outputs,
        conditional_moment_gaussian_score=g,unconditional_matrix_score=raw_gaussian,
        old_mean_sd_score=old,conditional_expected_paid_gain=4.,
        expected_positive_tail_certificate=2.5,joint_actions=[True,False],matrix_actions=[False,False],
        full_error_moment_error=float(np.max(abs(covA-covB))),
        smoothing_sd=eps,mean_difference=float(abs(outputs[0]['mean']-outputs[1]['mean'])),
        variance_difference=float(abs(outputs[0]['variance']-outputs[1]['variance'])))

if __name__=='__main__':
    answer=verify();Path(__file__).with_name('theory_checks.json').write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps(answer,indent=2))
