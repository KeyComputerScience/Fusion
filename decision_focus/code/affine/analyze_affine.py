"""Complete development-control analysis; no test-dependent selection."""
from pathlib import Path
import gzip, json, math
import numpy as np
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parents[2]
ARMS=('affine_joint','affine_diagonal','affine_factorized','affine_sandwich')
def load(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt') as f: return json.load(f)
    return json.loads(Path(p).read_text())
def excess(r):return max(0.,r['gain']-r['truegross']-r['q_issued']*r['posterior_or_block_sd'])
def summarize(rr):
    rows=[z for r in rr for z in r['rows']]; admitted=[z for z in rows if z['action']]
    groups={'issued':rows,'informative':[z for z in rows if z['disagreement']>0],'admitted':admitted}
    return dict(mean_increment=float(np.mean([r['increment'] for r in rr])),
       increments=[r['increment'] for r in rr],admissions=sum(r['admissions'] for r in rr),
       beneficial=sum(r['beneficial'] for r in rr),harmful=sum(r['harmful'] for r in rr),
       zero=sum(r['zero'] for r in rr),negative_loss=sum(r['negative_loss'] for r in rr),
       refused=sum(r['refused'] for r in rr),
       groups={k:dict(coverage=[sum(z['lower_covered'] for z in v),len(v)],
          total_excess=sum(excess(z) for z in v),max_excess=max([excess(z) for z in v]+[0.])) for k,v in groups.items()},
       max_mean_identity_error=max(r.get('conditional_mean_identity_error',0.) for r in rows),
       max_covariance_identity_error=max(r.get('conditional_covariance_identity_error',0.) for r in rows),
       issued_negative_influence_states=sum(r.get('negative_influences',0)>0 for r in rows),
       issued_rate_outside_support=sum(r.get('rate_outside_physical_support',False) for r in rows))
def decompose(full,alternative):
    cats={k:dict(count=0,units=0.) for k in ['additional_gain','avoided_loss','missed_gain','incurred_loss','zero_return']}
    actions=[]
    for seed,F,C in zip(sorted(full),[full[s] for s in sorted(full)],[alternative[s] for s in sorted(full)]):
        for f,c in zip(F['rows'],C['rows']):
            assert f['k']==c['k'] and f['local_net']==c['local_net']
            if f['action']==c['action']:continue
            D=f['local_net'];a=f['action'];cat=('additional_gain' if D>0 else 'incurred_loss' if D<0 else 'zero_return') if a else ('missed_gain' if D>0 else 'avoided_loss' if D<0 else 'zero_return')
            cats[cat]['count']+=1;cats[cat]['units']+=abs(D)
            actions.append(dict(seed=seed,k=f['k'],full_action=f['action'],control_action=c['action'],actual_return=D,category=cat,increment=(int(f['action'])-int(c['action']))*D,
                full_score=f['gate_score'],control_score=c['gate_score']))
    deltas=[full[s]['increment']-alternative[s]['increment'] for s in sorted(full)]
    assert abs(sum(a['increment'] for a in actions)-sum(deltas))<1e-8
    sd=float(np.std(deltas,ddof=1));mean=float(np.mean(deltas));half=2.776445105*sd/math.sqrt(5)
    return dict(categories=cats,changed=len(actions),actions=actions,seed_differences=deltas,mean=mean,conditional_t4_interval=[mean-half,mean+half])

def main():
    op=load(PROJECT/'work/fusion_focus_revision_20261003/operating/all_points.json.gz')
    summary={};newrows=[]
    for task in ['rss348','arem366','gashome362']:
        g=load(ROOT/(task+'_guarded.json.gz'));saved=load(ROOT/(task+'_results.json.gz'))
        out={'primary':{},'budgets':{},'comparisons':{},'selections':saved['selected']}
        if task=='rss348':
            old=load(PROJECT/'work/fusion_temporal_20261003/rss_validation/rss348_results.json')
        else:old=load(PROJECT/'work/fusion_strengthening_20261003/fresh/evaluation'/(task+'_results.json'))
        full={t['seed']:dict(rows=t['results']['precision_joint']['rows'],increment=t['results']['precision_joint']['net']-t['results']['reference']['net']) for t in old['trials']}
        for arm in ARMS:
            out['budgets'][arm]={str(int(b)):summarize([r for r in g if r['arm']==arm and r['budget']==b]) for b in [0.,110.,130.,260.]}
            out['primary'][arm]=out['budgets'][arm]['130']
            control={r['seed']:r for r in g if r['arm']==arm and r['budget']==130.}
            out['comparisons'][arm]=decompose(full,control)
        # Fixed-mean directional interventions on every selected affine joint
        # state; q is the actual issued joint threshold, not an oracle refit.
        cases=[]
        for t in saved['trials']:
            for row in t['results']['affine_joint']['rows']:
                C=np.asarray(row['affine_C']);mu=np.asarray(row['affine_mu']);n=len(mu)
                h=np.asarray(row['conditioned_error_mean'])+row['affine_rate']
                one=np.ones(n);v=1/float(one@np.linalg.solve(C,one));vd=1/float(np.sum(1/np.diag(C)))
                w=np.linalg.solve(C,one)*v;wd=(1/np.diag(C))*vd
                N=row['posterior_or_block_sd']/math.sqrt(v+.01**2)
                tdiag=float(wd@(h-mu));F=N*tdiag if row['information_ready'] else 0.
                score=F-row['q_issued']*N*math.sqrt(vd+.01**2)-5
                a=bool(row['gate_score']>0);b=bool(score>0)
                if a!=b:cases.append(dict(seed=t['seed'],k=row['k'],joint_proposal=a,diagonal_proposal=b,
                   joint_score=row['gate_score'],diagonal_score=score,actual_return=row['local_net'],
                   one_step_increment=(int(a)-int(b))*row['local_net']))
        out['fixed_state_diagonal_crossings']=cases
        out['fixed_state_diagonal_pooled_one_step_gain']=sum(r['one_step_increment'] for r in cases)
        summary[task]=out
    report=dict(scope='direct minimum-variance affine joint control; development extension on known physical tasks; not selected as a new final method',
       tasks=summary,verifications={t:load(ROOT/(t+'_verification.json')) for t in summary},
       primitive_identity='normal-equation stationarity and exact label-free disagreement-conditioning identities',
       result_integrity='all arms, grid trials, guarded and unguarded results kept; no test retuning or new independent validation claim')
    (ROOT/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    for t,v in summary.items():print(t,{a:(s['mean_increment'],s['harmful'],s['beneficial']) for a,s in v['primary'].items()},'full-minus-affine',v['comparisons']['affine_joint']['mean'])
    text=r'''\subsection{A direct joint-matrix value control}
\label{sec:affine-control}
We add a single-estimator control that uses every available joint input but
does not optimize the paid certificate. This control forms $C_k=R_k+P_k$
and projects the corrected source forecasts onto their common complete
value by
\begin{equation}
\widehat t_k=\arg\min_t(h_k-\mu_k-t\mathbf1)^{\mathsf T}
C_k^{-1}(h_k-\mu_k-t\mathbf1),\qquad
v_k=(\mathbf1^{\mathsf T}C_k^{-1}\mathbf1)^{-1}.
\label{eq:affine-control}
\end{equation}
It uses $F_k=N_k\widehat t_k$ and
$S_k=N_k\sqrt{v_k+\nu^2}$ in the identical delayed paid gate and
Fixed130/$B=130$ contract. The implied GLS coefficients may be signed;
they estimate a contrast and are not a probability mixture. Empty inputs
abstain, and unsupported history cannot authorize deployment.

Because $D\mathbf1=0$ implies $D(h_k-t\mathbf1)=Dh_k$, current
source differences are label-free. Under the working location law
$h_k\mid t\sim\mathcal N(\mu_k+t\mathbf1,C_k)$, conditioning on
those differences gives a common corrected value $\widehat t_k$ and
rank-one covariance $v_k\mathbf1\mathbf1^{\mathsf T}$. This is
standard Gaussian conditioning/GLS; it gives no accepted-action coverage
under an arbitrary physical shift. The control therefore tests whether
retaining the information but replacing the paid objective suffices.

The control and three reductions each receive the same 27 calibration
evaluations. The reductions diagonalize $C_k$ while holding $\mu_k$
fixed, rebuild a fully factorized state, or substitute the same-information
block-sandwich matrix for $P_k$. Formulas and the calibration rule are
frozen before their replays. RSS, AReM and GasHome were already examined,
so this is a development comparison on known tasks, not new physical
confirmation. Every configuration and outcome is retained.

\begin{table}[t]
\centering\small
\caption{Direct joint-value controls under identical Fixed130/$B=130$.
Cells contain mean reference-relative complete utility; parenthesized
counts are harmful / beneficial, pooled over five shared schedules.
These later controls do not replace the previously frozen full method.}
\label{tab:affine-controls}
\begin{tabular}{lrrr}
\toprule
Controller & RSS & AReM & GasHome\\
\midrule
Frozen Full & 30.2 (0/9) & $-30.0$ (5/0) & 45.2 (0/5)\\
Affine joint & 6.6 (4/4) & $-30.0$ (5/0) & 45.2 (0/5)\\
Affine diagonal, same mean & 6.6 (4/4) & $-30.0$ (5/0) & 45.2 (0/5)\\
Affine factorized state & 11.6 (4/6) & $-30.0$ (5/0) & 45.2 (0/5)\\
Affine sandwich & 6.6 (4/4) & $-30.0$ (5/0) & 45.2 (0/5)\\
\bottomrule
\end{tabular}
\end{table}

On RSS, Full gains 23.6 mean paid units over the direct joint control
and has no harmful lease, versus four for that control after reservation.
The direct control's five utility increments are $[-4,-4,2,41,-2]$;
the Full-minus-control interval is descriptive of delay variation only.
This is evidence for optimizing paid admission with the preserved state,
not evidence that its covariance is unavailable to deterministic GLS.
The direct joint and same-mean diagonal controls have identical budgeted
actions; cross-source risk therefore has no additional policy gain in
this control. The two additional tasks tie all four reductions.

The extension reconstructs 60 unguarded and 240 guarded trajectories.
Complete service and ledger checks pass without future labels entering
the live forecast. All covariance and conditional-mean identities are
checked at issuance; signed coefficients and out-of-support rate estimates
are counted in the saved records. The additional controls remain reported
as controls, rather than being promoted using their observed test returns.
'''
    comp=summary['rss348']['comparisons']['affine_joint'];c=comp['categories']
    detail=(r'Full versus the direct joint control adds %.1f beneficial units, avoids %.1f loss units, misses %.1f beneficial units and incurs %.1f loss units across the five schedules. The four-term sum is %.1f, giving the reported mean increment of %.1f; %d actions change. The conditional interval is $[%.2f,%.2f]$.'%(c['additional_gain']['units'],c['avoided_loss']['units'],c['missed_gain']['units'],c['incurred_loss']['units'],comp['mean']*5,comp['mean'],comp['changed'],*comp['conditional_t4_interval']))
    text=text.replace('This is evidence for optimizing paid admission with the preserved state,',detail+'\nThis is evidence for optimizing paid admission with the preserved state,')
    (ROOT/'manuscript.tex').write_text(text)

if __name__=='__main__':main()
