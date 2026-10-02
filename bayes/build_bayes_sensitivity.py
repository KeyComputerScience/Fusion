import json
from pathlib import Path
import numpy as np

W=Path(__file__).resolve().parent
s=json.loads((W/'bayes_results_sensitivity.json').read_text())
n=json.loads((W/'bayes_results_no_drift.json').read_text())
labels={'quality_power':r'$\alpha_q$', 'risk_scale':r'$\lambda$',
        'history_weight':r'$\kappa$', 'beta':r'$\beta$'}
rows=[]
audit={}
for field, settings in s['sensitivity'].items():
    audit[field]={}
    for value, result in settings.items():
        a=result['action_sensitivity']
        changes=sum(x['changed_update_slots'] for x in a)
        weight=np.mean([x['mean_executed_weight_l2_change'] for x in a])
        net=result['summary']['methods']['bm_joint']['net_return']['mean']
        delay=np.mean([x['recovery_sign_delay'] for x in a])
        gain=np.array([x['net_change_from_default'] for x in a])
        mean=gain.mean()
        radius=1.96*gain.std(ddof=1)/np.sqrt(len(gain))
        assert all(x['updates']==9 and x['steps']==288 and x['cost']==432 for x in a)
        rows.append(f"{labels[field]}={float(value):g} & ${weight:.4f}$ & {changes} & "
                    f"${net:.2f}$ & ${mean:.2f}$ & $[{mean-radius:.2f},{mean+radius:.2f}]$\\\\")
        audit[field][value]={'mean_weight_l2_change':float(weight),'changed_action_slots':changes,
                              'net_return_mean':net,'gain_mean':float(mean),'gain95':[float(mean-radius),float(mean+radius)],
                              'sign_recovery_mean':float(delay)}
table='\n'.join(rows)
c=n['summary']['paired_net_return']['frozen']
lo,hi=c['paired_95_normal_interval']
updates=n['summary']['methods']['bm_joint']['updates']['mean']
text=rf'''
\subsection{{Sensitivity of weights, actions, and cost-inclusive return}}
\label{{sec:bm-actual-sensitivity}}

Sensitivity reruns the entire service loop on the same ten physical seeds
93001--93010. It varies quality concentration $q_s^{{\alpha_q}}$,
residual-risk coefficient $\lambda$, the legacy historical-weight coefficient
$\kappa$, and archive forgetting $\beta$, one at a time. The HMM, value model,
context kernel, source observations, label delays, worker, and all other settings
remain fixed. The reference configuration is $\alpha_q=1$, $\lambda=1$,
$\kappa=0$, $\beta=0.985$. No setting is selected as a new winning test
configuration. Each rerun executes nine updates, 288 actual gradient steps,
432 step-fee units, and 135 lost service opportunities, so these action/return
changes cannot be explained by a larger realized update budget.

\begin{{table}}[htbp]
\centering\footnotesize
\caption{{Executed sensitivity over 5,600 decision slots. Weight change is mean
$\ell_2$ distance from the reference at each arm's deployed orientation.
Changed slots count launch/no-launch disagreements. Return differences and
descriptive paired intervals use the same ten seeds.}}
\label{{tab:bm-sensitivity}}
\begin{{tabular}}{{lrrrrr}}\toprule
Setting & Mean weight change & Changed slots & Net return & $\Delta J$ & Paired interval\\\midrule
{table}
\bottomrule\end{{tabular}}
\end{{table}}

Changes in source influence often fail to cross a deployment threshold, so
their action effect is much smaller than their weight effect. Some risk-free
or shorter-memory settings have a higher mean return than the reference.
The study therefore does not establish that a nonzero risk term, additional
inertia, or longer history is universally beneficial. The analytical bound in
Eq.~\eqref{{eq:bm-sensitivity}} explains fixed-mask weight perturbations;
Table~\ref{{tab:bm-sensitivity}} measures the subsequent decision and payoff
effects without extending that bound across mask or threshold discontinuities.
Per-seed expenditure and first-sign-alignment metrics accompany every setting;
the latter remain a supplementary, nonsustained recovery measure.

\subsection{{No-drift control and scope of the supported contribution}}
\label{{sec:bm-evidence-boundary}}

A separate no-relation-drift control uses ten independent seeds
94001--94010, retaining covariance regimes, measurement noise, masks,
and the unchanged deployment protocol. BM-DAF executes {updates:.2f} updates
on average and has a paired mean return difference of ${c['mean_gain']:.2f}$
against frozen, with descriptive interval $[{lo:.2f},{hi:.2f}]$.
Seven seeds have identical return and three have harmful unnecessary updates.
This reports false-update cost rather than asserting that fusion always avoids
unnecessary training. Its return equals the contextual-joint control on all
ten seeds. Benefits against expensive periodic schedules in this condition
are generic admission benefits, not evidence for the decision-weighted moment.

The supported theoretical contribution is a PSD posterior consequential-error
object with a strict same-information risk gain in a stated positive-stakes
construction, and a finite-archive correction consistent with fixed-memory
implementation. The supported empirical contribution is an actual delayed-label
service loop with cost-inclusive comparisons and executed sensitivity.
The current formal comparison does not establish a stable extra service gain
over the strongest contextual-joint, RF-C, or context-only-belief controls.
It also does not establish real-data external validity, sustained recovery of
a neural service policy, statistical coverage for adaptive moment estimates,
or wall-clock edge performance. These are material limits of the evidence.

Source hashes, frozen configuration, chronological/pilot/evaluation seeds,
per-seed outcomes, action audits, and full sensitivity records are supplied.
Dynamic fusion methods such as QMF and PDF provide relevant theoretical and
multimodal-learning context~\cite{{qmf2023,pdf2024}}; their complete neural
architectures are not claimed reproduced by these scalar forecast controls.
Delayed-learning and missing-covariance results also require their own
assumptions~\cite{{joulani2013,lounici2014}}. No prior empirical result is
reassigned to BM-DAF, and no planned experiment is presented as completed.
'''
(W/'bayes_sensitivity_results.tex').write_text(text)
(W/'bayes_sensitivity_table_audit.json').write_text(json.dumps(audit,indent=2))
print('wrote sensitivity table',len(rows),'settings')
