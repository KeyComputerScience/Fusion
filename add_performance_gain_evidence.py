"""Insert tables computed from immutable primary and new delay-replay logs."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work'
s=(WORK/'performance_gain_draft.tex').read_text()
primary=json.loads((WORK/'performance_gain_analysis.json').read_text())
extra=json.loads((WORK/'performance_delay_extension/results.json').read_text())
summary=json.loads((WORK/'performance_delay_extension/summary.json').read_text())
names=['occupancy357','occupancy864','mhealth319','har240']
labels=['Occupancy 357','Room 864','MHEALTH','HAR']

decomp=r'''\subsection{Measured sources of the paid gain}
The common-work contract permits an exact comparison between the full
controller $F$ and an aligned control $C$:
\begin{align}
 J_F-J_C
 &=C_F^{\rm served}-C_C^{\rm served}
   +3(A_C-A_F),\label{eq:measured-gain-decomposition}\\
 &=\sum_j(a_j^F-a_j^C)D_j.\label{eq:paired-executed-gain}
\end{align}
Here $C^{\rm served}$ counts correct served requests after all interruptions,
and $A$ counts admitted leases. Each lease pays two installation units and
one restoration unit; shared preparation and reference-refresh fees cancel.
The second identity is valid because the contrasted model versions and
complete lease increments are common across arms, with restoration and
nonoverlapping leases. We check both identities against saved trajectories.

\begin{table}[t]
\centering\small
\caption{Primary full-minus-joint gain decomposition. Utility columns are
means over five delay scenarios. Avoided harmful and retained beneficial
counts refer to paired origin/seed leases admitted by the joint controller;
their denominators are its harmful and beneficial admissions. A zero
denominator denotes no applicable event.}
\label{tab:gain-decomposition}
\begin{tabular}{lrrrrr}
\toprule
Task & $\Delta C^{\rm served}$ & Saved fees & $\Delta J$
 & Harmful avoided & Beneficial retained\\
\midrule
'''
for name,label in zip(names,labels):
    x=primary['tasks'][name]['comparisons']['joint']
    h=x['baseline_harmful_avoided'];b=x['beneficial_retention']
    decomp+=f"{label} & {x['served_correctness_change']['mean']:.1f} & {x['saved_fees']['mean']:.1f} & {x['paired_net_gain']['mean']:.1f} & {h['numerator']}/{h['denominator']} & {b['numerator']}/{b['denominator']}"+r'\\'+'\n'
decomp+=r'''\bottomrule
\end{tabular}
\end{table}

Room's 29.6-unit gain combines 26.6 more correct served predictions with
three saved fee units; MHEALTH combines 18.4 with 9.6. HAR gains 6.4
despite serving 2.6 fewer correct predictions, because nine fee units are
saved. Across the complete primary replay set, harmful admissions decrease
from 43 to 11: 32/43, or 74.42\%, are avoided. Four of eight joint-admitted
beneficial leases are retained; the four missed HAR leases sum to 24 utility
units. Equation~\eqref{eq:paired-executed-gain} includes both avoided losses
and missed gains, so the action analysis does not equate fewer deployments
with preserved recall of every useful update. These totals are descriptive
bookkeeping across 20 task--delay replays, not an estimate of population risk.

The full method also exceeds periodic admission on all 20 primary
task--delay comparisons, by task means of 312.4, 178.4, 92.8 and 138.4.
Periodic admission demonstrates the cost of indiscriminate updating;
the calibrated block gate remains the stronger comparator for isolating
the posterior-specific contribution.

'''
anchor=r'\subsection{Core contributions traced to posterior actions}'
assert s.count(anchor)==1;s=s.replace(anchor,decomp+anchor)

new=r'''\subsection{Supplementary locked-controller delay validation}
\label{sec:additional-delay}
After the primary results, we declared five additional delay schedules,
seeds 72006--72010, on all four existing test traces. Before running them,
we saved a separate protocol and controller/data hashes. The joint point
gate, posterior gate, full posterior and matched block gate retain their
original task-specific selected configurations and $q_0$, including
$\lambda=0$ where selected. They receive the same new delay schedule,
context, kernel, model work, forecasts, archive and costs. No refitting of
$q_0$, new search or task substitution is performed; the original causal
online calibration remains active. This is a later diagnostic
extension testing delay-schedule robustness on the same physical traces;
it does not supply new-site or new-participant evidence. Every result is
retained and the primary five-scenario tables remain unchanged.

\begin{table}[t]
\centering\small
\caption{Actually executed supplementary replays, seeds 72006--72010.
Entries are mean complete service net utility under the previously selected
controllers. No supplementary result enters controller selection.}
\label{tab:additional-net}
\begin{tabular}{lrrrr}
\toprule
Controller & Occupancy 357 & Room 864 & MHEALTH & HAR\\
\midrule
'''
for mode,label in [('joint','Joint predictive'),('bayes_gate','Posterior gate'),('bayes_both','Full posterior'),('frequentist_gate','Matched block gate')]:
    new+=label+' & '+' & '.join(f"{summary[n]['controllers'][mode]['mean_net']:.2f}" for n in names)+r'\\'+'\n'
new+=r'''\bottomrule
\end{tabular}
\end{table}

The full-minus-joint increments are $[0,0,0,0,0]$ on 357,
$[35,36,38,36,36]$ on Room, $[32,40,24,39,28]$ on MHEALTH and
$[8,7,3,4,3]$ on HAR. Their means are 0, 36.2, 32.6 and 5.0;
the conditional $t_4$ descriptive intervals are $[0,0]$,
$[34.84,37.56]$, $[24.02,41.18]$ and $[2.09,7.91]$.
Thus all 15 comparisons on Room and the wearable traces improve over
the point gate, while all five 357 comparisons tie. Room harmful admissions
decrease from 5 to 0, MHEALTH from 26 to 11 and HAR from 11 to 0.
The full controller retains 1/1 joint-admitted beneficial MHEALTH lease
and 3/6 beneficial HAR leases.

The added MHEALTH posterior-influence increment, full minus posterior
gate only, is $[22,22,0,22,22]$: mean 17.6 with conditional interval
$[5.38,29.82]$. Four additional delay schedules reproduce the avoided
$-22$ lease, without adopting a new risk coefficient or threshold.
On HAR, full minus matched block gate is $[0,17,0,0,17]$, mean 6.8
with interval $[-4.76,18.36]$. It captures two additional positive
leases that the calibrated block gate rejects. Both occur at window 68,
in seeds 72007 and 72010. With identical influence and gross forecasts,
their posterior masses are 21.488 and 21.692; posterior scales are 3.018
and 2.905, compared with block scales 13.026 and 12.489. Separately
calibrated issued thresholds give posterior gate scores $+0.915$ and
$+1.403$, versus block scores $-1.931$ and $-1.784$. Each admitted lease
realizes $+17$ complete net utility. These extend the two
identified action mechanisms under new delays; their physical observations
remain the same test traces.

\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[ybar,bar width=18pt,width=.9\linewidth,height=5.4cm,
 symbolic x coords={MHEALTH,HAR},xtick=data,
 ylabel={Mean posterior component gain},ymin=-10,ymax=34,
 enlarge x limits=.5,legend style={at={(.5,1.02)},anchor=south,
 legend columns=2,font=\small},ymajorgrids=true]
'''
orig=[primary['tasks']['mhealth319']['comparisons']['bayes_gate']['paired_net_gain'],primary['tasks']['har240']['comparisons']['frequentist_gate']['paired_net_gain']]
later=[summary['mhealth319']['paired']['bayes_both_minus_bayes_gate']['net'],summary['har240']['paired']['bayes_both_minus_frequentist_gate']['net']]
for values,ci_key in [(orig,'conditional_delay_t4_descriptive_interval'),(later,'ci95_t4')]:
    points=[]
    for label,vals in zip(['MHEALTH','HAR'],values):
        half=(vals[ci_key][1]-vals[ci_key][0])/2
        points.append(f"({label},{vals['mean']:.8f}) +- (0,{half:.8f})")
    new+=r'\addplot+[error bars/.cd,y dir=both,y explicit] coordinates {'+' '.join(points)+'};\n'
new+=r'''\legend{Primary delays,Additional delays}
\end{axis}
\end{tikzpicture}
\caption{Executed posterior component gains, with conditional five-delay
$t_4$ intervals. MHEALTH isolates full minus posterior-gate-only;
HAR isolates full minus the separately calibrated block gate. Additional
delays use unchanged selected controllers on the same physical traces.
The error bars describe replay variation and do not quantify site-level
generalization or correct for multiple comparisons.}
\label{fig:posterior-delay-validation}
\end{figure}

The matched block gate ties the full mean on MHEALTH in this extension,
with paired differences $[16,9,-18,2,-9]$ and interval $[-16.96,16.96]$.
Equal mean utility does not imply equal deployment risk: the full controller
has 11 harmful and one beneficial admission out of 12, whereas the block
gate has five harmful admissions out of five and no beneficial admission.
The full method remains 22.0 below the shared reference on MHEALTH;
HAR is 10.8 above it. Full informative coverage is 4/4 on 357, 0/5
on Room, 6/25 on MHEALTH and 27/28 on HAR. Admitted coverage is
1/12 on MHEALTH and 3/3 on HAR, with no occupancy admissions.
The extension therefore strengthens the observed action and utility
effects while retaining the calibration and strongest-control comparisons.
Aggregate full coverage is 301/320, 150/155, 72/105 and 150/160,
respectively; high aggregate coverage is not substituted for
decision-relevant coverage.

All 80 supplementary policy trajectories undergo independent request-level
service reconstruction, and 2,960 method-specific fork checks verify
complete labels, gross contrasts, actual interruptions and conservative
targets. Maximum service discrepancy is $3.56\times10^{-14}$,
fork discrepancy is zero and target-identity error is
$7.11\times10^{-15}$. All convex solves pass, with maximum KKT residual
$1.00\times10^{-9}$; delayed-calibration identity error is at most
$6.27\times10^{-14}$. The original executable, adapters, selections and
primary result files remain byte-identical. Protocol SHA-256 begins
\texttt{1853e0d747b2f406}. The separate extension archive contains
the protocol, pre-replay freeze, fixed controller inputs, trial logs,
auditor output and a portable rerun launcher.

'''
anchor=r'\subsection{Independent execution audit and reproduction details}'
assert s.count(anchor)==1;s=s.replace(anchor,new+anchor)
s=s.replace('The study has two prospective protocol rounds.','The primary study has two prospective protocol rounds.')
s=s.replace('Across 210 policy trajectories, 740 distinct','Across 210 primary policy trajectories, 740 distinct')
s=s.replace('The three figures are self-contained vector','The four figures are self-contained vector')
s=s.replace('contribution with measured admission-risk reduction and explicit return',
            'contribution with measured admission-risk reduction relative to the joint\npoint gate and explicit return')
out=WORK/'performance_gain_final.tex';out.write_text(s)
print('Prepared final Performance section with verified gain tables and new delay trials.')
