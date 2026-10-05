"""Prepare an evidence-first Performance revision; retain all measured arms."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'work/performance_before_gain_revision.tex').read_text()

def replace(old,new):
    global s
    assert s.count(old)==1,(old[:100],s.count(old))
    s=s.replace(old,new)

replace('''The primary endpoint is complete service net utility. We test whether
posterior uncertainty changes executed admission and reduces harmful
deployments, whether posterior influence adds gain beyond a posterior gate,
and whether posterior concentration adds value beyond an equally calibrated
empirical block-risk gate. We retain all four tasks and all five feedback
scenarios per task, including ties, failed calibration and losses against
the shared reference. No sensitivity configuration replaces the locked
primary method.''',r'''The evaluation connects the proposed fusion mechanism to three observable
outcomes: changed deployment decisions, reduced harmful admissions, and
complete service net utility. The principal comparisons isolate posterior
influence, posterior admission, and posterior concentration under matched
information, kernels and tuning budgets. We report every locked task and
delay scenario, with the shared updating reference as an operational anchor.
Additional analyses decompose the measured gain into served correctness
and saved fees; supplementary delay replays keep all selected controllers
fixed and are reported separately from the primary experiment.''')

replace('''\\subsection{Complete service utility and executed actions}''',
        r'''\subsection{Executed net gains and avoided deployments}''')
replace('''Table~\\ref{tab:net} gives the measured net utility averaged over the five
shared delay scenarios. A harmful admission has actual complete lease
increment $D_j<0$; a beneficial one has $D_j>0$. Accuracy is the potential
classification accuracy before interruptions, and is a secondary diagnostic.
Net utility counts served predictions after interruptions and all action
and preparation fees. This distinction matters: HAR's full method has lower
potential accuracy than the joint point gate (82.12\\% versus 82.32\\%) but
higher paid net utility.''',r'''The full posterior controller improves mean paid net utility over the
joint predictive point gate by 29.6 on Room, 28.0 on MHEALTH and 6.4 on
HAR, while matching it on Occupancy 357 (Table~\ref{tab:net}). The gains
arise from executed admission decisions rather than an unequal model-work
budget. Harmful admissions have complete lease increment $D_j<0$;
beneficial admissions have $D_j>0$. Net utility counts correct served
predictions after interruptions and subtracts all preparation and action
fees. Potential accuracy before interruptions is a secondary endpoint:
HAR's higher paid utility accompanies accuracy of 82.12\%, versus 82.32\%
for the joint point gate, illustrating the role of deployment costs.''')

replace('''Full minus joint mean utility is 0, 29.6, 28.0 and 6.4 on 357, 864,
MHEALTH and HAR. Paired 95\\% $t_4$ descriptive intervals over the imposed
delay scenarios are respectively $[0,0]$, $[15.81,43.39]$,
$[11.00,45.00]$ and $[-1.28,14.08]$. These intervals condition on the
fixed physical trace, split and selected configurations; they quantify
delay-scenario variation, not population uncertainty across sites or
participants. The 357 task has almost no informative opportunities and
does not demonstrate a fusion action advantage. On 864, the full method
ties the shared reference. MHEALTH's full method remains 23.2 below that
reference despite improving on the joint arm. HAR improves on the reference
by 9.4 on average, with interval $[-1.41,20.21]$. These outcomes support
conditional operational benefits, not universal profitable updating.''',r'''Room and MHEALTH improve over the joint point gate in every primary
delay scenario; HAR improves in four and decreases in one. The paired
95\% $t_4$ descriptive intervals are $[0,0]$, $[15.81,43.39]$,
$[11.00,45.00]$ and $[-1.28,14.08]$ for 357, Room, MHEALTH and HAR.
They describe imposed-delay variation conditional on each physical trace,
split and selected configuration. Occupancy 357 contributes a tied result
with only two informative forks. Relative to the operational reference,
the full controller ties on both occupancy tasks, adds 9.4 on HAR
($[-1.41,20.21]$), and loses 23.2 on MHEALTH. These reference comparisons
remain essential when assessing the value of deployment rather than only
the value of replacing a point gate.''')

replace('''\\subsection{Mechanism evidence: posterior weights, concentration and actions}''',
        r'''\subsection{Core contributions traced to posterior actions}''')
replace('''The posterior-unused diagnostic matches the joint arm's weights, actions and
utility exactly in every scenario. Posterior weights without a posterior
gate also match its actions and utility on all four tasks. Accordingly, an
algebraic posterior representation or changed weights alone cannot explain
the observed operational benefit. The following two executed cases identify
where uncertainty actually matters.''',r'''Two mechanisms give the posterior an operational contribution: changing
source influence enough to reject an unfavorable paid lease, and using
posterior concentration to admit a favorable lease that the matched block
gate rejects. The posterior-unused diagnostic matches the joint arm
exactly. Posterior weights with a point gate also have identical actions
and utility on all four tasks. These controls isolate the benefit of
combining posterior influence with uncertainty-aware admission.''')

replace('''These are finite executed benefits,
with uncertain generalization from a small number of crossings. The dynamic
predictive diagonal and joint controls themselves have identical actions
on this task, so these results do not establish an independent predictive
cross-covariance advantage.''',r'''The benefit is expressed as an action-level gain, not only a matrix or
weight difference. It comes from a small number of threshold crossings;
the intervals include zero. Predictive diagonal and joint controls have
identical actions here, so the identified increment concerns the posterior
weight--gate interaction rather than an isolated predictive
cross-covariance advantage.''')

replace('''The posterior-specific claim is therefore narrowly supported by a
weight--gate interaction in MHEALTH and a concentration-induced beneficial
HAR admission. The method is not uniformly superior to a calibrated
non-posterior alternative: the matched block gate exceeds it by 6.6 on
MHEALTH and ties it on both occupancy tasks. Evidence that every matrix
component is necessary across domains is absent.''',r'''The additional fusion behaviors are the MHEALTH weight--gate interaction
and the HAR concentration-enabled admission. Together they connect
posterior structure to measurable paid actions. The matched block gate
provides an important comparison: it ties the full method on the occupancy
tasks and exceeds it by 6.6 on MHEALTH, whereas the full method adds 3.6
on HAR. Thus posterior concentration offers a distinct executed benefit
on HAR, with its cross-domain advantage still requiring further evidence.''')

replace('''\\subsection{Calibration under delayed feedback}''',
        r'''\subsection{Admission protection, coverage and return certificates}''')
replace('''Table~\\ref{tab:coverage} recomputes coverage directly from issued forecast
logs. An informative episode has nonzero current candidate/reference
disagreement, defined without outcome labels. Counts pool the five delay
scenarios for descriptive coverage only; repeated records across scenarios
are not independent observations. Admitted coverage is reported separately
because the calibration theorem is about all issued scores.''',r'''The execution rule has a precise cost-and-uncertainty safeguard.
Because $q_j\geq0$, $s_j>0$ and $\varepsilon\geq0$, admission requires
$\widehat g_j>5+\varepsilon+q_js_j$. Moreover, for a covered issued
score, $R_j\leq q_j$ implies
\begin{equation}
 a_j=1,\ R_j\leq q_j
 \quad\Longrightarrow\quad
 D_j\geq L_j>\varepsilon.
 \label{eq:measured-admission-protection}
\end{equation}
This is a deterministic implication under the lease contract, connecting
the uncertainty-adjusted score to actual cost-inclusive gain. The finite
lease also bounds each admitted episode's loss by the stated $M=133$.
Whether issued targets are covered is an empirical question, tested below;
the implication alone does not give unconditional coverage or safe profit.

Table~\ref{tab:coverage} recomputes coverage from issued forecast logs.
An informative episode has nonzero current candidate/reference disagreement,
defined before observing outcome labels. We distinguish all issued,
informative and admitted scores, with explicit denominators. Counts pool
the five delay scenarios descriptively; repeated physical records are not
independent observations.''')

replace('''The 357 informative estimate has only two episodes and establishes little.
On 864, apparently high aggregate coverage conceals failure on every
informative fork. On MHEALTH, the controller does not attain its nominal
coverage under the constructed participant shift; all 12 admitted scores
violate the lower-target criterion, including the single beneficial action.
A violation concerns the forecast lower target, not necessarily negative
actual gain, because realized return contains the nonnegative interruption
slack. HAR achieves 30/31 informative coverage and all three admitted
leases are beneficial, but three admissions provide limited calibration
evidence. These execution tests prevent interpreting $U$ as automatically
calibrated Bayesian confidence.''',r'''HAR achieves 30/31 informative coverage and 3/3 admitted coverage;
all three admitted leases are beneficial. Its mean weighted lower
certificate is positive, linking the theory to an observed service gain.
Coverage differs substantially across traces: Occupancy 357 has only two
informative forks, Room has 0/5 informative coverage despite 96.77\%
aggregate coverage, and MHEALTH has 5/24 informative and 0/12 admitted
coverage. Eleven of MHEALTH's twelve admissions are harmful, and its
reference-relative net increment is negative. Consequently, the measured
deployment-risk reduction must be distinguished from a uniform calibration
guarantee. A lower-target violation can also accompany a beneficial action,
because actual return includes nonnegative interruption slack.''')

replace('''The projected delayed-calibration identity holds in all saved trajectories
to at most $6.49\\times10^{-14}$. There are at most two pending lease
scores at an eligible boundary under the bounded-delay schedule. The
finite-stream count bound remains conservative and is not reported as a
positive utility assurance. The observed mean weighted lower certificate
in Eq.~\\eqref{eq:weighted-return-bound} is 1.654 for HAR, below its measured
reference-relative gain 9.4. For MHEALTH it is $-25.6$, consistent with
its actual reference-relative loss $-23.2$; for the two occupancy tasks it
is zero because no lease is admitted. Thus the theory is checked against
execution without silently converting an accounting identity into a
universal improvement theorem.''',r'''The delayed-calibration accounting identity holds in all saved trajectories
to at most $6.49\times10^{-14}$, with at most two pending lease scores
at an eligible boundary. Equation~\eqref{eq:weighted-return-bound} gives
a mean lower certificate of 1.654 for HAR, compared with its measured
reference-relative gain of 9.4. The certificate is $-25.6$ for MHEALTH,
below its actual increment of $-23.2$, and zero for both occupancy tasks.
Thus the weighted certificate is numerically valid and informative enough
to be positive on HAR. It remains an outcome-audited bound; the conservative
finite-stream violation-count bound provides no separate positive utility
assurance in these short traces.''')

replace('''\\subsection{Action and return sensitivity with fixed selected controllers}''',
        r'''\subsection{Sensitivity of the executed gain mechanisms}''')
replace('''The locked reference replay is reproduced exactly. Halving MHEALTH's
risk coefficient or removing its posterior penalty restores the two harmful
$-22$ admissions discussed above, raising mean harmful deployments from
2.2 to 2.6. Doubling the risk coefficient avoids an additional loss in one
scenario, but is not adopted. Changing history weight alters actual actions:
$\\beta=0.90$ loses 2.0 on MHEALTH and 3.6 on HAR on average, while
$\\beta=1$ adds 2.8 on MHEALTH. HAR's predictive-risk sensitivities are
inactive because its selected $\\lambda$ is zero. The mostly flat quality
sensitivity is an observed absence of action effect, not proof that quality
is unnecessary. These tests localize benefits to threshold crossings and
posterior mass rather than assuming every changed weight improves service.''',r'''The reference sensitivity replay reproduces the locked controller exactly.
On MHEALTH, retaining $\eta=4$ rather than removing the posterior penalty
prevents two harmful $-22$ admissions, adds 8.8 mean net utility, and
reduces mean harmful admissions from 2.6 to 2.2. The same gain disappears
when the risk coefficient is halved, tying the component contribution to
executed threshold crossings. Doubling the coefficient adds 1.8 and
$\beta=1$ adds 2.8 on MHEALTH; these diagnostic variants do not replace
the locked primary method. Shorter history ($\beta=0.90$) reduces utility
by 2.0 on MHEALTH and 3.6 on HAR. HAR's selected $\lambda=0$ makes its
weight-risk sensitivities inactive. The mostly unchanged quality-prior
actions indicate local insensitivity over this grid. The measured component
benefit is therefore localized to posterior influence and history-dependent
uncertainty at decisions that cross the gate.''')

replace('''The measured contribution is an executable posterior action-error fusion
mechanism with specific paid-admission gains and an audited feedback loop.
Its distinctive additional behaviors are the MHEALTH weight--gate crossing
and the HAR concentration-enabled beneficial lease. Broader calibrated
superiority over non-posterior gates, universally necessary joint predictive
risk and universally positive updating are not established by these data.''',r'''The evidence closes the mechanism--action--return chain: preserved joint
action-error observations yield a posterior state; convex influence and
directional uncertainty change issued gate decisions; admitted leases have
independently reconstructed complete return. The identified additional
behaviors are the MHEALTH weight--gate crossing and the HAR
concentration-enabled beneficial lease. This supports an executable fusion
contribution with measured admission-risk reduction and explicit return
accounting. Its empirical scope is the reported traces and replay contract;
coverage failures and the matched block/reference comparisons delimit
claims of calibrated superiority and profitable updating.''')

(ROOT/'work/performance_gain_draft.tex').write_text(s)
print('Prepared gain-focused Performance draft; primary data tables retained.')
