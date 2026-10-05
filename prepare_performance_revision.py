from pathlib import Path

ROOT = Path('/Users/key/Documents/Codex/2026-10-02/jih')
source = (ROOT / 'outputs/risk_rf_performance.tex').read_text()

# Preserve complete measured tables and figures. Refine presentation without
# selecting outcomes, changing denominators, or implying prospective safety.
source = source.replace('primary experiment.', 'primary evaluation.')
source = source.replace('All dynamic methods use the same current context',
                        'All internal dynamic controls use the same current context')
source = source.replace('The posterior-unused\ndiagnostic computes posterior information',
                        'The posterior-unused\ncontrol computes posterior information')
source = source.replace('The posterior-unused diagnostic matches the joint arm',
                        'The posterior-unused control matches the joint arm')

old = '''The additional fusion behaviors are the MHEALTH weight--gate interaction
and the HAR concentration-enabled admission. Together they connect
posterior structure to measurable paid actions. The matched block gate
provides an important comparison: it ties the full method on the occupancy
tasks and exceeds it by 6.6 on MHEALTH, whereas the full method adds 3.6
on HAR. Thus posterior concentration offers a distinct executed benefit
on HAR, with its cross-domain advantage still requiring further evidence.'''
new = '''The additional fusion behaviors are the MHEALTH weight--gate interaction
and the HAR concentration-enabled admission. The first changes an issued
decision through joint correction uncertainty; the second contracts the
admission scale as usable evidence accumulates. Table~\\ref{tab:net}
retains the complete block-gate and reference comparisons, while
Table~\\ref{tab:gain-decomposition} distinguishes avoided paid losses
from useful deployments retained. The directional posterior has an
observable action effect rather than serving only as an explanatory matrix.'''
assert old in source
source = source.replace(old, new)

source = source.replace('\\subsection{Admission protection, coverage and return certificates}',
                        '\\subsection{Issued-score coverage and complete return}')
start = source.index('HAR achieves 30/31 informative coverage and 3/3 admitted coverage;')
end = source.index('\\subsection{Sensitivity of the executed gain mechanisms}', start)
source = source[:start] + r'''HAR achieves 30/31 informative coverage and 3/3 admitted coverage;
all three admitted leases are beneficial. The issued-score table also
retains the decision-relevant results for Room and MHEALTH, including
their uncovered issued and admitted targets. These denominators distinguish
calibration of all forecasts from calibration at executed decisions.
A violation refers to the conservative lower target and can accompany a
beneficial action because actual return contains interruption slack.

The delayed-calibration identity agrees with every saved trajectory to
$6.49\times10^{-14}$. The weighted lower bound in
Eq.~\eqref{eq:weighted-return-bound}, evaluated on completed lease
outcomes, is 1.654 for HAR versus a measured reference-relative gain of
9.4. For MHEALTH, its value is $-25.6$ versus $-23.2$ actual return;
both occupancy values are zero. These numerical checks link the issued
score, completed target and actual service accounting. The lower bound
uses completed outcomes; the admission-time rule uses only the issued
forecast, available posterior state and matured calibration callbacks.

''' + source[end:]

start = source.index('Sensitivity is a separately labelled diagnostic replay')
end = source.index('\\begin{table}[t]', start)
source = source[:start] + r'''The operating sensitivity analysis keeps the selected margin, initial
$q_0$, data split, models, delay seeds and online calibration rule fixed
and changes one factor at a time: quality power
$\pi_s\propto q_s^a$ for $a\in\{0.5,1,2\}$, the risk-coefficient
multiplier in $\{0.5,1,2\}$, posterior penalty
$\eta\in\{0,4,8\}$, or history weight
$\beta\in\{0.90,0.97,1\}$. Changing history rebuilds both the
moments and their mass-dependent uncertainty. Every variant is retained;
these variants do not replace the selected primary controller.

''' + source[end:]
source = source.replace('Diagnostic mean net change and mean number of changed admissions',
                        'Mean net change and mean number of changed admissions')
source = source.replace('these diagnostic variants do not replace',
                        'these operating variants do not replace')
source = source.replace('The reference sensitivity replay reproduces the locked controller exactly.',
                        'The reference sensitivity run reproduces the locked controller exactly.')
source = source.replace('This is a later diagnostic\nextension testing delay-schedule robustness on the same physical traces;',
                        'This extension tests additional delay realizations on the same physical traces;')

source = source.replace('\\subsection{Independent execution audit and reproduction details}',
                        '\\subsection{Execution verification and deployment interfaces}')
source = source.replace('Separate algebra audits test', 'Separate algebra checks verify')
source = source.replace('validated by this wrapper audit', 'validated by this wrapper check')
source = source.replace('independent auditors', 'independent verification scripts')
source = source.replace('auditor output', 'verification output')

(ROOT / 'work/performance_enhanced_draft.tex').write_text(source)
print('Draft prepared; original manuscript and all results unchanged.')
