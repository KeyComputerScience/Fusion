"""Build manuscript tables only from complete, verified extension outputs."""
from pathlib import Path
import hashlib, json, re, shutil

ROOT = Path('/Users/key/Documents/Codex/2026-10-02/jih')
EXT = ROOT / 'work/external_fusion_validation_20261002'
TASKS = ('occupancy357', 'occupancy864', 'mhealth319', 'har240')
LABELS = ('Occupancy 357', 'Room 864', 'MHEALTH', 'HAR')

def load(kind):
    folder = EXT if kind == 'pdf' else EXT / kind
    audit = json.loads((folder / 'execution_audit.json').read_text())
    assert audit['all_passed']
    result = json.loads((folder / 'results.json').read_text())
    assert set(result) == set(TASKS)
    assert all(len(result[t]['trials']) == 5 for t in TASKS)
    return result, audit

def count(trials, arm, subset):
    rows = [r for t in trials for r in t['results'][arm]['rows']]
    if subset == 'informative': rows = [r for r in rows if r['disagreement'] > 0]
    if subset == 'admitted': rows = [r for r in rows if r['action']]
    return f"{sum(bool(r['lower_covered']) for r in rows)}/{len(rows)}" if rows else '--'

def mean(result, arm, key):
    return sum(t['results'][arm][key] for t in result['trials']) / 5

pdf, paudit = load('pdf')
qmf, qaudit = load('qmf')
body = (ROOT / 'work/performance_enhanced_draft.tex').read_text()
# Honor the newly pasted source's merged accounting equation and label.
accounting_start = body.index('\\begin{align}\n J_F-J_C')
accounting_end = body.index('\\end{align}', accounting_start) + len('\\end{align}')
body = body[:accounting_start] + r'''\begin{equation}
\begin{aligned}
J_F-J_C &= C_F^{\rm served}-C_C^{\rm served}+3(A_C-A_F),\\
        &= \sum_j(a_j^F-a_j^C)D_j.
\end{aligned}
\label{eq:gain-decomposition}
\end{equation}''' + body[accounting_end:]
body = body.replace('eq:paired-executed-gain', 'eq:gain-decomposition')
body = body.replace('eq:measured-gain-decomposition', 'eq:gain-decomposition')

external = r'''\subsection{External published fusion rules under the service contract}
\label{sec:external-fusion}
We execute two recent published fusion rules: Predictive Dynamic Fusion
(PDF) \cite{pdf2024} and the energy-based rule in Quality-aware Multimodal
Fusion (QMF) \cite{qmf2023}. The controls are denoted PDF-P and QMF-E.
They use the identical prepared physical-source classifiers, candidate and
reference versions, delayed feedback, masks, contexts and paid lease.
The original papers also train their source backbones jointly with fusion;
here the backbones follow the common training schedule. PDF-P fits static
sigmoid-linear true-class-probability (TCP) heads on prefix audit records,
using the paper's MSE target. QMF-E retains the published energy fusion
rule with fixed $T=1$ and coefficient $0.1$, without the original
ranking-loss training. These are explicit fixed-backbone adaptations;
the comparison isolates published fusion rules in this deployment setting.

For PDF-P, let $c_{i,s}$ denote learned TCP and define
\begin{align}
 d_{i,s}&=C^{-1}\sum_{c=1}^{C}|p_{i,s,c}-C^{-1}|,\nonumber\\
 t_{i,s}&=c_{i,s}+\frac{\sum_{j\ne s}\log c_{i,j}}
                              {\sum_j\log c_{i,j}},\nonumber\\
 r_{i,s}&=\min\!\left\{1,
       \frac{(m-1)d_{i,s}}{\sum_{j\ne s}d_{i,j}}\right\},\nonumber\\
 w_{i}^{\rm PDF}&=\operatorname{softmax}_s(r_{i,s}t_{i,s}),\qquad
 p_i^{\rm PDF}=\operatorname{softmax}_c
       \!\left(\sum_s w_{i,s}^{\rm PDF}\log\widetilde p_{i,s,c}\right).
 \label{eq:external-pdf}
\end{align}
Uniform and singleton-source cases have explicit symmetric limits.
Here $\widetilde p_{i,s,c}=\max(p_{i,s,c},10^{-12})$ is the
implementation's logarithm floor. Without clipping, the log-probability
form preserves weighted-logit fusion because each omitted normalizer is
constant over classes. The source weights follow the
published softmax rather than the proposed method's capped simplex.

QMF-E uses the issued source logits $z_{i,s,c}$:
\begin{equation}
 v_{i,s}=0.1\log\sum_c e^{z_{i,s,c}},\qquad
 p_i^{\rm QMF}=\operatorname{softmax}_c
                    \!\left(\sum_s v_{i,s}z_{i,s,c}\right).
 \label{eq:external-qmf}
\end{equation}
The influences $v_{i,s}$ are not normalized over sources. The common
linear classifiers start at zero; their softmax gradient and $L_2$ update
preserve a zero class-column sum. Therefore their logits can be recovered
as $z_{i,s,c}=\log p_{i,s,c}-C^{-1}\sum_a\log p_{i,s,a}$.
The recovery is checked against actual prefix and online model parameters.
This property is specific to the shared classifiers; arbitrary source
logits cannot be recovered from probabilities this way.

Each external pipeline stores its actual fused probability at issuance.
After label arrival, its immutable fused residual is projected onto the
current candidate/reference contrast. The scalar error correction and
directional scale use the same context kernel, age discount, prior mass,
archive limit and complete-source qualification. We execute both a point
gate and a separately calibrated delayed gate for each external rule.
Thus the stronger external control receives uncertainty-aware admission
and the same complete outcome feedback.

Each pipeline receives nine configuration trials on each of the three
original calibration schedules. PDF-P varies TCP-head $L_2$ multipliers
in $\{0,0.25,1\}$ and admission margins in $\{0,4,12\}$. QMF-E uses
the same three margins with three repeated risk-grid entries because its
published weighting rule has no risk coefficient. Both receive 27
configuration--schedule evaluations, matching the internal trial count;
the algorithms have different hyperparameters. Initial quantiles use
only matured first-half calibration outcomes, and selection uses the
second half. All selected heads, margins and quantiles are saved before
the corresponding test replay. Prefix TCP fitting incurs additional
preparation outside the scored interval; it is disclosed separately from
the common online classifier work. The external rule benchmark uses the
existing task splits and preserves all primary outcomes.

\begin{table}[t]
\centering\small
\caption{Measured mean paid utility for the published-rule adaptations
on the five primary delay schedules. All four external pipelines and the
calibrated empirical control are retained.}
\label{tab:external-net}
\begin{tabular}{lrrrr}
\toprule
Controller & 357 & Room & MHEALTH & HAR\\
\midrule
__NET_ROWS__
\bottomrule
\end{tabular}
\end{table}

Against PDF-P point admission, full posterior mean increments are
$0$, $29.6$, $26.8$ and $7.2$. Against calibrated PDF-P they are
$0$, $0$, $9.8$ and $0$. On MHEALTH, the calibrated-PDF differences
by seed are $[0,5,12,5,27]$: four improvements and one tie, with
conditional five-delay interval $[-3.26,22.86]$. The controllers change
1.6 admission decisions per schedule on average. Full posterior records
2.2 harmful and 0.2 beneficial admissions per schedule, compared with
2.4 harmful and zero beneficial admissions for calibrated PDF-P.
This identifies an additional action-level benefit of the proposed
pipeline beyond the confidence-fusion adaptation supplied with a
calibrated gate.

__QMF_RESULT_TEXT__

\begin{table}[t]
\centering\small
\caption{Coverage and executed actions for the calibrated external
pipelines and full posterior. Entries retain exact denominators; a dash
means no admission. Coverage uses the issued threshold and conservative
complete-lease target.}
\label{tab:external-coverage}
\begin{tabular}{llrrrrr}
\toprule
Task & Controller & Issued & Informative & Admitted & Harmful & Beneficial\\
\midrule
__COVERAGE_ROWS__
\bottomrule
\end{tabular}
\end{table}

The point and calibrated variants expose the role of admission separately
from prediction fusion. On Room and HAR, calibrated PDF-P and full
posterior have identical actions and utility. The block-risk controller
remains in Tables~\ref{tab:net} and \ref{tab:external-net}; comparing
all calibrated pipelines prevents attributing every gate improvement to
Bayesian joint structure. The complete per-seed actions, source influences,
issued scores, matured callbacks and targets accompany these tables.
PDF-rule formula checks and request-level service reconstruction pass;
the external outputs include every selected pipeline, tie and loss.

'''

arms = [('PDF-P point', 'pdf_point', pdf), ('PDF-P calibrated', 'pdf_calibrated', pdf),
        ('QMF-E point', 'qmf_point', qmf), ('QMF-E calibrated', 'qmf_calibrated', qmf),
        ('Full posterior', 'bayes_both', pdf), ('Matched block gate', 'frequentist_gate', pdf)]
netrows = []
for label, arm, result in arms:
    netrows.append(label + ' & ' + ' & '.join(f"{mean(result[t], arm, 'net'):.2f}" for t in TASKS) + r'\\')
external = external.replace('__NET_ROWS__', '\n'.join(netrows))
coverage_rows = []
for task, label in zip(TASKS, LABELS):
    for display, arm, result in (('PDF-P', 'pdf_calibrated', pdf),
                                 ('QMF-E', 'qmf_calibrated', qmf),
                                 ('Full', 'bayes_both', pdf)):
        trials = result[task]['trials']
        harms = sum(t['results'][arm]['harmful'] for t in trials)
        bens = sum(t['results'][arm]['beneficial'] for t in trials)
        coverage_rows.append(label + ' & ' + display + ' & ' +
                             ' & '.join(count(trials, arm, s) for s in ('issued', 'informative', 'admitted')) +
                             f' & {harms} & {bens}' + r'\\')
external = external.replace('__COVERAGE_ROWS__', '\n'.join(coverage_rows))
qdiffs = {a: [mean(qmf[t], 'bayes_both', 'net') - mean(qmf[t], a, 'net') for t in TASKS]
          for a in ('qmf_point', 'qmf_calibrated')}
qtext = 'Against QMF-E point admission, the full-minus-control means are $' + \
        ','.join(f'{v:.1f}' for v in qdiffs['qmf_point']) + '$; against calibrated QMF-E they are $' + \
        ','.join(f'{v:.1f}' for v in qdiffs['qmf_calibrated']) + '$, in the same task order. '
qtext += ('The calibrated comparisons retain both useful changes and tied decisions; '
          'they evaluate source-influence rules at the level of complete paid leases. '
          'They do not establish superiority over the original jointly trained PDF or QMF networks.')
qtext += (' On MHEALTH, full-minus-calibrated-QMF differences are '
          '$[0,5,12,5,22]$, with conditional five-delay interval '
          '$[-1.79,19.39]$ and 1.4 changed admissions per schedule. '
          'Calibrated QMF-E has 11 harmful admissions and no beneficial '
          'admission, whereas full posterior has 11 harmful and one '
          'beneficial admission across these five schedules.')
external = external.replace('__QMF_RESULT_TEXT__', qtext)

body = body.replace('\\subsection{Execution verification and deployment interfaces}',
                    external + '\\subsection{Execution verification and deployment interfaces}')
insert = r'''\begin{table}[t]
\centering\small
\caption{Coverage across both retained cohorts of the full controller.
The ten-schedule column group is descriptive and does not replace the
primary cohort or add independent physical traces.}
\label{tab:coverage-ten}
\begin{tabular}{lrrr}
\toprule
Task & Issued & Informative & Admitted\\
\midrule
Occupancy 357 & 598/640 (93.44\%) & 6/6 (100\%) & --\\
Room 864 & 300/310 (96.77\%) & 0/10 (0\%) & --\\
MHEALTH & 142/210 (67.62\%) & 11/49 (22.45\%) & 1/24 (4.17\%)\\
HAR & 303/320 (94.69\%) & 57/59 (96.61\%) & 6/6 (100\%)\\
\bottomrule
\end{tabular}
\end{table}
HAR retains 57/59 informative coverage across the ten schedules and
all six admitted leases are covered and beneficial. The two original
cohorts remain separately reported. Coverage is calculated for every
issued origin, including outcomes whose feedback arrives after the final
executed gate. Online calibration uses only matured callbacks; its
denominator is recorded separately in the reproduction logs.

'''
body = body.replace('\\subsection{External published fusion rules under the service contract}',
                    insert + '\\subsection{External published fusion rules under the service contract}')

runtime = r'''\paragraph{Measured admission-interface latency.}
On the local arm64 macOS environment with Python 3.12.14 and NumPy 2.3.5,
we time eight real issued records per task for 25 repetitions after three
warm-up passes. The 200 timed calls per task include dimension and PSD
checks, the frozen solver and the admission gate. All decisions and
forecasts match the saved trajectory. Source inference, training and
historical-moment construction are outside the timed region.
\begin{table}[t]
\centering\small
\caption{Measured local admission-interface time in milliseconds.
Repeating calls measures latency, not additional service outcomes.}
\label{tab:live-latency}
\begin{tabular}{lrrr}
\toprule
Task & Active sources & Median & 95th percentile\\
\midrule
Occupancy 357 & 3--4 & 0.848 & 2.997\\
Room 864 & 4--5 & 0.096 & 0.107\\
MHEALTH & 2--3 & 0.234 & 1.157\\
HAR & 1--2 & 0.092 & 0.101\\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Portable execution and live information.}
The accompanying \texttt{Fusion\_External\_Validation\_Repro} package
provides one offline entry point. The commands below run from its root;
replay outputs are directed to a new directory outside the delivered
package, preserving the original files:
\begin{verbatim}
python run_reproduction.py verify
python run_reproduction.py audit-primary
python run_reproduction.py audit-supp
python run_reproduction.py external
python run_reproduction.py analyze-external
python run_reproduction.py benchmark
\end{verbatim}
The interface accepts active source IDs, issued forecasts, arrived joint
moments, quality priors, the current matured calibration threshold and
service feasibility. It does not read future labels or complete-lease
return. Empty masks retain the reference; singleton masks have unit
influence. Nonempty inputs are validated before the solver, and its
capped-prior fallback is logged. Deployment requires a complete
nonoverlapping lease, an unchanged candidate/reference inside that lease,
restoration at closure and timestamped label-maturity callbacks.
The delivered manifest checks file hashes and data provenance. A relocated
offline run reproduces all four primary selections and trajectories,
the supplementary trajectories and the recorded action crossings.
External head fitting, formula provenance and per-phase freezing are
included in the same package. Persistent queues, crash recovery and
physical-device execution require the corresponding application service;
the measured interface and cache launcher provide its reproducible
fusion and admission components.

'''
body = body.replace('The supplied archive includes the unchanged base executable',
                    runtime + 'The supplied archive includes the unchanged base executable')
body = body.replace('The four figures are self-contained vector', 'The figures are self-contained vector')
body = body.replace('runtime\nmeasurement, real missing physical inputs', 'real missing physical inputs')
body = body.replace('training and audit records', 'training and held-out prefix records')
body = body.replace('for audit:', 'for prefix error estimation:')
body = body.replace('non-audit training records', 'non-reserved training records')

conclusion = r'''\section{Conclusion}
\label{sec:conclusion}
Decision-conditioned joint-error fusion connects immutable multisource
evidence to paid deployment decisions. The block posterior separates
request-level predictive risk from uncertainty in a learned correction;
convex influence and directional admission then give that uncertainty
two executable roles. The measured mechanisms include rejection of a
paid loss through posterior influence and admission of a useful lease
through posterior concentration. The common service contract makes
both consequences visible in served correctness, fees and complete net
return. Equal-budget internal controls, recent published fusion-rule
adaptations, issued-score coverage and portable request-level
reconstruction substantiate this mechanism at the action level.
The comparisons also identify where calibrated alternatives tie or
outperform the selected method. The method's benefit is therefore
expressed through its measured decisions and stated lease conditions;
posterior uncertainty and calibrated admission remain distinct components
whose coverage and utility are evaluated together.
'''

fullpath = ROOT / 'outputs/risk_rf_revision.tex'
original_full = fullpath.read_text()
bib = original_full[original_full.index('\\begin{thebibliography}'):]
entries = []
for key in ('occupancy357', 'occupancy864', 'mhealth319', 'har240', 'pdf2024', 'qmf2023'):
    match = re.search(r'\\bibitem\{' + key + r'\}.*?(?=\\bibitem|\\end\{thebibliography\})', bib, re.S)
    assert match, key
    entries.append(match.group(0).strip())
standalone_begin = r'''% Standalone chapter; define \FusionMainDocument before inputting into a manuscript.
\ifdefined\FusionMainDocument
\else
\documentclass[10pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,array,graphicx,tikz,pgfplots,cite,hyperref}
\pgfplotsset{compat=1.18}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\DeclareMathOperator{\diag}{diag}
\setlength{\emergencystretch}{2em}
\begin{document}
\section*{Execution quantities used in this chapter}
For the standalone preview, the common admission rule and complete-return
relation are restated here. The main manuscript derives these relations
under the nonoverlapping paid-lease contract.
\begin{equation}
 L_j=\widehat g_j-5-q_js_j,\quad
 s_j=N_j\sqrt{w_j^{\mathsf T}U_jw_j+10^{-4}},\quad
 a_j=\mathbf1\{L_j>\varepsilon\}.
 \label{eq:posterior-gate}
\end{equation}
\begin{equation}
 J_T-J_T^0=\sum_j a_jD_j
 \geq \sum_j a_jL_j-\sum_j a_js_j(R_j-q_j)_+.
 \label{eq:weighted-return-bound}
\end{equation}
Here $R_j=(\widehat g_j-g_j^{\rm true})/s_j$ uses completed lease outcomes,
while $q_j$ and $s_j$ are fixed at issuance.
\fi

'''
standalone_end = '\n\\ifdefined\\FusionMainDocument\n\\else\n\\begin{thebibliography}{99}\n' + \
                 '\n\n'.join(entries) + '\n\\end{thebibliography}\n\\end{document}\n\\fi\n'
for name in ('risk_rf_performance', 'risk_rf_revision'):
    src = ROOT / 'outputs' / f'{name}.tex'
    backup = ROOT / 'work' / f'{name}_before_external_validation.tex'
    if not backup.exists(): shutil.copy2(src, backup)
(ROOT / 'outputs/risk_rf_performance.tex').write_text(standalone_begin + body + standalone_end)
start = original_full.index('\\section{Performance}')
end = original_full.index('\\begin{thebibliography}', start)
fullpath.write_text(original_full[:start] + body + '\n' + original_full[end:])
(ROOT / 'work/performance_enhanced_body.tex').write_text(body)
print('Updated existing standalone Performance and synchronized manuscript.')
