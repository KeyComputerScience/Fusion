from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'work'
O = ROOT / 'outputs'

def core():
    head = (W / 'bayes_method_head.tex').read_text()
    intro = head.split('\\subsection{One posterior predictive fusion object}')[0]
    intro = intro.replace('The information $\\mathcal F_k$ contains',
                          'The preforecast information $\\mathcal H_k$ contains')
    intro = intro.replace('hidden test regimes, future labels, drift-onset indicators, and outcomes of\n'
                          'unselected updates.',
                          'hidden test regimes, future labels, drift-onset indicators, and outcomes of\n'
                          'unselected updates. Current context, validity, and quality are available\n'
                          'when weights are selected. The current forecast values are then added to\n'
                          'form the action information $\\mathcal F_k=\\mathcal H_k\\vee\n'
                          '\\sigma(\\boldsymbol p_k)$. The predictive residual model conditions on\n'
                          '$\\mathcal H_k$, rather than asserting a full-rank residual Gaussian\n'
                          'conditional on an already fixed entire forecast vector.')
    theory = (W / 'bayes_theory_fragment.tex').read_text()
    theory = theory.replace('y_{r+1}', 'y_r').replace('a_r<k', 'a_r\\le k')
    theory = theory.replace('\\mathcal F_k', '\\mathcal H_k')
    theory = theory.replace('For active mask $\\mathcal A$, retain only origins whose saved forecasts\n'
                            'jointly cover it. With',
                            'For active mask $\\mathcal A$, retain completed origins\n'
                            '$\\mathcal I_k(\\mathcal A)=\\{r: a_r\\le k,\\ r\\ge k-L,\n'
                            '\\mathcal A\\subseteq\\mathcal A_r\\}$ whose saved forecasts jointly cover it.\n'
                            'Use $a_{k,r}=\\beta^{k-r}K(\\boldsymbol x_k,\\boldsymbol x_r)$,\n'
                            '$0<\\beta\\le1$, with the same nonnegative context kernel in every\n'
                            'optimizer control. Exact-mask filtering is an explicitly different\n'
                            'archive treatment. Returning to a larger mask requires genuinely\n'
                            'joint observations of that larger set. With')
    theory = theory.replace('\\mathcal A_r\\}$ whose', '\\mathcal A_r\\}$ whose')
    # The integration contains one predictive object, then its influence rule.
    split = theory.index('\\begin{proposition}[Positive-stakes gain')
    filter_part = theory[:split]
    benefit_part = theory[split:]
    influence = head[head.index('\\subsection{Quality-constrained influence'):]
    influence = influence.replace('\\widehat M_k', '\\widetilde M_k')
    influence = influence.replace('\\widetilde M-\\widetilde M}',
                                  "\\widetilde M-\\widetilde M'}")
    influence = influence.replace('common cap $b$', 'common effective cap $b$')
    influence = influence.replace('is nonempty.',
                                  'is nonempty; in execution the effective cap is\n'
                                  '$b=\\max\\{0.8,1/|\\mathcal A_k|\\}$.')
    # Rename a proof-specific lower stake to avoid collision with the weight cap.
    benefit_part = benefit_part.replace('$(a,b,b)$ with $a>b>0$',
                                       '$(d_+,d_-,d_-)$ with $d_+>d_->0$')
    benefit_part = benefit_part.replace('(aR_{12}+bR_{13}+bR_{23})/3',
                                       '(d_+R_{12}+d_-R_{13}+d_-R_{23})/3')
    benefit_part = benefit_part.replace('(a-b)', '(d_+-d_-)').replace('a-b=', 'd_+-d_-=')
    benefit_part = benefit_part.replace('The structural increment diminishes',
                                       'Positive trace normalization preserves this tangent argument.\n'
                                       'The structural increment diminishes')
    benefit_part = benefit_part.replace('\\begin{proposition}[Finite weighted archive, fixed design]',
                                       '\\begin{lemma}[Finite weighted archive, fixed design]')
    start = benefit_part.index('\\begin{lemma}[Finite weighted archive')
    end = benefit_part.index('\\end{proposition}', start)
    benefit_part = benefit_part[:end] + '\\end{lemma}' + benefit_part[end + len('\\end{proposition}'):]
    theory_parts = filter_part + influence + benefit_part
    # Restore the main action field only in the conditional admission statement.
    theory_parts = theory_parts.replace('true conditional full-horizon net value',
                                       'true conditional full-horizon net value given $\\mathcal F_k$')
    return intro + theory_parts + (W / 'bayes_policy_baselines.tex').read_text()

def pm(metric):
    return f"${metric['mean']:.2f}\\pm{metric['sd']:.2f}$"

def interval(x):
    lo, hi = x['paired_95_normal_interval']
    return f"${x['mean_gain']:.2f}$ & $[{lo:.2f},{hi:.2f}]$"

def main_results(data):
    s = data['summary']
    methods = [('bm_joint','BM-DAF'), ('context_joint','Contextual joint'),
               ('bm_diagonal','Decision diagonal'), ('context_diagonal','Contextual diagonal'),
               ('bm_context_only_belief','Context-only belief'), ('original_c','Original RF-C'),
               ('observed_loss','Observed audit loss'), ('periodic','Periodic'),
               ('random','Random'), ('frozen','Frozen'), ('random_matched','Random, count matched')]
    table = '\n'.join(f"{label} & {pm(s['methods'][m]['net_return'])} & "
                       f"${100*s['methods'][m]['mean_accuracy']['mean']:.2f}$ & "
                       f"${s['methods'][m]['training_steps']['mean']:.1f}$\\\\"
                       for m,label in methods)
    comparisons = [('context_joint','Contextual joint'),('bm_diagonal','Decision diagonal'),
                   ('bm_context_only_belief','Context-only belief'),('original_c','Original RF-C'),
                   ('observed_loss','Observed audit loss'),('periodic','Periodic'),
                   ('frozen','Frozen'),('random_matched','Random, count matched')]
    contrast = '\n'.join(f"{label} & {interval(s['paired_net_return'][m])}\\\\"
                          for m,label in comparisons)
    diag = data['per_seed']
    eig = min(x['diagnostics']['minimum_eigenvalue'] for x in diag)
    changes = sum(x['diagnostics'].get('moment_change_from_prefix',0) for x in diag)/len(diag)
    all_missing = sum(x['diagnostics']['all_missing_slots'] for x in diag)
    def totals(control):
        keys=['different_update_slots','beneficial_local_fork','harmful_local_fork','neutral_local_fork']
        return [sum(x['action_disagreements'][control][k] for x in diag) for k in keys]
    audits='\n'.join(f"{label} & {a} & {b} & {c} & {d}\\\\"
                     for label, (a,b,c,d) in [(label,totals(m)) for m,label in
                     [('context_joint','Contextual joint'),('bm_diagonal','Decision diagonal'),
                      ('original_c','Original RF-C')]])
    tests=data['causal_tests']
    return rf'''
\subsection{{Measured closed-loop outcomes and attribution}}
\label{{sec:bm-measured-outcomes}}

Table~\ref{{tab:bm-main}} reports the time-aligned final execution, with
mean $\pm$ between-seed SD for return. Source errors are learned from arrived
empirical targets while the physical relation, error correlation, and masks
change. These are executed policy outcomes, not a known-population-matrix
threshold construction.

\begin{{table}}[htbp]
\centering\small
\caption{{Twenty independent controlled service streams. Accuracy is potential
classification over all jobs before capacity drops; net return counts only
served correct jobs and subtracts actual step fees.}}
\label{{tab:bm-main}}
\begin{{tabular}}{{lrrr}}\toprule
Method & Net return & Accuracy (\%) & Gradient steps\\\midrule
{table}
\bottomrule\end{{tabular}}
\end{{table}}

\begin{{table}}[htbp]
\centering\small
\caption{{Paired BM-DAF minus control net return. Intervals are descriptive
seed-level 95\% normal intervals, with no multiplicity correction.}}
\label{{tab:bm-paired}}
\begin{{tabular}}{{lrr}}\toprule
Control & Mean gain & Paired interval\\\midrule
{contrast}
\bottomrule\end{{tabular}}
\end{{table}}

The comparison with frozen and observed-audit-loss policies assesses the value
of the complete updating policy. The comparison with contextual joint fusion
assesses the incremental value of decision-weighted cross-source structure.
Their interpretations are deliberately separate. An interval spanning zero
does not establish a stable incremental benefit; a positive point estimate
does not justify a superiority claim. The reported controls also prevent
generic updating, larger expenditure, or different available information from
being attributed to the new fusion estimand.

The delayed-data implementation changes the empirical moment relative to its
prefix prior: mean Frobenius change is ${changes:.5f}$. All retained active
matrices are PSD (minimum observed eigenvalue ${eig:.5f}$). There are
{all_missing} all-source-missing decision slots across the formal seeds,
with abstention. No solver failure or too-old label discard is observed.
These diagnostics show that delayed adaptive estimation executes; they do not
establish a consistency rate. Predictive-to-realized-state covariance discrepancy
includes irreducible latent-regime uncertainty and is not a pure matrix
estimation-error statistic.

\begin{{table}}[htbp]
\centering\small
\caption{{Actual launch disagreements over the twenty streams. Local advantages
use evaluation-only paired SGD forks from BM-DAF's state; they are not additive
whole-policy effects.}}
\label{{tab:bm-actions}}
\begin{{tabular}}{{lrrrr}}\toprule
Control & Different slots & Beneficial & Harmful & Neutral\\\midrule
{audits}
\bottomrule\end{{tabular}}
\end{{table}}

Most physical slots have unchanged update actions despite different fusion
weights. This limits the empirical size of the core mechanism and is reported
explicitly. Bayesian regime history and full joint structure are useful modeling
choices, but their individual incremental return must be established by the
corresponding ablations rather than by counting new components.

The replay/checkpoint test crosses the retained-history boundary and agrees
with full-history filtering to ${tests['late_factor_replay_max_abs_error']:.3g}$.
Future-data mutation leaves earlier posteriors, models, and adaptive actions
unchanged. Training-arrival, deployment-delay, actual gradient, and exactly
matched count/cost contracts pass. These are software/protocol checks, not
evidence of real-world sensor performance or hardware latency.
'''

def references():
    old=(W/'risk_rf_before_bayesian_revision.tex').read_text()
    chosen=['bates1969','boyd2004','qmf2023','pdf2024','joulani2013']
    entries=[]
    for key in chosen:
        m=re.search(r'\\bibitem\{'+key+r'\}.*?(?=\n\\bibitem|\n% Official publisher|\n\\end\{thebibliography\})',old,re.S)
        if m:
            entry=m.group(0).split('\n%')[0].strip()
            entries.append(entry)
    extra=r'''
\bibitem{rabiner1989}
L. R. Rabiner, ``A tutorial on hidden Markov models and selected applications
in speech recognition,'' \emph{Proceedings of the IEEE}, vol. 77, no. 2,
pp. 257--286, 1989. \url{https://doi.org/10.1109/5.18626}.
\bibitem{oosm2003}
S. Challa, R. J. Evans, and X. Wang, ``A Bayesian solution and its
approximations to out-of-sequence measurement problems,'' \emph{Information
Fusion}, vol. 4, no. 3, pp. 185--199, 2003.
\url{https://doi.org/10.1016/S1566-2535(03)00037-X}.
\bibitem{spo2022}
A. N. Elmachtoub and P. Grigas, ``Smart `Predict, then Optimize',''
\emph{Management Science}, vol. 68, no. 1, pp. 9--26, 2022.
\url{https://doi.org/10.1287/mnsc.2020.3922}.
\bibitem{lounici2014}
K. Lounici, ``High-dimensional covariance matrix estimation with missing
observations,'' \emph{Bernoulli}, vol. 20, no. 3, pp. 1029--1058, 2014.
\url{https://doi.org/10.3150/12-BEJ487}.
'''
    return '\n\\begin{thebibliography}{99}\n'+'\n\n'.join(entries)+extra+'\n\\end{thebibliography}\n\\end{document}\n'

if __name__=='__main__':
    data=json.loads((W/'bayes_results_main.json').read_text())
    # Final sensitivity section is assembled from executed artifacts separately.
    tail=(W/'bayes_sensitivity_results.tex').read_text() if (W/'bayes_sensitivity_results.tex').exists() else ''
    text=core()+main_results(data)+tail+references()
    (W/'risk_rf_bayesian_draft.tex').write_text(text)
    print('draft',len(text.splitlines()),len(text.split()),'words/tokens')
