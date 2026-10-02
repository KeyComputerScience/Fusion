from pathlib import Path
import re

root = Path('/Users/key/Documents/Codex/2026-10-02/jih')
old = (root/'work/risk_rf_before_methodology.tex').read_text()
def block(a,b):
    return old[old.index(a):old.index(b)]
preamble = old[:old.index('\\title')]
opening = r'''\begin{document}
\section{Methodology}
\label{sec:methodology}

We develop decision-aware reliability fusion (DA-RF) for services in which
incomplete operational evidence informs costly, delayed model updates.
The central benefit is to reduce decision-relevant joint prediction error:
sources with identical marginal accuracy can have different value when their
errors coincide in consequential decision contexts. Measurement quality,
common-target residual structure, and intervention recoverability are estimated
separately. The resulting weights feed a persistent-loss monitor and a
candidate-specific update score; every update must execute gradient steps,
consume its declared resources, and deploy a completed model.

Three claims are distinguished. Joint structure can strictly reduce executed
decision regret in a matched-marginal construction. Decision weighting can add
information beyond a scalar change of risk scale in a pooled-context
construction. A service update is admitted with a conditional net-value
certificate only when its independently calibrated recovery value covers
cost, estimation allowances, and uncertainty exposure. None of these statements
alone establishes neural-policy recovery in a new service experiment.

Forecast combination, regularization, and dynamic reliability have established
foundations~\cite{bates1969,bregman1967,boyd2004,diebold2019,qmf2023,pdf2024,
huang2025}. Our contribution is the decision-weighted joint-error estimand,
its quality-constrained estimation and influence rule, and the explicit bridge
from fusion sensitivity to update admission and executed benefit. We do not
claim a new KL or quadratic optimization primitive. Throughout, RF means
\emph{reliability fusion}, not random forest. Original RF denotes the supplied
multiplicative-quality baseline; previous Risk RF denotes the already evaluated
unweighted joint-risk implementation; DA-RF denotes the extension defined here.

\begin{figure}[htbp]
\centering\small
\fbox{\begin{minipage}{0.93\linewidth}\centering
\textbf{Completed observations, validity masks, and saved forecasts}\\[2pt]
$\Downarrow$\\[-1pt]
\textbf{Common-target prediction and separate observation quality}\\
Persistent loss $y$; diagnostics $d$; quality $q$; delayed residuals $e$\\[2pt]
$\Downarrow$\\[-1pt]
\textbf{Decision-weighted joint moments and constrained fusion}\\
$\widehat M$, empirical buffer $\delta$, KL reference, inertia, masked cap\\[2pt]
$\Downarrow$\\[-1pt]
\textbf{Persistent-loss admission and full-horizon candidate value}\\
Recovery support, uncertainty $UQ_i$, capacity, delay, and active costs\\[2pt]
$\Downarrow$\\[-1pt]
\textbf{Executed update or no update; measured deployment and outcomes}\\
Delayed labels update saved forecasts; paired interventions update recovery.
\end{minipage}}
\caption{DA-RF methodology. Forecast reliability and update recoverability have
different labels. Missing evidence cannot create an update certificate.}
\label{fig:framework}
\end{figure}
'''

evidence = block('\\subsection{Time, execution, and data provenance}',
                 '\\subsection{Quality reference, delayed residual risk, and masked fusion}')
evidence = evidence.replace('A consistent external reward is',
r'''Write gross service utility as
$u_t=s_t-\alpha_\ell\ell_t-\alpha_q\bar q_t$.
A consistent external reward is''')
evidence = evidence.replace('All existing operational results used in this revision are synthetic.',
                            'The operational results supplied with the manuscript are synthetic.')
start = evidence.index('DA-RF therefore predicts persistent loss')
end = evidence.index('Every source issues a forecast')
evidence = evidence[:start]+r'''DA-RF therefore predicts persistent loss rather than relying exclusively on the
positive adjacent-window difference. After selecting inference profile $j_t$,
but before service, issue
\begin{equation}
 \widehat u_t^{\rm ref}=g_{\rm ref}(\boldsymbol x_t^{\rm exo},B_t,j_t),
\end{equation}
using a prefix-fitted, frozen model and only currently known exogenous arrivals,
capacity, and profile information. Save its inputs and issuance time.
Completion, final queue state, and subsequent window summaries cannot be inputs
to that reference. For the same valid slot set $\mathcal V_k\subseteq\mathcal T_k$,
\begin{equation}
 J_k^{\rm ref}=\frac1{|\mathcal V_k|}\sum_{t\in\mathcal V_k}\widehat u_t^{\rm ref},
 \qquad J_k=\frac1{|\mathcal V_k|}\sum_{t\in\mathcal V_k}u_t,
 \qquad
 y_k=\clip\left(\frac{J_k^{\rm ref}-J_k}{B_J},0,1\right),\ B_J>0.
 \label{eq:persistent-target}
\end{equation}
The target is unavailable below a declared valid-slot support threshold.
Gross utility excludes the separately charged direct training fee while
retaining resource-mediated service losses. Reference extrapolation, composition
changes, and policy effects can confound $y_k$: it is an operational loss target,
not an identified causal drift probability. The previous Risk RF used its
original next-window degradation target; changing the target is a separate
experimental factor.

Distinguish physical observation validity $m^{\rm obs}_{s,k}$, diagnostic validity
$m^d_{s,k}$, and forecast validity $m^p_{s,k}$. A service source with fresh absolute
loss but no matched adjacent contexts may issue a supported \emph{loss-only}
forecast; its adjacent diagnostic remains unavailable. This mode and its support
are logged. Forecast availability requires valid source observations, a fitted
predictor, and declared age/support limits. It cannot be inferred merely from a
quality floor or an imputed diagnostic.

'''+evidence[end:]
start = evidence.index('A minimal declared feature vector')
end = evidence.index('The common service history')
evidence = evidence[:start]+r'''A declared feature vector is
\begin{equation}
 \boldsymbol\phi_{s,k}=(1,m^d_{s,k}d_{s,k},m^d_{s,k-1}d_{s,k-1},
 m^y_k y_k,m^y_k,m^d_{s,k},m^d_{s,k-1})^{\mathsf T},
\end{equation}
where $m^y_k$ records target validity and the products with an unavailable value
are implemented as masked feature entries, not fabricated observations.
'''+evidence[end:]

quality = block('\\subsection{Quality reference, delayed residual risk, and masked fusion}',
                'Let $\\mathcal A_k$ be the valid enabled sources')
quality = quality.replace('Quality reference, delayed residual risk, and masked fusion',
                          'Observation quality and delayed unweighted residuals')
quality = quality.replace('R_k=\\beta R_{k-1}+(1-\\beta)',
                          'R_k=\\beta_R R_{k-1}+(1-\\beta_R)')
quality = quality.replace('0\\le\\beta<1', '0\\le\\beta_R<1')
quality += r'''The unweighted matrix is retained as a diagnostic and as the Risk RF
control. The proposed optimizer instead uses the decision-weighted matrix
defined next. Quality factors and residual moments are never merged into one
uninterpretable reliability product.

'''
moment = (root/'work/method_decision_moment.tex').read_text()
moment = moment.replace('The response coefficients are fitted',
'''The coefficients are independent of the current fusion weights and are fitted''')
moment = moment.replace('The population target is\n$M=\\E[B^2',
r'''The estimation target is a declared local pooled distribution of decision
contexts. Its population moment is $M=\E[B^2''')
moment = moment.replace('for the\nrelevant decision-context distribution.',
r'''for that distribution. The information $\mathcal F$ here freezes the
combination rule before the associated evaluation contexts; it does not include
their as-yet unobserved stakes.''')
moment += r'''For a current context whose $B$ is already known under the complete
conditioning information, $M=B^2\E[\boldsymbol e\boldsymbol e^{\mathsf T}
\mid\mathcal F]$. Thus an exactly conditioned current-state analysis cannot
attribute a new matrix shape to decision weighting. Its possible additional
benefit concerns pooling over varying stakes/error regimes, or a precommitted
rule for future decision contexts. The historical estimator is a policy for
that pooled objective; interpreting it as the next conditional moment requires
the separately checked estimation-error condition below.

At each window boundary, learn from a rolling archive of completed decision
cases and freeze the weights throughout the following window. The empirical
criterion is
\begin{equation}
 \widehat{\mathcal L}_k(\boldsymbol w)=
 \frac{a_0\boldsymbol w^{\mathsf T}M_0\boldsymbol w+
 \sum_{r\in\mathcal I_k}a_{k,r}\chi_r
 (\boldsymbol w^{\mathsf T}\boldsymbol e_r)^2}
 {a_0+\sum_{r\in\mathcal I_k}a_{k,r}}
 =\boldsymbol w^{\mathsf T}\widehat M_k\boldsymbol w.
\end{equation}
We call $\widehat M_k$ a local pooled archive moment. A narrower context kernel
can improve matching while removing the variation in stakes that makes
decision weighting distinct. Test this support--specificity tradeoff explicitly.

'''

fusion = block('Let $\\mathcal A_k$ be the valid enabled sources',
               '\\subsection{Original RF and matched fusion baselines}')
fusion = r'''\subsection{Quality-constrained robust fusion and its decision state}
\label{sec:fusion-rule}

'''+fusion
fusion = fusion.replace('be the valid enabled sources',
                       'be the enabled sources with $m^p_{s,k}=1$')
fusion = fusion.replace('R_k', 'S_k')
fusion = fusion.replace('and\n$\\bar a_k=', 'and, when $N_k>0$,\n$\\bar a_k=')
fusion = fusion.replace('launches no new training.',
'''launches no new training, including probes. This all-missing rule has
priority over every subsequent admission exception.''')
start = fusion.index('The fusion state retains different interpretations:')
end = fusion.index('The nonnegative coefficients are selected')
fusion = fusion[:start]+r'''The decision state retains distinct interpretations:
\begin{align}
 \widehat y_{k+1|k}&=\sum_{s\in\mathcal A_k}w_{s,k}p_{s,k},&
 D_k^p&=\sum_{s\in\mathcal A_k}w_{s,k}(p_{s,k}-\widehat y_{k+1|k})^2,\\
 c_k&=N_k/S_{\rm enabled},&
 r_k^{\rm dec}&=\boldsymbol w_k^{\mathsf T}S_k\boldsymbol w_k,\quad
 r_k^{\rm pred}=\boldsymbol w_k^{\mathsf T}R_k\boldsymbol w_k.
 \label{eq:fusion-state}
\end{align}
With $Z_k^d=\sum_{s\in\mathcal A_k}w_{s,k}m^d_{s,k}>0$, the descriptive
diagnostic is $\Gamma_k=\sum_s w_{s,k}m^d_{s,k}d_{s,k}/Z_k^d$.
If $Z_k^d=0$, it is unavailable. Diagnostic coverage is reported separately;
missing diagnostics are not zero change scores. $\Gamma_k$ is not a drift
probability and is not the mandatory update gate.
Disagreement compares forecasts of one target. Agreement alone does not
establish accuracy, because all sources can share an error.
The exposure index is
\begin{equation}
 U_k=\min\left\{1,\omega_D4D_k^p+\omega_M(1-c_k)
       +\omega_R\frac{r_k^{\rm dec}}{B_U^2}
       +\omega_A\frac{A_k^{\rm risk}}{A_{\max}}\right\},\quad B_U>0.
 \label{eq:decision-uncertainty}
\end{equation}
$B_U$ is a prefix-fixed reward scale, since $M$ has squared-utility units.
'''+fusion[end:]

decision = (root/'work/decision_risk_fragment.tex').read_text()
decision = decision[:decision.index('For a causal estimator, store')]
decision = decision.replace('\\mathscr', '\\mathcal')
decision = decision.replace('The global sensitivity proposition remains valid with $R$ replaced by\n$S$.',
'''The sensitivity result below uses $S$.''')
decision += r'''A weight fixed before a future test context need not have independent
stakes and errors for this bound. If the stakes are already measurable under
the conditioning information, the weighted moment reduces to a scalar times
the conditional unweighted moment. The result then remains valid but does not
provide a new joint structure. The regret comparison uses a common fixed
candidate set; forecast-dependent changes of admission are analyzed separately.

'''

theory = block('\\subsection{Stability and the incremental value of joint residual risk}',
               '\\subsection{Propagation to admission and executed profile selection}')
start = theory.index('Fix an availability mask')
split = theory.index('\\begin{proposition}[Complementarity at matched marginal error]')
stability = theory[start:split]
stability = stability.replace('R_D', 'S_D').replace('\\widetilde R','\\widetilde S')
stability = re.sub(r'\bR\b','S',stability)
stability = r'''\subsection{Sensitivity to quality, residual risk, and historical weights}
\label{sec:joint-risk-theory}

'''+stability
stability += r'''For the operational input $S=\widehat M+\delta I$,
\begin{equation}
 \|S-\widetilde S\|_{\rm op}
 \le\|\widehat M-\widetilde{\widehat M}\|_{\rm op}
       +|\delta-\widetilde\delta|.
\end{equation}
When both radii exceed $\delta_{\min}$, the denominator can be sharpened to
$\tau/b+\kappa+\lambda\delta_{\min}$. Increasing the buffer can therefore
reduce concentration, but cannot establish that the forecasts or updates are
correct. Floors can make a quality factor locally inactive; cap $b=1/n$ makes
all weights uniform. These are declared negative controls.

The radius and inertia have an explicit overlap. On the simplex, their
sum equals, up to a constant,
\begin{equation}
 \frac{\kappa+\lambda\delta}{2}\|\boldsymbol w-\boldsymbol a_{\rm eff}\|_2^2,
 \qquad
 \boldsymbol a_{\rm eff}=
 \frac{\kappa\boldsymbol a+\lambda\delta\boldsymbol u}
      {\kappa+\lambda\delta},\quad \boldsymbol u=\boldsymbol1/n,
\end{equation}
when $\kappa+\lambda\delta>0$. Only a uniform historical anchor allows the
radius to be described solely as additional inertia. With a general anchor,
it also pulls the effective reference toward uniform weights. Radius and
history are therefore varied separately in the sensitivity analysis.

\paragraph{Prespecified sensitivity protocol.}
Analytical bounds are supplemented by paired perturbations on saved forecasts.
For each available source, perturb its log quality by
$\{-1,-0.5,0,0.5,1\}$, and separately vary the physical support, age, and
bootstrap variance that generate the quality. Sweep $\kappa/\tau$ over
$\{0,0.1,1,10\}$ with all other settings fixed. Perturb a PSD moment using
nonnegative rank-one terms, with relative operator perturbations
$\{0,0.01,0.05,0.10,0.25\}$ against a prefix-fixed nonzero risk scale; include
off-diagonal changes with unchanged marginals using the construction below.
Vary label delay and risk age over $\{0,1,2,4\}$ windows, and include both
fixed-mask and source-loss treatments. Only fixed-mask/cap cases are compared
to Eq.~\eqref{eq:three-way-sensitivity}.
Report weight changes, bound ratios, forecast error, candidate-set changes,
executed action differences, net service return, and recovery time. Parameters
are frozen before held-out evaluation; favorable test outcomes do not select
the sweep range or the winning configuration. The accompanying numerical audit
already checks 100 fixed-set perturbations; its largest observed change-to-bound
ratio is $0.63412$ at tolerance $2\times10^{-6}$. This check is neither a new
online estimation experiment nor a queue-recovery result.

'''
joint = r'''\subsection{Core benefit beyond marginal accuracy}
\label{sec:joint-benefit}

For predictions of one shared target $z$, with $e_s=p_s-z$ and
$R=\E[\boldsymbol e\boldsymbol e^{\mathsf T}]$, simplex fusion gives
\begin{equation}
 \E[(\boldsymbol w^{\mathsf T}\boldsymbol p-z)^2]
 =\boldsymbol w^{\mathsf T}R\boldsymbol w.
 \label{eq:joint-risk-identity}
\end{equation}
$R$ is an uncentered second moment, including bias and covariance.
The following population constructions isolate a benefit available from joint
structure. They are not assumptions that $\widehat M$ is the true service moment.
Here $b$ is a cap and $b_0$ is a target half-width.

'''+theory[split:]
increment=(root/'work/decision_risk_fragment.tex').read_text()
increment=increment[increment.index('\\begin{proposition}[Decision weighting adds structure'):]
joint+=increment

baseline=r'''\subsection{Original RF and controlled comparisons}
\label{sec:baselines}

\paragraph{Version-faithful Original RF.}
The supplied rule multiplies support, freshness, variability, and scalar
outcome consistency:
\begin{align}
 q^N_{s,k}&=\frac{n_{s,k}^{\rm eff}}{n_{s,k}^{\rm eff}+n_0},&
 q^A_{s,k}&=e^{-A_{s,k}/\tau_A},&
 q^V_{s,k}&=\frac1{\nu_{s,k}+\epsilon_\nu},&
 q^E_{s,k}&=e^{-E_{s,k}/\tau_E},\\
 r^{\rm orig}_{s,k}&=q^N_{s,k}q^A_{s,k}q^V_{s,k}q^E_{s,k},&
 \boldsymbol\alpha_k^{\rm orig}&=argmin_{\boldsymbol\alpha\in\mathcal C_k}
 D_{\rm KL}(\boldsymbol\alpha\Vert
          \operatorname{normalize}(\boldsymbol r_k^{\rm orig})).
 \label{eq:original-rf}
\end{align}
Its historical affine forecasts predict the next matched service-change
diagnostic $d^{\rm svc}_{3,k+1}$; its scalar MSE updates only after that label:
\begin{equation}
 E_{s,k+1}=(1-\zeta)E_{s,k}
 +\zeta(\widehat\psi_{s,k+1|k}-d^{\rm svc}_{3,k+1})^2.
\end{equation}
The original meaning and computation of $n^{\rm eff}$, floors, availability,
fallback, and gate thresholds are retained in the version-faithful reproduction.
A raw valid-row count is not retrospectively called an independent effective
sample size. Original RF combines diagnostics into
$\Gamma^{\rm orig}=\sum_s\alpha_s^{\rm orig}d_s$ and
$D^{\rm orig}=\sum_s\alpha_s^{\rm orig}(d_s-\Gamma^{\rm orig})^2$, with
$U^{\rm orig}=\min\{1,4D^{\rm orig}+\kappa_{\rm miss}(1-c)\}$.
It has no full joint residual matrix, decision-weighted moment, or inertia term.
Its supplied scheduler subtracts $-\lambda_U U^{\rm orig}$ equally from all
candidates. That subtraction cancels from an argmax and cannot be credited
with changing update intensity. Any separate original threshold gate is retained
and reported independently.

\paragraph{Matched Original RF-C control.}
For an estimator-only comparison, use the identical DA-RF forecasts, target,
source masks, measurement-quality factors, cap, and decision layer, but replace
the proposed optimizer by the original multiplicative scalar-MSE/KL rule.
Label this target-aligned control \emph{Original RF-C}; it is not the untouched
Original RF. Report the version-faithful end-to-end baseline separately.
The previous evaluated Risk RF is also retained under its original target,
unweighted residual moment, and gate; historical results are not relabeled DA-RF.

\paragraph{Structure, scale, and decision-layer controls.}
The primary cross-error comparator replaces $\widehat M$ by
$\diag(\widehat M)$, retaining $\delta I$, quality, inertia, masks, forecasts,
and gate. Compare decision-weighted versus unweighted full matrices under the
same context kernel and prior protocol. Rescale the unweighted matrix to the
weighted trace before adding the same radius; zero trace uses a declared common
fallback. This matched-scale control separates matrix shape from risk amplitude.
Use separate ablations for no risk, no inertia, no cap, no quality factor,
no sentinel, no probe, and no uncertainty exposure. Equal, prefix-fixed, inverse
marginal-MSE, and declared online-expert controls share forecast/label budgets.
Inverse MSE uses common-target errors, not heterogeneous diagnostic differences.
An evidential control declares its masses and normalization; it is not assigned
fictitious linear weights. QMF/PDF comparisons use compatible branch-prediction
tasks and are described as core-mechanism comparisons unless their entire
systems are reproduced~\cite{qmf2023,pdf2024}.

Fusion-only experiments freeze the target, forecast fitter, gate, recovery
estimator, and service-policy feature pathway. Gate-only experiments freeze
forecasts and weights. New labels, larger predictors, or a more permissive
sentinel are separate factors rather than evidence for joint-moment fusion.

'''

coord=block('\\subsection{Persistent-loss admission and recovery probes}',
            '\\subsection{Execution order and computational audit}')
coord=coord.replace('At slot $t$, let $k(t)$',
r'''The highest-priority rule is all-missing forecast evidence: if
$\mathcal A_{k(t)}=\varnothing$, choose resource-feasible no-update inference
and prohibit both exploitation and probe launches. A valid service loss-only
forecast counts as an available source under the explicitly logged mode above;
an unsupported or fabricated observation does not override this rule.

At slot $t$, let $k(t)$''')
coord=coord.replace('Set $C_t(i_0)=0$. Let',
r'''Set $C_t(i_0)=0$ and $\lambda_C=\alpha_C$ in
Eq.~\eqref{eq:external-reward}, with identical active-slot accounting.
For an undiscounted evaluation endpoint use $\gamma=1$ in both prediction
labels and costs. Values, allowances, margins, and exposure penalties all have
the same horizon-reward units. Gross rollout loss already includes training
resource competition and is not deducted a second time as a direct charge.
Let''')
calstart=coord.index('Let $\\widehat W_{j,t}$ predict')
coord=coord[:calstart]+r'''Candidate calibration has three declared states: supported, extrapolated,
and unsupported. Prefix/calibration simulator forks cover profile, capacity,
delay, and loss contexts, with held-out paired rollouts. Log support counts,
context distances, and prediction residuals. Only supported candidates with a
justified allowance enter certificate-based exploitation. Extrapolated and
unsupported candidates may be explored through the budgeted probe mechanism,
but their empirical residual buffer is not a uniform counterfactual guarantee.
Factual deployment logs alone do not label all unselected candidates.

'''+coord[calstart:]
cs=coord.index('\\begin{proposition}[Conditional net-value certificate]')
recover=(root/'work/conditional_recoverability.tex').read_text()
recover=recover[:recover.index('% Minimal baseline naming scheme')]
coord=coord[:cs]+recover+'\n'

order=r'''\subsection{Execution order and computational audit}

At a completed window boundary: finalize the valid target and diagnostic masks;
score previously saved forecasts against the newly arrived target; update their
old feature--target pairs and joint residual archive; then extract the current
evidence and quality. Fit or retrieve independently calibrated candidate-response
coefficients from the pre-fusion context, save their slope span, and issue new
forecasts with identifiers and timestamps. Build the context-weighted moment,
radius, and restricted anchor; solve fusion; publish the snapshot for the next
slot. A refitted prediction is never scored against the label used to fit it.

At each slot: ingest the last valid snapshot and current capacity/worker status;
apply the all-missing rule; update the persistent-loss hysteresis from completed
feedback; form supported exploitation, no-update, and budgeted-probe candidates;
score once and enumerate the admitted pairs. After selecting $j_t$, issue the
per-slot reference before service. Execute service and active training occupancy,
log costs and gradient steps, and deploy only completed workers at the declared
boundary. Probe launch/completion, its observations, and any later change of
calibration support are logged separately.

An archive of $K$ complete vectors costs $O(KS^2)$ to reweight and $O(KS)$
to store; a cached unweighted outer-product update costs $O(S^2)$.
Report source extraction/bootstrap, delayed fitting, moment construction,
convex solution, recovery prediction, pair enumeration, and actual training
latency separately. Enumeration over five update and six inference profiles
is small, but its timing is not end-to-end fusion or neural-update latency.

'''

downstream=block('\\subsection{Propagation to admission and executed profile selection}',
                 '\\section{Known-moment mechanism verification}')
downstream=downstream.replace('\\Delta R','\\Delta S').replace('\\widetilde R','\\widetilde S')
downstream=re.sub(r'\bR\b','S',downstream)
downstream=downstream.replace('\\S', '\\R')
downstream=downstream.replace('r^{\\rm pred}', 'r^{\\rm dec}')
downstream=downstream.replace('\\omega_R\\{','\\frac{\\omega_R}{B_U^2}\\{')
downstream=downstream.replace('identical source forecasts,\ndiagnostics, masks, cap, resource-feasible candidates, worker state, and',
'''identical source forecasts, masks, cap, physical candidate feasibility,
worker state, and''')
gs=downstream.index(' |\\Gamma-')
ge=downstream.index(' |\\widehat y-',gs)
downstream=downstream[:gs]+downstream[ge:]
downstream=downstream.replace('Since simplex weight differences sum to zero,',
                            'Since simplex weight differences sum to zero,')
start=downstream.index('Suppose the recovery estimator')
end=downstream.index('This result explains why')
downstream=downstream[:start]+r'''Suppose the validated recovery and allowance models yield a uniform absolute
score perturbation $B_L$ on their supported domain. Let $B_0$ bound the change
of the no-update upper reference and $B_h$ the change of the persistent-loss
statistic. The latter is no greater than the forecast bound above when the
observed-loss term and validity states remain fixed. Resource feasibility alone
does not fix the actual admitted set. Require the loss statistic to be farther
than $B_h$ from its applicable hysteresis threshold, and every exploitation
comparison $L_t(i,j)-B_t^0-m_t$ to be farther than $B_L+B_0$ from zero.
Together with fixed support status, cooldown, coverage, fallback state, worker
eligibility, probe budget, and deterministic tie rules, these margins fix the
actual admitted set. Within that set, a winner gap greater than $2B_L$ preserves
the selected pair. A gate-margin crossing is a genuine possible action change,
not a failure of the fixed-set weight bound. Log all margins and crossings.

'''+downstream[end:]

validation=(root/'work/method_validation_compact.tex').read_text()
refs=(root/'work/fusion_references.tex').read_text()
text = preamble+opening+evidence+quality+moment+fusion+decision+stability+joint+baseline+coord+downstream+order+validation+'\\begin{thebibliography}{99}\n'+refs+'\n\\end{thebibliography}\n\\end{document}\n'
# Preserve literal LaTeX control sequences in replacements assembled above.
text=text.replace('\x07lpha','\\alpha')
(root/'outputs/risk_rf_revision.tex').write_text(text)
print(len(text.splitlines()), 'lines written to existing source')
