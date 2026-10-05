# Paired information, paid actions, and recoverable conservatism

This is a development theory review. It does not modify the manuscript or frozen
reproduction sources. The numerical companion uses constructed finite laws and
the existing callable CJ implementation; it reads no physical trace.

## 0. CJ-R: effective-support normalization, its invariant, and its limit

For a fixed set of eligible historical **blocks**, let their positive raw
relevance weights be `w_j`, `S=sum(w)`, `Q=sum(w²)`. CJ-R replaces them by

`w'_j = (S/Q) w_j = n_eff w_j/S`, where `n_eff=S²/Q`.

The same transformed whole-block weights must enter the complete-block `R`,
masked correction `mu/P`, paired atom masses, missing-coordinate completion,
and all matched controls. Masks and observed residuals stay unchanged. Do not
normalize coordinates separately: that changes the pairing experiment.

**Proposition (scale invariance at fixed eligibility).** For `n≥1` positive
eligible weights,

`sum(w')=n_eff ∈ [1,n]`, `w'_j/w'_l=w_j/w_l`,

and replacing every raw `w_j` by `c w_j`, `c>0`, leaves every `w'_j` unchanged.
Consequently the consistently rebuilt CJ-R state, line posterior, and issued
score are unchanged when the residuals, masks, quality, configuration, current
forecasts, issued calibration state and ledger are held fixed.

**Proof.** The normalization gives `sum(w')=S²/Q` directly. Nonnegative
cross-products give `S²≥Q`; Cauchy--Schwarz gives `S²≤nQ`. Ratios cancel the
common multiplier. Scaling sends `(S,Q)` to `(cS,c²Q)` and hence
`(S/Q)w_j` to itself. The remaining state is a deterministic function of the
unchanged transformed records and held-fixed inputs. □

The proposition does not extend across an absolute cutoff such as
`w_j>1e−12`: common scaling can change the eligible set. Empty history retains
the reference. Finite precision can also change threshold ties; checks must
report their tolerances.

For normalized proportions `p_j=w_j/S`, `n_eff=1/sum(p_j²)` is an inverse
concentration measure. If scalar observations were independent with a common
variance `sigma²`, their weighted average would have variance
`sigma²/n_eff`. For serial/dependent blocks the actual variance is
`pᵀ Gamma p`, and for heterogeneous observations it is
`sum(p_j² sigma_j²)` even without dependence. Masks further require the
observed-coordinate precision. Therefore `n_eff` is not a calibrated count of
independent observations in this application, and Gaussian completion is not
new observed evidence.

The prior atom's empirical mass share changes from `S/(a0+S)` to
`n_eff/(a0+n_eff)`. This removes an arbitrary common attenuation when many
relative weights are meaningful, but also removes absolute overlap evidence.
If all blocks are distant from the current decision yet their tiny weights are
positive, CJ-R still assigns their total mass near their concentration count.
This can amplify irrelevant history, biased corrections, and extrapolation.
Keep raw `S,Q`, the normalization multiplier, eligible count and masks in the
audit; do not present the replacement as an empirically calibrated likelihood.

For raw weights in `(0,1]`, the multiplier is at least one. If `R` were held
fixed, the masked precision `K+sum(w'_j H_jᵀ R_OO^−1 H_j)` would increase and
its inverse `P` would decrease in Loewner order. **CJ-R also refits `R`**, so
this conditional observation does not prove that predictive variance,
posterior variance, admission score, acceptance, or paid return improve.

This normalization is a shared prior-balance/weighting change. Pairing's
additional contribution must still be demonstrated against identically
normalized exact-marginal and full-Gaussian controls, followed through changed
actions and complete paid return. The shared loss ledger remains independent
of both contributions.

## 1. Conditions needed before speaking about improvement

Fix one candidate/reference lease, its model versions, service interruptions,
restoration rule, fees, and complete counterfactual paid return `D`. Let `R ⊆ I`
be the reduced and retained information. Both include the current forecasts,
active source mask and an identical execution state. Let `G ∈ {0,1}` be the
common `R`-measurable feasibility indicator. Take `D` integrable. All actions
below satisfy `0 ≤ A ≤ G`.

These hypotheses hold for a single decision conditioned on the same ledger, or
for a set of decisions whose eligibility is externally fixed and identical.
They do **not** automatically hold throughout separately executed budget
trajectories: an earlier admission changes subsequent spent loss and reserves.
For an endogenous sequential budget, the optimal retained policy can emulate a
reduced policy, but the correct comparison involves the complete feasible
policy/value function. Summing the one-decision strict gaps is invalid.

The current service contract is `D = N U − 5 + δ`, with `δ = 2 − ell ∈ [0,2]`.
Thus `N E[U|I] − 5` is a worst-cost paid surrogate, not generally
`E[D|I]`. A claim about actual paid value must retain `E[δ|I]` or declare its
omission as an additional conservative penalty.

## 2. Exact information value and fitted-policy regret

Write `M_H = E[D|H]` for `H ∈ {R,I}`. The oracle value for information `H` is

`V(H) = E[G (M_H)_+]`.

Its exact information gap, already available for CJ, is

`Δ = V(I) − V(R)
   = E[G min{ E[(M_I)_+|R], E[(-M_I)_+|R] }] ≥ 0`.

Strict gain requires positive retained conditional mean paid returns on both
sides of zero inside an available reduced state. Merely changing a covariance,
source weight, posterior mean, or accepted count does not establish this gap.

For any fitted `H`-measurable estimate `hatM_H`, conservative term `r_H ≥ 0`,
and strict-positive gate `A_H = G 1{hatM_H − r_H > 0}`, define

`Reg_H = V(H) − E[A_H D]`.

**Proposition (exact action regret and a sufficient paid-gain condition).**

`Reg_H = E[G |M_H| 1{1{hatM_H − r_H > 0} != 1{M_H > 0}}]`,

and

`0 ≤ Reg_H ≤ E[G |hatM_H − M_H|] + E[G r_H]`.

Consequently, for a fitted retained controller and **any** reduced-information
feasible controller `A_R`,

`E[A_I D] − E[A_R D]
 ≥ Δ − E[G |hatM_I − M_I|] − E[G r_I]`.

For the two fitted sign policies the exact comparison is

`E[A_I D] − E[A_R D] = Δ − Reg_I + Reg_R`.

This identifies the gain source: pairing can increase `Δ`, estimation can
consume it, and unnecessary conservative deductions can consume it. A
computable sufficient improvement claim additionally needs a valid bound on
the *physical conditional paid-mean error*. Comparing two fitted posterior
densities does not supply that bound.

**Proof.** The tower property gives `E[A_H D] = E[A_H M_H]`. Pointwise
maximization over `[0,G]` selects `G 1{M_H>0}`. Subtraction gives the exact
regret identity, including ties under the strict-positive gate. At a harmful
admission, `M_H ≤ 0 < hatM_H−r_H` gives `|M_H| ≤ |hatM_H−M_H|`. At a missed
beneficial action, `M_H > 0 ≥ hatM_H−r_H` gives
`|M_H| ≤ |hatM_H−M_H| + r_H`. Integrate. Any reduced policy has value at most
`V(R)`; subtract this envelope and use the regret bound. The final identity
follows by subtracting the two oracle-minus-regret expressions. □

For the present CJ API set

`hatM_I = N hatμ_I − 5`,
`r_I = q N sqrt(hatv_I + nu²)`.

When the physical posterior `p*` and fitted line posterior `hatp` both concern
the same bounded `U`, total variation gives

`|N hatμ_I − N μ*_I| ≤ 2 N TV(hatp,p*)`.

With `κ_I=E[δ|I]≥0`, the old score may equivalently be written as
`(N hatμ_I−5+κ_I) − (r_I+κ_I)`. Hence

`Reg_I ≤ E[2 G N TV(hatp,p*)] + E[G(r_I+κ_I)]`.

Here `TV(hatp,p*)` is unknown on an ordinary physical stream. It is **not** the
TV between a fitted joint density and its fitted marginal or Gaussian control.
The latter is an information sensitivity diagnostic only.

## 3. What a less conservative gate can and cannot promise

At the identical issuance state, preserving `hatM`, `G`, and target scale while
reducing `r` weakly increases proposal acceptance. Its actual value difference
is exactly

`E[G M 1{ r_new < hatM ≤ r_old }]`

for `0 ≤ r_new ≤ r_old`; the inclusive endpoint follows the strict gate.
Therefore the relaxed gate improves expected paid value exactly when this
newly admitted band has positive weighted true conditional paid mean. With a
known conditional error bound `|hatM−M|≤ε`, every new admission is beneficial
if `hatM>ε` on that band. Without such a bound, more acceptance can introduce
harmful leases and can consume a binding budget.

If `hatM=M` exactly and the only objective is expected one-lease paid return,
`r=0` is the Bayes-optimal gate. Positive predictive variance is not itself
conditional mean error: an exactly known law can have high outcome variance
while its positive expected return remains correctly known. The old
`q sqrt(v+nu²)` gate addresses a different tail/coverage preference and does
not generally optimize expected paid return. If that preference is retained,
its risk objective and calibration assumptions should be explicit.

More retained information has no monotone acceptance-rate theorem even for
perfect oracle policies. For two rich states, `M={1,−100}` with probabilities
`{.99,.01}` yields a reduced mean `−.01`: rich oracle acceptance is `.99` and
reduced acceptance is zero. With `M={100,−1}` and probabilities `{.01,.99}`,
the reduced mean is `.01`: rich acceptance is `.01` and reduced acceptance is
one. In both examples retained information improves expected return by `.99`.
Thus "best return and best acceptance" are distinct objectives.

## 4. The bounded uniform prior is a working model

The existing line law is

`hatp(u|h) ∝ 1[-1≤u≤1] hatf_Z(h−u*1)`.

It is valid as Bayes conditioning under the *declared* additive independent
working model `H=U*1+Z`, `U~Uniform[-1,1]`, `U ⟂ Z`. Neither empirical residual
pairing nor analytic Gaussian integration proves these physical assumptions.

There is also a support incompatibility if one insists that this exact model
generate physical source forecasts in `[-1,1]^m`. Since the full uniform prior
puts mass arbitrarily near both boundaries, independence and
`H_s=U+Z_s ∈[-1,1]` almost surely force `Z_s=0` almost surely for every source.
For example, conditional on any `Z_s=z>0`, the uniform target places positive
probability in `(1−z,1]`, forcing `H_s>1`; negative `z` is symmetric. Any
nondegenerate Gaussian-smoothed residual law violates this bounded-forecast
support when viewed as the exact physical generative law. Interior-support
target constructions can be coherent, and the current formula remains a useful
working likelihood evaluated at observed `h`; it is not calibrated confidence.

A physically coherent extension may model `p(U,Z)` or `p(U|H,context)` directly,
use a prior with compatible interior support, or condition the joint model on
the forecast support with its `U`-dependent normalization. Simply truncating
the old posterior after fitting does not establish a correct physical law.

## 5. Implementation audit and verification scope

The current `joint_posterior(h,alpha,locations,covariances)` returns analytic
bounded line mean and variance. Its evidence includes the determinant,
orthogonal residual term, line variance factor and truncation mass. The strong
`product_posterior` uses the same exact coordinate mixture marginals and the
same `h`. `conditional_forecast` returns `F=N*mean` (zero when no eligible
history), `S=N*sqrt(variance+nu²)`, and then a separate calibration score
`(F−truegross)/S`. The current/future truth field enters that score, which is
used for offline callbacks; it does not enter issued `F` or `S`.

The read-only verifier checks exact finite oracle/regret identities, both
acceptance counterexamples, newly admitted beneficial and harmful bands,
analytic CJ moments against independent composite quadrature, exact-marginal
parity, source-weight invariance, and poisoning of `truegross` in actual CJ
issuance. It also measures bounded-model overflow in a simple additive
construction. These checks are mechanism and implementation checks, not a
new calibration guarantee or performance result.
