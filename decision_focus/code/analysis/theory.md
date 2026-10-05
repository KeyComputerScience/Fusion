# Focused mechanism: label-free disagreement conditioning

**Development status:** the subsequently executed affine variant did not
improve the retained controller and produced harmful RSS leases. It is not
a replacement for the final original method. The directly applicable original
capped-objective theorem is now in `theory_original.tex`; it preserves every
existing measured result and uses h=.112 rather than the affine example below.

This proposal is one inference-to-admission mechanism. It does not introduce another execution safeguard, optimizer regularizer, or purportedly new Gaussian identity. The existing RSS +38 intervention remains an outcome of the immutable original controller. The numerical example below is a mechanism construction, not a new RSS observation or a reassignment of old outcomes to a new controller.

## 1. Exact working law and the information already available

Restrict the globally reconstructed state to the currently observed source set A. Write h=h_A, μ=μ_A, C=R_AA+P_AA ≻0, and t=g_complete/N for the complete shared candidate/reference contrast. The exact identity is z=h−t1. Let D be any full-row-rank (m−1)×m contrast matrix with D1=0. Consequently Dz=Dh is available without any future label.

The Gaussian location working law is

    h | t, matured history ∼ N(μ+t1,C),

with μ,C fixed before the current complete target, C independent of t in this law. Equivalently the predictive error is z∼N(μ,C). In the empirical-Bayes implementation, this is a working approximation: μ,P are the masked precision state and R is estimated from the same declared complete-history rule. It is not a verified independence statement about the physical recordings.

This assumption deserves three explicit limits:

- Probability-derived h and t are bounded. Therefore a globally nondegenerate untruncated Gaussian cannot be the exact physical distribution; exactness below is mathematical conditional on the working law, while actual calibration is assessed on complete matured targets.
- The law describes a common location target plus residual noise. Marginal Gaussian errors alone, when their relationship with t is unrestricted, do not establish this location model.
- Source outages restrict h and C to observed coordinates. The method does not invent a missing current forecast. Partial historical blocks can still affect the global correction state before this restriction.

## 2. Proposition: all admissible influence vectors collapse to one target estimate

Condition the Gaussian error on the exact observed current disagreement:

    μ_c = μ + C Dᵀ(D C Dᵀ)⁻¹ D(h−μ),
    C_c = C − C Dᵀ(D C Dᵀ)⁻¹ D C.

Define

    v = (1ᵀC⁻¹1)⁻¹,
    t_hat = v 1ᵀC⁻¹(h−μ).

Then

    h−μ_c = t_hat·1,               C_c = v·11ᵀ.

Hence any a satisfying 1ᵀa=1 has the same corrected target aᵀ(h−μ_c)=t_hat and the same conditional error variance aᵀC_c a=v. A separate influence optimizer, capped-simplex entropy penalty and independently weighted R/P norms are not needed after this conditioning. Conditional θ and new residual noise can become correlated, so retaining separate conditioned R and P penalties is not justified by simply adding their marginal reductions.

Proof. D(h−μ_c)=0, so h−μ_c lies in span(1). Let w=C⁻¹1/(1ᵀC⁻¹1). Then wᵀCDᵀ=v1ᵀDᵀ=0, giving the claimed scalar coefficient. Also D C_c=0. Symmetry and rank imply C_c=a11ᵀ; multiplying by C⁻¹1 gives C_c C⁻¹1=1, hence a=v. The final statements follow by substitution. Singleton sources use t_hat=h−μ and v=C; empty observed sets abstain.

Equivalent deterministic formulation:

    t_hat = argmin_t (h−μ−t1)ᵀC⁻¹(h−μ−t1).

This is standard generalized least squares/Mahalanobis projection. A deterministic implementation with the same information must match. The implied GLS coefficients w may be signed; they are contrast-estimation coefficients, not a convex probability-mixture or a capped-simplex solution. A constrained t∈[−1,1] estimate or posterior must be described separately if used.

No prior is needed for the location confidence pivot: under the stated law, (t_hat−t)/sqrt(v) is N(0,1) and independent of D h. Calling t|entire h a posterior requires a prior. A flat location prior on R gives N(t_hat,v); a proper uniform prior on bounded t∈[−1,1] instead gives a truncated normal. Do not conflate these interpretations or infer accepted-action coverage from the location pivot.

## 3. One paid certificate and the completed execution connection

Use

    F=N t_hat,       S=N sqrt(v),
    L=F−q S−5,       proposal=1{L>0},

with the existing complete-target calibration and shared execution contract. Readiness, maturity and execution liability remain common service requirements; they are not additional fusion inventions.

Let T=(F−g_complete)/S and D_return=g_complete−5+δ, δ∈[0,2], as in the immutable lease contract. Algebra gives

    D_return = L−S(T−q)+δ.

Therefore an executable proposal with L>0 and the **completed** coverage event T≤q has D_return≥L>0. This is the precise conditional bridge from joint information to an action and completed paid benefit. It is not ex ante unconditional profit. Aggregate adaptive calibration or coverage on rejected forks does not imply the accepted-action coverage event or its conditional probability.

If a valid accepted-action conditional coverage bound Pr(T≤q | issued state,admit)≥1−α were independently justified and the shared reserve is M, then E[D_return | issued state,admit]≥(1−α)L−αM. Such a bound is not established by the current adaptive feedback accounting alone.

## 4. Same marginals, same counts, different joint information: realizable example

Take two sources, σ²=.01, equal quality and P0=.01 I. Construct four residual patterns

    z(ξ,η)=.1 (ξ, r ξ + sqrt(1−r²) η),  ξ,η∈{−1,+1},

with r=+.9 in history H+ and r=−.9 in history H−. Each pattern has total history weight one, so total full-block mass is four. Every source is co-observed in every record. Both histories have **identical source marginal sample distributions**, zero mean, equal quality, equal masks/counts and equal total history mass; only pairing changes. A factorized-state conditioner is identical in both histories.

The patterns can be repeated with nonnegative weights totaling one per pattern; no unequal singleton counts are used. With the implemented covariance prior a0=2, σ0²=.01, their predictive R is

    R_± = .01 [[1, ±.6],[±.6,1]].

This follows because the raw correlation ±.9 is multiplied by 4/(2+4). The precision likelihood has total full-block mass four and mean zero, yielding

    P_± = .01/(25−.6²) [[5−.6², ±4·.6],[±4·.6,5−.6²]].

Their diagonals are .001883116883116883, and their off-diagonals are ±.000974025974025974. Thus C_±=R_±+P_± have exactly the same marginals and mean but different jointly observed relationships. The fully factorized conditioner instead has R_fac=.01I, P_fac=.002I and C_fac=.012I in **both** histories.

These residuals are realizable with immutable binary probability forecasts: let candidate/reference make opposite decisions, let the historical complete contrast be zero with balanced lease labels, and set source probabilities to ((1+z_s)/2,(1−z_s)/2). All values are inside the probability simplex. Equal marginal distributions also permit equal prefix quality. This establishes sample realizability, not literal Gaussian generation of those finite bounded records.

Choose identical current μ=0 and h=.09·1, N=128 and q=1. Symmetry fixes t_hat=.09. The only action difference is decision-direction uncertainty:

| State | v | Issued L |
|---|---:|---:|
| Joint H− | .002454545454545454 | +.17844871287216346 |
| Joint H+ | .009428571428571429 | −5.908906399426874 |
| Fully factorized state | .006 | −3.3948373662909876 |
| Diagonal reduction of C± | .005941558441558441 | −3.3464326636577972 |

If the common execution budget is feasible, H− admits whereas both reductions reject. A complete realized g=10 gives D_return∈[5,7] and satisfies the H− coverage inequality g≥F−qS≈5.17845. Thus the joint relationship can retain an actual paid benefit at fixed current corrected mean, all marginal information, observed-source counts and execution contract. The opposite-correlation history correctly yields a smaller certificate; the theorem does not say joint information always raises a gate or beats every matrix estimator.

For literal fixed β=.97 history weights, repeated pattern groups can be assigned the same total weight by choosing allowed context-kernel factors below one. The construction requires those balanced weighted totals, not 48 artificial independent copies. It is an information-identification example, not an empirical replication count.

## 5. Why partial matured masks still matter within this one mechanism

For a current historical correction state θ∼N(μ,P), observe a matured block z_O with selector H and covariance R_OO/a independent conditional on θ. Let Σ=H P Hᵀ+R_OO/a. Exact Gaussian/Woodbury updating gives

    μ+ = μ + P HᵀΣ⁻¹(z_O−Hμ),
    P+ = P − P HᵀΣ⁻¹ H P.

For a previously unobserved coordinate U, its correction changes by P_UO Σ⁻¹(z_O−μ_O). This is a conditional **latent error correction**, not a fabricated source forecast or zero-filled residual. The complete factorized prior/noise model has P_UO=0 and cannot propagate that correction. In the example after the four full-block mass, adding the same source-1 residual .1 changes the missing source-2 mean by ±.00819672131147541; the factorized state changes it by zero. Source counts and marginal inputs are still identical across the two joint histories.

Directional information gain for any fixed vector w is

    wᵀ(P−P+)w = ||Σ^(−1/2) H P w||².

It is zero precisely when the observed block is irrelevant to that direction under the working state. With innovation zero, μ stays fixed; with R frozen, C+=R+P+≼C. Consequently v+=min_{1ᵀa=1} aᵀC+ a ≤v, strictly if the old GLS direction w satisfies H P w≠0. The variational identity follows from a constrained quadratic minimization or Cauchy–Schwarz; evaluating C+ at the old minimizer proves the order and strict condition.

If h−μ=c1 (or t_hat is otherwise held fixed), q>0, and

    L≤0 < L + N q(sqrt(v)−sqrt(v+)),

this added directional information causes an admission. A completed coverage event then proves its positive paid return using Section 3. Nonzero innovation, changed R or an altered contrast can change the mean and invalidate this fixed-mean monotonicity; no blanket improvement is claimed.

The Schur/Woodbury algebra is established Gaussian conditioning. The contribution is retaining genuine jointly observed error relationships and aligning their conditional direction with the complete paid action target. Sourcewise scalar summaries cannot generally reconstruct that information; a same-information non-Bayesian full-matrix method can.

## Checks and reporting boundary

- Numerical joint-versus-factorized figures above were computed independently with NumPy.
- Random SPD checks for m=2,…,8 (20 examples each) verified h−μ_c=t_hat1 and C_c=v11ᵀ to maximum absolute error 2.35e−13.
- This new mechanism must be prospectively amended and tested as its own variant. Existing RSS +38, exhaustive fixed-state interventions, GLS equivalence and every mixed/negative task result remain attributed to the original tested controller. No new collection becomes untouched validation because of this theoretical revision.
