# Final independent mathematical and output audit

Reviewed `outputs/risk_rf_performance.tex`, 1381 lines, SHA256
`d68e82309d7f5351a53db7cd05e617e06d581bd95d12c8223f221a94081deb9a`.
This audit is read-only; the manuscript and immutable scientific code were not changed.

**Disposition: no substantive blocking error found in the requested checks.**
Two small assumption clarifications below would make the algebraic scope completely explicit.

## 1. Original-objective joint-pairing construction

The proposition at manuscript lines 287–354 is a genuine joint-information construction, rather than a singleton-count example. For

\[
z=.1(\xi,r\xi+\sqrt{1-r^2}\eta),\qquad r=\pm .9,
\]

the four balanced weighted patterns have zero mean and raw covariance
`.01[[1, ±.9], [±.9, 1]]`. Reversing the sign of `r` preserves each source's full marginal multiset; only which simultaneous source errors are paired changes. With prior mass two and historical mass four, the stated `R±` and `(100 I + 4 R±^{-1})^{-1}` are correct. The covariance eigenvalue floor is inactive. Diagonal-penalty states are identical between histories, and fully factorized states are also identical.

The symmetry proof establishes the unique feasible optimizer `(1/2,1/2)`: the linear value is constant, both positive-definite directional norms are minimized there, and the KL deduction is uniquely minimized there. The cap `.8` permits this optimizer. An independent numerical substitution gives:

| State | F | S | F − S − 5 |
|---|---:|---:|---:|
| Joint negative pairing | 11.321755031973558 | 5.865696889543475 | 0.45605814243008247 |
| Joint positive pairing | 9.331590515783677 | 11.52 | −7.188409484216322 |
| Diagonal penalty, common mean | 10.205032376489317 | 9.14102838853485 | −3.935996012045532 |
| Fully factorized state | 10.090720268345088 | 9.14102838853485 | −4.0503081201897615 |

The binary probability realization is valid: `|z_s|≤.1335889895`, so `((1+z_s)/2,(1−z_s)/2)` lies in the simplex and produces the required origin contrast. The added agreement positions permit an even number of served disagreement requests, allowing historical complete contrast zero and integer test contrasts 10 or 0. This avoids the earlier all-disagreement odd-parity problem. Balanced *total* pattern weights are part of the constructed working-state assumption; repeated patterns are not treated as independent samples. The statement is an information-identification example, not a claim that the measured RSS archive has this designed state.

For `g−=10`, the independently calculated coverage statistic is
`(F−−10)/S−=.2253364019422472≤1`. Therefore the exact complete-return identity gives `D−∈[5,7]`. For `g+=0`, it gives `D+∈[−5,−3]`. A reduced state that excludes pairing must use the same launch probability `u` in both cases. Its conditional regret is at least
`5 p (1−u)+3 (1−p)u`, whose minimum is `min(5p,3(1−p))>0`.
The proof is correct with shared feasible execution budgets. The manuscript appropriately limits necessity to the stated reduction, and does not infer universal benefit or Bayesian exclusivity.

## 2. Partial observations and directional information

For a fixed working covariance and fixed existing posterior state, the update at lines 204–231 is the Gaussian/GLS information update. `Ω=H P Hᵀ+R_OO/a` is positive definite; Woodbury gives both the mean correction and the PSD covariance decrement. The directional identity

\[
w^\top(P-P^+)w=\|\Omega^{-1/2}HPw\|^2
\]

is exact. An unobserved latent error correction can change through `P_UO`; no missing source forecast is synthesized. A factorized posterior cannot transfer that correction when `P_UO=0`. The text already distinguishes this fixed-state identity from rebuilding the empirical archive.

Optional wording clarification: fixed-state addition also holds the prior and all old weights/projections fixed. At a later origin, ageing, kernel changes, current-model reprojection or prior changes require rebuilding even if the numerical `R` happens to be unchanged. Adding “prior, old weights and projection” to the rebuilding sentence would remove any ambiguity; this is not a discovered implementation inconsistency.

## 3. Reserve 130 and every service prefix

The reserve at lines 473–491 is valid under the declared four-window service contract. The actual eligible origin is a probe and has at least one common request drop (`probe_drop=1`); deployment adds two drops. At most 29 origin requests remain served by the candidate, and the next three windows contain at most `3×32=96` requests. Charging all three lease fee units conservatively gives `3+2+29+96=130`. More generally, origin extra drops plus remaining origin disagreements are bounded by the origin requests remaining after common work. All request-level differences are at least `−1`, so this same reserve covers an adverse service prefix, not merely the terminal lease result. Restoration/nonoverlap make the reference-relative increments additive.

The source `current_decision_reserve` independently uses
`3 + extra + current_served_disagreement + (H−1)×window`; `partial_lease_return` checks request and fee boundaries. The proof's ledger condition is correct: admission preserves `C+Q≤B`; settlement replaces a valid reserve by `(-D)+≤M`; unresolved leases retain their liability. Profits do not replenish loss. The guarantee requires the declared interruptions and immutable nonoverlapping leases, which the manuscript states; it would not transfer the value 130 to a service with removed common-work interruption.

## 4. Deterministic GLS and direct affine control

The manuscript correctly separates two different comparisons:

* Deterministic penalized GLS in Eq. (deterministic-gls) equals the complete masked Gaussian working-state estimator and can reproduce Full's subsequent paid objective and actions. Its reproduction is not an inferior-method experiment and establishes no exclusive Bayesian gain.
* The direct affine control uses the same joint information but changes the estimator/paid criterion to `C=R+P`, signed GLS contrast coefficients and a single predictive deduction. Full's difference from this control is an objective/controller comparison, not an independent joint-information ablation.

The affine section explicitly says joint and same-mean diagonal have identical budgeted actions, hence no additional cross-source policy gain in this control. It also reports stronger factorized performance and all harmful outcomes. This is the correct interpretation of the actual outputs.

Optional wording clarification: the rank-one conditional covariance identity requires a complete difference operator, `D1=0` **and** `rank(D)=m−1`. Conditioning on fewer independent differences does not reduce covariance to rank one. “Those differences” currently reasonably implies the complete operator, but stating its rank would make the identity formally explicit. The working-law conditioning gives no physical coverage theorem, as the manuscript already acknowledges.

## 5. Actual affine files and outcome scope

Read `work/fusion_focus_revision_20261003/affine/analysis.json`, the declared protocol and service/guard implementation. The main table agrees with actual saved results:

| Primary Fixed130/B130 arm | RSS mean | RSS harmful/beneficial | AReM mean | GasHome mean |
|---|---:|---:|---:|---:|
| Frozen Full, manuscript original record | 30.2 | 0/9 | −30.0 | 45.2 |
| Affine joint | 6.6 | 4/4 | −30.0 | 45.2 |
| Affine diagonal, same mean | 6.6 | 4/4 | −30.0 | 45.2 |
| Affine factorized state | 11.6 | 4/6 | −30.0 | 45.2 |
| Affine sandwich | 6.6 | 4/4 | −30.0 | 45.2 |

Joint RSS increments are `[-4,−4,2,41,−2]`; their mean is 6.6. The Full-minus-joint category totals are additional benefit 142, avoided loss 14, missed benefit 38 and incurred loss 0, so `(142+14−38)/5=23.6`; 17 actions differ. Its delay-conditional interval is `[-28.25787056,75.45787056]`, which includes zero. The unguarded affine-joint RSS mean is 48.6 with five harmful leases; it is distinct from its budgeted mean. AReM primary arms admit five harmful and zero beneficial leases; GasHome primary arms admit five beneficial and zero harmful leases.

Saved affine verification passes for each of RSS, AReM and GasHome: 20 unguarded plus 80 guarded trajectories per task, maximum service reconstruction error zero, and unchanged immutable core. Totals 60 and 240 match the manuscript. Maximum saved GLS/conditioning identity error on RSS is about `1.25e−16` for means and `1.39e−17` for covariance.

The main lead restricts its no-harm observation to the original eight tasks, explicitly excluding a generalization to later AReM outcomes. The direct-control section makes its RSS no-harm statement locally about Full and reports the additional task failures in the same table. No unsupported universal no-harm claim was found.

## Submission interpretation

The conditional joint-information theorem is valid and adds an exact information-to-cost-action link. The measured RSS `+38` geometry intervention, direct-control utility differences and the constructed theorem remain distinct evidential levels. The present direct-control data do not establish an independent joint-versus-diagonal improvement, and the manuscript correctly admits that limitation. Neither standard Gaussian algebra nor the shared loss ledger should be described as a new fusion primitive; the current framing appropriately locates the contribution in preserving decision-projected complete-horizon co-observation information and testing its paid action consequences.
