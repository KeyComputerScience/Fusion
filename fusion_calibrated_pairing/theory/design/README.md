# Prequential paired conditional-return fusion

This is a new development design. Previously inspected selfBACK and RSS data remain development data. No former algorithm, freeze, outcome, or manuscript is modified.

The method has one information-to-return line:

1. One common strong PDF forecaster supplies all physical-source class probabilities and its fused forecast. Source models use the active published training objectives; the tabular architecture is disclosed rather than presented as original image/text architecture reproduction.
2. At issuance, compute the complete-lease target prediction anchor `a`, source action contrasts `h`, and descriptor `d=h-a`. Store the immutable pair `(E=U-a,d)` only after complete labels mature. Here `U` is the original issued candidate/reference gross full-lease contrast divided by actual lease length. Historical targets are never relabeled with models trained on them.
3. Fit one smoothed full joint residual/descriptor law, then condition on current descriptors and shift by current anchor. A dedicated earlier state-fit library persists across physical-chain resets; only mature local tuples update the online archive.
4. One lower-tail score, followed by one signed conditional residual-quantile calibration transform, determines paid admission. Every information control receives identical forecasters, library origins, quality, context, score-calibration population, nine-configuration budget, and execution guarantee.

The source quality must be recomputed on the permanent model-excluded prefix error set using the common nonlinear PDF models. `strong_prefix_quality` accepts those held-out arrays, preserves the previous linear quality separately, and returns the new common quality for every internal control. Empty source masks are explicitly unready and cannot cause prior-only deployment.

## Information controls

- `paired`: retain the complete residual/source tuple and condition its full covariance-smoothed mixture.
- `pair_factorized`: preserve the residual marginal and every `(residual,source descriptor)` pair marginal exactly before current support truncation; retain a product conditional-source law given the residual.
- `full_gaussian`: retain full residual/source mean and covariance and use the resulting Gaussian conditional law.
- `joint_diagonal_kernel`: retain all full paired tuple locations with diagonal smoothing. This is a legitimate full conditional-density competitor, not an information-reduced control.
- `unconditional`: retain the same residual marginal and common strong anchor while discarding descriptor conditioning.
- `conditional_moment`: condition on the full paired descriptor first, retain the paired law's conditional mean and variance, and fit the bounded maximum-entropy scalar law. This is a stronger tuple-aware non-Bayesian competitor than a global covariance-only control. Its API is in `conditional_moment_control.py`.

A non-Bayesian full conditional-density method can retain the same information. An adaptive matrix method can also exploit retained tuples through joint-context localization. The theoretical distinction therefore concerns specified stored-information reductions, not every non-Bayesian estimator.

## API and partitions

`conditional_pairing.py` provides:

```python
stream = build_stream(pdf_events, pre, potential_forks, cfg,
                      prior_records=state_fit_library, recording=physical_id)
state_fit_library = library(state_fit_stream, complete_maturity_only=True)
raw = raw_run(stream, arm, cfg)
```

`pdf_events` contains immutable physical-source `p` and `external_probability` from the common nonlinear PDF world. `potential_forks` comes from the shared service engine. A canonical delay schedule supplies each earlier physical library origin once; delay replicas are not treated as independent prior samples. The caller freezes separate STATE-FIT, SCORE-CAL and SELECTION physical blocks. Source/model training and quality use only the permanent training prefix and its reserved error subset.

The raw density bandwidth is fixed at 1 and its lower expected-shortfall mass is fixed at 0.5. Nine total score-calibration configurations are bandwidth 0.25/0.5/1 times residual-quantile probability 0.8/0.9/0.95. The signed transform and actual-admission report are supplied in `../risk/immutable_conditional_calibration.py`. No 81-combination density/calibration search is concealed.

Gaussian completion and kernel smoothing are working models. The empirical local quantile is not automatically a future conditional-coverage certificate. Actual admission coverage, optimistic excess, harmful leases, negative loss and budget refusals must be reported for every locked physical unit and comparator.

`verify_design.py` validates the residual-to-return affine tail identity, causality before label maturity, and an analytically specified higher-order information witness. Its synthetic score is not a physical deployment result. The production mixture and factorized quadrature remain distinct from a mathematical oracle or an externally certified conditional risk law.

The target-mutation check fixes the issued common forecaster probabilities and model versions and poisons not-yet-mature complete target fields. Individual label arrival is governed separately by the shared predictor's causal update schedule; some labels can arrive before the full lease is complete. This check does not establish invariance to changing labels that a shared forecaster has already legitimately received.

## Completed known-selfBACK development

The six-arm run is preserved in `development_results.json.gz`, with every selected configuration and all counterfactual targets. `development_summary.json` records calibration and actual budget behavior. `analyze_development.py` reconstructs all changed-action terms into `development_analysis.json`; it never changes or selects a policy.

| Controller | Mean net increment over five delays | Beneficial leases | Harmful leases | Actual admitted coverage |
| --- | ---: | ---: | ---: | ---: |
| Paired | -4.0 | 32 | 18 | 10/50 |
| Pair-factorized | -6.8 | 27 | 14 | 6/41 |
| Full Gaussian | -12.4 | 28 | 15 | 6/43 |
| Full joint diagonal-kernel KDE | 19.6 | 32 | 17 | 11/49 |
| Unconditional residual law | 111.2 | 40 | 23 | 20/63 |
| Postconditional maximum entropy | 3.8 | 33 | 17 | 10/50 |

The paired law changes 15 actions against the factorized control: seven additional beneficial leases contribute +361 units, five added harmful leases -234, two missed beneficial leases -134, and one avoided harmful lease +21. Their sum is +14 across five delays, or +2.8 mean. Against the stronger postconditional moment law, it adds a -27 lease and misses a +12 lease, resulting in -7.8 mean. The complete changed-action accounting agrees with measured service differences to at most 5.7e-14; request-level reconstruction differs by at most 1.14e-13.

These are already-used data, so they do not establish independent physical generalization. They do not establish superior net benefit over every matched competitor or calibration at executed admissions. All six selected controllers and all 13 physical units remain in the evidence. The common guard's per-chain budget constraint holds; this execution fact must not be mistaken for a fusion-specific gain. Future raw data must use the separately frozen physical protocol rather than controller replacement based on these development outcomes.
