# Non-Bayesian matrix risk extension

This extension imports the frozen `work/fusion_temporal_20261003/temporal_fusion.py` without editing it or any original package. All new artifacts are written in this directory. The source hashes, formulas, seeds, calibration selection rules and limits are declared in `protocol.json` before this extension's calibration and test execution.

The deterministic `gls_exact` control minimizes a quality-weighted ridge-penalized generalized least-squares objective over all valid matured masked blocks. Its mean and inverse curvature equal the frozen joint-precision equations. This is an exact deterministic implementation, not a distinct performance competitor. Agreement is verified against the original calibration grid and five original RSS test trajectories.

The `block_sandwich` control uses the same penalized mean, full predictive covariance, source-quality ridge, physical source masks, contrast support, context kernel and age weights. With `A = K + sum a_j H_j^T R_j^-1 H_j`, each complete matured lease has score `psi_j = a_j H_j^T R_j^-1 (z_j - H_j mu)`. Its HC0 risk is `A^-1 [sum psi_j psi_j^T] A^-1`. Every score outer product is positive semidefinite and uses only the jointly observed coordinates in that lease. Missing errors are never filled and pairwise covariances from incompatible records are never assembled. The global matrix has an eigenvalue floor of `1e-6`, then is restricted to current active source IDs. The deterministic quality ridge affects the estimator's bread, not random prior observations in the meat. No small-sample correction, test tuning or coverage theorem is added.

Both controls receive the original nine `(prior_sd, q_floor)` settings on three calibration delays, 27 evaluations each. Original initialization, second-half complete-return selection, delayed issued-score updates, solver and paid service are retained. The final five RSS delay arrangements share one held-out physical trace. RSS had been examined for the original study; this comparison is explicitly post hoc and does not create a new independent validation dataset. Gas interventions are development evidence only.

The fixed-state evaluation uses every original full-controller issued state on RSS and Gas. It holds current mean, predictive covariance, threshold, quality and all model versions fixed, and changes only the risk matrix. Every changed action is retained. Categories refer to the full controller relative to the control: `retained_gain` is a positive full-only admission; `missed_gain` a positive control-only admission; `avoided_loss` a negative control-only admission; `incurred_loss` a negative full-only admission. Changed zero-return admissions form a separate category. These sums are one-step mechanism measurements, not independent dynamic policy return.

For every issued state, scaled excess severity is `S*(T-q)_+ = max(F-g-q*S, 0)`. It is reported separately for all issued, informative, proposed and admitted states. Dynamic RSS also retains the complete original diagnostic grid of reserves `fixed130`/`decision` and budgets `0,110,130,260`; none of these operating points selects or changes the fusion parameters.

Run using the bundled Python environment:

```bash
/Users/key/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 work/fusion_strengthening_20261003/matrix/run_matrix_controls.py --phase freeze
/Users/key/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 work/fusion_strengthening_20261003/matrix/run_matrix_controls.py --phase calibrate
/Users/key/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 work/fusion_strengthening_20261003/matrix/run_matrix_controls.py --phase test
/Users/key/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 work/fusion_strengthening_20261003/matrix/run_matrix_controls.py --phase interventions
```

Phases refuse to overwrite final results. Reproduction can use a clean copy of this directory with generated JSON files removed, while retaining the identical runner. The freeze checks both the runner and original input hashes after every execution phase. The cluster sandwich is an adaptive empirical matrix comparator; dependence, short local histories, estimated `R`, ridge shrinkage bias, and adaptive reprojection prevent interpreting it as a finite-sample confidence guarantee.

Background for the sandwich principle: [White (1980), A Heteroskedasticity-Consistent Covariance Matrix Estimator](https://doi.org/10.2307/1912934) and [Cameron and Miller (2015), A Practitioner's Guide to Cluster-Robust Inference](https://escholarship.org/uc/item/1jq5d0pq). This extension applies the estimating-equation score-outer-product form to the declared masked penalized GLS blocks; it does not reproduce a separately published fusion algorithm or inherit those papers' independence/asymptotic assumptions.
