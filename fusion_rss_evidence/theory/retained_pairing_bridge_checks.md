# Retained pairing bridge: independent checks

These synthetic checks validate the algebra used by the new TeX fragment. They do not rerun or change scientific results.

Seed: 20261004. Target support: [-1, 1]. N=128, q=1.2815515655, nu=0.01. Gaussian quadrature: 512 nodes.

| Check | Maximum absolute error |
| --- | ---: |
| posterior_density | 1.42108547152e-14 |
| conditional_mean | 4.16333634234e-17 |
| conditional_variance | 5.20417042793e-18 |
| paid_score | 7.1054273576e-15 |
| score_derivative | 3.88073068791e-07 |
| source_marginal | 1.7763568394e-15 |
| source_mean | 0 |
| source_covariance | 1.73472347598e-18 |
| posterior_TV_scaling | 4.4408920985e-16 |

All 12101 finite-distribution pairs passed the mean and variance bounds; the paid-score bound passed at three nu and three q values.
Mean, variance and score maximum bound ratios: 1, 1, 1.

The checks include exact-marginal→joint, exact-marginal→joint with diagonal P, and full moment-matched Gaussian→joint bridges. The first two keep coordinate densities fixed. The last keeps the full residual mean and covariance fixed. Direct normalized line quadrature independently checks the posterior bridge and its moments.

The bridge weights are evidence weighted: rho generally differs from tau. Its variance includes rho(1-rho)(mu1-mu0)^2. Retained pairing is not guaranteed to produce a monotone score path.

## Scope and estimation limit

The source densities in this diagnostic are declared fitted working laws. Their computable posterior TV measures the response to a specific information reduction. No finite estimated value here certifies closeness to the unknown physical law. A true-law certificate requires an independently justified posterior-TV or density-estimation bound; the score inequality then propagates that bound to the gate.

A bridge-induced admission difference becomes a paid benefit only after checking the candidate/reference return and common feasible execution state. Sequential reserve differences must be handled by complete replay accounting.
