# Frozen non-Bayesian matrix controls: results

The extension's formulas and protocol were declared before its calibration/test execution. This is a post hoc comparator study on the previously inspected RSS trace, not a new independent physical dataset. Gas is development evidence. The original core and every hashed source input remain unchanged.

| RSS control | Mean reference-relative complete return, no binding budget | Admissions | Beneficial | Harmful | Zero |
|---|---:|---:|---:|---:|---:|
| Full joint precision | 30.2 | 10 | 9 | 0 | 1 |
| Deterministic penalized GLS | 30.2 | 10 | 9 | 0 | 1 |
| Empirical block sandwich | 48.2 | 19 | 16 | 1 | 2 |

Both new controls received the same 27 calibration evaluations. Both selected `(prior_sd,q_floor,q_initial)=(0.2,0,0)`. GLS reproduces all nine original full-controller calibration outcomes, all 210 RSS issued decisions, and all test actions. Maximum differences: weights `4.44e-16`, gain `4.44e-15`, predictive scale `3.55e-15`, mean `1.39e-17`, inverse-curvature matrix `0`, threshold `0`. This directly rules out an exclusive Bayesian implementation advantage for these equations.

Full increments across the five delays are `[9,76,47,9,10]`; block-sandwich increments are `[37,76,59,41,28]`. Full minus sandwich is `[-28,0,-12,-32,-18]`, mean `-18.0`, conditional paired `t_4` interval `[-33.90106,-2.09894]`. This interval concerns the five imposed delays on one physical trace and does not support population/site-level significance.

| Reserve / budget | Full mean increment | Sandwich mean increment | Sandwich harmful / admissions |
|---|---:|---:|---:|
| Fixed130 / 0 | 0 | 0 | 0 / 0 |
| Fixed130 / 110 | 0 | 0 | 0 / 0 |
| Fixed130 / 130 | 30.2 | 44.4 | 1 / 18 |
| Fixed130 / 260 | 30.2 | 48.2 | 1 / 19 |
| Decision / 0 | 0 | 0 | 0 / 0 |
| Decision / 110 | 10.6 | 28.6 | 1 / 15 |
| Decision / 130 | 30.2 | 48.2 | 1 / 19 |
| Decision / 260 | 30.2 | 48.2 | 1 / 19 |

The shared guard satisfies its budget and service-prefix bounds on all 120 diagnostic trajectories. The sandwich controller's sole negative complete lease is `-1`; at Fixed130/B130 it causes one later budget refusal. The full controller has zero settled negative loss. None of the guard operating points was used to retune the fusion rules.

| Dynamic RSS stratum | Full covered / total | Sandwich covered / total | Full total scaled excess | Sandwich total scaled excess |
|---|---:|---:|---:|---:|
| All issued | 201 / 210 | 192 / 210 | 8.718899 | 61.393203 |
| Informative | 46 / 50 | 37 / 50 | 3.718899 | 56.393203 |
| Proposed / admitted, unguarded | 6 / 10 | 10 / 19 | 3.718899 | 47.937027 |

Scaled excess is `max(F-g-q*S,0)`. Full's maximum issued scaled excess is `1.859382`, versus sandwich `9.696494`. Full is more conservative and has lower excess severity, but loses profitable actions. Neither controller establishes nominal acceptance-conditional 90% coverage.

Every original full-controller state was also subjected to a risk-matrix-only intervention with current mean, `R`, issued `q`, source quality and models fixed. GLS changes zero actions on 210 RSS and 400 Gas states. Sandwich changes nine RSS actions: seven missed gains totaling `91`, one avoided loss of `1`, and one zero-return action; full-minus-control one-step return sums to `-90`. On Gas it changes 11 actions: nine missed gains totaling `245`, two avoided losses totaling `23`, and full-minus-control return `-222`. There are no retained gains or incurred losses among these crossings. The complete crossing list, scores, matrices and scaled excess values are in `fixed_state_interventions.json`; all issued intervention rows are retained in the task-specific compressed JSON files. These are fixed-state measurements and are not added to dynamic policy gains.

The HC0 sandwich is an empirical matrix comparator, not a confidence theorem. It does not automatically handle ridge shrinkage bias, dependent lease blocks, local weighting, estimated working covariance or adaptive reprojection. Its predeclared `1e-6` global eigenvalue floor is a numerical regularizer. The evidence supports a transparent risk/return tradeoff and exact deterministic equivalence; it does not support superiority over general non-Bayesian matrix risk methods.

Machine-readable aggregation and audit: `matrix_summary.json`. Complete calibration outcomes: `rss348_calibration_trials.json.gz`. Selected states and all grid scores: `rss348_selection.json`. Dynamic test results: `rss348_results.json`. All budget states: `rss348_budget_results.json.gz`. Provenance and reproduction: `protocol.json`, `README.md`, `run_matrix_controls.py`, and `verify_matrix_results.py`.
