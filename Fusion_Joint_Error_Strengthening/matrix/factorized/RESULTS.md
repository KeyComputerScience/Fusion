# Fully factorized information: actual RSS results

The complete factorized conditioning formula and protocol were frozen before evaluating this control on RSS. The previously inspected RSS trace makes this a known-trace extension. Existing matrix-control scripts, results and frozen source hashes remain unchanged.

The control diagonalizes the global working `R` before conditioning and rebuilds both `mu` and `P` coordinate by coordinate. Its observed blocks, masks, support/context/age weights, quality ridge and physical model states are identical to the full controller at the same prior scale. It is not the earlier penalty-only diagonal ablation. All 27 calibration evaluations are retained. The selected `(prior_sd,q_floor,q_initial)=(0.2,0,0)` matches Full's selected setting.

| RSS controller | Mean complete increment | Net utility | Admitted | Beneficial | Harmful | Zero |
|---|---:|---:|---:|---:|---:|---:|
| Full joint precision | 30.2 | 3304.0 | 10 | 9 | 0 | 1 |
| Fully factorized information | 44.8 | 3318.6 | 14 | 13 | 0 | 1 |

These rows hold both without a binding budget and at the original Fixed130/B130 point; neither controller has a harmful completed lease or a refusal there. Factorized increments are `[31,76,47,41,29]`, versus Full `[9,76,47,9,10]`. Full minus factorized is `[-22,0,0,-32,-19]`, mean `-14.6`; the conditional paired `t_4` interval is `[-32.19487,2.99487]`. The five delays share one physical trace, so this interval is not an independent-site inference.

The factorized controller adds four profitable admissions totaling `73`: seed88001/origin164=`+22`, seed88004/origin28=`+4`, seed88004/origin164=`+28`, seed88005/origin164=`+19`. Full retains no exclusive profitable admissions and avoids no losses in this comparison. There are no exclusively incurred losses or changed zero-return actions. The same decomposition holds for the fixed original-Full-threshold information intervention and Fixed130/B130. Thus all extra complete return here comes from profitable opportunities that Full rejects.

On the 210 original full-controller states, factorization changes the correction mean and precision on 82 states. The largest coordinate mean difference is `0.0786329123`, and the largest matrix-entry precision-covariance difference is `0.0148639672`. Dynamic calibration thresholds can diverge after callbacks, although selected settings and exogenous models are identical. The local intervention explicitly fixes Full's original prior scale and issued `q` to isolate the conditioning change.

| Dynamic RSS stratum | Full coverage | Factorized coverage | Full scaled excess | Factorized scaled excess |
|---|---:|---:|---:|---:|
| All issued | 201/210 | 199/210 | 8.718899 | 14.792191 |
| Informative | 46/50 | 44/50 | 3.718899 | 9.792191 |
| Proposed/admitted | 6/10 | 8/14 | 3.718899 | 9.792191 |

Scaled excess is `max(F-g-q*S,0)`. Its maximum is `1.859382` for Full and `3.477391` for factorized information. Factorized gains more utility and has more score excess, but both have zero harmful complete actions. The RSS evidence does not establish a positive return increment from joint conditioning; neither does it negate the mathematical non-equivalence of the information states or prove factorized dominance on unseen streams.

All 210 method-specific forks and five service trajectories pass independent reconstruction. All 80 budget trajectories retain their diagnostic results, with unchanged selected fusion parameters. `results.json` contains the full budget grid, every changed action, local matrices, gain/loss decompositions, severity, and state relations. `calibration_trials.json.gz`, `budget_trials.json.gz`, and `fixed_state_all_rows.json.gz` retain complete evaluation records. `verification.json` checks source integrity, positivity, matching masks/weights and reconstruction errors.
