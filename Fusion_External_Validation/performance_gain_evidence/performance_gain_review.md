# Complete locked performance gain analysis

All four tasks and all five test delay schedules (72001–72005) are retained. The same-trace schedules are not independent sites. Signs and deployment categories are recomputed from complete lease increments, and every paired return is exactly decomposed into changed actions. Saved analysis.json means agree to 1e−8. No method or parameter is selected by this analysis.

## Task-specific paired gains

| Task | Comparator | Mean net gain | Five gains, in seed order | Positive/tie/negative | Harmful reduction | Beneficial retention |
|---|---|---:|---|---|---|---|
| Occupancy 357 | Joint predictive point gate | 0.00 | [0, 0, 0, 0, 0] | 0/5/0 | NA (0 denominator) | NA (0 denominator) |
| Occupancy 357 | Posterior gate only | 0.00 | [0, 0, 0, 0, 0] | 0/5/0 | NA (0 denominator) | NA (0 denominator) |
| Occupancy 357 | Matched empirical-block gate | 0.00 | [0, 0, 0, 0, 0] | 0/5/0 | NA (0 denominator) | NA (0 denominator) |
| Occupancy 357 | Periodic admission | 312.40 | [315, 316, 306, 311, 314] | 5/0/0 | 319/319 (100.00%) | 0/1 (0.00%) |
| Occupancy 357 | Keep shared reference | 0.00 | [0, 0, 0, 0, 0] | 0/5/0 | NA (0 denominator) | NA (0 denominator) |
| Room 864 | Joint predictive point gate | 29.60 | [16, 38, 37, 19, 38] | 5/0/0 | 5/5 (100.00%) | NA (0 denominator) |
| Room 864 | Posterior gate only | 0.00 | [0, 0, 0, 0, 0] | 0/5/0 | NA (0 denominator) | NA (0 denominator) |
| Room 864 | Matched empirical-block gate | 0.00 | [0, 0, 0, 0, 0] | 0/5/0 | NA (0 denominator) | NA (0 denominator) |
| Room 864 | Periodic admission | 178.40 | [166, 186, 185, 169, 186] | 5/0/0 | 155/155 (100.00%) | NA (0 denominator) |
| Room 864 | Keep shared reference | 0.00 | [0, 0, 0, 0, 0] | 0/5/0 | NA (0 denominator) | NA (0 denominator) |
| MHEALTH | Joint predictive point gate | 28.00 | [10, 24, 31, 27, 48] | 5/0/0 | 16/27 (59.26%) | 1/1 (100.00%) |
| MHEALTH | Posterior gate only | 8.80 | [0, 0, 22, 0, 22] | 2/3/0 | 2/13 (15.38%) | 1/1 (100.00%) |
| MHEALTH | Matched empirical-block gate | -6.60 | [-17, -10, -5, -5, 4] | 1/0/4 | -7/4 (-175.00%) | 1/1 (100.00%) |
| MHEALTH | Periodic admission | 92.80 | [74, 108, 79, 86, 117] | 5/0/0 | 88/99 (88.89%) | 1/6 (16.67%) |
| MHEALTH | Keep shared reference | -23.20 | [-39, -5, -27, -27, -18] | 0/0/5 | NA (0 denominator) | NA (0 denominator) |
| HAR | Joint predictive point gate | 6.40 | [7, 14, 10, -2, 3] | 4/0/1 | 11/11 (100.00%) | 3/7 (42.86%) |
| HAR | Posterior gate only | 0.00 | [0, 0, 0, 0, 0] | 0/5/0 | NA (0 denominator) | 3/3 (100.00%) |
| HAR | Matched empirical-block gate | 3.60 | [0, 0, 18, 0, 0] | 1/4/0 | NA (0 denominator) | 2/2 (100.00%) |
| HAR | Periodic admission | 138.40 | [146, 155, 131, 115, 145] | 5/0/0 | 147/147 (100.00%) | 3/12 (25.00%) |
| HAR | Keep shared reference | 9.40 | [15, 0, 18, 0, 14] | 3/2/0 | NA (0 denominator) | NA (0 denominator) |

Harmful reduction compares total harmful admissions. Beneficial retention is the paired overlap with comparator-admitted positive leases; it does not count a new positive lease as a retained one. Zero-denominator retention is undefined, not 100%.

## Descriptive action totals across all 20 task-seed replays

There are 740 eligible origin/seed lease records. Full admits 15: 11 harmful, 4 beneficial, 0 neutral. Thus its harmful fraction among admissions is 11/15=73.33%, and beneficial fraction 4/15=26.67%. These denominators must accompany safety wording.

| Comparator | Admissions | Harmful | Beneficial | Full harmful reduction | Beneficial retention | Full new beneficial leases |
|---|---:|---:|---:|---|---|---:|
| Joint predictive point gate | 51 | 43 | 8 | 32/43 (74.42%) | 4/8 (50.00%) | 0 |
| Posterior gate only | 17 | 13 | 4 | 2/13 (15.38%) | 4/4 (100.00%) | 0 |
| Matched empirical-block gate | 7 | 4 | 3 | -7/4 (-175.00%) | 3/3 (100.00%) | 1 |
| Periodic admission | 740 | 720 | 19 | 709/720 (98.47%) | 4/19 (21.05%) | 0 |
| Keep shared reference | 0 | 0 | 0 | NA (0 denominator) | NA (0 denominator) | 4 |

## Net gain separates served correctness and saved fees

For every seed, full−control net utility equals (full served gross−control served gross)+(control fees−full fees). Common fees cancel, and saved fees equal three times the reduction in admission count. Served gross already includes actual interruption losses; it is not potential classification accuracy.

| Task | Full−joint mean net | Mean served correctness change | Mean saved fees |
|---|---:|---:|---:|
| Occupancy 357 | 0.00 | 0.00 | 0.00 |
| Room 864 | 29.60 | 26.60 | 3.00 |
| MHEALTH | 28.00 | 18.40 | 9.60 |
| HAR | 6.40 | -2.60 | 9.00 |

## Complete changed-action gain decomposition

Each entry is a sum over five schedules per task, with counts of replayed origin/seed events. The values below can be divided by five to recover each task mean. Benefits from losses avoided and gains newly admitted must be reported together with useful updates missed and losses newly admitted.

| Task | Comparator | Avoided harmful: count / gain | Missed beneficial: count / change | New harmful: count / change | New beneficial: count / gain | Net sum |
|---|---|---|---|---|---|---:|
| Occupancy 357 | Joint predictive point gate | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| Occupancy 357 | Posterior gate only | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| Occupancy 357 | Matched empirical-block gate | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| Occupancy 357 | Periodic admission | 319 / 1563 | 1 / -1 | 0 / 0 | 0 / 0 | 1562 |
| Occupancy 357 | Keep shared reference | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| Room 864 | Joint predictive point gate | 5 / 148 | 0 / 0 | 0 / 0 | 0 / 0 | 148 |
| Room 864 | Posterior gate only | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| Room 864 | Matched empirical-block gate | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| Room 864 | Periodic admission | 155 / 892 | 0 / 0 | 0 / 0 | 0 / 0 | 892 |
| Room 864 | Keep shared reference | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| MHEALTH | Joint predictive point gate | 16 / 140 | 0 / 0 | 0 / 0 | 0 / 0 | 140 |
| MHEALTH | Posterior gate only | 2 / 44 | 0 / 0 | 0 / 0 | 0 / 0 | 44 |
| MHEALTH | Matched empirical-block gate | 2 / 44 | 0 / 0 | 9 / -77 | 0 / 0 | -33 |
| MHEALTH | Periodic admission | 88 / 501 | 5 / -37 | 0 / 0 | 0 / 0 | 464 |
| MHEALTH | Keep shared reference | 0 / 0 | 0 / 0 | 11 / -121 | 1 / 5 | -116 |
| HAR | Joint predictive point gate | 11 / 56 | 4 / -24 | 0 / 0 | 0 / 0 | 32 |
| HAR | Posterior gate only | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| HAR | Matched empirical-block gate | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 18 | 18 |
| HAR | Periodic admission | 147 / 730 | 9 / -38 | 0 / 0 | 0 / 0 | 692 |
| HAR | Keep shared reference | 0 / 0 | 0 / 0 | 0 / 0 | 3 / 47 | 47 |

## Accurate advantage-first statements

- Relative to the joint predictive point gate, full reduces harmful admissions from 43 to 11 across the complete replay set: 32/43=74.42%. Four of the comparator’s eight beneficial admissions are retained (50%); the lower loss count does not establish preserved recall of all beneficial updates.
- On Room 864, all five point-gate losses are avoided; gains are [16,38,37,19,38], mean 29.6. This equals keeping the reference and matches the calibrated block gate, so it is uncertainty-admission benefit rather than Bayesian-exclusive gain.
- On MHEALTH, harmful point-gate admissions decrease from 27 to 11 (16/27=59.26%), and the sole point-gate beneficial lease is retained (1/1). Gains against the point gate are positive in all five schedules. Relative to posterior-gate-only, full avoids two additional −22 leases, harmful 13→11 (2/13=15.38%), while retaining its one beneficial lease.
- On HAR, all 11 joint-point harmful leases are avoided, while 3/7 comparator beneficial leases are retained. Gains against the point gate are [7,14,10,−2,3], positive in 4/5 schedules. Relative to the calibrated block gate, full retains both positive admissions (2/2) and adds one +18 lease: mean extra gain 3.6, positive in 1/5 and tied in 4/5 schedules.
- Relative to periodic admission, full has positive net gain in every one of the 20 task-seed comparisons. Avoiding most periodic leases contributes this result; periodic admission is weaker than the matched calibrated comparator and must not be the sole primary comparison.
- Full exactly matches the posterior-gate-only controller on both occupancy tasks and HAR. Its incremental weight contribution is confined to two MHEALTH scenario crossings, not evidence of independent weight benefit in every task.

## Material boundaries that cannot be deleted

- Full is 6.6 mean units below the matched block gate on MHEALTH, with four negative and one positive paired schedule. It is 23.2 below keeping the reference there.
- Full still makes 11 harmful admissions, all on MHEALTH, versus one beneficial MHEALTH admission. A statement of zero harmful deployments applies only to the two occupancy tasks and HAR, and must identify those domains.
- Nominal one-sided coverage is not attained on MHEALTH: all 70/105, informative 5/24 and admitted 0/12. Room 864 informative coverage is 0/5 despite aggregate 150/155. Posterior moments or algebra audits do not remove these failures.
- The t4 intervals describe imposed-delay variability on a fixed trace. They do not provide independent-site or participant generalization evidence. Mechanism examples are two delayed replays of one MHEALTH origin and one HAR origin.
- Raw net-utility totals have different trace lengths; per-task differences remain the main reporting unit. The JSON also contains pooled bookkeeping totals, not an across-site estimated effect.

All events, control denominators, per-seed values, conditional descriptive intervals, positive-gain retention and changed-action gain decompositions are in performance_gain_analysis.json.
