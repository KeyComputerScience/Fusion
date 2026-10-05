# Independent external-fusion validation audit

All four tasks, both external pipelines, and all five original delay schedules are retained. This is a later diagnostic extension on the same four previously evaluated physical traces. It does not add independent tasks or sites. Intervals describe delay variability on each fixed trace.

The external rule is PDF-P: a probability-interface adaptation of Predictive Dynamic Fusion with static prefix-fitted TCP heads, not a complete end-to-end reproduction. Both point and delayed-calibrated pipelines have nine prefix trials; parameter families differ from internal fusion, and additional static TCP-head prefix training is disclosed.

## Schema conversion and integrity

Original external trials have no baseline/windows fields. The immutable shim adds baseline={gross,fees,drops,net} from the same trial’s frozen arm and windows from the original frozen trial; both are independently checked against original baseline, ceil(test_rows/32), and explicit-service audit metadata. No original result, frozen runner, parameter, or old coverage artifact is edited.

All 120 arm×seed log audits pass; original internal arm results match every common saved field (maximum error 0). Source-weight simplex maximum error is 4.44e-16, and external gate penalty mismatch is 0. Original files and old 290-trial analysis remain unchanged.

Point-arm q0=0; its adaptive q tracker is not used in admission. Its q-based coverage must be labelled a diagnostic and cannot supply an executable lower-gate protection claim. Calibrated-arm penalty=q_issued×SD is checked directly. Scalar gate weights=[1] are not physical-source weights: request-level PDF softmax weights are checked separately. Softmax does not have the internal source cap; kkt=0 is an unused interface placeholder, not a source-weight optimization certificate.

## Complete paired full versus external results

| Task | Comparator | Mean net gain | Conditional t4 interval | Five gains | +/=/- | Avoided harmful / comparator harmful | Retained beneficial / comparator beneficial |
|---|---|---:|---|---|---|---|---|
| occupancy357 | pdf_point | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0/5/0 | 0/0 (undefined) | 0/0 (undefined) |
| occupancy357 | pdf_calibrated | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0/5/0 | 0/0 (undefined) | 0/0 (undefined) |
| occupancy864 | pdf_point | 29.6000 | [15.8125, 43.3875] | [16.0, 38.0, 37.0, 19.0, 38.0] | 5/0/0 | 5/5 (100.00%) | 0/0 (undefined) |
| occupancy864 | pdf_calibrated | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0/5/0 | 0/0 (undefined) | 0/0 (undefined) |
| mhealth319 | pdf_point | 26.8000 | [9.3415, 44.2585] | [10.0, 24.0, 31.0, 21.0, 48.0] | 5/0/0 | 15/26 (57.69%) | 1/1 (100.00%) |
| mhealth319 | pdf_calibrated | 9.8000 | [-3.2641, 22.8641] | [0.0, 5.0, 12.0, 5.0, 27.0] | 4/1/0 | 4/12 (33.33%) | 0/0 (undefined) |
| har240 | pdf_point | 7.2000 | [-0.1142, 14.5142] | [7.0, 14.0, 10.0, -2.0, 7.0] | 4/0/1 | 12/12 (100.00%) | 3/7 (42.86%) |
| har240 | pdf_calibrated | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0/5/0 | 0/0 (undefined) | 3/3 (100.00%) |

## Action and coverage denominators

| Task | Arm | Admissions / harmful / beneficial | All-issued coverage | Current-disagreement coverage | Admitted coverage |
|---|---|---|---|---|---|
| occupancy357 | pdf_point | 0 / 0 / 0 | 297/320 (92.81%) | 2/2 (100.00%) | 0/0 (undefined) |
| occupancy357 | pdf_calibrated | 0 / 0 / 0 | 297/320 (92.81%) | 2/2 (100.00%) | 0/0 (undefined) |
| occupancy357 | bayes_both | 0 / 0 / 0 | 297/320 (92.81%) | 2/2 (100.00%) | 0/0 (undefined) |
| occupancy357 | frequentist_gate | 0 / 0 / 0 | 297/320 (92.81%) | 2/2 (100.00%) | 0/0 (undefined) |
| occupancy357 | frozen | 0 / 0 / 0 | 297/320 (92.81%) | 2/2 (100.00%) | 0/0 (undefined) |
| occupancy864 | pdf_point | 5 / 5 / 0 | 138/155 (89.03%) | 0/5 (0.00%) | 0/5 (0.00%) |
| occupancy864 | pdf_calibrated | 0 / 0 / 0 | 150/155 (96.77%) | 0/5 (0.00%) | 0/0 (undefined) |
| occupancy864 | bayes_both | 0 / 0 / 0 | 150/155 (96.77%) | 0/5 (0.00%) | 0/0 (undefined) |
| occupancy864 | frequentist_gate | 0 / 0 / 0 | 150/155 (96.77%) | 0/5 (0.00%) | 0/0 (undefined) |
| occupancy864 | frozen | 0 / 0 / 0 | 138/155 (89.03%) | 0/5 (0.00%) | 0/0 (undefined) |
| mhealth319 | pdf_point | 27 / 26 / 1 | 63/105 (60.00%) | 3/24 (12.50%) | 0/27 (0.00%) |
| mhealth319 | pdf_calibrated | 12 / 12 / 0 | 76/105 (72.38%) | 8/24 (33.33%) | 0/12 (0.00%) |
| mhealth319 | bayes_both | 12 / 11 / 1 | 70/105 (66.67%) | 5/24 (20.83%) | 0/12 (0.00%) |
| mhealth319 | frequentist_gate | 5 / 4 / 1 | 82/105 (78.10%) | 8/24 (33.33%) | 0/5 (0.00%) |
| mhealth319 | frozen | 0 / 0 / 0 | 62/105 (59.05%) | 2/24 (8.33%) | 0/0 (undefined) |
| har240 | pdf_point | 19 / 12 / 7 | 128/160 (80.00%) | 26/31 (83.87%) | 7/19 (36.84%) |
| har240 | pdf_calibrated | 3 / 0 / 3 | 153/160 (95.62%) | 30/31 (96.77%) | 3/3 (100.00%) |
| har240 | bayes_both | 3 / 0 / 3 | 153/160 (95.62%) | 30/31 (96.77%) | 3/3 (100.00%) |
| har240 | frequentist_gate | 2 / 0 / 2 | 155/160 (96.88%) | 30/31 (96.77%) | 2/2 (100.00%) |
| har240 | frozen | 0 / 0 / 0 | 125/160 (78.12%) | 22/31 (70.97%) | 0/0 (undefined) |

## Exact utility and changed-action decomposition

Values are sums over all five delay schedules; dividing by five gives task means. Gains include useful admissions missed and harmful admissions newly added.

| Task | Comparator | Served gross change | Saved fees | Avoided harmful: count / gain | Missed beneficial: count / gain | New harmful: count / gain | New beneficial: count / gain |
|---|---|---:|---:|---|---|---|---|
| occupancy357 | pdf_point | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| occupancy357 | pdf_calibrated | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| occupancy864 | pdf_point | 133 | 15 | 5 / 148 | 0 / 0 | 0 / 0 | 0 / 0 |
| occupancy864 | pdf_calibrated | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| mhealth319 | pdf_point | 89 | 45 | 15 / 134 | 0 / 0 | 0 / 0 | 0 / 0 |
| mhealth319 | pdf_calibrated | 49 | 0 | 4 / 59 | 0 / 0 | 3 / -15 | 1 / 5 |
| har240 | pdf_point | -12 | 48 | 12 / 60 | 4 / -24 | 0 / 0 | 0 / 0 |
| har240 | pdf_calibrated | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

## Interpretation

- Full improves over PDF-P point admission on Room 864 and MHEALTH in every delay schedule, with HAR four positive and one negative schedule. The point gate is less protective than the calibrated external pipeline and must not be the sole modern comparator.
- Against PDF-P calibrated admission, full matches both occupancy tasks and HAR exactly in actions/returns. MHEALTH gains are [0,5,12,5,27], mean9.8, four positive and one tie; its conditional t4 interval includes zero. The gain is real and must be presented as task-specific, not universal superiority.
- Both calibrated pipelines remain harmful on MHEALTH: full11/12 harmful, external12/12 harmful. Full is still below reference by23.2 mean units and below the matched empirical-block gate by6.6. The external comparison strengthens the available baseline set without removing this boundary.
- Full and calibrated PDF-P agree on all HAR admissions. Their equality means the +3.6 versus empirical-block gate cannot be described as uniquely Bayesian relative to the new external calibrated pipeline.
- Full versus each external comparator satisfies ΔJ=Δserved_gross+saved_fees, and the sum of changed-action increments equals the same paired gain. Complete per-origin events remain in JSON; pooled bookkeeping is not a cross-task population effect.
