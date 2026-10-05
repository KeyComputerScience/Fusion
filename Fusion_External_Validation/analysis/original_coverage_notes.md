# Read-only coverage, actions, and gain audit

The original 72001–72005 cohort remains primary; 72006–72010 is a later diagnostic replication under the same bounded IID delay law. The ten-seed combination is descriptive and never replaces the primary result. All ten schedules reuse the same physical trace per task. The t intervals describe imposed-delay variability and are not site/subject generalization intervals.

## Definitions and reconstruction

- Issued: every complete four-window lease origin logged, whether admitted or rejected. Outcomes are evaluated offline for every issued lease, including feedback not yet received at replay end.
- Informative: current-window candidate/reference prediction disagreement > 0, exactly the original declared subset. It is not a definition of all decisions with nonzero historical forecasts; an admission can occur without current disagreement.
- Admitted: actual policy action=true. Beneficial/harmful depend on actual local net >0/<0 after drop and admission/restoration fees. Zero denominators are undefined.
- Coverage: (gain−truegross)/issued SD ≤ q issued at the decision. It compares the pessimistic target truegross−5 with gain−5−q×SD. For arms that do not execute q×SD in their gate, this is a calibration diagnostic, not their executed guarantee.
- Online calibration: only callbacks whose complete lease feedback matured by replay end. This subset differs from all issued; its exact ACI identity uses its own callback count.
- Accounting: J=Jreference+sum(admitted local_net); served gross=reference gross+sum(admitted(local_net+3)); fees=reference fees+3×admissions; extra dropped requests=2×admissions. The audit checks all terms, standardized scores, q-update states, one callback per mature origin, causality, target/slack bounds, counts, and reported coverage.
- This recomputes arithmetic from saved logs, not request-level predictions. The earlier separate physical service/fork reconstruction remains the independent truth audit. No old engine was rerun and no method or parameter changed.

## primary

| Task | Issued coverage | Informative coverage | Admitted coverage | Online-matured coverage | Pending feedback coverage | Admit / harmful / beneficial |
|---|---|---|---|---|---|---|
| occupancy357 | 297/320 (92.81%) | 2/2 (100.00%) | 0/0 (undefined) | 294/316 (93.04%) | 3/4 (75.00%) | 0 / 0 / 0 |
| occupancy864 | 150/155 (96.77%) | 0/5 (0.00%) | 0/0 (undefined) | 150/155 (96.77%) | 0/0 (undefined) | 0 / 0 / 0 |
| mhealth319 | 70/105 (66.67%) | 5/24 (20.83%) | 0/12 (0.00%) | 69/104 (66.35%) | 1/1 (100.00%) | 12 / 11 / 1 |
| har240 | 153/160 (95.62%) | 30/31 (96.77%) | 3/3 (100.00%) | 150/157 (95.54%) | 3/3 (100.00%) | 3 / 0 / 3 |

| Task | Comparator | Mean net gain | Conditional 95% t interval | Gains by seed | + / = / − |
|---|---|---:|---|---|---|
| occupancy357 | reference | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 5 / 0 |
| occupancy357 | frequentist_gate | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 5 / 0 |
| occupancy864 | reference | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 5 / 0 |
| occupancy864 | frequentist_gate | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 5 / 0 |
| mhealth319 | reference | -23.2000 | [-38.8666, -7.5334] | [-39.0, -5.0, -27.0, -27.0, -18.0] | 0 / 0 / 5 |
| mhealth319 | frequentist_gate | -6.6000 | [-16.1616, 2.9616] | [-17.0, -10.0, -5.0, -5.0, 4.0] | 1 / 0 / 4 |
| har240 | reference | 9.4000 | [-1.4103, 20.2103] | [15.0, 0.0, 18.0, 0.0, 14.0] | 3 / 2 / 0 |
| har240 | frequentist_gate | 3.6000 | [-6.3952, 13.5952] | [0.0, 0.0, 18.0, 0.0, 0.0] | 1 / 4 / 0 |

## supplement

| Task | Issued coverage | Informative coverage | Admitted coverage | Online-matured coverage | Pending feedback coverage | Admit / harmful / beneficial |
|---|---|---|---|---|---|---|
| occupancy357 | 301/320 (94.06%) | 4/4 (100.00%) | 0/0 (undefined) | 298/316 (94.30%) | 3/4 (75.00%) | 0 / 0 / 0 |
| occupancy864 | 150/155 (96.77%) | 0/5 (0.00%) | 0/0 (undefined) | 150/155 (96.77%) | 0/0 (undefined) | 0 / 0 / 0 |
| mhealth319 | 72/105 (68.57%) | 6/25 (24.00%) | 1/12 (8.33%) | 69/102 (67.65%) | 3/3 (100.00%) | 12 / 11 / 1 |
| har240 | 150/160 (93.75%) | 27/28 (96.43%) | 3/3 (100.00%) | 147/157 (93.63%) | 3/3 (100.00%) | 3 / 0 / 3 |

| Task | Comparator | Mean net gain | Conditional 95% t interval | Gains by seed | + / = / − |
|---|---|---:|---|---|---|
| occupancy357 | reference | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 5 / 0 |
| occupancy357 | frequentist_gate | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 5 / 0 |
| occupancy864 | reference | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 5 / 0 |
| occupancy864 | frequentist_gate | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 5 / 0 |
| mhealth319 | reference | -22.0000 | [-38.9568, -5.0432] | [-6.0, -13.0, -40.0, -20.0, -31.0] | 0 / 0 / 5 |
| mhealth319 | frequentist_gate | 0.0000 | [-16.9568, 16.9568] | [16.0, 9.0, -18.0, 2.0, -9.0] | 3 / 0 / 2 |
| har240 | reference | 10.8000 | [-1.5357, 23.1357] | [0.0, 17.0, 0.0, 20.0, 17.0] | 3 / 2 / 0 |
| har240 | frequentist_gate | 6.8000 | [-4.7615, 18.3615] | [0.0, 17.0, 0.0, 0.0, 17.0] | 2 / 3 / 0 |

## combined_ten_seed_descriptive

| Task | Issued coverage | Informative coverage | Admitted coverage | Online-matured coverage | Pending feedback coverage | Admit / harmful / beneficial |
|---|---|---|---|---|---|---|
| occupancy357 | 598/640 (93.44%) | 6/6 (100.00%) | 0/0 (undefined) | 592/632 (93.67%) | 6/8 (75.00%) | 0 / 0 / 0 |
| occupancy864 | 300/310 (96.77%) | 0/10 (0.00%) | 0/0 (undefined) | 300/310 (96.77%) | 0/0 (undefined) | 0 / 0 / 0 |
| mhealth319 | 142/210 (67.62%) | 11/49 (22.45%) | 1/24 (4.17%) | 138/206 (66.99%) | 4/4 (100.00%) | 24 / 22 / 2 |
| har240 | 303/320 (94.69%) | 57/59 (96.61%) | 6/6 (100.00%) | 297/314 (94.59%) | 6/6 (100.00%) | 6 / 0 / 6 |

| Task | Comparator | Mean net gain | Conditional 95% t interval | Gains by seed | + / = / − |
|---|---|---:|---|---|---|
| occupancy357 | reference | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 10 / 0 |
| occupancy357 | frequentist_gate | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 10 / 0 |
| occupancy864 | reference | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 10 / 0 |
| occupancy864 | frequentist_gate | 0.0000 | [0.0000, 0.0000] | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 0 / 10 / 0 |
| mhealth319 | reference | -22.6000 | [-31.4786, -13.7214] | [-39.0, -5.0, -27.0, -27.0, -18.0, -6.0, -13.0, -40.0, -20.0, -31.0] | 0 / 0 / 10 |
| mhealth319 | frequentist_gate | -3.3000 | [-11.1801, 4.5801] | [-17.0, -10.0, -5.0, -5.0, 4.0, 16.0, 9.0, -18.0, 2.0, -9.0] | 4 / 0 / 6 |
| har240 | reference | 10.1000 | [3.7781, 16.4219] | [15.0, 0.0, 18.0, 0.0, 14.0, 0.0, 17.0, 0.0, 20.0, 17.0] | 6 / 4 / 0 |
| har240 | frequentist_gate | 5.2000 | [-0.7927, 11.1927] | [0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 17.0, 0.0, 0.0, 17.0] | 3 / 7 / 0 |

## Statements supported and limits

- Both occupancy tasks and HAR have no harmful full admissions in both seed cohorts. Occupancy safety follows complete abstention in these traces; HAR admits only actual-positive leases. This does not establish general safety on other streams.
- MHEALTH remains a material boundary: report its admitted and informative coverage and harmful admission denominator together with its gains over weaker gates.
- The matched empirical-block gate is the essential strong control. It has the same interface/kernel/model work and its own prefix calibration. An increase relative to a point gate or periodic admission is not by itself a Bayesian-exclusive benefit.
- Primary and supplemental counts/intervals remain separate. Combining seeds increases repeated-delay Monte Carlo observations, not the number of physical validation traces (still one per task). No all-task population utility interval is computed.
- Origin identifiers are included for changed actions. Repeated changes at the same physical origin under different delays are not new sites or independent drift episodes.
- Row mass is the model's context/decay pseudo-mass, not an observed number of independent blocks. Saved rows do not retain all historical kernel weights, so a Kish effective sample size or joint block-bootstrap posterior cannot be reconstructed faithfully. Resampling logged episodes would also keep the original q/action trajectory and would not constitute a fresh causal replay. No such confidence claim is made.
- Existing same-IID-delay replicas are not a structural sensitivity experiment for tail delays, missing labels, interruption price, lease horizon, or reference refresh. Those require a separately locked replay protocol.

## Reuse for external comparisons

Import `analyze_trial(trial)` or `analyze_study(study, full_arm='bayes_both', comparators=('reference','NEW_ARM'))`. New arms must log the same issuance-time forecast/q/SD, actual action, complete-lease fork outcome, and q callbacks. A missing/inconsistent schema fails visibly rather than being silently treated as covered. An external arm that does not use posterior q still gets calibration diagnostics, with its executable penalty/gate coverage reported separately.
Run with `--additional-results path/to/results.json`; additional studies are audited separately and are not pooled with old cohorts. This analyzer uses no engine imports and performs no training or tuning.

Input hashes unchanged after analysis: True. All 290 arm/seed log audits passed. Detailed per-arm and per-seed numerators, denominators, callback cohorts, action events, intervals, and accounting errors are in coverage_analysis.json.
