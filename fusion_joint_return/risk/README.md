# Actual-admission calibration development

This directory retains a real, magnitude-aware prequential calibration run
on already known RSS and gas issued-score trajectories. It does not claim
a newly frozen validation or universal superiority.

`prequential_quantile.py` transforms the single full-return score to S-c.
Its offset is an empirical quantile of immutable issued-score residuals
that have completely matured. Current reprojected training errors and
future complete truth cannot enter the offset. Two feedback populations
are evaluated separately: all ready counterfactual leases under the common
replay contract, and actual admissions only. All nine probability/history
configurations are retained for every controller and population. The
unchanged Fixed130 guard is used throughout. Three future-truth poisoning
cutoffs per trajectory verify that pending truth cannot alter earlier
actions. The original scientific source/results are read-only inputs.

Run from the project root using the bundled Python (NumPy available):

```
python work/fusion_joint_return_20261005/risk/prequential_quantile.py
```

The protocol file deliberately prevents overwriting the first execution;
copy the script to a clean sibling directory if rerunning. `selection.json`
contains all calibration trials; `results.json.gz` retains immutable issued
scores, changed scores, callbacks and budget ledgers; `summary.json` includes
all selected results. Positive and unfavorable changes are both retained.

Actual selected joint results:

| Task | Original net | Quantile net | Original admitted coverage | Quantile coverage | Original total excess | Quantile total excess |
|---|---:|---:|---:|---:|---:|---:|
| RSS |22.4|22.4|9/13|9/13|38.2757|37.9802 ready / 38.2273 admitted|
| Gas |831.4|783.6|37/50|38/47|109.2007|82.1367|

The gas calibration transform removes one harmful lease and two beneficial
leases, improving target coverage and exceedance while reducing total
paid gain. It therefore supports a factual calibration improvement, not
a claim that this transform outperforms the strongest complete pipeline.
Both feedback populations yield the same joint actions here.

The numerical binomial values in the JSON are IID feasibility references,
not valid confidence limits for these dependent adaptive replay traces.
Even IID eligible residuals do not justify ordinary fixed-n binomial
intervals after an outcome-dependent budget guard stops admission; use a
time-uniform confidence envelope instead. `risk_envelope.py` provides an
executed-prefix conditional-risk upper envelope with all error allocations
explicit and a separate independent-whole-chain risk contrast bound.

`risk_claims.tex` gives executable score semantics, the realized results,
proofs, and the explicit assumptions needed for stronger risk claims.
Rolling empirical quantiles do not give conditional 90% coverage under
arbitrary drift. The prefix envelope controls the average conditional risk
of already executed actions, not that of the next lease. The independent
whole-chain theorem covers new chains under the same declared sampling law,
not arbitrary new sites. Current small fitting/acceptance counts do not
reach a 10% risk certificate.
