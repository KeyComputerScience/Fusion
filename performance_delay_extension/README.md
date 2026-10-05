# Supplementary fixed-controller feedback-delay replays

This is a diagnostic extension declared after the original four-trace results.
It uses five additional delay seeds (72006–72010) and keeps each saved controller
configuration and initial calibration value unchanged. It adds no physical site
or independent participant sample. Do not pool these averages with primary runs.

The original `run_extension.py` is frozen at SHA256
`07e27b7d0a394aebe0117e8a92bc722c9de8ed5eb78b386c3c8689a29b28583b`.
Its workspace-relative default path was retained for execution provenance.
The separate portable launcher changes runtime paths only:

```sh
python portable_launcher.py --package ../bayes_closed_loop_repro --output reproduced
```

NumPy is the only numerical dependency; the recorded execution used Python 3.12
and NumPy 2.3.5. Supply the unchanged reproduction package, including cached
input arrays and original metadata. The launcher copies the original declaration
to a new output directory, verifies the frozen runner, injects only package/output
paths, and suppresses bytecode writes. The actual alternate-directory rerun
matched all 80 original supplementary trajectories exactly in net return,
actions and causal calibration updates (`portable_reproduction_audit.json`).

`results.json` is the unified new-results file; per-trace files and `summary.json`
retain all outcomes. `execution_audit.json` independently reconstructs all service
requests, fees, restorations and local forks, alongside solver and calibration
checks. `crossing_audit.json` adds raw posterior-concentration and opportunity-
slack checks. `mechanism_cases.json` records the two HAR origin-68 crossings and
four MHEALTH origin-4 crossings, including issued thresholds and threshold swaps.
Those repeated origins are the same physical episode under different artificial
feedback delays, not six new independently observed episodes.

`pooled_coverage.json` uses total covered/total issued forks. `summary.json`
retains the mean of per-seed coverage rates, so its informative rates can differ.
The compact LaTeX fragment uses pooled denominators. All paired intervals are
descriptive Student-t4 intervals conditional on the fixed trace and controllers;
they are unadjusted for multiple comparisons. Adverse outcomes are retained,
including MHEALTH loss against the common reference and poor informative
coverage. No posterior-specific general superiority is claimed.
