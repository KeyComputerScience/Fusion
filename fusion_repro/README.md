# AC-DAF-I real-sensor replay and reproduction

This package preserves the real UCI sensor replay supporting the accompanying Methodology and Performance chapters. Its main positive method combines current-action projection of saved source errors, archived mean-bias correction, full cross-source covariance, and tangent-trace invariant scaling. A block Bayesian bootstrap module gives the analytic posterior predictive representation of the same archive estimator; it is not a newly measured additional performance component.

The invariant method is **post-test exploratory**: its design followed inspection of the original Air replay. The 10 scenarios vary imposed feedback delay on one chronological dataset. Reported paired intervals are conditional on that dataset and fit, and do not include method-selection uncertainty. They are not independent confirmation or field deployment measurements. The untouched original Air and Gas result files are included to preserve neutral and negative comparisons.

## Contents

- `real_air_invariant_fusion.py`: positive exploratory invariant Air variant; the original estimator and gate are preserved.
- `real_air_paired_fusion.py`: original Air action-projected, centered covariance variant with full-trace scaling.
- `real_action_fusion.py`: Gas real-sensor replay and shared softmax/optimizer adapters.
- `da_rf_fusion.py`: capped KL plus PSD quadratic risk solver; larger generic wrapper is not used in these real replays.
- `block_bayesian_fusion.py`: analytic block posterior predictive mean/covariance and optional Dirichlet sampling diagnostics. No Monte Carlo is required for production weights.
- `reproduce_audit.py`: one preserved Air seed, five primary comparators, full three-source outage compatibility, 0/1/3-source API checks, and actual-archive Bayesian equivalence audit.
- `data/AirQualityUCI.csv`: original UCI Air CSV.
- `data/gas_drift/Dataset/batch*.dat`: original 10 Gas batches.
- `results/*.json.gz`: original full result records, compressed without modifying their contents.
- `results/original_real_air_invariant_fusion.py.gz`: pre-repair source used in the outage compatibility audit; its original code cannot handle arbitrary missing masks.
- `reproduction_audit.json`: audit of packaged deployment repairs against preserved results.
- `audit_real_recovery.py` and `results/real_recovery_causal_audit.json`: independent same-state recovery/return reconstruction with no further admissions in each local fork.
- `manifest.json`: SHA256, sizes, roles, and provenance of delivered artifacts.
- `analyze_fusion_results.py`: regenerates paired t intervals, utility accounting, action denominators, monthly statistics and sensitivity from all saved scenarios.
- `analysis_reproduced.json`: generated analysis with 757 values checked against the chapter evidence.
- `bayesian_block_audit.json` and `projection_gain_audit.json`: numerical algebra audits, not independent performance experiments.

## Run

Observed environment: Python 3.12.14, NumPy 2.3.5. Only NumPy and the Python standard library are needed; the CSV loader does not need Pandas, SciPy, sklearn or Excel libraries.

Run these commands inside this directory:

```sh
python -m pip install -r requirements.txt
python reproduce_audit.py
python analyze_fusion_results.py
python real_air_invariant_fusion.py --data data/AirQualityUCI.csv --output results/air_invariant_reproduced.json
python real_air_paired_fusion.py --data data/AirQualityUCI.csv --output results/air_original_reproduced.json
python real_action_fusion.py --evaluate --data data/gas_drift/Dataset --output results/gas_reproduced.json
python audit_real_recovery.py
```

The full commands repeat prefix calibration, all scenarios/comparators, and sensitivity studies. `--pilot` on either Air script runs calibration only. Keep the Python filenames unchanged: Air imports `real_action_fusion`, and that module imports `da_rf_fusion`. Output parent directories must exist. The scripts write pilot JSON beside their source even when `--output` points elsewhere.

Read compressed source results using `gzip.open(path, 'rt')` followed by `json.load`. Gzip timestamps are fixed to zero for package reproducibility. The main replay is deterministic for a fixed source file, configuration and seed; extremely small BLAS-dependent floating point differences may occur on another platform. Raw return and action arrays are included for comparison.

## Data protocol

Air data source: [UCI Air Quality, dataset 360](https://archive.ics.uci.edu/dataset/360/air+quality). CSV SHA256: `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`.

The loader retains 7,344 of 9,357 raw hours after excluding missing current CO targets or any of four current sources; 2,013 rows are excluded. Sensors are PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), and PT08.S4(NO2). Each source uses its current value and preceding 31 raw hours; past sensor missing values are forward-filled causally before target filtering. Missing initial history starts at zero. Each source classifier has 32 lag features plus intercept; the candidate has 128 lag features plus intercept.

Six classes are March training CO quantile bins [1.1,1.6,2.1,2.6,3.5], not health thresholds. March non-audit rows fit initial models and normalization. April audit rows supply prefix error moments/quality. March and April non-audit rows seed the probe-training buffer. May is calibration; June 2004–April 2005 provides 5,792 complete-case test records in 187 queue windows.

Audit membership is fixed by first-eight-hex SHA256(timestamp) modulo10 <3. These audit labels are never used for model training. Source forecasts, masks, contexts and origin timestamps are saved before feedback. Labels enter the archive only at their declared arrival. Historical records are reprojected through the currently compared deployed/candidate pair without replacing their original source forecasts.

Calibration uses lambda [.25,1,4] and admission threshold [0,8,24] on three scenarios, seeds 98001–98003. **Every May grid point had zero admissions and the same 265.2 mean return.** Lambda .25/threshold0 come from a predetermined tie break, not an informative hyperparameter win. Test scenarios use seeds 99001–99010.

Gas source: [UCI Gas Sensor Array Drift, dataset224](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset). Each delivered batch hash is recorded in the original Gas JSON and manifest. B1 trains, B2 sets moments/quality, B3 calibrates, B4–B10 evaluate. Within-batch permutation scenarios are declared interventions, not true timestamp order or independent datasets.

## What actually executes

Every policy executes the same five-model numerical gradient updates: every eighth queue window, each model takes16 steps on its latest512 arrived non-audit training labels; learning rate .035 and L2 .002. The complete test has24 probes, or1,920 model-gradient steps per scenario, costing38.4 utility units at .02/step. Frozen deploy also pays this common candidate-generation cost. The controlled decision is **paid candidate deployment admission**, not whether numerical retraining occurs.

The current model serves launch-window jobs. A launch costs12 once, drops four service jobs, activates next window, observes a two-window cooldown and a monthly cap of eight. A common probe drops two jobs. Service prices1 or3 depend on known current sensor context. Gain extrapolates the current window over a four-window horizon. Delays are2–5 **complete-case queue windows**, not hours. These prices, fees, latency and occupancy are experimentally imposed; they are not measured business or hardware values.

The estimator weights within-block means and second moments by beta^(current-origin)*exp(-context_distance_squared/3^2), beta .97; the April pool is one prefix block of mass2. It uses up to64 completed archive blocks, currently selected by arrival insertion order. An origin-age control uses arrival instead of origin for decay; it is not an HMM replay ablation.

The matrix entering risk optimization is full centered action covariance divided by tr(TST), T=I-11'/n, if that trace is at least1e-6. The whole matrix is retained. Degenerate scale uses the raw matrix. Full, contextual-joint and projected-diagonal controls receive the same construction and own tangent normalization. RF-C-V uses raw scalar diagonal risk at fixed .15 temperature; its normalized variant is distinctly named. Neither is the original historical RF whole system.

## Preserved evidence

The exploratory invariant method has mean net utility2963.0. Using paired Student-t intervals with nine degrees of freedom, its mean gain is+41.9 [-1.10,84.90] versus contextual joint and+40.2 [-1.08,81.48] versus projected diagonal. These principal fusion comparisons do not establish a stable superiority claim. Mean actual action disagreements versus contextual joint are3.5 of187 per scenario. Adaptive archive versus prefix-only contributes+149.1 [114.02,184.18]; mean-bias correction versus uncorrected gain contributes+177.7 [140.76,214.64]. These contrasts are setting-specific rather than universal necessity claims. Original JSON files retain their earlier descriptive normal1.96 intervals; the accompanying Performance chapter uses the Student-t intervals as primary summaries.

Removing context or origin-age decay yields small differences whose intervals cross zero. Sensitivity and original neutral/negative outcomes remain in the preserved JSONs. The first Air scale formulation produced matching actions against contextual joint; the original Gas full method did not stably outperform its same-information joint control. Bayesian posterior block-weight uncertainty has no separate tested gain.

Same-state local forks use future labels only in offline evaluation; they do not feed the gate. They hold the current deployment state fixed and compare a candidate admission versus keep over four future windows with no later admissions. Their local sums are not the full-policy causal return difference because actual policies have different later budgets, deployment states and actions.

The independent recovery audit requires three consecutive potential-accuracy gains of at least .03 within the four-window no-further-admission fork; valuable sustained recovery additionally requires positive fee/opportunity-inclusive local utility. In the exploratory full method,35 launches contain13 positive,20 harmful and2 zero local gains;17 show sustained accuracy recovery and11 show valuable sustained recovery. Contextual joint has42 launches,14 positive,26 harmful and2 zero local gains;18 show sustained and12 valuable sustained recovery. This verifies recovery is feasible after some actual parameter updates, while clearly retaining harmful updates. The legacy `recovery` field in original result JSON follows later policy trajectories relative to the initial model and is confounded by subsequent admissions; it is not used for these causal recovery counts.

## Deployment boundary repairs

The packaged versions add cap=max(.8,1/n) for one active source and abstention for zero active sources. All-source absence blocks admissions including periodic, records unavailable gain, and keeps service with the currently deployed model. The covariance scaling avoids division by zero on empty masks. Archive storage is sliced to its last64 records after each feedback ingestion. These branches preserve the published four-source and three-source calculations; `reproduction_audit.json` records replay comparison and API checks.

Original outage stress removes one of four sources. Corner checks for zero/one source establish defined behavior, not measured missing-source performance improvement. The offline runner preconstructs complete event streams and retains output traces; this is a replay implementation, not a claim that the entire program uses bounded memory. Production can retain the64-block estimation buffer and512-row training buffer while sending full provenance logs to separate storage.

The stress mask suppresses a source forecast at fusion, while the candidate and training still receive complete sensor features. It represents source prediction/communication availability, not a physical sensor-input failure. Naturally missing current sensor/target rows are excluded from the evaluated queue; this study does not show recovery during complete physical input loss.

The block posterior module analytically decomposes predictive covariance into within-block covariance and between-block variation, and into expected conditional covariance plus posterior mean covariance. It does not provide a frequentist service certificate, covariance estimation error bound, or independent benefit claim. In the current positive replay, the actual admission gate uses an empirical score and cost threshold; it does not use a posterior confidence interval.
