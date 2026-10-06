# Complete frozen OPPORTUNITY execution

Run from the delivered project root using Python 3.12 and NumPy 2.3.5. The executed environment is recorded separately from the pre-acquisition scientific freeze. No GPU, SciPy, or executable mirror code is required. These commands use the unchanged numerical V2 pipeline; the two-worker command is a separately recorded resource-only executor.

Keep the declared `opportunity/selection.json` and its lock unchanged when replaying test. The process and calibration phases intentionally reject overwriting existing cache/selection files; reproduce into a copied project with new output directories while preserving the delivered outputs as reference artifacts. Frozen source manifests contain original absolute paths, so relocation requires a path-resolving verification adapter and must be tested separately before claiming portable relocation.

```sh
TASK_PYTHON=/Users/key/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/run_opportunity_v2.py verify
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/verify_mirror_transport.py verify work/fusion_calibrated_pairing_20261005/physical/raw/opportunity_mirror_e90bded.zip
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/run_opportunity_v2.py process
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/run_opportunity_v2.py calibrate
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/run_opportunity_test_2worker.py
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/analyze_opportunity_v2.py
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/risk/final_risk_report.py work/fusion_calibrated_pairing_20261005/physical/opportunity/results.json.gz work/fusion_calibrated_pairing_20261005/physical/opportunity/risk_report.json
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/run_opportunity_v2.py verify
```

The original sequential `run_opportunity_v2.py test` is an alternative to the two-worker test executor. Do not run both into the same result directory. They use the same selected models, law library, score pools, maturity callbacks, service contract, guard, nine pipelines and 60 recording-delay worlds. The two-worker executor records an exact serial-versus-worker first-world comparison of all numerical score/action/target/budget fields, deterministic original-order merging, checkpointed world outputs and scientific source verification before and after execution. The duplicated first world serves implementation verification, not an additional independent validation unit.

The physical cache retains 289,803 rows from all 869,387 native rows via the declared stride of three, all 18 native gesture/null classes, all 24 native sessions and all three original body/object/ambient source groups. There are two held-out test participants at one collection, with six nested sessions each and five imposed delay schedules per session. Sessions and delay repetitions are not additional physical sites.

Report all nine pipelines, the strongest calibration-selected complete control, complete action-return decomposition, actual admission coverage and excess, negative complete leases and loss, all four budgets, empty-stratum fallback outcomes and independent reconstruction errors. The source and metadata-only repair freezes precede numerical payload decoding; the pinned-mirror acquisition amendment records exact byte equivalence for the 15 available complete official members plus both legends, with remaining files attributed to the pinned mirror. No test outcomes alter the law, score calibration, candidate work, data selection or selected configuration.
