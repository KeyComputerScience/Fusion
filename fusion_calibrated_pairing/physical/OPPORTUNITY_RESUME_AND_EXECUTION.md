# OPPORTUNITY acquisition and exact execution plan

The same official collection remains declared. Its full archive is pending. All 24 native sessions must be present; do not process selected complete entries from a truncated ZIP. No sensor or label .dat values have been decoded and no OPPORTUNITY performance result exists.

The current native Chrome download of `opportunity+activity+recognition.zip` is resumed. Continue only this existing official transfer through Chrome's supported download controls if needed; preserve partial files and TLS validation. No additional automatic task or scheduled validation was created.

After Chrome visibly reports completion, use its “Show in folder” control to identify the completed file. The default expected location is `/Users/key/Downloads/opportunity+activity+recognition.zip`; do not substitute the `.crdownload` temporary file. Copy the completed file into the workspace as `raw/opportunity226.zip`, preserving the browser original and every failed partial.

Before sensor decoding, validate the ZIP central directory, all 24 native names and the two official legend hashes against `opportunity_metadata_repair_freeze.json`. Record the complete archive SHA256, byte count and UTC in a new acquisition-completion manifest. This is acquisition provenance, not a new scientific freeze.

Run from the project root after the full archive has passed those checks:

```sh
TASK_PYTHON=/Users/key/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/run_opportunity_v2.py verify
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/run_opportunity_v2.py process
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/run_opportunity_v2.py calibrate
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/run_opportunity_v2.py test
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/analyze_opportunity_v2.py
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/risk/final_risk_report.py work/fusion_calibrated_pairing_20261005/physical/opportunity/results.json.gz work/fusion_calibrated_pairing_20261005/physical/opportunity/risk_report.json
```

Calibration selects nine configurations per pipeline on three declared delays across both S2 selection sessions, saving the selection lock before test. Test retains all nine pipelines, all12 S3/S4 sessions and all five delays. This is two new test participants at one collection, not12 sites or60 independent physical streams. Report the strongest calibration-selected complete control, all action changes and four-term return decomposition, actual admission coverage and excess, harmful leases/loss, budget activity, empty-stratum suppressed/missed beneficial opportunities and independent service/fork checks. Preserve ties and losses and do not change the numerical method after viewing physical outcomes.

The original candidate/authorization freezes and metadata-only repair supplement are preserved; use V2 adapter/runner/analyzer. No main manuscript, editor document or scientific source was changed by this transport status report.
