# selfBACK physical validation: proposal, freeze and completed outcomes

The metadata-first proposal below preceded raw acquisition. The validation is now complete: all thirteen withheld participants and five fixed delay schedules were evaluated under nine pre-acquisition frozen pipelines. A later DBF published-rule benchmark is separately frozen and retains all thirteen participants. The original proposal, all freeze manifests, parser/runner versions, and failure logs are preserved.

## Completed evidence and interpretation

The delivered `selfback_report.json`, `selfback_results.csv`, `selfback_participant_results.csv`, `selfback_budget_results.csv`, and `selfback_changed_actions.csv` retain every pipeline and action comparison. CJRT-R's mean paid increment across the thirteen-participant replay is -103.2, compared with +198.8 for the strongest calibration-selected PDF pipeline and +226.2 for QMF. The paired, marginal, factorized and zero-increment policies have identical executed actions. Actual CJRT-R admission coverage is 4/25, with twelve beneficial and thirteen harmful leases. These frozen observations do not demonstrate independent paired-information improvement, complete-pipeline superiority or nominal 90% admission calibration. They must not be represented as positive independent confirmation.

All per-participant loss ledgers obey their budget. Pooled loss aggregates distinct ledgers and is not charged to one global budget. All 650 primary/extension policy trajectories, 2,600 budget paths and 120,600 ledger states pass complete-service and target reconstruction, with maximum discrepancies 1.14e-13 and 5.68e-14. The original fifty-one frozen source files remain byte-identical. Physical participants are separate retained subjects in one collection, rather than thirteen sites; delay schedules repeat those physical observations.

## Commands and versioned repairs

Use the bundled Python 3.12.14 and NumPy 2.3.5 runtime, with BLAS threads set to one. From the project root:

```text
python work/fusion_joint_return_20261005/new_physical/run_selfback_v4.py verify
python work/fusion_joint_return_20261005/new_physical/run_selfback_v2.py process
python work/fusion_joint_return_20261005/new_physical/run_selfback_v2.py calibrate
python work/fusion_joint_return_20261005/new_physical/run_selfback_v4.py test
python work/fusion_joint_return_20261005/new_physical/run_selfback_dbf_v2.py verify
python work/fusion_joint_return_20261005/new_physical/run_selfback_dbf_v2.py calibrate
python work/fusion_joint_return_20261005/new_physical/run_selfback_dbf_v2.py test
python work/fusion_joint_return_20261005/new_physical/analyze_selfback.py
```

Calibration/test commands protect completed outputs from overwriting. A fresh recomputation requires a separate copy/output tree containing the same frozen input sources and official raw archive. The analysis script recomputes reports from retained trajectories, checks the original source closure, and performs independent direct ratio-score/proposal and budget-ledger checks. It changes no controller.

The original pre-acquisition source/protocol freeze is `selfback_freeze.json` (SHA-256 `0e8def5239a0dda123102090f71cebbccebbb097cbb196361a630e5190e3ebef`). Two signal-before-read adapter repairs exclude macOS resource-fork entries and parse the native CSV headers while excluding the redundant thigh class column from features; `selfback_freeze_repair2.json` is `a80c8f882c6b10e88b6a293c2ac80ade187f1e6992c6e4bf00dc9d0f7f25bd46`.

Execution/reporting repairs preserve all numerical methods, data splits, selections, scores and actions. `run_selfback_v3.py` presents copied ratio rows to the legacy fork checker with audit-only q=0, because a ratio's saved gain already includes its lower-tail penalty; issued native q remains unchanged. `run_selfback_v4.py` attaches readiness from each corresponding issued archive and saves all raw results before summary. Its manifest `selfback_execution_repair4.json` is `50f4a2a0a76d602f05d53098ce43c5bb12c398746d7f1c1d9e1abd6bc340b81a`. Failed logs are retained; no test-based parameter, algorithm or participant replacement occurs.

DBF's original later-extension protocol hash is `dcdb8646994744c9c5363a44cf723537ffeb95759278b24d7a6a9e3d79d81010`; its metadata-only pre-calibration supplement is `1ba16cf2c7e9685ae75abe155a86e243088d2a75b5ecc6a287add9e0df6ef1ed`. It uses the published conflict-discount rule with an explicit common evidence adapter, not the original evidential backbone/loss. The extension occurs after primary raw acquisition/calibration and is not a tenth prospectively registered primary pipeline.

## Original metadata-first proposal

## Fresh collection selected

[selfBACK, UCI521](https://archive.ics.uci.edu/dataset/521/selfback), DOI `10.24432/C5SK65`, supplies native activity annotations and paired wrist/thigh accelerometers. Use twelve fit, eight calibration, and thirteen test participant ranks, determined from original numeric filename IDs before inspecting signal values. Retain all thirteen test participants. They are physical participants in one collection, not thirteen sites. The original panel timestamps, rather than an activity-sorted concatenation, determine the replay chronology.

`selfback_adapter.py` is an unexecuted metadata-first adapter. It reads original `w`/`t` sensor files, forms actual paired calendar-second bins, and retains observable timestamp gaps. The public description says the already merged `wt` panel uses timestamp pairing, but its six value columns do not provide an absolute timestamp; therefore the adapter uses raw panels to retain and independently check that provenance. No synthetic replicas or outcome-selected participants are introduced.

A native nine-class recognition task has diverse hand and locomotor activities, providing a scientifically plausible setting for sensor-specific errors. Whether those errors create useful joint paid-admission differences remains an experimental question. A one-second covariate record is twelve causal moments per physical sensor; labels, participant IDs, activity-folder names and timestamps are not prediction inputs. Labels do not trigger model resets or lag resets.

## Prior-use inventory

The workspace inventory found evaluated data/protocols for occupancy357/864, MHEALTH319, HAR240, DSADS256, gas224, RSS348, HAR70+780, AReM366, gas-home362, Daphnet245, localization196, HARTH779, gas-flow308, Appliances374 and SML274. PAMAP231 had previous partial raw acquisition, despite no completed evaluation; it cannot be described as newly acquired after this freeze. A filename search for selfBACK, WISDM, heterogeneity/HHAR or Opportunity226 found no raw/cache/evaluation artifacts. Only an earlier metadata discussion of OPPORTUNITY existed.

`inventory.json` records representative direct evidence and methodology. Replicated release directories were excluded from the content audit, while a whole-workspace filename search was also performed.

## Backups, not outcome-driven substitutes

[WISDM507](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) has 51 participants and four sensor panels on two devices, but the cross-device timestamp origin must be validated before exact physical pairing is claimed. It remains an independent optional collection to predeclare before any transfer; do not replace selfBACK based on its results.

[OPPORTUNITY226](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition) offers body, object and ambient sources with four participants and six runs per participant, but its adapters and missing-sensor interpretation require more development. [CASAS506](https://archive.ics.uci.edu/dataset/506/human+activity+recognition+from+continuous+ambient+sensor+data) offers genuinely separate homes but its full official archive is13GB. [REALDISP305](https://archive.ics.uci.edu/dataset/305/realistic+activity+recognition) offers real sensor-placement changes but the archive is2.5GB. These are metadata candidates, not evidence already collected.

The paper must distinguish physical participants/sites, imposed replay delays, and simulated deployment costs, and keep coverage failures, abstentions, and complete comparator results if they occur. Running a previously frozen algorithm does not guarantee a positive independent joint contribution.
