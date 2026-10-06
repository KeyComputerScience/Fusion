# Transport provenance supplement

The original UCI static and legacy transfers are preserved, including two transport-closed partial archives and the paused native Chrome partial. No selected incomplete collection was evaluated. The declared physical sessions, targets, preprocessing, score law, calibration, tuning grids, source hashes, and service contract are unchanged.

The authorized transport fallback is the third-party repository `MiguelCastilloSanchez/har-with-opportunity-dataset`, pinned to commit `e90bded40b26e32edb92f6ff6e1c255aebb3ab37` (resolved root tree `bdb925f8fe6115f288f03c0544d41c07bd009a13`). Its consortium README is preserved. The complete compressed repository archive was acquired over certificate-validated HTTPS from the pinned GitHub codeload URL. None of its executable code is used.

The compressed source archive is `raw/opportunity_mirror_e90bded.zip`, 328,439,165 bytes, SHA256 `f92c5da5f207160d4a044734ba4b1ada5f22279188e1d0a252f523e5560841d5`. `opportunity_mirror_integrity.json` records all 24 declared native session files and both official legends. Every required member passes its ZIP CRC, exact byte count, and pinned Git blob SHA1; the independently retrieved root-tree metadata corroborates every required blob. The 15 complete `.dat` members available from the official UCI partial archive and both legends are byte-identical by SHA256 and CRC. The remaining nine files retain pinned-mirror provenance. This does not claim the compressed mirror or a derived transport ZIP has the original official archive SHA256.

All of these checks operate on byte streams. No sensor or annotation values were numerically parsed before the complete collection passed integrity review. A derived ZIP can retain exactly the same 24 native sessions and two legends under their original `OpportunityUCIDataset/dataset/` names, allowing the unchanged frozen V2 adapter to execute. The original official partials, original compressed pinned mirror, transport-only amendment, metadata-only repair freeze, and scientific source freezes remain available.

Transport verification is separate from the scientific numerical pipeline:

```sh
TASK_PYTHON=/Users/key/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
"$TASK_PYTHON" work/fusion_calibrated_pairing_20261005/physical/verify_mirror_transport.py verify work/fusion_calibrated_pairing_20261005/physical/raw/opportunity_mirror_e90bded.zip
```

After integrity authorization, the `repack` operation records the derived ZIP provenance; then use the exact `run_opportunity_v2.py process/calibrate/test`, `analyze_opportunity_v2.py`, and final risk-report commands listed in `OPPORTUNITY_RESUME_AND_EXECUTION.md`. Retain all nine pipelines and all 60 test recording-delay chains. No post-acquisition algorithm or outcome-driven dataset substitution is permitted.
