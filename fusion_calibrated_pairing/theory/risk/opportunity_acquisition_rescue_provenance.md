# OPPORTUNITY 226 read-only acquisition rescue provenance

This metadata investigation obtained no `.dat` sensor or label values and changed no scientific algorithm, scorer, calibration rule, split, metric, comparator, or frozen numerical source.

## Verified HTTPS source inventory

- Repository: https://github.com/MiguelCastilloSanchez/har-with-opportunity-dataset
- Pinned commit: `e90bded40b26e32edb92f6ff6e1c255aebb3ab37` (2024-12-13).
- Commit's root tree: `bdb925f8fe6115f288f03c0544d41c07bd009a13`.
- Recursive tree metadata: https://api.github.com/repos/MiguelCastilloSanchez/har-with-opportunity-dataset/git/trees/bdb925f8fe6115f288f03c0544d41c07bd009a13?recursive=1
- All 24 expected native files are present: S1--S4, each ADL1--ADL5 and Drill. The complete nontruncated metadata reports 885,433,599 uncompressed bytes. Every sensor file is an ordinary Git blob, 22.66--72.12 MB, rather than an LFS pointer.
- Pinned archive transport candidate: https://codeload.github.com/MiguelCastilloSanchez/har-with-opportunity-dataset/zip/e90bded40b26e32edb92f6ff6e1c255aebb3ab37
- Per-member pinned HTTPS URLs, sizes and Git blob IDs are recorded in `opportunity_mirror_verified_metadata.json`.
- The repository is a third-party mirror, not the original author or UCI host. It retains the consortium's June 2012 / March 2013 README, documentation, and original scripts. Original full UCI archive byte equivalence is not yet established by this metadata alone.

The README content recovered through the GitHub connector was re-encoded as its original Windows-1252 bytes; 2,321 bytes give Git blob ID `12ce22d4e664c6525cc0a5935d7a997b8a294007`, exactly matching the pinned tree. The saved original-byte documentation has SHA-256 `146e483e98fa19e3e9f556be653b1366fe9ebcc5e5468a99b031e15ffc5751fa`. This checks the mirror README's transport integrity, not its equality to an independently obtained UCI member.

## Primary provenance references

UCI's official metadata identifies the dataset, its authors, DOI 10.24432/C5M027, and original `OpportunityUCIDataset` directory:
https://archive.ics.uci.edu/dataset/226/opportunity%2Bactivity%2Brecognition

The original authors' institutional GitHub code downloads the original UCI archive and uses the same native filenames. Its processed 18-session challenge output excludes S4 and cannot replace the frozen 24-session protocol:
https://github.com/STRCWearlab/DeepConvLSTM/blob/master/preprocess_data.py

The original author's later IEEE Opportunity++ release documents the same four people with six runs each and the original physical sensor groups, but adds video and skeleton data. Its 2.37 GB archive needs IEEE login and its native-member equivalence is unverified; it is therefore only a corroborating provenance route:
https://ieee-dataport.org/open-access/opportunity-multimodal-dataset-video-and-wearable-object-and-ambient-sensors-based

The related author paper corroborates four people, six runs each and 242 physical sensor channels:
https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.792065/full

## Acceptance criteria before numerical decoding

The authorized acquisition runner should independently:
1. Verify the pinned commit/root tree and complete 24-name inventory against the saved metadata.
2. Validate full acquired ZIP structure and every member CRC, with no truncated archives.
3. Match each downloaded native member's declared size and Git blob hash before decoding.
4. Compare, by raw-byte hash/CRC only, every fully obtained original UCI partial-archive member with its corresponding mirror member; do not compare an incomplete decompression as if complete.
5. Compare independently obtained original UCI documentation bytes/text to the mirror's original documentation, preserving encoding or explicitly normalizing text.
6. Record all matching original members and all remaining members without upstream byte verification separately.
7. Preserve the numerical source freeze and execute the same whole 24-session dataset and frozen splits only after transport validation.

A complete pinned mirror with all member CRC/hash checks and matching available original members supports a transparent mirror-assisted acquisition. It does not establish the SHA-256 of an unavailable complete original UCI ZIP or prove byte equivalence of all remaining UCI members. The archive URL and mirror commit must be disclosed.

