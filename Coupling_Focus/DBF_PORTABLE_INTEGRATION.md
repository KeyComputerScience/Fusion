# Portable secondary DBF/EAvg launcher

Copy `portable_dbf_launcher.py` to the new package root, beside the unchanged
`run_reproduction.py`, `SOURCE_IDENTITIES.json` and `project` directory from
the original CJ package. Copy the complete DBF directory under
`project/work/fusion_coupling_focus_20261005/dbf`, preserving bytes and paths.
The builder must include every project-relative source/data/comparison file
listed by DBF `protocol.json`, plus the HARTH `algorithm_freeze.json` and
`acquisition_authorization.json`. RSS's cache appears under
`work/fusion_temporal_20261003/interface_relocated_validation/physical/data/rss348`.
The launcher checks that complete closure and uses only the unchanged earlier
launcher's `install_paths` to rebase filesystem globals/import locations in
memory. Numerical source bytes and function/class bodies are preserved.

From the package root, with a Python environment containing NumPy:

```sh
python portable_dbf_launcher.py verify
python portable_dbf_launcher.py calibrate-rss --out /tmp/dbf-fresh
python portable_dbf_launcher.py calibrate-harth --out /tmp/dbf-fresh
python portable_dbf_launcher.py locktest --out /tmp/dbf-fresh
python portable_dbf_launcher.py test-rss --out /tmp/dbf-fresh
python portable_dbf_launcher.py test-harth --out /tmp/dbf-fresh
```

Fresh calibration copies frozen code, protocols, operator/analysis checks,
caches and original CJ comparison records. It does **not** copy secondary
selections or secondary outcomes. Existing fresh replay directories are
checked against the archived protocol and launcher hash. All new output must
be outside the package and archived project; original archives are preserved.
The complete closure includes both tasks because the unchanged secondary
protocol verifies both tasks' input hashes in every phase. No redownload is
required. HARTH refitting/replay is expensive; archived analysis is smaller:

```sh
python portable_dbf_launcher.py analyze-rss --out /tmp/dbf-rss-analysis
python portable_dbf_launcher.py analyze-harth --out /tmp/dbf-harth-analysis
```

Analysis copies the relevant secondary outcomes and locked selections into
a fresh directory, excludes the old summary, calls the unchanged analyzer,
and compares all decoded summary values against the archive. For a small
portable RSS test using the archived locked configurations, without repeating
HARTH calibration:

```sh
python portable_dbf_launcher.py test-rss --archived-selections --out /tmp/dbf-rss-replay
```

That explicit mode copies both task selection locks, but no secondary test
outcomes. It is a replay of archived configurations, not fresh calibration.
For local integration testing before package construction, pass
`--archive-project /path/to/jih` and
`--reproduction-launcher /path/to/Fusion_CJ_Frozen_Physical_Repro/run_reproduction.py`.

The frozen locked-score readiness diagnostic has its own namespace at
`dbf/readiness_intervention` and consumes byte-identical original secondary
selection/calibration/outcome inputs. Its protocol hashes raw gzip files;
freshly generated gzip timestamps may differ even when decoded values agree,
so this diagnostic explicitly uses the archived original inputs.

```sh
python portable_dbf_launcher.py readiness-verify
python portable_dbf_launcher.py readiness-calibrate --out /tmp/dbf-ready-replay
python portable_dbf_launcher.py readiness-test-rss --out /tmp/dbf-ready-replay
python portable_dbf_launcher.py readiness-test-harth --out /tmp/dbf-ready-replay
python portable_dbf_launcher.py readiness-analyze-rss --out /tmp/dbf-ready-rss-analysis
python portable_dbf_launcher.py readiness-analyze-harth --out /tmp/dbf-ready-harth-analysis
```

Readiness calibration covers both tasks without selecting again. A new
`readiness-freeze` replay is also supported; it copies the original archived
inputs but omits the old readiness protocol/marker before calling the unchanged
freeze function, and is explicitly a new replay registration. Other readiness
phases preserve the archived readiness protocol and pre-intervention marker.
Archive readiness analysis copies the relevant saved diagnostic outcomes and
calibration summary, excludes its old summary, and compares the regenerated
summary against the archive. No readiness output is copied for its fresh
calibrate/test phases. The original DBF analysis aliases remain separate.
`--entry-module`, `--entry-function`, `--entry-args` and `--readiness-protocol`
permit explicit dispatch if an additional frozen integration entrypoint is
later supplied; default paths match the delivered frozen readiness source.
