# Target-dependent joint-return fusion development

These are actual **known-data development experiments**, not new physical
confirmation. RSS and UCI308 had already been inspected. The original frozen
scientific sources and result files were never edited.

## One operator and strong reductions

`joint_return.py` exports `augment(events, pre, base, cfg)` and
`make_law(d, pre, cfg, arm, api)`. Each law returns a bounded conditional target
distribution with `mean`, `variance`, and `lower_tail(alpha)`. All methods use
the same paid score `N * lower_tail(alpha) - 5`, delayed callback rule and
Fixed130/B130 execution contract.

The three laws are:

- `return_joint`: retain paired complete return and all source contrasts,
  then condition the joint Gaussian mixture on the current source vector.
- `return_factorized`: preserve **every** `(U,H_s)` pair marginal and the
  same target marginal, then condition `prod_s f(U,H_s)/f_U(U)^(m-1)`.
- `return_gaussian`: preserve the complete `(U,H)` mean and covariance,
  then use its full Gaussian conditional law.

Each receives the same nine return/source bandwidth configurations and
three calibration delay schedules. Selection uses actual guarded complete
second-half calibration utility. Every configuration and outcome is saved.
Both reductions are substantially stronger than unconditional source means.

## Retained versions

| Directory | Historical target / source representation |
|---|---|
| `core` | Original immutable issued full-lease `U=truegross/N` and original issued source contrasts. Avoids retrospective current-model relabeling, but model regimes differ across origins. |
| `crossfit` | Current candidate/reference contrasts reconstructed only on permanent hash-held-out audit rows excluded from all online training. Uses Hájek sample means. |
| `crossfit_sampling` | Same audit-only reconstruction plus declared sampling covariance, propagated through Schur completion. |
| `mean_anchor` | Same audit-only conditional laws transported monotonically within physical bounds to one common strongest calibration-selected native corrected mean. No OR or selector of actions. |

`crossfit`/`crossfit_sampling` source top docstrings were inherited from the
immutable implementation; their actual `augment`/`state` code and frozen
protocol `labels` field specify audit-only current-contrast targets. This
documentation erratum does not modify the frozen source hashes.

The audit hash is deterministic. Representativeness is an assumption, not
a randomized sample guarantee. Sampling covariance uses nominal audit rate
0.3 and the working finite-population factor 0.7. Target/source sampling
covariance is block diagonal; source/source sampling covariance remains full.
Actual calibration and utility use **full-lease** truth rather than audit
sample estimates.

The mean anchor uses prior native selection files. Their current dependency
hashes are recorded in `anchor_dependency_snapshot.json`; that later snapshot
is not presented as a pre-acquisition freeze. Original per-version protocols
were written before their new calibration/development replay runs.

## Reproduction

Use the supplied project runtime, NumPy, and the existing project data/source
closure. The work directories are project extensions, not a standalone bundle.
For each retained directory and task, phases ran sequentially:

```text
python run_return.py freeze
python run_return.py calibrate --task rss
python run_return.py calibrate --task gas
python run_return.py test --task rss
python run_return.py test --task gas
```

The scripts intentionally refuse to overwrite completed phases. To rerun,
copy source to a new output directory while preserving the same project-root
layout and inputs. The `test` command name denotes original known held-out
physical partitions; these are explicitly development data for this work.

`verify_information.py` independently enumerates an exact paired-return
counterexample. `information_witness.json` is **analytical**, not measured
physical data. `joint_return_theory.tex` derives the target-dependent law and
the strict information distinction, and states its scope.

`audit_distribution.py DIRECTORY --task rss` reconstructs each selected law
and checks every issued score before inspecting full target CDFs and tenth
quantiles. It makes no new selections. Its distribution quantiles do not
replace the executed score or its coverage denominator.

## What the runs establish

`development_results.csv` retains paid net, admissions, beneficial/harmful
actions, coverage denominators and excess. Independent request-level service
reconstruction and fork accounting are performed by the unchanged shared
auditors; `development_audit.json` records their numerical discrepancies.

The immutable RSS joint law captures one covered +25 lease missed by its
reductions (mean +5 across five delays), but does not beat native mean +44.8.
The audit-only RSS law yields calibration +85 versus +81.67/+81.33 for its
reductions and development mean +18.8 versus +15.4/+7.8. Its actual admission
coverage is only 4/9 and it incurs harmful leases. Adding sampling covariance
or mean anchoring does not establish stable extra joint utility. The anchor
recovers RSS mean +44.8 for **all** three laws and produces gas mean +677
for **all** three laws, below the native gas +904.2 pipeline. Complete outputs
prevent these known-data results from being represented as confirmed new-site
dominance or nominal admission calibration.

The information distinction is genuine: dependence between complete return
and the joint source vector can contain decision information absent from
every target/source pair and from full second-order summaries. Gaussian
mixture conditioning, covariance smoothing, Schur completion and lower-tail
utility are established mathematical tools. Full non-Bayesian joint-density
methods can preserve the same information. The performed experiments do not
demonstrate the requested comprehensive complete-pipeline superiority.
