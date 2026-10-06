# Native-anchored paired-information posterior: development record

This folder implements one normalized conditional density, rather than an OR/committee admission rule:

`p_zeta(u|h) proportional to p_native(u|h) exp(zeta (log f_joint(h-u1) - log f_Gaussian(h-u1)))`, for `u in [-1,1]`.

The reference Gaussian matches the complete paired error law's full mean and covariance, including the same bandwidth, missing-source covariance completion and posterior correction state. It isolates information beyond those moments in the paired arm. Source marginals and factorized information receive direct matched controls. A single lower expected-shortfall score determines paid admission. The shared native issued `q` maps to tail mass `alpha` by `phi(Phi^-1(alpha))/alpha=q`; `alpha(0)=1`. This reproduces an untruncated native Gaussian score at zero increment. The bounded native anchor and original native controller are retained separately.

The archive and native anchor share historical observations and are **not** treated as independent likelihoods. This is a normalized conditional information correction; another full-density estimator can represent the same information. Algebraic normalization and Gaussian compatibility do not imply conditional calibration or positive utility.

## Evidence boundary

RSS and gas were both previously inspected. These runs are **known-data development**, not frozen new-environment validation. Native parameters had already been selected from nine configurations per task. Each new arm receives nine additional configurations (bandwidth 0.25/0.5/1, exponent 0/0.5/1), evaluated on the same three calibration delays, with all outcomes retained. Thus this is a disclosed second-stage matched intervention. Original native callbacks and issued `q` states remain shared; `q` is not separately refitted to the new law. Complete actual service reconstruction and fork targets use the unchanged service contract and fixed-130 budget guard.

The first written quantile-score protocol was superseded **before any experiment ran** by the ES functional. Both `*_v0_no_experiment.json` records are retained. The active protocol hash is `83a3593c7c0e071feb037b454a78619cef6b76b9024b284abce961ed975dd77c`.

## Primary locked selection and results

Calibration tie-breaks favor smaller exponent, then larger bandwidth. In both tasks every primary arm selects exponent zero. Therefore the selected rule adds **no independent paired-information increment**.

| Task | Mean complete net increment | Beneficial / harmful / zero leases | Admitted coverage |
|---|---:|---:|---:|
| RSS | 44.8 | 13 / 0 / 1 | 8/14 |
| Gas | 904.2 | 45 / 0 / 0 | 42/45 |

All four selected arms and the original strongest native controller have identical actions and net increments. These results preserve the strongest native baseline; they do **not** establish superiority over it. No primary result is replaced with an intervention selected from inspected test outcomes.

`theory.tex` gives the exact exponential-tilt and paid-ES influence identities. `identity_checks.json` independently evaluates the derivative with error 3.44e-6, and the Gaussian ES mapping with error at most 1.21e-9, using 131,073 integration nodes. The production replay uses 2,049 nodes; its numerical approximation is not a statistical coverage certificate.

The separate `grid_diagnostic_protocol.json` was declared after the primary known-data runs. It evaluates all configurations as operating diagnostics only and never changes the locked primary selections. On RSS all paired exponents preserve the native actions and utility; nonzero exponents change admitted coverage from 8/14 to 9/14 and reduce total optimistic excess. This is a conditional fixed-trace diagnostic, not nominal coverage certification or independent deployment gain.

## Reproduction

Run from the project root with the supplied Python/NumPy runtime. Original dependency modules are addressed by their project paths, so this folder is not a standalone portable package.

```sh
python work/fusion_joint_return_20261005/ratio/run_ratio.py calibrate --task rss
python work/fusion_joint_return_20261005/ratio/run_ratio.py test --task rss
python work/fusion_joint_return_20261005/ratio/run_ratio.py calibrate --task gas
python work/fusion_joint_return_20261005/ratio/run_ratio.py test --task gas
python work/fusion_joint_return_20261005/ratio/verify_identity.py
python work/fusion_joint_return_20261005/ratio/run_grid_diagnostic.py run --task rss
python work/fusion_joint_return_20261005/ratio/run_grid_diagnostic.py run --task gas
```

The selection/test commands reject overwriting existing results. For a fresh run copy the folder to a new output location and preserve the frozen source/dependency closure, or remove only newly generated result subdirectories from a separately archived working copy. A pre-existing protocol is verified before use. `protocol.json` lists the imported scientific source hashes and selected native parameter hashes. No original manuscript or numerical source is edited by this runner.

The gas operating grid is also complete. At the common bandwidth 1, exponent 1, paired information preserves the native 904.2 mean increment and 45 beneficial/0 harmful leases. The marginal and factorized counterparts each realize 855.4, with 43 beneficial/0 harmful leases: two additional beneficial paired admissions account for 244 pooled utility units, or 48.8 per delay schedule. Their strongest bandwidth-0.25 controls also realize 904.2, so the operating-point result does not establish superiority over the strongest complete reduced pipelines. No posterior exponent achieves more than the native baseline on either task. `complete_grid.csv` preserves all 72 task/arm/configuration rows, and `grid_action_decomposition.json` gives every changed action at the shared bandwidth-1/exponent-1 diagnostic. This record cannot be used as prospective independent validation.

## Prospective version 2 (CJRT-R)

`prospective_v2.py` adds a reusable API without changing any original prototype code or result. Its pre-fresh-data selection prefers the largest information exponent, then largest bandwidth, when paid calibration return ties; all four ratio arms use exactly this tie-break. The change is a declared known-data design choice, not evidence of a new benefit. Method-definition SHA-256 is `6d41bb15fd81dfa0429bab1315f162b97f3013176a94c54e62e10680b639c33a`.

The definition is method-only: a complete physical runner must separately freeze its adapters, sources, costs, windows, splits, calibration/test seeds and raw-acquisition status before reading new data. The helper's `freeze_definition` can record unread raw paths and rejects paths that already exist. Each ratio arm is charged the common nine native calibration trials plus its nine additional ratio trials. It does not refit source models or scalar calibration with test feedback. `summary` reports all-issued, ready, informative and actual-admitted exact coverage denominators, excess max/sum/mean, harmful leases, negative loss and guard refusals. Empty admission denominators represent missing evidence, not perfect coverage.

The Gaussian compatibility theorem is mathematical. Production scoring imposes a numerical tail-mass floor `1e-12`; compatibility applies where this guard and the finite normal-inversion search are inactive. Such quadrature/support guards do not provide a confidence or coverage guarantee.
