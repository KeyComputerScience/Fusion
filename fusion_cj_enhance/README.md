# CJ-R development candidate: retained unsuccessful result

CJ-R normalizes the raw relevance weights of whole mature masked error blocks to their concentration-based effective mass, `wprime = w * sum(w) / sum(w**2)`. The same weights enter covariance, correction, Gaussian completion and paired mixture construction. It preserves relative weights and is invariant to common positive scaling at a fixed eligible set. Zero-coordinate blocks cannot add effective mass. ESS is not a count of independent observations; the original eligibility cutoff and absolute-overlap limitations remain explicit.

The scientific sources, known input caches, grids and paid selection rule were frozen in `protocol.json` before calibration. Every arm has nine configurations. Initial q uses only fully mature, ready first-half calibration scores. Selection maximizes actual second-half Fixed130/B130 guarded return, with the declared common tie rules. Every selected configuration was locked before development replay. All four physical datasets, including their earlier test partitions, were already inspected and are development evidence here.

**CJ-R is not a demonstrated improvement and is not promoted as the final candidate.** It improves guarded RSS calibration from 58.0 to 86.0 mean units, but the selected development mean falls from frozen CJ's 35.6 to 2.0. Full outcomes follow; counts pool the five delay schedules.

| RSS controller | Guarded mean increment | Beneficial | Harmful | Admitted coverage |
|---|---:|---:|---:|---:|
| Original CJ | 35.6 | 13 | 1 | 11/16 |
| CJ-R joint | 2.0 | 4 | 4 | 3/10 |
| CJ-R exact marginal | 2.0 | 4 | 4 | 2/10 |
| CJ-R fully factorized | 3.4 | 6 | 4 | 3/12 |
| CJ-R full Gaussian | 12.6 | 7 | 3 | 5/13 |
| CJ-R block sandwich | 12.6 | 7 | 3 | 5/13 |
| CJ-R joint, diagonal P | 2.0 | 4 | 4 | 3/10 |
| CJ-R Gaussian copula | 8.2 | 6 | 1 | 4/9 |
| Legacy factorized | 44.8 | 13 | 0 | 8/14 |
| Legacy block sandwich | 44.4 | 15 | 1 | 9/18 |

CJ-R's guarded seed increments are `[-4,-4,2,13,3]`. Its aggregate loss is 14, and the ledger refuses ten later proposals. Its unguarded mean is 17.4, which must not be substituted for the primary guarded value. Against original CJ, the complete four-term difference is `+11 -166 -14 +1 = -168` across the five schedules, or -33.6 mean units. Four early harmful admissions at origin 28 have effective masses 1.256–1.354 from three or four weakly relevant blocks. Original CJ rejects each. At identical CJ-R issued state/q, four joint-only crossings relative to the full Gaussian total -14, so retaining pairing in a misspecified fitted law can harm realized service.

AReM remains -30.0 mean, with five harmful leases and 0/5 admitted coverage. GasHome remains 45.2 mean, five beneficial leases and 5/5 admitted coverage. CJ-R still makes no paid admission on the previously inspected UCI196 development recordings; its normalized factorized comparator makes one harmful -13 lease. There is no newly acquired physical evaluation of CJ-R.

All served-request, fork-sum, final settlement, prefix and ledger identities pass with zero reconstruction error; current-truth poisoning leaves F/S unchanged and all history blocks are mature. The original CJ and strong legacy RSS guarded outcomes reproduce their unchanged archives. Independent mathematical/API tests under `../theory` verify coherent normalization, masked fitting, zero-mask exclusion and the concentration-versus-independent-information distinction. Mathematical consistency does not establish calibration or profitable transfer.

`*_calibration_trials.json.gz` retain every grid trial, `*_selection.json` and `all_selections_before_development.json` retain selection locks, and `*_development_results.json.gz` retain all selected rows, four budgets and fixed-state comparisons. `development_analysis.json`, `development_comparison.csv`, `changed_actions.csv` and `harmful_CJR_admissions.csv` preserve the full outcome accounting. No scientific source or old result was revised after these findings.
