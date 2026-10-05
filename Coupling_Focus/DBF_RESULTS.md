Completed DBF/EAvg secondary benchmark and locked-score readiness diagnostic
===========================================================================

Original science, previously published/frozen results, and manuscript source are
unchanged. The only new files are inside this DBF extension folder. This is a
known-data benchmark extension and a subsequently declared diagnostic; neither
constitutes fresh physical validation or original DBF EDL training.

Primary paper: https://proceedings.mlr.press/v258/bezirganyan25a.html
Author operator: https://github.com/bezirganyan/DBF_uncertainty/blob/main/model.py
`DBFModel.get_doc_belief_fusion`, primary lambda=1, author epsilon=1e-6.
Common evidence adapter: `e_s=kappa*(q_s/mean(prefix q))*p_s`; `alpha=e+1`.
EAvg uses the same adapter/search and arithmetic mean evidence.

Protocol froze before secondary calibration (SHA256
`6e08d55dde6e2b0b35e5f7b350d378d991883a9d946828ddb000f330bc94a3e3`).
The temporary initial name `operator.py` was renamed to `dbf_operator.py` before
this freeze and before any calibration. All frozen source paths bind the latter.
Both rules selected highest evidence strength and margin0: RSS kappa8,
HARTH kappa48. RSS q_initial DBF0.0029220949481223394, EAvg0.00036718596486584327;
HARTH both0. The RSS and HARTH selection SHA256 values are respectively
`2200b2f8dda4096547ebe10d65b5cb876245b74b139c3c8bf4fd1aec2359988e`
and `5f0ab4de142f187952312454142402192b09e8f97f41a0bd42b20ca6e90b4e5f`.
Both selections jointly locked at 2026-10-05T00:40:24.464584 UTC before secondary
test replay. All18 configurations per task and three calibration delays are
retained in the compressed calibration-trial files.

| Collection | Policy | Original mean delta | Ready-only mean delta | Ready-only actions (+/-/0) | Covered ready-only admissions |
|---|---|---:|---:|---:|---:|
| RSS | CJ archived | 35.6 | 35.6 | 16 (13/1/2) | 11/16 |
| RSS | DBF adapter | 16.2 | 16.2 | 10 (6/3/1) | 6/10 |
| RSS | EAvg adapter | 19.4 | 19.4 | 12 (8/3/1) | 7/12 |
| HARTH | CJ archived | 48.2 | 48.2 | 12 (10/1/1) | 9/12 |
| HARTH | DBF adapter | 654.4 | 110.2 | 23 (15/7/1) | 15/23 |
| HARTH | EAvg adapter | 664.8 | 108.8 | 26 (16/9/1) | 15/26 |

Original HARTH DBF admitted67 (59/7/1), coverage59/67, negative loss54;
EAvg admitted75 (62/12/1), coverage61/75, negative loss102. The original
unready admissions account for44 (all beneficial) and mean544.2 for DBF;
49 (46 beneficial,3 harmful) and mean556.0 for EAvg.

The readiness diagnostic was declared after original benchmark outcomes, with
its own protocol SHA256
`be0d66cba05e4eeb314aa1448345943a78fe4c6d7c0555623f222523353c7b09`.
Only unready original proposals are prohibited. Original locked kappa, margin,
F/S, q_initial, issuance q and callbacks remain unchanged. Every chronological
ledger and explicit service path was reconstructed from raw proposals, including
all budgets0/110/130/260. No configuration was reselected. RSS selected
calibration91.333 remains unchanged; HARTH selected13.0 becomes4.333 for both
rules. The intervention does not align original all-origin q calibration or the
empirical information law with CJ.

Initial-q denominators: RSS60=51ready+9unready; HARTH176=69ready+107unready.
Test callbacks: RSS205=190ready+15unready per rule;
HARTH1360=764ready+596unready per rule. These all-origin external-helper
denominators differ from CJ's ready-only calibration and remain so in the
locked-score diagnostic.

Pooled five-delay four-term CJ-minus-rule differences (kept - missed - incurred + avoided):

| Collection / diagnostic | Comparator | Four terms | Total / mean | Changed actions |
|---|---|---|---|---:|
| RSS original and ready | DBF | 129 - 41 - 1 + 10 | 97 / 19.4 | 16 |
| RSS original and ready | EAvg | 129 - 57 - 1 + 10 | 81 / 16.2 | 18 |
| HARTH original | DBF | 2 - 3076 - 11 + 54 | -3031 / -606.2 | 59 |
| HARTH original | EAvg | 0 - 3174 - 11 + 102 | -3083 / -616.6 | 65 |
| HARTH ready | DBF | 2 - 355 - 11 + 54 | -310 / -62.0 | 15 |
| HARTH ready | EAvg | 0 - 355 - 11 + 63 | -303 / -60.6 | 16 |

DBF-minus-EAvg is -3.2 on RSS (two missed gains totaling16); ready-only
HARTH is +1.4 (missed2, avoided9; three changed actions). The conflict operator
therefore has no consistent return advantage over the common-adapter control.

Checks completed:

- Independent scalar loop DBF versus broadcast operator error <=1.11e-16;
  source permutations, zero evidence, no-epsilon singleton and identical-opinion
  invariants, opinion normalization and evidence round trip pass. Author epsilon
  introduces singleton contraction5.14e-7, explicitly recorded.
- Original RSS10 and HARTH1950 controller trajectories and40/7800 budget
  trajectories retain all five original test delay schedules and all195 HARTH
  chains, including zero-admission chains. All service/fork/causal/ledger audits pass.
- Readiness diagnostic retains40/7800 budget trajectories and10/1950 exact
  original q-trajectory matches. All420/3120 world origin/return/maturity matches pass.
  Maximum explicit service/fork error0; maximum target error1.42e-14;
  maximum q identity error1.78e-15. All diagnostic admissions are ready.
- Loss<=B per chain, minimum prefix>=-B, and spent+reserved<=B are checked
  at every ledger event. HARTH aggregate bound is195*B per delay schedule.
- The exact prefix gradient cache recomputes its first hit bitwise. HARTH test
  cache2922hits/3misses and RSS20hits/5misses; frozen models and work charges
  are unchanged. No neural network was trained.

Numerical summaries and complete row/action ledgers are in `rss348/`,
`harth779/`, and `readiness_intervention/`. `benchmark_fragment.tex` is an
integration fragment only. `portable_dbf_launcher.py` and its integration notes
provide the filesystem-only portable launcher without altering numerical bodies.
