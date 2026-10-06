# Single conditional prequential score calibration

Scientific source: `immutable_conditional_calibration.py`.
Final SHA-256:
`f6095b0c75bbe008b89ff6865723bc3a0ae2d79fa5ef530fd2f038b4fab097c7`.
The source is fixed before the new physical protocol; later analysis can
add reports without changing any scored decision.

The raw conditional-return law is fixed at lower ES mass 0.5. Every
information arm receives the same nine calibration configurations:
bandwidth `{0.25,0.5,1}` × residual quantile `{0.8,0.9,0.95}`.
For the immutable raw paid score S, record E=(S-X)/N only after complete
counterfactual feedback matures. The signed local quantile corrects one
score, L=clip(S-N*q,-N-5,N-5); there are no OR rules or extra admission
gates. The shared Fixed130 service contract applies afterward.

The common context contains fused anchor, current disagreement fraction
and active-source fraction. It does not supply omitted historical
target–source pairing to a reduced arm. State fitting, score-residual
fitting, configuration selection and test physical records are disjoint.
The complete canonical score-fit library persists across session resets
with total mass two, and only the latest 48 mature online residuals enter
the adaptive part, with discount 0.97. Persistent fitting records are never
given a fake participant-local age. Each physical score-fit origin enters
once from a canonical delay seed, rather than once per repeated delay.

API:

```
raw = conditional_pairing.raw_run(stream, arm, raw_cfg)
pool = complete_initial_pool(score_fit_stream, raw_score_fit_rows,
    provenance={'stage': 'score_fit', 'canonical_seed': canonical_seed,
                'physical_partition': physical_partition})
issued = calibrated_run(stream, raw['rows'], calibration_cfg,
    initial_pool=pool, selection=is_selection_session)
# Apply unchanged common guard and request-level service reconstruction.
report = actual_admission_report(guarded['rows'], guarded['ledger'])
```

`selection=True` labels a whole, dedicated selection session. It never
implicitly removes half of a dedicated score-fit or selection session.
The legacy half-session mode requires an explicit `selection_boundary`.
`complete_initial_pool` defaults to callbacks matured by the score-fit
session end. A later completed flush can supply its explicitly declared
deadline as `boundary`; immature tail feedback is otherwise excluded.
Pool items retain stage, recording, origin, maturity and caller-provided
provenance. Initial test/selection records are rejected.

`actual_admission_report` reads **actual guarded actions** and the original
guard schema. Missing ledger fields raise an error rather than silently
reporting zero spent budget. Reports separate issued/ready/informative/
admitted coverage with denominators, maximum/sum/mean exceedance, negative
returns, paid increment, refusals and maximum settled-plus-reserved loss.
Empty admission coverage is `None`, never 100%.

`verify_calibration_api.py` executes malformed-input, immaturity, duplicate,
out-of-order callback, physical-support, future-truth poisoning, dedicated
selection and persistent-library tests. It also invokes the actual shared
budget guard and checks report/refusal/ledger agreement. The deliberately
synthetic checks are implementation evidence, not physical advantage data.

Risk claims:

- The operating local adaptive quantile is empirical. Setting p=0.9
  does not prove selected conditional 90% coverage.
- A valid full-information conditional residual quantile would preserve
  its risk bound under predictable admission and the common ledger. The
  tested weighted estimate is not automatically such a quantile.
- A finite-sample tolerance certificate requires independently sampled
  fixed-size score-fit errors, sufficient context strata, unchanged
  conditional residual laws, familywise allocation and adequate counts.
  Those assumptions are not asserted for participant-shift replay.
- Time-uniform risk envelopes describe mean conditional risk of already
  executed complete admission prefixes, not the next environment.
- Multiple delays of the same participant/session do not add independent
  sites. The unconditional execution statement is the shared per-chain
  loss budget; actual empirical calibration and extra paid actions must
  be measured separately.

`main_formula.tex` contains compact method prose; `appendix_proofs.tex`
contains all assumptions, proofs, certificate scope and the link between
target exceedance and cost-inclusive service. No new physical superiority
is asserted by this directory before the frozen runner produces results.
