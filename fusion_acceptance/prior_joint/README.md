# Native prior-regularization development comparison

The frozen source-level fusion operator is unchanged. This development run
exposes the native correction-prior standard deviation as a tuning axis,
using nine `prior_sd × bandwidth` configurations and fixing tail cap one.
All five information arms receive the same 27 configuration-delay trials.
Every configuration rebuilds correction means, precision and completed
paired atoms. Shared quality, history weighting, physical model work,
complete-lease costs and the B130 guard remain unchanged.

Protocol SHA-256:
`81b10501512d389a8082786753bb76d0dab35251ca1a922fe038eec83a327003`.
The protocol was frozen before new calibration. Selections and all trial
files were hashed before separate-process test replay. RSS is already
known development data; this comparison is not new physical validation.

Joint calibration selects prior standard deviation .2, bandwidth .5,
tail cap one, and initial alpha .9711180409527267, with mean guarded
second-half increment 86.0. Its test increment is 6.6 under B130 versus
22.4 for the original frozen CJRT pipeline. The enlarged native prior
grid therefore cannot be presented as an established improvement.
No configuration is reselected using test results. All results remain
stored, including changed actions, adverse complete leases, calibration
exceedance and binding budget refusals.

Under B130 the new joint controller makes nine admissions: four beneficial,
four harmful and one zero-return lease, cumulative negative return 14.
Sixteen proposed deployments are refused. Under the unchanged selected
controller with B260 it makes 25 admissions, 18 beneficial and five harmful,
and achieves mean increment 43.4 with cumulative negative return 15.
The corresponding factorized arm achieves 54.0 at B260, so the budget
diagnostic does not establish a joint advantage over every comparator.
The primary B130 result is not replaced by the B260 diagnostic.

Independent request-level service reconstruction has zero discrepancy;
1,050 complete-lease fork checks have maximum discrepancy 3.55e-15.
All budgets satisfy the shared pathwise guard bound.

`summary.json` retains all five controllers, all four budgets, actual
action counts, all/ready/informative/admitted coverage, maximum and
cumulative admitted score excess, and four-term action-return comparisons
against both strongest native complete policies. `results.json.gz` retains
every issued score, matured callback, guarded action and ledger row.

Verification from the project root:

```sh
/Users/key/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 work/fusion_acceptance_20261005/prior_joint/run_prior.py verify
```

The original execution used separate `freeze`, `calibrate`, and `test`
commands. Existing outputs are protected from replacement by assertions.
The diagnostic imports unchanged shared functions from the sibling
strong-policy harness and the original frozen source closure.
