# Predictive Fusion Revision

This package replaces the six-item method outline with a code-aligned
algorithm and exports numerical forecasts using the supplied implementation.
The manuscript files use explicit inputs, forecast variables, candidate
values, execution branches, and feedback fields.

## Reproduce

Run with Python 3.10 or later and the standard library:

```bash
python3 export_predictions.py
python3 -m unittest test_predictions -v
```

`predictable_data.json` contains full-precision checkpoint inputs, source
scores, reliability weights, next-window forecasts, every training/inference
value, resource feasibility, and conditional recovery trajectories.
`slot_predictions.csv` contains 1,200 slot records. `training_values.csv` and
`inference_values.csv` contain the candidate values at six checkpoints.
`prediction_errors.csv` separates the decision-time forecast from its
subsequently available target. A computable forecast does not by itself
establish forecast accuracy.

## LaTeX

Load `amsmath`, `amssymb`, `algorithm`, `algpseudocode`, and `booktabs`.
The algorithm packages are documented by
[CTAN algorithms](https://ctan.org/pkg/algorithms) and
[CTAN algorithmicx](https://ctan.org/pkg/algorithmicx).
Insert `prediction_equations.tex`, `algorithm_revised.tex`, and
`prediction_case.tex` into the manuscript. The algorithm uses automatic
numbering and does not hardcode its number in the caption.
The snippets inherit the manuscript font; use Times New Roman in its
XeLaTeX preamble. All text, rules, and equations use black on white.

## Data Interpretation

The supplied stream and profile priors are synthetic and illustrative. The
export is deterministic conditional forecast data, not submission-ready
experimental evidence or a real-service performance improvement. The three
source predictors all target next-window performance degradation. Their
outputs are neither drift probabilities nor acceptance probabilities.

Logged execution feedback updates recovery and quality even when a
recommended pair differs from the logged pair. Replay does not simulate
the outcome of the alternative recommendation. The default forecast has
constant retention and zero pending gains; the live backend may supply
other decision-time forecasts.

## Implementation Alignment

| Manuscript operation | Implementation |
| --- | --- |
| Receive current inputs | `ServiceBackend.begin_slot` interface |
| Admission and retention | `Coordinator._allowed`, `Coordinator.retention` |
| Baseline, delay factors, value | `RecoveryForecast.construct`, `factors`, `coefficient` |
| Candidate cache and selection | `Coordinator.choose` |
| Feasible dispatch or service handling | `execute_one_slot` interface |
| Recovery and executed-profile quality | `Coordinator.observe` |
| Full-window evidence and predictors | `EvidenceFusion.complete_window` |

The engine files are unchanged dependency copies. The revised manuscript
algorithm does not introduce an internal event queue or claim that this
reference implementation trains a DRL policy. At startup the first complete
evaluation window establishes the comparison data and yields no source
scores. Complete-window results affect decisions in the following slot.

## Verification

The 27 supplied engine tests and six additional export checks passed. The
LaTeX fragments compile with XeLaTeX. The compilation check uses an available
Times-compatible font; the delivered fragments inherit the final manuscript's
Times New Roman setting.
