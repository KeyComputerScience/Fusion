"""Read-only coverage/actions/accounting audit of saved fusion trial logs.

No models, policies, q values, parameters, or data splits are refitted. The public
``analyze_trial`` / ``analyze_study`` functions accept the original trial schema,
including new external-baseline arms. This audit reconstructs arithmetic from
immutable logs; request-level physical service remains a separate audit.

Default: audit four primary and four supplemental result files, and their
ten-delay-seed descriptive combinations. Optional additional files are audited
separately and never pooled with the locked cohorts. Only Python + NumPy needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("occupancy357", "occupancy864", "mhealth319", "har240")
FULL = "bayes_both"
COMPARATORS = ("reference", "frequentist_gate", "joint", "bayes_gate", "periodic")
EPS = 1e-8
CALIBRATED_INTERNAL_ARMS = {"bayes_gate", "bayes_both", "frequentist_gate", "diagonal_posterior_gate"}
# Two-sided 0.95 t critical values, applicable to the requested 5/10 replays.
T_CRITICAL = {4: 2.7764451051977987, 9: 2.262157162798205}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction(num: int, den: int) -> dict[str, Any]:
    return dict(numerator=int(num), denominator=int(den),
                rate=float(num / den) if den else None)


def descriptive_stats(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if not n:
        return dict(by_seed=[], n=0, mean=None, interval=None)
    critical = T_CRITICAL.get(n - 1)
    interval = None
    if n > 1 and critical is not None:
        half = critical * float(x.std(ddof=1)) / math.sqrt(n)
        interval = [float(x.mean() - half), float(x.mean() + half)]
    return dict(by_seed=x.tolist(), n=n, sum=float(x.sum()), mean=float(x.mean()),
                min=float(x.min()), max=float(x.max()),
                positive=int(np.sum(x > EPS)), tie=int(np.sum(abs(x) <= EPS)),
                negative=int(np.sum(x < -EPS)),
                conditional_delay_t95_descriptive_interval=interval,
                interval_df=n - 1 if interval is not None else None,
                interval_scope="Monte Carlo delay schedules on one fixed physical trace; not independent-subject/site generalization")


def coverage_counts(rows: list[dict], predicate=lambda r: True) -> dict:
    chosen = [r for r in rows if predicate(r)]
    return fraction(sum(bool(r["lower_covered"]) for r in chosen), len(chosen))


def action_counts(rows: list[dict]) -> dict:
    admitted = [r for r in rows if r["action"]]
    harmful = sum(float(r["local_net"]) < 0 for r in admitted)
    beneficial = sum(float(r["local_net"]) > 0 for r in admitted)
    neutral = sum(float(r["local_net"]) == 0 for r in admitted)
    return dict(issued=len(rows), admissions=len(admitted), harmful=harmful,
                beneficial=beneficial, neutral=neutral,
                harmful_fraction=fraction(harmful, len(admitted)),
                beneficial_fraction=fraction(beneficial, len(admitted)),
                admitted_actual_increment_sum=float(sum(r["local_net"] for r in admitted)),
                distinct_admitted_origins=sorted({int(r["k"]) for r in admitted}),
                distinct_harmful_origins=sorted({int(r["k"]) for r in admitted if r["local_net"] < 0}),
                distinct_beneficial_origins=sorted({int(r["k"]) for r in admitted if r["local_net"] > 0}))


def analyze_arm(trial: dict, mode: str, *, admission_restore_fee=3., drop=2,
                conservative_cost=5., alpha=.1, gamma=.05) -> dict:
    """Audit one arm against row logs and the same trial's reference baseline.

    Required row fields: k/action/local_net/truegross/gain/posterior_or_block_sd/
    q_issued/standardized_score/lower_covered/disagreement/maturity. Required arm
    fields: net/gross/fees/drops/rows/q_updates/q_initial/q_final. No truth is used
    to recompute an action. Any mismatch raises rather than silently omitting it.
    """
    saved = trial["results"][mode]
    rows = saved["rows"]
    reference = trial["baseline"]
    windows = int(trial["windows"])
    by_origin = {int(r["k"]): r for r in rows}
    assert len(by_origin) == len(rows), "duplicate issued origin"
    assert rows == sorted(rows, key=lambda r: r["k"]), "origins out of order"
    errors = dict(score=0., penalty=0., target=0., weights=0., callback=0.,
                  callback_state=0., calibration_identity=0., service_log=0.,
                  closed_loop=0., saved_counts=0., issued_q_state=0.)
    derived_rows = []
    for r in rows:
        sd = float(r["posterior_or_block_sd"])
        assert sd > 0 and math.isfinite(sd), "nonpositive forecast scale"
        score = (float(r["gain"]) - float(r["truegross"])) / sd
        errors["score"] = max(errors["score"], abs(score - r["standardized_score"]))
        covered = score <= float(r["q_issued"])
        assert bool(r["lower_covered"]) == covered, "saved coverage does not match issued score/q"
        assert float(r["q_issued"]) >= 0, "negative projected q"
        if mode in CALIBRATED_INTERNAL_ARMS:
            errors["penalty"] = max(errors["penalty"], abs(float(r["penalty"]) - float(r["q_issued"]) * sd))
        if mode == "periodic":
            assert bool(r["action"]), "periodic action contract broken"
        elif mode == "frozen":
            assert not bool(r["action"]), "keep-reference action contract broken"
        else:
            assert bool(r["action"]) == (float(r["gate_score"]) > 0), "action differs from issued gate score"
        assert int(r["k"]) < int(r["maturity"]), "feedback precedes origin"
        if "future_accuracy_gain" in r:
            assert int(r["k"]) + len(r["future_accuracy_gain"]) <= windows, "incomplete terminal lease"
        if "weights" in r:
            w = np.asarray(r["weights"], float)
            errors["weights"] = max(errors["weights"], abs(float(w.sum()) - 1.),
                                     max(0., -float(w.min())))
        pessimistic = float(r["truegross"]) - conservative_cost
        slack = float(r["local_net"]) - pessimistic
        # These are algebraically inferable from logs; no raw predictions needed.
        extra_correct = conservative_cost - admission_restore_fee - slack
        assert -EPS <= slack <= conservative_cost - admission_restore_fee + EPS, "opportunity-loss upper bound violated"
        if "pessimistic_net_target" in r:
            errors["target"] = max(errors["target"], abs(pessimistic - r["pessimistic_net_target"]))
        if "opportunity_slack" in r:
            errors["target"] = max(errors["target"], abs(slack - r["opportunity_slack"]))
        if "candidate_extra_correct" in r:
            errors["target"] = max(errors["target"], abs(extra_correct - r["candidate_extra_correct"]))
        rr = dict(r)
        rr["lower_covered"] = covered
        rr["feedback_available_at_end"] = int(r["maturity"]) <= windows
        rr["opportunity_slack_reconstructed"] = slack
        rr["forecast_penalty_bound_covered"] = float(r["gain"]) - float(r.get("penalty", 0.)) <= float(r["truegross"])
        rr["actual_gate_margin_covered"] = float(r["gate_score"]) <= float(r["local_net"])
        derived_rows.append(rr)
    assert errors["score"] < EPS and errors["target"] < EPS and errors["weights"] < EPS and errors["penalty"] < EPS
    counts = action_counts(derived_rows)
    for key, computed in (("deployments", counts["admissions"]), ("harmful", counts["harmful"]),
                          ("beneficial", counts["beneficial"]), ("zero", counts["neutral"])):
        if key in saved:
            errors["saved_counts"] = max(errors["saved_counts"], abs(float(saved[key]) - computed))
    assert errors["saved_counts"] == 0
    admitted = [r for r in derived_rows if r["action"]]
    ref_net = float(reference["gross"]) - float(reference["fees"])
    reconstructed = dict(net=ref_net + sum(r["local_net"] for r in admitted),
                         gross=float(reference["gross"]) + sum(r["local_net"] + admission_restore_fee for r in admitted),
                         fees=float(reference["fees"]) + admission_restore_fee * len(admitted),
                         drops=int(reference["drops"]) + drop * len(admitted))
    errors["service_log"] = max(abs(reconstructed[k] - float(saved[k])) for k in reconstructed)
    errors["closed_loop"] = abs(reconstructed["net"] - float(saved["net"]))
    assert errors["service_log"] < EPS
    assert abs(ref_net - reference["net"]) < EPS
    callbacks = saved.get("q_updates", [])
    callback_origins = [int(c["issued_index"]) for c in callbacks]
    expected = {int(r["k"]) for r in rows if int(r["maturity"]) <= windows}
    assert len(set(callback_origins)) == len(callback_origins), "multiple calibration callbacks for one origin"
    assert set(callback_origins) == expected, "callback cohort does not match mature issued cohort"
    assert [c["update_index"] for c in callbacks] == sorted(c["update_index"] for c in callbacks), "callback execution order reversed"
    q_state = float(saved["q_initial"])
    for c in callbacks:
        r = by_origin[int(c["issued_index"])]
        assert int(c["issued_index"]) < int(c["maturity"]) <= int(c["update_index"]) <= windows
        violation = int(float(r["standardized_score"]) > float(r["q_issued"]))
        assert int(c["violation"]) == violation
        assert int(c["maturity"]) == int(r["maturity"])
        errors["callback"] = max(errors["callback"], abs(c["issued_q"] - r["q_issued"]),
                                 abs(c["issued_std"] - r["posterior_or_block_sd"]),
                                 abs(c["standardized_score"] - r["standardized_score"]))
        proposal = q_state + gamma * (violation - alpha)
        next_q = max(0., proposal)
        regulator = next_q - proposal
        errors["callback_state"] = max(errors["callback_state"], abs(c["q_before"] - q_state),
                                       abs(c["q_after"] - next_q), abs(c["regulator"] - regulator))
        q_state = next_q
    # Independently verify each decision uses only q callbacks already executed.
    cursor = 0
    visible_q = float(saved["q_initial"])
    for r in rows:
        while cursor < len(callbacks) and callbacks[cursor]["update_index"] <= r["k"]:
            visible_q = float(callbacks[cursor]["q_after"])
            cursor += 1
        errors["issued_q_state"] = max(errors["issued_q_state"], abs(visible_q - r["q_issued"]))
    regulators = sum(c["regulator"] for c in callbacks)
    violations = sum(c["violation"] for c in callbacks)
    identity = violations - (alpha * len(callbacks) + (saved["q_final"] - saved["q_initial"] - regulators) / gamma)
    errors["calibration_identity"] = max(abs(identity), abs(q_state - saved["q_final"]))
    for key, computed in (("calibration_updates", len(callbacks)), ("clip_regulator_sum", regulators)):
        if key in saved:
            assert abs(saved[key] - computed) < EPS
    assert errors["callback"] < EPS and errors["callback_state"] < EPS and errors["calibration_identity"] < EPS and errors["issued_q_state"] < EPS
    subsets = dict(issued=lambda r: True,
                   informative_current_disagreement=lambda r: r["disagreement"] > 0,
                   admitted=lambda r: bool(r["action"]),
                   feedback_matured_by_end=lambda r: r["feedback_available_at_end"],
                   pending_feedback_at_end=lambda r: not r["feedback_available_at_end"],
                   informative_feedback_matured=lambda r: r["disagreement"] > 0 and r["feedback_available_at_end"],
                   admitted_feedback_matured=lambda r: r["action"] and r["feedback_available_at_end"],
                   admitted_informative=lambda r: r["action"] and r["disagreement"] > 0,
                   admitted_without_current_disagreement=lambda r: r["action"] and r["disagreement"] <= 0)
    cov = {key: coverage_counts(derived_rows, f) for key, f in subsets.items()}
    assert cov["feedback_matured_by_end"]["numerator"] == len(callbacks) - violations
    if saved.get("coverage") is not None:
        assert abs(saved["coverage"] - cov["issued"]["rate"]) < EPS
    if saved.get("informative_coverage") is not None:
        assert abs(saved["informative_coverage"] - cov["informative_current_disagreement"]["rate"]) < EPS
    masses = [float(r["mass"]) for r in rows if "mass" in r]
    return dict(seed=int(trial["seed"]), mode=mode, windows=windows,
                reference_net=ref_net, net=float(saved["net"]), gross=float(saved["gross"]),
                fees=float(saved["fees"]), drops=int(saved["drops"]), actions=counts,
                coverage=cov, calibration_callbacks=len(callbacks), callback_violations=violations,
                q_initial=float(saved["q_initial"]), q_final=float(saved["q_final"]),
                q_projection_regulator_sum=float(regulators), accounting_reconstructed=reconstructed,
                forecast_penalty_bound_coverage={key: fraction(sum(r["forecast_penalty_bound_covered"] for r in derived_rows if f(r)), sum(f(r) for r in derived_rows)) for key, f in list(subsets.items())[:3]},
                actual_gate_margin_coverage={key: fraction(sum(r["actual_gate_margin_covered"] for r in derived_rows if f(r)), sum(f(r) for r in derived_rows)) for key, f in list(subsets.items())[:3]},
                historical_mass_summary=dict(min=float(min(masses)), median=float(np.median(masses)), max=float(max(masses))) if masses else None,
                max_errors=errors, audit_passed=True)


def analyze_trial(trial: dict, *, modes=None, **kwargs) -> dict:
    """Public entry point for a future external-fusion result trial."""
    return {mode: analyze_arm(trial, mode, **kwargs)
            for mode in (modes if modes is not None else trial["results"])}


def paired_comparison(trials: list[dict], full_arm: str, comparator: str) -> dict:
    differences = []; gross_differences = []; saved_fees = []; events = []
    control_beneficial = retained_beneficial = control_harmful = avoided_harmful = 0
    full_harmful = full_beneficial = control_admissions = full_admissions = 0
    for trial in trials:
        f = trial["results"][full_arm]
        if comparator == "reference":
            b = dict(trial["baseline"], rows=[dict(r, action=False) for r in f["rows"]])
        else:
            b = trial["results"][comparator]
        indexed = {int(r["k"]): r for r in b["rows"]}
        assert set(indexed) == {int(r["k"]) for r in f["rows"]}
        event_sum = 0.
        for r in f["rows"]:
            s = indexed[int(r["k"])]; d = float(r["local_net"])
            assert abs(d - s["local_net"]) < EPS, "fork differences across methods violate common candidates"
            fa, ba = bool(r["action"]), bool(s["action"])
            full_harmful += fa and d < 0; full_beneficial += fa and d > 0; full_admissions += fa
            control_harmful += ba and d < 0; control_beneficial += ba and d > 0; control_admissions += ba
            retained_beneficial += fa and ba and d > 0
            avoided_harmful += not fa and ba and d < 0
            if fa == ba:
                continue
            event = ("avoided_harmful" if d < 0 else "missed_beneficial" if d > 0 else "changed_neutral") if ba else ("new_harmful" if d < 0 else "new_beneficial" if d > 0 else "changed_neutral")
            contribution = -d if ba else d
            event_sum += contribution
            events.append(dict(seed=int(trial["seed"]), origin=int(r["k"]),
                               event=event, local_net=d, contribution=contribution))
        delta = float(f["net"] - b["net"])
        gd = float(f["gross"] - b["gross"]); fs = float(b["fees"] - f["fees"])
        assert abs(delta - event_sum) < EPS and abs(delta - gd - fs) < EPS
        differences.append(delta); gross_differences.append(gd); saved_fees.append(fs)
    categories = ("avoided_harmful", "missed_beneficial", "new_harmful", "new_beneficial", "changed_neutral")
    origins = sorted({e["origin"] for e in events})
    return dict(comparator=comparator,
                paired_net_gain=descriptive_stats(differences),
                served_gross_change=descriptive_stats(gross_differences),
                saved_fees=descriptive_stats(saved_fees),
                full_counts=dict(admissions=full_admissions, harmful=full_harmful, beneficial=full_beneficial),
                control_counts=dict(admissions=control_admissions, harmful=control_harmful, beneficial=control_beneficial),
                harmful_count_reduction_fraction=fraction(control_harmful - full_harmful, control_harmful),
                harmful_control_leases_avoided=fraction(avoided_harmful, control_harmful),
                beneficial_control_leases_retained=fraction(retained_beneficial, control_beneficial),
                changed_origin_count=len(origins), changed_origins=origins,
                event_summary={key: dict(count=sum(e["event"] == key for e in events), contribution_sum=float(sum(e["contribution"] for e in events if e["event"] == key))) for key in categories},
                contribution_by_physical_origin={str(k): dict(events=sum(e["origin"] == k for e in events), contribution_sum=float(sum(e["contribution"] for e in events if e["origin"] == k))) for k in origins},
                events=events)


def aggregate_arms(per_trial: list[dict], mode: str) -> dict:
    summaries = [r[mode] for r in per_trial]
    cov_keys = list(summaries[0]["coverage"])
    cov = {key: fraction(sum(r["coverage"][key]["numerator"] for r in summaries),
                         sum(r["coverage"][key]["denominator"] for r in summaries)) for key in cov_keys}
    action_keys = ("issued", "admissions", "harmful", "beneficial", "neutral")
    counts = {key: sum(r["actions"][key] for r in summaries) for key in action_keys}
    counts["harmful_fraction"] = fraction(counts["harmful"], counts["admissions"])
    counts["beneficial_fraction"] = fraction(counts["beneficial"], counts["admissions"])
    counts["distinct_admitted_physical_origins"] = sorted(set().union(*(r["actions"]["distinct_admitted_origins"] for r in summaries)))
    counts["distinct_beneficial_physical_origins"] = sorted(set().union(*(r["actions"]["distinct_beneficial_origins"] for r in summaries)))
    counts["distinct_harmful_physical_origins"] = sorted(set().union(*(r["actions"]["distinct_harmful_origins"] for r in summaries)))
    return dict(seeds=[r["seed"] for r in summaries], seed_count=len(summaries),
                net=descriptive_stats([r["net"] for r in summaries]),
                gain_vs_shared_reference=descriptive_stats([r["net"] - r["reference_net"] for r in summaries]),
                actions=counts, coverage=cov,
                q_callback_cohort=dict(denominator=sum(r["calibration_callbacks"] for r in summaries),
                                      violations=sum(r["callback_violations"] for r in summaries)),
                max_errors={key: max(r["max_errors"][key] for r in summaries) for key in summaries[0]["max_errors"]},
                by_seed=summaries)


def analyze_study(study: dict, *, full_arm=FULL, comparators=COMPARATORS,
                  modes=None, **kwargs) -> dict:
    """Analyze one task's saved test trials; reusable for external controls."""
    trials = study["trials"]
    assert len({int(t["seed"]) for t in trials}) == len(trials), "duplicate delay seed"
    common_modes = set.intersection(*(set(t["results"]) for t in trials))
    chosen_modes = sorted(common_modes if modes is None else set(modes))
    assert set(chosen_modes) <= common_modes
    per_trial = [analyze_trial(t, modes=chosen_modes, **kwargs) for t in trials]
    pairs = {mode: paired_comparison(trials, full_arm, mode) for mode in comparators
             if mode == "reference" or mode in common_modes}
    return dict(dataset=study.get("dataset"), seeds=[int(t["seed"]) for t in trials],
                number_fixed_physical_traces=1,
                interpretation="Repeated simulated delay schedules on one fixed observation sequence; no independent-site/participant claim",
                split=study.get("split"),
                arms={mode: aggregate_arms(per_trial, mode) for mode in chosen_modes},
                full_comparisons=pairs,
                all_audits_passed=True,
                tested_modes=chosen_modes)


def load_result(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "trials" in data:
        return {data.get("dataset", path.stem): data}
    if isinstance(data, dict) and all(isinstance(v, dict) and "trials" in v for v in data.values()):
        return data
    raise ValueError(f"Unsupported trial-log schema: {path}")


def combine_fixed_trace(primary: dict, supplement: dict) -> dict:
    """Descriptive combination only; preserve the two original cohorts elsewhere."""
    assert primary["dataset"] == supplement["dataset"]
    assert primary["data_hashes"] == supplement["data_hashes"]
    assert primary["split"] == supplement["split"]
    assert all(primary["selected"][m] == v for m, v in supplement["selected"].items())
    assert all(primary["q_initial"][m] == v for m, v in supplement["q_initial"].items())
    psha = primary.get("code_sha", primary.get("base_code_sha"))
    assert psha == supplement["engine_sha256"]
    combined = dict(primary)
    combined["trials"] = list(primary["trials"]) + list(supplement["trials"])
    common = set.intersection(*(set(t["results"]) for t in combined["trials"]))
    combined["trials"] = [dict(t, results={m: t["results"][m] for m in sorted(common)}) for t in combined["trials"]]
    return combined


def fmt_ratio(value: dict) -> str:
    return f"{value['numerator']}/{value['denominator']}" + (f" ({100*value['rate']:.2f}%)" if value["rate"] is not None else " (undefined)")


def write_notes(report: dict, path: Path) -> None:
    lines = ["# Read-only coverage, actions, and gain audit", "",
             "The original 72001–72005 cohort remains primary; 72006–72010 is a later diagnostic replication under the same bounded IID delay law. The ten-seed combination is descriptive and never replaces the primary result. All ten schedules reuse the same physical trace per task. The t intervals describe imposed-delay variability and are not site/subject generalization intervals.", "",
             "## Definitions and reconstruction", "",
             "- Issued: every complete four-window lease origin logged, whether admitted or rejected. Outcomes are evaluated offline for every issued lease, including feedback not yet received at replay end.",
             "- Informative: current-window candidate/reference prediction disagreement > 0, exactly the original declared subset. It is not a definition of all decisions with nonzero historical forecasts; an admission can occur without current disagreement.",
             "- Admitted: actual policy action=true. Beneficial/harmful depend on actual local net >0/<0 after drop and admission/restoration fees. Zero denominators are undefined.",
             "- Coverage: (gain−truegross)/issued SD ≤ q issued at the decision. It compares the pessimistic target truegross−5 with gain−5−q×SD. For arms that do not execute q×SD in their gate, this is a calibration diagnostic, not their executed guarantee.",
             "- Online calibration: only callbacks whose complete lease feedback matured by replay end. This subset differs from all issued; its exact ACI identity uses its own callback count.",
             "- Accounting: J=Jreference+sum(admitted local_net); served gross=reference gross+sum(admitted(local_net+3)); fees=reference fees+3×admissions; extra dropped requests=2×admissions. The audit checks all terms, standardized scores, q-update states, one callback per mature origin, causality, target/slack bounds, counts, and reported coverage.",
             "- This recomputes arithmetic from saved logs, not request-level predictions. The earlier separate physical service/fork reconstruction remains the independent truth audit. No old engine was rerun and no method or parameter changed.", ""]
    for cohort in ("primary", "supplement", "combined_ten_seed_descriptive"):
        lines += [f"## {cohort}", "", "| Task | Issued coverage | Informative coverage | Admitted coverage | Online-matured coverage | Pending feedback coverage | Admit / harmful / beneficial |", "|---|---|---|---|---|---|---|"]
        for name, result in report["cohorts"][cohort].items():
            arm = result["arms"][FULL]; c = arm["coverage"]; a = arm["actions"]
            cells = [fmt_ratio(c[k]) for k in ("issued", "informative_current_disagreement", "admitted", "feedback_matured_by_end", "pending_feedback_at_end")]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {a['admissions']} / {a['harmful']} / {a['beneficial']} |")
        lines += ["", "| Task | Comparator | Mean net gain | Conditional 95% t interval | Gains by seed | + / = / − |", "|---|---|---:|---|---|---|"]
        for name, result in report["cohorts"][cohort].items():
            for mode in ("reference", "frequentist_gate"):
                s = result["full_comparisons"][mode]["paired_net_gain"]
                interval = s["conditional_delay_t95_descriptive_interval"]
                interval_text = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "not available"
                lines.append(f"| {name} | {mode} | {s['mean']:.4f} | {interval_text} | {s['by_seed']} | {s['positive']} / {s['tie']} / {s['negative']} |")
        lines.append("")
    lines += ["## Statements supported and limits", "",
              "- Both occupancy tasks and HAR have no harmful full admissions in both seed cohorts. Occupancy safety follows complete abstention in these traces; HAR admits only actual-positive leases. This does not establish general safety on other streams.",
              "- MHEALTH remains a material boundary: report its admitted and informative coverage and harmful admission denominator together with its gains over weaker gates.",
              "- The matched empirical-block gate is the essential strong control. It has the same interface/kernel/model work and its own prefix calibration. An increase relative to a point gate or periodic admission is not by itself a Bayesian-exclusive benefit.",
              "- Primary and supplemental counts/intervals remain separate. Combining seeds increases repeated-delay Monte Carlo observations, not the number of physical validation traces (still one per task). No all-task population utility interval is computed.",
              "- Origin identifiers are included for changed actions. Repeated changes at the same physical origin under different delays are not new sites or independent drift episodes.",
              "- Row mass is the model's context/decay pseudo-mass, not an observed number of independent blocks. Saved rows do not retain all historical kernel weights, so a Kish effective sample size or joint block-bootstrap posterior cannot be reconstructed faithfully. Resampling logged episodes would also keep the original q/action trajectory and would not constitute a fresh causal replay. No such confidence claim is made.",
              "- Existing same-IID-delay replicas are not a structural sensitivity experiment for tail delays, missing labels, interruption price, lease horizon, or reference refresh. Those require a separately locked replay protocol.", "",
              "## Reuse for external comparisons", "",
              "Import `analyze_trial(trial)` or `analyze_study(study, full_arm='bayes_both', comparators=('reference','NEW_ARM'))`. New arms must log the same issuance-time forecast/q/SD, actual action, complete-lease fork outcome, and q callbacks. A missing/inconsistent schema fails visibly rather than being silently treated as covered. An external arm that does not use posterior q still gets calibration diagnostics, with its executable penalty/gate coverage reported separately.",
              "Run with `--additional-results path/to/results.json`; additional studies are audited separately and are not pooled with old cohorts. This analyzer uses no engine imports and performs no training or tuning.", "",
              f"Input hashes unchanged after analysis: {report['input_hashes_unchanged']}. All {report['total_arm_trial_audits']} arm/seed log audits passed. Detailed per-arm and per-seed numerators, denominators, callback cohorts, action events, intervals, and accounting errors are in coverage_analysis.json.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--additional-results", nargs="*", type=Path, default=[])
    parser.add_argument("--external-full-arm", default=FULL)
    parser.add_argument("--external-comparators", nargs="*", default=list(COMPARATORS))
    args = parser.parse_args()
    primary_paths = {n: args.root / "outputs" / "bayes_closed_loop_repro" / ("new_bayes_fusion" if n.startswith("occupancy") else "new_bayes_extension") / f"{n}_results.json" for n in TASKS}
    supplement_paths = {n: args.root / "work" / "performance_delay_extension" / f"{n}_results.json" for n in TASKS}
    paths = list(primary_paths.values()) + list(supplement_paths.values()) + args.additional_results
    hashes = {str(p.resolve()): sha256(p) for p in paths}
    primary = {n: load_result(p)[n] for n, p in primary_paths.items()}
    supplement = {n: load_result(p)[n] for n, p in supplement_paths.items()}
    assert all([t["seed"] for t in d["trials"]] == list(range(72001, 72006)) for d in primary.values())
    assert all([t["seed"] for t in d["trials"]] == list(range(72006, 72011)) for d in supplement.values())
    report = dict(kind="Read-only immutable trial-log arithmetic audit; not new experiments",
                  analysis_code_sha256=sha256(Path(__file__)), input_sha256=hashes,
                  definitions=dict(issued="All complete lease origins, including rejected actions and eventual offline-evaluated outcomes",
                                   informative="Current-window disagreement>0, matching original fixed diagnostic subset",
                                   admitted="action=true", coverage="(gain−truegross)/issued_sd ≤ issued_q",
                                   interval="Paired descriptive Student t over independent imposed-delay RNG schedules conditional on the same physical trace; not independent-site evidence",
                                   combined="Later diagnostic seeds retained separately; ten-seed combination descriptive only",
                                   zero_denominator="null rate, never 0% or 100%"),
                  cohorts={}, external={}, artifact_metadata={}, total_arm_trial_audits=0)
    engine_path = args.root / "outputs" / "bayes_closed_loop_repro" / "independent_bayes_fusion.py"
    adapter_path = args.root / "outputs" / "bayes_closed_loop_repro" / "independent_bayes_extension.py"
    engine_sha = sha256(engine_path)
    adapter_sha = sha256(adapter_path)
    assert all(d.get("code_sha", d.get("base_code_sha")) == engine_sha for d in primary.values())
    assert all(d["engine_sha256"] == engine_sha for d in supplement.values())
    assert all(primary[n]["adapter_sha"] == adapter_sha for n in TASKS if not n.startswith("occupancy"))
    report["artifact_metadata"] = dict(engine_sha256=engine_sha, adapter_sha256=adapter_sha,
        per_task={n: dict(primary_protocol_sha256=primary[n]["protocol_sha"],
                         supplemental_protocol_sha256=supplement[n]["protocol_sha256"],
                         selection_sha256=primary[n]["selection_sha"],
                         raw_data_hashes=primary[n]["data_hashes"],
                         selected=supplement[n]["selected"], q_initial=supplement[n]["q_initial"]) for n in TASKS})
    for label, studies in (("primary", primary), ("supplement", supplement),
                           ("combined_ten_seed_descriptive", {n: combine_fixed_trace(primary[n], supplement[n]) for n in TASKS})):
        report["cohorts"][label] = {n: analyze_study(d) for n, d in studies.items()}
    for path in args.additional_results:
        report["external"][str(path.resolve())] = {n: analyze_study(d, full_arm=args.external_full_arm, comparators=args.external_comparators) for n, d in load_result(path).items()}
    # Unique original arm/seed audits; combined repetitions are not new checks.
    report["total_arm_trial_audits"] = sum(sum(a["seed_count"] for a in d["arms"].values()) for cohort in ("primary", "supplement") for d in report["cohorts"][cohort].values())
    report["input_hashes_unchanged"] = all(sha256(Path(path)) == digest for path, digest in hashes.items())
    assert report["input_hashes_unchanged"]
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "coverage_analysis.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_notes(report, args.output / "coverage_notes.md")
    for label, studies in report["cohorts"].items():
        for name, d in studies.items():
            a = d["arms"][FULL]
            print(label, name, "coverage", {k: fmt_ratio(a["coverage"][k]) for k in ("issued", "informative_current_disagreement", "admitted", "feedback_matured_by_end")}, "actions", {k: a["actions"][k] for k in ("admissions", "harmful", "beneficial")})
    print("ALL_LOG_AUDITS_PASS", report["total_arm_trial_audits"], "INPUTS_UNCHANGED", report["input_hashes_unchanged"])


if __name__ == "__main__":
    main()
