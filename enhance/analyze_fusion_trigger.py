"""Read-only aggregation of executed fusion-trigger runs and insertable tables.

This helper performs no experiments and assumes no method wins. Its sign-flip
test uses the paired MEAN (not signed ranks); seed-level bootstrap intervals are
descriptive for this frozen protocol. Raw outcomes and actual expenditure remain
separate from a common budget ceiling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np


HERE = Path(__file__).resolve().parent
METRICS = {
    "accuracy": ("accuracy", "overall_accuracy"),
    "post_boundary_accuracy": ("post_boundary_accuracy", "post_accuracy"),
    "last_512_accuracy": ("last_512_accuracy", "tail_accuracy", "last_window_accuracy"),
    "reward_sum": ("reward_sum", "return", "cumulative_reward"),
    "actual_gradient_updates": ("actual_gradient_updates", "gradient_updates", "gradient_steps"),
    "deployments": ("deployments", "deployed_jobs"),
    "training_jobs_started": ("training_jobs_started", "training_jobs", "jobs_started"),
    "normalized_training_cost": ("normalized_training_cost",),
}
ACCURACIES = {"accuracy", "post_boundary_accuracy", "last_512_accuracy"}
RISK_NAMES = ("risk_rf", "riskrf", "risk_fusion", "full_risk_rf", "full_risk", "revised", "revised_residual_fusion")
NAMES = {"risk_rf": "Risk RF", "riskrf": "Risk RF", "risk_fusion": "Risk RF",
         "full_risk_rf": "Risk RF", "revised": "Risk RF", "revised_residual_fusion": "Risk RF",
         "observed_loss": "Observed loss", "periodic": "Periodic", "equal": "Equal",
         "service_only": "Service only", "context_only": "Context only", "workload_only": "Context only", "no_rt": "No retraining",
         "no_variability": "RF - variability", "no_risk": "RF - risk", "no_residual": "RF - residual risk",
         "no_disagreement": "RF - disagreement", "no_inertia": "RF - inertia"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def first(row, aliases):
    return next((row[key] for key in aliases if row.get(key) not in (None, "")), None)


def numerical(value):
    if value in (None, ""):
        return None
    answer = float(value)
    if not math.isfinite(answer):
        raise ValueError("Refuse nonfinite observed result: " + repr(value))
    return answer


def budget_ceiling(row, config):
    direct = first(row, ("budget_ceiling_updates", "gradient_budget_ceiling", "maximum_gradient_updates", "update_budget"))
    if direct is not None:
        return numerical(direct)
    direct = first(config, ("budget_ceiling_updates", "gradient_budget_ceiling", "maximum_gradient_updates", "update_budget"))
    if direct is not None:
        return numerical(direct)
    jobs = first(row, ("maximum_jobs", "max_jobs", "maximum_training_jobs", "maximum_jobs_per_run"))
    steps = first(row, ("job_gradient_steps", "gradient_steps_per_job"))
    if jobs is None:
        jobs = first(config, ("maximum_jobs", "max_jobs", "maximum_training_jobs", "maximum_jobs_per_run"))
    if steps is None:
        steps = first(config, ("job_gradient_steps", "gradient_steps_per_job"))
    return numerical(jobs) * numerical(steps) if jobs is not None and steps is not None else None


def load_results(source):
    source = Path(source).resolve()
    directory = source if source.is_dir() else source.parent
    metadata = {}
    for name in ("manifest.json", "frozen_config.json", "config.json"):
        path = directory / name
        if path.exists():
            metadata[name] = read_json(path)
    config = metadata.get("frozen_config.json", metadata.get("config.json", {}))
    if source.is_dir():
        paths = [source / name for name in ("seed_results.csv", "seed_results.json") if (source / name).exists()]
        if paths:
            paths = paths[:1]
        else:
            paths = sorted((source / "runs").rglob("summary.json"))
    else:
        paths = [source]
    if not paths:
        raise FileNotFoundError("No seed_results.csv/json or runs/**/summary.json in " + str(source))
    rows = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                loaded = list(csv.DictReader(stream))
        else:
            value = read_json(path)
            if isinstance(value, list):
                loaded = value
            elif isinstance(value, dict) and "method" in value:
                loaded = [value]
            else:
                loaded = next((value[key] for key in ("seed_results", "runs", "summaries")
                               if isinstance(value, dict) and isinstance(value.get(key), list)), None)
                if loaded is None:
                    raise ValueError("JSON must contain individual run summaries: " + str(path))
        for original in loaded:
            required = ("method", "backbone", "scenario", "seed")
            if any(original.get(key) in (None, "") for key in required):
                raise ValueError("Every run needs method/backbone/scenario/seed: " + str(path))
            row = {key: str(original[key]) for key in required}
            row["seed"] = int(original["seed"])
            for metric, aliases in METRICS.items():
                row[metric] = numerical(first(original, aliases))
                if metric in ACCURACIES and row[metric] is not None and not 0 <= row[metric] <= 1:
                    raise ValueError("Accuracy must be a fraction in [0,1], not a percentage")
            row["budget_ceiling_updates"] = budget_ceiling(original, config)
            row["job_gradient_steps"] = numerical(first(original, ("job_gradient_steps", "gradient_steps_per_job"))
                or first(config, ("job_gradient_steps", "gradient_steps_per_job")))
            row["deployment_delay_slots"] = numerical(first(original, ("deployment_delay_slots", "deployment_delay"))
                or first(config, ("deployment_delay_slots", "deployment_delay")))
            if row["actual_gradient_updates"] is not None and row["budget_ceiling_updates"] is not None:
                if row["actual_gradient_updates"] > row["budget_ceiling_updates"] + 1e-9:
                    raise ValueError("Actual updates exceed declared common ceiling")
            rows.append(row)
    identities = [(r["backbone"], r["scenario"], r["method"], r["seed"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate run identities: separate different budgets/replicates into explicit scenarios")
    provenance = {"source_files": [{"path": str(p), "sha256": digest(p)} for p in paths],
                  "metadata": metadata, "individual_runs": len(rows)}
    return rows, provenance


def describe(values):
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values), "maximum": max(values)}


def stable_seed(base, key):
    return (int(base) + int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")) % (2 ** 64)


def sign_flip_mean(differences, rng_seed, draws=100000):
    values = np.asarray(differences, dtype=float)
    nonzero = values[np.abs(values) > 1e-14]
    n = len(nonzero)
    if not n:
        return 1.0, "exact_sign_flip_mean", 1, 0
    observed = abs(float(nonzero.sum()))
    tolerance = max(1e-14, float(np.abs(nonzero).sum()) * 1e-12)
    hits = 0
    if n <= 20:
        count = 2 ** n
        shifts = np.arange(n, dtype=np.uint64)
        for start in range(0, count, 32768):
            integers = np.arange(start, min(start + 32768, count), dtype=np.uint64)
            signs = 2.0 * ((integers[:, None] >> shifts) & 1).astype(float) - 1.0
            hits += int(np.count_nonzero(np.abs(signs @ nonzero) >= observed - tolerance))
        return hits / count, "exact_sign_flip_mean", count, n
    rng = np.random.default_rng(rng_seed)
    for start in range(0, draws, 2048):
        signs = 2 * rng.integers(0, 2, size=(min(2048, draws - start), n)) - 1
        hits += int(np.count_nonzero(np.abs(signs @ nonzero) >= observed - tolerance))
    return (hits + 1) / (draws + 1), "monte_carlo_sign_flip_mean_plus_one", draws, n


def paired_statistics(differences, *, key, base_seed, bootstrap_draws=10000):
    values = np.asarray(differences, dtype=float)
    seed = stable_seed(base_seed, key)
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_draws)
    for start in range(0, bootstrap_draws, 2048):
        stop = min(start + 2048, bootstrap_draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    interval = np.quantile(means, [0.025, 0.975])
    p, test, permutations, nonzero = sign_flip_mean(values, stable_seed(base_seed, key + "/signflip"))
    return {"mean_difference": float(values.mean()), "sd_difference": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95_low": float(interval[0]), "ci95_high": float(interval[1]),
            "two_sided_p": p, "test": test, "sign_flip_outcomes_or_draws": permutations,
            "nonzero_seed_pairs": nonzero, "bootstrap_draws": bootstrap_draws, "bootstrap_rng_seed": seed}


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["backbone"], row["scenario"], row["method"]), []).append(row)
    summary = []
    for (backbone, scenario, method), selected in sorted(groups.items()):
        result = {"backbone": backbone, "scenario": scenario, "method": method,
                  "n": len(selected), "seeds": sorted(r["seed"] for r in selected)}
        for metric in (*METRICS, "budget_ceiling_updates", "job_gradient_steps", "deployment_delay_slots"):
            values = [r[metric] for r in selected if r[metric] is not None]
            if values:
                result.update({metric + "_" + key: value for key, value in describe(values).items()})
            else:
                result[metric + "_n"] = 0
        summary.append(result)
    return summary


def paired_comparisons(rows, risk_method, base_seed):
    comparisons = []
    for backbone, scenario in sorted({(r["backbone"], r["scenario"]) for r in rows}):
        selected = [r for r in rows if (r["backbone"], r["scenario"]) == (backbone, scenario)]
        candidates = {r["seed"]: r for r in selected if r["method"] == risk_method}
        for control in sorted({r["method"] for r in selected} - {risk_method}):
            controls = {r["seed"]: r for r in selected if r["method"] == control}
            for metric in METRICS:
                seeds = sorted(seed for seed in candidates.keys() & controls.keys()
                               if candidates[seed][metric] is not None and controls[seed][metric] is not None)
                if not seeds:
                    continue
                diffs = [candidates[seed][metric] - controls[seed][metric] for seed in seeds]
                key = f"{backbone}/{scenario}/{risk_method}-{control}/{metric}"
                comparisons.append({"backbone": backbone, "scenario": scenario, "risk_method": risk_method,
                    "control": control, "metric": metric, "n_pairs": len(seeds), "seeds": seeds,
                    "unpaired_or_missing_seeds": sorted((candidates.keys() | controls.keys()) - set(seeds)),
                    "seed_differences": dict(zip(map(str, seeds), diffs)),
                    "positive_difference_meaning": "higher_accuracy_favors_Risk_RF" if metric in ACCURACIES
                    else "higher_reward_favors_Risk_RF" if metric == "reward_sum" else "greater_actual_expenditure_for_Risk_RF",
                    **paired_statistics(diffs, key=key, base_seed=base_seed)})
    return comparisons


def holm_clean_post(comparisons, clean_scenario, backbones):
    family = [r for r in comparisons if r["control"] == "observed_loss"
              and r["scenario"] == clean_scenario and r["metric"] == "post_boundary_accuracy"
              and r["backbone"] in backbones]
    if len(family) != len(backbones) or {r["backbone"] for r in family} != set(backbones):
        raise ValueError("Requested Holm family is incomplete: need each predefined backbone's clean post RiskRF-minus-observed_loss comparison")
    previous = 0.0
    for index, row in enumerate(sorted(family, key=lambda r: r["two_sided_p"])):
        previous = max(previous, min(1.0, (len(family) - index) * row["two_sided_p"]))
        row["holm_adjusted_p"] = previous
        row["holm_family"] = "clean_post_vs_observed_loss:" + ",".join(backbones)
    return len(family)


def latex_escape(value):
    mapping = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
               "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(character, character) for character in str(value))


def pm(row, metric, scale=1, digits=2, latex=False):
    if not row.get(metric + "_n"):
        return "NA"
    mean, sd = row[metric + "_mean"] * scale, row[metric + "_sd"] * scale
    return (f"${mean:.{digits}f}\\pm{sd:.{digits}f}$" if latex
            else f"{mean:.{digits}f} ± {sd:.{digits}f}")


def ceiling_text(row):
    if not row.get("budget_ceiling_updates_n"):
        return "NA"
    low, high = row["budget_ceiling_updates_minimum"], row["budget_ceiling_updates_maximum"]
    return f"{low:g}" if low == high else f"{low:g}--{high:g}"


def latex_table(summary, path, selected=False, label_prefix="fusion-trigger"):
    scope = ("This fragment shows explicitly selected scenarios; every executed scenario remains in summary.csv and trigger_comparison_full.tex."
             if selected else "All methods and scenarios are retained.")
    lines = ["% Insertable fragment: requires booktabs and longtable; no document preamble.",
        r"\begin{longtable}{lrrrrrr}",
        r"\caption{Executed fusion-trigger comparison. Post and tail accuracy are percentages; all means use individual seed runs and sample SD. Actual gradient steps and deployments are reported separately from the declared common gradient-step ceiling. An upper limit does not imply equal realized expenditure. This is a controlled synthetic two-source contextual-bandit adaptation, not a queue completion, full three-source service implementation, or hardware result. " + scope + r"}\label{tab:" + label_prefix + ("" if selected else "-full") + r"}\\",
        r"\toprule", r"Method & $n$ & Post (\%) & Tail (\%) & Actual steps / ceiling & Deployments & Reward \\",
        r"\midrule", r"\endfirsthead", r"\toprule",
        r"Method & $n$ & Post (\%) & Tail (\%) & Actual steps / ceiling & Deployments & Reward \\",
        r"\midrule", r"\endhead"]
    previous = None
    for row in summary:
        group = (row["backbone"], row["scenario"])
        if group != previous:
            lines.append(r"\multicolumn{7}{l}{\textbf{" + latex_escape(row["backbone"].upper() + ": " + row["scenario"]) + r"}} \\")
            previous = group
        name = latex_escape(NAMES.get(row["method"], row["method"]))
        cells = [name, str(row["n"]), pm(row, "post_boundary_accuracy", 100, latex=True),
                 pm(row, "last_512_accuracy", 100, latex=True),
                 pm(row, "actual_gradient_updates", digits=1, latex=True) + " / " + ceiling_text(row),
                 pm(row, "deployments", digits=1, latex=True), pm(row, "reward_sum", latex=True)]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def report(summary, comparisons, provenance, risk_method, output):
    lines = ["Executed fusion-trigger comparison", "",
        f"Analyzed {provenance['individual_runs']} individual runs. No experiments were rerun. The candidate method is `{risk_method}`.", "",
        "This new diagnostic uses factual context and reward channels: a two-source contextual-bandit adaptation, not the full three-source queue service. The common ceiling is a limit, not evidence that methods spent equally. Costs and gradient steps describe actually executed updates. Positive accuracy/reward differences favor Risk RF; positive cost/job differences mean that it spent more. No superiority or equivalence is assumed.", "",
        "| Backbone | Scenario | Method | n | Post accuracy % | Tail accuracy % | Actual steps | Ceiling | Deployments | Reward |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        if row["method"] in {risk_method, "observed_loss", "periodic", "no_rt"}:
            cells = [row["backbone"], row["scenario"], row["method"], str(row["n"]),
                     pm(row, "post_boundary_accuracy", 100), pm(row, "last_512_accuracy", 100),
                     pm(row, "actual_gradient_updates", digits=1), ceiling_text(row),
                     pm(row, "deployments", digits=1), pm(row, "reward_sum")]
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Primary descriptive comparison: Risk RF minus observed loss.", "",
                  "| Backbone | Scenario | Metric | Seed pairs | Difference (pp) | Paired-bootstrap 95% interval (pp) | Two-sided p | Holm p if requested |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in comparisons:
        if row["control"] == "observed_loss" and row["metric"] in {"post_boundary_accuracy", "last_512_accuracy"}:
            adjusted = f"{row['holm_adjusted_p']:.5f}" if "holm_adjusted_p" in row else "NA"
            lines.append(f"| {row['backbone']} | {row['scenario']} | {row['metric']} | {row['n_pairs']} | "
                         f"{100*row['mean_difference']:+.3f} | [{100*row['ci95_low']:+.3f}, {100*row['ci95_high']:+.3f}] | "
                         f"{row['two_sided_p']:.5f} | {adjusted} |")
    lines.extend(["", "Every method appears in summary.csv and trigger_comparison_full.tex; trigger_comparison.tex may show explicitly selected scenarios. paired.csv/json retains every available comparison and metric, including null or adverse effects.", "",
        "Statistics: 10,000 seed-pair bootstrap resamples with stable fixed random seeds; two-sided sign-flip test of the paired mean, exact for at most 20 nonzero differences. Larger samples use 100,000 sign flips with the plus-one Monte Carlo correction. All-zero differences return p=1. These are not signed-rank tests. Unadjusted comparisons are exploratory. Optional Holm adjustment applies only to the explicitly selected backbone/clean-post family.", "",
        "With five nonzero paired differences the smallest exact two-sided p is 0.0625. A confidence interval or p-value does not establish causation beyond this controlled protocol. Failure to reject does not establish equivalence. Actual expenditure must accompany an accuracy advantage before interpreting efficiency.", "",
        "Missing outcomes are not imputed; metric-specific n and omitted/unpaired seeds are exported. The generator's known boundary is for offline post-boundary scoring, not a trigger input. Source corruption, factual feedback availability, and delay remain responsibilities of the separately executed experiment.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def self_test():
    assert sign_flip_mean([0, 0, 0], 1)[0] == 1.0
    assert sign_flip_mean([1, 2, 3, 4, 5], 1)[0] == 0.0625
    assert sign_flip_mean([-1, -2, -3, -4, -5], 1)[0] == 0.0625
    assert sign_flip_mean([1, -1], 1)[0] == 1.0
    a = paired_statistics([0, 0, 0, 0, 0], key="test", base_seed=7)
    assert a["ci95_low"] == a["ci95_high"] == 0 and a["two_sided_p"] == 1
    a = paired_statistics([1, 2, 3, 4, 5], key="test", base_seed=7)
    assert a == paired_statistics([1, 2, 3, 4, 5], key="test", base_seed=7)
    family = [{"control": "observed_loss", "scenario": "clean", "metric": "post_boundary_accuracy", "backbone": b,
               "two_sided_p": p} for b, p in [("dqn", 0.02), ("ppo", 0.03)]]
    assert holm_clean_post(family, "clean", ["dqn", "ppo"]) == 2
    assert family[0]["holm_adjusted_p"] == family[1]["holm_adjusted_p"] == 0.04
    assert budget_ceiling({}, {"maximum_jobs": 12, "job_gradient_steps": 32}) == 384
    assert latex_escape("x_1 & 20%") == r"x\_1 \& 20\%"
    print("Analysis self-tests passed (constructed inputs only; no experiment outcomes generated).")


def structured_analysis(summary, comparisons, risk_method, base_seed, family_n):
    groups = []
    for row in summary:
        result = {key: row[key] for key in ("backbone", "scenario", "method", "n", "seeds")}
        for metric in (*METRICS, "budget_ceiling_updates", "job_gradient_steps", "deployment_delay_slots"):
            result[metric] = {field: row.get(metric + "_" + field)
                              for field in ("n", "mean", "sd", "minimum", "maximum")}
        groups.append(result)
    paired = []
    for row in comparisons:
        paired.append({**row, "method_minus_reference": row["risk_method"] + " - " + row["control"],
                       "difference_mean": row["mean_difference"], "ci95": [row["ci95_low"], row["ci95_high"]],
                       "pvalue": row["two_sided_p"]})
    return {"groups": groups, "paired_comparisons": paired, "metadata": {
        "risk_method": risk_method, "analysis_seed": base_seed, "bootstrap_draws": 10000,
        "test": "two_sided_sign_flip_of_paired_mean", "holm_family_size": family_n,
        "accuracy_units": "fraction; multiply differences by100 for percentage points",
        "common_ceiling_is_not_matched_actual_spending": True,
        "scope": "controlled_synthetic_two_source_contextual_bandit_adaptation",
        "default_superiority_or_necessity_claim": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE.parent / "results/fusion_trigger")
    parser.add_argument("--output", type=Path, help="Default: <results>/analysis; raw inputs are never edited")
    parser.add_argument("--risk-method", help="Full candidate method name; otherwise infer an unambiguous known name")
    parser.add_argument("--seed", type=int, default=20261001, help="Fixed analysis RNG seed")
    parser.add_argument("--holm-clean-post", action="store_true")
    parser.add_argument("--clean-scenario", default="clean")
    parser.add_argument("--holm-backbones", default="dqn,ppo")
    parser.add_argument("--table-scenarios", help="Comma-separated scenarios for the main insertable fragment; full supplemental fragment always retains all")
    parser.add_argument("--label-prefix", default="fusion-trigger", help="LaTeX label suffix after tab:, e.g. fusion-trigger-budget12")
    parser.add_argument("--show-budget-caps", action="store_true", help="Read-only query of declared caps in the individual results, then exit")
    parser.add_argument("--expected-budget-ceiling", type=float, help="Validate the declared cap; never fills or changes observed expenditure")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    rows, provenance = load_results(args.results)
    caps = sorted({row["budget_ceiling_updates"] for row in rows if row["budget_ceiling_updates"] is not None})
    missing_caps = sum(row["budget_ceiling_updates"] is None for row in rows)
    if args.expected_budget_ceiling is not None:
        if missing_caps or any(abs(cap - args.expected_budget_ceiling) > 1e-9 for cap in caps):
            raise ValueError(f"Declared budget ceiling differs from expected {args.expected_budget_ceiling:g}, or is missing: {caps}, missing={missing_caps}")
    if args.show_budget_caps:
        print(json.dumps({"individual_runs": len(rows), "declared_gradient_step_ceilings": caps,
                          "runs_with_unknown_ceiling": missing_caps}, indent=2))
        return
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", args.label_prefix):
        raise ValueError("--label-prefix must be a plain identifier without LaTeX commands")
    methods = {r["method"] for r in rows}
    candidates = [method for method in RISK_NAMES if method in methods]
    risk_method = args.risk_method
    if risk_method is None:
        if len(candidates) != 1:
            raise ValueError("Specify --risk-method explicitly; known full RiskRF candidate names found: " + repr(candidates))
        risk_method = candidates[0]
    if risk_method not in methods:
        raise ValueError("Candidate method absent from the supplied individual runs")
    source = args.results.resolve()
    output = (args.output or ((source if source.is_dir() else source.parent) / "analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_paths = {Path(item["path"]).resolve() for item in provenance["source_files"]}
    output_names = ["summary.csv", "summary.json", "paired.csv", "paired.json", "analysis.json", "report.md", "trigger_comparison.tex", "trigger_comparison_full.tex", "analysis_manifest.json"]
    if any((output / name).resolve() in input_paths for name in output_names):
        raise ValueError("Output would overwrite an input file; choose another --output")
    summary = summarize(rows)
    comparisons = paired_comparisons(rows, risk_method, args.seed)
    family_n = holm_clean_post(comparisons, args.clean_scenario, args.holm_backbones.split(",")) if args.holm_clean_post else 0
    write_csv(output / "summary.csv", summary)
    write_json(output / "summary.json", summary)
    write_csv(output / "paired.csv", comparisons)
    write_json(output / "paired.json", comparisons)
    write_json(output / "analysis.json", structured_analysis(summary, comparisons, risk_method, args.seed, family_n))
    if args.table_scenarios:
        table_scenarios = set(args.table_scenarios.split(","))
        unknown = table_scenarios - {r["scenario"] for r in summary}
        if unknown:
            raise ValueError("Selected table scenarios absent: " + repr(sorted(unknown)))
        displayed = [r for r in summary if r["scenario"] in table_scenarios]
    else:
        displayed = summary
    latex_table(displayed, output / "trigger_comparison.tex", selected=True, label_prefix=args.label_prefix)
    latex_table(summary, output / "trigger_comparison_full.tex", label_prefix=args.label_prefix)
    report(summary, comparisons, provenance, risk_method, output)
    write_json(output / "analysis_manifest.json", {"risk_method": risk_method, "analysis_seed": args.seed,
        "bootstrap_draws": 10000, "test": "two_sided_sign_flip_of_paired_mean", "holm_family_size": family_n,
        "holm_clean_scenario": args.clean_scenario if family_n else None,
        "group_count": len(summary), "paired_metric_count": len(comparisons),
        "main_table_scenarios": sorted({r["scenario"] for r in displayed}),
        "declared_gradient_step_ceilings": caps, "runs_with_unknown_ceiling": missing_caps,
        "latex_label_prefix": args.label_prefix,
        "source_provenance": provenance, "analysis_code_sha256": digest(__file__),
        "experiment_rerun": False, "default_superiority_claim": False,
        "outputs": {name: digest(output / name) for name in output_names if name != "analysis_manifest.json"}})
    print(f"Analyzed {len(rows)} runs in {len(summary)} groups; exported {len(comparisons)} paired metric comparisons to {output}")


if __name__ == "__main__":
    main()
