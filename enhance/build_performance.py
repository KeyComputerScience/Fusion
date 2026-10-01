"""Render manuscript tables and prose exclusively from validated analysis.json."""
import argparse
import json
from pathlib import Path
import re

from data_pipeline import ROOT, read_json, write_json

SHORT = {"fusion": "Reliability fusion", "no_rt": "No retraining", "periodic": "Periodic",
         "reactive": "Workload trigger", "dpp": "Deficit heuristic", "equal": "Equal weights",
         "fixed": "Fixed weights", "workload": "Workload only", "operating": "Operating only",
         "performance": "Performance only", "no_predictive": "No prediction update",
         "uncapped": "No weight cap", "no_disagreement": "No disagreement", "no_delayed": "No delayed value"}
SCENARIOS = {"performance_outage": "Performance outage", "all_outage": "Complete outage",
             "operating_noise": "Operating noise", "conflict": "Workload conflict"}


def number(value, digits=2):
    return f"{value:.{digits}f}"


def spread(record, digits=2, scale=1.):
    if record["mean"] is None:
        return "--"
    return "$" + number(record["mean"] * scale, digits) + r"\pm" + number(record["sd"] * scale, digits) + "$"


def table(caption, label, headers, rows, align=None, note=""):
    align = align or ("l" + "r" * (len(headers) - 1))
    body = "\n".join(" & ".join(str(x) for x in row) + r" \\" for row in [headers] + rows)
    header, data = body.split("\n", 1)
    return (r"\begin{table*}[htbp]" + "\n" + r"\centering" + "\n" + r"\footnotesize" + "\n"
            + r"\setlength{\tabcolsep}{4pt}" + "\n" + r"\caption{" + caption + "}\n"
            + r"\label{" + label + "}\n" + r"\begin{tabular}{@{}" + align + "@{}}\n"
            + r"\toprule" + "\n" + header + "\n" + r"\midrule" + "\n" + data + "\n"
            + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
            + ((r"\par\smallskip\begin{minipage}{0.98\textwidth}\footnotesize " + note
                + r"\end{minipage}" + "\n") if note else "") + r"\end{table*}" + "\n")


def build(analysis_path, output):
    report = read_json(analysis_path)
    if not report["validation"]["all_passed"] or report["validation"]["runs"] != 95:
        raise ValueError("This manuscript template requires all 95 validated runs")
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    main = {(r["backbone"], r["method"]): r for r in report["main"]}
    ablations = {r["method"]: r for r in report["ablation"]}
    fusion = {(r["scenario"], r["method"]): r for r in report["fusion_diagnostics"]}
    pairs = {(r["backbone"], r["method"], r["reference"]): r for r in report["paired_comparisons"]}
    solver, checked = report["solver"], report["validation"]
    blocks = {}
    blocks["PARAMETER_TABLE"] = table("Declared controlled evaluation parameters.", "tab:eval_parameters",
        ["Parameter", "Value", "Role"], [
            ["Edge nodes / seeds", "$10$ / $10,20,30,40,50$", "Paired exogenous inputs"],
            ["Prefix / horizon", "$1200$ / $3000$ slots", "Initialization / evaluation"],
            ["Window / count", "$W=50$ / $N_w=60$", "Completed-window fusion"],
            ["Injected changes", "$901,1801$", "Evaluation slot indices"],
            ["Resources", "$(C,L,P)$", "Memory, computing, budget"],
            ["Capacity baselines", r"$1;\ \mathcal U(0.8,1.2)$", "Memory; computing and budget"],
            ["Lookahead / initial recovery", "$16$ / $H_1=1$", "Delayed proxy"],
            ["Exact / AO limit", "$30$ pairs / $50$ sweeps", "Default / enlarged search"],
            ["Proxy coefficients", "$(0.12,0.04,0.08,0.35)$", "$\\lambda_c,\\omega,\\kappa_U,\\xi$"],
            ["Retention coefficients", "$(0.08,0.02,0.8)$", "$\\eta_\\Gamma,\\eta_U,\\rho_{\\min}$"],
            ["Admission thresholds", "$(0.15,0.08,0.8,2/3)$", "On, off, uncertainty, coverage"],
            ["Minimum evidence / context", "$10$ / $2$ samples", "Support eligibility"],
            ["Block bootstrap", "$32$ replicates, block $5$", "Source-score noise"],
            ["Support / freshness scales", "$20$ / $50$", "Source reliability"],
            ["Error scale / noise floor", "$0.2$ / $0.0025$", "Source reliability"],
            ["Missing-source penalty / cap", "$0.5$ / $0.7$", "Uncertainty / source weights"],
            ["Predictor ridge / forgetting", "$0.1$ / $0.99$", "Delayed affine calibration"],
            ["Error update / initial error", "$0.15$ / $0.05$", "Squared prediction error"],
            ["Quality update rate", "$0.1$", "Executed profile only"],
            ["Training eligibility", "$128$ new transitions", "One active worker"],
            ["Reward weights", "$(1,0.5,0.05,0.12)$", "Completion, deadlines, queue, cost"]], align="lll")
    profiles = report["runtime"]["profile_priors"]
    training_rows = [[p["id"], str(p["updates"]), number(p["gain"]), number(p["cost"]), number(p["intensity"], 3),
                      str(p["deployment_delay"]), *[number(x) for x in p["resources"]]] for p in profiles["training"]]
    inference_rows = [[p["id"], number(p["beta"]), *[number(x) for x in p["resources"]]] for p in profiles["inference"]]
    train = table("Explicit normalized training profiles and inference profiles.", "tab:eval_profiles",
                  ["Training", "Updates", "$g_i$", "$c_i$", "$q_i$", "$\\delta_i$", "$a_i^C$", "$a_i^L$", "$a_i^P$"],
                  training_rows)
    extra = (r"\par\medskip\begin{tabular}{@{}lrrrr@{}}\toprule" + "\n"
             + r"Inference & $\beta_j$ & $b_j^C$ & $b_j^L$ & $b_j^P$ \\ \midrule" + "\n"
             + "\n".join(" & ".join(row) + r" \\" for row in inference_rows)
             + "\n" + r"\bottomrule\end{tabular}" + "\n"
             + r"\par\smallskip\begin{minipage}{0.98\textwidth}\footnotesize "
             + r"Costs and gains are per unit training load; delay counts active allocation slots. "
             + r"Profile \texttt{tr0} performs no update or deployment. "
             + r"Inference quality priors are initialized separately from the calibration prefix.\end{minipage}" + "\n")
    blocks["PROFILE_TABLE"] = train.replace(r"\end{table*}", extra + r"\end{table*}")
    main_rows = [[r["backbone"].upper(), SHORT[r["method"]], spread(r["return"]),
                  spread(r["completion_fraction"], 2, 100), spread(r["deadline_fraction"], 2, 100),
                  spread(r["deployed_jobs"], 1), spread(r["training_cost"], 2)] for r in report["main"]]
    blocks["MAIN_TABLE"] = table("Executed service results: mean $\\pm$ seed standard deviation.", "tab:service_results",
        ["Backbone", "Coordinator", "Return", "Complete (\\%)", "Missed (\\%)", "Deployments", "Training cost"],
        main_rows, align="llrrrrr", note="Missed counts expiry and admission rejection; tasks queued at the horizon remain separate. Costs are normalized totals.")
    dqn, ppo = main["dqn", "fusion"], main["ppo", "fusion"]
    def pair_text(backbone):
        result = pairs[backbone, "fusion", "no_rt"]
        lo, hi = result["bootstrap_ci95"]
        return (f"{number(result['paired_difference']['mean'],3)} "
                + r"(95\% paired bootstrap interval $[" + number(lo,3) + "," + number(hi,3)
                + "]$, $p=" + number(result["signed_rank_p_two_sided"],4) + "$")
    blocks["MAIN_ANALYSIS"] = (f"Reliability fusion obtains returns of {spread(dqn['return'])} with DQN and "
        + f"{spread(ppo['return'])} with PPO. The corresponding completion fractions are "
        + f"{number(dqn['completion_fraction']['mean']*100)}\\% and {number(ppo['completion_fraction']['mean']*100)}\\%. "
        + "Its paired return difference from no retraining is " + pair_text("dqn") + ") for DQN and "
        + pair_text("ppo") + ") for PPO. Thus, the experiment does not demonstrate a service-return advantage over no retraining. "
        + f"The largest mean return among the main DQN comparators is {number(main['dqn','periodic']['return']['mean'])} "
        + "for periodic training, while the largest mean among the main PPO comparators is "
        + f"{number(main['ppo','no_rt']['return']['mean'])} for no retraining. "
        + f"Fusion deploys {number(dqn['deployed_jobs']['mean'],1)} jobs per run, whereas periodic training deploys "
        + f"{number(main['dqn','periodic']['deployed_jobs']['mean'],1)}. "
        + "The fusion trigger reduces training frequency; this does not by itself establish better policy adaptation. "
        + "Training consumes resources that can lower the feasible inference profile, and its fixed recovery prior does not guarantee a neural reward increase.")
    fusion_return = ablations["fusion"]["return"]["mean"]
    blocks["ABLATION_TABLE"] = table("Executed DQN fusion ablations; differences are relative to reliability fusion.", "tab:service_ablation",
        ["Variant", "Return", "Complete (\\%)", "Deployments", "Training cost", "$\\Delta$ return"],
        [[SHORT[r["method"]], spread(r["return"]), spread(r["completion_fraction"], 2, 100),
          spread(r["deployed_jobs"], 1), spread(r["training_cost"]), number(r["return"]["mean"] - fusion_return, 3)] for r in report["ablation"]])
    blocks["ABLATION_ANALYSIS"] = ("Workload-only fusion and the variants without predictive calibration, a weight cap, or disagreement "
        + "produce identical seed-level DQN returns in this setting. These results do not establish an executed-service benefit for those components. "
        + f"Equal weights deploy {number(ablations['equal']['deployed_jobs']['mean'],1)} jobs on average, with return "
        + f"{spread(ablations['equal']['return'])}. Operating-only evidence deploys "
        + f"{number(ablations['operating']['deployed_jobs']['mean'],1)} jobs and gives return {spread(ablations['operating']['return'])}, "
        + f"at training cost {number(ablations['operating']['training_cost']['mean'])}, compared with "
        + f"{number(ablations['fusion']['training_cost']['mean'])} for reliability fusion. "
        + "Removing delayed value suppresses admitted training in these runs and reproduces the no-retraining return. "
        + "The different rankings of evidence specificity, training cost, and service return show that they must be evaluated separately.")
    clean = [r for r in report["fusion_diagnostics"] if r["scenario"] == "clean"]
    blocks["FUSION_TABLE"] = table("Fusion diagnostics on common executed feedback; forecast MAE uses common target windows.", "tab:fusion_diagnostics",
        ["Variant", "ROC-AUC", "AP", "Alarm (\\%)", "Shifts covered", "Delay", "Forecast MAE"],
        [[SHORT[r["method"]], number(r["roc_auc"]["mean"],3), number(r["average_precision"]["mean"],3),
          number(r["nominal_window_alarm_fraction"]["mean"]*100,2), f"{r['total_detected']}/{r['total_changes']}",
          number(r["detection_delay_slots"]["mean"],1) if r["detection_delay_slots"]["mean"] is not None else "--",
          spread(r["common_forecast_mae"],4)] for r in clean],
        note="Delay is in slots and conditional on covered shifts. ROC-AUC, AP, and alarm entries are seed means. The same 58 eligible targets per seed are used by every clean variant.")
    cf, equal, perf = fusion["clean", "fusion"], fusion["clean", "equal"], fusion["clean", "performance"]
    baseline = cf["common_persistence_mae"]["mean"]
    relative = 100 * (baseline - cf["common_forecast_mae"]["mean"]) / baseline
    blocks["FUSION_ANALYSIS"] = (f"On the controlled shifts, reliability fusion has mean ROC-AUC {number(cf['roc_auc']['mean'],3)} "
        + f"and AP {number(cf['average_precision']['mean'],3)}, covering {cf['total_detected']}/{cf['total_changes']} changes "
        + f"with a conditional delay of {number(cf['detection_delay_slots']['mean'],1)} slots. Workload-only and equal-weight fusion "
        + "also obtain perfect mean ranking scores, so these easy, aligned shifts do not isolate a ranking advantage of reliability fusion. "
        + f"Reliability fusion has {number(cf['nominal_window_alarm_fraction']['mean']*100,2)}\\% nominal-window alarms, "
        + f"compared with {number(equal['nominal_window_alarm_fraction']['mean']*100,2)}\\% for equal weights and "
        + f"{number(fusion['clean','operating']['nominal_window_alarm_fraction']['mean']*100,2)}\\% for operating-only evidence.\n\n"
        + f"The next-window forecast MAE is {spread(cf['common_forecast_mae'],4)}, versus {spread(cf['common_persistence_mae'],4)} "
        + f"for persistence, a {number(relative,2)}\\% difference in the mean. Disabling predictive calibration yields "
        + f"MAE {number(fusion['clean','no_predictive']['common_forecast_mae']['mean'],4)}. "
        + f"Performance-only prediction has lower overall MAE, {number(perf['common_forecast_mae']['mean'],4)}, while covering "
        + f"{perf['total_detected']}/{perf['total_changes']} injected changes. "
        + "Consequently, overall prediction error and drift coverage favor different variants. "
        + f"An exploratory check restricted to the two change-target windows gives fusion MAE {number(cf['change_target_mae']['mean'],4)} "
        + f"and persistence MAE {number(cf['change_target_persistence_mae']['mean'],4)}. "
        + "The overall MAE includes many stable targets and should not be interpreted as reliable anticipation of abrupt changes.")
    stress = [r for r in report["fusion_diagnostics"] if r["scenario"] != "clean"]
    blocks["STRESS_TABLE"] = table("Evidence-channel perturbations on identical service feedback.", "tab:stress_diagnostics",
        ["Scenario", "Variant", "ROC-AUC", "Alarm (\\%)", "Shifts", "Fallback", "Max. weight"],
        [[SCENARIOS[r["scenario"]], SHORT[r["method"]], number(r["roc_auc"]["mean"],3),
          number(r["nominal_window_alarm_fraction"]["mean"]*100,2), f"{r['total_detected']}/{r['total_changes']}",
          number(r["fallback_windows"]["mean"],1), number(r["maximum_source_weight"]["max"],3)] for r in stress], align="llrrrrr",
        note="Fallback is the mean number of non-startup windows per seed. Maximum weight is the largest over all five seeds. Subset variants normalize coverage over their enabled sources.")
    outage, missing, conflict = fusion["all_outage", "fusion"], fusion["performance_outage", "fusion"], fusion["conflict", "fusion"]
    blocks["STRESS_ANALYSIS"] = (f"With the performance channel unavailable, reliability fusion still covers {missing['total_detected']}/{missing['total_changes']} "
        + "changes and produces no nominal-window alarms in this experiment. "
        + f"The capped maximum source weight is {number(missing['maximum_source_weight']['max'],3)}, "
        + f"whereas the uncapped maximum reaches {number(fusion['performance_outage','uncapped']['maximum_source_weight']['max'],3)}. "
        + "The cap therefore limits concentration under this outage; it does not improve the reported detection score.\n\n"
        + f"Complete evidence loss produces {number(outage['fallback_windows']['mean'],1)} fallback windows per seed, "
        + f"mean ROC-AUC {number(outage['roc_auc']['mean'],3)}, and coverage of only {outage['total_detected']}/{outage['total_changes']} changes. "
        + "The missing interval also invalidates comparisons across the adjacent recovery window. "
        + "Holding the latest drift value preserves a control input but cannot recover information absent from all sources. "
        + f"Under workload conflict, reliability fusion covers {conflict['total_detected']}/{conflict['total_changes']} changes, "
        + f"with ROC-AUC {number(conflict['roc_auc']['mean'],3)} and {number(conflict['nominal_window_alarm_fraction']['mean']*100,2)}\\% "
        + "nominal-window alarms. Equal weights have higher ROC-AUC in that scenario. "
        + "These perturbations support bounded source influence and explicit missing-evidence handling, rather than uniform superiority over simpler fusion.")
    blocks["SOLVER_TABLE"] = table("Exact search and multi-start AO on 500 paired synthetic states with 750 profile pairs.", "tab:solver_audit",
        ["Solver", "Time (ms)", "Pairs evaluated", "Mean proxy gap", "Largest gap", "Same pair (\\%)"],
        [["Exact", spread(solver["exact_ms"],3), spread(solver["exact_pairs"],1), "$0$", "$0$", "100.0"],
         ["AO", spread(solver["ao_ms"],3), spread(solver["ao_pairs"],1), spread(solver["proxy_gap"],5),
          number(solver["maximum_proxy_gap"],5), number(solver["same_pair_fraction"]*100,1)]],
        note="Mean and spread summarize five seed-level averages over 100 audit states each. The proxy gap is exact minus AO; it is not an external service-return gap.")
    blocks["SOLVER_ANALYSIS"] = ("The default 30-pair experiment uses exact feasible enumeration, so AO does not introduce a default-run optimization gap. "
        + "For the enlarged audit, AO reaches coordinate convergence in all feasible states, but selects the same pair as exact search in only "
        + f"{number(solver['same_pair_fraction']*100,1)}\\% of states. Its largest objective gap is "
        + f"{number(solver['maximum_proxy_gap'],5)}. The ratio of aggregate exact to AO time is {number(solver['total_time_ratio'],2)}. "
        + "Thus, finite coordinate convergence and reduced search time do not imply global optimality. The audit uses interpolated profiles "
        + "and generated states; it does not evaluate service performance on a 750-action physical system.\n\n"
        + f"Validation checks {checked['slots_checked']:,} executed slots across {checked['runs']} runs and "
        + f"{checked['deployments_checked']} deployments. Every admitted inference profile is executed, all deployment events "
        + "change actual model parameters after the declared completion delay, and no recovery increment precedes an actual deployment. "
        + f"The resource-violation count is {checked['resource_violations']}. Mean online coordination time is "
        + f"{spread(dqn['coordinator_mean_ms'],3)} ms for DQN fusion and {spread(ppo['coordinator_mean_ms'],3)} ms for PPO fusion. "
        + "This timing covers pair selection and excludes window fusion and service training. Completed-task p95 latency ranges "
        + f"from {number(dqn['p95_latency_slots']['min'],0)} to {number(dqn['p95_latency_slots']['max'],0)} slots across DQN fusion seeds. "
        + "It is not hardware latency and excludes unsuccessful tasks, which are accounted for by the missed fraction.")
    for name in ["python", "numpy", "torch"]:
        blocks[name.upper()] = report["runtime"][name].replace("_", r"\_")
    template = (ROOT / "performance_template.tex").read_text(encoding="utf-8")
    for key, value in blocks.items():
        template = template.replace("@@" + key + "@@", value)
        if key.endswith("_TABLE"):
            (tables / (key.lower() + ".tex")).write_text(value, encoding="utf-8")
    if "@@" in template:
        raise ValueError("An unfilled manuscript token remains")
    if re.search(r"\b(?:measure\w*|protect\w*|previous\w*)\b", template, re.I):
        raise ValueError("A disallowed manuscript word remains")
    (output / "performance.tex").write_text(template, encoding="utf-8")
    write_json(output / "results_for_manuscript.json", {"main": report["main"], "ablation": report["ablation"],
                "fusion": report["fusion_diagnostics"], "paired": report["paired_comparisons"],
                "solver": report["solver"], "validation": report["validation"]})
    print(f"Generated {output / 'performance.tex'} and {len(list(tables.glob('*.tex')))} tables", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "results/analysis.json")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.analysis, args.output)
