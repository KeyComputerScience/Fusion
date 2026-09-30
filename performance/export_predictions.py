"""Export deterministic forecasts and decisions from the supplied core engine."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))

from logio import read_csv, read_json, write_csv, write_json
from pipeline import FusionPipeline
from proxy import RecoveryForecast


def inspect_decision(pipeline, row, decision):
    config = pipeline.config["coordination"]
    engine = pipeline.coordinator
    snapshot = pipeline.fusion.snapshot
    remaining = pipeline.config["evaluation_horizon"] - (row["slot"] - pipeline.origin) - 1
    horizon = min(config["lookahead"], remaining)
    attenuation = math.exp(-config["xi"] * snapshot.gamma)
    forecast = RecoveryForecast.construct(engine.H, decision.rho, horizon, attenuation)
    training, inference, pairs = [], [], []
    for profile in engine.training:
        coefficient = (forecast.coefficient(profile["deployment_delay"], config["proxy_mode"])
                       if config["enable_delayed_value"] else 0.0)
        gain = row["training_load"] * profile["gain"]
        intensity = row["training_load"] * profile["intensity"]
        direct_cost = row["training_load"] * profile["cost"]
        value = (coefficient * gain - config["lambda_cost"] * direct_cost
                 - config["omega_intensity"] * intensity ** 2
                 - config["kappa_uncertainty"] * snapshot.uncertainty * intensity)
        factors = dict(forecast.factors(profile["deployment_delay"]))
        admitted = decision.training_allowed or profile["id"] == engine.base
        training.append({
            "profile": profile["id"], "admitted": admitted,
            "delay": profile["deployment_delay"], "gain": gain,
            "cost": direct_cost, "intensity": intensity,
            "V": coefficient, "F": value,
            "exact_proxy_increment": forecast.exact_increment(gain, profile["deployment_delay"]),
            "tangent_credit": forecast.coefficient(profile["deployment_delay"]) * gain,
            "conditional_recovery": [baseline + factors.get(h, 0.0) * gain
                                     for h, baseline in enumerate(forecast.baseline)]})
    for profile in engine.inference:
        quality = engine.service_quality(profile)
        inference.append({"profile": profile["id"], "quality_estimate": quality,
                          "W": attenuation * quality * (-math.expm1(-engine.H))})
    for i, terms_i in zip(engine.training, training):
        for j, terms_j in zip(engine.inference, inference):
            feasible = engine.feasible(i, j, row["training_load"], row["inference_load"], row["capacities"])
            pairs.append({"training": i["id"], "inference": j["id"],
                          "admitted": terms_i["admitted"], "resource_feasible": feasible,
                          "objective": terms_i["F"] + terms_j["W"]})
    eligible = [p for p in pairs if p["admitted"] and p["resource_feasible"]]
    if decision.status == "feasible":
        chosen = next(p for p in eligible if (p["training"], p["inference"]) ==
                      (decision.training_profile, decision.inference_profile))
        if not math.isclose(chosen["objective"], decision.objective, abs_tol=1e-12):
            raise AssertionError("Cached values do not match the engine decision")
        if decision.solver == "exact" and not math.isclose(
                max(p["objective"] for p in eligible), decision.objective, abs_tol=1e-12):
            raise AssertionError("Exact selection does not match the feasible optimum")
    elif eligible:
        raise AssertionError("A feasible admitted pair exists for an infeasible decision")
    return {"slot": row["slot"], "fusion": snapshot.to_dict(),
            "inputs": {"training_load": row["training_load"], "inference_load": row["inference_load"],
                       "capacities": row["capacities"], "pending_gain_forecast": {},
                       "retention_forecast": [decision.rho] * horizon},
            "lookahead": horizon, "attenuation": attenuation,
            "baseline_recovery": list(forecast.baseline), "decision": decision.to_dict(),
            "training_terms": training, "inference_terms": inference, "pair_terms": pairs,
            "quality_components": copy.deepcopy(engine.quality),
            "predictor_coefficients": [list(p.coefficients()) for p in pipeline.fusion.predictors]}


def export(engine_root=ROOT / "engine", checkpoints=(401, 601, 651, 1151, 1251, 1500)):
    config = read_json(engine_root / "config.json")
    profiles = read_json(engine_root / "profiles.json")
    reference = read_json(engine_root / "reference.json")
    rows = read_csv(engine_root / "example_stream.csv")
    pipeline = FusionPipeline(config, profiles, reference, collect_history=True)
    evaluation = [row for row in rows if row["slot"] > reference["fit_through_slot"]]
    checkpoints = set(checkpoints)
    available_slots = {row["slot"] for row in evaluation}
    if not checkpoints or not checkpoints <= available_slots:
        raise ValueError("Checkpoint slots must belong to the supplied evaluation stream")
    cases, timeline = [], []
    for row in evaluation:
        decision = pipeline.decide(row["slot"], row["training_load"], row["inference_load"], row["capacities"])
        detail = inspect_decision(pipeline, row, decision) if row["slot"] in checkpoints else None
        snapshot = pipeline.fusion.snapshot
        record = {"slot": row["slot"], "fusion_end": snapshot.completed_slot,
                  "gamma": snapshot.gamma, "D": snapshot.disagreement, "U": snapshot.uncertainty,
                  "coverage": snapshot.coverage, "fallback": snapshot.fallback,
                  "training_allowed": decision.training_allowed, "H": decision.recovery_state,
                  "rho": decision.rho, "training_profile": decision.training_profile,
                  "inference_profile": decision.inference_profile, "objective": decision.objective}
        for source, name in enumerate(("workload", "operating", "performance")):
            record[f"d_{name}"] = snapshot.scores[source]
            record[f"alpha_{name}"] = snapshot.weights[source]
            record[f"next_target_{name}"] = snapshot.next_target_predictions[source]
        observed = pipeline.observe(row)
        record.update(observed)
        timeline.append(record)
        if detail is not None:
            detail["feedback"] = dict(observed, logged_training=row["logged_training"],
                                      logged_inference=row["logged_inference"])
            cases.append(detail)
    windows = {window["window_index"]: window for window in pipeline.window_results}
    for case in cases:
        snap = case["fusion"]
        target_window = windows.get(snap["window_index"] + 1)
        target = target_window["scores"][2] if target_window else None
        case["posthoc_prediction_check"] = {
            "target_window": snap["window_index"] + 1,
            "target_available_from_slot": target_window["available_from_slot"] if target_window else None,
            "target": target,
            "squared_errors": [None if p is None or target is None else (p - target) ** 2
                               for p in snap["next_target_predictions"]],
            "used_in_slot_decision": False}
    digest = hashlib.sha256((engine_root / "example_stream.csv").read_bytes()).hexdigest()
    profile_priors = copy.deepcopy(profiles)
    profile_priors["provenance"] = "ILLUSTRATIVE: profile values require calibration."
    payload = {"data_kind": "deterministic_synthetic_logged_replay",
               "interpretation": "Conditional forecast and recommendation data; not real service outcomes.",
               "seed": config["fusion"]["seed"], "input_sha256": digest,
               "calibration_cutoff": reference["fit_through_slot"], "evaluation_slots": len(timeline),
               "next_target": "performance degradation d_3 in the next completed window",
               "normalization_reference": reference, "configuration": config,
               "profile_priors": profile_priors, "checkpoints": cases,
               "window_outputs": pipeline.window_results}
    return payload, timeline


def write_case_tex(payload, path, slot):
    case = next(item for item in payload["checkpoints"] if item["slot"] == slot)
    snap, decision = case["fusion"], case["decision"]
    lines = [r"\subsection*{Reproducible Numerical Example}",
             "The following values are generated from a deterministic synthetic log.",
             "They illustrate conditional forecasts and feasible recommendations, not real service gains.",
             f"At slot ${slot}$, the fusion window ends at slot ${snap['completed_slot']}$;",
             f"$H_t={decision['recovery_state']:.6f}$, $\\rho_t={decision['rho']:.6f}$,",
             f"$\\Gamma={snap['gamma']:.6f}$, $D={snap['disagreement']:.6f}$,",
             f"$U={snap['uncertainty']:.6f}$, and $c={snap['coverage']:.6f}$.",
             f"The loads are $D_t^{{\\mathrm{{tr}}}}={case['inputs']['training_load']:.6f}$ and",
             f"$D_t^{{\\mathrm{{inf}}}}={case['inputs']['inference_load']:.6f}$, with lookahead $L_t={case['lookahead']}$.",
             r"\begin{center}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Training profile & $\delta_i$ & $G_{i,t}$ & $V_{i,t}$ & $F_{i,t}$ \\", r"\midrule"]
    lines.extend(f"{t['profile']} & {t['delay']} & {t['gain']:.6f} & {t['V']:.6f} & {t['F']:.6f} \\\\"
                 for t in case["training_terms"])
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}",
                  f"The exact solver selects $({decision['training_profile']},{decision['inference_profile']})$",
                  f"with proxy objective ${decision['objective']:.6f}$.",
                  "The selected inference term is "
                  + "$W_{j_t,t}=" + f"{next(j['W'] for j in case['inference_terms'] if j['profile'] == decision['inference_profile']):.6f}" + "$.",
                  "The next-window performance forecasts from the three sources are",
                  "$\\widehat\\psi_{s,k+1|k}=(" + ",".join(f"{p:.6f}" for p in snap["next_target_predictions"]) + ")$.",
                  "The JSON output retains full precision, all candidate values, resource tests, and recovery trajectories."])
    check = case["posthoc_prediction_check"]
    if check["target"] is not None:
        lines.extend([
            f"After the target window completes, $d_{{3,k+1}}={check['target']:.6f}$ becomes available",
            f"at slot ${check['target_available_from_slot']}$; it is not an input at slot ${slot}$.",
            "The three squared prediction errors are $("
            + ",".join(f"{error:.6f}" for error in check["squared_errors"]) + ")$.",
            "These values do not establish predictive accuracy; they expose the error for this example."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[401, 601, 651, 1151, 1251, 1500])
    args = parser.parse_args()
    payload, timeline = export(checkpoints=args.checkpoints)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "predictable_data.json", payload)
    write_csv(args.output / "slot_predictions.csv", timeline)
    training = [{"slot": c["slot"], **{k: v for k, v in t.items() if k != "conditional_recovery"}}
                for c in payload["checkpoints"] for t in c["training_terms"]]
    write_csv(args.output / "training_values.csv", training)
    write_csv(args.output / "inference_values.csv", [{"slot": c["slot"], **j}
              for c in payload["checkpoints"] for j in c["inference_terms"]])
    write_csv(args.output / "prediction_errors.csv", [
        {"slot": c["slot"], "source": name,
         "forecast": c["fusion"]["next_target_predictions"][s],
         "target": c["posthoc_prediction_check"]["target"],
         "squared_error": c["posthoc_prediction_check"]["squared_errors"][s]}
        for c in payload["checkpoints"]
        for s, name in enumerate(("workload", "operating", "performance"))])
    if 651 in args.checkpoints:
        write_case_tex(payload, args.output / "prediction_case.tex", 651)
    print(json.dumps({"slots": len(timeline), "checkpoints": sorted(args.checkpoints),
                      "kind": payload["data_kind"]}, indent=2))


if __name__ == "__main__":
    main()
