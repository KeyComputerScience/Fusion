"""Generate a synthetic log and run the core, solver audit, and diagnostic suite."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from calibrate_reference import fit_reference
from generate_demo import generate
from logio import read_csv, read_json, write_csv, write_json
from pipeline import run_replay
from run_experiments import diagnose, run_suite


def save_replay(directory, decisions, windows, summary):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    write_csv(directory / "decisions.csv",decisions)
    write_json(directory / "summary.json",summary)
    (directory / "fusion_windows.jsonl").write_text(
        "".join(json.dumps(w,ensure_ascii=False,allow_nan=False)+"\n" for w in windows),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-slots",type=int,default=300)
    parser.add_argument("--seed",type=int,default=20260930)
    parser.add_argument("--with-visuals",action="store_true")
    parser.add_argument("--font-dir",help="Licensed Times New Roman folder, only for visual builds")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config, profiles = read_json(root / "config.json"),read_json(root / "profiles.json")
    write_csv(root / "example_stream.csv",generate(config,profiles,args.calibration_slots,args.seed))
    rows = read_csv(root / "example_stream.csv")
    reference = fit_reference(rows,config,args.calibration_slots)
    write_json(root / "reference.json",reference)
    audit_by_solver, choices_by_solver = {}, {}
    for solver in ("exact","ao"):
        choices, windows, summary = run_replay(rows,config,profiles,reference,solver)
        destination = root / ("example_results" if solver == "exact" else "example_results_ao")
        save_replay(destination,choices,windows,summary)
        audit_by_solver[solver] = diagnose(rows,config,profiles,reference,choices,windows)
        choices_by_solver[solver] = choices
        print(f"{solver}: slots={len(choices)}, windows={len(windows)}",flush=True)
    gaps, agreement = [], 0
    for exact, ao in zip(choices_by_solver["exact"],choices_by_solver["ao"]):
        if exact["status"] == ao["status"] == "feasible":
            gaps.append(exact["objective"]-ao["objective"])
            agreement += int((exact["training_profile"],exact["inference_profile"]) ==
                             (ao["training_profile"],ao["inference_profile"]))
    write_json(root / "solver_audit.json",{
        "scope":"current proxy on the same fixed synthetic logged feedback",
        "diagnostics":audit_by_solver, "compared_feasible_slots":len(gaps),
        "pair_agreements":agreement, "maximum_exact_minus_ao_proxy_gap":max(gaps,default=0),
        "minimum_exact_minus_ao_proxy_gap":min(gaps,default=0),
        "ao_global_optimality_claimed":False,
        "warning":"Agreement on this stream is not a guarantee. A separate test exhibits a positive AO global gap."})
    run_suite(rows,config,profiles,reference,read_json(root / "ablation_plan.json"),
              read_json(root / "stress_plan.json"),root / "experiment_results")
    if args.with_visuals:
        font_option = ["--font-dir",args.font_dir] if args.font_dir else []
        subprocess.run([sys.executable,"plot_demo.py",*font_option],cwd=root,check=True)
        subprocess.run([sys.executable,"build_preview.py",*font_option],cwd=root,check=True)
    subprocess.run([sys.executable,"validate_package.py"],cwd=root,check=True)
    print("Done. Synthetic functional diagnostics only; see validation_report.json.")


if __name__ == "__main__":
    main()
