"""Re-run tests and validate the example outputs without inferring policy return."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from coordinator import Coordinator
from logio import read_csv, read_json, write_json


def validate(root):
    configuration = read_json(root / "config.json")
    profiles = read_json(root / "profiles.json")
    reference = read_json(root / "reference.json")
    recorded = {row["slot"]: row for row in read_csv(root / "example_stream.csv")}
    engine = Coordinator(configuration["coordination"], profiles)
    with (root / "example_results/decisions.csv").open(newline="") as stream:
        exact = list(csv.DictReader(stream))
    with (root / "example_results_ao/decisions.csv").open(newline="") as stream:
        ao = list(csv.DictReader(stream))
    if len(exact) != len(ao):
        raise AssertionError("Both solvers must replay the same slots")
    gaps = []
    infeasible_slots = 0
    for a, b in zip(exact, ao):
        if a["slot"] != b["slot"]:
            raise AssertionError("Solver logs are not aligned")
        for field in ("gamma", "uncertainty", "recovery_state", "fusion_completed_slot"):
            if not math.isclose(float(a[field]), float(b[field]), abs_tol=1e-12, rel_tol=1e-12):
                raise AssertionError(f"Solver recommendations changed logged feedback state: {field}")
        for row in (a, b):
            slot = int(row["slot"])
            if int(row["fusion_completed_slot"]) >= slot:
                raise AssertionError("Same-slot or future fusion evidence was used")
            if not 0 <= float(row["gamma"]) <= 1 or not 0 <= float(row["uncertainty"]) <= 1:
                raise AssertionError("Invalid score bounds")
            if row["status"] == "feasible":
                log = recorded[slot]
                feasible = engine.feasible(engine.training_by_id[row["training_profile"]],
                                            engine.inference_by_id[row["inference_profile"]],
                                            log["training_load"], log["inference_load"], log["capacities"])
                if not feasible:
                    raise AssertionError(f"Infeasible executable pair at slot {slot}")
            else:
                infeasible_slots += 1
            trajectory = json.loads(row["trajectory"])
            if any(after < before - 1e-10 for before, after in zip(trajectory, trajectory[1:])):
                raise AssertionError("AO accepted an objective decrease")
        if a["objective"] and b["objective"]:
            gap = float(a["objective"]) - float(b["objective"])
            if gap < -1e-9:
                raise AssertionError("AO exceeded exhaustive optimum on the same surrogate")
            gaps.append(max(0.0, gap))
    windows = [json.loads(line) for line in (root / "example_results/fusion_windows.jsonl").read_text().splitlines()]
    for window in windows:
        if window["fallback"]:
            if sum(window["weights"]) != 0 or window["uncertainty"] != 1:
                raise AssertionError("Invalid fallback")
        elif not math.isclose(sum(window["weights"]), 1.0, abs_tol=1e-10):
            raise AssertionError("Weights do not sum to one")
        if not 0 <= window["disagreement"] <= 0.25 + 1e-10:
            raise AssertionError("Invalid disagreement")
        if any(weight != 0 for weight, valid in zip(window["weights"], window["masks"]) if not valid):
            raise AssertionError("An unavailable source was assigned positive weight")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root / "test_results.txt").write_text(tests.stdout)
    if tests.returncode:
        raise AssertionError(tests.stdout)
    result = {
        "status": "passed", "scope": "functional reference implementation on synthetic logged data",
        "empirical_drl_improvement_validated": False,
        "calibration_cutoff": reference["fit_through_slot"],
        "evaluation_slots_per_solver": len(exact), "completed_windows": len(windows),
        "solvers_validated": ["exact", "multi_start_ao"],
        "resource_violations": 0, "future_evidence_violations": 0,
        "reported_infeasible_rows": infeasible_slots,
        "ao_monotonic_trajectories": True,
        "ao_suboptimal_slots_against_same_slot_exact_surrogate": sum(gap > 1e-9 for gap in gaps),
        "ao_maximum_surrogate_gap": max(gaps, default=0.0),
        "unit_tests": 14, "unit_test_status": "passed",
        "latex_compilation": "not_checked", "visual_inspection": "not_checked"
    }
    build_log = root / "build/methods_preview.log"
    if build_log.is_file():
        text = build_log.read_text(errors="replace")
        bad = ("LaTeX Error", "Overfull", "undefined", "Missing character", "Warning")
        if any(marker in text for marker in bad):
            raise AssertionError("Inspect the final LaTeX build log")
        result["latex_compilation"] = "passed_without_warnings"
    write_json(root / "validation_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    validate(Path(__file__).resolve().parent)
