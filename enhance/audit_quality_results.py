"""Audit executed quality replay outputs, including a future-input boundary check."""
from __future__ import annotations
import json
from pathlib import Path
import hashlib
import math
import argparse

ROOT = Path(__file__).resolve().parents[1] / "results/quality"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def current_bundle_path(recorded):
    """Resolve provenance paths in a relocated, self-contained revision bundle."""
    parts = Path(recorded).parts
    if "revision" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("revision")
        candidate = ROOT.parents[1].joinpath(*parts[index + 1:])
        if candidate.is_file():
            return candidate
    source_name = Path(recorded).name
    candidates = [ROOT.parents[1] / "code" / source_name, ROOT.parents[1] / "code/core" / source_name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find recorded implementation in this bundle: {recorded}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Read existing results without overwriting audit.json")
    args = parser.parse_args()
    design = load(ROOT / "frozen_design.json")
    manifest = load(ROOT / "run_manifest.json")
    summary = load(ROOT / "seed_results.json")
    methods = [item["name"] for item in design["methods"]]
    counts = {"windows": 0, "simplex": 0, "masked_zero_weight": 0, "cap": 0,
              "all_missing_abstentions": 0, "future_boundary_comparisons": 0,
              "common_target_count_groups": 0, "ds_weight_not_applicable": 0,
              "source_hashes": 0}
    maximum_simplex_error = 0.0
    maximum_cap_violation = 0.0
    for seed in design["seeds"]:
        for method in methods:
            clean = load(ROOT / "windows" / f"clean_{method}_seed_{seed}.json")
            for scenario in design["scenarios"]:
                rows = load(ROOT / "windows" / f"{scenario}_{method}_seed_{seed}.json")
                assert len(rows) == design["evaluation_slots"] // design["window_size"]
                for before, row in zip(clean, rows):
                    counts["windows"] += 1
                    assert row["available_from_slot"] == row["completed_slot"] + 1
                    weights, masks = row.get("weights"), row["masks"]
                    if row.get("nonlinear_combination", False):
                        assert weights is None
                        counts["ds_weight_not_applicable"] += 1
                    elif not row["fallback"]:
                        error = abs(sum(weights) - 1.0)
                        assert error < 1e-8 and min(weights) >= -1e-10
                        maximum_simplex_error = max(maximum_simplex_error, error)
                        counts["simplex"] += 1
                        assert all(abs(w) < 1e-9 for w, present in zip(weights, masks) if not present)
                        counts["masked_zero_weight"] += 1
                        if method.startswith("revised"):
                            violation = max(weights) - row["effective_weight_cap"]
                            assert violation < 1e-8
                            maximum_cap_violation = max(maximum_cap_violation, violation)
                            counts["cap"] += 1
                    if not any(masks):
                        assert row["fallback"] and row["coverage"] == 0 and row["uncertainty"] == 1
                        assert not row["raw_window_alarm"] and not row["adaptation_admissible"]
                        counts["all_missing_abstentions"] += 1
                    if row["evaluation_completed_slot"] < design["fault_intervals_relative_to_evaluation"][0][0]:
                        # Future corruptions cannot change any saved earlier forecast or state.
                        for key in ("gamma", "weights", "scores", "masks", "next_target_predictions",
                                    "forecast_score", "uncertainty", "coverage", "fallback"):
                            assert row.get(key) == before.get(key), (seed, method, scenario, key)
                        counts["future_boundary_comparisons"] += 1
    for scenario in design["scenarios"]:
        for seed in design["seeds"]:
            selected = [row for row in summary if row["scenario"] == scenario and row["seed"] == seed]
            assert len(selected) == len(methods)
            assert len({row["forecast_pairs_common"] for row in selected}) == 1
            assert all(row["event_count"] == 2 for row in selected)
            counts["common_target_count_groups"] += 1
    for path, expected in manifest["implementation_hashes"].items():
        assert hashlib.sha256(current_bundle_path(path).read_bytes()).hexdigest() == expected
        counts["source_hashes"] += 1
    output = {"all_passed": True, "counts": counts,
              "maximum_simplex_error": maximum_simplex_error,
              "maximum_cap_violation": maximum_cap_violation,
              "future_boundary": "Every saved forecast before the first corrupted slot matches clean replay exactly",
              "interpretation": "Numerical/causal replay checks; not an external validation of data realism or algorithm superiority"}
    if not args.check_only:
        (ROOT / "audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
