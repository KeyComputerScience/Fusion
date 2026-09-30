"""Streaming adapter: decide before feedback, then complete evidence windows.

Offline replay diagnoses fusion and recommended profiles. Logged outcomes cannot
measure the return of a different recommended policy. For live use, call decide()
on current loads/capacities, execute that choice, then observe() with real feedback.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from coordinator import Coordinator
from fusion import EvidenceFusion
from logio import read_csv, read_json, validate_config, write_csv, write_json


class FusionPipeline:
    def __init__(self, config, profiles, reference, solver="auto", collect_history=False):
        validate_config(config)
        self.config = config
        self.reference = reference
        self.solver = solver
        self.collect_history = collect_history
        self.fusion = EvidenceFusion(config["fusion"], reference)
        if reference["state_columns"] != config["state_columns"]:
            raise ValueError("Reference and configured state columns must agree")
        self.coordinator = Coordinator(config["coordination"], profiles)
        self.next_slot = reference["fit_through_slot"] + 1
        self.origin = self.next_slot
        self.pending_decision = None
        self.window_rows = []
        self.window_results = []

    def decide(self, slot, training_load, inference_load, capacities,
               future_retentions=None, pending_gain_forecast=None):
        if self.pending_decision is not None:
            raise RuntimeError("Feedback for the preceding decision is required")
        if slot != self.next_slot:
            raise ValueError(f"Expected slot {self.next_slot}, received {slot}")
        remaining = self.config["evaluation_horizon"] - (slot - self.origin) - 1
        if remaining < 0:
            raise ValueError("Slot exceeds the declared evaluation horizon")
        self.pending_decision = self.coordinator.choose(
            slot, training_load, inference_load, capacities,
            self.fusion.snapshot, remaining, self.solver, future_retentions, pending_gain_forecast)
        return self.pending_decision

    def observe(self, feedback):
        if self.pending_decision is None or feedback["slot"] != self.pending_decision.slot:
            raise ValueError("Feedback must match the pending decision's slot")
        if feedback["slot"] <= self.reference["fit_through_slot"]:
            raise ValueError("Evaluation feedback must follow the reference prefix")
        result = self.coordinator.observe(feedback, self.pending_decision.rho)
        self.window_rows.append(dict(feedback))
        size = self.config["fusion"]["window_size"]
        if len(self.window_rows) == size:
            index = (feedback["slot"] - self.origin + 1) // size
            snapshot = self.fusion.complete_window(self.window_rows, index)
            if self.collect_history:
                self.window_results.append(snapshot.to_dict())
            self.window_rows = []
        self.pending_decision = None
        self.next_slot += 1
        return result


def run_replay(rows, config, profiles, reference, solver="auto"):
    pipeline = FusionPipeline(config, profiles, reference, solver, collect_history=True)
    evaluation = [row for row in rows if row["slot"] > reference["fit_through_slot"]]
    if not evaluation or evaluation[0]["slot"] != pipeline.origin:
        raise ValueError("Evaluation must start immediately after calibration cutoff")
    decisions = []
    for row in evaluation:
        decision = pipeline.decide(row["slot"], row["training_load"], row["inference_load"], row["capacities"])
        record = decision.to_dict()
        snapshot = pipeline.fusion.snapshot
        for source, name in enumerate(("workload", "operating", "performance")):
            record[f"score_{name}"] = snapshot.scores[source]
            record[f"weight_{name}"] = snapshot.weights[source]
        record["logged_training"] = row["logged_training"]
        record["logged_inference"] = row["logged_inference"]
        record["matches_logged_pair"] = (record["training_profile"] == row["logged_training"]
                                         and record["inference_profile"] == row["logged_inference"])
        record["trajectory"] = json.dumps(record["trajectory"])
        record.update(pipeline.observe(row))
        decisions.append(record)
    summary = {
        "mode": "offline_logged_replay", "causal_return_of_recommendations_evaluated": False,
        "provenance": sorted({row.get("provenance", "unspecified") for row in evaluation}),
        "slots": len(decisions), "complete_windows": len(pipeline.window_results),
        "unfinished_window_samples": len(pipeline.window_rows),
        "solver_counts": dict(Counter(row["solver"] for row in decisions)),
        "status_counts": dict(Counter(row["status"] for row in decisions)),
        "training_recommendations": dict(Counter(row["training_profile"] for row in decisions)),
        "inference_recommendations": dict(Counter(row["inference_profile"] for row in decisions)),
        "maximum_evaluated_pairs": max(row["evaluated_pairs"] for row in decisions),
        "warning": "Synthetic demo and illustrative profiles are not manuscript experimental results."
    }
    return decisions, pipeline.window_results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="example_stream.csv")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--profiles", default="profiles.json")
    parser.add_argument("--reference", default="reference.json")
    parser.add_argument("--solver", choices=("auto", "exact", "ao"), default="auto")
    parser.add_argument("--output", default="example_results")
    args = parser.parse_args()
    rows = read_csv(args.input)
    decisions, windows, summary = run_replay(rows, read_json(args.config), read_json(args.profiles),
                                           read_json(args.reference), args.solver)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "decisions.csv", decisions)
    (directory / "fusion_windows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in windows), encoding="utf-8")
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
