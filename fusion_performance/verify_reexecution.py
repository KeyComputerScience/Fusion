"""Repeat two complete reference runs and compare reward and parameter digests."""
import argparse
from pathlib import Path

import numpy as np

from closed_loop import ROOT, calibrate, run_closed
from data_pipeline import read_json, write_json


def verify(results):
    experiment = read_json(ROOT / "experiment_config.json")
    core, profiles = read_json(ROOT / "core/config.json"), read_json(ROOT / "core/profiles.json")
    data = dict(np.load(ROOT / "data/exogenous_seed_10.npz", allow_pickle=False))
    outcomes = []
    for backbone in experiment["backbones"]:
        expected = read_json(results / "runs" / f"{backbone}_fusion_seed_10" / "summary.json")
        agent, reference, calibrated, _ = calibrate(data, experiment, core, profiles, 10, backbone)
        _, _, events, observed = run_closed(data, experiment, core, calibrated, agent, reference, "fusion", 10)
        exact = ["return", "arrival_count", "completed_count", "deadline_count", "queue_end", "training_jobs",
                 "deployed_jobs", "gradient_updates", "training_cost", "initial_parameter_sha256", "final_parameter_sha256"]
        if any(expected[key] != observed[key] for key in exact):
            raise AssertionError(f"Full reexecution differs for {backbone}")
        outcomes.append({"backbone": backbone, "seed": 10, "slots": experiment["evaluation_slots"],
                         "return": observed["return"], "return_difference": 0., "equal_fields": exact,
                         "initial_parameter_sha256": agent.parameter_digest(),
                         "final_parameter_sha256": observed["final_parameter_sha256"],
                         "calibration_gradient_updates": agent.updates,
                         "parameter_changing_deployments": len(events), "all_passed": True})
        print(f"Full {backbone.upper()} seed=10 reexecution matches return and model digests", flush=True)
    write_json(results / "reexecution_checks.json", outcomes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    verify(parser.parse_args().results)
