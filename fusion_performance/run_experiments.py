"""Run actual service-policy backends on identical exogenous arrays per seed."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from closed_loop import ROOT, calibrate, run_closed, write_records
from data_pipeline import generate, read_json, save_stream, sha256, write_json


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(experiment, core, profiles, data_dir, output, quick=False):
    output.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries, manifest = [], []
    started = time.perf_counter()
    for seed in experiment["seeds"]:
        path = data_dir / f"exogenous_seed_{seed}.npz"
        if path.exists():
            data = dict(np.load(path, allow_pickle=False))
        else:
            data = generate(experiment, core["coordination"]["nodes"], seed)
            save_stream(path, data)
        if len(data["arrivals"]) != experiment["calibration_slots"] + experiment["evaluation_slots"]:
            raise ValueError("Exogenous stream length does not match this run")
        if int(data["seed"]) != seed:
            raise ValueError("Stored seed does not match the paired run")
        if str(data["provenance"]) == "controlled_synthetic":
            expected = generate(experiment, core["coordination"]["nodes"], seed)
            if any(not np.array_equal(data[key], expected[key]) for key in ("arrivals", "capacities", "rates", "mixes")):
                raise ValueError("Stale exogenous data: regenerate with the declared configuration")
        try:
            stream_path = str(path.resolve().relative_to(ROOT))
        except ValueError:
            stream_path = str(path.resolve())
        manifest.append({"seed": seed, "path": stream_path, "sha256": sha256(path),
                         "provenance": str(data["provenance"])})
        for backbone in experiment["backbones"]:
            initial, reference, calibrated, prefix = calibrate(data, experiment, core, profiles, seed, backbone)
            calibration_dir = output / "calibration" / f"{backbone}_seed_{seed}"
            calibration_dir.mkdir(parents=True, exist_ok=True)
            write_json(calibration_dir / "reference.json", reference)
            write_json(calibration_dir / "profiles.json", calibrated)
            torch.save(initial.checkpoint(), calibration_dir / "initial_policy.pt")
            write_records(calibration_dir / "prefix.csv.gz", prefix)
            methods = list(experiment["main_methods"])
            if backbone == "dqn" and not quick:
                methods += experiment["ablation_methods"]
            for method in methods:
                records, windows, events, summary = run_closed(data, experiment, core, calibrated, initial, reference, method, seed)
                directory = output / "runs" / f"{backbone}_{method}_seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                write_records(directory / "slots.csv.gz", records)
                write_json(directory / "windows.json", windows)
                write_json(directory / "deployments.json", events)
                summary["input_sha256"] = manifest[-1]["sha256"]
                summary["calibration_sha256"] = sha256(calibration_dir / "initial_policy.pt")
                summary["evaluation_slots"] = experiment["evaluation_slots"]
                write_json(directory / "summary.json", summary)
                summaries.append(summary)
                print(f"{backbone:3} seed={seed:3} method={method:16} return={summary['return']:.4f} "
                      f"jobs={summary['training_jobs']:3} deployments={summary['deployed_jobs']:3} "
                      f"violations={summary['resource_violations']} runtime={summary['runtime_seconds']:.2f}s", flush=True)
    write_csv(output / "seed_results.csv", summaries)
    write_json(output / "seed_results.json", summaries)
    write_json(output / "run_manifest.json", {
        "data_kind": "smoke_test" if quick else experiment["data_kind"],
        "mode": "executed_closed_loop", "configuration": experiment, "core_configuration": core,
        "profile_priors": profiles, "inputs": manifest, "runs": len(summaries),
        "python": platform.python_version(), "system": platform.platform(), "processor": platform.processor(),
        "numpy": np.__version__, "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "calibration_in_evaluation_return": False, "unexecuted_policy_returns_used": False,
        "elapsed_seconds": time.perf_counter() - started})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    experiment = read_json(ROOT / "experiment_config.json")
    core, profiles = read_json(ROOT / "core/config.json"), read_json(ROOT / "core/profiles.json")
    if args.quick:
        experiment = copy.deepcopy(experiment)
        experiment.update(seeds=[10], calibration_slots=256, evaluation_slots=300,
                          drift_onsets=[101, 201], main_methods=["fusion", "no_rt", "periodic", "dpp"])
        experiment["agent"]["warmup"] = 128
        core["evaluation_horizon"] = 300
        args.data, args.output = ROOT / "smoke_data", ROOT / "smoke_results"
    run(experiment, core, profiles, args.data, args.output, args.quick)


if __name__ == "__main__":
    main()
