"""Regenerate an interrupted gzip log and require identical returns/model hashes."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from closed_loop import ROOT, calibrate, run_closed, write_records
from data_pipeline import read_json, write_json
from run_experiments import write_csv


def repair(results):
    experiment, core = read_json(ROOT / "experiment_config.json"), read_json(ROOT / "core/config.json")
    profiles = read_json(ROOT / "core/profiles.json")
    summaries = read_json(results / "seed_results.json")
    cache, repaired = {}, []
    for index, summary in enumerate(summaries):
        directory = results / "runs" / f"{summary['backbone']}_{summary['method']}_seed_{summary['seed']}"
        try:
            with gzip.open(directory / "slots.csv.gz", "rb") as stream:
                while stream.read(1024 * 1024):
                    pass
            continue
        except (OSError, EOFError):
            pass
        key = (summary["seed"], summary["backbone"])
        if key not in cache:
            data = dict(np.load(ROOT / "data" / f"exogenous_seed_{key[0]}.npz", allow_pickle=False))
            agent, reference, calibrated, _ = calibrate(data, experiment, core, profiles, *key)
            cache[key] = data, agent, reference, calibrated
        data, agent, reference, calibrated = cache[key]
        rows, windows, events, replacement = run_closed(data, experiment, core, calibrated, agent, reference,
                                                       summary["method"], summary["seed"])
        for name in ("return", "final_parameter_sha256", "completed_count", "deadline_count", "training_cost", "deployed_jobs"):
            if replacement[name] != summary[name]:
                raise AssertionError(f"Repair is not deterministic: {directory.name}, {name}")
        write_records(directory / "slots.csv.gz", rows)
        write_json(directory / "windows.json", windows)
        write_json(directory / "deployments.json", events)
        replacement.update(input_sha256=summary["input_sha256"], calibration_sha256=summary["calibration_sha256"],
                           evaluation_slots=summary["evaluation_slots"])
        summaries[index] = replacement
        write_json(directory / "summary.json", replacement)
        repaired.append(directory.name)
        print(f"Regenerated and verified {directory.name}", flush=True)
    write_csv(results / "seed_results.csv", summaries)
    write_json(results / "seed_results.json", summaries)
    write_json(results / "log_regeneration.json", {"repaired": repaired,
                "criterion": "identical reward, task counts, training cost, deployment count and final model digest"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    repair(parser.parse_args().results)
