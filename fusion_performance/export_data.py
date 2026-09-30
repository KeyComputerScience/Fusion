"""Export data provenance and illustrative stored forecasts from executed runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from data_pipeline import ROOT, read_json, sha256, write_json
from run_experiments import write_csv


def export(results):
    experiment = read_json(ROOT / "experiment_config.json")
    entries = []
    for seed in experiment["seeds"]:
        path = ROOT / "data" / f"exogenous_seed_{seed}.npz"
        with np.load(path, allow_pickle=False) as values:
            entries.append({"seed": seed, "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                            "provenance": str(values["provenance"]),
                            "arrays": {name: {"shape": list(values[name].shape), "dtype": str(values[name].dtype)}
                                       for name in values.files},
                            "evaluation_arrivals": int(values["arrivals"][experiment["calibration_slots"]:].sum())})
    write_json(ROOT / "data/manifest.json", {"protocol": experiment, "streams": entries,
               "source": "generated Poisson tasks and normalized synthetic capacities; not Alibaba trace records"})
    examples = []
    for backbone in experiment["backbones"]:
        windows = read_json(results / "runs" / f"{backbone}_fusion_seed_10" / "windows.json")
        targets = {row["window_index"]: row["scores"][2] for row in windows}
        for row in windows:
            if row["window_index"] not in (18, 19, 20, 36, 37, 38):
                continue
            predictions = row["next_target_predictions"]
            available = (not row["fallback"] and
                         all(p is not None or w == 0 for p, w in zip(predictions, row["weights"])))
            combined = sum(p * w for p, w in zip(predictions, row["weights"]) if p is not None) if available else None
            target = targets.get(row["window_index"] + 1)
            examples.append({"backbone": backbone, "method": "fusion", "seed": 10,
                             "window": row["window_index"],
                             "available_evaluation_slot": row["available_from_slot"] - experiment["calibration_slots"],
                             "gamma": row["gamma"], "uncertainty": row["uncertainty"],
                             "source_scores": row["scores"], "source_weights": row["weights"],
                             "stored_source_forecasts": predictions, "combined_forecast": combined,
                             "next_window_performance_target_posthoc": target,
                             "absolute_error_posthoc": abs(combined - target) if combined is not None and target is not None else None,
                             "target_in_decision_inputs": False})
    write_json(results / "forecast_examples.json", {"mode": "stored predictions from actual closed-loop runs",
               "target": "next-window performance-degradation evidence; not task-success probability", "rows": examples})
    write_csv(results / "forecast_examples.csv", [
        {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
        for row in examples])
    print(f"Exported five input manifests and {len(examples)} stored forecast examples", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    export(parser.parse_args().results)
