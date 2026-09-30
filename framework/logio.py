"""CSV/JSON helpers and explicit input checks; no numerical dependencies."""
import csv
import json
import math
from pathlib import Path

FLOAT_COLUMNS = ("queue", "arrival_rate", "utilization", "cpu_fraction", "bandwidth",
                 "inference_load", "training_load", "environment_reward", "observed_training_cost",
                 "completion_ratio", "deadline_loss", "availability", "throughput_ratio",
                 "deployed_gain", "completed_training_load")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = {key: (None if value == "" else value) for key, value in raw.items()}
            row["slot"] = int(row["slot"])
            for key in FLOAT_COLUMNS:
                value = row.get(key)
                row[key] = float(value) if value is not None else None
                if row[key] is not None and not math.isfinite(row[key]):
                    raise ValueError(f"Nonfinite input at slot {row['slot']}: {key}")
            row["capacities"] = json.loads(row.pop("capacities_json"))
            for key in ("inference_load", "training_load"):
                if row.get(key) is None or row[key] < 0:
                    raise ValueError(f"Missing/negative load at slot {row['slot']}: {key}")
            if row.get("deadline_loss") is not None and row["deadline_loss"] < 0:
                raise ValueError("Deadline loss must use the positive part")
            for key in ("completion_ratio", "availability", "throughput_ratio"):
                if row.get(key) is not None and not 0 <= row[key] <= 1:
                    raise ValueError(f"{key} must lie in [0,1]")
            if rows and row["slot"] != rows[-1]["slot"] + 1:
                raise ValueError("CSV must have unique, contiguous, increasing slots")
            rows.append(row)
    if not rows:
        raise ValueError("The input CSV is empty")
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV without a schema")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config):
    """Fail before processing when scales, bounds or iteration limits are invalid."""
    def positive_integer(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    def positive(value, name):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    positive_integer(config["evaluation_horizon"], "evaluation_horizon")
    positive(config["normalization_std_floor"], "normalization_std_floor")
    if not config["state_columns"] or len(set(config["state_columns"])) != len(config["state_columns"]):
        raise ValueError("State columns must be nonempty and unique")
    fusion = config["fusion"]
    for name in ("window_size", "minimum_source_samples", "minimum_context_samples",
                 "bootstrap_replicates", "bootstrap_block_size"):
        positive_integer(fusion[name], name)
    if fusion["bootstrap_replicates"] < 2 or fusion["minimum_source_samples"] > fusion["window_size"]:
        raise ValueError("Bootstrap requires at least two replicates; support cannot exceed window size")
    for name in ("support_scale", "freshness_scale", "error_scale", "epsilon_noise", "epsilon_reward",
                 "unsupported_noise_variance"):
        positive(fusion[name], name)
    calibration = fusion["predictive_calibration"]
    positive(calibration["ridge"], "ridge")
    for name in ("forgetting", "error_rate"):
        if not 0 < calibration[name] <= 1:
            raise ValueError(f"{name} must lie in (0,1]")
    coordination = config["coordination"]
    for name in ("nodes", "lookahead", "exact_pair_limit", "ao_max_sweeps"):
        positive_integer(coordination[name], name)
    for name in ("initial_recovery", "eta_gamma", "eta_uncertainty", "xi", "lambda_cost",
                 "omega_intensity", "kappa_uncertainty"):
        value = coordination[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < coordination["rho_min"] <= 1 or not 0 < coordination["quality_update_rate"] <= 1:
        raise ValueError("rho_min and quality_update_rate must lie in (0,1]")
    weights = coordination["quality_weights"]
    if any(weight < 0 for weight in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Quality weights must be nonnegative and sum to one")
    if set(weights) != {"completion", "deadline_compliance", "availability", "throughput"}:
        raise ValueError("The quality metric names are fixed by the implementation")
    trigger = coordination["trigger"]
    if not 0 <= trigger["off"] < trigger["on"] <= 1:
        raise ValueError("Require 0 <= trigger.off < trigger.on <= 1")
    if not 0 <= trigger["minimum_coverage"] <= 1 or not 0 <= trigger["maximum_uncertainty"] <= 1:
        raise ValueError("Coverage and uncertainty thresholds must lie in [0,1]")
