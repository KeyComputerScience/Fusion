"""Generate a reproducible SYNTHETIC logging stream, not paper evaluation data."""
import argparse
import json
import math
import random

from fusion import clip
from logio import read_json, write_csv


def generate(config, profiles, calibration_slots=300, seed=20260930):
    rng = random.Random(seed)
    nodes = config["coordination"]["nodes"]
    capacities = [[1.0, rng.uniform(0.9, 1.15), rng.uniform(0.9, 1.15)] for _ in range(nodes)]
    tr = {profile["id"]: profile for profile in profiles["training"]}
    inf = {profile["id"]: profile for profile in profiles["inference"]}
    pending_gains = {}
    rows = []
    horizon = config["evaluation_horizon"]
    for slot in range(1, calibration_slots + horizon + 1):
        relative = slot - calibration_slots
        phase = 0 if relative <= 300 else (1 if relative <= 700 else 2)
        weights = ([0.75, 0.20, 0.05], [0.10, 0.30, 0.60], [0.55, 0.35, 0.10])[phase]
        regime = rng.choices(["light", "mixed", "cpu_heavy"], weights=weights)[0]
        arrival = max(0.05, (0.45, 0.85, 0.60)[phase] + rng.gauss(0, 0.035))
        queue = max(0.0, (0.18, 0.68, 0.30)[phase] + rng.gauss(0, 0.04))
        utilization = clip((0.40, 0.75, 0.52)[phase] + rng.gauss(0, 0.02))
        cpu_fraction = clip((0.18, 0.65, 0.28)[phase] + rng.gauss(0, 0.025))
        bandwidth = max(0.05, (0.88, 0.52, 0.76)[phase] + rng.gauss(0, 0.025))
        inference_load = clip(0.70 + 0.35 * arrival, 0.1, 1.2)
        training_load = 0.5
        # A fixed logging policy supplies context overlap in adjacent windows.
        logged_training = "tr1" if slot % 10 == 0 else "tr0"
        logged_inference = ["inf2", "inf3", "inf4"][slot % 3]
        profile_tr, profile_inf = tr[logged_training], inf[logged_inference]
        cost = training_load * profile_tr["cost"]
        if profile_tr["gain"]:
            deployment_slot = slot + profile_tr["deployment_delay"] - 1
            pending_gains[deployment_slot] = pending_gains.get(deployment_slot, 0.0) + training_load * profile_tr["gain"]
        deployed_gain = pending_gains.pop(slot, 0.0)
        # This external synthetic reward is deliberately independent of f(H).
        service_shift = (0.0, 0.18, 0.06)[phase]
        completion = clip(0.96 - service_shift + rng.gauss(0, 0.012))
        deadline_loss = max(0.0, 0.04 + 0.8 * service_shift + rng.gauss(0, 0.008))
        availability = clip(0.99 + rng.gauss(0, 0.003))
        throughput = clip(0.94 - service_shift + rng.gauss(0, 0.012))
        external_reward = (profile_inf["beta"] * completion - 0.2 * deadline_loss
                           - config["fusion"]["observed_cost_reward_scale"] * cost + rng.gauss(0, 0.008))
        row = {
            "slot": slot, "regime": regime, "queue": queue, "arrival_rate": arrival,
            "utilization": utilization, "cpu_fraction": cpu_fraction, "bandwidth": bandwidth,
            "inference_load": inference_load, "training_load": training_load,
            "capacities_json": json.dumps(capacities), "logged_training": logged_training,
            "logged_inference": logged_inference, "policy_version": "synthetic_policy_v1",
            "environment_reward": external_reward, "observed_training_cost": cost,
            "completion_ratio": completion, "deadline_loss": deadline_loss,
            "availability": availability, "throughput_ratio": throughput,
            "deployed_gain": deployed_gain, "completed_training_profile": "",
            "completed_training_load": "", "provenance": "synthetic"
        }
        # Deliberate source outages. Dispatch metadata remains observable.
        if 901 <= relative <= 1000:
            row["regime"] = ""
            for name in config["state_columns"]:
                row[name] = ""
            row["environment_reward"] = row["observed_training_cost"] = ""
        elif 801 <= relative <= 850:
            row["environment_reward"] = row["observed_training_cost"] = ""
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--profiles", default="profiles.json")
    parser.add_argument("--output", default="example_stream.csv")
    parser.add_argument("--calibration-slots", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    rows = generate(read_json(args.config), read_json(args.profiles), args.calibration_slots, args.seed)
    write_csv(args.output, rows)
    print(f"SYNTHETIC stream: {len(rows)} rows written to {args.output}")


if __name__ == "__main__":
    main()
