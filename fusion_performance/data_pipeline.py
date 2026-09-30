"""Paired exogenous streams and a provenance-preserving Alibaba import route."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(configuration, nodes, seed, trace=None):
    count = configuration["calibration_slots"] + configuration["evaluation_slots"]
    relative = np.arange(1, count + 1) - configuration["calibration_slots"]
    phase = np.searchsorted(configuration["drift_onsets"], relative, side="right")
    rates = np.asarray(configuration["phase_rates"])[phase]
    mixes = np.asarray(configuration["phase_mixes"])[phase]
    provenance = "controlled_synthetic"
    if trace is not None:
        if len(trace["rates"]) != count or trace["capacities"].shape != (count, nodes, 3):
            raise ValueError("Imported trace length and node count must match the complete protocol")
        rates, mixes = trace["rates"], trace["mixes"]
        provenance = "alibaba_2018_driven_with_synthetic_service_tasks"
    rng = np.random.default_rng(seed)
    arrivals = rng.poisson(rates[:, None, None] * mixes[:, None, :], size=(count, nodes, 3))
    capacities = np.ones((count, nodes, 3))
    baseline = rng.uniform(*configuration["capacity_uniform"], size=(nodes, 2))
    wave = 1 - configuration["capacity_wave_amplitude"] * np.sin(np.arange(count)[:, None] / 120 + np.arange(nodes)[None, :])
    capacities[:, :, 1:] = baseline[None, :, :] * wave[:, :, None]
    if trace is not None:
        capacities = trace["capacities"].copy()
    return {"arrivals": arrivals, "capacities": capacities, "rates": rates, "mixes": mixes,
            "phase": phase, "provenance": np.array(provenance), "seed": np.array(seed)}


def save_stream(path, data):
    np.savez_compressed(path, **data)
    return {"path": str(path), "sha256": sha256(path), "seed": int(data["seed"]),
            "provenance": str(data["provenance"]), "slots": len(data["arrivals"])}


def csv_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        yield from csv.reader(stream)


def import_alibaba(usage_path, task_path, configuration, nodes, slot_seconds, output):
    """Import official headerless fields; never substitute generated trace records."""
    count = configuration["calibration_slots"] + configuration["evaluation_slots"]
    if slot_seconds <= 0:
        raise ValueError("slot_seconds must be positive")
    cutoff = configuration["calibration_slots"]
    limit = count * slot_seconds
    prefix_seen, invalid = {}, 0
    # machine_id,time_stamp,cpu_util_percent,mem_util_percent,...,net_in,net_out,...
    for row in csv_rows(usage_path):
        try:
            stamp, cpu, memory = float(row[1]), float(row[2]) / 100, float(row[3]) / 100
            if not all(math.isfinite(v) for v in (stamp, cpu, memory)):
                raise ValueError
            if not 0 <= stamp < limit:
                continue
            if not 0 <= cpu <= 1 or not 0 <= memory <= 1:
                raise ValueError
            slot = int(stamp // slot_seconds)
            if slot < cutoff:
                prefix_seen.setdefault(row[0], bytearray(cutoff))[slot] = 1
        except (ValueError, IndexError):
            invalid += 1
    prefix_coverage = {machine: sum(slots) for machine, slots in prefix_seen.items()}
    selected = sorted((m for m in prefix_coverage if prefix_coverage[m] > 0),
                      key=lambda m: (-prefix_coverage[m], m))[:nodes]
    if len(selected) != nodes:
        raise ValueError("Insufficient machines with valid observations")
    # A second streaming pass aggregates only the selected nodes, avoiding
    # full-horizon storage for every machine in the original large trace.
    aggregate = {machine: {} for machine in selected}
    for row in csv_rows(usage_path):
        try:
            if row[0] not in aggregate:
                continue
            stamp, cpu, memory = float(row[1]), float(row[2]) / 100, float(row[3]) / 100
            if (not all(math.isfinite(v) for v in (stamp, cpu, memory))
                    or not 0 <= stamp < limit or not 0 <= cpu <= 1 or not 0 <= memory <= 1):
                continue
            slot = int(stamp // slot_seconds)
            stats = aggregate[row[0]].setdefault(slot, [0.0, 0.0, 0])
            stats[0] += cpu
            stats[1] += memory
            stats[2] += 1
        except (ValueError, IndexError):
            continue
    # batch_task: task_name,instance_num,job_name,task_type,status,start_time,end_time,plan_cpu,plan_mem.
    totals, cpu_sum, mem_sum = np.zeros(count), np.zeros(count), np.zeros(count)
    task_invalid = 0
    for row in csv_rows(task_path):
        try:
            stamp, number = float(row[5]), int(row[1])
            cpu, memory = float(row[7]) / 100, float(row[8]) / 100
            if not all(math.isfinite(v) for v in (stamp, cpu, memory)) or cpu < 0 or not 0 <= memory <= 1:
                raise ValueError
            if not 0 <= stamp < limit or number <= 0:
                continue
            slot = int(stamp // slot_seconds)
            totals[slot] += number
            cpu_sum[slot] += number * max(0.0, cpu)
            mem_sum[slot] += number * max(0.0, memory)
        except (ValueError, IndexError):
            task_invalid += 1
    mean = float(totals[:cutoff].mean())
    if mean <= 0:
        raise ValueError("The calibration prefix contains no valid task arrivals")
    rates = np.clip(1.15 * totals / mean, 0.05, 4.0)
    cpu_ratio = cpu_sum / np.maximum(totals, 1)
    mixes = np.tile([0.55, 0.30, 0.15], (count, 1))
    mixes[cpu_ratio >= 0.75] = [0.10, 0.30, 0.60]
    mixes[cpu_ratio < 0.25] = [0.75, 0.20, 0.05]
    capacities = np.ones((count, nodes, 3))
    missing = []
    for index, machine in enumerate(selected):
        last_cpu, last_mem = 0.5, 0.5
        for slot in range(count):
            stats = aggregate[machine].get(slot)
            if stats:
                last_cpu, last_mem = stats[0] / stats[2], stats[1] / stats[2]
            else:
                missing.append((machine, slot + 1))
            capacities[slot, index] = [max(0.05, 1 - last_mem), max(0.05, 1 - last_cpu), 1.0]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, rates=rates, mixes=mixes, capacities=capacities)
    write_json(output.with_suffix(".json"), {
        "mode": "alibaba_2018_driven_import", "machines": selected,
        "selection": "descending calibration-prefix valid slot coverage, then machine ID; evaluation coverage is not used",
        "calibration_machine_coverage": {m: prefix_coverage[m] for m in selected},
        "usage_sha256": sha256(usage_path), "task_sha256": sha256(task_path),
        "slot_seconds": slot_seconds, "time_interval": [0, limit],
        "calibration_cutoff": cutoff, "arrival_prefix_mean": mean,
        "fill": "forward-only CPU/memory, initial 0.5; missing slots logged",
        "budget": "synthetic unit capacity; not supplied by the trace",
        "invalid_usage_rows": invalid, "invalid_task_rows": task_invalid,
        "missing_slots": missing, "task_memory_mean": float(mem_sum.sum() / max(totals.sum(), 1)),
        "source": "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "import-alibaba"))
    parser.add_argument("--config", type=Path, default=ROOT / "experiment_config.json")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--machine-usage", type=Path)
    parser.add_argument("--batch-task", type=Path)
    parser.add_argument("--slot-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    config = read_json(args.config)
    nodes = read_json(ROOT / "core/config.json")["coordination"]["nodes"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "import-alibaba":
        if not args.machine_usage or not args.batch_task:
            parser.error("Both official CSV paths are required")
        import_alibaba(args.machine_usage, args.batch_task, config, nodes, args.slot_seconds, args.output / "trace.npz")
        return
    trace = dict(np.load(args.trace, allow_pickle=False)) if args.trace else None
    results = []
    for seed in config["seeds"]:
        data = generate(config, nodes, seed, trace)
        results.append(save_stream(args.output / f"exogenous_seed_{seed}.npz", data))
    write_json(args.output / "manifest.json", {"streams": results, "protocol": config,
               "trace_import": read_json(args.trace.with_suffix(".json")) if args.trace else None})
    print(json.dumps({"seeds": config["seeds"], "slots_per_seed": len(data["arrivals"]),
                      "kind": str(data["provenance"])}), flush=True)


if __name__ == "__main__":
    main()
