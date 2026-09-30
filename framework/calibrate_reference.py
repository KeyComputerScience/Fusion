"""Freeze normalization using only a pre-evaluation prefix."""
import argparse
import hashlib
import json
import statistics

from fusion import finite
from logio import read_csv, read_json, write_json


def fit_reference(rows, config, through_slot):
    prefix = [row for row in rows if row["slot"] <= through_slot]
    if not prefix or prefix[-1]["slot"] != through_slot:
        raise ValueError("Calibration cutoff must be present in the prefix")
    columns = config["state_columns"]
    scales = {}
    for column in columns:
        values = [row[column] for row in prefix if finite(row.get(column))]
        if len(values) < 2:
            raise ValueError(f"Insufficient calibration data for {column}")
        scales[column] = {"mean": statistics.fmean(values),
                          "std": max(config["normalization_std_floor"], statistics.pstdev(values)),
                          "samples": len(values)}
    # The digest intentionally covers only the prefix and normalization inputs.
    payload = [{"slot": row["slot"], **{name: row.get(name) for name in columns}} for row in prefix]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"schema_version": "1.0", "fit_through_slot": through_slot,
            "calibration_samples": len(prefix), "state_columns": columns,
            "state_scales": scales, "calibration_prefix_sha256": digest,
            "provenance": sorted({row.get("provenance", "unspecified") for row in prefix})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="example_stream.csv")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--through-slot", type=int, default=300)
    parser.add_argument("--output", default="reference.json")
    args = parser.parse_args()
    write_json(args.output, fit_reference(read_csv(args.input), read_json(args.config), args.through_slot))
    print(f"Frozen prefix through slot {args.through_slot}: {args.output}")


if __name__ == "__main__":
    main()
