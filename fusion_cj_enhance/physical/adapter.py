"""Metadata-prepared HARTH adapter; acquisition and ingestion require root freeze.

No network calls occur here. Labels never select rows, subjects, sensor groups,
class vocabulary, stride, history boundaries or stream boundaries. The adapter
contains no CJ scientific function; the next method runner supplies its engine.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NAME = "harth779"
FIELDS = ("timestamp", "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label")
SOURCES = ("lower_back_AX3", "right_front_thigh_AX3")
GROUPS = [[0, 1, 2], [3, 4, 5]]
CLASS_CODES = (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140)
CLASS_NAMES = ("walking", "running", "shuffling", "stairs_ascending", "stairs_descending",
               "standing", "sitting", "lying", "cycling_sit", "cycling_stand",
               "cycling_sit_inactive", "cycling_stand_inactive")
CLASS_INDEX = {code: i for i, code in enumerate(CLASS_CODES)}
SPEC = dict(classes=12, groups=GROUPS, sources=list(SOURCES), raw_sample_rate_hz=50,
            acquisition_row_stride=50, first_retained_acquisition_row=0,
            features="source-major back_xyz,thigh_xyz in g; direct simultaneous physical observations",
            source_timestamp_relation="all six source coordinates are from the same retained acquisition row",
            lag="current and previous two retained rows; repeat first available history row",
            history_reset="participant or raw timestamp gap above1second",
            split="numeric filename subject-ID ranks1:8fit,9:12calibration,13:22test",
            fit_audit="each fit participant chronological first70percent fit,last30percent prior audit")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_authorization(path):
    """Run before opening an archive or cache. Draft artifacts cannot authorize."""
    path = Path(path).resolve()
    auth = json.loads(path.read_text())
    protocol = ROOT / "protocol.json"
    if not protocol.exists():
        raise ValueError("Final physical protocol.json does not exist; draft cannot authorize ingestion")
    p = json.loads(protocol.read_text())
    if p.get("status") != "frozen_before_acquisition":
        raise ValueError("Physical protocol has not been frozen")
    if auth.get("authorized_stage") != "algorithm_and_physical_protocol_frozen_before_acquisition":
        raise ValueError("Root algorithm/physical acquisition authorization is required")
    if auth.get("adapter_sha256") != sha(__file__) or auth.get("physical_protocol_sha256") != sha(protocol):
        raise ValueError("Authorized adapter/protocol bytes differ")
    algorithm = Path(auth["algorithm_freeze_path"])
    if not algorithm.is_absolute():
        algorithm = path.parent / algorithm
    if sha(algorithm) != auth["algorithm_freeze_sha256"]:
        raise ValueError("Root final algorithm freeze differs")
    return auth


def timestamp_us(value):
    stamp = dt.datetime.fromisoformat(value.strip())
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return ((stamp.toordinal() * 86400 + stamp.hour * 3600 + stamp.minute * 60
             + stamp.second) * 1_000_000 + stamp.microsecond)


def parse_recording(handle, subject):
    """Stream full acquisition rows, retaining indices0,50,100,... causally."""
    reader = csv.DictReader(handle)
    header = tuple(reader.fieldnames or ())
    ignored_export_columns = tuple(name for name in header if name not in FIELDS)
    # Official S015/S021 include an inert named export index; S023 includes
    # an unnamed one. Preserve scientific column order and all row values.
    if (len(header) not in (len(FIELDS), len(FIELDS)+1)
            or len(set(header)) != len(header)
            or tuple(name for name in header if name in FIELDS) != FIELDS
            or len(ignored_export_columns) > 1
            or any(name not in ("index", "") for name in ignored_export_columns)):
        raise ValueError(f"Official HARTH schema differs for {subject}; metadata-only amendment required")
    kept = []
    previous_tick = None
    run = 0
    gap = True
    raw_rows = 0
    ties = 0
    full_gaps = 0
    for raw_row, fields in enumerate(reader):
        if None in fields or any(fields[name] is None for name in header):
            raise ValueError(f"Official HARTH row width differs for {subject}/row{raw_row}")
        raw_rows += 1
        tick = timestamp_us(fields["timestamp"])
        if previous_tick is not None:
            if tick < previous_tick:
                raise ValueError(f"Acquisition timestamps regress in {subject}/row{raw_row}")
            ties += tick == previous_tick
            if tick - previous_tick > 1_000_000:
                gap = True
                full_gaps += 1
        previous_tick = tick
        if raw_row % SPEC["acquisition_row_stride"]:
            continue
        if gap:
            run += 1
            gap = False
        feature = np.asarray([float(fields[name]) for name in FIELDS[1:7]], dtype=np.float64)
        if not np.isfinite(feature).all():
            raise ValueError(f"Nonfinite retained sensor readings in {subject}/row{raw_row}; no imputation")
        code = int(fields["label"])
        if code not in CLASS_INDEX:
            raise ValueError(f"Undeclared activity code in {subject}/row{raw_row}; no class substitution")
        kept.append(dict(raw=feature, y=CLASS_INDEX[code], raw_row=raw_row, tick=tick,
            documented_timestamp=fields["timestamp"], session=f"{subject}/run{run:04d}",
            recording=subject, origin=f"{NAME}/{subject}/raw{raw_row:09d}"))
    if not kept:
        raise ValueError(f"Empty declared participant {subject}; no substitution")
    return kept, dict(raw_rows=raw_rows, retained_rows=len(kept), equal_adjacent_timestamps=int(ties),
        raw_timestamp_gaps_above_one_second=full_gaps, history_runs=run,
        header_fields=list(header), header_field_count=len(header),
        scientific_header_fields=list(FIELDS), ignored_export_columns=list(ignored_export_columns),
        first_raw_row=kept[0]["raw_row"], last_raw_row=kept[-1]["raw_row"],
        first_timestamp=kept[0]["documented_timestamp"], last_timestamp=kept[-1]["documented_timestamp"])


def process(archive_path, out, authorization_path):
    auth = verify_authorization(authorization_path)
    out = Path(out)
    if (out / "cache_metadata.json").exists():
        raise ValueError("Prepared cache already exists; verify/load it instead")
    rows, audits, raw_hashes = [], {}, {}
    with zipfile.ZipFile(archive_path) as archive:
        subjects = []
        for name in archive.namelist():
            match = re.fullmatch(r"S(\d+)\.csv", Path(name).name)
            if match:
                subjects.append((int(match[1]), Path(name).stem, name))
        subjects.sort()
        if len(subjects) != 22 or len({s[0] for s in subjects}) != 22:
            raise ValueError("Official archive must contain22unique HARTH subject CSVs; no replacement")
        for rank, (number, subject, name) in enumerate(subjects):
            # Hash and parse in separate streaming passes; no full raw-body allocation.
            digest = hashlib.sha256()
            with archive.open(name) as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            raw_hashes[name] = digest.hexdigest()
            with archive.open(name) as handle:
                kept, audits[subject] = parse_recording(io.TextIOWrapper(handle, encoding="utf-8-sig"), subject)
            for row in kept:
                row.update(subject_number=number, subject_rank=rank+1,
                    partition=0 if rank < 8 else 1 if rank < 12 else 2)
            rows.extend(kept)
    arrays = dict(
        raw=np.stack([r["raw"] for r in rows]), y=np.asarray([r["y"] for r in rows], dtype=np.int64),
        subject=np.asarray([r["subject_number"] for r in rows], dtype=np.int64),
        subject_rank=np.asarray([r["subject_rank"] for r in rows], dtype=np.int64),
        partition=np.asarray([r["partition"] for r in rows], dtype=np.int64),
        timestamp=np.asarray([r["origin"] for r in rows]), origin=np.asarray([r["origin"] for r in rows]),
        documented_timestamp=np.asarray([r["documented_timestamp"] for r in rows]),
        recording=np.asarray([r["recording"] for r in rows]), session=np.asarray([r["session"] for r in rows]),
        anchor_tick=np.asarray([r["tick"] for r in rows], dtype=np.int64),
        anchor_raw_row=np.asarray([r["raw_row"] for r in rows], dtype=np.int64))
    arrays["source_raw_rows"] = np.repeat(arrays["anchor_raw_row"][:, None], 2, axis=1)
    arrays["source_ticks"] = np.repeat(arrays["anchor_tick"][:, None], 2, axis=1)
    arrays["source_age_seconds"] = np.zeros((len(rows), 2), dtype=np.float64)
    ntrain, ncal = int(np.sum(arrays["partition"] == 0)), int(np.sum(arrays["partition"] <= 1))
    if not 0 < ntrain < ncal < len(rows):
        raise ValueError("A predeclared subject partition is empty")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cached_dataset.npz"
    np.savez_compressed(cache, **arrays)
    roster = [s[1] for s in subjects]
    meta = dict(name=NAME, collection="UCI HARTH779", n=len(rows),
        raw_rows=sum(a["raw_rows"] for a in audits.values()), archive_sha256=sha(archive_path),
        cache_sha256=sha(cache), consumed_raw_hashes=raw_hashes,
        acquisition_freeze_sha256=sha(authorization_path), algorithm_freeze_sha256=auth["algorithm_freeze_sha256"],
        physical_protocol_sha256=sha(ROOT/"protocol.json"), adapter_sha256=sha(__file__),
        participants=roster, fit_participants=roster[:8], calibration_participants=roster[8:12], test_participants=roster[12:],
        subject_id_map={str(number): subject for number, subject, _ in subjects},
        class_codes=list(CLASS_CODES), class_names=list(CLASS_NAMES), derivation=SPEC,
        split_boundaries=dict(ntrain=ntrain, ncal=ncal), recordings=audits,
        ordering="numeric original subject-ID, then unchanged acquisition-row order",
        interpretation="one previously untouched physical collection; ten held-out participants, separate causal chains")
    (out/"cache_metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n")
    return meta


def load_data(root):
    root = Path(root)
    meta = json.loads((root/"cache_metadata.json").read_text())
    cache = root/"cached_dataset.npz"
    if sha(cache) != meta["cache_sha256"] or sha(__file__) != meta["adapter_sha256"]:
        raise ValueError("Frozen adapter/cache differs")
    if sha(ROOT/"protocol.json") != meta["physical_protocol_sha256"]:
        raise ValueError("Frozen physical protocol differs")
    with np.load(cache, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if any(len(value) != meta["n"] for value in data.values()) or data["raw"].shape != (meta["n"], 6):
        raise ValueError("Physical cache schema/lengths differ")
    if not np.isfinite(data["raw"]).all():
        raise ValueError("Nonfinite physical source features")
    if not np.array_equal(data["source_ticks"], np.repeat(data["anchor_tick"][:, None], 2, axis=1)):
        raise ValueError("Source timestamps do not equal their documented anchor")
    data["timestamp"], data["origin"] = data["timestamp"].tolist(), data["origin"].tolist()
    data.update(n=meta["n"], raw_rows=meta["raw_rows"], hashes=meta["consumed_raw_hashes"], metadata=meta)
    return data


def boundary_lagged(raw, mean, scale, sessions):
    z = np.clip((raw - mean) / scale, -8, 8)
    x = np.empty((len(z), 3*z.shape[1]+1), dtype=np.float64)
    first = 0
    for i in range(len(z)):
        if i == 0 or sessions[i] != sessions[i-1]:
            first = i
        x[i, :-1] = np.stack([z[max(first, i-lag)] for lag in range(3)], axis=1).reshape(-1)
        x[i, -1] = 1
    return x


def make_prefix(data, cfg, engine):
    """Common source models: fit subjects only, chronological fit/audit cuts."""
    meta = data["metadata"]
    ntrain, ncal = meta["split_boundaries"]["ntrain"], meta["split_boundaries"]["ncal"]
    trainidx, auditidx = [], []
    for subject in meta["fit_participants"]:
        ids = np.flatnonzero(data["recording"] == subject)
        cut = int(np.floor(.7*len(ids)))
        if not 0 < cut < len(ids):
            raise ValueError("Empty chronological fit/audit partition")
        trainidx.extend(ids[:cut])
        auditidx.extend(ids[cut:])
    trainidx, auditidx = np.asarray(trainidx), np.asarray(auditidx)
    mean = data["raw"][trainidx].mean(0)
    scale = np.maximum(data["raw"][trainidx].std(0), .05)
    x = boundary_lagged(data["raw"], mean, scale, data["session"])
    features = [np.r_[[j for v in group for j in range(3*v, 3*v+3)], 18] for group in GROUPS]+[np.arange(19)]
    models = [engine.gradient(np.zeros((len(ids),12)), x[trainidx][:,ids], data["y"][trainidx],
              cfg["initial_steps"], cfg, 12) for ids in features]
    pa = np.stack([engine.softmax(x[auditidx][:,features[s]]@models[s]) for s in range(2)])
    prior = []
    # Prior blocks never cross participant or timestamp-gap histories.
    for session in dict.fromkeys(data["session"][auditidx].tolist()):
        positions = np.flatnonzero(data["session"][auditidx] == session)
        for left in range(0, len(positions), cfg["window"]):
            pp = positions[left:left+cfg["window"]]
            ids = auditidx[pp]
            prior.append(dict(x=x[ids], y=data["y"][ids], p=pa[:,pp], context=engine.context(x[ids],GROUPS),
                mask=np.ones(2,bool), origin=-1))
    q = 1/(np.mean(np.sum((pa-np.eye(12)[data["y"][auditidx]][None])**2,axis=2),axis=1)+.05)
    split = dict(prefix_rows=ntrain, training_rows=len(trainidx), audit_rows=len(auditidx),
        calibration_rows=ncal-ntrain, test_rows=len(x)-ncal, fit_people=meta["fit_participants"],
        calibration_people=meta["calibration_participants"], test_people=meta["test_participants"],
        feature_history_reset_indices=np.flatnonzero(np.r_[True,data["session"][1:]!=data["session"][:-1]]).tolist())
    pre = dict(models=models, prior=prior, q=q, mean=mean, scale=scale, features=features, classes=12,
        m=2, groups=GROUPS, train_x=x[trainidx], train_y=data["y"][trainidx],
        cal_x=x[ntrain:ncal], cal_y=data["y"][ntrain:ncal], cal_timestamp=data["timestamp"][ntrain:ncal],
        test_x=x[ncal:], test_y=data["y"][ncal:], test_timestamp=data["timestamp"][ncal:], split=split,
        recording_streams=[])
    for subject in meta["participants"][8:]:
        for session in dict.fromkeys(data["session"][data["recording"] == subject].tolist()):
            ids = np.flatnonzero(data["session"] == session)
            pre["recording_streams"].append(dict(recording=session, person=subject,
                partition="calibration" if subject in meta["calibration_participants"] else "test",
                x=x[ids], y=data["y"][ids], timestamp=[data["timestamp"][int(i)] for i in ids],
                session=data["session"][ids].copy()))
    return pre


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    meta = process(args.archive, args.output, args.authorization)
    print(json.dumps({k:meta[k] for k in ("name","n","raw_rows","fit_participants","calibration_participants","test_participants")},indent=2))
