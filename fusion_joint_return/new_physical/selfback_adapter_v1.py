"""Unexecuted selfBACK521 physical-pair adapter; no acquisition functions.

Read raw original w/t panels only after the parent has saved a complete
algorithm/protocol freeze. Original calendar timestamps define pairing and
chronology. Folder annotations are targets, never feature inputs or resets.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import zipfile
import numpy as np

NAME = 'selfback521'
FIT_N, CAL_N, TEST_N = 12, 8, 13
MIN_SAMPLES = 50
NS_SECOND = 1_000_000_000
FEATURE_NAMES = [f'{axis}_{stat}' for axis in ('x', 'y', 'z')
                 for stat in ('mean', 'std', 'min', 'max')]
GROUPS = [list(range(12)), list(range(12, 24))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(archive):
    """Filename-only schema: no signal payload or label outcome read here."""
    found = {}
    for member in archive.namelist():
        parts = Path(member).parts
        if '__MACOSX' in parts or Path(member).name.startswith('._') or not member.lower().endswith('.csv'):
            continue
        panel_positions = [i for i, p in enumerate(parts) if p.lower() in ('w', 't')]
        if not panel_positions:
            continue
        assert len(panel_positions) == 1, member
        j = panel_positions[0]
        assert len(parts) == j + 3, ('Expected panel/activity/person.csv schema', member)
        panel, activity = parts[j].lower(), parts[j + 1]
        stem = Path(member).stem
        assert re.fullmatch(r'[0-9]+', stem), ('Non-numeric participant ID', member)
        key = (int(stem), activity, panel)
        assert key not in found, ('Duplicate filename key', key)
        found[key] = member
    ids = sorted({p for p, _, _ in found})
    activities = sorted({a for _, a, _ in found})
    assert len(ids) == FIT_N + CAL_N + TEST_N, ('Participant count', ids)
    assert len(activities) == 9, ('Activity-folder count', activities)
    expected = {(p, a, s) for p in ids for a in activities for s in ('w', 't')}
    assert set(found) == expected, ('Incomplete original two-panel participant/activity matrix',
                                    sorted(expected - set(found)), sorted(set(found) - expected))
    return found, ids, activities


def parse_sensor(payload, member):
    """Parse supplied timestamp,x,y,z CSV with optional CSV index column."""
    text = payload.decode('utf-8-sig')
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
    except csv.Error:
        dialect = csv.excel
    records, header_seen = [], False
    for line_n, row in enumerate(csv.reader(io.StringIO(text), dialect)):
        if not row or all(not x.strip() for x in row):
            continue
        row = [x.strip() for x in row]
        if len(row) == 5:
            # Pandas CSV index is permitted but never used as a timestamp.
            row = row[1:]
        assert len(row) == 4, ('Unexpected raw panel schema', member, line_n, len(row))
        try:
            value = np.asarray([float(x) for x in row[1:]], dtype=float)
        except ValueError:
            assert not records and not header_seen and all(
                a.lower() in ('x', 'y', 'z', 'acc_x', 'acc_y', 'acc_z') for a in row[1:]
            ), ('Malformed sensor header', member, line_n)
            header_seen = True
            continue
        assert np.isfinite(value).all(), ('Non-finite physical values', member, line_n)
        # Require an absolute calendar timestamp. A numeric per-device clock
        # must not be silently reinterpreted or aligned by row number.
        assert re.search(r'[0-9]{4}[-/][0-9]{2}[-/][0-9]{2}', row[0]), (
            'Calendar timestamps are required for original physical pairing', member, line_n)
        stamp = np.datetime64(row[0].replace('/', '-'), 'ns')
        assert not np.isnat(stamp), ('Invalid timestamp', member, line_n)
        records.append((int(stamp.astype(np.int64)), value))
    assert records, ('Empty physical sensor member', member)
    return records


def one_panel_bins(archive, member_map, person, activities, panel, raw_hashes):
    bins = {}
    raw_rows = 0
    for label, activity in enumerate(activities):
        member = member_map[(person, activity, panel)]
        payload = archive.read(member)
        raw_hashes[member] = hashlib.sha256(payload).hexdigest()
        for stamp, values in parse_sensor(payload, member):
            raw_rows += 1
            bucket = bins.setdefault(stamp // NS_SECOND, [])
            bucket.append((stamp, label, values))
    output = {}
    discarded = dict(sparse_bins=0, mixed_annotation_bins=0, identical_duplicate_rows=0)
    for second, records in bins.items():
        labels = {r[1] for r in records}
        if len(labels) != 1:
            discarded['mixed_annotation_bins'] += 1
            continue
        by_stamp = {}
        for stamp, label, value in records:
            if stamp in by_stamp:
                assert np.array_equal(by_stamp[stamp][1], value), (
                    'Conflicting same-time physical readings', person, panel, stamp)
                discarded['identical_duplicate_rows'] += 1
            else:
                by_stamp[stamp] = (label, value)
        if len(by_stamp) < MIN_SAMPLES:
            discarded['sparse_bins'] += 1
            continue
        v = np.stack([by_stamp[k][1] for k in sorted(by_stamp)])
        moments = np.column_stack((v.mean(0), v.std(0), v.min(0), v.max(0))).reshape(-1)
        output[second] = (next(iter(labels)), moments, len(v))
    return output, dict(raw_rows=raw_rows, eligible_bins=len(output), **discarded)


def process(path, out, freeze_path):
    """Process after explicit source-bound acquisition authorization exists."""
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    assert freeze.get('raw_access_authorized') is True, 'Root freeze must authorize raw access'
    assert freeze.get('dataset') == NAME, 'Freeze dataset mismatch'
    assert freeze.get('algorithm_source_hashes'), 'A complete algorithm source closure is required'
    for source, expected in freeze['algorithm_source_hashes'].items():
        assert sha(source) == expected, ('Frozen algorithm source mismatch', source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'cached_dataset.npz').exists(), 'Never overwrite a completed cache'
    arrays = {k: [] for k in ('raw', 'y', 'person', 'partition', 'timestamp_ns', 'run', 'counts')}
    raw_hashes, stats = {}, {}
    with zipfile.ZipFile(path) as archive:
        members, ids, activities = inventory(archive)
        ranks = {p: i for i, p in enumerate(ids)}
        for person in ids:
            panels, panel_stats = [], []
            for panel in ('w', 't'):
                b, s = one_panel_bins(archive, members, person, activities, panel, raw_hashes)
                panels.append(b)
                panel_stats.append(s)
            rank = ranks[person]
            partition = 'fit' if rank < FIT_N else 'calibration' if rank < FIT_N + CAL_N else 'test'
            paired, annotation_mismatch, previous, run = 0, 0, None, -1
            for second in sorted(set(panels[0]) & set(panels[1])):
                wrist, thigh = panels[0][second], panels[1][second]
                if wrist[0] != thigh[0]:
                    annotation_mismatch += 1
                    continue
                if previous is None or second - previous > 1:
                    run += 1
                previous = second
                arrays['raw'].append(np.r_[wrist[1], thigh[1]])
                arrays['y'].append(wrist[0])
                arrays['person'].append(person)
                arrays['partition'].append(partition)
                arrays['timestamp_ns'].append(second * NS_SECOND)
                arrays['run'].append(run)
                arrays['counts'].append([wrist[2], thigh[2]])
                paired += 1
            assert paired > 0, ('No actual timestamp-paired bins for a retained participant', person)
            stats[str(person)] = dict(rank=rank + 1, partition=partition, paired_bins=paired,
                runs=run + 1, annotation_mismatch_bins=annotation_mismatch, panels=panel_stats)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    assert arrays['raw'].shape == (len(arrays['y']), 24)
    np.savez_compressed(out / 'cached_dataset.npz', **arrays)
    meta = dict(name=NAME, n=len(arrays['y']), source_panels=['wrist', 'thigh'], groups=GROUPS,
        participant_ids=ids, fit_ids=ids[:FIT_N], calibration_ids=ids[FIT_N:FIT_N + CAL_N],
        test_ids=ids[FIT_N + CAL_N:], activities=activities, physical_sites='not established by metadata',
        physical_test_participants=TEST_N, panel_feature_names=FEATURE_NAMES, statistics=stats,
        recorded_input_rate_Hz=100, retained_bin_duration_seconds=1, minimum_actual_panel_samples=MIN_SAMPLES,
        pairing='Same original absolute calendar-second bin, never row-index or normalized-activity-time pairing',
        chronology='Original absolute participant timestamps; observable gaps retained',
        archive_sha256=sha(path), raw_hashes=raw_hashes,
        cache_sha256=sha(out / 'cached_dataset.npz'), freeze_sha256=sha(freeze_path))
    (out / 'cache_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def prepared(root, engine, cfg):
    """Common fit-only classifier preparation; test values never fit scales."""
    root = Path(root)
    meta = json.loads((root / 'cache_metadata.json').read_text())
    assert sha(root / 'cached_dataset.npz') == meta['cache_sha256']
    with np.load(root / 'cached_dataset.npz') as f:
        d = {k: f[k].copy() for k in f.files}
    fit = np.flatnonzero(d['partition'] == 'fit')
    reserve = np.array([int(hashlib.sha256(
        f"{int(d['person'][i])}:{int(d['timestamp_ns'][i])}".encode()).hexdigest()[:8], 16) % 10 < 3
        for i in fit])
    train, audit = fit[~reserve], fit[reserve]
    assert set(d['y'][train]) == set(range(9))
    mean = d['raw'][train].mean(0)
    scale = np.maximum(d['raw'][train].std(0), .05)
    z = np.clip((d['raw'] - mean) / scale, -8, 8)
    x, first = np.ones((len(z), 73)), 0
    for i in range(len(z)):
        if i == 0 or d['person'][i] != d['person'][i - 1] or d['run'][i] != d['run'][i - 1]:
            first = i
        x[i, :-1] = np.stack([z[max(first, i - lag)] for lag in range(3)], axis=1).reshape(-1)
    features = [np.r_[[j for v in group for j in range(3 * v, 3 * v + 3)], 72]
                for group in GROUPS] + [np.arange(73)]
    models = [engine.gradient(np.zeros((len(f), 9)), x[train][:, f], d['y'][train],
                             cfg['initial_steps'], cfg, 9) for f in features]
    pa = np.stack([engine.softmax(x[audit][:, features[s]] @ models[s]) for s in range(2)])
    prior = []
    for left in range(0, len(audit), cfg['window']):
        pp = np.arange(left, min(left + cfg['window'], len(audit)))
        ii = audit[pp]
        prior.append(dict(x=x[ii], y=d['y'][ii], p=pa[:, pp],
            context=engine.context(x[ii], GROUPS), mask=np.ones(2, bool), origin=-1))
    q = 1 / (np.mean(np.sum((pa - np.eye(9)[d['y'][audit]][None]) ** 2, axis=2), axis=1) + .05)
    pre = dict(models=models, prior=prior, q=q, mean=mean, scale=scale, features=features,
        classes=9, m=2, groups=GROUPS, train_x=x[train], train_y=d['y'][train],
        recording_streams=[], split=dict(training_rows=len(train), audit_rows=len(audit),
            fit=meta['fit_ids'], calibration=meta['calibration_ids'], test=meta['test_ids'],
            physical_test_participants=TEST_N))
    for person in meta['calibration_ids'] + meta['test_ids']:
        ii = np.flatnonzero(d['person'] == person)
        assert np.all(np.diff(d['timestamp_ns'][ii]) > 0)
        pre['recording_streams'].append(dict(recording=f'p{person:03d}', person=person,
            batch=f'p{person:03d}', partition=str(d['partition'][ii[0]]),
            x=x[ii], y=d['y'][ii], timestamp_ns=d['timestamp_ns'][ii].tolist(),
            observable_run=d['run'][ii].tolist()))
    return pre, meta
