"""UCI308 metadata-fixed causal physical-panel adapter, before acquisition.

Concentrations identify experiment gas classes; they never enter features.
Every experiment is a separate service, lag and history chain.
"""
import csv, gzip, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np

GROUPS = [list(range(i, i+4)) for i in (0, 4, 8, 12)]
FIT = ('day-1-morning', 'day-2-morning')
CAL = ('day-2-afternoon',)
TEST = ('day-3-morning', 'day-4-afternoon')
BATCH_COUNTS = dict(zip(FIT+CAL+TEST, (19, 10, 10, 11, 8)))
STRIDE = 5
RATE = 25
POINTS = 7500

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def label(acetone, ethanol):
    assert 0 <= acetone <= 1 and 0 <= ethanol <= 1
    return (1 if acetone > 0 else 0) + (2 if ethanol > 0 else 0)

def process(path, out):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    assert not (out/'cache_metadata.json').exists()
    points = np.arange(0, POINTS, STRIDE)
    records = {}
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if Path(n).name == 'rawdata.csv.gz']
        assert len(names) == 1, names
        member = names[0]
        compressed = archive.read(member)
        payload = gzip.decompress(compressed)
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    needed = {'exp', 'batch', 'ace_conc', 'eth_conc', 'sensor', 'sample'}
    assert needed <= set(reader.fieldnames), reader.fieldnames[:12]
    columns = [f'dR_t{i+1}' for i in range(POINTS)]
    assert set(columns) <= set(reader.fieldnames)
    for rawrow, row in enumerate(reader, 2):
        exp = int(row['exp']); sample = int(row['sample']); sensor = int(row['sensor'])
        batch = row['batch']; ace = float(row['ace_conc']); eth = float(row['eth_conc'])
        assert batch in BATCH_COUNTS and 1 <= sensor <= 16 and 1 <= sample <= 58
        identity = (sample, batch, ace, eth)
        record = records.setdefault(exp, dict(identity=identity, sensors={}, rows={}))
        assert record['identity'] == identity and sensor not in record['sensors']
        series = np.array([float(row[k]) for k in columns])
        assert np.isfinite(series).all()
        record['sensors'][sensor] = series[points]
        record['rows'][sensor] = rawrow
    assert len(records) == 58 and len({r['identity'][0] for r in records.values()}) == 58
    arrays = {k: [] for k in ('raw', 'y', 'person', 'session', 'timestamp', 'batch', 'sample_index', 'source_raw_rows')}
    stats = {}
    for batch in FIT+CAL+TEST:
        exps = sorted(e for e, r in records.items() if r['identity'][1] == batch)
        assert len(exps) == BATCH_COUNTS[batch], (batch, len(exps))
        for exp in exps:
            r = records[exp]; sample, _, ace, eth = r['identity']
            assert set(r['sensors']) == set(range(1,17))
            recording = f'flow308/exp{exp:03d}'
            x = np.column_stack([r['sensors'][s] for s in range(1,17)])
            n = len(points)
            arrays['raw'].extend(x)
            arrays['y'].extend([label(ace, eth)]*n)
            arrays['person'].extend([exp]*n)
            arrays['session'].extend([recording]*n)
            arrays['batch'].extend([batch]*n)
            arrays['sample_index'].extend(points+1)
            arrays['source_raw_rows'].extend([[r['rows'][s] for s in range(1,17)]]*n)
            arrays['timestamp'].extend([f'{recording}/point{p+1:05d}/tick{p/RATE:.2f}' for p in points])
            stats[recording] = dict(experiment=exp, official_sample=sample, batch=batch,
                raw_points=POINTS, retained=n, sensor_ids=list(range(1,17)),
                source_raw_rows=[r['rows'][s] for s in range(1,17)],
                first_point=int(points[0]+1), last_point=int(points[-1]+1),
                first_relative_seconds=float(points[0]/RATE), last_relative_seconds=float(points[-1]/RATE))
    arrays = {k: np.array(v) for k,v in arrays.items()}
    assert arrays['raw'].shape == (87000,16)
    np.savez_compressed(out/'cached_dataset.npz', **arrays)
    meta = dict(name='flow308', n=len(arrays['y']), experiments=len(records), raw_sensor_rows=16*58,
        points_per_sensor=POINTS, acquisition_rate_hz=RATE, stride=STRIDE, retained_rate_hz=RATE/STRIDE,
        fit_batches=FIT, calibration_batches=CAL, test_batches=TEST, batch_counts=BATCH_COUNTS,
        groups=GROUPS, physical_sensor_ids=[[v+1 for v in g] for g in GROUPS],
        task='experiment gas identity throughout exposure and recovery, not instantaneous concentration',
        classes={'0':'air','1':'acetone','2':'ethanol','3':'mixture'}, file_statistics=stats,
        raw_hashes={member:hashlib.sha256(compressed).hexdigest(),member+'::decompressed':hashlib.sha256(payload).hexdigest()},
        archive_sha256=sha(path), cache_sha256=sha(out/'cached_dataset.npz'),
        provenance='raw CSV sensor row and documented sample column retained for every panel value')
    (out/'cache_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta

def prepared(root, engine, cfg):
    root = Path(root)
    meta = json.loads((root/'cache_metadata.json').read_text())
    assert sha(root/'cached_dataset.npz') == meta['cache_sha256']
    with np.load(root/'cached_dataset.npz') as f:
        d = {k:f[k].copy() for k in f.files}
    train=[]; audit=[]
    for recording in dict.fromkeys(d['session'].tolist()):
        ids=np.flatnonzero(d['session']==recording)
        if d['batch'][ids[0]] not in FIT: continue
        cut=int(.7*len(ids)); assert 0 < cut < len(ids)
        train.extend(ids[:cut]); audit.extend(ids[cut:])
    train=np.array(train); audit=np.array(audit)
    mean=d['raw'][train].mean(0); scale=np.maximum(d['raw'][train].std(0),.05)
    z=np.clip((d['raw']-mean)/scale,-8,8)
    x=np.ones((len(z),49)); first=0
    for i in range(len(z)):
        if i==0 or d['session'][i]!=d['session'][i-1]: first=i
        x[i,:-1]=np.stack([z[max(first,i-lag)] for lag in range(3)],axis=1).reshape(-1)
    features=[np.r_[[j for v in group for j in range(3*v,3*v+3)],48] for group in GROUPS]+[np.arange(49)]
    models=[engine.gradient(np.zeros((len(ids),4)),x[train][:,ids],d['y'][train],cfg['initial_steps'],cfg,4) for ids in features]
    pa=np.stack([engine.softmax(x[audit][:,features[s]]@models[s]) for s in range(4)])
    prior=[]
    for session in dict.fromkeys(d['session'][audit].tolist()):
        positions=np.flatnonzero(d['session'][audit]==session)
        for left in range(0,len(positions),cfg['window']):
            pp=positions[left:left+cfg['window']]; ii=audit[pp]
            prior.append(dict(x=x[ii],y=d['y'][ii],p=pa[:,pp],context=engine.context(x[ii],GROUPS),mask=np.ones(4,bool),origin=-1))
    q=1/(np.mean(np.sum((pa-np.eye(4)[d['y'][audit]][None])**2,axis=2),axis=1)+.05)
    pre=dict(models=models,prior=prior,q=q,mean=mean,scale=scale,features=features,classes=4,m=4,groups=GROUPS,
        train_x=x[train],train_y=d['y'][train],recording_streams=[],
        split=dict(training_rows=len(train),audit_rows=len(audit),fit_batches=FIT,calibration_batches=CAL,test_batches=TEST,
            calibration_rows=int(np.isin(d['batch'],CAL).sum()),test_rows=int(np.isin(d['batch'],TEST).sum()),
            fit_experiments=29,calibration_experiments=10,test_experiments=19))
    for session in dict.fromkeys(d['session'].tolist()):
        ids=np.flatnonzero(d['session']==session); batch=str(d['batch'][ids[0]])
        if batch in FIT: continue
        pre['recording_streams'].append(dict(recording=session,person=int(d['person'][ids[0]]),batch=batch,
            partition='calibration' if batch in CAL else 'test',x=x[ids],y=d['y'][ids],timestamp=d['timestamp'][ids].tolist()))
    assert len(pre['recording_streams'])==29
    return pre,meta
