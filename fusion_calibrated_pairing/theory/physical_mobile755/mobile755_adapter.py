"""Metadata-declared ordinal mobile755 adapter; no acquisition facility.

The two source groups are accelerometer and gyroscope modalities on a
mobile phone. No participant/site IDs or valid physical-clock mapping
are asserted. All CSV records and native binary activity codes remain.
"""
from pathlib import Path
import csv,hashlib,io,json,zipfile
import numpy as np

NAME='mobile755'
CSV_NAME='accelerometer_gyro_mobile_phone_dataset.csv'
SENSOR_COLUMNS=('accX','accY','accZ','gyroX','gyroY','gyroZ')
COLUMNS=SENSOR_COLUMNS+('timestamp','Activity')
GROUPS=[list(range(3)),list(range(3,6))]
PARTITIONS=('fit','state_fit','score_cal','selection','test')
FRACTIONS=(0.,.3,.4,.5,.6,1.)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def partition_vector(n):
    boundaries=[int(np.floor(n*f)) for f in FRACTIONS]
    assert boundaries[0]==0 and boundaries[-1]==n and all(b>a for a,b in zip(boundaries,boundaries[1:]))
    out=np.empty(n,dtype='<U10')
    for p,a,b in zip(PARTITIONS,boundaries,boundaries[1:]):out[a:b]=p
    return out,boundaries

def csv_payload(archive):
    names=[name for name in archive.namelist() if Path(name).name==CSV_NAME and '__MACOSX' not in Path(name).parts]
    assert len(names)==1,('Exactly one native CSV is required',names)
    return names[0],archive.read(names[0])

def decode(payload):
    reader=csv.DictReader(io.StringIO(payload.decode('utf-8-sig',errors='strict')))
    assert reader.fieldnames is not None and len(reader.fieldnames)==len(COLUMNS) and set(reader.fieldnames)==set(COLUMNS),('Unexpected official CSV header',reader.fieldnames)
    sensors=[];codes=[];stamps=[]
    for i,row in enumerate(reader):
        assert set(row)==set(COLUMNS) and None not in row,('Malformed complete row',i)
        values=[float(row[name]) if row[name].strip() else float('nan') for name in SENSOR_COLUMNS]
        assert not np.isinf(values).any(),('Infinite sensor value',i)
        target=float(row['Activity']);assert np.isfinite(target) and target==round(target),('Native integer target required',i)
        sensors.append(values);codes.append(int(target));stamps.append(row['timestamp'])
    assert len(sensors)>=1280,'All five fixed ordinal partitions must be nonempty'
    raw=np.asarray(sensors,float);native=np.asarray(codes,int)
    vocabulary=tuple(sorted(set(codes)))
    assert len(vocabulary)==2,('Official standing/walking target requires exactly two native integer codes',vocabulary)
    # Vocabulary validation is a schema operation, not a numerical target
    # fit, class-removal choice, or tuning outcome. All native rows remain.
    y=np.asarray([vocabulary.index(code) for code in codes],int)
    part,boundaries=partition_vector(len(y))
    return dict(raw=raw,y=y,native_activity=native,timestamp=np.asarray(stamps),
                row_index=np.arange(len(y)),partition=part,run=np.zeros(len(y),int)),vocabulary,boundaries

def process(archive_path,out,freeze_path):
    freeze=json.loads(Path(freeze_path).read_text());assert freeze.get('raw_access_authorized') is True and freeze['dataset']==NAME
    for path,want in freeze['algorithm_source_hashes'].items():assert sha(path)==want,('Frozen numerical source changed',path)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);assert not (out/'cached_dataset.npz').exists()
    with zipfile.ZipFile(archive_path) as archive:
        member,payload=csv_payload(archive);arrays,vocabulary,boundaries=decode(payload)
    np.savez_compressed(out/'cached_dataset.npz',**arrays)
    statistics=[]
    for part,a,b in zip(PARTITIONS,boundaries,boundaries[1:]):
        statistics.append(dict(partition=part,record_start=a,record_stop=b,records=b-a,
            native_class_counts=np.bincount(arrays['y'][a:b],minlength=2).tolist(),
            missing_cells=int(np.isnan(arrays['raw'][a:b]).sum())))
    metadata=dict(name=NAME,uci_id=755,classes=2,native_target_codes=vocabulary,
        groups=GROUPS,source_names=['accelerometer','gyroscope'],records=len(arrays['y']),
        advertised_records=31991,metadata_cardinality_matches=len(arrays['y'])==31991,
        partition_fractions=FRACTIONS,partition_boundaries=boundaries,statistics=statistics,
        raw_order='Original native CSV record order; no sorting, row filtering, interpolation, or resampling',
        timestamp_scope='Categorical native timestamp retained for provenance only; units/participant/site mapping not validated; no physical lease duration claimed',
        physical_collection='King Saud University mobile-phone inertial collection,2022 per UCI metadata',
        independent_test_participants=None,independent_test_sites=None,clock_units=None,
        archive_sha256=sha(archive_path),raw_hashes={member:hashlib.sha256(payload).hexdigest()},
        cache_sha256=sha(out/'cached_dataset.npz'),freeze_sha256=sha(freeze_path))
    (out/'cache_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata

def prepared(root,engine,cfg):
    root=Path(root);meta=json.loads((root/'cache_metadata.json').read_text());assert sha(root/'cached_dataset.npz')==meta['cache_sha256']
    with np.load(root/'cached_dataset.npz') as file:d={k:file[k].copy() for k in file.files}
    fit=np.flatnonzero(d['partition']=='fit')
    reserved=np.asarray([int(hashlib.sha256(f'mobile755:{int(i)}'.encode()).hexdigest()[:8],16)%10<3 for i in d['row_index'][fit]])
    train,audit=fit[~reserved],fit[reserved];assert len(train)>0 and len(audit)>0
    missing=np.isnan(d['raw']);median=np.zeros(6)
    for j in range(6):
        finite=d['raw'][train,j][~missing[train,j]];median[j]=np.median(finite) if len(finite) else 0.
    observed=np.concatenate((np.where(missing,median[None,:],d['raw']),missing.astype(float)),axis=1)
    mean=observed[train].mean(0);scale=np.maximum(observed[train].std(0),.05)
    z=np.clip((observed-mean)/scale,-8,8);x=np.ones((len(z),37));first=0
    for i in range(len(z)):
        if i==0 or d['partition'][i]!=d['partition'][i-1]:first=i
        x[i,:-1]=np.stack([z[max(first,i-lag)] for lag in range(3)],axis=1).reshape(-1)
    allgroups=[g+[j+6 for j in g] for g in GROUPS]
    features=[np.r_[[j for v in g for j in range(3*v,3*v+3)],36] for g in allgroups]+[np.arange(37)]
    train_x=x[train];train_y=d['y'][train];audit_x=x[audit];audit_y=d['y'][audit]
    models=[engine.gradient(np.zeros((len(ix),2)),train_x[:,ix],train_y,cfg['initial_steps'],cfg,2) for ix in features]
    pa=np.stack([engine.softmax(audit_x[:,features[s]]@models[s]) for s in range(2)])
    q=1/(np.mean(np.sum((pa-np.eye(2)[audit_y][None])**2,axis=2),axis=1)+.05)
    prior=[]
    for start in range(0,len(audit),cfg['window']):
        ii=audit[start:start+cfg['window']]
        prior.append(dict(x=x[ii],y=d['y'][ii],p=pa[:,start:start+cfg['window']],
            context=engine.context(x[ii],allgroups),mask=np.ones(2,bool),origin=-1))
    pre=dict(models=models,prior=prior,q=q,median=median,mean=mean,scale=scale,
        features=features,classes=2,m=2,groups=allgroups,train_x=train_x,train_y=train_y,
        audit_x=audit_x,audit_y=audit_y,recording_streams=[],
        split=dict(training_rows=len(train),audit_rows=len(audit),fit_fraction=.3,
            state_fit_fraction=.1,score_cal_fraction=.1,selection_fraction=.1,test_fraction=.4,
            split_unit='Original contiguous ordinal records',participant_disjoint=False,site_disjoint=False))
    source_present=np.stack([~missing[:,g].all(1) for g in GROUPS],axis=1)
    for partition in PARTITIONS[1:]:
        ii=np.flatnonzero(d['partition']==partition);assert np.array_equal(ii,np.arange(ii[0],ii[-1]+1))
        block=slice(int(ii[0]),int(ii[-1]+1))
        pre['recording_streams'].append(dict(recording=f'mobile755-{partition}',person='collection755',
            batch='collection755',session=partition,partition=partition,x=x[block],y=d['y'][block],
            timestamp=[f'mobile755:ordinal{int(v)}' for v in d['row_index'][block]],
            original_timestamp=d['timestamp'][block],physical_source_present=source_present[block]))
    return pre,meta
