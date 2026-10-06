"""Metadata-first OPPORTUNITY226 adapter V2: official physical grouping repair.

No downloader and no raw observation is accessed until a complete numerical
method/protocol source freeze authorizes it. The native matrix clock already
pairs body/object/ambient sources; no reconstructed interdevice clock is used.
"""
from pathlib import Path
import hashlib,io,json,re,zipfile
import numpy as np
NAME='opportunity226'
SESSIONS=('Drill','ADL1','ADL2','ADL3','ADL4','ADL5')
GROUPS=[list(range(133))+list(range(230,242)),list(range(133,193)),list(range(193,230))]
TARGET_CODES=(0,406516,406517,404516,404517,406520,404520,406505,404505,
              406519,404519,406511,404511,406508,404508,408512,407521,405506)
TARGET_INDEX=249;STRIDE=3
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def partition(person,session):
    if person==1:return 'fit'
    if person in (3,4):return 'test'
    assert person==2
    return 'state_fit' if session in SESSIONS[:2] else 'score_cal' if session in SESSIONS[2:4] else 'selection'
def inventory(archive):
    found={}
    for name in archive.namelist():
        if '__MACOSX' in Path(name).parts or Path(name).name.startswith('._'):continue
        match=re.fullmatch(r'S([1-4])-(Drill|ADL[1-5])\.dat',Path(name).name)
        if not match:continue
        key=(int(match[1]),match[2]);assert key not in found,('Duplicate original session',key)
        found[key]=name
    assert set(found)=={(p,s) for p in range(1,5) for s in SESSIONS},'All original 24 sessions must be retained'
    return found
def metadata_gate(archive):
    """Official legends are checked before opening any .dat sensor payload."""
    found={}
    for name in archive.namelist():
        low=Path(name).name.lower()
        if low in ('column_names.txt','label_legend.txt'):
            assert low not in found,('Duplicate official metadata',low)
            found[low]=(name,archive.read(name))
    assert set(found)=={'column_names.txt','label_legend.txt'},'Official column/label legends required before raw processing'
    columns=found['column_names.txt'][1].decode('utf-8-sig',errors='strict')
    labels=found['label_legend.txt'][1].decode('utf-8-sig',errors='strict')
    # Numbering is one-based in the official legend and zero-based in arrays.
    mapping={int(k):v for k,v in re.findall(r'(?im)^\s*(?:column\s*:?\s*)?(\d+)\s*[:;\t ]+([^\r\n]+)',columns)}
    assert len(mapping)>=250 and 1 in mapping and 250 in mapping,('Unexpected official column legend',len(mapping))
    assert any(word in mapping[1].lower() for word in ('time','millisec')),'First column must be recorded time'
    assert 'ml' in mapping[250].lower() and 'arm' in mapping[250].lower(),'Last column must be native mid-level gesture annotation'
    for code in TARGET_CODES[1:]:assert str(code) in labels,('Native gesture code absent from official legend',code)
    return dict(metadata_hashes={name:hashlib.sha256(blob).hexdigest() for name,blob in found.values()},
                columns={str(k):mapping[k] for k in range(1,251)},target_codes=TARGET_CODES)
def process(archive_path,out,freeze_path):
    freeze=json.loads(Path(freeze_path).read_text());assert freeze.get('raw_access_authorized') is True and freeze['dataset']==NAME
    hashes=freeze.get('algorithm_source_hashes',freeze.get('source_hashes'));assert hashes
    for path,want in hashes.items():assert sha(path)==want,('Numerical source changed',path)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);assert not (out/'cached_dataset.npz').exists()
    arrays={k:[] for k in ('raw','y','person','session','partition','timestamp_ms','row_index','run')};stats=[];raw_hashes={}
    codes={v:i for i,v in enumerate(TARGET_CODES)}
    with zipfile.ZipFile(archive_path) as archive:
        metadata=metadata_gate(archive);members=inventory(archive)
        for person in range(1,5):
            for session in SESSIONS:
                member=members[person,session];payload=archive.read(member);raw_hashes[member]=hashlib.sha256(payload).hexdigest()
                data=np.loadtxt(io.BytesIO(payload));assert data.ndim==2 and data.shape[1]==250,(member,data.shape)
                assert np.isfinite(data[:,0]).all() and np.all(np.diff(data[:,0])>=0),('Invalid native session time',member)
                assert np.isfinite(data[:,TARGET_INDEX]).all() and np.all(data[:,TARGET_INDEX]==np.round(data[:,TARGET_INDEX]))
                assert set(data[:,TARGET_INDEX].astype(int))<=set(TARGET_CODES),('Unknown native gesture label',member)
                ii=np.arange(0,len(data),STRIDE);sample=data[ii];raw=sample[:,1:243]
                assert not np.isinf(raw).any(),'Only native missing NaN, not infinite values, is supported'
                labels=np.array([codes[int(v)] for v in sample[:,TARGET_INDEX]],int)
                # Observable time gaps reset causal lags. Labels never do.
                runs=np.cumsum(np.r_[True,np.diff(sample[:,0])>250.])-1
                n=len(ii);arrays['raw'].append(raw);arrays['y'].append(labels);arrays['person'].append(np.full(n,person))
                arrays['session'].append(np.full(n,session));arrays['partition'].append(np.full(n,partition(person,session)))
                arrays['timestamp_ms'].append(sample[:,0]);arrays['row_index'].append(ii);arrays['run'].append(runs)
                stats.append(dict(person=person,session=session,partition=partition(person,session),raw_rows=len(data),retained=n,
                    missing_cells=int(np.isnan(raw).sum()),source_absent_rows=[int(np.isnan(raw[:,g]).all(1).sum()) for g in GROUPS],
                    timestamp_first=float(sample[0,0]),timestamp_last=float(sample[-1,0]),gap_runs=int(runs[-1]+1),
                    class_counts=np.bincount(labels,minlength=18).tolist()))
    arrays={k:np.concatenate(v,axis=0) for k,v in arrays.items()};np.savez_compressed(out/'cached_dataset.npz',**arrays)
    meta=dict(name=NAME,n=len(arrays['y']),groups=GROUPS,source_names=['body','objects','ambient'],target='native mid-level gestures including null',
        classes=18,target_codes=TARGET_CODES,session_order=SESSIONS,fit_ids=[1],calibration_ids=[2],test_ids=[3,4],
        original_Hz=30,deterministic_row_stride=STRIDE,nominal_retained_Hz=10,timestamp_unit='milliseconds per independent session',
        physical_test_participants=2,physical_test_sessions=12,physical_sites=1,statistics=stats,metadata=metadata,
        archive_sha256=sha(archive_path),raw_hashes=raw_hashes,cache_sha256=sha(out/'cached_dataset.npz'),freeze_sha256=sha(freeze_path))
    (out/'cache_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta
def prepared(root,engine,cfg):
    root=Path(root);meta=json.loads((root/'cache_metadata.json').read_text());assert sha(root/'cached_dataset.npz')==meta['cache_sha256']
    with np.load(root/'cached_dataset.npz') as f:d={k:f[k].copy() for k in f.files}
    fit=np.flatnonzero(d['partition']=='fit');reserved=np.array([int(hashlib.sha256(f"S1-{d['session'][i]}:{int(d['row_index'][i])}".encode()).hexdigest()[:8],16)%10<3 for i in fit])
    train,audit=fit[~reserved],fit[reserved];assert len(audit)>0
    missing=np.isnan(d['raw']);median=np.zeros(242)
    for j in range(242):
        finite=d['raw'][train,j][~missing[train,j]];median[j]=np.median(finite) if len(finite) else 0.
    physical=np.where(missing,median[None],d['raw']);observed=np.concatenate((physical,missing.astype(float)),axis=1)
    mean=observed[train].mean(0);scale=np.maximum(observed[train].std(0),.05);z=np.clip((observed-mean)/scale,-8,8)
    x=np.ones((len(z),3*484+1));first=0
    for i in range(len(z)):
        if i==0 or d['person'][i]!=d['person'][i-1] or d['session'][i]!=d['session'][i-1] or d['run'][i]!=d['run'][i-1]:first=i
        x[i,:-1]=np.stack([z[max(first,i-lag)] for lag in range(3)],axis=1).reshape(-1)
    source_present=np.stack([~missing[:,g].all(1) for g in GROUPS],axis=1)
    # Resource-only release: the retained feature values are already fixed.
    # Session feature arrays below are contiguous views, not second copies.
    del missing,physical,observed,z
    del d['raw']
    train_x=x[train];train_y=d['y'][train];audit_x=x[audit];audit_y=d['y'][audit]
    allgroups=[g+[v+242 for v in g] for g in GROUPS]
    features=[np.r_[[j for v in g for j in range(3*v,3*v+3)],1452] for g in allgroups]+[np.arange(1453)]
    models=[engine.gradient(np.zeros((len(g),18)),train_x[:,g],train_y,cfg['initial_steps'],cfg,18) for g in features]
    pa=np.stack([engine.softmax(audit_x[:,features[s]]@models[s]) for s in range(3)])
    q=1/(np.mean(np.sum((pa-np.eye(18)[audit_y][None])**2,axis=2),axis=1)+.05)
    # The legacy common service precomputer needs a nonempty error set to
    # populate descriptive fields. The new conditional law does not read
    # this prefix prior: its dedicated S2 state-fit library is explicit.
    prior=[]
    for start in range(0,len(audit),cfg['window']):
        ii=audit[start:start+cfg['window']]
        prior.append(dict(x=x[ii],y=d['y'][ii],p=pa[:,start:start+cfg['window']],
            context=engine.context(x[ii],allgroups),mask=np.ones(3,bool),origin=-1))
    pre=dict(models=models,prior=prior,q=q,mean=mean,scale=scale,median=median,features=features,classes=18,m=3,groups=allgroups,
        train_x=train_x,train_y=train_y,audit_x=audit_x,audit_y=audit_y,recording_streams=[],
        split=dict(training_rows=len(train),audit_rows=len(audit),fit=[1],calibration=[2],test=[3,4],physical_test_participants=2,physical_test_sessions=12))
    for person in (2,3,4):
        for session in SESSIONS:
            ii=np.flatnonzero((d['person']==person)&(d['session']==session));part=partition(person,session)
            assert np.array_equal(ii,np.arange(ii[0],ii[-1]+1)),('Session rows must remain contiguous',person,session)
            block=slice(int(ii[0]),int(ii[-1]+1))
            pre['recording_streams'].append(dict(recording=f'S{person}-{session}',person=person,batch=f'S{person}',session=session,partition=part,
                x=x[block],y=d['y'][block],timestamp=[f'S{person}-{session}:{int(d["row_index"][i])}' for i in ii],timestamp_ms=d['timestamp_ms'][block],
                physical_source_present=source_present[block]))
    return pre,meta
