"""Pre-acquisition Daphnet245 adapter; fixed participants, physical source groups."""
import io,re,zipfile,json,hashlib
from pathlib import Path
import numpy as np

GROUPS=[[0,1,2],[3,4,5],[6,7,8]]
FIT=tuple(range(1,6));CAL=(6,7);TEST=(8,9,10)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def process(path,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'cache_metadata.json').exists():raise ValueError('Existing cache')
    raw=[];ys=[];people=[];sessions=[];times=[];stats={};hashes={}
    with zipfile.ZipFile(path) as archive:
        files=sorted(n for n in archive.namelist() if re.fullmatch(r'S\d\dR\d\d\.txt',Path(n).name) and '/dataset/' in n)
        assert files
        for name in files:
            content=archive.read(name);hashes[name]=hashlib.sha256(content).hexdigest()
            z=np.loadtxt(io.BytesIO(content));assert z.ndim==2 and z.shape[1]==11 and np.isfinite(z).all()
            person=int(Path(name).name[1:3]);assert person in FIT+CAL+TEST
            assert np.isin(z[:,-1],[0,1,2]).all() and (np.diff(z[:,0])>=0).all()
            run=0;new=True;kept=0
            for i,row in enumerate(z):
                if row[-1]==0 or (i and row[0]-z[i-1,0]>1000):new=True
                if i%64 or row[-1]==0:continue
                if new:run+=1;new=False
                raw.append(row[1:10]);ys.append(int(row[-1])-1);people.append(person)
                sessions.append(f'{Path(name).stem}/run{run:04d}');times.append(f'daphnet245/{Path(name).stem}/raw{i:09d}')
                kept+=1
            stats[Path(name).stem]=dict(raw_rows=len(z),retained=kept,runs=run,
                first_ms=float(z[0,0]),last_ms=float(z[-1,0]),median_step_ms=float(np.median(np.diff(z[:,0]))))
    arrays=dict(raw=np.array(raw),y=np.array(ys),person=np.array(people),session=np.array(sessions),timestamp=np.array(times))
    assert set(people)==set(FIT+CAL+TEST)
    np.savez_compressed(out/'cached_dataset.npz',**arrays)
    meta=dict(name='daphnet245',raw_rows=sum(v['raw_rows'] for v in stats.values()),n=len(raw),
        fit=FIT,calibration=CAL,test=TEST,groups=GROUPS,stride=64,annotation_codes=[1,2],
        file_statistics=stats,raw_hashes=hashes,archive_sha256=sha(path),cache_sha256=sha(out/'cached_dataset.npz'))
    (out/'cache_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta

def prepared(root,engine,cfg):
    root=Path(root);meta=json.loads((root/'cache_metadata.json').read_text())
    assert sha(root/'cached_dataset.npz')==meta['cache_sha256']
    with np.load(root/'cached_dataset.npz') as f:d={k:f[k].copy() for k in f.files}
    train=[];audit=[]
    for person in FIT:
        ids=np.flatnonzero(d['person']==person);cut=int(.7*len(ids));assert cut>0 and cut<len(ids)
        train.extend(ids[:cut]);audit.extend(ids[cut:])
    train=np.array(train);audit=np.array(audit)
    mean=d['raw'][train].mean(0);scale=np.maximum(d['raw'][train].std(0),.05)
    z=np.clip((d['raw']-mean)/scale,-8,8);x=np.ones((len(z),28));first=0
    for i in range(len(z)):
        if i==0 or d['session'][i]!=d['session'][i-1]:first=i
        x[i,:-1]=np.stack([z[max(first,i-lag)] for lag in range(3)],axis=1).reshape(-1)
    features=[np.r_[[j for v in group for j in range(3*v,3*v+3)],27] for group in GROUPS]+[np.arange(28)]
    models=[engine.gradient(np.zeros((len(ids),2)),x[train][:,ids],d['y'][train],cfg['initial_steps'],cfg,2) for ids in features]
    pa=np.stack([engine.softmax(x[audit][:,features[s]]@models[s]) for s in range(3)])
    prior=[]
    for session in dict.fromkeys(d['session'][audit].tolist()):
        positions=np.flatnonzero(d['session'][audit]==session)
        for left in range(0,len(positions),cfg['window']):
            pp=positions[left:left+cfg['window']];ii=audit[pp]
            prior.append(dict(x=x[ii],y=d['y'][ii],p=pa[:,pp],context=engine.context(x[ii],GROUPS),mask=np.ones(3,bool),origin=-1))
    q=1/(np.mean(np.sum((pa-np.eye(2)[d['y'][audit]][None])**2,axis=2),axis=1)+.05)
    pre=dict(models=models,prior=prior,q=q,mean=mean,scale=scale,features=features,classes=2,m=3,groups=GROUPS,
        train_x=x[train],train_y=d['y'][train],recording_streams=[],split=dict(training_rows=len(train),audit_rows=len(audit),
        calibration_rows=int(np.isin(d['person'],CAL).sum()),test_rows=int(np.isin(d['person'],TEST).sum()),
        fit_people=FIT,calibration_people=CAL,test_people=TEST))
    for session in dict.fromkeys(d['session'].tolist()):
        ids=np.flatnonzero(d['session']==session);person=int(d['person'][ids[0]])
        if person in FIT:continue
        pre['recording_streams'].append(dict(recording=session,person=person,partition='calibration' if person in CAL else 'test',
            x=x[ids],y=d['y'][ids],timestamp=d['timestamp'][ids].tolist()))
    return pre,meta
