"""Lossless cache I/O and boundary-aware prefix construction for locked tasks.

The model helper is passed in rather than imported here. The adapter changes
partition I/O and resets feature history at supplied session boundaries; it
does not change fusion, admission, label delays, or model-update schedules.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np


def load_dataset(root, name):
    folder=Path(root)/name
    meta=json.loads((folder/"cache_metadata.json").read_text())
    cache=folder/"cached_dataset.npz"
    if hashlib.sha256(cache.read_bytes()).hexdigest()!=meta["cache_sha256"]:
        raise ValueError("New-task cache hash differs")
    with np.load(cache,allow_pickle=False) as z:
        arrays={key:z[key].copy() for key in ("raw","y","timestamp","origin","subject","session","partition")}
    n=meta["n"]
    if any(len(a)!=n for a in arrays.values()):raise ValueError("New-task cache lengths differ")
    if arrays["raw"].dtype!=np.float64 or not np.isfinite(arrays["raw"]).all():
        raise ValueError("Malformed new-task features")
    for key in ("timestamp","origin"):
        arrays[key]=arrays[key].tolist()
    arrays.update(n=n,hashes=meta["consumed_raw_hashes"],raw_rows=meta["raw_rows"],
                  participants=meta["participants"],metadata=meta)
    return arrays,meta["derivation"]


def boundary_lagged(raw, mean, scale, session):
    z=np.clip((raw-mean)/scale,-8,8)
    out=np.empty((len(z),3*z.shape[1]+1),np.float64)
    start=0
    for i in range(len(z)):
        if i==0 or session[i]!=session[i-1]:start=i
        # Identical variable-major current / previous / second-previous layout
        # to the frozen engine; missing history repeats the first session row.
        out[i,:-1]=np.stack([z[max(start,i-j)] for j in range(3)],axis=1).reshape(-1)
        out[i,-1]=1.
    return out


def make_prefix(data,spec,cfg,engine):
    ntrain=data["metadata"]["split_boundaries"]["ntrain"]
    ncal=data["metadata"]["split_boundaries"]["ncal"]
    timestamps=data["timestamp"]
    audit=np.array([int(hashlib.sha256(t.encode()).hexdigest()[:8],16)%10<3 for t in timestamps])
    trainidx=np.flatnonzero(~audit[:ntrain]);auditidx=np.flatnonzero(audit[:ntrain])
    if not len(trainidx) or not len(auditidx):raise ValueError("Empty held-out prefix partition")
    mean=data["raw"][trainidx].mean(0)
    scale=np.maximum(data["raw"][trainidx].std(0),.05)
    x=boundary_lagged(data["raw"],mean,scale,data["session"])
    y=data["y"];classes=spec["classes"];groups=spec["groups"];m=len(groups);d=data["raw"].shape[1]
    features=[np.r_[[j for v in g for j in range(3*v,3*v+3)],3*d] for g in groups]+[np.arange(3*d+1)]
    models=[engine.gradient(np.zeros((len(ids),classes)),x[trainidx][:,ids],y[trainidx],cfg["initial_steps"],cfg,classes) for ids in features]
    pa=np.stack([engine.softmax(x[auditidx][:,features[s]]@models[s]) for s in range(m)])
    prior=[]
    for j in range(0,len(auditidx),cfg["window"]):
        sel=auditidx[j:j+cfg["window"]]
        prior.append(dict(x=x[sel],y=y[sel],p=pa[:,j:j+cfg["window"]],context=engine.context(x[sel],groups),mask=np.ones(m,bool),origin=-1))
    q=1/(np.mean(np.sum((pa-np.eye(classes)[y[auditidx]][None])**2,axis=2),axis=1)+.05)
    split=dict(prefix_rows=ntrain,training_rows=len(trainidx),audit_rows=len(auditidx),calibration_rows=ncal-ntrain,
               test_rows=len(x)-ncal,train_start=timestamps[0],prefix_end=timestamps[ntrain-1],
               calibration_start=timestamps[ntrain],calibration_end=timestamps[ncal-1],test_start=timestamps[ncal],test_end=timestamps[-1],
               fit_subjects=sorted(np.unique(data["subject"][:ntrain]).tolist()),
               calibration_subjects=sorted(np.unique(data["subject"][ntrain:ncal]).tolist()),
               test_subjects=sorted(np.unique(data["subject"][ncal:]).tolist()),
               fit_sessions=sorted(np.unique(data["session"][:ntrain]).tolist()),
               calibration_sessions=sorted(np.unique(data["session"][ntrain:ncal]).tolist()),
               test_sessions=sorted(np.unique(data["session"][ncal:]).tolist()),
               feature_history_reset_indices=np.flatnonzero(np.r_[True,data["session"][1:]!=data["session"][:-1]]).tolist())
    return dict(models=models,prior=prior,q=q,mean=mean,scale=scale,features=features,classes=classes,m=m,groups=groups,
                train_x=x[trainidx],train_y=y[trainidx],cal_x=x[ntrain:ncal],cal_y=y[ntrain:ncal],cal_timestamp=timestamps[ntrain:ncal],
                test_x=x[ncal:],test_y=y[ncal:],test_timestamp=timestamps[ncal:],split=split)

