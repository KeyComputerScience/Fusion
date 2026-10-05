"""Actual fit-prefix-only equivalence checks for the optional execution cache.

Compressed NPY members are streamed only through ntrain rows. No calibration
or held-out feature/label values and no outcome records are materialized.
The cache audit is separate from any scientific calibration/test run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    answer=importlib.util.module_from_spec(spec);spec.loader.exec_module(answer);return answer


def prefix_member(archive,name,rows):
    with archive.open(name+'.npy') as handle:
        version=np.lib.format.read_magic(handle)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(handle)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(handle)
        else:raise ValueError('Unsupported NPY version: '+str(version))
        assert not fortran and not dtype.hasobject and 0<rows<=shape[0]
        count=rows*int(np.prod(shape[1:],dtype=np.int64));expected=count*dtype.itemsize
        body=handle.read(expected);assert len(body)==expected
        array=np.frombuffer(body,dtype=dtype).copy().reshape((rows,*shape[1:]))
        return array,dict(member=name+'.npy',archive_rows=shape[0],materialized_rows=rows,
            dtype=dtype.str,shape=array.shape,materialized_bytes=expected,
            sha256=hashlib.sha256(body).hexdigest())


def state_equal(a,b):
    return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]


def equal(a,b):
    am,aw=a;bm,bw=b
    assert len(am)==len(bm) and aw==bw
    error=0.;arrays=0
    for x,z in zip(am,bm):
        assert list(x)==list(z)
        for key in x:
            assert x[key].dtype==z[key].dtype and x[key].shape==z[key].shape
            assert x[key].tobytes(order='C')==z[key].tobytes(order='C'),key
            error=max(error,float(np.max(np.abs(x[key]-z[key]))));arrays+=1
    return dict(bitwise_equal=True,work_equal=True,arrays=arrays,maximum_array_error=error)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'execution_cache_report.json')
    args=ap.parse_args();assert not args.output.exists(),'Use a new proof filename; preserve prior audits.'
    frozen=json.loads((ROOT/'algorithm_freeze.json').read_text())
    before={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in frozen['source_hashes']}
    assert before==frozen['source_hashes']
    cache_module=module('independent_optional_execution_cache',ROOT/'execution_cache.py')
    neural=next(Path(p) for p in before if p.endswith('/fusion_strengthening_20261003/external/neural_external.py'))
    nn=module('prefix_only_unchanged_neural',neural)
    ad=module('prefix_only_unchanged_adapter',ROOT/'adapter.py')
    cfg=frozen['base_cfg'];nn.helper.CFG=dict(cfg)
    meta=json.loads((ROOT/'data/cache_metadata.json').read_text())
    rows=int(meta['split_boundaries']['ntrain']);data={};member_records=[]
    with zipfile.ZipFile(ROOT/'data/cached_dataset.npz') as archive:
        for name in ('raw','y','recording','session'):
            data[name],record=prefix_member(archive,name,rows);member_records.append(record)
    trainidx=[]
    assert set(data['recording'].tolist())==set(meta['fit_participants'])
    for person in meta['fit_participants']:
        ids=np.flatnonzero(data['recording']==person);cut=int(np.floor(.7*len(ids)))
        assert 0<cut<len(ids);trainidx.extend(ids[:cut])
    trainidx=np.asarray(trainidx);mean=data['raw'][trainidx].mean(0)
    scale=np.maximum(data['raw'][trainidx].std(0),.05)
    x=ad.boundary_lagged(data['raw'],mean,scale,data['session'])
    features=[np.r_[[j for v in group for j in range(3*v,3*v+3)],18] for group in ad.GROUPS]
    xs=[x[trainidx][:,ids] for ids in features];y=data['y'][trainidx]
    assert len(y)>cfg['train_buffer']
    rng=np.random.default_rng(741003)
    models=[dict(W=rng.normal(0,1/np.sqrt(z.shape[1]),(z.shape[1],32)),
        V=rng.normal(0,.05,(33,12)),tcp=np.zeros(33)) for z in xs]
    snapshots=copy.deepcopy((xs,y,models));original=nn.train
    cache=cache_module.install(nn,cfg['train_buffer']);checks=[]
    prior_global_rng=np.random.get_state()
    status='failed';report={}
    try:
        for rule in ('pdf','qmf'):
            print('PREFIX_CACHE_PROOF',rule,'rows',len(y),flush=True)
            np.random.seed(701);state=np.random.get_state();started=time.perf_counter()
            cold=original(xs,y,rule,.02,20,12,models)
            cold_seconds=time.perf_counter()-started
            assert state_equal(state,np.random.get_state()),'Original consumed global RNG.'
            np.random.seed(932);state=np.random.get_state();started=time.perf_counter()
            miss=nn.train(xs,y,rule,.02,20,12,models)
            miss_seconds=time.perf_counter()-started
            assert state_equal(state,np.random.get_state()),'Cache miss consumed global RNG.'
            miss_check=equal(cold,miss)
            started=time.perf_counter();hit=nn.train(xs,y,rule,.02,20,12,models)
            hit_seconds=time.perf_counter()-started;hit_check=equal(cold,hit)
            assert state_equal(state,np.random.get_state()),'Cache hit consumed global RNG.'
            key=cache.key(xs,y,rule,.02,20,12,models)
            for got,stored in zip(hit[0],cache.entries[key][0]):
                assert all(not np.shares_memory(got[n],stored[n]) for n in got)
            hit[0][0]['W'][0,0]+=123.;hit[1]['first_loss']=-999.
            isolation_check=equal(cold,nn.train(xs,y,rule,.02,20,12,models))
            checks.append(dict(rule=rule,cold_vs_miss=miss_check,cold_vs_hit=hit_check,
                mutation_isolation=isolation_check,cold_seconds=cold_seconds,
                miss_seconds=miss_seconds,hit_seconds=hit_seconds,
                global_rng_unchanged=True,key=key,work=cold[1]))
        for got,want in zip(xs,snapshots[0]):assert got.tobytes()==want.tobytes()
        assert y.tobytes()==snapshots[1].tobytes()
        for got,want in zip(models,snapshots[2]):
            for name in got:assert got[name].tobytes()==want[name].tobytes()
        basekey=cache.key(xs,y,'pdf',.02,20,12,models);key_checks={}
        changed_x=[z.copy(order='K') for z in xs];changed_x[0][0,0]+=1e-6
        changed_y=y.copy();changed_y[0]=(int(y[0])+1)%12
        changed_models=copy.deepcopy(models);changed_models[0]['W'][0,0]+=1e-6
        variants=dict(xs=(changed_x,y,'pdf',.02,20,12,models),y=(xs,changed_y,'pdf',.02,20,12,models),
            models=(xs,y,'pdf',.02,20,12,changed_models),rule=(xs,y,'qmf',.02,20,12,models),
            lr=(xs,y,'pdf',.08,20,12,models),steps=(xs,y,'pdf',.02,19,12,models),
            classes=(xs,y,'pdf',.02,20,13,models),
            row_order=([z[::-1] for z in xs],y[::-1],'pdf',.02,20,12,models))
        for name,values in variants.items():
            key_checks[name]=cache.key(*values)!=basekey;assert key_checks[name]
        smallxs=[z[:12].copy() for z in xs];smally=y[:12].copy()
        equal(original(smallxs,smally,'pdf',.02,20,12,models),nn.train(smallxs,smally,'pdf',.02,20,12,models))
        equal(original(xs,y,'pdf',.02,1,12,models),nn.train(xs,y,'pdf',.02,1,12,models))
        after={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before};assert after==before
        status='passed';report=dict(checks=checks,input_unchanged=True,global_rng_unchanged=True,
            key_changes=key_checks,small_buffer_and_non20_bypass_equal=True,
            frozen_sources_unchanged=True,frozen_source_hashes=before,
            fit_prefix_rows=rows,training_rows=len(y),fit_participants=meta['fit_participants'],
            streamed_members=member_records,calibration_rows_materialized=0,test_rows_materialized=0,
            test_outcome_records_read=False,prefix_initial_models='constructed fixed local seed741003; no selected or held-out model/result inspected',
            no_scientific_trial_or_selection_performed=True)
    finally:
        np.random.set_state(prior_global_rng)
        report.update(status=status,cache=cache.report(),wrapper_sha256=cache_module.sha(ROOT/'execution_cache.py'),
            verifier_sha256=cache_module.sha(__file__))
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False,
            default=lambda value:value.item() if isinstance(value,np.generic) else str(value))+'\n')
    print(json.dumps(dict(status=status,checks=len(checks),hits=cache.hits,misses=cache.misses,
        output=str(args.output)),indent=2),flush=True)


if __name__=='__main__':main()
