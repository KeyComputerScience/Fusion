"""Synthetic-only adapter, source-mask and complete-pipeline preflight.

It neither transfers nor reads OPPORTUNITY sensor values. Synthetic model
steps are shortened for an interface check and never enter the protocol.
"""
from pathlib import Path
import io,json,tempfile,zipfile,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import run_opportunity as r
np=r.np

def main():
    fixture=io.BytesIO()
    with zipfile.ZipFile(fixture,'w') as archive:
        for person in range(1,5):
            for session in r.adapter.SESSIONS:archive.writestr(f'dataset/S{person}-{session}.dat','SYNTHETIC payload must not be read by metadata checks')
        archive.writestr('dataset/column_names.txt','\n'.join(f'Column: {i}: '+('time milliseconds' if i==1 else 'ML_Both_Arms' if i==250 else f'synthetic_{i}') for i in range(1,251)))
        archive.writestr('dataset/label_legend.txt','\n'.join(map(str,r.adapter.TARGET_CODES)))
    fixture.seek(0)
    with zipfile.ZipFile(fixture) as archive:
        assert len(r.adapter.inventory(archive))==24
        assert len(r.adapter.metadata_gate(archive)['columns'])==250
    engine,guard,cfg,service,fork=r.config();cfg=dict(cfg,initial_steps=2,probe_steps=1)
    cache=r.api.db.ExactGradientCache(engine);engine.gradient=cache
    rng=np.random.default_rng(6161);arrays={key:[] for key in ('raw','y','person','session','partition','timestamp_ms','row_index','run')}
    for person in range(1,5):
        for session in r.adapter.SESSIONS:
            n=256;raw=rng.normal(size=(n,242));raw[5,8]=np.nan
            arrays['raw'].append(raw);arrays['y'].append(rng.integers(0,18,n));arrays['person'].append(np.full(n,person))
            arrays['session'].append(np.full(n,session));arrays['partition'].append(np.full(n,r.adapter.partition(person,session)))
            arrays['timestamp_ms'].append(np.arange(n)*100.);arrays['row_index'].append(np.arange(n)*3);arrays['run'].append(np.zeros(n,int))
    arrays={key:np.concatenate(value) for key,value in arrays.items()}
    with tempfile.TemporaryDirectory(prefix='opportunity_synthetic_') as path:
        path=Path(path);np.savez_compressed(path/'cached_dataset.npz',**arrays)
        (path/'cache_metadata.json').write_text(json.dumps(dict(cache_sha256=r.adapter.sha(path/'cached_dataset.npz'),test_ids=[3,4])))
        pre,meta=r.adapter.prepared(path,engine,cfg)
    assert pre['train_x'].shape[1]==1453 and [len(ix) for ix in pre['features'][:-1]]==[871,361,223]
    assert len(pre['audit_x'])+len(pre['train_x'])==1536 and len(pre['prior'])>0
    assert all(np.shares_memory(record['x'],pre['recording_streams'][0]['x'].base) for record in pre['recording_streams'])
    r.nn.helper.CFG=cfg;r.api.db.helper.CFG=cfg
    models={}
    for rule in ('pdf','qmf'):
        models[rule],_=r.nn.train([pre['train_x'][:,ix] for ix in pre['features'][:-1]],pre['train_y'],rule,.2,2,18)
    pre=r.cp.strong_prefix_quality(models['pdf'],pre['audit_x'],pre['audit_y'],pre,r.nn.forward)
    record=pre['recording_streams'][0]
    original=engine.world(record['x'],record['y'],record['timestamp'],pre,161001,cfg)
    base=engine.precompute(original,pre,cfg)
    available=r.views.forecast_events(original,record,cfg)
    plain=r.nn.world(available,pre,'pdf',.2,models['pdf'])
    safe=r.views.neural_world(r.nn,available,pre,'pdf',.2,models['pdf'])
    nonempty_error=max(float(np.max(abs(a['external_probability']-b['external_probability']))) for a,b in zip(plain,safe))
    assert nonempty_error==0.
    empty_record=dict(record,physical_source_present=record['physical_source_present'].copy())
    empty_record['physical_source_present'][:32]=False
    altered=r.views.forecast_events(original,empty_record,cfg)
    pdf=r.views.neural_world(r.nn,altered,pre,'pdf',.2,models['pdf'])
    qmf=r.views.neural_world(r.nn,altered,pre,'qmf',.2,models['qmf'])
    assert not pdf[0]['mask'].any() and pdf[0]['external_weights'].shape==(32,0)
    assert np.max(abs(pdf[0]['external_probability']-r.nn.helper.sm(pdf[0]['x']@pdf[0]['reference'])))==0.
    stream=r.cp.build_stream(pdf,pre,base,cfg,(),record['recording'])
    prior=r.cp.library(stream)
    streams,raw=r.raw_pipelines(record,base,pdf,qmf,pre,cfg,prior)
    checks=[]
    for arm in r.ARMS:
        assert all(np.isfinite(row['gate_score']) for row in raw[arm]['rows'])
        assert not raw[arm]['rows'][0]['information_ready'] and not raw[arm]['rows'][0]['action']
        pool=r.cal.complete_initial_pool(streams[arm],raw[arm]['rows'],boundary=streams[arm]['windows'])
        issued=r.cal.calibrated_run(streams[arm],raw[arm]['rows'],r.cal.configurations()[0],pool)
        assert not issued['rows'][0]['action']
        guarded=r.api.db.guarded(original,pre,cfg,issued,guard,service,130.)
        check=r.target_check(original,issued['rows'],cfg,fork);assert check['maximum_error']<1e-8
        assert guarded['service_error']<1e-8
        checks.append(dict(arm=arm,rows=len(issued['rows']),forks=check['forks'],service_error=guarded['service_error'],target_error=check['maximum_error']))
    report=dict(role='Synthetic interface preflight only; not physical empirical evidence',raw_data_transferred=False,
        metadata_files=2,filename_sessions=24,feature_count=1453,source_feature_counts=[871,361,223],
        nonempty_nn_forecast_error=nonempty_error,empty_mask_reference_probability_error=0.,
        controls=r.ARMS,checks=checks,dbf_operator=r.op.audit(),cache=cache.report())
    r.dump('synthetic_integration_checks.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
