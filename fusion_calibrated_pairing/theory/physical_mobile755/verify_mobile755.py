"""Synthetic-only adapter/forecast/score/service preflight, no real payload."""
from pathlib import Path
import io,json,tempfile,zipfile,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import run_mobile755 as r
np=r.np

def main():
    rng=np.random.default_rng(7171);n=2560
    raw=rng.normal(size=(n,6));raw[5,1]=np.nan
    labels=rng.integers(0,2,n)
    header=','.join(r.adapter.COLUMNS)
    payload=(header+'\n'+'\n'.join(','.join([*(str(v) if np.isfinite(v) else '' for v in raw[i]),f'opaque-{i}',str(labels[i])]) for i in range(n))+'\n').encode()
    fixture=io.BytesIO()
    with zipfile.ZipFile(fixture,'w') as archive:archive.writestr(r.adapter.CSV_NAME,payload)
    fixture.seek(0)
    with zipfile.ZipFile(fixture) as archive:
        name,blob=r.adapter.csv_payload(archive);arrays,vocabulary,boundaries=r.adapter.decode(blob)
    assert vocabulary==(0,1) and boundaries==[0,768,1024,1280,1536,2560]
    assert arrays['timestamp'][0]=='opaque-0' and len(arrays['y'])==n
    engine,guard,cfg,service,fork=r.config();cfg=dict(cfg,initial_steps=2,probe_steps=1)
    cache=r.api.db.ExactGradientCache(engine);engine.gradient=cache
    with tempfile.TemporaryDirectory(prefix='mobile755_synthetic_') as directory:
        directory=Path(directory);np.savez_compressed(directory/'cached_dataset.npz',**arrays)
        (directory/'cache_metadata.json').write_text(json.dumps(dict(cache_sha256=r.adapter.sha(directory/'cached_dataset.npz'),classes=2)))
        pre,meta=r.adapter.prepared(directory,engine,cfg)
        baseline_median=pre['median'].copy();baseline_mean=pre['mean'].copy();baseline_scale=pre['scale'].copy()
        poisoned={k:v.copy() for k,v in arrays.items()};poisoned['raw'][1536:]=1e9
        np.savez_compressed(directory/'cached_dataset.npz',**poisoned)
        (directory/'cache_metadata.json').write_text(json.dumps(dict(cache_sha256=r.adapter.sha(directory/'cached_dataset.npz'),classes=2)))
        future,unused=r.adapter.prepared(directory,engine,cfg)
    assert np.array_equal(baseline_median,future['median']) and np.array_equal(baseline_mean,future['mean']) and np.array_equal(baseline_scale,future['scale'])
    assert pre['train_x'].shape[1]==37 and [len(v) for v in pre['features'][:-1]]==[19,19]
    assert len(pre['train_x'])+len(pre['audit_x'])==768 and len(pre['prior'])>0
    assert [x['partition'] for x in pre['recording_streams']]==['state_fit','score_cal','selection','test']
    for record in pre['recording_streams']:
        first=record['x'][0,:36].reshape(12,3)
        assert np.max(abs(first[:,0]-first[:,1]))==0 and np.max(abs(first[:,0]-first[:,2]))==0
    r.nn.helper.CFG=cfg;r.api.db.helper.CFG=cfg;models={}
    for rule in ('pdf','qmf'):models[rule],_=r.nn.train([pre['train_x'][:,ix] for ix in pre['features'][:-1]],pre['train_y'],rule,.2,2,2)
    pre=r.cp.strong_prefix_quality(models['pdf'],pre['audit_x'],pre['audit_y'],pre,r.nn.forward)
    record=pre['recording_streams'][0]
    original=engine.world(record['x'],record['y'],record['timestamp'],pre,171001,cfg);base=engine.precompute(original,pre,cfg)
    available=r.views.forecast_events(original,record,cfg)
    plain=r.nn.world(available,pre,'pdf',.2,models['pdf']);safe=r.views.neural_world(r.nn,available,pre,'pdf',.2,models['pdf'])
    nonempty_error=max(float(np.max(abs(a['external_probability']-b['external_probability']))) for a,b in zip(plain,safe))
    assert nonempty_error==0.
    state=r.cp.build_stream(safe,pre,base,cfg,(),record['recording']);library=r.cp.library(state)
    record=pre['recording_streams'][1]
    original=engine.world(record['x'],record['y'],record['timestamp'],pre,171001,cfg);base=engine.precompute(original,pre,cfg)
    empty=dict(record,physical_source_present=record['physical_source_present'].copy());empty['physical_source_present'][:32]=False
    available=r.views.forecast_events(original,empty,cfg)
    pdf=r.views.neural_world(r.nn,available,pre,'pdf',.2,models['pdf']);qmf=r.views.neural_world(r.nn,available,pre,'qmf',.2,models['qmf'])
    assert not pdf[0]['mask'].any()
    assert np.max(abs(pdf[0]['external_probability']-r.nn.helper.sm(pdf[0]['x']@pdf[0]['reference'])))==0.
    streams,raws=r.raw_pipelines(record,base,pdf,qmf,pre,cfg,library);checks=[]
    for arm in r.ARMS:
        rows=raws[arm]['rows'];assert all(np.isfinite(x['gate_score']) for x in rows)
        assert not rows[0]['information_ready'] and not rows[0]['action']
        issued=r.cal.calibrated_run(streams[arm],rows,r.cal.configurations()[0],())
        assert not issued['rows'][0]['action']
        cutoff=min(x['maturity'] for x in rows)-1
        poisoned=r.cal.calibrated_run(streams[arm],rows,r.cal.configurations()[0],(),poison_after=cutoff)
        assert [(x['k'],x['gate_score'],x['action']) for x in issued['rows'] if x['k']<=cutoff]==[(x['k'],x['gate_score'],x['action']) for x in poisoned['rows'] if x['k']<=cutoff]
        g=r.api.db.guarded(original,pre,cfg,issued,guard,service,130.);check=r.target_check(original,issued['rows'],cfg,fork)
        assert g['service_error']<1e-8 and check['maximum_error']<1e-8
        report=r.cal.actual_admission_report(g['rows'],g['ledger'])
        checks.append(dict(arm=arm,rows=len(rows),forks=check['forks'],target_error=check['maximum_error'],service_error=g['service_error'],future_complete_target_poison_unchanged=True,empty_source_proposal=False,budget_bound=report['max_spent_plus_pending']))
    report=dict(role='Synthetic interface/schema checks only; no physical empirical result',real_sensor_or_class_observed=False,local_raw_transfer=False,
        synthetic_records=n,boundaries=boundaries,feature_count=37,source_feature_counts=[19,19],
        fit_only_preprocessing_future_poison_unchanged=True,partition_lags_reset=True,
        nonempty_neural_probability_error=nonempty_error,empty_reference_probability_error=0.,controls=r.ARMS,checks=checks,dbf_operator=r.op.audit(),cache=cache.report())
    r.dump('synthetic_integration_checks.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
