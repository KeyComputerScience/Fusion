"""Metadata-first unused-collection mobile755 validation of frozen pipelines.

Only the physical adapter/protocol is new. Scientific conditional laws,
calibration V2, physical-availability interface, native PDF/QMF objectives,
DBF operator and service/guard functions are the exact OPPORTUNITY sources.
No archive is downloaded by this runner.
"""
from pathlib import Path
import argparse,datetime,importlib.util,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'physical')]
import mobile755_adapter as adapter

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
r=module('mobile755_shared_final_pipelines',ROOT.parent/'physical/run_opportunity.py')
api,np,nn=r.api,r.np,r.nn
cp,cm,cal,views,op=r.cp,r.cm,r.cal,r.views,r.op
ARMS=r.ARMS;RATE=r.RATE;BUDGETS=r.BUDGETS
CAL_SEEDS=(171001,171002,171003);TEST_SEEDS=(172001,172002,172003,172004,172005)
CANONICAL=CAL_SEEDS[0];OUT=ROOT/'mobile755'
config=r.config;world=r.world;raw_pipelines=r.raw_pipelines;target_check=r.target_check

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,value):api.dump(ROOT/name,value)
def load(name):return api.load(ROOT/name)

def freeze():
    assert not (ROOT/'mobile755_freeze_candidate.json').exists()
    assert not (ROOT/'raw/mobile755.zip').exists() and not (ROOT/'mobile755_data/cached_dataset.npz').exists()
    engine,guard,cfg,service,fork=config()
    sources={Path(__file__),Path(adapter.__file__),ROOT/'verify_mobile755.py',ROOT/'analyze_mobile755.py'}
    upstream_path=PROJECT/'work/fusion_revision_20261005/new_physical/upstream_source_closure_audit.json'
    upstream=api.load(upstream_path)['union_source_hashes']
    for path,want in upstream.items():assert api.sha(path)==want,path
    sources|={Path(path) for path in upstream}
    for obj in list(sys.modules.values()):
        filename=getattr(obj,'__file__',None)
        if filename and str(filename).startswith(str(PROJECT)) and str(filename).endswith('.py'):sources.add(Path(filename).resolve())
    proposal=load('mobile755_protocol_proposal.json')
    p=dict(version=1,dataset=adapter.NAME,utc=now(),raw_access_authorized=False,
        status='Source and protocol candidate freeze before sensor/class observation and before successful local raw transfer; root review required',
        failed_endpoint_resolution=proposal['failed_endpoint_resolution'],raw_archive=proposal['raw_url'],raw_path=proposal['raw_path'],
        algorithm_source_hashes={str(path):api.sha(path) for path in sorted(sources)},source_count=len(sources),
        declaration_hashes={str(ROOT/name):api.sha(ROOT/name) for name in ('mobile755_protocol_proposal.json','prior_use_inventory.json','official_metadata.json','synthetic_integration_checks.json')},
        upstream_closure_sha256=api.sha(upstream_path),arms=ARMS,
        native_task='Standing(stop) versus walking; all native CSV rows, sorted native two-code target vocabulary is schema encoding only',
        physical_groups=dict(accelerometer=['accX','accY','accZ'],gyroscope=['gyroX','gyroY','gyroZ']),
        context_scope='One separately collected mobile-phone collection; no participant/site/device ID column. Original ordinal record order, not independently validated calendar/sessions. timestamp is categorical metadata only; no physical lease seconds stated',
        partitions=dict(fit=(0.,.3),state_fit=(.3,.4),score_cal=(.4,.5),selection=(.5,.6),test=(.6,1.)),
        preprocessing='No row filtering/resampling/reordering. Fit-only median/missing indicators, scale floor.05, clip8, three causal lags reset at ordinal partition boundaries only; no label-triggered resets or inference of clock units',
        numerical_cfg=cfg,classes=2,feature_count=37,source_feature_counts=[19,19],
        common_models='Exact final OPPORTUNITY pipeline source: linear service models; PDF/QMF32tanh tabular physical encoders, fixed known-development rate.2,500 initialsteps and20steps every4windows. DBF uses common PDF probabilities',
        model_cost='Fixed protocol work/interruption units identical to OPPORTUNITY; prefix nonlinear preparation outside scored interval separately reported; no identical-FLOP-price claim',
        quality='Exact strong_prefix_quality on permanently model-excluded hashed FIT error rows of common nonlinear PDF; identical quality across internal controls/DBF; old linear quality retained separately',
        legacy_prior='Held-out FIT chunks populate only original common-precompute descriptive fields. New paired law uses explicit STATE-FIT library and fully matured online tuples',
        persistent_state='STATE-FIT ordinal30–40% under canonical171001 once per physical origin; complete maturity by block end required; exact immutable complete lease E=U−anchor and descriptor D=h−anchor',
        score_cal='Dedicated ordinal40–50% canonical171001; all nine exact raw pipelines. Immutable mature counterfactual score errors, original final V2 stratification/global48/callback-rank age/conditional prior mass2, emptystratum physical-min fallback',
        selection='Dedicated ordinal50–60%, three delays, whole block guarded paid return; same9configs perarm; ties higher probability then largerbandwidth. Strongest calibration-selected non-proposed completepipeline locked beforetest; all co-winners retained',
        grids={arm:cal.configurations() for arm in ARMS},configurations_per_pipeline=9,scenario_trials_per_pipeline=27,
        calibration_seeds=CAL_SEEDS,test_seeds=TEST_SEEDS,canonical_prior_seed=CANONICAL,
        scientific_method='Unchanged conditional_pairing.py and conditional_moment_control.py plus immutable_conditional_calibration_v2.py; all full/factor/Gaussian/completeKDE/unconditional/postconditional moment controls retained',
        external_scope='Native active-objective PDF/QMF adapters, DBF2025 fixed published conflict operator; source objectives/operators exactly shared with OPP. Tabular sensor architecture disclosure retained',
        service=dict(window_records=32,horizon_windows=4,lease_requests_max=128,install_fee=2,restore_fee=1,extra_interruptions=2,conservative_cost=5,reserve=130,primary_budget=130),budgets=BUDGETS,
        ledger_unit='Per ordinal physical block and delay chain; no pooled-loss comparison against a single130unit cap',
        physical_missing='Unchanged source mask wrapper: all raw coordinates absent on every current-window row hides that modality; other native missing values receive fit-only imputation/masks; empty forecasts retain reference',
        no_test_reselection=True,all_outcomes_retained=True,
        outputs='Every nine-pipeline trial/tie/loss; actual action-return decomposition, request/fork reconstruction, admitted coverage/excess/negative return, ledger binding, emptystratum suppressed/missed beneficial opportunities, raw/source freeze provenance')
    dump('mobile755_freeze_candidate.json',p)
    dump('mobile755_candidate_lock.json',dict(utc=now(),sha256=api.sha(ROOT/'mobile755_freeze_candidate.json'),local_raw_exists=False,sensor_or_class_observed=False))
    print('CANDIDATE_FREEZE',len(sources),api.sha(ROOT/'mobile755_freeze_candidate.json'),flush=True)

def verify(candidate=False):
    name='mobile755_freeze_candidate.json' if candidate else 'mobile755_freeze.json'
    lock='mobile755_candidate_lock.json' if candidate else 'mobile755_freeze_lock.json'
    p=load(name);assert api.sha(ROOT/name)==load(lock)['sha256']
    for kind in ('algorithm_source_hashes','declaration_hashes'):
        for path,want in p[kind].items():assert api.sha(path)==want,('Frozen source/declaration changed',path)
    return p

def authorize():
    p=verify(True);assert not (ROOT/'mobile755_freeze.json').exists()
    assert not (ROOT/'raw/mobile755.zip').exists() and not (ROOT/'mobile755_data/cached_dataset.npz').exists()
    p.update(raw_access_authorized=True,authorized_utc=now(),candidate_freeze_sha256=api.sha(ROOT/'mobile755_freeze_candidate.json'),
        status='Root-reviewed source/protocol freeze before successful local transfer and any sensor/class observation; failed metadata endpoint request disclosed')
    dump('mobile755_freeze.json',p);dump('mobile755_freeze_lock.json',dict(utc=now(),sha256=api.sha(ROOT/'mobile755_freeze.json'),local_raw_exists=False,sensor_or_class_observed=False))
    print('AUTHORIZED_FREEZE',api.sha(ROOT/'mobile755_freeze.json'),flush=True)

def process():
    p=verify();assert p['raw_access_authorized'] is True
    print(adapter.process(ROOT/'raw/mobile755.zip',ROOT/'mobile755_data',ROOT/'mobile755_freeze.json'),flush=True)

def prepared(selection=None):
    verify();engine,guard,cfg,service,fork=config()
    cache=api.db.ExactGradientCache(engine);engine.gradient=cache
    pre,meta=adapter.prepared(ROOT/'mobile755_data',engine,cfg)
    api.db.helper.CFG=cfg;nn.helper.CFG=cfg;models={};training=[]
    for rule in ('pdf','qmf'):
        if selection is None:
            initial,work=nn.train([pre['train_x'][:,ix] for ix in pre['features'][:-1]],pre['train_y'],rule,RATE,500,pre['classes'])
            training.append(dict(rule=rule,lr=RATE,**work))
        else:initial=[{k:np.asarray(value) for k,value in model.items()} for model in selection['models'][rule]]
        models[rule]=initial
    pre=cp.strong_prefix_quality(models['pdf'],pre['audit_x'],pre['audit_y'],pre,nn.forward)
    return engine,guard,cfg,service,fork,pre,meta,cache,models,training

def calibrate():
    verify();OUT.mkdir(exist_ok=True);assert not (OUT/'selection.json').exists()
    engine,guard,cfg,service,fork,pre,meta,cache,models,training=prepared()
    records=pre['recording_streams'];library=[];state_log=[]
    for record in [x for x in records if x['partition']=='state_fit']:
        original,base,pdf,qmf=world(record,CANONICAL,pre,cfg,engine,models)
        stream=cp.build_stream(pdf,pre,base,cfg,(),record['recording']);new=cp.library(stream);library.extend(new)
        state_log.append(dict(recording=record['recording'],canonical_seed=CANONICAL,records=new,pending_excluded=sum(d['disagreement']>0 and d['maturity']>stream['windows'] for d in stream['decisions'])))
    pools={arm:[] for arm in ARMS};score_raw=[]
    for record in [x for x in records if x['partition']=='score_cal']:
        original,base,pdf,qmf=world(record,CANONICAL,pre,cfg,engine,models);streams,raw=raw_pipelines(record,base,pdf,qmf,pre,cfg,library)
        for arm in ARMS:
            pools[arm].extend(cal.complete_initial_pool(streams[arm],raw[arm]['rows'],boundary=streams[arm]['windows'],
                provenance=dict(stage='score_fit',dataset=adapter.NAME,recording=record['recording'],canonical_seed=CANONICAL)))
            score_raw.append(dict(arm=arm,recording=record['recording'],seed=CANONICAL,**raw[arm]))
    dump('mobile755/state_fit_library.json.gz',state_log);dump('mobile755/score_cal_raw.json.gz',score_raw)
    entries={(arm,c['bandwidth'],c['probability']):dict(arm=arm,config=c,net=0.) for arm in ARMS for c in cal.configurations()};trials=[];raw_log=[]
    for record in [x for x in records if x['partition']=='selection']:
        for seed in CAL_SEEDS:
            original,base,pdf,qmf=world(record,seed,pre,cfg,engine,models);streams,raw=raw_pipelines(record,base,pdf,qmf,pre,cfg,library)
            for arm in ARMS:
                raw_log.append(dict(arm=arm,recording=record['recording'],seed=seed,**raw[arm]))
                for conf in cal.configurations():
                    result=cal.calibrated_run(streams[arm],raw[arm]['rows'],conf,pools[arm],True);g=api.db.guarded(original,pre,cfg,result,guard,service,130.)
                    entries[arm,conf['bandwidth'],conf['probability']]['net']+=g['increment']/len(CAL_SEEDS)
                    trials.append(dict(arm=arm,recording=record['recording'],person=record['person'],seed=seed,config=conf,guarded=g,issued=result))
            print('CAL',record['recording'],seed,flush=True)
    grid=list(entries.values());selected={arm:max((x for x in grid if x['arm']==arm),key=lambda x:(x['net'],x['config']['probability'],x['config']['bandwidth'])) for arm in ARMS}
    controls=[arm for arm in ARMS if arm!='paired'];winner=max(ARMS,key=lambda arm:selected[arm]['net']);control=max(controls,key=lambda arm:selected[arm]['net'])
    co_winners=[arm for arm in controls if abs(selected[arm]['net']-selected[control]['net'])<1e-12]
    dump('mobile755/calibration_trials.json.gz',trials);dump('mobile755/selection_raw.json.gz',raw_log)
    dump('mobile755/selection.json',dict(selected=selected,grid=grid,strongest_calibration_pipeline=winner,strongest_calibration_control=control,strongest_calibration_control_co_winners=co_winners,
        prior_library=library,score_pools=pools,models={rule:[{k:value.tolist() for k,value in model.items()} for model in initial] for rule,initial in models.items()},
        training=training,quality=pre['q'],previous_linear_quality=pre['previous_linear_quality'],metadata=meta))
    dump('mobile755/selection_lock.json',dict(utc=now(),selection_sha256=api.sha(OUT/'selection.json'),freeze_sha256=api.sha(ROOT/'mobile755_freeze.json'),test_exists=False,cache=cache.report()))
    print('SELECTED',{arm:(x['config'],x['net']) for arm,x in selected.items()},'STRONGEST',winner,flush=True)

def test():
    verify();assert not (OUT/'results.json.gz').exists()
    assert api.sha(OUT/'selection.json')==load('mobile755/selection_lock.json')['selection_sha256']
    selection=load('mobile755/selection.json');selected=selection['selected'];library=selection['prior_library'];pools=selection['score_pools']
    engine,guard,cfg,service,fork,pre,meta,cache,models,training=prepared(selection)
    issued=[];guarded=[];budget=[];checks=[];raw_log=[];availability=[]
    for record in [x for x in pre['recording_streams'] if x['partition']=='test']:
        for seed in TEST_SEEDS:
            original,base,pdf,qmf=world(record,seed,pre,cfg,engine,models);streams,raw=raw_pipelines(record,base,pdf,qmf,pre,cfg,library)
            availability.append(dict(recording=record['recording'],seed=seed,empty_forecast_windows=sum(e['empty_forecast_mask'] for e in pdf),physical_all_absent_windows=sum(e['physical_all_absent'] for e in pdf),
                physically_absent_by_source=[sum(not e['physical_available'][s] for e in pdf) for s in range(pre['m'])]))
            for arm in ARMS:
                result=cal.calibrated_run(streams[arm],raw[arm]['rows'],selected[arm]['config'],pools[arm]);g=api.db.guarded(original,pre,cfg,result,guard,service,130.)
                identity=dict(arm=arm,recording=record['recording'],session=record['session'],person=record['person'],seed=seed)
                issued.append(dict(**identity,**result));guarded.append(dict(**identity,**g));raw_log.append(dict(**identity,**raw[arm]))
                checks.append(dict(**identity,**target_check(original,result['rows'],cfg,fork)))
                for cap in BUDGETS:budget.append(dict(**identity,budget=cap,**(g if cap==130. else api.db.guarded(original,pre,cfg,result,guard,service,cap))))
            print('TEST',record['recording'],seed,flush=True)
    dump('mobile755/results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budget,raw=raw_log,availability=availability,selected=selected))
    dump('mobile755/test_checks.json',dict(target_checks=checks,max_service_error=max(g['service_error'] for g in budget),cache=cache.report()))
    print('COMPLETE',len(guarded),'policies',len(budget),'ledger paths',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','authorize','verify','process','calibrate','test']);args=parser.parse_args();globals()[args.phase]()
