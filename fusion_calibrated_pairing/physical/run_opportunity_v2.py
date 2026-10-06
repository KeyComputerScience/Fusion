"""Prospective OPPORTUNITY native-gesture paid fusion validation.

The candidate freeze, authorization freeze, original sensor archive,
dedicated state/score/selection partitions and every result are retained.
No acquisition occurs in this runner. Authorize is run only after the root
agent has reviewed the complete pre-acquisition numerical source closure.
"""
from pathlib import Path
import argparse,datetime,importlib.util,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'design'),str(ROOT.parent/'risk')]
import conditional_pairing as cp
import conditional_moment_control as cm
import immutable_conditional_calibration_v2 as cal
import opportunity_adapter_v2 as adapter
import physical_views as views

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module('opportunity_shared_service',PROJECT/'work/fusion_joint_return_20261005/new_physical/run_selfback_v4.py')
api=old.api;np=api.np;nn=old.nn
op=module('opportunity_published_dbf_operator',PROJECT/'work/fusion_coupling_focus_20261005/dbf/dbf_operator.py')
ARMS=cp.ARMS+('conditional_moment','pdf','qmf','dbf')
CAL_SEEDS=(161001,161002,161003);TEST_SEEDS=(162001,162002,162003,162004,162005)
CANONICAL=CAL_SEEDS[0];BUDGETS=(0.,110.,130.,260.);RATE=.2
OUT=ROOT/'opportunity'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,value):api.dump(ROOT/name,value)
def load(name):return api.load(ROOT/name)
def config():
    engine,recovery,guard,source,cfg,study,service,fork=api.p.r.binding()
    cfg=api.p.base_cfg(cfg)
    assert cfg['window']==32 and cfg['horizon']==4 and cfg['initial_steps']==500 and cfg['probe_steps']==20 and cfg['probe_period']==4
    return engine,guard,cfg,service,fork

def freeze():
    assert not (ROOT/'opportunity_freeze_candidate.json').exists()
    assert not (ROOT/'raw/opportunity226.zip').exists() and not (ROOT/'opportunity_data/cached_dataset.npz').exists()
    engine,guard,cfg,service,fork=config()
    sources={Path(__file__),ROOT/'verify_integration.py',ROOT/'analyze_opportunity.py',Path(cp.__file__),Path(cm.__file__),Path(cal.__file__),Path(adapter.__file__),Path(views.__file__),Path(nn.__file__),Path(op.__file__)}
    upstream_path=PROJECT/'work/fusion_revision_20261005/new_physical/upstream_source_closure_audit.json'
    upstream=api.load(upstream_path)['union_source_hashes']
    for path,want in upstream.items():assert api.sha(path)==want,path
    sources|={Path(path) for path in upstream}
    for obj in list(sys.modules.values()):
        filename=getattr(obj,'__file__',None)
        if filename and str(filename).startswith(str(PROJECT)) and str(filename).endswith('.py'):sources.add(Path(filename).resolve())
    proposal=load('opportunity_protocol_v2_pre_acquisition.json')
    p=dict(version=1,utc=now(),dataset='opportunity226',raw_access_authorized=False,
        status='Complete candidate numerical and physical protocol freeze; root review required before transfer',
        raw_archive=proposal['raw_url'],raw_path=proposal['raw_path'],
        algorithm_source_hashes={str(path):api.sha(path) for path in sorted(sources)},
        declaration_hashes={str(ROOT/name):api.sha(ROOT/name) for name in ('opportunity_protocol_proposal.json','opportunity_protocol_v2_pre_acquisition.json','prior_use_inventory.json','official_metadata.json','synthetic_integration_checks.json')},
        upstream_closure_sha256=api.sha(upstream_path),source_count=len(sources),arms=ARMS,
        native_task='All 17 mid-level gestures plus null; zero-based raw target column249, all24 sessions retained',
        participants=dict(fit=[1],state_fit=[2],score_cal=[2],selection=[2],test=[3,4]),
        fixed_sessions=dict(fit=adapter.SESSIONS,state_fit=adapter.SESSIONS[:2],score_cal=adapter.SESSIONS[2:4],selection=adapter.SESSIONS[4:],test=adapter.SESSIONS),
        physical_units='Two new test participants, six nested sessions each, one physical collection; not twelve independent sites',
        preprocessing='Native synchronized body145/object60/ambient37 coordinates. Original30Hz row-stride3 from row0 -> nominal10Hz. Fit-only median+missing indicators, stdfloor.05/clip8, three causal lags/session or observedtime-gap>250ms; no label filtering, ordering or label-triggered resets',
        raw_column_validation='Official column_names and label_legend read/validated before any .dat payload; original24 filename inventory',
        physical_missing='Mask a forecast source only if every raw coordinate is missing on every current-window row; partial missing retains trained median/mask view. Empty combined forecast mask yields common reference probability and no proposal. All originally nonempty NN forecasts unchanged',
        numerical_cfg=cfg,classes=18,feature_count=1453,source_feature_counts=[871,361,223],
        common_model='Linear service candidate/reference and common physical-source PDF32tanh forecaster; fixed known-development lr.2,500 prefix fullbatch steps/20steps every4windows. Native QMF same architecture/steps/rate. DBF reuses the common PDF sources without extra training',
        model_cost='Prefix preparation outside scored interval disclosed. All service arms pay the same declared source/candidate scheduled preparation fees; costs are protocol units, not measured FLOP prices',
        quality='New nonlinear PDF inverse Brier+.05 only on permanently model-excluded hashed S1 error rows; identical fixed quality across information arms and DBF; old linear quality separately retained',
        legacy_prior_scope='S1 held-out chunks only populate unused legacy common-precompute descriptive fields. The conditional residual law receives only dedicated S2 STATE-FIT library and fully matured immutable online tuples',
        persistent_state='S2 Drill/ADL1 canonical seed161001 only; original issued full lease U/PDFanchor and source descriptors; pending end-of-session targets excluded. Same state-fit library passed unchanged to all later information arms',
        score_cal='S2 ADL2/ADL3 canonical161001 only; each arm immutable raw residual (Sraw-X)/N only fully matured by session end. Dedicated score pool disjoint from state fit and hyperparameter selection. V2 query filters predictable support>0 versus support=0; global latest48 mature online callbacks preserve actual arrival age, then same-stratum filtering; persistent score pool conditional mass2. Empty current stratum q=2 and L=-N-5 physical-support fallback, not a nonempty empirical calibration certificate',
        selection='S2 ADL4/ADL5, three shared delays; whole-session actual guarded complete paid return. Nine configurations per pipeline, ties larger residual probability then bandwidth. Strongest non-proposed calibration-selected complete control is the primary comparator, locked before test; control ties follow declared arm order and all co-winners retained',
        grids={arm:cal.configurations() for arm in ARMS},configuration_count_per_pipeline=9,scenario_trials_per_pipeline=27,
        calibration_seeds=CAL_SEEDS,test_seeds=TEST_SEEDS,canonical_prior_seed=CANONICAL,
        distribution='Fixed conditional joint (originalissued residual E,physical disagreement descriptors D), Scott full kernel/ESmass.5; pair-factorized preserves every(E,D_s) margin; Gaussian global moments; full joint KDE with diagonal smoothing still retains joint locations; unconditional; strongest conditional-moment maxentropy retains paired post-conditioning mean/variance',
        native_controls='PDF/QMF raw N*issued fused anchor-5 plus identical signed conditional calibration nine-grid; fixed lr.2. Active original published objectives on tabular32tanh adapters, not original paper architectures',
        dbf_control='DBF2025 published get_doc_belief_fusion conflict discount lambda1 epsilon1e-6, common PDFprobabilities evidence e_s=C*(q_s/mean_all_q)*p_s; fixed kappa=C from known selfBACK design. Same nine signed calibration configurations, no evidential-backbone training or extra search',
        service=dict(window_records=32,horizon_windows=4,nominal_seconds=12.8,install_fee=2,restore_fee=1,extra_interruptions=2,conservative_cost=5,reserve=130,primary_budget=130),budgets=BUDGETS,
        ledger_unit='Budget/reset per physical session and delay history chain; pooled losses are not compared with one130unit budget',
        no_test_reselection=True,all_outcomes_retained=True,
        outputs='Every issue/proposal/actual admission, callback, action-return decomposition, independent request/fork reconstruction, admitted coverage/excess magnitude, harmful count/loss, ledger binding, person/session results, empty-stratum suppressed and missed-beneficial potential outcomes, source/raw hashes')
    dump('opportunity_freeze_candidate.json',p)
    dump('opportunity_candidate_lock.json',dict(utc=now(),sha256=api.sha(ROOT/'opportunity_freeze_candidate.json'),raw_exists=False))
    print('CANDIDATE_FREEZE',p['source_count'],api.sha(ROOT/'opportunity_freeze_candidate.json'),flush=True)

def verify(candidate=False):
    name='opportunity_freeze_candidate.json' if candidate else 'opportunity_freeze.json'
    lock='opportunity_candidate_lock.json' if candidate else 'opportunity_freeze_lock.json'
    p=load(name);assert api.sha(ROOT/name)==load(lock)['sha256']
    for kind in ('algorithm_source_hashes','declaration_hashes'):
        for path,want in p[kind].items():assert api.sha(path)==want,('Frozen source/declaration changed',path)
    supplement=load('opportunity_metadata_repair_freeze.json')
    assert api.sha(ROOT/'opportunity_metadata_repair_freeze.json')==load('opportunity_metadata_repair_lock.json')['sha256']
    assert supplement['original_freeze_sha256']==api.sha(ROOT/'opportunity_freeze.json')
    for kind in ('algorithm_source_hashes','official_metadata_hashes'):
        for path,want in supplement[kind].items():assert api.sha(path)==want,('Metadata repair source changed',path)
    return supplement

def authorize():
    # Invoked only after explicit root authorization, never by freeze().
    p=verify(True);assert not (ROOT/'opportunity_freeze.json').exists()
    assert not (ROOT/'raw/opportunity226.zip').exists() and not (ROOT/'opportunity_data/cached_dataset.npz').exists()
    p.update(raw_access_authorized=True,authorized_utc=now(),candidate_freeze_sha256=api.sha(ROOT/'opportunity_freeze_candidate.json'),
        status='Root-reviewed complete prospective freeze before any OPPORTUNITY sensor transfer')
    dump('opportunity_freeze.json',p);dump('opportunity_freeze_lock.json',dict(utc=now(),sha256=api.sha(ROOT/'opportunity_freeze.json'),raw_exists=False))
    print('AUTHORIZED_FREEZE',api.sha(ROOT/'opportunity_freeze.json'),flush=True)

def process():
    p=verify();assert p['raw_access_authorized'] is True
    print(adapter.process(ROOT/'raw/opportunity226.zip',ROOT/'opportunity_data',ROOT/'opportunity_metadata_repair_freeze.json'),flush=True)

def prepared(selection=None):
    verify();engine,guard,cfg,service,fork=config()
    cache=api.db.ExactGradientCache(engine);engine.gradient=cache
    pre,meta=adapter.prepared(ROOT/'opportunity_data',engine,cfg)
    api.db.helper.CFG=cfg;nn.helper.CFG=cfg
    models={};training=[]
    for rule in ('pdf','qmf'):
        if selection is None:
            initial,work=nn.train([pre['train_x'][:,ix] for ix in pre['features'][:-1]],pre['train_y'],rule,RATE,500,pre['classes'])
            training.append(dict(rule=rule,lr=RATE,**work))
        else:initial=[{k:np.asarray(value) for k,value in model.items()} for model in selection['models'][rule]]
        models[rule]=initial
    pre=cp.strong_prefix_quality(models['pdf'],pre['audit_x'],pre['audit_y'],pre,nn.forward)
    return engine,guard,cfg,service,fork,pre,meta,cache,models,training

def world(record,seed,pre,cfg,engine,models):
    original=engine.world(record['x'],record['y'],record['timestamp'],pre,seed,cfg)
    # Native imposed masks remain nonempty for legacy descriptive moments.
    # Physical masks affect forecast information; complete service targets
    # do not depend on which forecasts are available.
    base=engine.precompute(original,pre,cfg)
    available=views.forecast_events(original,record,cfg)
    pdf=views.neural_world(nn,available,pre,'pdf',RATE,models['pdf'])
    qmf=views.neural_world(nn,available,pre,'qmf',RATE,models['qmf'])
    return original,base,pdf,qmf

def raw_pipelines(record,base,pdf,qmf,pre,cfg,library):
    stream=cp.build_stream(pdf,pre,base,cfg,library,record['recording'])
    raw={arm:cp.raw_run(stream,arm,cfg) for arm in cp.ARMS}
    raw['conditional_moment']=cm.raw_run(stream,raw['paired'],cfg)
    streams={arm:stream for arm in cp.ARMS+('conditional_moment',)}
    for arm,events in (('pdf',pdf),('qmf',qmf),('dbf',views.dbf_world(op,pdf,pre))):
        streams[arm]=views.native_stream(cp,events,base,pre,record['recording'])
        raw[arm]=views.native_raw(streams[arm])
    return streams,raw

def target_check(original,rows,cfg,fork):
    canonical=[dict(row,q_issued=0.) for row in rows]
    return old.base.target_checks(original,canonical,cfg,fork)

def calibrate():
    verify();OUT.mkdir(exist_ok=True);assert not (OUT/'selection.json').exists()
    engine,guard,cfg,service,fork,pre,meta,cache,models,training=prepared()
    records=pre['recording_streams'];library=[];state_log=[]
    for record in [r for r in records if r['partition']=='state_fit']:
        original,base,pdf,qmf=world(record,CANONICAL,pre,cfg,engine,models)
        stream=cp.build_stream(pdf,pre,base,cfg,(),record['recording']);rows=cp.library(stream)
        library.extend(rows);state_log.append(dict(recording=record['recording'],canonical_seed=CANONICAL,records=rows,pending_excluded=sum(d['disagreement']>0 and d['maturity']>stream['windows'] for d in stream['decisions'])))
        print('STATE',record['recording'],len(rows),flush=True)
    pools={arm:[] for arm in ARMS};score_raw=[]
    for record in [r for r in records if r['partition']=='score_cal']:
        original,base,pdf,qmf=world(record,CANONICAL,pre,cfg,engine,models)
        streams,raw=raw_pipelines(record,base,pdf,qmf,pre,cfg,library)
        for arm in ARMS:
            pool=cal.complete_initial_pool(streams[arm],raw[arm]['rows'],boundary=streams[arm]['windows'],
                provenance=dict(stage='score_fit',dataset='opportunity226',recording=record['recording'],person=record['person'],canonical_seed=CANONICAL))
            pools[arm].extend(pool);score_raw.append(dict(arm=arm,recording=record['recording'],seed=CANONICAL,**raw[arm]))
        print('SCORE_CAL',record['recording'],flush=True)
    dump('opportunity/state_fit_library.json.gz',state_log);dump('opportunity/score_cal_raw.json.gz',score_raw)
    entries={(arm,conf['bandwidth'],conf['probability']):dict(arm=arm,config=conf,net=0.) for arm in ARMS for conf in cal.configurations()}
    trials=[];raw_log=[]
    for record in [r for r in records if r['partition']=='selection']:
        for seed in CAL_SEEDS:
            original,base,pdf,qmf=world(record,seed,pre,cfg,engine,models)
            streams,raw=raw_pipelines(record,base,pdf,qmf,pre,cfg,library)
            for arm in ARMS:
                raw_log.append(dict(arm=arm,recording=record['recording'],seed=seed,**raw[arm]))
                for conf in cal.configurations():
                    result=cal.calibrated_run(streams[arm],raw[arm]['rows'],conf,pools[arm],True)
                    guarded=api.db.guarded(original,pre,cfg,result,guard,service,130.)
                    entry=entries[arm,conf['bandwidth'],conf['probability']];entry['net']+=guarded['increment']/len(CAL_SEEDS)
                    trials.append(dict(arm=arm,recording=record['recording'],person=record['person'],seed=seed,config=conf,guarded=guarded,issued=result))
            print('SELECT_REPLAY',record['recording'],seed,flush=True)
    grid=list(entries.values())
    selected={arm:max((entry for entry in grid if entry['arm']==arm),key=lambda x:(x['net'],x['config']['probability'],x['config']['bandwidth'])) for arm in ARMS}
    winner=max(ARMS,key=lambda arm:selected[arm]['net'])
    controls=tuple(arm for arm in ARMS if arm!='paired')
    control=max(controls,key=lambda arm:selected[arm]['net'])
    co_winners=[arm for arm in controls if abs(selected[arm]['net']-selected[control]['net'])<1e-12]
    dump('opportunity/calibration_trials.json.gz',trials);dump('opportunity/selection_raw.json.gz',raw_log)
    dump('opportunity/selection.json',dict(selected=selected,grid=grid,strongest_calibration_pipeline=winner,
        strongest_calibration_control=control,strongest_calibration_control_co_winners=co_winners,
        prior_library=library,score_pools=pools,models={rule:[{k:value.tolist() for k,value in model.items()} for model in initial] for rule,initial in models.items()},
        training=training,quality=pre['q'],previous_linear_quality=pre['previous_linear_quality'],metadata=meta))
    dump('opportunity/selection_lock.json',dict(utc=now(),selection_sha256=api.sha(OUT/'selection.json'),freeze_sha256=api.sha(ROOT/'opportunity_freeze.json'),metadata_repair_freeze_sha256=api.sha(ROOT/'opportunity_metadata_repair_freeze.json'),test_exists=False,cache=cache.report()))
    print('SELECTED',{arm:(entry['config'],entry['net']) for arm,entry in selected.items()},'STRONGEST',winner,flush=True)

def test():
    verify();assert not (OUT/'results.json.gz').exists()
    assert api.sha(OUT/'selection.json')==load('opportunity/selection_lock.json')['selection_sha256']
    selection=load('opportunity/selection.json');selected=selection['selected'];library=selection['prior_library'];pools=selection['score_pools']
    engine,guard,cfg,service,fork,pre,meta,cache,models,training=prepared(selection)
    issued=[];guarded=[];budget=[];checks=[];raw_log=[];availability=[]
    for record in [r for r in pre['recording_streams'] if r['partition']=='test']:
        for seed in TEST_SEEDS:
            original,base,pdf,qmf=world(record,seed,pre,cfg,engine,models)
            streams,raw=raw_pipelines(record,base,pdf,qmf,pre,cfg,library)
            availability.append(dict(recording=record['recording'],person=record['person'],seed=seed,
                empty_forecast_windows=sum(event['empty_forecast_mask'] for event in pdf),
                physical_all_absent_windows=sum(event['physical_all_absent'] for event in pdf),
                physically_absent_by_source=[sum(not event['physical_available'][s] for event in pdf) for s in range(pre['m'])]))
            for arm in ARMS:
                result=cal.calibrated_run(streams[arm],raw[arm]['rows'],selected[arm]['config'],pools[arm],False)
                g=api.db.guarded(original,pre,cfg,result,guard,service,130.)
                identity=dict(arm=arm,recording=record['recording'],session=record['session'],person=record['person'],seed=seed)
                issued.append(dict(**identity,**result));guarded.append(dict(**identity,**g));raw_log.append(dict(**identity,**raw[arm]))
                checks.append(dict(**identity,**target_check(original,result['rows'],cfg,fork)))
                for B in BUDGETS:budget.append(dict(**identity,budget=B,**(g if B==130. else api.db.guarded(original,pre,cfg,result,guard,service,B))))
            print('TEST',record['recording'],seed,flush=True)
    # Save numerical output before independent interpretation/reporting.
    dump('opportunity/results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budget,raw=raw_log,availability=availability,selected=selected))
    dump('opportunity/test_checks.json',dict(target_checks=checks,max_service_error=max(g['service_error'] for g in budget),cache=cache.report()))
    print('COMPLETE',len(guarded),'policies',len(budget),'ledger paths',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['verify','process','calibrate','test'])
    args=parser.parse_args();globals()[args.phase]()
