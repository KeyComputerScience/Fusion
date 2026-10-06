"""One prospectively declared DEVELOPMENT candidate on already-used selfBACK.

No new physical confirmation. No OPPORTUNITY records are opened. All nine
selected pipelines, trials and outcomes remain recorded.
"""
from pathlib import Path
import argparse,datetime,sys,importlib.util,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'design'));sys.path.insert(0,str(ROOT.parent/'risk'))
import horizon_law as horizon
import immutable_conditional_calibration_v2 as cal
import conditional_moment_control as moment

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
known=module('horizon_known_development_source',ROOT.parent/'design/run_selfback_development.py')
views=module('horizon_native_view_source',ROOT.parent/'physical/physical_views.py')
op=module('horizon_known_dbf_operator',PROJECT/'work/fusion_coupling_focus_20261005/dbf/dbf_operator.py')
api=known.api;np=known.np;nn=known.nn
ARMS=horizon.ARMS+('conditional_moment','pdf','qmf','dbf')
CONTROL_ORDER=('pair_factorized','full_gaussian','joint_diagonal_kernel','conditional_moment','unconditional','pdf','qmf','dbf')
CAL_SEEDS=known.CAL_SEEDS;DEV_SEEDS=known.DEV_SEEDS;CANONICAL=CAL_SEEDS[0]

def dump(name,x):api.dump(ROOT/name,x)
def load(name):return api.load(ROOT/name)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def prepared():
    output=known.prepared()
    selection=known.old.load('selfback/selection.json')
    qmf=[{k:np.asarray(v) for k,v in model.items()} for model in selection['models']['qmf:0.2']]
    return (*output,qmf)

def worlds(records,pre,cfg,engine,pdf_initial,qmf_initial,seeds):
    for rec in records:
        for seed in seeds:
            original=engine.world(rec['x'],rec['y'],rec['timestamp'],pre,seed,cfg)
            base=engine.precompute(original,pre,cfg)
            pdf=nn.world(original,pre,'pdf',.2,pdf_initial)
            qmf=nn.world(original,pre,'qmf',.2,qmf_initial)
            yield rec,seed,original,base,pdf,qmf

def raw_results(base,pdf,qmf,pre,cfg,library,recording):
    stream=horizon.build_stream(pdf,pre,base,cfg,library,recording)
    out={arm:horizon.raw_run(stream,arm) for arm in horizon.ARMS}
    out['conditional_moment']=moment.raw_run(stream,out['paired'],cfg)
    streams={arm:stream for arm in out}
    for arm,events in (('pdf',pdf),('qmf',qmf),('dbf',views.dbf_world(op,pdf,pre))):
        native=views.native_stream(known.cp,events,base,pre,recording)
        streams[arm]=native;out[arm]=views.native_raw(native)
    return streams,out

def freeze():
    assert not (ROOT/'protocol.json').exists()
    known.verify();known.old.verify()
    sources=set()
    for mod in list(sys.modules.values()):
        filename=getattr(mod,'__file__',None)
        if filename and str(filename).startswith(str(PROJECT)) and str(filename).endswith('.py'):sources.add(Path(filename))
    sources.add(Path(__file__))
    data=known.old.ROOT/'selfback_data'
    protocol=dict(version='horizon-conditional-development-v0',utc=now(),
        status='ONE new candidate on already inspected selfBACK development; not physical confirmation; OPPORTUNITY remains inaccessible to this runner',
        arms=ARMS,source_hashes={str(p):api.sha(p) for p in sorted(sources)},
        data_hashes={str(p):api.sha(p) for p in data.glob('*') if p.is_file()},
        source_model_selection_sha256=api.sha(known.old.ROOT/'selfback/selection.json'),
        descriptor='One6-dimensional block per PHYSICAL source: current+two preceding immutable issued soft directional source evidence and mean probability top-two margin. Common8: hard/softPDFanchors,C/Rmargins,hard/softdisagreement,activefraction,startupfraction. Historiesresetperphysicalrecording; masked lag coordinatesremainunobserved.',
        target='unchanged original fulllease hardcorrectness contrast U=G/128; residualE=U-originalhardPDFanchor, never current-model historical reprojection',
        state='ONE complete-return conditional Gaussian-completed tuple-mixture, qualityprior mass2, contextband1,48localfullymaturedtuples,origin-agebeta.97,Scott smoothing,ESalpha.5; include quiet origins when softTV>0; weight max(harddisagree,softTV)',
        controls='factor preserves every multivariate (E,entirephysicalsourcepath) margin AFTER commonconditioning; fullGaussian/fulljointdiagonalkernel/commononly/postconditionalmaxentropy receive same descriptor/library/quality/currentmasks. NativePDF/QMF activeoldobjectives andDBF operator receive common signed calibration andguard, distinctnativeeligibilitydisclosed.',
        partitions='unchanged known development calibration ranks: first4 statefit; next2 scorefit; last2 selection; all13formerheldoutpeople used-development evaluation. Canonical151001 state/score originsonce.',
        forecasters='fixed known-data PDF/QMF lr.2 initialmodels; no new source/backbone search; commoncandidate/reference andwork unchanged',
        calibration='unchanged scientific V2: immutable readyissued residual quantile; predictable hardquiet/informative filter, global48callbackcapacity; persistent same-stratum mass2; signed supportclipped score,emptystratum physicalminimum',
        configurations={arm:cal.configurations() for arm in ARMS},configurations_per_arm=9,
        selection='maximize guarded fullpaid selectionreturn averaged3delayseeds; ties higherp then largerband; strongestcompletecontrol selected CAL-only over8controls with declared tieorder',
        strongest_control_tie_order=CONTROL_ORDER,canonical_seed=CANONICAL,
        calibration_seeds=CAL_SEEDS,development_seeds=DEV_SEEDS,
        reserve=130.,primary_budget=130.,budgets=(0.,110.,130.,260.),
        fail_policy='anysolver/runtimefailure recorded; no algorithm replacement or hiddenfallback; all valid arms/outcomes retained',
        no_outcome_selection=True,no_opp_access=True)
    dump('protocol.json',protocol);dump('pre_run_freeze.json',dict(utc=now(),protocol_sha256=api.sha(ROOT/'protocol.json')))
    print('FREEZE',api.sha(ROOT/'protocol.json'),len(sources),flush=True)

def verify():
    p=load('protocol.json');assert api.sha(ROOT/'protocol.json')==load('pre_run_freeze.json')['protocol_sha256']
    for key in ('source_hashes','data_hashes'):
        for path,want in p[key].items():assert api.sha(path)==want,path
    return p

def calibrate():
    verify();assert not (ROOT/'selection.json').exists()
    engine,guard,cfg,service,fork,pre,meta,cache,pdf_initial,qmf_initial=prepared()
    partitions=known.load('selection.json')['partitions'];recs=pre['recording_streams']
    library=[]
    for rec,seed,original,base,pdf,qmf in worlds([r for r in recs if r['person'] in partitions['statefit']],pre,cfg,engine,pdf_initial,qmf_initial,(CANONICAL,)):
        stream=horizon.build_stream(pdf,pre,base,cfg,(),rec['recording']);library.extend(horizon.library(stream))
    pools={a:[] for a in ARMS};score_raw=[]
    for rec,seed,original,base,pdf,qmf in worlds([r for r in recs if r['person'] in partitions['scorefit']],pre,cfg,engine,pdf_initial,qmf_initial,(CANONICAL,)):
        streams,raw=raw_results(base,pdf,qmf,pre,cfg,library,rec['recording'])
        for arm in ARMS:
            pools[arm].extend(cal.complete_initial_pool(streams[arm],raw[arm]['rows'],provenance=dict(stage='score_fit',canonical_seed=seed,person=rec['person'])))
            score_raw.append(dict(arm=arm,recording=rec['recording'],person=rec['person'],seed=seed,**raw[arm]))
    selection_raw=[]
    for rec,seed,original,base,pdf,qmf in worlds([r for r in recs if r['person'] in partitions['selection']],pre,cfg,engine,pdf_initial,qmf_initial,CAL_SEEDS):
        streams,raw=raw_results(base,pdf,qmf,pre,cfg,library,rec['recording'])
        selection_raw.extend((rec,seed,original,streams[arm],arm,raw[arm]) for arm in ARMS)
    trials=[];grid=[]
    for arm in ARMS:
        for config in cal.configurations():
            issued=[];guarded=[]
            for rec,seed,original,stream,a,raw in selection_raw:
                if a!=arm:continue
                r=cal.calibrated_run(stream,raw['rows'],config,pools[arm],selection=True)
                g=api.db.guarded(original,pre,cfg,r,guard,service,130.)
                identity=dict(arm=arm,recording=rec['recording'],person=rec['person'],seed=seed)
                issued.append(dict(**identity,**r));guarded.append(dict(**identity,**g))
            value=dict(arm=arm,config=config,net=sum(g['increment'] for g in guarded)/len(CAL_SEEDS))
            grid.append(value);trials.append(dict(configuration=value,issued=issued,guarded=guarded))
            print('CAL',arm,config,value['net'],flush=True)
    selected={a:max((r for r in grid if r['arm']==a),key=lambda r:(r['net'],r['config']['probability'],r['config']['bandwidth'])) for a in ARMS}
    strongest=max(CONTROL_ORDER,key=lambda a:(selected[a]['net'],-CONTROL_ORDER.index(a)))
    dump('score_fit_raw.json.gz',score_raw);dump('calibration_trials.json.gz',trials)
    dump('selection.json',dict(selected=selected,strongest_control=strongest,grid=grid,
         prior_library=library,score_pools=pools,partitions=partitions,metadata=meta,quality=pre['q']))
    dump('selection_lock.json',dict(utc=now(),selection_sha256=api.sha(ROOT/'selection.json'),evaluation_results_exist=False))
    print('SELECT',selected,'STRONGEST',strongest,flush=True)

def evaluate():
    verify();assert not (ROOT/'development_results.json.gz').exists()
    assert api.sha(ROOT/'selection.json')==load('selection_lock.json')['selection_sha256']
    engine,guard,cfg,service,fork,pre,meta,cache,pdf_initial,qmf_initial=prepared()
    selection=load('selection.json');issued=[];guarded=[];budgets=[];raw_all=[];checks=[];diagnosis=[]
    records=[r for r in pre['recording_streams'] if r['person'] in selection['partitions']['development']]
    for rec,seed,original,base,pdf,qmf in worlds(records,pre,cfg,engine,pdf_initial,qmf_initial,DEV_SEEDS):
        streams,raw=raw_results(base,pdf,qmf,pre,cfg,selection['prior_library'],rec['recording'])
        for d in streams['paired']['decisions']:
            diagnosis.append(dict(recording=rec['recording'],person=rec['person'],seed=seed,k=d['k'],
                old_hard_disagreement=d['disagreement'],soft_support=d['soft_support'],
                soft_disagreement=d['soft_disagreement'],potential_return=d['localnet'],
                ready=d['temporal']['eligible'],current_soft_source_evidence=d['descriptor'][:,0].tolist()))
        for arm in ARMS:
            r=cal.calibrated_run(streams[arm],raw[arm]['rows'],selection['selected'][arm]['config'],selection['score_pools'][arm])
            g=api.db.guarded(original,pre,cfg,r,guard,service,130.)
            identity=dict(arm=arm,recording=rec['recording'],person=rec['person'],seed=seed)
            issued.append(dict(**identity,**r));guarded.append(dict(**identity,**g));raw_all.append(dict(**identity,**raw[arm]))
            checks.append(dict(**identity,**known.target_check(original,r['rows'],cfg,fork)))
            for B in (0.,110.,130.,260.):budgets.append(dict(**identity,budget=B,**(g if B==130. else api.db.guarded(original,pre,cfg,r,guard,service,B))))
        print('DEV',rec['recording'],seed,flush=True)
    summary={}
    for arm in ARMS:
        gg=[g for g in guarded if g['arm']==arm];rows=[r for g in gg for r in g['rows']]
        report=cal.actual_admission_report(rows,[row for g in gg for row in g['ledger']])
        increments=[sum(g['increment'] for g in gg if g['seed']==s) for s in DEV_SEEDS]
        summary[arm]=dict(report,mean_increment=float(np.mean(increments)),seed_increments=increments)
    dump('development_results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budgets,raw=raw_all,selected=selection['selected']))
    dump('development_summary.json',dict(status='KNOWN selfBACK DEVELOPMENT ONLY; no new physical confirmation',
        summary=summary,selected=selection['selected'],strongest_cal_control=selection['strongest_control'],
        max_service_error=max(g['service_error'] for g in budgets),target_checks=checks,
        policy_trajectories=len(guarded),budget_paths=len(budgets),cache=cache.report()))
    dump('descriptor_diagnosis.json',diagnosis)
    print('SUMMARY',summary,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','calibrate','evaluate'))
    args=parser.parse_args()
    try:globals()[args.phase]()
    except Exception as exc:
        dump(f'{args.phase}_failure.json',dict(utc=now(),type=type(exc).__name__,message=str(exc),status='retained failure; algorithm unchanged'))
        raise
