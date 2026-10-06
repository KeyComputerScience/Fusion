"""New paired-return development on ALREADY INSPECTED selfBACK521.

No fresh confirmation claim; old source/model selections are immutable.
Canonical state-fit participants precede disjoint score-fit/selection people.
All new candidate outcomes remain recorded, including failures and ties.
"""
from pathlib import Path
import argparse,datetime,sys,importlib.util,json,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'risk'))
import conditional_pairing as cp
import immutable_conditional_calibration as cal
import conditional_moment_control as cm

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module('selfback_development_shared',PROJECT/'work/fusion_joint_return_20261005/new_physical/run_selfback_v4.py')
api=old.api;np=api.np;nn=old.nn
ARMS=cp.ARMS+('conditional_moment',)
CAL_SEEDS=old.CAL;DEV_SEEDS=old.TEST;CANONICAL=CAL_SEEDS[0]

def dump(name,x):api.dump(ROOT/name,x)
def load(name):return api.load(ROOT/name)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def prepared():
    engine,guard,cfg,service,fork,pre,meta,cache=old.prepare()
    selection=old.load('selfback/selection.json')
    # This is known-data design: lr .2 comes from previous development, not
    # selected again using these new scores or inspected test outcomes.
    initial=[{k:np.asarray(v) for k,v in model.items()} for model in selection['models']['pdf:0.2']]
    audit_x=np.concatenate([row['x'] for row in pre['prior']]);audit_y=np.concatenate([row['y'] for row in pre['prior']])
    pre=cp.strong_prefix_quality(initial,audit_x,audit_y,pre,nn.forward)
    api.db.helper.CFG=cfg;nn.helper.CFG=cfg
    return engine,guard,cfg,service,fork,pre,meta,cache,initial

def worlds(records,pre,cfg,engine,initial,seeds):
    out=[]
    for record in records:
        for seed in seeds:
            original=engine.world(record['x'],record['y'],record['timestamp'],pre,seed,cfg)
            base=engine.precompute(original,pre,cfg)
            pdf=nn.world(original,pre,'pdf',.2,initial)
            out.append((record,seed,original,base,pdf))
    return out

def freeze():
    assert not (ROOT/'protocol.json').exists();old.verify()
    sources={Path(__file__),Path(cp.__file__),Path(cal.__file__)}
    for mod in list(sys.modules.values()):
        filename=getattr(mod,'__file__',None)
        if filename and str(filename).startswith(str(PROJECT)) and str(filename).endswith('.py'):sources.add(Path(filename))
    data=old.ROOT/'selfback_data'
    dump('protocol.json',dict(utc=now(),status='KNOWN selfBACK development; data and all former outcomes already inspected; no new physical confirmation',
        arms=ARMS,source_hashes={str(p):api.sha(p) for p in sorted(sources)},data_hashes={str(p):api.sha(p) for p in data.glob('*') if p.is_file()},
        common_forecaster='unchanged old nonlinearPDF active-objective prefix model atlr.2, predeclared known-data design; every informationarm sameprobabilities/immutabletargets/anchor',
        quality='recomputed common nonlinearPDF Brier only on permanent model-excluded prefix error rows; oldlinearqualityretainedseparately',
        partitions='ordered calibration people: first4 STATE-FIT; next2 SCORE-FIT; last2 SELECT. Alloriginal13testpeople are known-data DEVELOPMENT. Canonicaldelay151001alone provides each persistent prior/score-fit origin once.',
        raw_method='conditional_pairing.py: originalissuedE=U−PDFanchor,D=hphysical−PDFanchor; persistentstatefitpairedlibrary; onlineonlyfullymaturedtuples; fixedScott/fullmomentdensityband1 andESmass.5',
        grids={arm:cal.configurations() for arm in ARMS},configs_per_arm=9,selection='guarded complete paidnet/3 selectiondelaychains; ties higher residualquantileprobability then largerbandwidth',
        model_work='allmethods same PDFsourceforecaster andcommonlinearservicecandidate/reference,500oldprefixsteps/20onlineprobe; preparatory oldNNtrainingcost disclosed/shared',
        calibrator='signed empiricalconditionalquantile ofimmutableissued (Sraw−X)/N, contextanchor/disagreementfraction/activefraction; prior scorefit pool plus maturedonline; no futurecoveragecertificate',
        budgets=(0.,110.,130.,260.),primary_budget=130.,reserve=130.,
        maturity='state/score-fit tuples require complete maturity <=theirphysicalchainend; allonlinecallbacks onlyaftertheircomplete labels arrive',
        selection_seeds=CAL_SEEDS,development_seeds=DEV_SEEDS,source_model_sha256=api.sha(old.ROOT/'selfback/selection.json'),no_test_selection=True))
    dump('pre_run_freeze.json',dict(utc=now(),protocol_sha256=api.sha(ROOT/'protocol.json')))
    print('FROZEN',api.sha(ROOT/'protocol.json'),flush=True)

def verify():
    p=load('protocol.json');assert api.sha(ROOT/'protocol.json')==load('pre_run_freeze.json')['protocol_sha256']
    for key in ('source_hashes','data_hashes'):
        for path,want in p[key].items():assert api.sha(path)==want,path
    return p

def raw_results(stream,cfg):
    out={arm:cp.raw_run(stream,arm,cfg) for arm in cp.ARMS}
    out['conditional_moment']=cm.raw_run(stream,out['paired'],cfg)
    return out

def target_check(original,rows,cfg,fork):
    # New calibrated scores already include their single uncertainty term.
    canonical=[dict(r,q_issued=0.) for r in rows]
    return old.base.target_checks(original,canonical,cfg,fork)

def calibrate():
    verify();assert not (ROOT/'selection.json').exists()
    engine,guard,cfg,service,fork,pre,meta,cache,initial=prepared();ids=meta['calibration_ids']
    state_ids=ids[:4];score_ids=ids[4:6];select_ids=ids[6:]
    recs=pre['recording_streams']
    stateworlds=worlds([r for r in recs if r['person'] in state_ids],pre,cfg,engine,initial,(CANONICAL,))
    library=[]
    for rec,seed,original,base,pdf in stateworlds:
        stream=cp.build_stream(pdf,pre,base,cfg,(),rec['recording']);library.extend(cp.library(stream))
    scoreworlds=worlds([r for r in recs if r['person'] in score_ids],pre,cfg,engine,initial,(CANONICAL,))
    pools={arm:[] for arm in ARMS};score_raw=[]
    for rec,seed,original,base,pdf in scoreworlds:
        stream=cp.build_stream(pdf,pre,base,cfg,library,rec['recording'])
        raw_by_arm=raw_results(stream,cfg)
        for arm in ARMS:
            raw=raw_by_arm[arm]
            pool=cal.complete_initial_pool(stream,raw['rows'],boundary=stream['windows'],provenance=dict(stage='score_fit',canonical_seed=seed,person=rec['person']))
            for row in pool:row.update(physical_partition='score_fit',canonical_seed=seed,person=rec['person'])
            pools[arm].extend(pool);score_raw.append(dict(recording=rec['recording'],person=rec['person'],seed=seed,arm=arm,**raw))
    selectionworlds=worlds([r for r in recs if r['person'] in select_ids],pre,cfg,engine,initial,CAL_SEEDS)
    selectionraw=[]
    for rec,seed,original,base,pdf in selectionworlds:
        stream=cp.build_stream(pdf,pre,base,cfg,library,rec['recording'])
        raw_by_arm=raw_results(stream,cfg)
        for arm in ARMS:selectionraw.append((rec,seed,original,stream,arm,raw_by_arm[arm]))
    grid=[];trials=[]
    for arm in ARMS:
        for conf in cal.configurations():
            guarded=[];issued=[]
            for rec,seed,original,stream,a,raw in selectionraw:
                if a!=arm:continue
                r=cal.calibrated_run(stream,raw['rows'],conf,pools[arm],False)
                g=api.db.guarded(original,pre,cfg,r,guard,service,130.)
                guarded.append(dict(arm=arm,recording=rec['recording'],person=rec['person'],seed=seed,**g));issued.append(dict(arm=arm,recording=rec['recording'],person=rec['person'],seed=seed,**r))
            entry=dict(arm=arm,config=conf,net=sum(g['increment'] for g in guarded)/3)
            grid.append(entry);trials.append(dict(configuration=entry,guarded=guarded,issued=issued));print('CAL',arm,conf,entry['net'],flush=True)
    selected={arm:max((entry for entry in grid if entry['arm']==arm),key=lambda x:(x['net'],x['config']['probability'],x['config']['bandwidth'])) for arm in ARMS}
    dump('score_fit_raw.json.gz',score_raw);dump('calibration_trials.json.gz',trials)
    dump('selection.json',dict(selected=selected,grid=grid,prior_library=library,score_pools=pools,partitions=dict(statefit=state_ids,scorefit=score_ids,selection=select_ids,development=meta['test_ids']),quality=pre['q'],previous_linear_quality=pre['previous_linear_quality'],metadata=meta))
    dump('selection_lock.json',dict(utc=now(),selection_sha256=api.sha(ROOT/'selection.json'),results_exist=False,cache=cache.report()))
    print('SELECT',selected,flush=True)

def development():
    verify();assert not (ROOT/'development_results.json.gz').exists();assert api.sha(ROOT/'selection.json')==load('selection_lock.json')['selection_sha256']
    engine,guard,cfg,service,fork,pre,meta,cache,initial=prepared();selection=load('selection.json');library=selection['prior_library'];pools=selection['score_pools'];selected=selection['selected']
    recs=[r for r in pre['recording_streams'] if r['person'] in meta['test_ids']]
    issued=[];guarded=[];budget=[];checks=[];raws=[]
    for rec in recs:
        for rec0,seed,original,base,pdf in worlds([rec],pre,cfg,engine,initial,DEV_SEEDS):
            stream=cp.build_stream(pdf,pre,base,cfg,library,rec['recording'])
            raw_by_arm=raw_results(stream,cfg)
            for arm in ARMS:
                raw=raw_by_arm[arm];r=cal.calibrated_run(stream,raw['rows'],selected[arm]['config'],pools[arm],False)
                g=api.db.guarded(original,pre,cfg,r,guard,service,130.)
                identity=dict(arm=arm,recording=rec['recording'],person=rec['person'],seed=seed)
                issued.append(dict(**identity,**r));guarded.append(dict(**identity,**g));raws.append(dict(**identity,**raw));checks.append(dict(**identity,**target_check(original,r['rows'],cfg,fork)))
                for B in (0.,110.,130.,260.):budget.append(dict(**identity,budget=B,**(g if B==130. else api.db.guarded(original,pre,cfg,r,guard,service,B))))
            print('DEV',rec['recording'],seed,flush=True)
    summary={}
    for arm in ARMS:
        gg=[g for g in guarded if g['arm']==arm];rows=[r for g in gg for r in g['rows']];report=cal.actual_admission_report(rows,[row for g in gg for row in g['ledger']])
        increments=[sum(g['increment'] for g in gg if g['seed']==seed) for seed in DEV_SEEDS]
        summary[arm]=dict(report,mean_increment=float(np.mean(increments)),seed_increments=increments)
    dump('development_results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budget,raw=raws,selected=selected))
    dump('development_summary.json',dict(summary=summary,selected=selected,max_service_error=max(g['service_error'] for g in budget),target_checks=checks,cache=cache.report()))
    print('SUMMARY',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','calibrate','development']);args=p.parse_args();globals()[args.phase]()
