"""One final predeclared used-data calibration candidate; no further iteration."""
from pathlib import Path
import argparse,sys,datetime,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;V0=ROOT.parent
sys.path.insert(0,str(V0));sys.path.insert(0,str(V0.parent/'risk'))
import run_development as v0
import positive_calibration as cal
api=v0.api;np=v0.np;ARMS=v0.ARMS

def dump(name,value):api.dump(ROOT/name,value)
def load(name):return api.load(ROOT/name)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def freeze():
    assert not (ROOT/'protocol.json').exists();v0.verify()
    files=[ROOT/'run_v1.py',ROOT/'positive_calibration.py']
    inputs=[V0/'protocol.json',V0/'pre_run_freeze.json',V0/'selection.json',V0/'selection_lock.json',
            V0/'calibration_trials.json.gz',V0/'development_results.json.gz',V0/'development_summary.json']
    dump('protocol.json',dict(utc=now(),status='Known-selfBACK development v1 designed after retainedV0 outcomes; not new physical confirmation; no OPP access',
        source_hashes={str(p):api.sha(p) for p in files},input_hashes={str(p):api.sha(p) for p in inputs},
        unchanged='same V0 immutable rawscores/fullmultivariatephysicalsourcepath/state/forecasters/modelwork/pools/readiness; no density/backbone retraining',
        only_change='one calibratedscore withqplus=max(0,empirical signedquantile); no extra predicate/guard; physicalsupportclip andshared130ledger unchanged',
        arms=ARMS,configurations={a:cal.configurations() for a in ARMS},trials_per_arm=9,
        selection='all same3oldcal delays andwholelast2selectionpeople; maximizeguardedpaidreturn, tieshigherp then largerband; strongestcontrol CAL-only declaredV0order',
        development='all13knownformerheldoutpeople x5delays x9arms, all outcomes retained',
        stop_rule='After this one refinement stop and report all V0/V1 outcomes; no guaranteedpositive search',
        risk_scope='Monotonicity relative raw ES andsigned correction does not guarantee positive return or calibrated accepted risk',
        budgets=(0.,110.,130.,260.),primary_budget=130.,reserve=130.))
    dump('pre_run_freeze.json',dict(utc=now(),protocol_sha256=api.sha(ROOT/'protocol.json')))
    print('FREEZE_V1',api.sha(ROOT/'protocol.json'),flush=True)

def verify():
    p=load('protocol.json');assert api.sha(ROOT/'protocol.json')==load('pre_run_freeze.json')['protocol_sha256'];v0.verify()
    for key in ('source_hashes','input_hashes'):
        for path,want in p[key].items():assert api.sha(path)==want,path
    return p

def restore_raw(issued):
    rows=[]
    for row in issued['rows']:
        r=dict(row);score=r['raw_gate_score']
        r.update(gate_score=score,gain=score+5.,q_issued=0.,action=bool(r['information_ready'] and score>0))
        rows.append(r)
    return dict(rows=rows)

def minimal_stream(raw,windows,recording):
    return dict(windows=windows,recording=recording,
        decisions=[dict(k=r['k'],maturity=r['maturity'],N=r['N'],cal_context=r['cal_context']) for r in raw['rows']])

def context():
    engine,guard,cfg,service,fork,pre,meta,cache,pdf_initial,qmf_initial=v0.prepared()
    records={r['recording']:r for r in pre['recording_streams']}
    return engine,guard,cfg,service,fork,pre,records

def calibrate():
    verify();assert not (ROOT/'selection.json').exists()
    engine,guard,cfg,service,fork,pre,records=context();old=v0.load('selection.json')
    trials=v0.load('calibration_trials.json.gz');raws={};worlds={}
    for trial in trials:
        for issued in trial['issued']:
            key=(issued['arm'],issued['recording'],issued['seed'])
            if key not in raws:raws[key]=restore_raw(issued)
    grid=[];retained=[]
    for arm in ARMS:
        for config in cal.configurations():
            issued_all=[];guarded=[]
            for (a,recording,seed),raw in raws.items():
                if a!=arm:continue
                rec=records[recording];wkey=(recording,seed)
                if wkey not in worlds:worlds[wkey]=engine.world(rec['x'],rec['y'],rec['timestamp'],pre,seed,cfg)
                original=worlds[wkey];stream=minimal_stream(raw,len(original),recording)
                r=cal.calibrated_run(stream,raw['rows'],config,old['score_pools'][arm],selection=True)
                g=api.db.guarded(original,pre,cfg,r,guard,service,130.)
                identity=dict(arm=arm,recording=recording,person=rec['person'],seed=seed)
                issued_all.append(dict(**identity,**r));guarded.append(dict(**identity,**g))
            row=dict(arm=arm,config=config,net=sum(g['increment'] for g in guarded)/len(v0.CAL_SEEDS))
            grid.append(row);retained.append(dict(configuration=row,issued=issued_all,guarded=guarded))
            print('CAL',row,flush=True)
    selected={a:max((g for g in grid if g['arm']==a),key=lambda g:(g['net'],g['config']['probability'],g['config']['bandwidth'])) for a in ARMS}
    strongest=max(v0.CONTROL_ORDER,key=lambda a:(selected[a]['net'],-v0.CONTROL_ORDER.index(a)))
    dump('calibration_trials.json.gz',retained);dump('selection.json',dict(selected=selected,grid=grid,strongest_control=strongest,
        score_pools=old['score_pools'],partitions=old['partitions'],unchanged_raw_inputs_sha256=api.sha(V0/'development_results.json.gz')))
    dump('selection_lock.json',dict(utc=now(),selection_sha256=api.sha(ROOT/'selection.json'),development_results_exist=False))
    print('SELECT',selected,strongest,flush=True)

def evaluate():
    verify();assert not (ROOT/'development_results.json.gz').exists()
    assert api.sha(ROOT/'selection.json')==load('selection_lock.json')['selection_sha256']
    engine,guard,cfg,service,fork,pre,records=context();selection=load('selection.json');old=v0.load('development_results.json.gz')
    worlds={};issued=[];guarded=[];budgets=[];checks=[]
    for raw in old['raw']:
        arm=raw['arm'];recording=raw['recording'];seed=raw['seed'];rec=records[recording];key=(recording,seed)
        if key not in worlds:worlds[key]=engine.world(rec['x'],rec['y'],rec['timestamp'],pre,seed,cfg)
        original=worlds[key];stream=minimal_stream(raw,len(original),recording)
        r=cal.calibrated_run(stream,raw['rows'],selection['selected'][arm]['config'],selection['score_pools'][arm])
        g=api.db.guarded(original,pre,cfg,r,guard,service,130.);identity=dict(arm=arm,recording=recording,person=rec['person'],seed=seed)
        issued.append(dict(**identity,**r));guarded.append(dict(**identity,**g))
        checks.append(dict(**identity,**v0.known.target_check(original,r['rows'],cfg,fork)))
        for B in (0.,110.,130.,260.):budgets.append(dict(**identity,budget=B,**(g if B==130. else api.db.guarded(original,pre,cfg,r,guard,service,B))))
    summary={}
    for a in ARMS:
        gs=[g for g in guarded if g['arm']==a];rows=[r for g in gs for r in g['rows']]
        summary[a]=dict(cal.actual_admission_report(rows,[x for g in gs for x in g['ledger']]),
                       mean_increment=float(np.mean([sum(g['increment'] for g in gs if g['seed']==s) for s in v0.DEV_SEEDS])),
                       seed_increments=[sum(g['increment'] for g in gs if g['seed']==s) for s in v0.DEV_SEEDS])
    dump('development_results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budgets,raw=old['raw'],selected=selection['selected']))
    dump('development_summary.json',dict(status='Known-selfBACK used development v1 ONLY; not fresh confirmation',summary=summary,
        strongest_cal_control=selection['strongest_control'],selected=selection['selected'],policy_trajectories=len(guarded),
        budget_paths=len(budgets),max_service_error=max(g['service_error'] for g in budgets),target_checks=checks))
    print('SUMMARY',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('freeze','calibrate','evaluate'));args=p.parse_args()
    try:globals()[args.phase]()
    except Exception as exc:
        dump(f'{args.phase}_failure.json',dict(utc=now(),type=type(exc).__name__,message=str(exc),status='retained failure; no replacement'))
        raise
