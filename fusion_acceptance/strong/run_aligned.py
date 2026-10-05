"""Fresh guarded RSS selection with frozen numerical policy functions.

Known RSS development benchmark, not new physical-environment validation.
Writes only below this directory. All calibration and test outcomes retained.
"""
from pathlib import Path
import argparse, datetime, hashlib, importlib.util, sys, contextlib
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[3]
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'work/fusion_joint_tail_20261005'))
import run_replay as api
import numpy as np
ARMS=('joint','legacy_factorized','legacy_sandwich')
CAL=(87001,87002,87003);TEST=(88001,88002,88003,88004,88005)
PRIORS=(.05,.1,.2);FLOORS=(0.,.64,1.2815515655446004)
matrix=api.c.parent.parent.matrix
factor=api.c.parent.parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,obj):api.dump(ROOT/name,obj)
def load(name):return api.load(ROOT/name)
def sha(p):return api.sha(p)
def dispatch(d,pre,cfg,arm,q):
    if arm=='legacy_factorized':return factor.factorized_forecast(d,pre,cfg,'factorized_information',q)
    if arm=='legacy_sandwich':return matrix.matrix_forecast(d,pre,cfg,'block_sandwich',q)
    raise ValueError(arm)
@contextlib.contextmanager
def bindings(state=None):
    before_state,before_forecast=api.core.state,api.core.forecast
    if state is not None:api.core.state=state
    api.core.forecast=dispatch
    try:yield
    finally:api.core.state,api.core.forecast=before_state,before_forecast
def prepared():
    result=api.db.prepared('rss348')
    engine,recovery,guard,cfg,service,fork,data,pre,records,cache=result
    cfg=api.p.base_cfg(cfg)
    return engine,guard,cfg,service,fork,data,pre,cache
def freeze():
    assert not (ROOT/'protocol.json').exists()
    api.verify()
    sources={Path(__file__)}
    for obj in list(sys.modules.values()):
        filename=getattr(obj,'__file__',None)
        if filename and str(filename).startswith(str(PROJECT)) and str(filename).endswith('.py'):sources.add(Path(filename))
    # The frozen shared preparation selects this original cache.
    data_dir=PROJECT/'work/fusion_temporal_20261003/physical/data/rss348'
    ds=list(data_dir.glob('*.npz'))+list(data_dir.glob('*metadata*.json'))
    dump('protocol.json',dict(utc=now(),status='known RSS guarded native full-policy benchmark; frozen before calibration and this fresh test replay',
        source_hashes={str(p):sha(p) for p in sorted(sources)},data_hashes={str(p):sha(p) for p in sorted(ds)},arms=ARMS,
        calibration_seeds=CAL,test_seeds=TEST,grid_joint=dict(bandwidth=api.BANDS,tail_cap=api.CAPS),grid_legacy=dict(prior_sd=PRIORS,q_floor=FLOORS),
        trials_per_arm=27,shared='one prepared physical world/base per delay seed; immutable forecasts, physical history, quality, context/support/age weighting, service/fork targets, model work and Fixed130 B130 ledger',
        state_adapter='numerically unchanged cumulant_state for joint; factorized_state for legacy; unused posterior/other-tail diagnostics skipped, no source edited',
        firsthalf_fit='all callbacks used for initial fit require complete maturity at/before floor(windows/2); ready records only; CJRT largest alpha violation <=.1; native legacy higher90% standardized-error quantile at q_floor',
        online='CJRT ready-only alpha += .02*(.1-violation); legacy native all-issued q += .05*(violation-.1), q floor; callbacks only after complete label maturity',
        selection='all policies maximize mean independently request-reconstructed guarded actual return over second-half eligibility on exactly three original calibration seeds',
        tie_joint='smaller cap then larger bandwidth',tie_legacy='larger q_floor then smaller prior_sd',legacy_bandwidth=1.,budget=130.,diagnostic_budgets=(0.,110.,130.,260.),reserve=130.,
        initial_model_states='prefix-fitted restart for calibration and test; no calibration-trained models carried into test',
        limitations=['RSS was previously used for method development; this is not independent physical validation','native risk/calibration functionals and tuning axes differ; only common preparation/history and guarded tuning objective are matched','five delay seeds share one physical trace; all outcomes retained'],
        runtime=dict(python=sys.version,numpy=np.__version__)))
    dump('pre_calibration_freeze.json',dict(utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),selection_exists=False,test_exists=False))
    print('FROZEN',sha(ROOT/'protocol.json'),flush=True)
def verify():
    pr=load('protocol.json');assert sha(ROOT/'protocol.json')==load('pre_calibration_freeze.json')['protocol_sha256']
    for f in ('source_hashes','data_hashes'):
        for path,want in pr[f].items():assert sha(path)==want,path
    return pr
def worlds(part):
    engine,guard,cfg,service,fork,data,pre,cache=prepared();key='cal' if part=='calibration' else 'test'
    seeds=CAL if part=='calibration' else TEST
    events=[engine.world(pre[key+'_x'],pre[key+'_y'],pre[key+'_timestamp'],pre,s,cfg) for s in seeds]
    bases=[engine.precompute(e,pre,cfg) for e in events]
    return engine,guard,cfg,service,fork,data,pre,cache,events,bases
def stream(events,pre,base,cfg,joint):
    state=api.c.parent.cumulant_state if joint else factor.factorized_state
    with bindings(state):s=api.core.augment(events,pre,base,cfg)
    if joint:
        for d in s['decisions']:d['tail_laws']={'joint':api.law(d,pre,cfg,'joint')}
    return s
def joint_initial(streams,cap):
    eligible=[d for s in streams for d in s['decisions'] if d['maturity']<=s['windows']//2 and d['temporal']['eligible']]
    if not eligible:return 1e-4,0
    left,right=1e-4,cap
    for _ in range(35):
        mid=(left+right)/2
        violation=np.mean([d['truegross']<d['N']*d['tail_laws']['joint'].lower_tail(mid) for d in eligible])
        if violation<=.1:left=mid
        else:right=mid
    return left,len(eligible)
def run_native(engine,s,pre,cfg,arm,q0,select=False):
    if arm=='joint':return api.run(s,'joint',cfg['tail_cap'],q0,select)
    with bindings():return api.core.run(engine,s,pre,cfg,arm,q0,select)
def target_checks(events,rows,cfg,fork):
    maximum=0.
    for row in rows:
        f=fork(events,row,cfg)
        maximum=max(maximum,abs(f['gross']-row['truegross']),abs(f['actual']-row['local_net']),f['identity_error'])
        assert f['covered']==row['lower_covered']
    assert maximum<1e-8
    return dict(forks=len(rows),maximum_error=maximum)
def calibrate():
    verify();assert not (ROOT/'selection.json').exists()
    engine,guard,cfg0,service,fork,data,pre,cache,events,bases=worlds('calibration');trials=[];grid=[]
    for b in api.BANDS:
        cfg=dict(cfg0,slice_bandwidth=b)
        ss=[stream(e,pre,base,cfg,True) for e,base in zip(events,bases)]
        for cap in api.CAPS:
            cc=dict(cfg,tail_cap=cap);q0,count=joint_initial(ss,cap)
            rr=[run_native(engine,s,pre,cc,'joint',q0,True) for s in ss]
            gg=[api.db.guarded(e,pre,cc,r,guard,service,130.) for e,r in zip(events,rr)]
            conf=dict(arm='joint',bandwidth=b,cap=cap,prior_sd=cfg['prior_sd'],q_floor=None,initial=q0,fit_count=count,net=sum(g['increment'] for g in gg)/3)
            grid.append(conf);trials.append(dict(configuration=conf,issued=rr,guarded=gg))
        print('CAL_JOINT',b,flush=True)
    for sd in PRIORS:
        cfg=dict(cfg0,slice_bandwidth=1.,prior_sd=sd)
        ss=[stream(e,pre,base,cfg,False) for e,base in zip(events,bases)]
        for floor in FLOORS:
            cc=dict(cfg,q_floor=floor)
            for arm in ARMS[1:]:
                with bindings():q0,scores=api.core.initial_q(ss,pre,cc,arm)
                rr=[run_native(engine,s,pre,cc,arm,q0,True) for s in ss]
                gg=[api.db.guarded(e,pre,cc,r,guard,service,130.) for e,r in zip(events,rr)]
                conf=dict(arm=arm,bandwidth=1.,cap=None,prior_sd=sd,q_floor=floor,initial=q0,fit_count=len(scores),net=sum(g['increment'] for g in gg)/3)
                grid.append(conf);trials.append(dict(configuration=conf,fit_scores=scores,issued=rr,guarded=gg))
        print('CAL_LEGACY',sd,flush=True)
    selected={}
    for arm in ARMS:
        key=(lambda x:(x['net'],-x['cap'],x['bandwidth'])) if arm=='joint' else (lambda x:(x['net'],x['q_floor'],-x['prior_sd']))
        selected[arm]=max((x for x in grid if x['arm']==arm),key=key)
    dump('calibration_trials.json.gz',trials);dump('selection.json',dict(selected=selected,grid=grid,split=pre['split'],quality=pre['q'],data_hashes=data['hashes'],cfg=cfg0))
    dump('selection_lock.json',dict(utc=now(),selection_sha256=sha(ROOT/'selection.json'),protocol_sha256=sha(ROOT/'protocol.json'),trials_sha256=sha(ROOT/'calibration_trials.json.gz'),test_results_exist=False,cache=cache.report()))
    print('SELECT',selected,flush=True)
def summary(issued,guarded):
    out={}
    for arm in ARMS:
        gg=[g for g in guarded if g['arm']==arm];ii=[r for r in issued if r['arm']==arm];rows=[r for x in gg for r in x['rows']]
        groups=dict(issued=rows,ready=[r for r in rows if r.get('information_ready')],informative=[r for r in rows if r['disagreement']>0],admitted=[r for r in rows if r['action']])
        acts=groups['admitted'];excess=[max(0.,r['gate_score']-(r['truegross']-5.)) for r in acts]
        out[arm]=dict(seed_increments=[g['increment'] for g in gg],mean_increment=float(np.mean([g['increment'] for g in gg])),admissions=len(acts),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),zero=sum(r['local_net']==0 for r in acts),negative_loss=sum(max(0.,-r['local_net']) for r in acts),coverage={k:[sum(r['lower_covered'] for r in v),len(v)] for k,v in groups.items()},admitted_max_excess=max(excess,default=0.),admitted_total_excess=sum(excess),refused=sum(g['refused'] for g in gg))
    return out
def decomposition(guarded):
    by={(g['arm'],g['seed']):g for g in guarded};out={}
    for arm in ARMS[1:]:
        cell={k:dict(count=0,value=0.) for k in ('added_gain','avoided_loss','missed_gain','incurred_loss')};zero=0;changes=[]
        for seed in TEST:
            J,C=by['joint',seed],by[arm,seed];jr={r['k']:r for r in J['rows']};cr={r['k']:r for r in C['rows']}
            assert jr.keys()==cr.keys()
            for k,r in jr.items():
                c=cr[k];assert r['local_net']==c['local_net'] and r['truegross']==c['truegross']
                if r['action']==c['action']:continue
                D=r['local_net'];changes.append(dict(seed=seed,k=k,joint=r['action'],control=c['action'],actual_net=D,joint_score=r['gate_score'],control_score=c['gate_score']))
                if D==0:zero+=1;continue
                key=('added_gain' if D>0 else 'incurred_loss') if r['action'] else ('missed_gain' if D>0 else 'avoided_loss')
                cell[key]['count']+=1;cell[key]['value']+=abs(D)
        val=cell['added_gain']['value']+cell['avoided_loss']['value']-cell['missed_gain']['value']-cell['incurred_loss']['value']
        actual=sum(by['joint',s]['increment']-by[arm,s]['increment'] for s in TEST);assert abs(val-actual)<1e-8
        out[arm]=dict(terms=cell,zero_changes=zero,total_difference=val,mean_difference=val/len(TEST),changed_actions=changes)
    return out
def test():
    verify();assert not (ROOT/'results.json.gz').exists();assert load('selection_lock.json')['selection_sha256']==sha(ROOT/'selection.json')
    engine,guard,cfg0,service,fork,data,pre,cache,events,bases=worlds('test');selected=load('selection.json')['selected'];issued=[];guarded=[];audits=[];budget_rows=[]
    for seed,e,base in zip(TEST,events,bases):
        streams={}
        for arm,s in selected.items():
            cfg=dict(cfg0,slice_bandwidth=s['bandwidth'],prior_sd=s['prior_sd'])
            if arm=='joint':cfg['tail_cap']=s['cap']
            else:cfg['q_floor']=s['q_floor']
            key=(arm=='joint',s['bandwidth'],s['prior_sd'])
            if key not in streams:streams[key]=stream(e,pre,base,cfg,arm=='joint')
            r=run_native(engine,streams[key],pre,cfg,arm,s['initial']);g=api.db.guarded(e,pre,cfg,r,guard,service,130.)
            issued.append(dict(arm=arm,seed=seed,**r));guarded.append(dict(arm=arm,seed=seed,**g));audits.append(dict(arm=arm,seed=seed,**target_checks(e,r['rows'],cfg,fork)))
            for B in (0.,110.,130.,260.):
                bg=g if B==130. else api.db.guarded(e,pre,cfg,r,guard,service,B)
                budget_rows.append(dict(arm=arm,seed=seed,budget=B,**bg))
        print('TEST',seed,flush=True)
    sm=summary(issued,guarded);decomp=decomposition(guarded)
    budget_summary={str(B):summary(issued,[g for g in budget_rows if g['budget']==B]) for B in (0.,110.,130.,260.)}
    dump('results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budget_rows,selected=selected));dump('summary.json',dict(summary=sm,budget_summary=budget_summary,decomposition=decomp,selected=selected,max_service_error=max(g['service_error'] for g in budget_rows),target_checks=audits,cache=cache.report()))
    print('SUMMARY',sm,'DECOMPOSITION',decomp,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','verify','calibrate','test']);a=p.parse_args()
    {'freeze':freeze,'verify':lambda:print(verify()['status']),'calibrate':calibrate,'test':test}[a.phase]()
