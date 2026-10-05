"""Unseen UCI308 batch validation; algorithms and protocol lock before acquisition."""
from pathlib import Path
import datetime,importlib.util,sys,zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
base=module('flow308_shared_policy',ROOT.parent/'strong/run_aligned.py')
api=base.api;np=api.np
adapter=module('flow308_adapter',ROOT/'adapter.py')
TAIL=api.ARMS;NATIVE=('legacy_factorized','legacy_sandwich');EXTERNAL=('dbf','eavg');ARMS=TAIL+NATIVE+EXTERNAL
CAL=(142001,142002,142003);TEST=(143001,143002,143003,143004,143005)
BUDGETS=(0.,110.,130.,260.)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,obj):api.dump(ROOT/name,obj)
def load(name):return api.load(ROOT/name)
def sha(p):return api.sha(p)
def choice():return load('algorithm_choice.json')
def install():
    if choice()['mode']=='geometry':
        module('flow308_geometry_operator',ROOT.parent/'geometry_joint/run_geometry.py')
def freeze():
    assert not (ROOT/'protocol.json').exists() and not (ROOT/'raw/flow308.zip').exists()
    install();base.verify();files={Path(__file__),ROOT/'adapter.py',ROOT/'algorithm_choice.json'}
    for obj in list(sys.modules.values()):
        name=getattr(obj,'__file__',None)
        if name and str(name).startswith(str(PROJECT)) and str(name).endswith('.py'):files.add(Path(name))
    dump('protocol.json',dict(utc=now(),status='frozen before raw acquisition and before any UCI308 evaluation',
        source_hashes={str(x):sha(x) for x in sorted(files)},algorithm=choice(),dataset='UCI308 Gas sensor array under flow modulation',
        official_url='https://archive.ics.uci.edu/dataset/308/gas+sensor+array+under+flow+modulation',doi='10.24432/C5BG7G',
        fit_batches=adapter.FIT,calibration_batches=adapter.CAL,test_batches=adapter.TEST,batch_counts=adapter.BATCH_COUNTS,
        sampling=dict(raw_hz=25,stride=5,retained_hz=5,seconds=300,points=1500,anti_alias=False),
        source_panels=[[v+1 for v in g] for g in adapter.GROUPS],task='4-class experiment gas identity throughout exposure/recovery; no concentration feature',
        histories='separate experiment chains; lags, posterior, calibration model work and ledger restart; leases never cross experiment',
        arms=ARMS,native_grid=dict(prior_sd=base.PRIORS,q_floor=base.FLOORS),external_grid=dict(kappa_multiplier=(.25,1.,4.),margin=(0.,4.,12.)),
        external_scope='published DBF2025 operator and matched evidence averaging on shared frozen physical classifiers; explicit evidence adapter, no original EDL backbone claim',
        calibration_seeds=CAL,test_seeds=TEST,configs_per_arm=9,trials_per_arm=27,
        service=dict(window=32,horizon=4,fees=(2,1),extra_drops=2,lower_cost=5,reserve=130,budgets=BUDGETS,primary_budget=130),
        select='guarded second-half actual service increment /3 delay schedules; native per-method deterministic tie rule; all trials retained',
        maturity='initial calibration only complete first-half outcomes; tail ready-only alpha callbacks, native q all-issued callbacks; external initial q all-issued per published adapter',
        tail_grid=choice()['grid'],quality='prefix heldout rows only, fixed; no test fit or normalization',
        output='all selected arms, actions, all/ready/informative/admitted coverage and exceedance, ledger budgets, complete fork and request service checks'))
    dump('pre_acquisition_freeze.json',dict(utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),raw_exists=False,test_exists=False))
    print('FROZEN',sha(ROOT/'protocol.json'),flush=True)
def verify():
    p=load('protocol.json');assert sha(ROOT/'protocol.json')==load('pre_acquisition_freeze.json')['protocol_sha256']
    for path,want in p['source_hashes'].items():assert sha(path)==want,path
    return p
def prepare():
    verify();install();engine,recovery,guard,source,cfg,study,service,fork=api.p.r.binding();cfg=api.p.base_cfg(cfg)
    cache=api.db.ExactGradientCache(engine);engine.gradient=cache
    pre,meta=adapter.prepared(ROOT/'data',engine,cfg);api.db.helper.CFG=cfg
    return engine,guard,cfg,service,fork,pre,meta,cache
def worlds(part):
    engine,guard,cfg,service,fork,pre,meta,cache=prepare();records=[]
    for rec in pre['recording_streams']:
        if rec['partition']!=part:continue
        for seed in CAL if part=='calibration' else TEST:
            events=engine.world(rec['x'],rec['y'],rec['timestamp'],pre,seed,cfg)
            records.append((rec,seed,events,engine.precompute(events,pre,cfg)))
    return engine,guard,cfg,service,fork,pre,meta,cache,records
def configurations():
    mode=choice()['mode']
    if mode=='geometry':return [dict(prior_sd=.1,slice_bandwidth=bt,b_parallel=bp,b_perp=bt,tail_cap=1.) for bp in api.BANDS for bt in api.BANDS]
    return [dict(prior_sd=.1,slice_bandwidth=b,tail_cap=cap) for b in api.BANDS for cap in api.CAPS]
def tailstream(events,pre,initial,cfg):
    with base.bindings(api.c.parent.cumulant_state):s=api.core.augment(events,pre,initial,cfg)
    return api.decorate(s,pre,cfg)
def run_tail(s,arm,cfg,alpha,select=False):return api.run(s,arm,cfg['tail_cap'],alpha,select)
def calibration():
    verify();assert not (ROOT/'selection.json').exists();engine,guard,cfg0,service,fork,pre,meta,cache,data=worlds('calibration');grid=[];trials=[]
    for conf in configurations():
        cfg=dict(cfg0,**conf);ss=[tailstream(e,pre,init,cfg) for _,_,e,init in data];initials,count=api.fit(ss,cfg['tail_cap'])
        for arm in TAIL:
            rr=[run_tail(s,arm,cfg,initials[arm],True) for s in ss];gg=[api.db.guarded(e,pre,cfg,r,guard,service,130.) for (_,_,e,_),r in zip(data,rr)]
            entry=dict(arm=arm,config=conf,initial=initials[arm],fit_count=count,net=sum(g['increment'] for g in gg)/3)
            grid.append(entry);trials.append(dict(configuration=entry,issued=rr,guarded=gg))
        print('CAL_TAIL',conf,flush=True)
    for sd in base.PRIORS:
        cfg=dict(cfg0,prior_sd=sd,slice_bandwidth=1.)
        ss=[base.stream(e,pre,init,cfg,False) for _,_,e,init in data]
        for floor in base.FLOORS:
            cc=dict(cfg,q_floor=floor)
            for arm in NATIVE:
                with base.bindings():q0,values=api.core.initial_q(ss,pre,cc,arm)
                rr=[base.run_native(engine,s,pre,cc,arm,q0,True) for s in ss];gg=[api.db.guarded(e,pre,cc,r,guard,service,130.) for (_,_,e,_),r in zip(data,rr)]
                entry=dict(arm=arm,config=dict(prior_sd=sd,slice_bandwidth=1.,q_floor=floor),initial=q0,fit_count=len(values),net=sum(g['increment'] for g in gg)/3)
                grid.append(entry);trials.append(dict(configuration=entry,issued=rr,guarded=gg))
        print('CAL_NATIVE',sd,flush=True)
    for rule in EXTERNAL:
        for mult in (.25,1.,4.):
            ss=[api.db.streams(e,pre,rule,pre['classes']*mult) for _,_,e,_ in data]
            q0,values=api.db.helper.ext.qi(ss,pre,cfg0,rule+'_calibrated')
            for margin in (0.,4.,12.):
                cc=dict(cfg0,threshold=margin)
                rr=[api.db.execute(engine,s,pre,cc,rule,q0,True) for s in ss];gg=[api.db.guarded(e,pre,cc,r,guard,service,130.) for (_,_,e,_),r in zip(data,rr)]
                entry=dict(arm=rule,config=dict(kappa_multiplier=mult,threshold=margin),initial=q0,fit_count=len(values),net=sum(g['increment'] for g in gg)/3)
                grid.append(entry);trials.append(dict(configuration=entry,issued=rr,guarded=gg))
        print('CAL_EXTERNAL',rule,flush=True)
    def key(x):
        c=x['config']
        if x['arm'] in TAIL:return (x['net'],-c['tail_cap'],c['slice_bandwidth'],c.get('b_parallel',0.))
        if x['arm'] in NATIVE:return (x['net'],c['q_floor'],-c['prior_sd'],0.)
        return (x['net'],-c['threshold'],-c['kappa_multiplier'],0.)
    selected={arm:max((x for x in grid if x['arm']==arm),key=key) for arm in ARMS}
    dump('calibration_trials.json.gz',trials);dump('selection.json',dict(selected=selected,grid=grid,split=pre['split'],quality=pre['q'],metadata=meta))
    dump('selection_lock.json',dict(utc=now(),selection_sha256=sha(ROOT/'selection.json'),protocol_sha256=sha(ROOT/'protocol.json'),trials_sha256=sha(ROOT/'calibration_trials.json.gz'),test_results_exist=False))
    print('SELECT',selected,flush=True)
def summary(gg):
    out={}
    for arm in ARMS:
        part=[g for g in gg if g['arm']==arm];rows=[r for g in part for r in g['rows']];acts=[r for r in rows if r['action']]
        groups=dict(issued=rows,ready=[r for r in rows if r.get('information_ready')],informative=[r for r in rows if r['disagreement']>0],admitted=acts)
        values=[sum(g['increment'] for g in part if g['seed']==seed) for seed in TEST]
        excess=[max(0.,r['gate_score']-(r['truegross']-5)) for r in acts]
        out[arm]=dict(mean_increment=float(np.mean(values)),seed_increments=values,admissions=len(acts),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),zero=sum(r['local_net']==0 for r in acts),negative_loss=sum(max(0.,-r['local_net']) for r in acts),coverage={k:[sum(r['lower_covered'] for r in rr),len(rr)] for k,rr in groups.items()},max_excess=max(excess,default=0.),sum_excess=sum(excess),refused=sum(g['refused'] for g in part))
    return out
def test():
    verify();assert not (ROOT/'results.json.gz').exists();assert sha(ROOT/'selection.json')==load('selection_lock.json')['selection_sha256'];selected=load('selection.json')['selected']
    engine,guard,cfg0,service,fork,pre,meta,cache,data=worlds('test');issued=[];guarded=[];budgets=[];audit=[]
    for rec,seed,events,initial in data:
        ss={}
        for arm,v in selected.items():
            cfg=dict(cfg0,**v['config'])
            if arm in TAIL:
                key=tuple(sorted(v['config'].items()))
                if key not in ss:ss[key]=tailstream(events,pre,initial,cfg)
                result=run_tail(ss[key],arm,cfg,v['initial'])
            elif arm in NATIVE:
                s=base.stream(events,pre,initial,cfg,False);result=base.run_native(engine,s,pre,cfg,arm,v['initial'])
            else:
                s=api.db.streams(events,pre,arm,pre['classes']*cfg['kappa_multiplier']);result=api.db.execute(engine,s,pre,cfg,arm,v['initial'])
            g=api.db.guarded(events,pre,cfg,result,guard,service,130.)
            identity=dict(arm=arm,seed=seed,recording=rec['recording'],batch=rec['batch'])
            issued.append(dict(**identity,**result));guarded.append(dict(**identity,**g))
            audit.append(dict(**identity,**base.target_checks(events,result['rows'],cfg,fork)))
            for B in BUDGETS:budgets.append(dict(**identity,budget=B,**(g if B==130. else api.db.guarded(events,pre,cfg,result,guard,service,B))))
        print('TEST',rec['recording'],seed,flush=True)
    decomposition={};by={(g['arm'],g['seed'],g['recording']):g for g in guarded}
    for arm in ARMS[1:]:
        terms={k:dict(count=0,value=0.) for k in ('added_gain','avoided_loss','missed_gain','incurred_loss')};changes=[]
        for j in [g for g in guarded if g['arm']=='joint']:
            c=by[arm,j['seed'],j['recording']]
            for r,x in zip(j['rows'],c['rows']):
                assert r['k']==x['k'] and r['local_net']==x['local_net']
                if r['action']==x['action']:continue
                D=r['local_net'];changes.append(dict(seed=j['seed'],recording=j['recording'],k=r['k'],D=D,joint=r['action'],control=x['action']))
                if D==0:continue
                k=('added_gain' if D>0 else 'incurred_loss') if r['action'] else ('missed_gain' if D>0 else 'avoided_loss');terms[k]['count']+=1;terms[k]['value']+=abs(D)
        total=terms['added_gain']['value']+terms['avoided_loss']['value']-terms['missed_gain']['value']-terms['incurred_loss']['value']
        assert abs(total-sum(g['increment'] for g in guarded if g['arm']=='joint')+sum(g['increment'] for g in guarded if g['arm']==arm))<1e-8
        decomposition[arm]=dict(terms=terms,pooled_difference=total,mean_difference=total/5,changed_actions=changes)
    dump('results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budgets))
    dump('summary.json',dict(summary=summary(guarded),budget_summary={str(B):summary([g for g in budgets if g['budget']==B]) for B in BUDGETS},decomposition=decomposition,selected=selected,metadata=meta,max_service_error=max(g['service_error'] for g in budgets),target_checks=audit,cache=cache.report()))
    print('SUMMARY',summary(guarded),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','verify','calibrate','test','process']);a=p.parse_args()
    if a.phase=='process':verify();print(adapter.process(ROOT/'raw/flow308.zip',ROOT/'data')['n'])
    else:{'freeze':freeze,'verify':lambda:print(verify()['status']),'calibrate':calibration,'test':test}[a.phase]()
