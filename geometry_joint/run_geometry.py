"""Development regularization grid; unchanged CJRT paired-law/tail operator.

The sole primary pipeline change replaces bandwidth x tail-cap search with
prior_sd x bandwidth search at tail cap one. All outcome data retained.
"""
from pathlib import Path
import importlib.util,sys,datetime
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
spec=importlib.util.spec_from_file_location('guarded_prior_shared',ROOT.parent/'strong/run_aligned.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
api=base.api;np=api.np;ARMS=api.ARMS;PRIORS=(.25,.5,1.);BANDS=(.25,.5,1.);CAP=1.
original_covariance_law=api.c.law

def covariance_law(mm,pre,ids,cfg,factorized=False):
    a,l,V,mu,C,old=original_covariance_law(mm,pre,ids,cfg,factorized)
    Q=cfg['prior_variance']*np.diag(np.mean(pre['q'])/pre['q'][ids])
    half=np.diag(np.sqrt(np.diag(Q)))
    e=np.linalg.solve(half,np.ones(len(ids)));e/=np.linalg.norm(e)
    projection=np.outer(e,e)
    bp=cfg.get('b_parallel',cfg['slice_bandwidth']);bt=cfg.get('b_perp',cfg['slice_bandwidth'])
    new=half@(bp**2*projection+bt**2*(np.eye(len(ids))-projection))@half
    if bp==bt:assert np.max(np.abs(new-old))<1e-15
    return a,l,V-old+new,mu,C-old+new,new
api.c.law=covariance_law
def dump(name,obj):api.dump(ROOT/name,obj)
def load(name):return api.load(ROOT/name)
def sha(p):return api.sha(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def freeze():
    assert not (ROOT/'protocol.json').exists();base.verify();original=api.verify()
    sources={**original['source_hashes'],**base.load('protocol.json')['source_hashes'],str(Path(__file__)):sha(__file__)}
    dump('protocol.json',dict(utc=now(),status='Known RSS decision-aligned joint kernel development; frozen before calibration/test replay',
        source_hashes=sources,data_hashes=base.load('protocol.json')['data_hashes'],arms=ARMS,prior_sd=.1,b_parallel=PRIORS,b_perp=BANDS,tail_cap=CAP,
        native_operator='same immutable finite paired error law, working masked precision completion and conditional paid lower-tail readout',
        sole_change='quality-whitened anisotropic paired-law smoothing, separate common-target and transverse-disagreement scales; same error locations/state, one conditional law and paid score;9configs',
        fit='largest alpha with <=10% first-half fully mature ready violations, 35 bisections, alpha min1e-4',
        update='ready-only immutable issued-score callbacks; alpha += .02*(.1−violation) clipped[1e-4,1]',
        selection='mean second-half independently reconstructed B130 guarded actual net on three original calibration delays; ties larger transverse scale then larger parallel scale',
        configs_per_arm=9,trial_count_per_arm=27,calibration_seeds=base.CAL,test_seeds=base.TEST,budget=130.,reserve=130.,budgets=(0.,110.,130.,260.),
        shared='same current h, prefix quality, matured projected full-lease source residuals, model work/versions, masks, delays, fees and ledger',
        limitation='RSS calibration/test already known in previous development; comparisons retained; post-change independent physical validation required'))
    dump('pre_calibration_freeze.json',dict(utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),selection_exists=False,test_exists=False))
    print('FROZEN',sha(ROOT/'protocol.json'),flush=True)
def verify():
    p=load('protocol.json');assert sha(ROOT/'protocol.json')==load('pre_calibration_freeze.json')['protocol_sha256']
    for field in ('source_hashes','data_hashes'):
        for path,want in p[field].items():assert sha(path)==want,path
    return p
def stream(e,pre,initial,cfg):
    with base.bindings(api.c.parent.cumulant_state):s=api.core.augment(e,pre,initial,cfg)
    for d in s['decisions']:d['tail_laws']={a:api.law(d,pre,cfg,a) for a in ARMS}
    return s
def calibrate():
    verify();assert not (ROOT/'selection.json').exists()
    engine,guard,cfg0,service,fork,data,pre,cache,events,bases=base.worlds('calibration');grid=[];trials=[]
    for bp in PRIORS:
        for b in BANDS:
            cfg=dict(cfg0,prior_sd=.1,slice_bandwidth=b,b_parallel=bp,b_perp=b)
            ss=[stream(e,pre,s,cfg) for e,s in zip(events,bases)]
            initials,count=api.fit(ss,CAP)
            for arm in ARMS:
                rr=[api.run(s,arm,CAP,initials[arm],True) for s in ss]
                gg=[api.db.guarded(e,pre,cfg,r,guard,service,130.) for e,r in zip(events,rr)]
                conf=dict(arm=arm,prior_sd=.1,bandwidth=b,b_parallel=bp,b_perp=b,cap=CAP,alpha_initial=initials[arm],fit_count=count,
                    net=sum(g['increment'] for g in gg)/3,beneficial=sum(g['beneficial'] for g in gg),harmful=sum(g['harmful'] for g in gg))
                grid.append(conf);trials.append(dict(configuration=conf,issued=rr,guarded=gg))
            print('CAL',bp,b,flush=True)
    selected={arm:max((g for g in grid if g['arm']==arm),key=lambda g:(g['net'],g['b_perp'],g['b_parallel'])) for arm in ARMS}
    dump('calibration_trials.json.gz',trials);dump('selection.json',dict(selected=selected,grid=grid,split=pre['split'],cfg=cfg0,quality=pre['q'],data_hashes=data['hashes']))
    dump('selection_lock.json',dict(utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),selection_sha256=sha(ROOT/'selection.json'),trials_sha256=sha(ROOT/'calibration_trials.json.gz'),test_exists=False,cache=cache.report()))
    print('SELECT',selected,flush=True)
def summarize(guarded):
    out={}
    for arm in ARMS:
        gg=[g for g in guarded if g['arm']==arm];rows=[r for g in gg for r in g['rows']]
        groups=dict(issued=rows,ready=[r for r in rows if r['information_ready']],informative=[r for r in rows if r['disagreement']>0],admitted=[r for r in rows if r['action']])
        acts=groups['admitted'];excess=[max(0.,r['gate_score']-(r['truegross']-5.)) for r in acts]
        out[arm]=dict(mean_increment=float(np.mean([g['increment'] for g in gg])),seed_increments=[g['increment'] for g in gg],admissions=len(acts),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),zero=sum(r['local_net']==0 for r in acts),negative_loss=sum(max(0.,-r['local_net']) for r in acts),coverage={k:[sum(r['lower_covered'] for r in rr),len(rr)] for k,rr in groups.items()},admitted_max_excess=max(excess,default=0.),admitted_total_excess=sum(excess),refused=sum(g['refused'] for g in gg))
    return out
def differences(guarded,comparison):
    rows={(g['arm'],g['seed']):g for g in guarded};out={}
    J={seed:rows['joint',seed] for seed in base.TEST}
    for name,C in comparison.items():
        terms={k:dict(count=0,value=0.) for k in ('added_gain','avoided_loss','missed_gain','incurred_loss')};changes=[];zero=0;increments=[]
        for seed in base.TEST:
            jr={r['k']:r for r in J[seed]['rows']};cr={r['k']:r for r in C[seed]['rows']};assert jr.keys()==cr.keys();increments.append(J[seed]['increment']-C[seed]['increment'])
            for k,r in jr.items():
                c=cr[k];assert r['truegross']==c['truegross'] and r['local_net']==c['local_net']
                if r['action']==c['action']:continue
                D=r['local_net'];changes.append(dict(seed=seed,k=k,joint=r['action'],control=c['action'],actual_net=D,joint_score=r['gate_score'],control_score=c['gate_score']))
                if D==0:zero+=1;continue
                key=('added_gain' if D>0 else 'incurred_loss') if r['action'] else ('missed_gain' if D>0 else 'avoided_loss')
                terms[key]['count']+=1;terms[key]['value']+=abs(D)
        value=terms['added_gain']['value']+terms['avoided_loss']['value']-terms['missed_gain']['value']-terms['incurred_loss']['value']
        assert abs(value-sum(increments))<1e-8
        out[name]=dict(terms=terms,changed_actions=changes,zero_changes=zero,seed_differences=increments,mean_difference=value/5,total_difference=value)
    return out
def test():
    verify();assert not (ROOT/'results.json.gz').exists();assert load('selection_lock.json')['selection_sha256']==sha(ROOT/'selection.json')
    engine,guard,cfg0,service,fork,data,pre,cache,events,bases=base.worlds('test');selected=load('selection.json')['selected'];issued=[];guarded=[];budget_rows=[];audits=[]
    for seed,e,initial in zip(base.TEST,events,bases):
        ss={}
        for arm,s in selected.items():
            cfg=dict(cfg0,prior_sd=.1,slice_bandwidth=s['b_perp'],b_parallel=s['b_parallel'],b_perp=s['b_perp']);key=(s['b_parallel'],s['b_perp'])
            if key not in ss:ss[key]=stream(e,pre,initial,cfg)
            r=api.run(ss[key],arm,CAP,s['alpha_initial']);g=api.db.guarded(e,pre,cfg,r,guard,service,130.)
            issued.append(dict(arm=arm,seed=seed,**r));guarded.append(dict(arm=arm,seed=seed,**g));audits.append(dict(arm=arm,seed=seed,**base.target_checks(e,r['rows'],cfg,fork)))
            for B in (0.,110.,130.,260.):
                bg=g if B==130. else api.db.guarded(e,pre,cfg,r,guard,service,B)
                budget_rows.append(dict(arm=arm,seed=seed,budget=B,**bg))
        print('TEST',seed,flush=True)
    comp={a:{g['seed']:g for g in guarded if g['arm']==a} for a in ARMS[1:]}
    old=base.load('results.json.gz')['guarded']
    comp.update({a:{g['seed']:g for g in old if g['arm']==a} for a in ('legacy_factorized','legacy_sandwich')})
    sm=summarize(guarded);decomp=differences(guarded,comp)
    dump('results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budget_rows,selected=selected))
    dump('summary.json',dict(summary=sm,budget_summary={str(B):summarize([g for g in budget_rows if g['budget']==B]) for B in (0.,110.,130.,260.)},decomposition=decomp,selected=selected,target_checks=audits,max_service_error=max(g['service_error'] for g in budget_rows),cache=cache.report()))
    print('SUMMARY',sm,flush=True);print('STRONG', {a:decomp[a]['mean_difference'] for a in ('legacy_factorized','legacy_sandwich')},flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','verify','calibrate','test']);a=p.parse_args()
    {'freeze':freeze,'verify':lambda:print(verify()['status']),'calibrate':calibrate,'test':test}[a.phase]()
