"""Frozen known-data comparison of complete joint conditional laws.

New outputs only. Separate freeze, calibration, test and report processes.
No external classifier, full-network or independent physical claim is made.
"""
from pathlib import Path
import argparse,datetime,importlib.util,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module('complete_joint_shared',PROJECT/'work/fusion_acceptance_20261005/strong/run_aligned.py')
physical=module('complete_joint_gas_shared',PROJECT/'work/fusion_acceptance_20261005/physical/run_physical.py')
api=base.api;np=api.np
from joint_return import make_law, augment, conditional_gaussian
ARMS=('return_joint','return_factorized','return_gaussian')
BANDS=(.25,.5,1.);CAPS=(.1,.5,1.);DFS=(3.,5.,10.)
BUDGETS=(0.,110.,130.,260.)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,obj):api.dump(ROOT/name,obj)
def load(name):return api.load(ROOT/name)
def configurations(arm):
    return [dict(b_return=bu,b_source=bh,cap=1.) for bu in BANDS for bh in BANDS]
def synthetic_checks():
    C=np.array([[.13,.04,-.015],[.04,.20,.03],[-.015,.03,.17]])
    mean=np.array([.1,.05,.03]);h=np.array([.13,-.08])
    loc,var=conditional_gaussian(h,mean,C,api)
    u=np.linspace(-1,1,200001);r=np.column_stack((u-mean[0],np.repeat(h[0]-mean[1],len(u)),np.repeat(h[1]-mean[2],len(u))))
    inv=np.linalg.inv(C);dens=np.exp(-.5*np.einsum('is,st,it->i',r,inv,r))
    mass=np.trapezoid(dens,u);numeric=np.trapezoid(u*dens,u)/mass
    from joint_return import TruncatedMixture
    exact=TruncatedMixture([loc],[var],[0.])
    assert abs(numeric-exact.mean)<1e-9
    return dict(conditional_joint_mean_error=abs(numeric-exact.mean))
def freeze():
    assert not (ROOT/'protocol.json').exists();base.verify();physical.verify();api.verify()
    checks=synthetic_checks();sources={Path(__file__),ROOT/'joint_return.py'}
    for obj in list(sys.modules.values()):
        file=getattr(obj,'__file__',None)
        if file and str(file).startswith(str(PROJECT)) and str(file).endswith('.py'):sources.add(Path(file))
    data={**base.load('protocol.json')['data_hashes']}
    for file in (physical.ROOT/'data').glob('*'):data[str(file)]=api.sha(file)
    protocol=dict(utc=now(),status='KNOWN RSS/UCI308 DEVELOPMENT: target-dependent permanent-heldout current-contrast law WITH sampling uncertainty; not independent physical confirmation',
        source_hashes={str(p):api.sha(p) for p in sorted(sources)},data_hashes=data,arms=ARMS,
        grids={arm:configurations(arm) for arm in ARMS},configurations_per_arm=9,
        calibration_seeds=dict(rss=base.CAL,gas=physical.CAL),development_seeds=dict(rss=base.TEST,gas=physical.TEST),
        shared='same current issued h, active source masks, complete lease permanent-heldout current-contrast h/U, physical context, prefix quality, age weights, archive48, model versions/work, delayed complete targets, Fixed130/B130 ledger',
        law='p(U,H) as Gaussian-smoothed paired immutable complete-lease observations; condition H=h; U truncated[-1,1]; N*LT_alpha(U|h)-5',
        labels='current-model complete-horizon U and immutable source forecasts projected only on permanent hash audit rows; every candidate update excludes these rows; target estimate is normalized by observed audit positions and requires representativeness; calibration uses exact actual full-lease targets',
        missing='all U matured, only present forecast H observed; Gaussian Schur completion over [U,H_observed] with explicit uncertainty, complete-block covariance ridge',
        sampling='Hajek means over audit positions; U uses all4window audit positions with common interruption indicators; working FPC .7 times samplevariance/auditcount; H full sample covariance; target/source sampling covariance blockdiagonal; propagated through Schur conditional-completion transfer; hash determinism requires representativeness assumption',
        prior='one zero Gaussian atom mass2, U variance.01, H variance.01*meanquality/quality; Gaussian working prior, not true physical law',
        kernel='Scott neff^(-2/(m+5))*D(moment+prior)D; Ddiag(b_return,b_source,..); all3arms identical joint moments/atoms',
        reductions='factorization exact f(U,H_s) and f_U marginals combined prod_s f(U,H_s)/f_U(U)^(m-1); matrix single full Gaussian matching all [U,H] moments',
        calibration='same original empirical firsthalf mature ready <=10% violation alpha fit,35 bisections; same ready-only alpha update .02; not conditional coverage certificate',
        selection='guarded complete actual secondhalf net/3 original calibration schedules; ties larger source then return bandwidth',
        budgets=BUDGETS,primary_budget=130.,reserve=130.,runtime=dict(python=sys.version,numpy=np.__version__),
        limitations=['all underlying traces known development data','Gaussian smoothing and missing completion are working models','no universal profit or actual-admission coverage assumed'])
    dump('mathematical_checks.json',checks);dump('protocol.json',protocol)
    dump('pre_calibration_freeze.json',dict(utc=now(),protocol_sha256=api.sha(ROOT/'protocol.json'),checks_sha256=api.sha(ROOT/'mathematical_checks.json'),selection_exists=False,test_exists=False))
    print('FROZEN',api.sha(ROOT/'protocol.json'),flush=True)
def verify():
    p=load('protocol.json');assert api.sha(ROOT/'protocol.json')==load('pre_calibration_freeze.json')['protocol_sha256']
    for field in ('source_hashes','data_hashes'):
        for path,want in p[field].items():assert api.sha(path)==want,path
    return p
def worlds(task,part):
    if task=='rss':
        engine,guard,cfg,service,fork,data,pre,cache,events,bases=base.worlds(part)
        seeds=base.CAL if part=='calibration' else base.TEST
        records=[(dict(recording='rss348',batch='known-rss3' if part=='test' else 'known-rss2'),seed,e,b) for seed,e,b in zip(seeds,events,bases)]
    else:
        # Reuse immutable physical preparation without its global c.law patch.
        physical.verify();engine,recovery,guard,source,cfg,study,service,fork=api.p.r.binding();cfg=api.p.base_cfg(cfg)
        cache=api.db.ExactGradientCache(engine);engine.gradient=cache
        pre,data=physical.adapter.prepared(physical.ROOT/'data',engine,cfg);api.db.helper.CFG=cfg
        records=[]
        for rec in pre['recording_streams']:
            if rec['partition']!=part:continue
            for seed in physical.CAL if part=='calibration' else physical.TEST:
                e=engine.world(rec['x'],rec['y'],rec['timestamp'],pre,seed,cfg)
                records.append((rec,seed,e,engine.precompute(e,pre,cfg)))
    cfg=dict(cfg,prior_sd=.1)
    return engine,guard,cfg,service,fork,pre,data,cache,records
def prepared_streams(records,pre,cfg):
    streams=[]
    for _,_,e,b in records:
        streams.append(augment(e,pre,b,cfg))
    return streams
def decorate(streams,pre,cfg,arm):
    diagnostics=[]
    for stream in streams:
        for d in stream['decisions']:
            law=make_law(d,pre,cfg,arm,api);d['tail_laws']={arm:law}
            diagnostics.append(dict(k=d['k'],dimension=law.active_dimension,components=law.component_count,neff=law.neff,
                quadrature_error=getattr(law,'quadrature_error',0.),quadrature_nodes=getattr(law,'quadrature_nodes',0),quadrature_converged=getattr(law,'quadrature_converged',True)))
    return dict(laws=len(diagnostics),max_quadrature_error=max(x['quadrature_error'] for x in diagnostics),max_nodes=max(x['quadrature_nodes'] for x in diagnostics),nonconverged=sum(not x['quadrature_converged'] for x in diagnostics),diagnostics=diagnostics)
def fit(streams,arm,cap):
    eligible=[d for s in streams for d in s['decisions'] if d['maturity']<=s['windows']//2 and d['temporal']['eligible']]
    if not eligible:return 1e-4,0,0
    lo,hi=1e-4,cap
    for _ in range(35):
        mid=(lo+hi)/2;v=np.mean([d['truegross']<d['N']*d['tail_laws'][arm].lower_tail(mid) for d in eligible])
        if v<=.1:lo=mid
        else:hi=mid
    v=sum(d['truegross']<d['N']*d['tail_laws'][arm].lower_tail(lo) for d in eligible)
    return lo,len(eligible),int(v)
def tie(entry):
    cfg=entry['config'];return entry['net'],cfg['b_source'],cfg['b_return']
def calibrate(task):
    verify();out=ROOT/task;out.mkdir(exist_ok=True);assert not (out/'selection.json').exists()
    engine,guard,cfg0,service,fork,pre,data,cache,records=worlds(task,'calibration');streams=prepared_streams(records,pre,cfg0);grid=[];trials=[]
    for arm in ARMS:
        for conf in configurations(arm):
            cfg=dict(cfg0,**conf);diagnostic=decorate(streams,pre,cfg,arm);alpha,count,violations=fit(streams,arm,cfg['cap'])
            rr=[api.run(s,arm,cfg['cap'],alpha,True) for s in streams]
            gg=[dict(recording=rec['recording'],batch=rec['batch'],seed=seed,**api.db.guarded(e,pre,cfg,r,guard,service,130.)) for (rec,seed,e,_),r in zip(records,rr)]
            entry=dict(arm=arm,config=conf,alpha_initial=alpha,fit_count=count,fit_violations=violations,net=sum(g['increment'] for g in gg)/3)
            grid.append(entry);trials.append(dict(configuration=entry,issued=[dict(recording=rec['recording'],seed=seed,**r) for (rec,seed,_,_),r in zip(records,rr)],guarded=gg,numerical=diagnostic))
            print('CAL',task,arm,conf,entry['net'],flush=True)
    selected={arm:max((x for x in grid if x['arm']==arm),key=tie) for arm in ARMS}
    dump(task+'/calibration_trials.json.gz',trials);dump(task+'/selection.json',dict(selected=selected,grid=grid,quality=pre['q'],split=pre['split'],cfg=cfg0))
    dump(task+'/selection_lock.json',dict(utc=now(),protocol_sha256=api.sha(ROOT/'protocol.json'),selection_sha256=api.sha(out/'selection.json'),trials_sha256=api.sha(out/'calibration_trials.json.gz'),test_exists=False,cache=cache.report()))
    print('SELECT',task,selected,flush=True)
def summarize(guarded,seeds):
    out={}
    for arm in ARMS:
        gg=[g for g in guarded if g['arm']==arm];rows=[r for g in gg for r in g['rows']]
        groups=dict(issued=rows,ready=[r for r in rows if r['information_ready']],admitted=[r for r in rows if r['action']]);acts=groups['admitted']
        excess=[max(0.,r['gate_score']-(r['truegross']-5.)) for r in acts]
        totals=[sum(g['increment'] for g in gg if g['seed']==seed) for seed in seeds]
        out[arm]=dict(seed_increments=totals,mean_increment=float(np.mean(totals)),per_admission_actual_net=sum(r['local_net'] for r in acts)/len(acts) if acts else None,
            admissions=len(acts),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),zero=sum(r['local_net']==0 for r in acts),negative_loss=sum(max(0.,-r['local_net']) for r in acts),
            coverage={k:[sum(r['lower_covered'] for r in v),len(v)] for k,v in groups.items()},admitted_max_excess=max(excess,default=0.),admitted_total_excess=sum(excess),refused=sum(g['refused'] for g in gg),
            positive_outcome_certificate=(sum(r['gate_score'] for r in acts)-sum(excess))/len(seeds))
    return out
def differences(guarded,seeds):
    by={(g['arm'],g['seed'],g['recording']):g for g in guarded};out={}
    for arm in ARMS[1:]:
        terms={k:dict(count=0,value=0.) for k in ('added_gain','avoided_loss','missed_gain','incurred_loss')};changes=[];zeros=0
        for J in [g for g in guarded if g['arm']=='return_joint']:
            C=by[arm,J['seed'],J['recording']];assert len(J['rows'])==len(C['rows'])
            for r,c in zip(J['rows'],C['rows']):
                assert r['k']==c['k'] and r['truegross']==c['truegross'] and r['local_net']==c['local_net']
                if r['action']==c['action']:continue
                D=r['local_net'];changes.append(dict(recording=J['recording'],seed=J['seed'],k=r['k'],actual_net=D,joint_action=r['action'],competitor_action=c['action'],joint_score=r['gate_score'],competitor_score=c['gate_score']))
                if D==0:zeros+=1;continue
                key=('added_gain' if D>0 else 'incurred_loss') if r['action'] else ('missed_gain' if D>0 else 'avoided_loss')
                terms[key]['count']+=1;terms[key]['value']+=abs(D)
        total=terms['added_gain']['value']+terms['avoided_loss']['value']-terms['missed_gain']['value']-terms['incurred_loss']['value']
        actual=sum(g['increment'] for g in guarded if g['arm']=='return_joint')-sum(g['increment'] for g in guarded if g['arm']==arm);assert abs(total-actual)<1e-8
        out[arm]=dict(terms=terms,zero_changes=zeros,changed_actions=changes,pooled_difference=total,mean_difference=total/len(seeds))
    return out
def test(task):
    verify();out=ROOT/task;assert not (out/'results.json.gz').exists()
    assert load(task+'/selection_lock.json')['selection_sha256']==api.sha(out/'selection.json')
    selected=load(task+'/selection.json')['selected'];engine,guard,cfg0,service,fork,pre,data,cache,records=worlds(task,'test')
    issued=[];guarded=[];budget_rows=[];audits=[];diagnostics=[]
    for rec,seed,e,b in records:
        streams=prepared_streams([(rec,seed,e,b)],pre,cfg0)
        for arm,s in selected.items():
            cfg=dict(cfg0,**s['config']);diagnostic=decorate(streams,pre,cfg,arm);r=api.run(streams[0],arm,cfg['cap'],s['alpha_initial'])
            g=api.db.guarded(e,pre,cfg,r,guard,service,130.);identity=dict(arm=arm,recording=rec['recording'],batch=rec['batch'],seed=seed)
            issued.append(dict(**identity,**r));guarded.append(dict(**identity,**g));diagnostics.append(dict(**identity,**diagnostic))
            audits.append(dict(**identity,**base.target_checks(e,r['rows'],cfg,fork)))
            for B in BUDGETS:budget_rows.append(dict(**identity,budget=B,**(g if B==130. else api.db.guarded(e,pre,cfg,r,guard,service,B))))
        print('TEST',task,rec['recording'],seed,flush=True)
    seeds=base.TEST if task=='rss' else physical.TEST
    dump(task+'/results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budget_rows,numerical=diagnostics,selected=selected))
    dump(task+'/summary.json',dict(summary=summarize(guarded,seeds),budget_summary={str(B):summarize([g for g in budget_rows if g['budget']==B],seeds) for B in BUDGETS},decomposition=differences(guarded,seeds),selected=selected,
        max_service_error=max(g['service_error'] for g in budget_rows),target_checks=audits,max_calibration_identity=max(abs(r['identity_error']) for r in issued),numerical=dict(max_quadrature_error=max(x['max_quadrature_error'] for x in diagnostics),max_nodes=max(x['max_nodes'] for x in diagnostics),nonconverged=sum(x['nonconverged'] for x in diagnostics)),cache=cache.report()))
    print('SUMMARY',task,summarize(guarded,seeds),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','verify','calibrate','test']);p.add_argument('--task',choices=['rss','gas'],default='rss');a=p.parse_args()
    if a.phase=='freeze':freeze()
    elif a.phase=='verify':print(verify()['status'])
    elif a.phase=='calibrate':calibrate(a.task)
    else:test(a.task)
