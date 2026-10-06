"""Known-data development of a single native-anchored paired posterior.

No committee or OR admission: one normalized density, one quantile score.
Original native q states are shared issuance covariates; all four ratio arms
receive the same nine additional configurations after native selection.
"""
from pathlib import Path
import argparse,datetime,importlib.util,sys,math
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT/'work/fusion_revision_20261005/joint_competitors'))
import run_competitors as rc
api=rc.api;np=api.np;base=rc.base
ARMS=('paired','marginal','factorized','zero_increment')
BANDS=(.25,.5,1.);ZETAS=(0.,.5,1.)
U=np.linspace(-1.,1.,2049)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,x):api.dump(ROOT/name,x)
def load(name):return api.load(ROOT/name)
def configurations():return [dict(bandwidth=b,zeta=z) for b in BANDS for z in ZETAS]
def lognormal_line(h,a,l,V):
    inv=np.linalg.inv(V);r=h[None,None,:]-U[:,None,None]-l[None,:,:]
    ld=np.linalg.slogdet(V)[1]
    logs=np.log(a)[None,:]-.5*(len(h)*api.c.LOG2PI+ld[None,:]+np.einsum('uis,ist,uit->ui',r,inv,r))
    return api.c.logsumexp(logs,axis=1)
def logmarginal(h,a,l,V):
    variance=np.diagonal(V,axis1=1,axis2=2);logs=np.zeros(len(U))
    for s in range(len(h)):
        r=h[s]-U[:,None]-l[None,:,s]
        logs+=api.c.logsumexp(np.log(a)[None,:]-.5*(api.c.LOG2PI+np.log(variance[:,s])[None,:]+r*r/variance[:,s][None,:]),axis=1)
    return logs
def alpha_from_q(q):
    if q<=0:return 1.
    lo,hi=-12.,12.
    for _ in range(60):
        z=(lo+hi)/2;p=.5*math.erfc(-z/math.sqrt(2));ratio=math.exp(-z*z/2)/math.sqrt(2*math.pi)/max(p,1e-300)
        if ratio>q:lo=z
        else:hi=z
    return .5*math.erfc(-(lo+hi)/2/math.sqrt(2))
def density_tail(logs,alpha):
    weights=np.exp(logs-logs.max());mass=(weights[:-1]+weights[1:])/2*np.diff(U)
    first=(U[:-1]*weights[:-1]+U[1:]*weights[1:])/2*np.diff(U)
    total=mass.sum();cum=np.r_[0.,np.cumsum(mass)]/total;moment=np.r_[0.,np.cumsum(first)]/total
    if alpha>=1:return float(moment[-1])
    quantile=np.interp(alpha,cum,U);return float(np.interp(quantile,U,moment)/alpha)
def density_logs(d,pre,cfg):
    mm=d['temporal'];a,l,V,mu,C,J=api.c.law(mm,pre,d['ids'],cfg)
    g=lognormal_line(d['h'],np.array([1.]),mu[None,:],C[None,:,:])
    paired=lognormal_line(d['h'],a,l,V)-g
    marginal=logmarginal(d['h'],a,l,V)-g
    fa,fl,fV,fmu,fC,_=api.c.law(mm,pre,d['ids'],cfg,True)
    factorized=logmarginal(d['h'],fa,fl,fV)-g
    return dict(paired=paired,marginal=marginal,factorized=factorized,zero_increment=np.zeros(len(U)))
def freeze():
    assert not (ROOT/'protocol.json').exists();api.verify();base.verify();rc.physical.verify()
    sources={Path(__file__)}
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and str(p).startswith(str(PROJECT)) and str(p).endswith('.py'):sources.add(Path(p))
    dump('protocol.json',dict(utc=now(),status='Known RSS/gas development; both held-out data streams were previously inspected. This is not independent new physical validation.',arms=ARMS,grid=configurations(),
        operator='pi*(u|h) proportional to Normal(u;F_native/N,(S_native/N)^2) times exp(zeta[log f_arm(h-u1)-log f_G(h-u1)]) on[-1,1]; G matched full paired mean and covariance including same kernel; J paired same atoms; one lower expected-shortfall score N LT_alpha(q_native)(pi*)-5, where phi(Phi^-1(alpha))/alpha=q_native and alpha(0)=1',
        anchor='previously selected strongest native factorized_information controller; all its q callbacks and q-dependent optimization remain unchanged and are supplied to every new arm',
        shared='same current contrasts, histories, quality, maturity, kernels, masks, preparation/model work, fixed130 reserve and B130 ledger',
        parameter_budget='native prior/q-floor previously selected from9 configs per task; EACH new arm receives9 ADDITIONAL configs bandwidth .25,.5,1 times exponent0,.5,1; 3 calibration delays; second-stage search is disclosed; original native outcomes remain comparator',
        selection='guarded actual complete net on calibration secondhalf, pooled/3; ties favor smaller exponent then larger bandwidth; exact same for all four arms',
        callbacks='shared native q is computed by original native all-issued maturity callbacks, not independently refit to new score; a failed selective calibration is retained, not relabeled calibrated',
        quadrature='2049 uniformly spaced nodes on[-1,1], trapezoidal CDF with linear quantile interpolation; no coverage guarantee inferred from integration',
        source_hashes={str(p):api.sha(p) for p in sorted(sources)},reference_selections=dict(rss=api.sha(base.ROOT/'selection.json'),gas=api.sha(rc.physical.ROOT/'selection.json'))))
    dump('pre_run_freeze.json',dict(utc=now(),protocol_sha256=api.sha(ROOT/'protocol.json')))
    print('FROZEN',api.sha(ROOT/'protocol.json'),flush=True)
def verify():
    p=load('protocol.json');assert api.sha(ROOT/'protocol.json')==load('pre_run_freeze.json')['protocol_sha256']
    for path,want in p['source_hashes'].items():assert api.sha(path)==want,path
    return p
def anchor(task):
    s=(base.load('selection.json') if task=='rss' else rc.physical.load('selection.json'))['selected']['legacy_factorized']
    if task=='rss':return dict(prior_sd=s['prior_sd'],q_floor=s['q_floor'],slice_bandwidth=1.),s['initial']
    return s['config'],s['initial']
def prepare(task,part):
    engine,guard,cfg,service,fork,pre,data,cache,records=rc.worlds(task,part)
    cc,q0=anchor(task);cfg=dict(cfg,**cc);streams=rc.prepared_streams(records,pre,cfg)
    native=[]
    for stream in streams:
        with base.bindings():r=api.core.run(engine,stream,pre,cfg,'legacy_factorized',q0,part=='calibration')
        native.append(r)
        for d,row in zip(stream['decisions'],r['rows']):
            d['native']=row
            d['loganchor']=-.5*((U-row['gain']/d['N'])/(row['posterior_or_block_sd']/d['N']))**2
            d['prob']=alpha_from_q(row['q_issued'])
            d['ratios']={}
        for b in BANDS:
            for d in stream['decisions']:d['ratios'][b]=density_logs(d,pre,dict(cfg,slice_bandwidth=b))
    return engine,guard,cfg,service,fork,pre,cache,records,streams,native

def issue(s,arm,conf,selection=False):
    rows=[];boundary=s['windows']//2 if selection else 0
    for d in s['decisions']:
        n=d['native'];logdensity=d['loganchor']+conf['zeta']*d['ratios'][conf['bandwidth']][arm]
        value=density_tail(logdensity,max(1e-12,d['prob']));score=d['N']*value-5
        ready=bool(d['temporal']['eligible']);covered=bool(d['truegross']-5>=score)
        rows.append(dict(k=d['k'],action=bool(ready and score>0 and d['k']>=boundary),maturity=d['maturity'],local_net=d['localnet'],truegross=d['truegross'],gate_score=score,gain=score+5,posterior_or_block_sd=1.,q_issued=n['q_issued'],standardized_score=score-(d['truegross']-5),lower_covered=covered,information_ready=ready,disagreement=d['disagreement'],active_source_ids=d['ids'].tolist(),posterior_tail_mass=d['prob'],native_score=n['gate_score']))
    return dict(rows=rows)
def summarize(guarded,seeds):
    out={}
    for arm in ARMS+('native_original',):
        gs=[g for g in guarded if g['arm']==arm];rows=[r for g in gs for r in g['rows']];acts=[r for r in rows if r['action']]
        increments=[sum(g['increment'] for g in gs if g['seed']==seed) for seed in seeds]
        excess=[max(0.,r['gate_score']-(r['truegross']-5)) for r in acts]
        out[arm]=dict(mean_increment=float(np.mean(increments)),seed_increments=increments,admissions=len(acts),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),zero=sum(r['local_net']==0 for r in acts),negative_loss=sum(max(0.,-r['local_net']) for r in acts),admitted_coverage=[sum(r['lower_covered'] for r in acts),len(acts)],admitted_max_excess=max(excess,default=0.),admitted_total_excess=sum(excess))
    return out

def phase(task,part):
    verify();out=ROOT/task;out.mkdir(exist_ok=True)
    engine,guard,cfg,service,fork,pre,cache,records,streams,native=prepare(task,part)
    if part=='calibration':
        assert not (out/'selection.json').exists();trials=[];grid=[]
        for arm in ARMS:
            for conf in configurations():
                rr=[issue(s,arm,conf,True) for s in streams]
                gg=[dict(recording=rec['recording'],seed=seed,arm=arm,**api.db.guarded(e,pre,cfg,r,guard,service,130.)) for (rec,seed,e,_),r in zip(records,rr)]
                entry=dict(arm=arm,config=conf,net=sum(g['increment'] for g in gg)/3)
                grid.append(entry);trials.append(dict(configuration=entry,guarded=gg));print('CAL',task,arm,conf,entry['net'],flush=True)
        selected={arm:max((x for x in grid if x['arm']==arm),key=lambda x:(x['net'],-x['config']['zeta'],x['config']['bandwidth'])) for arm in ARMS}
        dump(task+'/calibration_trials.json.gz',trials);dump(task+'/selection.json',dict(selected=selected,grid=grid,anchor=anchor(task)))
        dump(task+'/selection_lock.json',dict(utc=now(),selection_sha256=api.sha(out/'selection.json'),test_exists=False))
        print('SELECT',task,selected,flush=True)
    else:
        assert not (out/'results.json.gz').exists();assert api.sha(out/'selection.json')==load(task+'/selection_lock.json')['selection_sha256']
        selected=load(task+'/selection.json')['selected'];issued=[];guarded=[];audits=[]
        for arm in ARMS:
            for (rec,seed,e,_),s in zip(records,streams):
                r=issue(s,arm,selected[arm]['config']);g=api.db.guarded(e,pre,cfg,r,guard,service,130.)
                identity=dict(arm=arm,recording=rec['recording'],seed=seed);issued.append(dict(**identity,**r));guarded.append(dict(**identity,**g));audits.append(dict(**identity,**base.target_checks(e,r['rows'],cfg,fork)))
        for (rec,seed,e,_),r in zip(records,native):guarded.append(dict(arm='native_original',recording=rec['recording'],seed=seed,**api.db.guarded(e,pre,cfg,r,guard,service,130.)))
        seeds=base.TEST if task=='rss' else rc.physical.TEST
        summary=summarize(guarded,seeds)
        dump(task+'/results.json.gz',dict(issued=issued,guarded=guarded,selected=selected,audits=audits))
        dump(task+'/summary.json',dict(summary=summary,selected=selected,max_service_error=max(g['service_error'] for g in guarded),target_checks=audits,cache=cache.report()))
        print('SUMMARY',task,summary,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','calibrate','test']);p.add_argument('--task',choices=['rss','gas'],default='rss');a=p.parse_args()
    freeze() if a.phase=='freeze' else phase(a.task,'calibration' if a.phase=='calibrate' else 'test')
