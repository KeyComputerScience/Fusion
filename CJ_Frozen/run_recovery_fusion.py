"""Causal complete-lease joint-error Bayesian fusion, isolated development extension.

All labels used in issued-lease residual moments mature before the current gate.
Known traces are development replays; held-out new tasks are declared separately.
Original packages are imported but never modified.
"""
from __future__ import annotations
import argparse,datetime,hashlib,itertools,json,math,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent
ARMS=('mean_transfer','diagonal_transfer','joint_transfer','full_transfer','bound_only','point_transfer')
LAMS=(0.,.25,1.)
MARGINS=(0.,4.,12.)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False))
def psd(a):
    eig,v=np.linalg.eigh((a+a.T)/2);return (v*np.maximum(eig,0))@v.T

def capped_prior(q,cap):
    q=np.maximum(np.asarray(q,float),1e-14);q=q/q.sum();w=np.zeros(len(q));free=np.ones(len(q),bool);remaining=1.
    for _ in q:
        ids=np.flatnonzero(free);z=remaining*q[ids]/q[ids].sum();over=z>cap
        if not over.any():w[ids]=z;break
        w[ids[over]]=cap;free[ids[over]]=False;remaining-=cap*over.sum()
    return w

def convex_influence(q,rate,S,U,cfg,arm):
    """Exact active-upper-face Newton solution; sqrt term has PSD Hessian."""
    m=len(q);cap=max(cfg['cap'],1/m);pi=np.maximum(np.asarray(q,float),1e-14)**cfg['quality_power'];pi/=pi.sum()
    lam=0. if arm in ('mean_transfer','point_transfer') else cfg['lam']
    kappa=cfg['norm_kappa'] if arm=='full_transfer' else 0.
    if arm=='diagonal_transfer':S=np.diag(np.diag(S));U=np.diag(np.diag(U))
    floor=cfg['norm_floor'];tau=cfg['tau'];fallback=capped_prior(pi,cap)
    def value(w):return float(tau*w@np.log(w/pi)-w@rate+lam/2*w@S@w+kappa*math.sqrt(float(w@U@w)+floor**2))
    def gh(w):
        a=U@w;s=math.sqrt(float(w@a)+floor**2)
        g=tau*(np.log(w/pi)+1)-rate+lam*S@w+kappa*a/s
        h=np.diag(tau/w)+lam*S+kappa*(U/s-np.outer(a,a)/s**3)
        return g,h
    if m==1 or m*cap<=1+1e-12:return fallback,dict(kkt=0.,primal=0.,converged=True,iterations=0)
    if lam==0 and kappa==0:
        log=np.log(pi)+rate/tau;w=capped_prior(np.exp(log-log.max()),cap)
        g,_=gh(w);free=w<cap-1e-9;common=float(g[free].mean()) if free.any() else 0.;res=float(np.max(abs(g[free]-common))) if free.any() else 0.
        if (~free).any():res=max(res,float(np.maximum(g[~free]-common,0).max()))
        return w,dict(kkt=res,primal=float(abs(w.sum()-1)),converged=res<1e-7,iterations=0)
    best=None;iters=0
    for nupper in range(m):
        if nupper*cap>=1:break
        for fixed in itertools.combinations(range(m),nupper):
            free=np.array([i for i in range(m) if i not in fixed]);w=np.zeros(m);w[list(fixed)]=cap;rem=1-nupper*cap;w[free]=rem*pi[free]/pi[free].sum()
            for _ in range(70):
                iters+=1;g,h=gh(w);gf=g[free];hf=h[np.ix_(free,free)]
                if len(free)==1:break
                if np.max(abs(gf-gf.mean()))<1e-9:break
                kk=np.block([[hf,np.ones((len(free),1))],[np.ones((1,len(free))),np.zeros((1,1))]])
                try:d=np.linalg.solve(kk,np.r_[-gf,0.])[:-1]
                except np.linalg.LinAlgError:break
                step=1.;neg=d<0
                if neg.any():step=min(step,float(np.min(-.99*w[free][neg]/d[neg])))
                old=value(w);descent=float(gf@d)
                for _ in range(35):
                    cand=w.copy();cand[free]+=step*d
                    if np.all(cand>0) and value(cand)<=old+1e-4*step*descent:break
                    step*=.5
                if step<1e-12:break
                w=cand
            if (w>cap+1e-8).any():continue
            g,_=gh(w);common=float(np.mean(g[free]));r=float(np.max(abs(g[free]-common)))
            if fixed:r=max(r,float(np.maximum(g[list(fixed)]-common,0).max()))
            primal=max(abs(float(w.sum())-1),float(np.maximum(w-cap,0).max()),float(np.maximum(-w,0).max()))
            if best is None or r<best[0]:best=(r,w.copy(),primal)
            if r<1e-7 and primal<1e-9:return w,dict(kkt=r,primal=primal,converged=True,iterations=iters)
    if best is not None and best[0]<1e-6 and best[2]<1e-8:return best[1],dict(kkt=best[0],primal=best[2],converged=True,iterations=iters)
    return fallback,dict(kkt=float(best[0]) if best else 1e9,primal=0.,converged=False,iterations=iters)

def lease_moments(arrived,current,pre,cfg):
    """Current-decision reprojection of completed immutable-probability leases.

    Mathematical symmetric prior atoms: zero mean, prior_variance I and total
    concentration prior_mass. These are regularization rather than observations.
    """
    ids=current['ids'];m=len(ids);a0=cfg['prior_mass'];first=np.zeros(m);second=a0*cfg['prior_variance']*np.eye(m);mass=a0;eligible=[]
    for r in arrived[-cfg['lease_archive']:]:
        if not r['mask'][ids].all():continue
        a=cfg['beta']**(current['k']-r['k'])*math.exp(-float(np.sum((current['context']-r['context'])**2))/cfg['kernel_bandwidth']**2)
        if a<1e-12:continue
        reference=(r['x']@current['reference']).argmax(1);candidate=(r['x']@current['candidate']).argmax(1)
        b=np.eye(pre['classes'])[candidate]-np.eye(pre['classes'])[reference]
        issued=np.einsum('sic,ic->s',r['p'][ids],b)/len(r['x'])
        actual=0.
        for f in r['future']:
            pa=(f['x']@current['reference']).argmax(1);pc=(f['x']@current['candidate']).argmax(1);keep=f['keep']
            actual+=float(np.sum((pc[keep]==f['y'][keep]).astype(float)-(pa[keep]==f['y'][keep]).astype(float)))
        z=issued-actual/r['N'];first+=a*z;second+=a*np.outer(z,z);mass+=a;eligible.append(dict(origin=r['k'],maturity=r['maturity'],weight=a,reprojected_residual=z.tolist()))
    mu=first/mass;B=psd(second/mass-np.outer(mu,mu));U=B/(mass+1);T=np.eye(m)-np.ones((m,m))/m
    scale=float(np.trace(T@B@T));scale=scale if scale>=1e-6 else 1.
    return dict(mu=mu,B=B,U=U,S=B/scale,Us=U/scale,mass=mass,scale=scale,eligible=eligible,blocks=len(eligible),prior_mass=a0)

def augment_stream(events,pre,base,cfg):
    pending=[];arrived=[];rows=[]
    for d in base['decisions']:
        k=d['k'];e=events[k];arrived.extend([r for r in pending if r['maturity']<=k]);pending=[r for r in pending if r['maturity']>k];arrived=arrived[-cfg['lease_archive']:]
        current=dict(d,context=e['context'].copy(),candidate=e['candidate'],reference=e['reference']);mm=lease_moments(arrived,current,pre,cfg);row=dict(current,lease=mm)
        future=[]
        for j in range(k,k+cfg['horizon']):
            f=events[j];cd=(cfg['probe_drop'] if f['probe'] else 0)+(2 if f['refresh'] else 0)
            future.append(dict(x=f['x'],y=f['y'],keep=np.arange(len(f['x']))>=cd))
        # Offline lease truths are queued; only complete-label maturity callbacks
        # allow the record to enter the reprojected posterior above.
        pending.append(dict(k=k,maturity=d['maturity'],context=e['context'].copy(),x=e['x'],p=e['p'].copy(),mask=e['mask'].copy(),future=future,N=d['N']))
        rows.append(row)
    return dict(base,decisions=rows)

def forecast(d,pre,cfg,arm):
    mm=d['lease'];rate=d['h']-mm['mu'];w,cert=convex_influence(pre['q'][d['ids']],rate,mm['S'],mm['U'],cfg,arm)
    variance=np.diag(np.diag(mm['B'])) if arm=='diagonal_transfer' else mm['B']
    kf=cfg['norm_kappa'] if arm in ('full_transfer','bound_only') else 0.
    raw_u=np.diag(np.diag(mm['U'])) if arm=='diagonal_transfer' else mm['U']
    norm=math.sqrt(max(0.,float(w@raw_u@w))+cfg['norm_floor']**2)
    gain=float(d['N']*(w@rate-kf*norm));sd=d['N']*math.sqrt(max(0.,float(w@variance@w))+cfg['gate_floor']);score=(gain-d['truegross'])/sd
    return w,cert,gain,sd,score

def run(engine,stream,pre,cfg,arm,qinit,selection=False):
    old=engine.decision_forecast;engine.decision_forecast=lambda d,p,c,m:forecast(d,p,c,arm)
    try:r=engine.run(stream,pre,cfg,'joint' if arm=='point_transfer' else 'bayes_gate',qinit,selection)
    finally:engine.decision_forecast=old
    for row,d in zip(r['rows'],stream['decisions']):
        mm=d['lease'];row.update(lease_mu=mm['mu'].tolist(),lease_B=mm['B'].tolist(),lease_U=mm['U'].tolist(),current_source_rate=d['h'].tolist(),active_source_ids=d['ids'].tolist(),lease_mass=mm['mass'],lease_blocks=mm['blocks'],eligible_residuals=mm['eligible'],norm_penalty=float(cfg['norm_kappa']*math.sqrt(np.asarray(row['weights'])@mm['U']@np.asarray(row['weights'])+cfg['norm_floor']**2)) if arm in ('full_transfer','bound_only') else 0.,contrast_context=d['context'].tolist())
        assert all(z['maturity']<=row['k'] for z in row['eligible_residuals'])
    r['extension_arm']=arm;r['complete_lease_residual_state']=True
    return r

def initial_q(streams,pre,cfg,arm):
    scores=[]
    for stream in streams:
        boundary=stream['windows']//2
        for d in stream['decisions']:
            if d['maturity']<=boundary:scores.append(forecast(d,pre,cfg,arm)[4])
    return max(0.,float(np.quantile(scores,.9,method='higher'))) if scores else 0.,scores

def summary(trials):
    out={}
    for arm in ARMS+('reference',):
        rs=[t['results'][arm] for t in trials];rows=[d for r in rs for d in r.get('rows',[])];inf=[r for r in rows if r['disagreement']>0];acts=[r for r in rows if r['action']]
        out[arm]=dict(mean_net=float(np.mean([r['net'] for r in rs])),mean_actions=float(np.mean([r['deployments'] for r in rs])),harmful=sum(r['harmful'] for r in rs),beneficial=sum(r['beneficial'] for r in rs),coverage=dict(all=[sum(r['lower_covered'] for r in rows),len(rows)],informative=[sum(r['lower_covered'] for r in inf),len(inf)],admitted=[sum(r['lower_covered'] for r in acts),len(acts)]),max_kkt=max((r.get('max_kkt',0) for r in rs)),solver_failures=sum(r.get('solver_failures',0) for r in rs))
    pairs={}
    for arm in ARMS+('reference',):
        if arm=='full_transfer':continue
        values=np.array([t['results']['full_transfer']['net']-t['results'][arm]['net'] for t in trials]);half=2.776445105*values.std(ddof=1)/math.sqrt(len(values)) if len(values)==5 else None
        pairs['full_minus_'+arm]=dict(values=values.tolist(),mean=float(values.mean()),conditional_t4_ci=[float(values.mean()-half),float(values.mean()+half)] if half is not None else None,changed_actions=int(sum(np.sum(np.array(t['results']['full_transfer']['actions'])!=np.array(t['results'][arm]['actions'])) for t in trials)))
    return dict(controllers=out,paired=pairs)

def execute_check(events,pre,cfg,result,explicit_service,explicit_fork):
    actual=explicit_service(events,pre,result['rows'],cfg);errors={k:abs(float(actual[k])-float(result[k])) for k in ('gross','fees','net','drops','accuracy','deployments')};fork=0.;target=0.
    for row in result['rows']:
        f=explicit_fork(events,row,cfg);fork=max(fork,abs(f['gross']-row['truegross']),abs(f['actual']-row['local_net']));target=max(target,f['identity_error'],abs(f['standardized']-row['standardized_score']))
    passed=max(errors.values())<1e-8 and fork<1e-8 and target<1e-8 and result['solver_failures']==0 and result['max_kkt']<1e-6
    return dict(passed=bool(passed),service_errors=errors,fork_error=fork,target_error=target,independent_service=actual,forks=len(result['rows']),q_identity_error=result['calibration_identity_error'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=ROOT.parent.parent/'outputs'/'Fusion_External_Validation_Repro'/'bayes_closed_loop_repro');p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--phase',required=True,choices=('calibrate','test'));p.add_argument('--datasets',nargs='*');args=p.parse_args();out=args.output.resolve();package=args.package.resolve();out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(package));from cached_runner import install_cached_loaders
    engine,adapter=install_cached_loaders();from audit_canary_execution import load_study,explicit_service,explicit_fork
    protocol=json.loads((out/'protocol.json').read_text());studies={k:v for k,v in protocol['studies'].items() if not args.datasets or k in args.datasets};sys.path.insert(0,str(ROOT));from new_data_adapter import load_dataset as load_new,make_prefix
    def study(name,entry):
        if entry.get('new_task'):
            data,spec=load_new(ROOT/'data',name);return data,make_prefix(data,spec,cfgbase,engine),None
        return load_study(package/'independent_data',package/entry['folder'],name)
    cfgbase=dict(engine.BASE,**protocol['extension']);codehash=sha(__file__);prohash=sha(out/'protocol.json')
    if args.phase=='calibrate':
        assert not (out/'results.json').exists(),'test results already exist; no retune'
        previous=json.loads((out/'prefix_freeze.json').read_text()) if (out/'prefix_freeze.json').exists() else None
        if previous:assert previous['code_sha256']==codehash and previous['protocol_sha256']==prohash
        else:dump(out/'pre_calibration_freeze.json',dict(utc=now(),code_sha256=codehash,protocol_sha256=prohash,known_traces_are_development=True))
        hashes=previous['selection_hashes'] if previous else {}
        for name,entry in studies.items():
            assert name not in hashes,'selection already frozen';data,pre,saved=study(name,entry);events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,cfgbase) for s in entry.get('calibration_seeds',protocol['calibration_seeds'])];streams=[augment_stream(e,pre,engine.precompute(e,pre,cfgbase),cfgbase) for e in events];grid=[]
            for lam in LAMS:
                for margin in MARGINS:
                    cfg=dict(cfgbase,lam=lam,threshold=margin)
                    for arm in ARMS:
                        qi,scores=initial_q(streams,pre,cfg,arm);rr=[run(engine,s,pre,cfg,arm,qi,True) for s in streams];grid.append(dict(arm=arm,lam=lam,margin=margin,q_initial=qi,q_fit_scores=scores,mean_selection_net=float(np.mean([r['selection_net'] for r in rr])),mean_selection_actions=float(np.mean([r['deployments'] for r in rr])),solver_failures=sum(r['solver_failures'] for r in rr)))
                    print('CAL',name,lam,margin,flush=True)
            selected={a:max((r for r in grid if r['arm']==a),key=lambda r:(r['mean_selection_net'],-r['margin'],-r['lam'])) for a in ARMS};path=out/(name+'_prefix_selection.json');dump(path,dict(dataset=name,data_hashes=data['hashes'],split=pre['split'],grid=grid,selected=selected,code_sha256=codehash,protocol_sha256=prohash));hashes[name]=sha(path);print('SELECTED',name,{a:(x['lam'],x['margin'],x['q_initial'],x['mean_selection_net']) for a,x in selected.items()},flush=True)
        dump(out/'prefix_freeze.json',dict(utc=now(),code_sha256=codehash,protocol_sha256=prohash,selection_hashes=hashes));return
    freeze=json.loads((out/'prefix_freeze.json').read_text());assert freeze['code_sha256']==codehash and freeze['protocol_sha256']==prohash;all_results=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else {};checks=[]
    for name,entry in studies.items():
        path=out/(name+'_prefix_selection.json');assert sha(path)==freeze['selection_hashes'][name];selection=json.loads(path.read_text());data,pre,saved=study(name,entry);assert data['hashes']==selection['data_hashes'];trials=[]
        for seed in entry.get('test_seeds',protocol['test_seeds']):
            events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfgbase);stream=augment_stream(events,pre,engine.precompute(events,pre,cfgbase),cfgbase);results={};audit={}
            for arm in ARMS:
                chosen=selection['selected'][arm];cfg=dict(cfgbase,lam=chosen['lam'],threshold=chosen['margin']);r=run(engine,stream,pre,cfg,arm,chosen['q_initial']);check=execute_check(events,pre,cfg,r,explicit_service,explicit_fork);assert check['passed'],(name,seed,arm,check);results[arm]=r;audit[arm]=check;checks.append(dict(dataset=name,seed=seed,arm=arm,**check))
            ref=engine.run(stream,pre,cfgbase,'frozen',0);results['reference']=ref;audit['reference']=execute_check(events,pre,cfgbase,ref,explicit_service,explicit_fork);checks.append(dict(dataset=name,seed=seed,arm='reference',**audit['reference']));trials.append(dict(seed=seed,windows=stream['windows'],episodes=len(stream['decisions']),results=results,execution_checks=audit));print('TEST',name,seed,{a:(round(r['net'],2),r['deployments'],r['harmful'],r['beneficial']) for a,r in results.items()},flush=True);dump(out/(name+'_results_partial.json'),dict(trials=trials))
        result=dict(dataset=name,split=pre['split'],selected=selection['selected'],trials=trials,summary=summary(trials),code_sha256=codehash,protocol_sha256=prohash);dump(out/(name+'_results.json'),result);all_results[name]=result
    dump(out/'results.json',all_results);dump(out/'summary.json',{k:v['summary'] for k,v in all_results.items()});dump(out/'execution_checks.json',dict(passed=all(r['passed'] for r in checks),trajectories=len(checks),forks=sum(r['forks'] for r in checks),max_service_error=max(max(r['service_errors'].values()) for r in checks),max_fork_error=max(r['fork_error'] for r in checks),checks=checks));print('COMPLETE',flush=True)

if __name__=='__main__':main()
