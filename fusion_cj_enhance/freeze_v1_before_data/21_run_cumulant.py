"""Paired complete-error predictive cumulant: declared development extension.

All physical traces here have been inspected previously. This runner never
edits the frozen core, never selects from test outcomes, and retains every
calibration trial and every held-out trajectory. No universal gain is claimed.
"""
from pathlib import Path
import argparse, datetime, gzip, hashlib, importlib.util, itertools, json, math, sys
sys.dont_write_bytecode = True
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
FACTOR=PROJECT/'work/fusion_strengthening_20261003/matrix/factorized/factorized_control.py'
spec=importlib.util.spec_from_file_location('cumulant_parent',FACTOR)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
core=parent.core
ADAPTER=PROJECT/'work/fusion_strengthening_20261003/fresh/new_data_adapter.py'
spec=importlib.util.spec_from_file_location('cumulant_adapter',ADAPTER)
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
CORE=PROJECT/'work/fusion_temporal_20261003/temporal_fusion.py'
ARMS=('paired_cumulant','marginal_cumulant','factorized_cumulant','matrix_cumulant','sandwich_cumulant')
TASKS=('rss348','arem366','gashome362')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,v):
    s=json.dumps(v,indent=2,allow_nan=False)
    if str(p).endswith('.gz'):
        with gzip.open(p,'wt') as f:f.write(s)
    else:Path(p).write_text(s)
def load(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt') as f:return json.load(f)
    return json.loads(Path(p).read_text())

def atom_state(observations,R,mu,ids,cfg):
    """Real joint atoms; missing entries integrated with a declared Gaussian law.

    Missing values are not treated as observations or independent sources.
    The working conditional completion includes its Schur uncertainty. A
    prior atom prevents zero empirical variance. Centering holds the issued
    correction mean fixed across cumulant and second-order controls.
    """
    atoms=[np.zeros(len(mu))];covariances=[R.copy()];weights=[cfg['prior_mass']]
    masks=[[]]
    for b in observations:
        ix=b['ids'];cross=R[:,ix]
        inverse=np.linalg.solve(R[np.ix_(ix,ix)],np.eye(len(ix)))
        mean=mu+cross@inverse@(b['residual']-mu[ix])
        V=R-cross@inverse@cross.T;V=(V+V.T)/2
        # Numerically restore PSD without assigning a positive missing-mask
        # likelihood to any coordinate that was never observed.
        ev,U=np.linalg.eigh(V);V=(U*np.maximum(ev,0.))@U.T
        atoms.append(mean);covariances.append(V);weights.append(b['weight']);masks.append(ix.tolist())
    alpha=np.array(weights);alpha/=alpha.sum()
    z=np.array(atoms);center=alpha@z;v=(z-center)[:,ids]
    V=np.array([x[np.ix_(ids,ids)] for x in covariances])
    C=np.einsum('j,js,jt->st',alpha,v,v)+np.einsum('j,jst->st',alpha,V)
    return dict(alpha=alpha,v=v,V=V,C=C,masks=masks,center=center[ids],mass=float(sum(weights)))

def cumulant_state(arrived,current,pre,cfg):
    mm=parent.factorized_state(arrived,current,pre,cfg)
    obs,complete=parent.matrix.collect_blocks(arrived,current,pre,cfg)
    R,_=core.covariance(complete,pre['m'],cfg['prior_mass'],cfg['prior_variance'])
    K=np.diag((pre['q']/np.mean(pre['q']))/cfg['prior_sd']**2)
    A=K.copy();rhs=np.zeros(pre['m'])
    diagonal=np.diag(R);facprec=np.diag(K).copy();facrhs=np.zeros(pre['m'])
    for b in obs:
        ix=b['ids'];inv=np.linalg.solve(R[np.ix_(ix,ix)],np.eye(len(ix)))
        A[np.ix_(ix,ix)]+=b['weight']*inv;rhs[ix]+=b['weight']*(inv@b['residual'])
        facprec[ix]+=b['weight']/diagonal[ix];facrhs[ix]+=b['weight']*b['residual']/diagonal[ix]
    mu=np.linalg.solve(A,rhs);facmu=facrhs/facprec
    mm['paired_atoms']=atom_state(obs,R,mu,current['ids'],cfg)
    mm['factorized_atoms']=atom_state(obs,np.diag(diagonal),facmu,current['ids'],cfg)
    assert np.max(np.abs(mu[current['ids']]-mm['mu']))<1e-10
    assert np.max(np.abs(facmu[current['ids']]-mm['factorized_mu']))<1e-10
    return mm

def paired_terms(w,atoms,P,theta):
    v,V,a=atoms['v'],atoms['V'],atoms['alpha']
    Vw=np.einsum('jst,t->js',V,w)
    log=np.log(a)+theta*(v@w)+.5*theta**2*np.einsum('js,s->j',Vw,w)
    shift=float(log.max());exp=np.exp(log-shift);rho=exp/exp.sum()
    d=v+theta*Vw;grad=rho@d
    Hess=theta*(np.einsum('j,js,jt->st',rho,d,d)-np.outer(grad,grad))+theta*np.einsum('j,jst->st',rho,V)
    value=(shift+math.log(float(exp.sum())))/theta
    return value+.5*theta*float(w@P@w),grad+theta*(P@w),Hess+theta*P

def marginal_terms(w,atoms,P,theta):
    # Same global alpha, same per-source marginals, same correction and P;
    # only the joint pairing of predictive atoms is removed.
    v,V,a=atoms['v'],atoms['V'],atoms['alpha'];diag=np.diagonal(V,axis1=1,axis2=2)
    log=np.log(a)[:,None]+theta*v*w[None,:]+.5*theta**2*diag*w[None,:]**2
    shift=log.max(0);exp=np.exp(log-shift);rho=exp/exp.sum(0)
    d=v+theta*diag*w[None,:];grad=(rho*d).sum(0)
    Hess=np.diag(theta*((rho*d*d).sum(0)-grad**2+(rho*diag).sum(0)))
    value=float(np.sum(shift+np.log(exp.sum(0)))/theta)
    return value+.5*theta*float(w@P@w),grad+theta*(P@w),Hess+theta*P

def risk_terms(w,atoms,P,theta,arm):
    if arm in ('marginal_cumulant','factorized_cumulant'):return marginal_terms(w,atoms,P,theta)
    if arm in ('matrix_cumulant','sandwich_cumulant'):
        C=atoms['C']+P
        return .5*theta*float(w@C@w),theta*(C@w),theta*C
    return paired_terms(w,atoms,P,theta)

def solve(d,pre,cfg,arm,q):
    mm=d['temporal'];factor=arm=='factorized_cumulant'
    mu=mm['factorized_mu'] if factor else mm['mu']
    P=mm['factorized_P'] if factor else (mm['sandwich_P'] if arm=='sandwich_cumulant' else mm['P'])
    R=mm['factorized_R'] if factor else mm['R']
    atoms=mm['factorized_atoms'] if factor else mm['paired_atoms']
    rate=d['h']-mu;m=len(rate);cap=max(cfg['cap'],1/m)
    pi=pre['q'][d['ids']]**cfg['quality_power'];pi/=pi.sum()
    tau=cfg['tau'];theta=cfg['tail_temperature'];floor=cfg['norm_floor']
    def terms(w):
        v,g,H=risk_terms(w,atoms,P,theta,arm);rn,rg,rH=core.norm_terms(w,R,floor)
        entropy=tau*float(np.sum(w*np.log(w/pi)))
        return (-float(w@rate)+v+entropy+q*rn,
                -rate+g+tau*(np.log(w/pi)+1)+q*rg,
                H+np.diag(tau/w)+q*rH)
    w=core.capped(pi,cap);best=None;iters=0
    if m==1:
        val,g,H=terms(w);best=(val,w,0.)
    else:
        # Exact low-dimensional cap-face enumeration keeps convergence
        # certificates independent of which risk representation is used.
        for count in range(m):
            for fixed in itertools.combinations(range(m),count):
                remaining=1-cap*count
                if remaining<=0:continue
                free=np.array([i for i in range(m) if i not in fixed]);x=np.full(m,cap)
                x[free]=remaining*pi[free]/pi[free].sum()
                for it in range(100):
                    val,g,H=terms(x);gg=g[free];kk=float(np.max(np.abs(gg-gg.mean())))
                    if kk<2e-9:break
                    A=np.block([[H[np.ix_(free,free)],np.ones((len(free),1))],[np.ones((1,len(free))),np.zeros((1,1))]])
                    step=np.zeros(m);step[free]=np.linalg.solve(A,np.r_[-gg,0.])[:-1]
                    alpha=1.;negative=step<0
                    if negative.any():alpha=min(alpha,float(np.min(-x[negative]/step[negative]))*.99)
                    slope=float(g@step)
                    for _ in range(40):
                        candidate=x+alpha*step
                        if candidate.min()>0 and terms(candidate)[0]<=val+1e-4*alpha*slope+1e-14:break
                        alpha*=.5
                    x=candidate
                iters+=it+1;val,g,H=terms(x);gg=g[free];nu=gg.mean()
                kk=max(float(np.max(np.abs(gg-nu))),float(np.max(np.maximum(g[list(fixed)]-nu,0.))) if fixed else 0.)
                if x.max()<=cap+1e-9 and kk<1e-6 and (best is None or val<best[0]):best=(val,x,kk)
    if best is None:
        # A numerical failure cannot authorize deployment or create a zero
        # calibration denominator. The caller retains the failed certificate.
        S=float(d['N']*core.norm_terms(w,R,floor)[0])
        return w,dict(converged=False,kkt=1e6,primal=0.,iterations=iters),0.,S,0.,atoms
    val,w,kkt=best;risk=risk_terms(w,atoms,P,theta,arm)[0]
    entropy=tau*float(np.sum(w*np.log(w/pi)))
    F=float(d['N']*(w@rate-risk-entropy))
    S=float(d['N']*core.norm_terms(w,R,floor)[0])
    primal=max(abs(float(w.sum())-1),float(w.max()-cap),float(-w.min()))
    cert=dict(converged=bool(kkt<1e-6 and primal<1e-8),kkt=kkt,primal=primal,iterations=iters)
    return w,cert,F,S,risk,atoms

def cumulant_forecast(d,pre,cfg,arm,q):
    if arm not in ARMS:return parent.factorized_forecast(d,pre,cfg,arm,q)
    w,cert,F,S,risk,atoms=solve(d,pre,cfg,arm,q);raw=F
    ready=bool(d['temporal']['eligible'])
    if not ready:F=0.
    return w,cert,F,S,(F-d['truegross'])/S,dict(information_ready=ready,raw_optimized_gain=raw,
        predictive_cumulant=risk,tail_temperature=cfg['tail_temperature'],atoms=len(atoms['alpha']),
        atom_mass=atoms['mass'],pairing_control=arm,posterior_norm=0.,entropy=0.)

core.state=cumulant_state;core.forecast=cumulant_forecast
def binding():return parent.matrix.binding()
def task_data(name,cfg,engine,source,study):
    if name=='rss348':return study(name,source['studies'][name])
    data,derivation=adapter.load_dataset(PROJECT/'work/fusion_strengthening_20261003/fresh/data',name)
    return data,adapter.make_prefix(data,derivation,cfg,engine)
def seeds(name):return ([87001,87002,87003],list(range(88001,88006))) if name=='rss348' else ([91001,91002,91003],list(range(92001,92006)))

def freeze():
    assert not (ROOT/'protocol.json').exists()
    inputs=[CORE,FACTOR,parent.MATRIX_SOURCE,ADAPTER]
    dump(ROOT/'protocol.json',dict(utc=now(),runner_sha256=sha(__file__),input_sha256={str(p):sha(p) for p in inputs},
        tasks=TASKS,arms=ARMS,status='development extension on previously examined physical traces',
        objective='N[w^T(h-mu)-Gamma(w)-tau KL]-q*N*sqrt(w^T R w+nu^2)-5',
        Gamma='0.5*t*w^T P w + log(sum alpha_j exp(t*w^T v_j+0.5*t^2*w^T V_j*w))/t',
        prior_atom='weight a0, zero raw location and predictive covariance R',
        masked_atoms='conditional Gaussian means and Schur covariances from existing global R and mu; observed blocks unchanged',
        center='weighted global atom center, so all joint and marginal controls have identical mean and sourcewise atom marginals',
        matrix='exact second-order cumulant of same atoms plus either posterior curvature P or cluster sandwich P',
        factorized='global diagonal R before correction/conditional completion; product of marginal cumulants',
        shared='same raw records, masks, kernel, history, quality, source models, preparation, q feedback and fixed130/B130 guard',
        calibration=dict(temperature=[2.,8.,32.],q_floor=[0.,.64,1.2815515655446004],prior_sd=.1,
            trials_per_arm=27,q_initial='ready matured first-half higher 90th percentile',
            selection='max second-half mean utility; ties larger floor then smaller temperature',seeds={n:seeds(n) for n in TASKS}),
        primary_guard=dict(reserve=130.,budget=130.),diagnostic_budgets=[0.,110.,130.,260.],
        guarantees='strict convexity and information distinction; budget safety is shared and independent of statistical correctness',
        reporting='every result retained; no test-driven revision; no claim of independent sites or universal superiority',
        runtime=dict(python=sys.version,numpy=np.__version__)))
def verify():
    p=load(ROOT/'protocol.json');assert sha(__file__)==p['runner_sha256']
    for path,want in p['input_sha256'].items():assert sha(path)==want
    return p
def calibrate():
    verify();engine,recovery,guard,source,cfg0,study,service,fork=binding()
    cfg0.update(prior_sd=.1)
    for name in TASKS:
        assert not (ROOT/(name+'_selection.json')).exists()
        data,pre=task_data(name,cfg0,engine,source,study);cal,_=seeds(name)
        events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,cfg0) for s in cal]
        bases=[engine.precompute(e,pre,cfg0) for e in events]
        streams=[core.augment(e,pre,b,cfg0) for e,b in zip(events,bases)];grid=[];trials=[]
        for theta in (2.,8.,32.):
            for arm in ARMS:
                for floor in (0.,.64,1.2815515655446004):
                    cfg=dict(cfg0,tail_temperature=theta,q_floor=floor,threshold=0.)
                    qi,scores=core.initial_q(streams,pre,cfg,arm)
                    rr=[core.run(engine,s,pre,cfg,arm,qi,True) for s in streams]
                    assert all(r['solver_failures']==0 for r in rr)
                    g=dict(arm=arm,temperature=theta,q_floor=floor,prior_sd=.1,q_initial=qi,
                        fit_scores=scores,net=float(np.mean([r['selection_net'] for r in rr])))
                    grid.append(g);trials.append(dict(configuration=g,results=rr))
                print('CAL',name,theta,arm,flush=True)
        selected={a:max((g for g in grid if g['arm']==a),key=lambda g:(g['net'],g['q_floor'],-g['temperature'])) for a in ARMS}
        dump(ROOT/(name+'_calibration_trials.json.gz'),trials)
        dump(ROOT/(name+'_selection.json'),dict(selected=selected,grid=grid,data_hashes=data['hashes'],split=pre['split']))
        print('SELECT',name,{a:(v['temperature'],v['q_floor'],v['q_initial'],v['net']) for a,v in selected.items()},flush=True)
    dump(ROOT/'before_test.json',dict(utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),selections={n:sha(ROOT/(n+'_selection.json')) for n in TASKS}))
def test():
    p=verify();frozen=load(ROOT/'before_test.json');assert frozen['protocol_sha256']==sha(ROOT/'protocol.json')
    engine,recovery,guard,source,cfg0,study,service,fork=binding();cfg0.update(prior_sd=.1)
    for name in TASKS:
        assert not (ROOT/(name+'_results.json.gz')).exists()
        selection=load(ROOT/(name+'_selection.json'));assert sha(ROOT/(name+'_selection.json'))==frozen['selections'][name]
        data,pre=task_data(name,cfg0,engine,source,study);assert data['hashes']==selection['data_hashes']
        _,testseeds=seeds(name);trials=[];guarded=[];maxerr=0.;maxkkt=0.
        for seed in testseeds:
            events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg0)
            base=engine.precompute(events,pre,cfg0);stream=core.augment(events,pre,base,cfg0);results={};checks={};reference=service(events,pre,[],cfg0)
            for arm in ARMS:
                ss=selection['selected'][arm];cfg=dict(cfg0,tail_temperature=ss['temperature'],q_floor=ss['q_floor'],threshold=0.)
                r=core.run(engine,stream,pre,cfg,arm,ss['q_initial']);assert r['solver_failures']==0
                maxkkt=max(maxkkt,r['max_kkt']);check=recovery.execute_check(events,pre,cfg,r,service,fork);assert check['passed']
                results[arm]=r;checks[arm]=check
                for budget in p['diagnostic_budgets']:
                    gg=guard.replay(r['rows'],len(events),budget,'gross_loss',{x['k']:130. for x in r['rows']})
                    actual=service(events,pre,gg['rows'],cfg);prefix=guard.reconstruct_prefix_increment(events,gg['rows'],cfg)
                    act=[x for x in gg['rows'] if x['action']];inc=actual['net']-reference['net'];loss=sum(max(0.,-x['local_net']) for x in act)
                    err=max(abs(inc-sum(x['local_net'] for x in act)),abs(inc-prefix['final']),abs(inc-gg['final_settled_increment']))
                    maxerr=max(maxerr,err);assert err<1e-8 and loss<=budget+1e-8 and prefix['minimum']>=-budget-1e-8
                    assert all(l['spent_loss']+l['reserved']<=budget+1e-8 for l in gg['ledger'])
                    guarded.append(dict(task=name,seed=seed,arm=arm,budget=budget,reserve=130.,increment=inc,net=actual['net'],
                        negative_loss=loss,minimum_prefix=prefix['minimum'],admissions=len(act),harmful=sum(x['local_net']<0 for x in act),
                        beneficial=sum(x['local_net']>0 for x in act),zero=sum(x['local_net']==0 for x in act),
                        refused=sum(x['proposed_action'] and not x['action'] for x in gg['rows']),rows=gg['rows'],ledger=gg['ledger']))
            trials.append(dict(seed=seed,reference=reference,results=results,checks=checks))
            print('TEST',name,seed,{a:(round(r['net']-reference['net'],2),r['harmful'],r['beneficial']) for a,r in results.items()},flush=True)
        dump(ROOT/(name+'_results.json.gz'),dict(trials=trials,selected=selection['selected']))
        dump(ROOT/(name+'_guarded.json.gz'),guarded)
        dump(ROOT/(name+'_verification.json'),dict(passed=True,max_service_error=maxerr,max_kkt=maxkkt,unguarded=25,guarded=len(guarded),unchanged_core=sha(CORE)==p['input_sha256'][str(CORE)]))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','calibrate','test']);args=ap.parse_args()
    {'freeze':freeze,'calibrate':calibrate,'test':test}[args.phase]()
