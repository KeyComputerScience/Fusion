"""Decision-conditioned complete-horizon masked Gaussian precision fusion.

All historical outcomes enter only after complete lease-label maturity.
The prior, predictive covariance and Gaussian working likelihood are explicit.
Empirical coverage is measured, never inferred from the Gaussian notation.
"""
from __future__ import annotations
import argparse, datetime, hashlib, itertools, json, math, sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'work/fusion_focus_20261003/controls'))
import run_strong_controls as support

ARMS=('precision_joint','precision_diagonal','scalar_mass','no_posterior','complete_only')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def covariance(records,dimension,prior_mass=2.,prior_variance=.01):
    first=np.zeros(dimension);second=prior_mass*prior_variance*np.eye(dimension);mass=prior_mass
    for a,z in records:first+=a*z;second+=a*np.outer(z,z);mass+=a
    mu=first/mass;R=(second/mass-np.outer(mu,mu));R=(R+R.T)/2
    val,vec=np.linalg.eigh(R);return (vec*np.maximum(val,1e-6))@vec.T,mass

def state(arrived,current,pre,cfg):
    """Power-likelihood conditioning of partial joint Gaussian observations.

    Predictive covariance uses complete FULL-source blocks plus PSD prior;
    incomplete blocks remain valid marginal likelihood observations. No
    pairwise covariance imputation or pseudo-observation of missing errors.
    """
    H=cfg['horizon'];m=pre['m'];dim=m;observations=[];complete=[]
    for r in arrived[-cfg['lease_archive']:]:
        a=cfg['beta']**(current['k']-r['k'])*math.exp(-float(np.sum((current['context']-r['context'])**2))/cfg['kernel_bandwidth']**2)
        if a<1e-12:continue
        pa=(r['x']@current['reference']).argmax(1);pc=(r['x']@current['candidate']).argmax(1)
        b=np.eye(pre['classes'])[pc]-np.eye(pre['classes'])[pa]
        ids=np.flatnonzero(r['mask'])
        issued=np.einsum('sic,ic->s',r['p'][ids],b)/len(r['x'])
        rates=[];support_count=0;support_total=0
        for f in r['future']:
            p0=(f['x']@current['reference']).argmax(1);p1=(f['x']@current['candidate']).argmax(1);keep=f['keep']
            rates.append(float(np.sum((p1[keep]==f['y'][keep]).astype(float)-(p0[keep]==f['y'][keep]).astype(float)))/len(r['x']))
            support_count+=int(np.sum(p1!=p0));support_total+=len(p1)
        # A record on which the CURRENT actions coincide contains no contrast
        # information. Its zero projected residual is not an error observation.
        decision_support=support_count/max(1,support_total)
        a*=decision_support
        if a<1e-12:continue
        indices=ids
        z=issued-float(np.sum(rates))/H
        observations.append(dict(a=a,indices=indices,z=z,origin=r['k'],maturity=r['maturity']))
        if len(ids)==m:complete.append((a,z))
    R,mass=covariance(complete,dim,cfg['prior_mass'],cfg['prior_variance'])
    # Zero prior correction. Quality adjusts prior variance rather than its
    # posterior sample count; no outcome-defined masks are used.
    quality=pre['q']/np.mean(pre['q']);base=1/quality
    P0=np.diag(cfg['prior_sd']**2*base)
    precision=np.diag(1/np.diag(P0));rhs=np.zeros(dim)
    complete_precision=precision.copy();complete_rhs=rhs.copy()
    for o in observations:
        ix=o['indices'];inv=np.linalg.inv(R[np.ix_(ix,ix)])
        precision[np.ix_(ix,ix)]+=o['a']*inv;rhs[ix]+=o['a']*(inv@o['z'])
        if len(ix)==dim:complete_precision+=o['a']*inv;complete_rhs+=o['a']*(inv@o['z'])
    P=np.linalg.inv(precision);P=(P+P.T)/2;mu=P@rhs
    Pc=np.linalg.inv(complete_precision);muc=Pc@complete_rhs
    active=current['ids'];ix=active
    Pa=P[np.ix_(ix,ix)];Ra=R[np.ix_(ix,ix)]
    c=float(np.sum(Pa*Ra)/np.sum(Ra*Ra));non_scalar=float(np.linalg.norm(Pa-c*Ra)/max(np.linalg.norm(Pa),1e-15))
    return dict(mu=mu[ix],R=Ra,P=Pa,complete_mu=muc[ix],complete_P=Pc[np.ix_(ix,ix)],mass=mass,all_mass=cfg['prior_mass']+sum(o['a'] for o in observations),non_scalar=non_scalar,
                eligible=[dict(origin=o['origin'],maturity=o['maturity'],weight=o['a'],observed_components=len(o['indices'])) for o in observations],
                complete_blocks=len(complete),partial_blocks=sum(len(o['indices'])<dim for o in observations))

def augment(events,pre,base,cfg):
    pending=[];arrived=[];rows=[]
    for d in base['decisions']:
        k=d['k'];e=events[k];arrived.extend(r for r in pending if r['maturity']<=k);pending=[r for r in pending if r['maturity']>k]
        arrived=arrived[-cfg['lease_archive']:]
        current=dict(d,context=e['context'].copy(),candidate=e['candidate'],reference=e['reference'])
        mm=state(arrived,current,pre,cfg);rows.append(dict(current,temporal=mm))
        future=[]
        for j in range(k,k+cfg['horizon']):
            f=events[j];cd=(cfg['probe_drop'] if f['probe'] else 0)+(2 if f['refresh'] else 0)
            future.append(dict(x=f['x'],y=f['y'],keep=np.arange(len(f['x']))>=cd))
        pending.append(dict(k=k,maturity=d['maturity'],context=e['context'].copy(),x=e['x'],p=e['p'].copy(),mask=e['mask'].copy(),future=future,N=d['N']))
    return dict(base,decisions=rows)

def norm_terms(w,M,floor):
    z=M@w;s=math.sqrt(max(0.,float(w@z))+floor**2)
    return s,z/s,M/s-np.outer(z,z)/s**3

def solve(rate,R,P,q,quality,cfg,H):
    """Active-set Newton solve on a product of capped simplices.

    Weight objective and gate use identical directional risk terms. The
    KL term supplies strict convexity and is also deducted in admission.
    """
    m=len(quality);dim=H*m;cap=max(cfg['cap'],1/m)
    pi=quality**cfg['quality_power'];pi=pi/pi.sum();prior=np.tile(support.np.asarray(pi),H)
    w=np.tile(support.np.asarray(support.np.minimum(pi,cap)),H).reshape(H,m)
    for t in range(H):w[t]=capped(pi,cap)
    w=w.reshape(-1);tau=cfg['tau']/H;f=cfg['norm_floor'];R=R/H**2;P=P/H**2;rate=rate/H
    def value(x):
        p=norm_terms(x,P,f)[0];r=norm_terms(x,R,f)[0]
        return float(-x@rate+tau*np.sum(x*np.log(x/prior))+cfg['posterior_kappa']*p+q*r)
    active=w>=cap-1e-10;iters=0
    for iters in range(160):
        pn,pg,ph=norm_terms(w,P,f);rn,rg,rh=norm_terms(w,R,f)
        g=-rate+tau*(np.log(w/prior)+1)+cfg['posterior_kappa']*pg+q*rg
        Hess=np.diag(tau/w)+cfg['posterior_kappa']*ph+q*rh
        # Release a cap if its KKT multiplier has the wrong sign.
        for t in range(H):
            ids=np.arange(t*m,(t+1)*m);free=ids[~active[ids]]
            if not len(free):active[ids[np.argmax(g[ids])]]=False;free=ids[~active[ids]]
            nu=float(np.mean(g[free]))
            # Optimize a face before checking whether its cap must be released.
            # Releasing from an unconverged face can produce a zero-step cycle.
            if np.max(np.abs(g[free]-nu))<1e-7:
                release=ids[active[ids] & (g[ids]>nu+1e-10)]
                active[release]=False
        free=np.flatnonzero(~active);A=np.zeros((H,len(free)))
        for t in range(H):A[t,free//m==t]=1
        K=np.block([[Hess[np.ix_(free,free)],A.T],[A,np.zeros((H,H))]])
        sol=np.linalg.solve(K,np.r_[-g[free],np.zeros(H)]);step=np.zeros(dim);step[free]=sol[:len(free)]
        residual=[]
        for t in range(H):
            ix=np.arange(t*m,(t+1)*m);fr=ix[~active[ix]];nu=float(np.mean(g[fr]))
            residual.extend(np.abs(g[fr]-nu));residual.extend(np.maximum(g[ix[active[ix]]]-nu,0))
        kkt=max(residual,default=0.)
        if kkt<2e-8:break
        if np.max(np.abs(step))<1e-13:break
        alpha=1.;pos=step>1e-15;neg=step<-1e-15
        if pos.any():alpha=min(alpha,float(np.min((cap-w[pos])/step[pos])))
        if neg.any():alpha=min(alpha,float(np.min((w[neg]-1e-14)/-step[neg]))*.99)
        old=value(w);slope=float(g@step)
        for _ in range(50):
            candidate=w+alpha*step
            if np.min(candidate)>0 and value(candidate)<=old+1e-4*alpha*slope+1e-14:break
            alpha*=.5
        w=candidate;active=w>=cap-1e-9
    if kkt>=1e-6 and H==1:
        # Exact small-source fallback: enumerate cap faces, optimize each
        # positive free simplex, and retain only primal/dual-valid solutions.
        best=None
        for count in range(m):
            for fixed in itertools.combinations(range(m),count):
                remaining=1-cap*count
                if remaining<=0:continue
                free=np.array([i for i in range(m) if i not in fixed],int);x=np.full(m,cap)
                x[free]=remaining*pi[free]/pi[free].sum()
                for _ in range(200):
                    _,pg,ph=norm_terms(x,P,f);_,rg,rh=norm_terms(x,R,f)
                    gg=-rate+tau*(np.log(x/prior)+1)+cfg['posterior_kappa']*pg+q*rg
                    hh=np.diag(tau/x)+cfg['posterior_kappa']*ph+q*rh
                    ff=gg[free];kk=np.max(np.abs(ff-ff.mean()))
                    if kk<2e-9:break
                    mat=np.block([[hh[np.ix_(free,free)],np.ones((len(free),1))],[np.ones((1,len(free))),np.zeros((1,1))]])
                    step=np.zeros(m);step[free]=np.linalg.solve(mat,np.r_[-ff,0])[:-1];alpha=1.;neg=step<0
                    if neg.any():alpha=min(alpha,float(np.min(-x[neg]/step[neg]))*.99)
                    old=value(x);slope=float(gg@step)
                    for _ in range(50):
                        xx=x+alpha*step
                        if np.min(xx)>0 and value(xx)<=old+1e-4*alpha*slope+1e-14:break
                        alpha*=.5
                    x=xx
                nu=float(np.mean(gg[free]));face_kkt=max(float(kk),float(np.max(np.maximum(gg[list(fixed)]-nu,0))) if fixed else 0.)
                if max(x)<=cap+1e-9 and face_kkt<1e-6:
                    val=value(x)
                    if best is None or val<best[0]:best=(val,x,face_kkt)
        if best is not None:_,w,kkt=best
    primal=max(float(np.max(np.abs(w.reshape(H,m).sum(1)-1))),float(np.max(w-cap)),float(-np.min(w)))
    return w,dict(kkt=float(kkt),primal=primal,iterations=iters+1,converged=bool(kkt<1e-6 and primal<1e-8)),value(w)

def capped(pi,cap):
    z=np.zeros(len(pi));free=np.ones(len(pi),bool);remaining=1.
    for _ in pi:
        ix=np.flatnonzero(free);v=remaining*pi[ix]/pi[ix].sum();over=v>cap
        if not over.any():z[ix]=v;break
        z[ix[over]]=cap;free[ix[over]]=False;remaining-=cap*int(over.sum())
    return z

def forecast(d,pre,cfg,arm,q):
    H=1;m=len(d['ids']);mm=d['temporal'];mu=mm['mu'];R=mm['R'];P=mm['P'];rate=d['h']-mu
    config=dict(cfg,posterior_kappa=0. if arm=='no_posterior' else 1.)
    if arm=='precision_diagonal':R=np.diag(np.diag(R));P=np.diag(np.diag(P))
    if arm=='scalar_mass':P=R/(mm['all_mass']+1)
    if arm=='complete_only':rate=d['h']-mm['complete_mu'];P=mm['complete_P']
    Hsolve=1
    w,cert,obj=solve(rate,R,P,q,pre['q'][d['ids']],config,Hsolve)
    pnorm=math.sqrt(max(0.,float(w@P@w))/Hsolve**2+cfg['norm_floor']**2)
    rnorm=math.sqrt(max(0.,float(w@R@w))/Hsolve**2+cfg['norm_floor']**2)
    pi=pre['q'][d['ids']]**cfg['quality_power'];pi/=pi.sum()
    entropy=cfg['tau']/Hsolve*float(np.sum(w*np.log(w/np.tile(pi,Hsolve))))
    gain=float(d['N']*(w@rate/Hsolve-config['posterior_kappa']*pnorm-entropy))
    raw_gain=gain
    # A regularizing prior alone cannot license paid deployment. All methods
    # share this causal information-availability condition; the partial-block
    # ablation can qualify only through the blocks it actually retains.
    ready=bool(mm['complete_blocks']>0) if arm=='complete_only' else bool(mm['eligible'])
    if not ready:gain=0.
    sd=float(d['N']*rnorm)
    # truegross is used only to create the offline callback, never the solve.
    return w,cert,gain,sd,(gain-d['truegross'])/sd,dict(posterior_norm=pnorm,entropy=entropy,objective=obj,horizon_weights=Hsolve,information_ready=ready,raw_optimized_gain=raw_gain)

def accounted_run(stream,pre,cfg,mode,qinit,selection=False):
    q=float(qinit);pending=[];qlog=[];rows=[];actions=[];net=stream['basegross']-stream['commonfees'];gross=stream['basegross'];fees=stream['commonfees'];drops=stream['commondrops'];correct=stream['basecorrect'];maxkkt=0.;failures=0;boundary=stream['windows']//2 if selection else 0;select_net=0.;clip_sum=0.;violations=0;updates=0
    for d in stream['decisions']:
        k=d['k']
        for due,issuedk,issuedq,score,issuedstd in sorted(pending,key=lambda p:p[0]):
            if due<=k:
                violation=int(score>issuedq);proposal=q+.05*(violation-.1);newq=max(cfg.get("q_floor",0.),proposal);regulator=newq-proposal;clip_sum+=regulator;violations+=violation;updates+=1;qlog.append(dict(maturity=due,issued_index=issuedk,update_index=k,issued_q=issuedq,issued_std=issuedstd,standardized_score=score,violation=violation,q_before=q,q_after=newq,regulator=regulator));q=newq
        pending=[p for p in pending if p[0]>k]
        w,cert,gain,sd,standardized=decision_forecast(d,pre,cfg,mode);maxkkt=max(maxkkt,cert['kkt']);failures+=not cert['converged'];gated=mode in ('bayes_gate','bayes_both','frequentist_gate');penalty=q*sd if gated else 0.;score=gain-penalty-5.-cfg['threshold'];action=score>0
        if mode=='periodic':action=True
        if mode=='frozen':action=False
        # Prefixq fit uses only completedfirsthalfcanaryforks; gridselection secondhalf.
        if selection and k<boundary:action=False
        if action:net+=d['localnet'];gross+=d['grossdifference'];fees+=3.;drops+=cfg['deploy_drop'];correct+=d['accdifference'];select_net+=d['localnet'] if k>=boundary else 0.
        actions.append(bool(action));row=dict(k=k,action=bool(action),weights=w.tolist(),gain=gain,posterior_or_block_sd=sd,q_issued=q,penalty=penalty,gate_score=score,standardized_score=standardized,truegross=d['truegross'],local_net=d['localnet'],disagreement=d['disagreement'],maturity=d['maturity'],lower_covered=bool(standardized<=q),future_accuracy_gain=d['future'],blocks=d['blocks'],mass=d['mass'],scale=d['scale'],solver_converged=cert['converged'],kkt=cert['kkt']);rows.append(row)
        # All localforks have outcomes once their4windowsfeedback hasmatured; the
        # simulation keeps offline truth separate from causal q state transitions.
        if not selection or k>=boundary:pending.append((d['maturity'],k,q,standardized,sd))
    # Flush only feedback available by replayend; no additionalexecutedgates.
    for due,issuedk,issuedq,score,issuedstd in sorted(pending,key=lambda p:p[0]):
        if due<=stream['windows']:
            violation=int(score>issuedq);proposal=q+.05*(violation-.1);newq=max(cfg.get("q_floor",0.),proposal);regulator=newq-proposal;clip_sum+=regulator;violations+=violation;updates+=1;qlog.append(dict(maturity=due,issued_index=issuedk,update_index=stream['windows'],issued_q=issuedq,issued_std=issuedstd,standardized_score=score,violation=violation,q_before=q,q_after=newq,regulator=regulator));q=newq
    identity=violations-(.1*updates+(q-qinit-clip_sum)/.05);acts=[r for r in rows if r['action']];selected=[r for r in rows if r['k']>=boundary];informative=[r for r in selected if r['disagreement']>0]
    return dict(net=net,gross=gross,fees=fees,drops=drops,accuracy=correct/stream['n'],deployments=sum(actions),actions=actions,rows=rows,q_updates=qlog,q_initial=qinit,q_final=q,clip_regulator_sum=clip_sum,calibration_updates=updates,miscoverage=violations/updates if updates else None,calibration_identity_error=identity,coverage=float(np.mean([r['lower_covered'] for r in selected])) if selected else None,informative_coverage=float(np.mean([r['lower_covered'] for r in informative])) if informative else None,informative_episodes=len(informative),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),zero=sum(r['local_net']==0 for r in acts),solver_failures=int(failures),max_kkt=maxkkt,selection_net=select_net,closed_loop_identity_error=net-(stream['basegross']-stream['commonfees']+sum(r['local_net'] for r in acts)))


def run(engine,stream,pre,cfg,arm,qinit,selection=False):
    # The unchanged engine retains callback accounting and service accounting.
    # Its forecast hook receives a causally mirrored q state before each solve.
    pending=[];q=float(qinit);boundary=stream['windows']//2 if selection else 0;metadata=[]
    def hook(d,p,c,mode):
        nonlocal pending,q
        k=d['k']
        for due,issuedq,R in sorted(pending,key=lambda x:x[0]):
            if due<=k:q=max(cfg.get('q_floor',0.),q+.05*(int(R>issuedq)-.1))
        pending=[v for v in pending if v[0]>k]
        w,cert,F,S,R,extra=forecast(d,p,c,arm,q)
        if not selection or k>=boundary:pending.append((d['maturity'],q,R))
        metadata.append(dict(q=q,extra=extra))
        return w,cert,F,S,R
    global decision_forecast
    decision_forecast=hook
    result=accounted_run(stream,pre,cfg,'bayes_gate',qinit,selection)
    for row,d,meta in zip(result['rows'],stream['decisions'],metadata):
        assert abs(row['q_issued']-meta['q'])<1e-12
        mm=d['temporal'];row.update(arm=arm,**meta['extra'],joint_mu=mm['mu'].tolist(),joint_R=mm['R'].tolist(),joint_P=mm['P'].tolist(),
                                  precision_non_scalar=mm['non_scalar'],active_source_ids=d['ids'].tolist(),eligible=mm['eligible'],partial_blocks=mm['partial_blocks'])
        row['point_value_opportunity']=bool(row['gain']-5-cfg['threshold']>0)
        row['pre_gate_proposal']=bool(row['gate_score']>0)
        assert all(o['maturity']<=row['k'] for o in mm['eligible'])
    return result

def initial_q(streams,pre,cfg,arm):
    scores=[]
    for stream in streams:
        for d in stream['decisions']:
            if d['maturity']<=stream['windows']//2:
                result=forecast(d,pre,cfg,arm,cfg.get('q_floor',0.))
                if result[5]['information_ready']:scores.append(result[4])
    return max(cfg.get('q_floor',0.),float(np.quantile(scores,.9,method='higher'))) if scores else cfg.get('q_floor',0.),scores

def summarize(trials):
    out={}
    for arm in trials[0]['results']:
        rs=[t['results'][arm] for t in trials];rows=[z for r in rs for z in r['rows']]
        groups=dict(issued=rows,origin=[z for z in rows if z['disagreement']>0],proposed=[z for z in rows if z.get('pre_gate_proposal',False)],admitted=[z for z in rows if z['action']])
        out[arm]=dict(mean_net=float(np.mean([r['net'] for r in rs])),harmful=sum(r['harmful'] for r in rs),beneficial=sum(r['beneficial'] for r in rs),
                      admissions=sum(r['deployments'] for r in rs),coverage={k:[sum(z['lower_covered'] for z in rr),len(rr)] for k,rr in groups.items()})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tasks',nargs='+',default=['mhealth319','gas224']);ap.add_argument('--phase',choices=['calibrate','test'],required=True);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    engine,recovery,guard,source,cfg0,study,service,fork=support.setup(support.DEFAULT_PACKAGE,support.DEFAULT_GUARD)
    cfg0.update(prior_sd=.1,tau=.01,q_floor=1.2815515655446004);names=args.tasks
    if any(n in ('har780','pamap231','rss348') for n in names):
        sys.path.insert(0,str(PROJECT/'work/fusion_focus_20261003/fresh'))
        from new_data_adapter import load_dataset,make_prefix
        old_study=study
        def study(name,spec):
            if name=='har780':folder=PROJECT/'work/fusion_focus_20261003/fresh/data'
            elif name in ('pamap231','rss348'):folder=ROOT/'physical/data'
            else:return old_study(name,spec)
            data,derivation=load_dataset(folder,name)
            return data,make_prefix(data,derivation,cfg0,engine)
        source['studies'].update(har780=dict(calibration_seeds=[83001,83002,83003],test_seeds=[84001,84002,84003,84004,84005]),pamap231=dict(calibration_seeds=[85001,85002,85003],test_seeds=[86001,86002,86003,86004,86005]))
        source['studies']['rss348']=dict(calibration_seeds=[87001,87002,87003],test_seeds=[88001,88002,88003,88004,88005])
    if args.phase=='calibrate':
        assert not (out/'results.json').exists();dump(out/'protocol.json',dict(utc=now(),code_sha256=sha(__file__),tasks=names,arms=ARMS,prior_sd=[.05,.1,.2],q_floor=[0.,.64,1.2815515655446004],margin=0.,evidence_scope='all seven previous physical tasks are development; new PAMAP/RSS protocols are separate frozen validation',cfg=cfg0))
    else:
        protocol=json.loads((out/'protocol.json').read_text());assert protocol['code_sha256']==sha(__file__)
    all_results={}
    for name in names:
        data,pre=study(name,source['studies'][name]);spec=source['studies'][name]
        if args.phase=='calibrate':
            seeds=spec.get('calibration_seeds',source['calibration_seeds']);events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,cfg0) for s in seeds];bases=[engine.precompute(e,pre,cfg0) for e in events];grid=[]
            for sd in (.05,.1,.2):
                cfg=dict(cfg0,prior_sd=sd);streams=[augment(e,pre,b,cfg) for e,b in zip(events,bases)]
                for arm in ARMS:
                    for floor in (0.,.64,1.2815515655446004):
                        cc=dict(cfg,threshold=0.,q_floor=floor);qi,scores=initial_q(streams,pre,cc,arm)
                        rr=[run(engine,s,pre,cc,arm,qi,True) for s in streams]
                        grid.append(dict(arm=arm,prior_sd=sd,margin=0.,q_floor=floor,q_initial=qi,fit_scores=scores,net=float(np.mean([r['selection_net'] for r in rr])),failures=sum(r['solver_failures'] for r in rr)))
                    print('CAL',name,sd,arm,flush=True)
            selected={a:max((g for g in grid if g['arm']==a),key=lambda g:(g['net'],g['q_floor'],-g['prior_sd'])) for a in ARMS}
            dump(out/(name+'_selection.json'),dict(split=pre['split'],data_hashes=data['hashes'],grid=grid,selected=selected));print('SELECT',name,{a:(v['prior_sd'],v['margin'],v['net']) for a,v in selected.items()},flush=True)
        else:
            selection=json.loads((out/(name+'_selection.json')).read_text());assert data['hashes']==selection['data_hashes'];trials=[]
            for seed in spec.get('test_seeds',source['test_seeds']):
                e=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg0);base=engine.precompute(e,pre,cfg0);streams={};results={};checks={}
                for arm in ARMS:
                    ss=selection['selected'][arm];cfg=dict(cfg0,prior_sd=ss['prior_sd'],threshold=ss['margin'],q_floor=ss['q_floor'])
                    if ss['prior_sd'] not in streams:streams[ss['prior_sd']]=augment(e,pre,base,cfg)
                    result=run(engine,streams[ss['prior_sd']],pre,cfg,arm,ss['q_initial'])
                    check=recovery.execute_check(e,pre,cfg,result,service,fork);assert check['passed'],check
                    results[arm]=result;checks[arm]=check
                results['reference']=engine.run(base,pre,cfg0,'frozen',0)
                trials.append(dict(seed=seed,results=results,checks=checks));print('DEV',name,seed,{a:(round(r['net'],2),r['harmful'],r['beneficial']) for a,r in results.items()},flush=True)
            all_results[name]=dict(selected=selection['selected'],trials=trials,summary=summarize(trials));dump(out/(name+'_results.json'),all_results[name])
    if args.phase=='test':dump(out/'results.json',all_results)

if __name__=='__main__':main()
