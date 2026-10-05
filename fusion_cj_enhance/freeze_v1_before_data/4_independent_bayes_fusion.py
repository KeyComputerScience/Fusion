"""Frozen-protocol held-out multisensor deployment experiments.

No test label enters model selection, uncertainty calibration, source reliability,
or preprocessing. Test feedback enters online model/archive updates only after its
recorded arrival. Real covariates/targets, simulated service/latency/cost queue.
"""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
MODES=('mean_only','diagonal','joint','bayes_weights','bayes_gate','bayes_both','posterior_unused','periodic','frozen')
BASE=dict(window=32,train_fraction=.30,audit_fraction=.30,lr=.08,l2=.003,
          initial_steps=500,probe_steps=20,probe_period=4,train_buffer=512,
          archive=48,beta=.97,prior_mass=2.,kernel_bandwidth=3.,tau=.05,
          lam=1.,eta=4.,kappa=0.,threshold=0.,horizon=4,deploy_fee=2.,
          deploy_drop=2,probe_drop=1,step_fee=.001,cooldown=2,budget=8,
          delay_low=1,delay_high=4,cap=.8,target_coverage=.9,
          source_dropout_period=13,source_dropout_length=2,
          quality_power=1.,uncertainty_floor=1e-12)


def softmax(z):
    z=z-z.max(axis=-1,keepdims=True);p=np.exp(z);return p/p.sum(axis=-1,keepdims=True)


def gradient(w,x,y,steps,cfg,classes):
    target=np.eye(classes)[y]
    for _ in range(steps):
        w=w-cfg['lr']*(x.T@(softmax(x@w)-target)/len(x)+cfg['l2']*w)
    return w


def capped_pi(pi,cap):
    pi=np.maximum(pi,1e-14);pi/=pi.sum();out=np.zeros(len(pi));free=np.ones(len(pi),bool);remaining=1.
    for _ in pi:
        ids=np.flatnonzero(free);z=remaining*pi[ids]/pi[ids].sum();over=z>cap
        if not over.any():out[ids]=z;break
        out[ids[over]]=cap;free[ids[over]]=False;remaining-=cap*over.sum()
    return out


def convex_fuse(q,S,U,cfg,mode):
    """Enumerated active-cap Newton solve of convex KL + predictive/posterior risk.

    Both predictive and posterior matrices are positive semidefinite.
    Every accepted solution has a numerical primal/KKT certificate. Failed solves
    retain a logged capped-prior fallback. Sizes here are three/four real sources.
    """
    m=len(q)
    if not m:return np.empty(0),dict(kkt=0.,primal=0.,converged=True,iterations=0)
    cap=max(cfg['cap'],1/m);pi=q**cfg['quality_power'];pi=pi/pi.sum();fallback=capped_pi(pi,cap)
    lam=0. if mode in ('mean_only','frozen','periodic') else cfg['lam']
    eta=cfg['eta'] if mode in ('bayes_weights','bayes_both') else 0.
    if mode=='diagonal':S=np.diag(np.diag(S))
    risk=S+eta*U
    def value(w):return cfg['tau']*float(w@np.log(w/pi))+lam/2*float(w@risk@w)
    def gh(w):
        g=cfg['tau']*(np.log(w/pi)+1)+lam*risk@w
        h=np.diag(cfg['tau']/w)+lam*risk
        return g,h
    if m==1 or m*cap<=1+1e-12:return fallback,dict(kkt=0.,primal=0.,converged=True,iterations=0)
    if lam==0:return fallback,dict(kkt=0.,primal=0.,converged=True,iterations=0)
    best=None;iters=0
    for nupper in range(m):
        if nupper*cap>=1:break
        for fixed in itertools.combinations(range(m),nupper):
            free=np.array([i for i in range(m) if i not in fixed]);w=np.zeros(m);w[list(fixed)]=cap
            rem=1-nupper*cap;w[free]=rem*pi[free]/pi[free].sum()
            for it in range(60):
                iters+=1;g,h=gh(w);gf=g[free];hf=h[np.ix_(free,free)]
                if len(free)==1:break
                kkt=np.block([[hf,np.ones((len(free),1))],[np.ones((1,len(free))),np.zeros((1,1))]])
                try:d=np.linalg.solve(kkt,np.r_[-gf,0.])[:-1]
                except np.linalg.LinAlgError:break
                if np.max(np.abs(gf-gf.mean()))<1e-9:break
                step=1.;negative=d<0
                if negative.any():step=min(step,float(np.min(-.99*w[free][negative]/d[negative])))
                old=value(w);descent=float(gf@d)
                for _ in range(35):
                    cand=w.copy();cand[free]+=step*d
                    if np.all(cand>0) and value(cand)<=old+1e-4*step*descent:break
                    step*=.5
                if step<1e-12:break
                w=cand
            if (w>cap+1e-8).any():continue
            g,_=gh(w);common=float(np.mean(g[free]));r=float(np.max(np.abs(g[free]-common)))
            if fixed:r=max(r,float(np.max(np.maximum(g[list(fixed)]-common,0))))
            primal=max(abs(float(w.sum())-1),float(np.maximum(w-cap,0).max()),float(np.maximum(-w,0).max()))
            if best is None or r<best[0]:best=(r,w.copy(),primal)
            if r<1e-7 and primal<1e-9:return w,dict(kkt=r,primal=primal,converged=True,iterations=iters)
    return fallback,dict(kkt=float(best[0]) if best else 1e9,primal=0.,converged=False,iterations=iters)


def load_dataset(data_root,name,spec):
    import datetime as dt
    records=[];hashes={}
    for filename in spec['files']:
        path=data_root/name/filename;raw=path.read_bytes();hashes[filename]=hashlib.sha256(raw).hexdigest();reader=csv.reader(raw.decode('utf-8-sig').splitlines());header=next(reader)
        for values in reader:
            if len(values)==len(header)+1:values=values[1:]
            if len(values)!=len(header):raise ValueError('Malformed CSV')
            r=dict(zip(header,values));stamp=r.get('date',r.get('Date',''))+' '+r.get('Time','')
            stamp=stamp.strip();fmt='%Y-%m-%d %H:%M:%S' if '-' in stamp else '%Y/%m/%d %H:%M:%S'
            timestamp=dt.datetime.strptime(stamp,fmt).isoformat();x=np.array([float(r[f]) for f in spec['features']]);y=int(float(r[spec['target']]))
            if not np.isfinite(x).all():raise ValueError('Nonfinite raw sensor value')
            records.append((timestamp,x,y))
    records.sort(key=lambda r:r[0]);unique=[];duplicates=0
    for r in records:
        if unique and r[0]==unique[-1][0]:
            if not np.array_equal(r[1],unique[-1][1]) or r[2]!=unique[-1][2]:raise ValueError('Conflicting duplicate timestamps')
            duplicates+=1;continue
        unique.append(r)
    return dict(raw=np.stack([r[1] for r in unique]),y=np.array([r[2] for r in unique]),timestamp=[r[0] for r in unique],hashes=hashes,duplicates=duplicates,raw_rows=len(records),n=len(unique))


def lagged(raw,mean,scale):
    z=np.clip((raw-mean)/scale,-8,8);out=[]
    for i in range(len(z)):
        hist=np.stack([z[max(0,i-j)] for j in range(3)],axis=1).reshape(-1);out.append(np.r_[hist,1.])
    return np.asarray(out)


def context(x,groups):
    return np.array([np.mean(x[:,np.array(g)*3]) for g in groups])


def prefix(data,spec,cfg):
    ntrain=int(data['n']*.30);ncal=int(data['n']*.60);timestamps=data['timestamp'];audit=np.array([int(hashlib.sha256(t.encode()).hexdigest()[:8],16)%10<3 for t in timestamps])
    trainidx=np.flatnonzero(~audit[:ntrain]);auditidx=np.flatnonzero(audit[:ntrain]);mean=data['raw'][trainidx].mean(0);scale=np.maximum(data['raw'][trainidx].std(0),.05);x=lagged(data['raw'],mean,scale);y=data['y'];classes=spec['classes'];groups=spec['groups'];m=len(groups);d=data['raw'].shape[1]
    f=[np.r_[[j for v in g for j in range(3*v,3*v+3)],3*d] for g in groups]+[np.arange(3*d+1)]
    models=[gradient(np.zeros((len(ids),classes)),x[trainidx][:,ids],y[trainidx],cfg['initial_steps'],cfg,classes) for ids in f]
    pa=np.stack([softmax(x[auditidx][:,f[s]]@models[s]) for s in range(m)]);prior=[]
    for j in range(0,len(auditidx),cfg['window']):
        sel=auditidx[j:j+cfg['window']];prior.append(dict(x=x[sel],y=y[sel],p=pa[:,j:j+cfg['window']],context=context(x[sel],groups),mask=np.ones(m,bool),origin=-1))
    q=1/(np.mean(np.sum((pa-np.eye(classes)[y[auditidx]][None])**2,axis=2),axis=1)+.05)
    return dict(models=models,prior=prior,q=q,mean=mean,scale=scale,features=f,classes=classes,m=m,groups=groups,train_x=x[trainidx],train_y=y[trainidx],cal_x=x[ntrain:ncal],cal_y=y[ntrain:ncal],cal_timestamp=timestamps[ntrain:ncal],test_x=x[ncal:],test_y=y[ncal:],test_timestamp=timestamps[ncal:],split=dict(prefix_rows=ntrain,training_rows=len(trainidx),audit_rows=len(auditidx),calibration_rows=ncal-ntrain,test_rows=len(x)-ncal,train_start=timestamps[0],prefix_end=timestamps[ntrain-1],calibration_end=timestamps[ncal-1],test_start=timestamps[ncal],test_end=timestamps[-1]))


def world(x,y,timestamps,pre,seed,cfg):
    rng=np.random.default_rng(seed);m=pre['m'];models=[w.copy() for w in pre['models']];pending=[];tx=pre['train_x'].copy();ty=pre['train_y'].copy();events=[];cid=0;reference=pre['models'][-1].copy()
    for start in range(0,len(x),cfg['window']):
        k=len(events)
        for due,xx,yy in pending:
            if due<=k:tx=np.concatenate([tx,xx])[-cfg['train_buffer']:];ty=np.concatenate([ty,yy])[-cfg['train_buffer']:]
        pending=[p for p in pending if p[0]>k];probe=k%cfg['probe_period']==0
        if probe:
            models=[gradient(w,tx[:,ids],ty,cfg['probe_steps'],cfg,pre['classes']) for w,ids in zip(models,pre['features'])];cid+=1
        refresh=k>0 and k%8==0
        if refresh:reference=models[-1].copy()
        xx=x[start:start+cfg['window']];yy=y[start:start+cfg['window']];delay=int(rng.integers(cfg['delay_low'],cfg['delay_high']+1));mask=np.ones(m,bool)
        if k%cfg['source_dropout_period']<cfg['source_dropout_length']:mask[(k//cfg['source_dropout_period'])%m]=False
        ar=np.array([int(hashlib.sha256(t.encode()).hexdigest()[:8],16)%10<3 for t in timestamps[start:start+cfg['window']]])
        if (~ar).any():pending.append((k+delay,xx[~ar].copy(),yy[~ar].copy()))
        events.append(dict(x=xx,y=yy,context=context(xx,pre['groups']),mask=mask,audit=ar,probe=probe,refresh=refresh,delay=delay,origin=k,candidate=models[-1].copy(),reference=reference.copy(),cid=cid,p=np.stack([softmax(xx[:,pre['features'][s]]@models[s]) for s in range(m)])))
    return events


def psd(a):
    eig,v=np.linalg.eigh((a+a.T)/2);return (v*np.maximum(eig,0))@v.T


def moments(archive,pre,current,cfg):
    ids=np.flatnonzero(current['mask']);chunks=[]
    for r in pre['prior']:chunks.append((cfg['prior_mass']/len(pre['prior']),r))
    for r in archive:
        if not r['mask'][ids].all():continue
        a=cfg['beta']**(current['origin']-r['origin'])*math.exp(-float(np.sum((current['context']-r['context'])**2))/cfg['kernel_bandwidth']**2);chunks.append((max(a,1e-12),r))
    means=[];seconds=[];alphas=[]
    for a,r in chunks:
        active=(r['x']@current['reference']).argmax(1);candidate=(r['x']@current['candidate']).argmax(1);b=np.eye(pre['classes'])[candidate]-np.eye(pre['classes'])[active];e=r['p'][ids]-np.eye(pre['classes'])[r['y']][None];z=np.einsum('sic,ic->si',e,b)
        means.append(z.mean(1));seconds.append(z@z.T/len(r['y']));alphas.append(a)
    alpha=np.asarray(alphas);mass=float(alpha.sum());p=alpha/mass;means=np.asarray(means);mu=p@means;Q=np.einsum('g,gij->ij',p,np.asarray(seconds));B=psd(np.einsum('g,gi,gj->ij',p,means,means)-np.outer(mu,mu));Sigma=psd(Q-np.outer(mu,mu));U=B/(mass+1)
    T=np.eye(len(ids))-np.ones((len(ids),len(ids)))/len(ids);ttrace=float(np.trace(T@Sigma@T));scale=ttrace if ttrace>=1e-6 else 1.
    return dict(S=Sigma/scale,Us=U/scale,mu=mu,U=U,B=B,mass=mass,blocks=len(chunks),scale=scale,ttrace=ttrace,identity_error=float(np.max(np.abs(Sigma-(Sigma-U+U)))))


def precompute(events,pre,cfg):
    pending=[];archive=[];decisions=[];basegross=0.;commonfees=0.;commondrops=0;basen=0;basecorrect=0
    for k,e in enumerate(events):
        for due,r in pending:
            if due<=k:archive.append(r)
        archive=archive[-cfg['archive']:];pending=[p for p in pending if p[0]>k]
        common_drop=(cfg['probe_drop'] if e['probe'] else 0)+(2 if e['refresh'] else 0);active=(e['x']@e['reference']).argmax(1);keep=np.arange(len(active))>=common_drop
        basegross+=float(np.sum(active[keep]==e['y'][keep]));commonfees+=(pre['m']+1)*cfg['probe_steps']*cfg['step_fee'] if e['probe'] else 0.;commonfees+=2. if e['refresh'] else 0.;commondrops+=min(len(active),common_drop);basecorrect+=int(np.sum(active==e['y']));basen+=len(active)
        if k%cfg['horizon']==0 and k+cfg['horizon']<=len(events):
            mm=moments(archive,pre,e,cfg);cand=(e['x']@e['candidate']).argmax(1);b=np.eye(pre['classes'])[cand]-np.eye(pre['classes'])[active];ids=np.flatnonzero(e['mask']);h=np.einsum('sic,ic->s',e['p'][ids],b)/len(active);N=cfg['horizon']*len(active)
            truegross=0.;grossdifference=0.;accdifference=0.;future=[];maturity=0
            for j in range(k,k+cfg['horizon']):
                f=events[j];pa=(f['x']@e['reference']).argmax(1);pc=(f['x']@e['candidate']).argmax(1);cd=(cfg['probe_drop'] if f['probe'] else 0)+(2 if f['refresh'] else 0);keepj=np.arange(len(pa))>=cd;g=float(np.sum((pc[keepj]==f['y'][keepj]).astype(float)-(pa[keepj]==f['y'][keepj]).astype(float)));truegross+=g;accdifference+=int(np.sum(pc==f['y'])-np.sum(pa==f['y']));future.append(float(np.mean(pc==f['y'])-np.mean(pa==f['y'])));maturity=max(maturity,j+f['delay'])
            candidateextra=float(np.sum(cand[common_drop:common_drop+cfg['deploy_drop']]==e['y'][common_drop:common_drop+cfg['deploy_drop']]))
            grossdifference=truegross-candidateextra;localnet=grossdifference-3.
            decisions.append(dict(k=k,S=mm['S'],Us=mm['Us'],U=mm['U'],B=mm['B'],mu=mm['mu'],h=h,N=N,ids=ids,truegross=truegross,localnet=localnet,grossdifference=grossdifference,accdifference=accdifference,maturity=maturity,future=future,disagreement=float(np.mean(active!=cand)),mass=mm['mass'],blocks=mm['blocks'],scale=mm['scale'],identity_error=mm['identity_error']))
        ar=e['audit']
        if ar.any():pending.append((k+e['delay'],dict(x=e['x'][ar].copy(),y=e['y'][ar].copy(),p=e['p'][:,ar].copy(),context=e['context'].copy(),mask=e['mask'].copy(),origin=k)))
    return dict(decisions=decisions,basegross=basegross,commonfees=commonfees,commondrops=commondrops,basecorrect=basecorrect,n=basen,windows=len(events))


def decision_forecast(d,pre,cfg,mode):
    w,cert=convex_fuse(pre['q'][d['ids']],d['S'],d['Us'],cfg,mode);gain=float(d['N']*w@(d['h']-d['mu']));variance=d['B'] if mode=='frequentist_gate' else d['U'];sd=d['N']*math.sqrt(max(0.,float(w@variance@w))+1e-4);standardized=(gain-d['truegross'])/sd
    return w,cert,gain,sd,standardized


def initial_q(streams,pre,cfg,mode):
    scores=[]
    for stream in streams:
        boundary=stream['windows']//2
        for d in stream['decisions']:
            if d['maturity']<=boundary:
                scores.append(decision_forecast(d,pre,cfg,mode)[4])
    return max(0.,float(np.quantile(scores,.9,method='higher'))) if scores else 0.,scores


def run(stream,pre,cfg,mode,qinit,selection=False):
    q=float(qinit);pending=[];qlog=[];rows=[];actions=[];net=stream['basegross']-stream['commonfees'];gross=stream['basegross'];fees=stream['commonfees'];drops=stream['commondrops'];correct=stream['basecorrect'];maxkkt=0.;failures=0;boundary=stream['windows']//2 if selection else 0;select_net=0.;clip_sum=0.;violations=0;updates=0
    for d in stream['decisions']:
        k=d['k']
        for due,issuedk,issuedq,score,issuedstd in sorted(pending,key=lambda p:p[0]):
            if due<=k:
                violation=int(score>issuedq);proposal=q+.05*(violation-.1);newq=max(0.,proposal);regulator=newq-proposal;clip_sum+=regulator;violations+=violation;updates+=1;qlog.append(dict(maturity=due,issued_index=issuedk,update_index=k,issued_q=issuedq,issued_std=issuedstd,standardized_score=score,violation=violation,q_before=q,q_after=newq,regulator=regulator));q=newq
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
            violation=int(score>issuedq);proposal=q+.05*(violation-.1);newq=max(0.,proposal);regulator=newq-proposal;clip_sum+=regulator;violations+=violation;updates+=1;qlog.append(dict(maturity=due,issued_index=issuedk,update_index=stream['windows'],issued_q=issuedq,issued_std=issuedstd,standardized_score=score,violation=violation,q_before=q,q_after=newq,regulator=regulator));q=newq
    identity=violations-(.1*updates+(q-qinit-clip_sum)/.05);acts=[r for r in rows if r['action']];selected=[r for r in rows if r['k']>=boundary];informative=[r for r in selected if r['disagreement']>0]
    return dict(net=net,gross=gross,fees=fees,drops=drops,accuracy=correct/stream['n'],deployments=sum(actions),actions=actions,rows=rows,q_updates=qlog,q_initial=qinit,q_final=q,clip_regulator_sum=clip_sum,calibration_updates=updates,miscoverage=violations/updates if updates else None,calibration_identity_error=identity,coverage=float(np.mean([r['lower_covered'] for r in selected])) if selected else None,informative_coverage=float(np.mean([r['lower_covered'] for r in informative])) if informative else None,informative_episodes=len(informative),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),zero=sum(r['local_net']==0 for r in acts),solver_failures=int(failures),max_kkt=maxkkt,selection_net=select_net,closed_loop_identity_error=net-(stream['basegross']-stream['commonfees']+sum(r['local_net'] for r in acts)))


def algebra_audit():
    rng=np.random.default_rng(51517);errors=[]
    for m in (3,4,5):
        a=rng.normal(size=(m,m));S=a@a.T;b=rng.normal(size=(m,m));U=b@b.T/20;q=rng.uniform(.5,2,m);cfg=dict(BASE,lam=0)
        ws=[convex_fuse(q,S,U,cfg,mode)[0] for mode in ('mean_only','joint','bayes_weights','bayes_both')];zero=max(float(np.max(np.abs(w-ws[0]))) for w in ws);cfg=dict(BASE,eta=0);wj,cj=convex_fuse(q,S,U,cfg,'joint');wb,cb=convex_fuse(q,S,U,cfg,'bayes_weights');eta0=float(np.max(np.abs(wj-wb)));assert zero<1e-12 and eta0<1e-12;assert cj['converged'] and cb['converged'];errors.append(dict(m=m,lambda0_weight_error=zero,eta0_weight_error=eta0,joint_kkt=cj['kkt']))
    return dict(passed=True,rows=errors)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'independent_data');parser.add_argument('--output',type=Path,default=ROOT/'new_bayes_fusion');parser.add_argument('--audit-only',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    audit=algebra_audit();(args.output/'algebra_audit.json').write_text(json.dumps(audit,indent=2))
    if args.audit_only:print(json.dumps(audit),flush=True);return
    protocol_path=args.output/'protocol.json';protocol=json.loads(protocol_path.read_text());sha=hashlib.sha256(protocol_path.read_bytes()).hexdigest();code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(args.output/'pretest_freeze.json').write_text(json.dumps(dict(protocol_sha=sha,code_sha=code_sha,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_metrics_read=False),indent=2));print('PRETEST_FREEZE',sha,code_sha,flush=True)
    summaries={};all_results={}
    for name,spec in protocol['datasets'].items():
        data=load_dataset(args.data,name,spec);pre=prefix(data,spec,BASE);calstreams=[precompute(world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,seed,BASE),pre,BASE) for seed in protocol['randomness']['calibration_seeds']];cal=[];selected={};qfit={}
        for lam in (0.,.25,1.):
            for margin in (0.,4.,12.):
                cfg=dict(BASE,lam=lam,threshold=margin)
                for mode in MODES[:7]:
                    qi,scores=initial_q(calstreams,pre,cfg,mode);rr=[run(s,pre,cfg,mode,qi,True) for s in calstreams];cal.append(dict(mode=mode,lam=lam,margin=margin,q_initial=qi,q_fit_scores=scores,mean_selection_net=float(np.mean([r['selection_net'] for r in rr])),mean_selection_actions=float(np.mean([r['deployments'] for r in rr]))))
                print('CAL',name,lam,margin,flush=True)
        for mode in MODES[:7]:
            choice=max([r for r in cal if r['mode']==mode],key=lambda r:(r['mean_selection_net'],-r['margin'],-r['lam']));selected[mode]=dict(BASE,lam=choice['lam'],threshold=choice['margin']);qfit[mode]=choice['q_initial']
        selected['posterior_unused']=dict(selected['joint']);qfit['posterior_unused']=qfit['joint'];selection_path=args.output/f'{name}_prefix_selection.json';selection_path.write_text(json.dumps(dict(protocol_sha=sha,code_sha=code_sha,data_hashes=data['hashes'],split=pre['split'],grid=cal,selected=selected,q_initial=qfit),indent=2));selection_sha=hashlib.sha256(selection_path.read_bytes()).hexdigest();print('SELECTION_FROZEN',name,selection_sha,{m:(c['lam'],c['threshold'],qfit[m]) for m,c in selected.items()},flush=True)
        trials=[]
        for seed in protocol['randomness']['test_seeds']:
            stream=precompute(world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,BASE),pre,BASE);rr={mode:run(stream,pre,selected.get(mode,BASE),mode,qfit.get(mode,0)) for mode in MODES};trials.append(dict(seed=seed,windows=stream['windows'],episodes=len(stream['decisions']),baseline=dict(gross=stream['basegross'],fees=stream['commonfees'],drops=stream['commondrops'],net=stream['basegross']-stream['commonfees']),results=rr));print('TEST',name,seed,{m:(round(r['net'],2),r['deployments'],r['harmful']) for m,r in rr.items()},flush=True);(args.output/f'{name}_results_partial.json').write_text(json.dumps(dict(trials=trials),indent=2))
        summary={mode:dict(mean_net=float(np.mean([t['results'][mode]['net'] for t in trials])),mean_deployments=float(np.mean([t['results'][mode]['deployments'] for t in trials])),mean_harmful=float(np.mean([t['results'][mode]['harmful'] for t in trials])),mean_beneficial=float(np.mean([t['results'][mode]['beneficial'] for t in trials])),mean_accuracy=float(np.mean([t['results'][mode]['accuracy'] for t in trials])),mean_coverage=float(np.mean([t['results'][mode]['coverage'] for t in trials])),mean_informative_coverage=float(np.mean([t['results'][mode]['informative_coverage'] for t in trials if t['results'][mode]['informative_coverage'] is not None])),solver_failures=sum(t['results'][mode]['solver_failures'] for t in trials),max_kkt=max(t['results'][mode]['max_kkt'] for t in trials)) for mode in MODES}
        result=dict(dataset=name,protocol_sha=sha,code_sha=code_sha,selection_sha=selection_sha,data_hashes=data['hashes'],raw_rows=data['raw_rows'],unique_rows=data['n'],duplicates=data['duplicates'],split=pre['split'],quality=pre['q'].tolist(),selected=selected,q_initial=qfit,calibration=cal,trials=trials,summary=summary);(args.output/f'{name}_results.json').write_text(json.dumps(result,indent=2));summaries[name]=summary;all_results[name]=result;print('DATASET_COMPLETE',name,summary,flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2));(args.output/'results.json').write_text(json.dumps(all_results,indent=2));print('COMPLETE',flush=True)

MODES=('mean_only','diagonal','joint','bayes_weights','bayes_gate','bayes_both','frequentist_gate','posterior_unused','periodic','frozen')
if __name__=='__main__':main()
