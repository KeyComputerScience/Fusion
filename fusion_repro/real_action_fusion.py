"""Decision-projected fusion on the real UCI Gas Sensor Array Drift benchmark.

Raw sensor readings and gas labels are real. Windows, feedback delay, outages,
service prices, SGD costs, and deployment costs are declared replay interventions.
All policies pay exactly the same actual five-model batch-gradient training. Their only
controlled action is paid admission of the shared candidate model.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'uci_gas'/'Dataset'
sys.path.insert(0,str(ROOT))
from da_rf_fusion import solve_capped_fusion
MODES=('decision_full','context_joint','decision_diagonal','quality_only',
       'rf_c','frozen_moment','no_context','no_delay_replay','periodic','frozen_deploy')
BASE=dict(window=32, probe_period=8, probe_steps=16, train_buffer=512,
          learning_rate=.035, l2=.002, beta=.97, archive=64, prior_mass=2.,
          context_bandwidth=3., tau=.05, lam=1., deploy_fee=12.,
          deploy_drop=4, probe_drop=2, step_fee=.02, horizon=4,
          cooldown=2, budget_per_batch=8, delay_low=2, delay_high=5,
          threshold=0., mask_stress=False, quality_power=1.)

def softmax(z):
    z=z-z.max(axis=-1,keepdims=True); e=np.exp(z);return e/e.sum(axis=-1,keepdims=True)

def load():
    batches={}; hashes={}
    for b in range(1,11):
        p=DATA/f'batch{b}.dat'; raw=p.read_bytes();hashes[str(b)]=hashlib.sha256(raw).hexdigest()
        xx=[];yy=[]
        for line in raw.decode().splitlines():
            fields=line.split();v=np.zeros(128)
            for field in fields[1:]:
                i,x=field.split(':');v[int(i)-1]=float(x)
            xx.append(v);yy.append(int(fields[0])-1)
        batches[b]=(np.asarray(xx),np.asarray(yy))
    mean=batches[1][0].mean(0);scale=np.maximum(batches[1][0].std(0),.05)
    for b,(x,y) in batches.items():
        batches[b]=(np.column_stack([np.clip((x-mean)/scale,-8,8),np.ones(len(x))]),y)
    return batches,hashes

def features(s):
    return np.r_[np.arange(32*s,32*(s+1)),128] if s<4 else np.arange(129)

def gradient(w,x,y,steps,cfg):
    target=np.eye(6)[y]
    for _ in range(steps):
        p=softmax(x@w);g=x.T@(p-target)/len(x)+cfg['l2']*w
        w=w-cfg['learning_rate']*g
    return w

def capped_pi(v,cap=.8):
    if len(v)==0:return np.empty(0)
    cap=max(cap,1/len(v))
    v=np.maximum(v,1e-14);out=np.zeros(len(v));free=np.ones(len(v),bool);rem=1.
    for _ in range(len(v)):
        z=rem*v[free]/v[free].sum();idx=np.flatnonzero(free);over=z>cap
        if not over.any():out[idx]=z;break
        out[idx[over]]=cap;free[idx[over]]=False;rem-=cap*over.sum()
    return out

def capped_log_pi(log_rates,cap=.8):
    """Exact capped exponential normalization without a global underflow floor."""
    if len(log_rates)==0:return np.empty(0)
    cap=max(cap,1/len(log_rates))
    out=np.zeros(len(log_rates));free=np.ones(len(out),bool);rem=1.
    for _ in range(len(out)):
        idx=np.flatnonzero(free);log=log_rates[free];rates=np.exp(np.maximum(log-log.max(),-690.));z=rem*rates/rates.sum();over=z>cap
        if not over.any():out[idx]=z;break
        out[idx[over]]=cap;free[idx[over]]=False;rem-=cap*over.sum()
    return out

def fuse_weights(q,s,cfg,mode):
    if len(q)==0:
        return np.empty(0),dict(converged=True,kkt_residual=0.,primal_residual=0.,closed_form=True,abstention=True,cap=None)
    cap=max(.8,1/len(q))
    q=np.maximum(q,1e-9)**cfg['quality_power'];pi=capped_pi(q)
    if mode=='rf_c':
        logits=np.log(q)-np.diag(s)/.15
        return capped_log_pi(logits),dict(converged=True,kkt_residual=0.,primal_residual=0.,closed_form=True)
    lam=0. if mode=='quality_only' else cfg['lam']
    result=solve_capped_fusion(q,s,pi,cap,cfg['tau'],lam,0.,tolerance=1e-9,max_iterations=100)
    return result.weights,dict(converged=bool(result.converged),kkt_residual=float(result.kkt_residual),primal_residual=float(result.primal_residual),closed_form=False)

def context(x):
    # Four physical-group magnitude summaries use only current covariates.
    return np.log1p(np.mean(np.abs(x[:,:128]).reshape(len(x),4,32),axis=(0,2)))

def tensor(p,y):
    e=p-np.eye(6)[y][None,:,:]
    return np.einsum('sic,tid->stcd',e,e,optimize=True)/len(y)

def moments(records,prior,c,k,cfg,mask,mode):
    if mode=='frozen_moment':return prior[np.ix_(mask,mask)]
    numerator=cfg['prior_mass']*prior;den=cfg['prior_mass'];mass=[]
    for r in records[-cfg['archive']:]:
        if not np.all(r['mask'][mask]):continue
        age=k-r['origin'] if mode!='no_delay_replay' else k-r['arrival']
        kern=1. if mode=='no_context' else math.exp(-np.sum((c-r['context'])**2)/cfg['context_bandwidth']**2)
        a=cfg['beta']**max(age,0)*kern
        numerator=numerator+a*r['tensor'];den+=a;mass.append(a)
    out=numerator/den
    return out[np.ix_(mask,mask)]

def make_prefix(batches,cfg):
    weights=[]
    x,y=batches[1]
    for s in range(5):
        f=features(s);w=np.zeros((len(f),6))
        w=gradient(w,x[:,f],y,500,cfg);weights.append(w)
    x2,y2=batches[2]
    ps=np.stack([softmax(x2[:,features(s)]@weights[s]) for s in range(4)])
    prior=tensor(ps,y2)
    q=1/(np.array([np.mean(np.sum((ps[s]-np.eye(6)[y2])**2,axis=1)) for s in range(4)])+.05)
    return weights,prior,q

def world(batches,prefix,batch_ids,seed,cfg):
    rng=np.random.default_rng(seed);events=[];global_k=0
    for b in batch_ids:
        x,y=batches[b];ix=rng.permutation(len(x))
        for start in range(0,len(ix),cfg['window']):
            ids=ix[start:start+cfg['window']];cx=x[ids];cy=y[ids]
            c=context(cx);price=np.full(len(ids),1.+2.*(c[0]>1.))
            mask=np.ones(4,bool)
            if cfg['mask_stress'] and global_k%13 in (4,5,6):mask[global_k//13%4]=False
            events.append(dict(batch=b,x=cx,y=cy,context=c,price=price,mask=mask,
                               delay=int(rng.integers(cfg['delay_low'],cfg['delay_high']+1))))
            global_k+=1
    models=[v.copy() for v in prefix];arrived_x=[batches[1][0],batches[2][0]];arrived_y=[batches[1][1],batches[2][1]]
    pending=[];candidate_id=0;arrived_origins=[-1]
    for k,e in enumerate(events):
        for due,r in pending:
            if due<=k:arrived_x.append(r['x']);arrived_y.append(r['y']);arrived_origins.append(r['origin'])
        pending=[(due,r) for due,r in pending if due>k]
        probe=k%cfg['probe_period']==0
        if probe:
            tx=np.concatenate(arrived_x)[-cfg['train_buffer']:];ty=np.concatenate(arrived_y)[-cfg['train_buffer']:]
            models=[gradient(w,tx[:,features(s)],ty,cfg['probe_steps'],cfg) for s,w in enumerate(models)]
            candidate_id+=1
        e['source']=np.stack([softmax(e['x'][:,features(s)]@models[s]) for s in range(4)])
        e['candidate']=models[4].copy();e['candidate_id']=candidate_id;e['probe']=probe
        e['probe_max_training_origin']=max(arrived_origins) if probe else None
        e['origin']=k;pending.append((k+max(1,e['delay']),e))
    return events

def run(events,prefix,prior,q,cfg,mode):
    deployed=prefix[4].copy();pending_labels=[];archive=[];pending_deploy=None
    used=set();actions=[];weights=[];scores=[];batches=[];correct=[];served=[];labels=[];returns=[]
    candidate_preds=[];active_preds=[];fees=0.;probe_fees=0.;deploy_fees=0.;drops=0;probe_steps=0
    cooldown_until=-1;budget={};matrix_min=1.;matrix_change=[];causal_errors=0
    local_benefits=[];decision_records=[];solver_failures=0;max_kkt=0.;max_primal=0.
    for k,e in enumerate(events):
        for r in pending_labels:
            if r['arrival']<=k:
                archive.append(dict(origin=r['origin'],arrival=r['arrival'],mask=r['mask'],context=r['context'],tensor=tensor(r['saved_source'],r['revealed_y'])))
        archive=archive[-cfg['archive']:]
        pending_labels=[r for r in pending_labels if r['arrival']>k]
        if pending_deploy is not None and pending_deploy[0]==k:
            deployed=pending_deploy[1];pending_deploy=None
        active=(e['x']@deployed).argmax(1);cand=(e['x']@e['candidate']).argmax(1)
        active_preds.extend(active.tolist());candidate_preds.extend(cand.tolist())
        b=e['price'][:,None]*(np.eye(6)[cand]-np.eye(6)[active])
        d=b.T@b/len(b);mask=np.flatnonzero(e['mask']);pp=e['source'][mask]
        t=moments(archive,prior,e['context'],k,cfg,mask,mode)
        r=np.einsum('stcc->st',t);m=np.einsum('stcd,cd->st',t,d)
        if np.trace(m)>1e-12:m*=np.trace(r)/np.trace(m)
        else:m=r.copy()
        if mode in ('context_joint','rf_c'):s=r
        elif mode=='decision_diagonal':s=np.diag(np.diag(m))
        else:s=m
        s=(s+s.T)/2
        if s.size:matrix_min=min(matrix_min,float(np.linalg.eigvalsh(s).min()))
        matrix_change.append(float(np.linalg.norm(r-np.einsum('stcc->st',prior[np.ix_(mask,mask)]))))
        w,certificate=fuse_weights(q[mask],s,cfg,mode)
        solver_failures+=int(not certificate['converged']);max_kkt=max(max_kkt,certificate['kkt_residual']);max_primal=max(max_primal,certificate['primal_residual'])
        fused=np.einsum('s,sic->ic',w,pp)
        estimated=float(np.sum(b*fused)*cfg['horizon'])
        deployment_cost=cfg['deploy_fee']+cfg['deploy_drop']*float(e['price'].mean())
        eligible=(mask.size>0 and e['candidate_id'] not in used and k>=cooldown_until and budget.get(e['batch'],0)<cfg['budget_per_batch'] and k+1<len(events))
        if mode=='frozen_deploy':launch=False
        elif mode=='periodic':launch=eligible and e['probe']
        else:launch=eligible and estimated>deployment_cost+cfg['threshold']
        shadow={}
        for alt in ('context_joint','decision_diagonal','quality_only','rf_c'):
            alt_matrix=np.diag(np.diag(m)) if alt=='decision_diagonal' else r
            alt_w,alt_certificate=fuse_weights(q[mask],alt_matrix,cfg,alt)
            alt_p=np.einsum('s,sic->ic',alt_w,pp)
            alt_score=float(np.sum(b*alt_p)*cfg['horizon'])-deployment_cost
            shadow[alt]=dict(score=alt_score,action=bool(eligible and alt_score>cfg['threshold']),weight_l2=float(np.linalg.norm(w-alt_w)))
        if launch:
            pending_deploy=(k+1,e['candidate'].copy());used.add(e['candidate_id']);cooldown_until=k+cfg['cooldown'];budget[e['batch']]=budget.get(e['batch'],0)+1
            deploy_fees+=cfg['deploy_fee'];fees+=cfg['deploy_fee']
        # Service at k uses the model already deployed at its beginning. Launch
        # occupancy is charged at k once; activation is one window later.
        drop=(cfg['probe_drop'] if e['probe'] else 0)+(cfg['deploy_drop'] if launch else 0)
        drop=min(drop,len(active));keep=np.arange(len(active))>=drop
        gross=float(np.sum(e['price'][keep]*(active[keep]==e['y'][keep])))
        probe_cost=0.
        if e['probe']:
            step_count=5*cfg['probe_steps'];probe_steps+=step_count;probe_cost=step_count*cfg['step_fee'];probe_fees+=probe_cost;fees+=probe_cost
        net=gross-probe_cost-(cfg['deploy_fee'] if launch else 0.)
        returns.append(net);drops+=drop;correct.extend((active==e['y']).tolist());served.extend(keep.tolist());labels.extend(e['y'].tolist());batches.extend([e['batch']]*len(active))
        actions.append(bool(launch));weights.append(dict(origin=k,active=mask.tolist(),value=w.tolist()));scores.append(estimated-deployment_cost)
        pending_labels.append(dict(origin=k,arrival=k+max(1,e['delay']),mask=e['mask'].copy(),context=e['context'],saved_source=e['source'],revealed_y=e['y']))
        if any(v['arrival']>k for v in archive):causal_errors+=1
        # Evaluation-only local causal fork: same deployed state, same future
        # candidate, no additional subsequent admissions in either fork.
        local= -cfg['deploy_fee']
        for j in range(k+1,min(len(events),k+1+cfg['horizon'])):
            f=events[j];pa=(f['x']@deployed).argmax(1);pc=(f['x']@e['candidate']).argmax(1)
            future_keep=np.arange(len(pa))>=(cfg['probe_drop'] if f['probe'] else 0)
            local+=float(np.sum(f['price'][future_keep]*((pc[future_keep]==f['y'][future_keep]).astype(float)-(pa[future_keep]==f['y'][future_keep]).astype(float))))
        first=cfg['probe_drop'] if e['probe'] else 0;last=first+cfg['deploy_drop']
        local-=float(np.sum(e['price'][first:last]*(active[first:last]==e['y'][first:last])))
        local_benefits.append(local)
        decision_records.append(dict(origin=k,candidate_id=e['candidate_id'],eligible=eligible,
            action=bool(launch),gain_score=estimated-deployment_cost,local_deploy_advantage=local,
            gain_available=bool(mask.size),
            prediction_disagreement=float(np.mean(active!=cand)),same_state_shadow=shadow))
    correct=np.asarray(correct);labels=np.asarray(labels);served=np.asarray(served);batches=np.asarray(batches)
    batch_scores={}
    for batch in sorted(set(batches.tolist())):
        sel=batches==batch;recalls=[float(correct[sel&(labels==c)].mean()) for c in range(6) if np.any(sel&(labels==c))]
        batch_scores[str(batch)]=dict(n=int(sel.sum()),classes_present=int(len(recalls)),potential_accuracy=float(correct[sel].mean()),balanced_accuracy=float(np.mean(recalls)))
    # Persistence is measured against a read-only initially deployed model and
    # only after an admission; all labels here are evaluation-only.
    base_acc=[];active_acc=[];ptr=0
    for e in events:
        n=len(e['y']);base=(e['x']@prefix[4]).argmax(1);base_acc.append(float(np.mean(base==e['y'])));active_acc.append(float(correct[ptr:ptr+n].mean()));ptr+=n
    recovery=[]
    for k,a in enumerate(actions):
        if a:
            hits=[j for j in range(k+1,len(events)-2) if all(active_acc[h]>=base_acc[h]+.03 for h in range(j,j+3))]
            first=hits[0] if hits else None;recovery.append(dict(launch=k,first_three_window_improvement=first,latency=None if first is None else first-k))
    return dict(net_return=float(sum(returns)),potential_accuracy=float(correct.mean()),served_correct=int(np.sum(correct&served)),n=int(len(labels)),deployments=int(sum(actions)),probe_steps=probe_steps,probe_fees=probe_fees,deploy_fees=deploy_fees,total_fees=fees,dropped_service=drops,actions=actions,weights=weights,scores=scores,local_deploy_advantages=local_benefits,decision_records=decision_records,batch_scores=batch_scores,recovery=recovery,matrix_min_eigenvalue=matrix_min,mean_matrix_change=float(np.mean(matrix_change)),causal_errors=causal_errors,solver_failures=solver_failures,max_solver_kkt=max_kkt,max_solver_primal=max_primal)

def paired(a,b):
    d=np.asarray(a)-np.asarray(b);mean=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else 0
    return dict(mean=mean,ci95=[mean-1.96*se,mean+1.96*se],positive=int(np.sum(d>1e-9)),negative=int(np.sum(d< -1e-9)),ties=int(np.sum(abs(d)<=1e-9)))

def main():
    global DATA
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--evaluate',action='store_true');parser.add_argument('--data',type=Path,default=DATA,help='Directory containing official batch1.dat through batch10.dat');parser.add_argument('--output',default=str(ROOT/'real_action_results.json'));args=parser.parse_args();DATA=args.data
    batches,hashes=load();cfg=BASE.copy();prefix,prior,q=make_prefix(batches,cfg)
    pilot=[]
    # Every adaptive comparator receives identical calibration grid budget.
    for lam in (.25,1.,4.):
        for threshold in (0.,8.,24.):
            cc=dict(cfg,lam=lam,threshold=threshold);worlds=[world(batches,prefix,[3],s,cc) for s in (96001,96002,96003)]
            for mode in MODES[:5]:
                rr=[run(w,prefix,prior,q,cc,mode) for w in worlds]
                pilot.append(dict(lam=lam,threshold=threshold,mode=mode,mean_return=float(np.mean([r['net_return'] for r in rr])),mean_actions=float(np.mean([r['deployments'] for r in rr]))))
    best={mode:max([r for r in pilot if r['mode']==mode],key=lambda x:(x['mean_return'],-x['lam'],-x['threshold'])) for mode in MODES[:5]}
    chosen=best['decision_full'];cfg.update(lam=chosen['lam'],threshold=chosen['threshold'])
    result=dict(dataset='UCI Gas Sensor Array Drift, dataset 224',data_hashes=hashes,split=dict(train=[1],prior=[2],calibrate=[3],test=list(range(4,11))),config=cfg,quality=q.tolist(),quality_definition='inverse B2 labeled prefix source Brier risk plus .05; outcome reliability prior, not physical measurement quality',effective_delay='max(1,requested_delay) windows, label unavailable before its forecast',calibration_grid=pilot,calibration_best=best,seeds=list(range(97001,97011)),limitations=['real sensor values and labels; artificial replay windows, delays, masks, prices and costs','within-batch permutations are not independent datasets or genuine timestamp order','all five probe models trained and paid identically; policy controls candidate admission, not retraining','no HMM increment claimed in this extension','normal paired intervals conditional on fixed shared dataset and fit'])
    result['scalar_baseline_semantics']='RF-C uses diagonal uncentered source-class Brier second moment; aligned component control, not full untouched historical RF.'
    (ROOT/'real_action_pilot.json').write_text(json.dumps(result,indent=2));print('CALIBRATION_FROZEN',json.dumps(dict(config=cfg,best=best)),flush=True)
    if args.pilot and not args.evaluate:return
    per_seed=[]
    for seed in result['seeds']:
        ww=world(batches,prefix,list(range(4,11)),seed,cfg);rr={mode:run(ww,prefix,prior,q,cfg,mode) for mode in MODES}
        # Separately prefix-tuned strong comparators share the same grid budget.
        for mode in ('context_joint','decision_diagonal','quality_only','rf_c'):
            cc=dict(cfg,lam=best[mode]['lam'],threshold=best[mode]['threshold']);rr[mode+'_prefix_tuned']=run(ww,prefix,prior,q,cc,mode)
        diag={}
        full=rr['decision_full']
        for mode,r in rr.items():
            diff=np.asarray(full['actions'])!=np.asarray(r['actions']);local=np.asarray(full['local_deploy_advantages'])
            chosen=np.asarray(full['actions']);signed=np.where(chosen,local,-local)
            diag[mode]=dict(action_disagreements=int(diff.sum()),full_state_fork_positive_on_actual_diff=int(np.sum(signed[diff]>0)),full_state_fork_negative_on_actual_diff=int(np.sum(signed[diff]<0)),full_state_fork_sum_on_actual_diff=float(signed[diff].sum()))
        shadow_audit={}
        for alt in ('context_joint','decision_diagonal','quality_only','rf_c'):
            crossing=np.array([v['action']!=v['same_state_shadow'][alt]['action'] for v in full['decision_records']])
            signed=np.array([v['local_deploy_advantage']*(1 if v['action'] else -1) for v in full['decision_records']])
            shadow_audit[alt]=dict(crossings=int(crossing.sum()),beneficial=int(np.sum(signed[crossing]>0)),harmful=int(np.sum(signed[crossing]<0)),local_sum=float(signed[crossing].sum()),mean_weight_l2=float(np.mean([v['same_state_shadow'][alt]['weight_l2'] for v in full['decision_records']])))
        per_seed.append(dict(seed=seed,results=rr,diagnostics=diag,same_state_shadow_audit=shadow_audit));print('SEED',seed,{m:(round(r['net_return'],2),r['deployments']) for m,r in rr.items()},flush=True)
    result['per_seed']=per_seed;keys=list(per_seed[0]['results']);result['summary']={mode:dict(mean_return=float(np.mean([v['results'][mode]['net_return'] for v in per_seed])),mean_deployments=float(np.mean([v['results'][mode]['deployments'] for v in per_seed])),mean_accuracy=float(np.mean([v['results'][mode]['potential_accuracy'] for v in per_seed])),mean_action_disagreements=float(np.mean([v['diagnostics'][mode]['action_disagreements'] for v in per_seed])),paired_vs_full=paired([v['results']['decision_full']['net_return'] for v in per_seed],[v['results'][mode]['net_return'] for v in per_seed])) for mode in keys}
    Path(args.output).write_text(json.dumps(result,indent=2));print('MAIN_RESULTS_SAVED',args.output,json.dumps(result['summary']),flush=True)
    # Prespecified, paired sensitivities keep sources/SGD probes unchanged.
    settings=[('delay_0',dict(delay_low=0,delay_high=0)),('delay_8',dict(delay_low=8,delay_high=8)),('beta_90',dict(beta=.9)),('beta_1',dict(beta=1.)),('lambda_0',dict(lam=0.)),('lambda_4',dict(lam=4.)),('quality_flat',dict(quality_power=0.)),('fee_4',dict(deploy_fee=4.)),('fee_24',dict(deploy_fee=24.)),('mask_outage',dict(mask_stress=True))]
    sensitivity=[]
    for name,change in settings:
        cc=dict(cfg,**change);rows=[]
        for seed in result['seeds'][:5]:
            ww=world(batches,prefix,list(range(4,11)),seed,cc);a=run(ww,prefix,prior,q,cc,'decision_full');b=run(ww,prefix,prior,q,cc,'context_joint');base=next(v['results']['decision_full'] for v in per_seed if v['seed']==seed)
            rows.append(dict(seed=seed,full_return=a['net_return'],joint_return=b['net_return'],deployments=a['deployments'],actions_changed_vs_base=int(np.sum(np.asarray(a['actions'])!=np.asarray(base['actions']))),weight_l2_vs_base=float(np.mean([np.linalg.norm(np.asarray(x['value'])-np.asarray(y['value'])) for x,y in zip(a['weights'],base['weights']) if x['active']==y['active']]))))
        sensitivity.append(dict(setting=name,change=change,rows=rows,paired_vs_joint=paired([v['full_return'] for v in rows],[v['joint_return'] for v in rows]),mean_return=float(np.mean([v['full_return'] for v in rows])),mean_actions=float(np.mean([v['deployments'] for v in rows])),mean_action_changes=float(np.mean([v['actions_changed_vs_base'] for v in rows]))));print('SENSITIVITY',name,sensitivity[-1]['paired_vs_joint'],flush=True)
    result['sensitivity']=sensitivity
    Path(args.output).write_text(json.dumps(result,indent=2));print('COMPLETE',args.output,json.dumps(result['summary']),flush=True)

if __name__=='__main__':main()
