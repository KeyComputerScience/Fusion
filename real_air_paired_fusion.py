"""Independent chronological Air Quality validation of paired action fusion.

Real hourly records, four physical sensor histories, and CO targets. Six classes
are prefix quantile bins, not health categories. Training/audit disjoint by fixed
timestamp hash. Feedback latency and service costs are explicit interventions.
"""
from __future__ import annotations
import argparse,csv,datetime as dt,hashlib,json,math
from pathlib import Path
import numpy as np
import real_action_fusion as gas

ROOT=Path(__file__).resolve().parent
MODES=('decision_full','context_joint','decision_diagonal','quality_only','rf_c',
       'prefix_only_archive','no_context','no_delay_replay','uncorrected_gain','periodic','frozen_deploy')
CFG=dict(gas.BASE)

def load_air(path):
    sensor_names=('PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)')
    records=[]
    with path.open(newline='',encoding='utf-8-sig') as file:
        for r in csv.DictReader(file,delimiter=';'):
            if not r['Date'].strip():continue
            date=dt.datetime.strptime(r['Date']+' '+r['Time'],'%d/%m/%Y %H.%M.%S')
            vals=np.array([float(r[name].replace(',','.')) for name in sensor_names])
            co=float(r['CO(GT)'].replace(',','.'))
            records.append((date,vals,co))
    records.sort(key=lambda r:r[0]);last=np.zeros(4);history=[];kept=[]
    for raw_i,(date,val,co) in enumerate(records):
        current_ok=val!=-200
        last=np.where(current_ok,val,last);history.append(last.copy())
        # All raw hours contribute to lag construction before target filtering.
        lag=np.stack([history[max(0,len(history)-1-j)] for j in range(32)],axis=1).reshape(-1)
        if co==-200 or not current_ok.all():continue
        timestamp=date.isoformat();audit=int(hashlib.sha256(timestamp.encode()).hexdigest()[:8],16)%10<3
        kept.append(dict(date=date,timestamp=timestamp,raw_index=raw_i,x=lag,co=co,audit=audit,month=date.strftime('%Y-%m')))
    march=[r for r in kept if r['month']=='2004-03'];train=[r for r in march if not r['audit']]
    bins=np.quantile([r['co'] for r in train],np.arange(1,6)/6)
    xtrain=np.stack([r['x'] for r in train]);mean=xtrain.mean(0);scale=np.maximum(xtrain.std(0),.05)
    for r in kept:
        r['x']=np.r_[np.clip((r['x']-mean)/scale,-8,8),1.]
        r['y']=int(np.searchsorted(bins,r['co'],side='right'))
    monthly={m:dict(n=sum(r['month']==m for r in kept),audit=sum(r['month']==m and r['audit'] for r in kept)) for m in sorted(set(r['month'] for r in kept))}
    gaps=[(b['date']-a['date']).total_seconds()/3600 for a,b in zip(kept,kept[1:])]
    meta=dict(raw_rows=len(records),complete_case_rows=len(kept),excluded_rows=len(records)-len(kept),co_bins=bins.tolist(),classes='six CO(GT) prefix quantile bins; not health thresholds',sensor_columns=list(sensor_names),feature_definition='current plus preceding 31 raw hours, source-only past forward fill before target filtering',audit_rule='SHA256(timestamp) first 8 hex digits modulo10 <3; fixed once',monthly=monthly,raw_start=records[0][0].isoformat(),raw_end=records[-1][0].isoformat(),complete_case_gap_hours=dict(mean=float(np.mean(gaps)),max=float(max(gaps)),gaps_above_one=int(sum(g>1 for g in gaps))),csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return kept,meta

def prefix_fit(rows,cfg):
    march_train=[r for r in rows if r['month']=='2004-03' and not r['audit']]
    april_audit=[r for r in rows if r['month']=='2004-04' and r['audit']]
    train_rows=[r for r in rows if r['month'] in ('2004-03','2004-04') and not r['audit']]
    x=np.stack([r['x'] for r in march_train]);y=np.array([r['y'] for r in march_train]);models=[]
    for s in range(5):
        f=gas.features(s);models.append(gas.gradient(np.zeros((len(f),6)),x[:,f],y,500,cfg))
    ax=np.stack([r['x'] for r in april_audit]);ay=np.array([r['y'] for r in april_audit]);ap=np.stack([gas.softmax(ax[:,gas.features(s)]@models[s]) for s in range(4)])
    prior=dict(x=ax,p=ap,y=ay,context=gas.context(ax),price=np.array([1.+2.*(gas.context(r['x'][None,:])[0]>1.) for r in april_audit]))
    q=1/(np.mean(np.sum((ap-np.eye(6)[ay][None,:,:])**2,axis=2),axis=1)+.05)
    return models,prior,q,train_rows

def world(rows,models,train_rows,months,seed,cfg):
    rng=np.random.default_rng(seed);events=[];models=[m.copy() for m in models]
    for month in months:
        rr=[r for r in rows if r['month']==month]
        for start in range(0,len(rr),cfg['window']):
            block=rr[start:start+cfg['window']];x=np.stack([r['x'] for r in block]);y=np.array([r['y'] for r in block]);c=gas.context(x);mask=np.ones(4,bool);k=len(events)
            if cfg['mask_stress'] and k%13 in (4,5,6):mask[k//13%4]=False
            events.append(dict(x=x,y=y,context=c,price=np.array([1.+2.*(gas.context(r['x'][None,:])[0]>1.) for r in block]),month=month,audit=np.array([r['audit'] for r in block]),delay=int(rng.integers(cfg['delay_low'],cfg['delay_high']+1)),mask=mask,timestamps=[r['timestamp'] for r in block]))
    arrived_x=[np.stack([r['x'] for r in train_rows])];arrived_y=[np.array([r['y'] for r in train_rows])];pending=[];origins=[-1];cid=0
    for k,e in enumerate(events):
        for due,r in pending:
            train=~r['audit']
            if due<=k and train.any():arrived_x.append(r['x'][train]);arrived_y.append(r['y'][train]);origins.append(r['origin'])
        pending=[(due,r) for due,r in pending if due>k]
        probe=k%cfg['probe_period']==0
        if probe:
            tx=np.concatenate(arrived_x)[-cfg['train_buffer']:];ty=np.concatenate(arrived_y)[-cfg['train_buffer']:]
            models=[gas.gradient(w,tx[:,gas.features(s)],ty,cfg['probe_steps'],cfg) for s,w in enumerate(models)];cid+=1
        e.update(source=np.stack([gas.softmax(e['x'][:,gas.features(s)]@models[s]) for s in range(4)]),candidate=models[4].copy(),candidate_id=cid,probe=probe,origin=k,probe_max_training_origin=max(origins) if probe else None)
        pending.append((k+max(1,e['delay']),e))
    return events

def paired_moments(records,prior,c,k,mask,deployed,candidate,cfg,mode):
    chunks=[(cfg['prior_mass'],prior)]
    if mode!='prefix_only_archive':
        for r in records[-cfg['archive']:]:
            if not np.all(r['mask'][mask]):continue
            age=k-r['origin'] if mode!='no_delay_replay' else k-r['arrival']
            kernel=1. if mode=='no_context' else math.exp(-float(np.sum((c-r['context'])**2))/cfg['context_bandwidth']**2)
            chunks.append((cfg['beta']**max(0,age)*kernel,r))
    den=sum(a for a,r in chunks);ez=np.zeros(len(mask));zz=np.zeros((len(mask),len(mask)));ee=np.zeros((len(mask),len(mask)));mean_e=np.zeros((len(mask),6))
    for a,r in chunks:
        e=r['p'][mask]-np.eye(6)[r['y']][None,:,:]
        active=(r['x']@deployed).argmax(1);cand=(r['x']@candidate).argmax(1)
        b=r['price'][:,None]*(np.eye(6)[cand]-np.eye(6)[active]);z=np.einsum('sic,ic->si',e,b)
        ez+=a*z.mean(1);zz+=a*(z@z.T/len(r['y']));ee+=a*np.einsum('sic,tic->st',e,e)/len(r['y']);mean_e+=a*e.mean(1)
    ez/=den;mean_e/=den;ss=zz/den-np.outer(ez,ez);rr=ee/den-mean_e@mean_e.T
    # Symmetry/PSD only correct numerical roundoff, not statistical deficits.
    def psd(a):
        eig,v=np.linalg.eigh((a+a.T)/2);return (v*np.maximum(eig,0))@v.T
    return psd(ss),psd(rr),ez

def run(events,models,prior,q,cfg,mode):
    deployed=models[4].copy();pending=[];archive=[];pending_deploy=None;used=set();budget={};cooldown=-1;results=[];actions=[];audit=[];fees=0;probe_fees=0;deploy_fees=0;steps=0;drops=0;maxkkt=0.;fails=0;min_eig=1.;correct=[];labels=[];months=[];baseacc=[];activeacc=[]
    for k,e in enumerate(events):
        for due,r in pending:
            if due<=k:archive.append(r)
        pending=[(due,r) for due,r in pending if due>k]
        if pending_deploy is not None and pending_deploy[0]==k:deployed=pending_deploy[1];pending_deploy=None
        active=(e['x']@deployed).argmax(1);cand=(e['x']@e['candidate']).argmax(1);b=e['price'][:,None]*(np.eye(6)[cand]-np.eye(6)[active]);mask=np.flatnonzero(e['mask']);pp=e['source'][mask]
        action_cov,r,mu=paired_moments(archive,prior,e['context'],k,mask,deployed,e['candidate'],cfg,mode)
        if np.trace(action_cov)>1e-12:action_cov=action_cov*np.trace(r)/np.trace(action_cov)
        else:action_cov=r.copy()
        s=r if mode in ('context_joint','rf_c') else (np.diag(np.diag(action_cov)) if mode=='decision_diagonal' else action_cov)
        w,cert=gas.fuse_weights(q[mask],s,cfg,mode);maxkkt=max(maxkkt,cert['kkt_residual']);fails+=int(not cert['converged']);min_eig=min(min_eig,float(np.linalg.eigvalsh(s).min()))
        fused=np.einsum('s,sic->ic',w,pp);raw=float(np.sum(b*fused)*cfg['horizon']);bias=float(cfg['horizon']*len(b)*(w@mu));gain=raw-(0 if mode=='uncorrected_gain' else bias)
        cost=cfg['deploy_fee']+cfg['deploy_drop']*float(e['price'].mean());eligible=e['candidate_id'] not in used and k>=cooldown and budget.get(e['month'],0)<cfg['budget_per_batch'] and k+1<len(events)
        if mode=='frozen_deploy':launch=False
        elif mode=='periodic':launch=eligible and e['probe']
        else:launch=eligible and gain>cost+cfg['threshold']
        shadow={}
        for alt in ('context_joint','decision_diagonal','quality_only','rf_c'):
            am=np.diag(np.diag(action_cov)) if alt=='decision_diagonal' else r;aw,ac=gas.fuse_weights(q[mask],am,cfg,alt);af=np.einsum('s,sic->ic',aw,pp);ag=float(cfg['horizon']*np.sum(b*af)-cfg['horizon']*len(b)*(aw@mu))
            shadow[alt]=dict(action=bool(eligible and ag>cost+cfg['threshold']),score=ag-cost,weight_l2=float(np.linalg.norm(w-aw)))
        if launch:
            pending_deploy=(k+1,e['candidate'].copy());used.add(e['candidate_id']);cooldown=k+cfg['cooldown'];budget[e['month']]=budget.get(e['month'],0)+1;deploy_fees+=cfg['deploy_fee'];fees+=cfg['deploy_fee']
        common_drop=cfg['probe_drop'] if e['probe'] else 0;drop=min(len(active),common_drop+(cfg['deploy_drop'] if launch else 0));keep=np.arange(len(active))>=drop;gross=float(np.sum(e['price'][keep]*(active[keep]==e['y'][keep])))
        pf=0.
        if e['probe']:steps+=5*cfg['probe_steps'];pf=5*cfg['probe_steps']*cfg['step_fee'];probe_fees+=pf;fees+=pf
        net=gross-pf-(cfg['deploy_fee'] if launch else 0);results.append(net);drops+=drop;actions.append(bool(launch));correct.extend((active==e['y']).tolist());labels.extend(e['y'].tolist());months.extend([e['month']]*len(active));base=(e['x']@models[4]).argmax(1);baseacc.append(float(np.mean(base==e['y'])));activeacc.append(float(np.mean(active==e['y'])))
        ar=e['audit']
        if ar.any():
            rec=dict(x=e['x'][ar],p=e['source'][:,ar,:].copy(),y=e['y'][ar],context=e['context'],price=e['price'][ar],mask=e['mask'],origin=k,arrival=k+max(1,e['delay']))
            pending.append((rec['arrival'],rec))
        local=-cfg['deploy_fee']
        for j in range(k+1,min(len(events),k+1+cfg['horizon'])):
            f=events[j];pa=(f['x']@deployed).argmax(1);pc=(f['x']@e['candidate']).argmax(1);fk=np.arange(len(pa))>=(cfg['probe_drop'] if f['probe'] else 0)
            local+=float(np.sum(f['price'][fk]*((pc[fk]==f['y'][fk]).astype(float)-(pa[fk]==f['y'][fk]).astype(float))))
        lost=slice(common_drop,common_drop+cfg['deploy_drop']);local-=float(np.sum(e['price'][lost]*(active[lost]==e['y'][lost])))
        audit.append(dict(origin=k,action=bool(launch),eligible=bool(eligible),weight=w.tolist(),active_mask=mask.tolist(),gain_score=gain-cost,raw_gain=raw,bias_correction=bias,local_advantage=local,shadow=shadow,audit_support=sum(len(a['y']) for a in archive[-cfg['archive']:]),prediction_disagreement=float(np.mean(active!=cand))))
    corr=np.array(correct);labels=np.array(labels);mm=np.array(months);monthly={}
    for month in sorted(set(months)):
        sel=mm==month;rec=[float(corr[sel&(labels==c)].mean()) for c in range(6) if np.any(sel&(labels==c))];monthly[month]=dict(n=int(sel.sum()),accuracy=float(corr[sel].mean()),balanced_accuracy=float(np.mean(rec)),classes_present=len(rec))
    recov=[]
    for k,a in enumerate(actions):
        if a:
            hits=[j for j in range(k+1,len(events)-2) if all(activeacc[h]>=baseacc[h]+.03 for h in range(j,j+3))];j=hits[0] if hits else None;recov.append(dict(launch=k,first_three_window_improvement=j,latency=None if j is None else j-k))
    return dict(net_return=float(sum(results)),deployments=sum(actions),accuracy=float(corr.mean()),monthly=monthly,actions=actions,audit=audit,recovery=recov,probe_steps=steps,probe_fee=probe_fees,deploy_fee=deploy_fees,total_fee=fees,dropped_service=drops,max_kkt=maxkkt,solver_failures=fails,min_eigenvalue=min_eig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'uci_air'/'AirQualityUCI.csv');p.add_argument('--pilot',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'real_air_results.json');args=p.parse_args()
    rows,meta=load_air(args.data);cfg=dict(CFG);models,prior,q,train_rows=prefix_fit(rows,cfg);calib=[]
    for lam in (.25,1.,4.):
        for threshold in (0.,8.,24.):
            cc=dict(cfg,lam=lam,threshold=threshold)
            ww=[world(rows,models,train_rows,['2004-05'],seed,cc) for seed in (98001,98002,98003)]
            for mode in MODES[:5]:
                rr=[run(w,models,prior,q,cc,mode) for w in ww];calib.append(dict(mode=mode,lam=lam,threshold=threshold,mean_return=float(np.mean([r['net_return'] for r in rr])),mean_actions=float(np.mean([r['deployments'] for r in rr]))))
    best={mode:max([r for r in calib if r['mode']==mode],key=lambda x:(x['mean_return'],-x['lam'],-x['threshold'])) for mode in MODES[:5]};cfg.update(lam=best['decision_full']['lam'],threshold=best['decision_full']['threshold'])
    months=sorted(set(r['month'] for r in rows if r['month']>='2004-06'))
    result=dict(dataset='UCI Air Quality, dataset360',metadata=meta,config=cfg,split=dict(train='2004-03 train hash70%',prior='2004-04 audit hash30%',calibration='2004-05',test=months),quality=q.tolist(),quality_definition='inverse April audit source Brier risk+.05, an outcome reliability prior',calibration_grid=calib,calibration_best=best,seeds=list(range(99001,99011)),limitations=['10 seeds vary artificial feedback delay on one fixed chronological dataset','complete-case evaluated queue excludes missing target or current source records; no full-hour availability claim','costs, queue windows, deployment lag and feedback latency are simulated on real covariates/targets','paired audit counterfactuals reduce marginal product assumption, but context and future transfer remain empirical','centering/bias correction and adaptive archived contrasts do not imply a martingale certificate'])
    result['scalar_baseline_semantics']='RF-C-V: diagonal centered source-class covariance, hence scalar variance, plus common action-bias correction; not bias-inclusive MSE or untouched Original RF.'
    (ROOT/'real_air_pilot.json').write_text(json.dumps(result,indent=2));print('AIR_FROZEN',json.dumps(dict(config=cfg,best=best,metadata=meta)),flush=True)
    if args.pilot:return
    ps=[]
    for seed in result['seeds']:
        ww=world(rows,models,train_rows,months,seed,cfg);rr={mode:run(ww,models,prior,q,cfg,mode) for mode in MODES}
        for mode in ('context_joint','decision_diagonal','quality_only','rf_c'):
            cc=dict(cfg,lam=best[mode]['lam'],threshold=best[mode]['threshold']);rr[mode+'_prefix_tuned']=run(ww,models,prior,q,cc,mode)
        full=rr['decision_full'];shadow={}
        for alt in MODES[1:5]:
            crossing=np.array([v['action']!=v['shadow'][alt]['action'] for v in full['audit']]);signed=np.array([v['local_advantage']*(1 if v['action'] else -1) for v in full['audit']]);shadow[alt]=dict(crossings=int(crossing.sum()),beneficial=int(np.sum(signed[crossing]>0)),harmful=int(np.sum(signed[crossing]<0)),local_sum=float(signed[crossing].sum()),mean_weight_l2=float(np.mean([v['shadow'][alt]['weight_l2'] for v in full['audit']])))
        diag={mode:int(np.sum(np.array(full['actions'])!=np.array(r['actions']))) for mode,r in rr.items()};ps.append(dict(seed=seed,results=rr,actual_action_disagreements=diag,same_state_shadow_audit=shadow));print('AIR_SEED',seed,{m:(round(r['net_return'],2),r['deployments']) for m,r in rr.items()},flush=True)
    result['per_seed']=ps;keys=list(ps[0]['results']);result['summary']={mode:dict(mean_return=float(np.mean([v['results'][mode]['net_return'] for v in ps])),mean_deployments=float(np.mean([v['results'][mode]['deployments'] for v in ps])),mean_accuracy=float(np.mean([v['results'][mode]['accuracy'] for v in ps])),mean_action_disagreements=float(np.mean([v['actual_action_disagreements'][mode] for v in ps])),paired_vs_full=gas.paired([v['results']['decision_full']['net_return'] for v in ps],[v['results'][mode]['net_return'] for v in ps])) for mode in keys};args.output.write_text(json.dumps(result,indent=2));print('AIR_MAIN_SAVED',json.dumps(result['summary']),flush=True)
    settings=[('beta_90',dict(beta=.9)),('beta_1',dict(beta=1.)),('lambda_0',dict(lam=0.)),('lambda_4',dict(lam=4.)),('fee_4',dict(deploy_fee=4.)),('fee_24',dict(deploy_fee=24.)),('delay_1',dict(delay_low=1,delay_high=1)),('delay_8',dict(delay_low=8,delay_high=8)),('quality_flat',dict(quality_power=0.)),('outage',dict(mask_stress=True))];sensitivity=[]
    for name,change in settings:
        cc=dict(cfg,**change);rr=[]
        for seed in result['seeds'][:5]:
            ww=world(rows,models,train_rows,months,seed,cc);a=run(ww,models,prior,q,cc,'decision_full');b=run(ww,models,prior,q,cc,'context_joint');base=next(v['results']['decision_full'] for v in ps if v['seed']==seed);rr.append(dict(seed=seed,full_return=a['net_return'],joint_return=b['net_return'],deployments=a['deployments'],actions_changed_vs_base=int(np.sum(np.array(a['actions'])!=np.array(base['actions'])))))
        sensitivity.append(dict(setting=name,change=change,rows=rr,unique_replay_count=1 if name in ('delay_1','delay_8') else 5,interval_interpretation='Fixed-delay seeds repeat one deterministic replay.' if name in ('delay_1','delay_8') else 'Conditional artificial-delay replay interval.',mean_return=float(np.mean([r['full_return'] for r in rr])),mean_actions=float(np.mean([r['deployments'] for r in rr])),mean_action_changes=float(np.mean([r['actions_changed_vs_base'] for r in rr])),paired_vs_joint=gas.paired([r['full_return'] for r in rr],[r['joint_return'] for r in rr])));print('AIR_SENSITIVITY',name,sensitivity[-1]['paired_vs_joint'],flush=True)
    result['sensitivity']=sensitivity;args.output.write_text(json.dumps(result,indent=2));print('AIR_COMPLETE',args.output,flush=True)

if __name__=='__main__':main()
