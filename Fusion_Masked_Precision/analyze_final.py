"""Exact analysis of every frozen outcome and fixed-state interventions."""
import sys,json,math,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import temporal_fusion as core
ROOT=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def paired(v):
    x=np.asarray(v,float);half=2.776445105*x.std(ddof=1)/math.sqrt(len(x))
    return dict(values=x.tolist(),mean=float(x.mean()),conditional_t4_ci=[float(x.mean()-half),float(x.mean()+half)])
allr=json.loads((ROOT/'final_development/results.json').read_text())
allr.update(json.loads((ROOT/'rss_validation/results.json').read_text()))
report={};checks=[];non_scalar=[]
for name,r in allr.items():
    groups={};full=[t['results']['precision_joint'] for t in r['trials']]
    for arm in r['summary']:
        if arm=='precision_joint':continue
        groups[arm]=paired([t['results']['precision_joint']['net']-t['results'][arm]['net'] for t in r['trials']])
    rows=[x for t in r['trials'] for x in t['results']['precision_joint']['rows']]
    non_scalar.extend(x['precision_non_scalar'] for x in rows if len(x['weights'])>1)
    cert=sum(sum(x['gain']-5-x['q_issued']*x['posterior_or_block_sd']-x['posterior_or_block_sd']*max(0,x['standardized_score']-x['q_issued']) for x in rr['rows'] if x['action']) for rr in full)/5
    selection=ROOT/('rss_validation' if name=='rss348' else 'final_development')/(name+'_selection.json')
    sel=json.loads(selection.read_text())
    report[name]=dict(summary=r['summary'],paired=groups,selected=r['selected'],split=sel['split'],mean_weighted_certificate=cert)
    for t in r['trials']:
        for arm,c in t['checks'].items():checks.append(dict(task=name,seed=t['seed'],arm=arm,**c))
audit=dict(trajectories=len(checks),forks=sum(c['forks'] for c in checks),max_service_error=max(max(c['service_errors'].values()) for c in checks),max_fork_error=max(c['fork_error'] for c in checks),max_target_error=max(c['target_error'] for c in checks),max_q_identity_error=max(abs(c['q_identity_error']) for c in checks),passed=all(c['passed'] for c in checks),non_scalar=dict(multisource_origins=len(non_scalar),above_1e_8=sum(x>1e-8 for x in non_scalar),median=float(np.median(non_scalar))))
dump(ROOT/'analysis_summary.json',dict(tasks=report,execution=audit))

engine,recovery,guard,source,cfg0,study,service,fork=core.support.setup(core.support.DEFAULT_PACKAGE,core.support.DEFAULT_GUARD)
cfg0.update(prior_sd=.1,tau=.01)
sys.path.insert(0,str(core.PROJECT/'work/fusion_focus_20261003/fresh'))
from new_data_adapter import load_dataset,make_prefix
cases=[];causality=0;score_dominance=[]
for name in ('gas224','rss348'):
    if name=='rss348':
        data,der=load_dataset(ROOT/'physical/data',name);pre=make_prefix(data,der,cfg0,engine)
    else:data,pre=study(name,source['studies'][name])
    rr=allr[name];sel=rr['selected']['precision_joint'];cfg=dict(cfg0,prior_sd=sel['prior_sd'],q_floor=sel['q_floor'],threshold=sel['margin'])
    for t in rr['trials']:
        events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,t['seed'],cfg)
        stream=core.augment(events,pre,engine.precompute(events,pre,cfg),cfg)
        for row,d in zip(t['results']['precision_joint']['rows'],stream['decisions']):
            q=row['q_issued'];a=row['action'];w,cert,F,S,_,_=core.forecast(d,pre,cfg,'precision_joint',q)
            score=F-q*S-5;assert abs(score-row['gate_score'])<1e-7
            poison=dict(d,truegross=d['truegross']+1000,localnet=-1000)
            other=core.forecast(poison,pre,cfg,'precision_joint',q)
            assert np.max(np.abs(w-other[0]))<1e-12 and F==other[2] and S==other[3];causality+=1
            variants={}
            dd=dict(d,temporal=dict(d['temporal'],P=np.diag(np.diag(d['temporal']['P']))))
            variants['P_diagonal_only']=core.forecast(dd,pre,cfg,'precision_joint',q)
            variants['scalar_mass']=core.forecast(d,pre,cfg,'scalar_mass',q)
            variants['complete_only']=core.forecast(d,pre,cfg,'complete_only',q)
            variants['no_posterior']=core.forecast(d,pre,cfg,'no_posterior',q)
            variants['decoupled']=core.forecast(d,pre,cfg,'precision_joint',0.)
            for variant,v in variants.items():
                alt=v[2]-q*v[3]-5;admit=bool(alt>0)
                if variant=='decoupled':score_dominance.append(score-alt)
                if a!=admit:
                    cases.append(dict(task=name,seed=t['seed'],origin=row['k'],variant=variant,full_action=a,variant_action=admit,full_score=score,variant_score=alt,actual_lease_increment=row['local_net'],one_step_full_gain=(int(a)-int(admit))*row['local_net'],full_weights=w.tolist(),variant_weights=v[0].tolist(),issued_q=q,R=d['temporal']['R'].tolist(),P=d['temporal']['P'].tolist(),mu=d['temporal']['mu'].tolist(),current_h=d['h'].tolist(),non_scalar=d['temporal']['non_scalar']))
out={}
for name in ('gas224','rss348'):
    out[name]={v:dict(changed_actions=len(c),one_step_gain=sum(z['one_step_full_gain'] for z in c),positive_changes=sum(z['one_step_full_gain']>0 for z in c),negative_changes=sum(z['one_step_full_gain']<0 for z in c)) for v in ('P_diagonal_only','scalar_mass','complete_only','no_posterior','decoupled') for c in [[z for z in cases if z['task']==name and z['variant']==v]]}
dump(ROOT/'fixed_state_interventions.json',dict(scope='fixed issued state/q one-step mechanism interventions, not adaptive-policy trajectory comparisons',summaries=out,cases=cases,causal_future_truth_checks=causality,min_score_optimality_gap=min(score_dominance)))
print(json.dumps(dict(execution=audit,mechanism=out),indent=2))
