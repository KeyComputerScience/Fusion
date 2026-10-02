"""Independent same-state, single-admission recovery and return audit.

Future labels are evaluation-only. Subsequent admissions are disabled in the
local forks. Local fork values are not summed to claim whole-policy gains.
"""
from pathlib import Path
import json
import gzip
import numpy as np
import real_air_paired_fusion as air

ROOT=Path(__file__).resolve().parent

def inspect(events, result, initial, cfg):
    deployed=initial.copy();pending=None;rows=[];net=0.;max_local_error=0.
    for k,event in enumerate(events):
        if pending is not None and pending[0]==k:
            deployed=pending[1];pending=None
        active=(event['x']@deployed).argmax(1)
        launch=bool(result['actions'][k])
        common_drop=cfg['probe_drop'] if event['probe'] else 0
        n_drop=min(len(active),common_drop+(cfg['deploy_drop'] if launch else 0))
        keep=np.arange(len(active))>=n_drop
        net+=float(np.sum(event['price'][keep]*(active[keep]==event['y'][keep])))
        if event['probe']:net-=5*cfg['probe_steps']*cfg['step_fee']
        if launch:
            net-=cfg['deploy_fee'];candidate=event['candidate'];gains=[]
            local=-cfg['deploy_fee']
            lost=slice(common_drop,common_drop+cfg['deploy_drop'])
            local-=float(np.sum(event['price'][lost]*(active[lost]==event['y'][lost])))
            for j in range(k+1,min(len(events),k+1+cfg['horizon'])):
                future=events[j]
                old=(future['x']@deployed).argmax(1)
                updated=(future['x']@candidate).argmax(1)
                gains.append(float(np.mean(updated==future['y'])-np.mean(old==future['y'])))
                retained=np.arange(len(old))>=(cfg['probe_drop'] if future['probe'] else 0)
                local+=float(np.sum(future['price'][retained]*
                    ((updated[retained]==future['y'][retained]).astype(float)-
                     (old[retained]==future['y'][retained]).astype(float))))
            sustained=any(all(g>=.03 for g in gains[start:start+3])
                          for start in range(max(0,len(gains)-2)))
            recorded=result['audit'][k]['local_advantage']
            max_local_error=max(max_local_error,abs(local-recorded))
            rows.append(dict(origin=k,accuracy_gains=gains,local_net_gain=local,
                parameter_change=float(np.linalg.norm(candidate-deployed)),
                sustained_three_window_improvement=sustained,
                valuable_sustained_recovery=bool(sustained and local>0),
                truncated_horizon=len(gains)<cfg['horizon']))
            pending=(k+1,candidate.copy())
    return dict(events=rows,return_recomputed=net,
        absolute_return_error=abs(net-result['net_return']),max_local_error=max_local_error)

def main():
    rows,metadata=air.load_air(ROOT/'data'/'AirQualityUCI.csv')
    output=dict(definition='same-deployed-state update-vs-keep fork, no later admissions; '
        'four postdeployment windows; at least three consecutive potential accuracy '
        'gains >=.03 and positive full local fee/opportunity-inclusive gain',studies={})
    for name,filename in [('formal','real_air_results.json'),
                          ('exploratory','real_air_invariant_results.json')]:
        with gzip.open(ROOT/'results'/(filename+'.gz'),'rt') as file:data=json.load(file)
        cfg=data['config']
        models,prior,q,train_rows=air.prefix_fit(rows,cfg)
        audits={mode:[] for mode in ['decision_full','context_joint',
                    'decision_diagonal','quality_only','rf_c','periodic']}
        for seedrow in data['per_seed']:
            events=air.world(rows,models,train_rows,data['split']['test'],seedrow['seed'],cfg)
            for mode in audits:
                audited=inspect(events,seedrow['results'][mode],models[4],cfg)
                audits[mode].append(dict(seed=seedrow['seed'],**audited))
        summary={}
        for mode,cases in audits.items():
            records=[r for case in cases for r in case['events']]
            summary[mode]=dict(admissions=len(records),
                positive_local_gain=sum(r['local_net_gain']>0 for r in records),
                harmful_local_gain=sum(r['local_net_gain']<0 for r in records),
                sustained=sum(r['sustained_three_window_improvement'] for r in records),
                valuable_sustained=sum(r['valuable_sustained_recovery'] for r in records),
                max_return_error=max(case['absolute_return_error'] for case in cases),
                max_local_error=max(case['max_local_error'] for case in cases),
                minimum_parameter_change=min(r['parameter_change'] for r in records) if records else None)
        output['studies'][name]=dict(summary=summary,per_seed=audits)
        print(name,summary,flush=True)
    (ROOT/'results'/'real_recovery_causal_audit_reproduced.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
