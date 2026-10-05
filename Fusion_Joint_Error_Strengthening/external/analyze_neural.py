"""Request-level verification and complete operating grid for frozen neural controls."""
import argparse, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import neural_external as nn
PROJECT=nn.PROJECT
sys.path.insert(0,str(PROJECT/'outputs/Fusion_Loss_Budget_Extension_Repro/guard'))
import run_loss_budget_guard as guard

def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def severity(rows):
    result={}
    for name,rr in [('issued',rows),('informative',[r for r in rows if r['disagreement']>0]),('admitted',[r for r in rows if r['action']])]:
        excess=[max(0.,r['gain']-r['truegross']-r['q_issued']*r['posterior_or_block_sd']) for r in rr]
        result[name]=dict(covered=sum(r['lower_covered'] for r in rr),total=len(rr),total_scaled_excess=sum(excess),max_scaled_excess=max(excess,default=0.))
    return result
def decompose(full,control):
    out=dict(additional_gain=0.,avoided_loss=0.,missed_gain=0.,incurred_loss=0.,changed_actions=0,shared_beneficial=0)
    for f,c in zip(full,control):
        assert f['k']==c['k'] and abs(f['local_net']-c['local_net'])<1e-8
        d=f['local_net'];out['shared_beneficial']+=bool(f['action'] and c['action'] and d>0)
        if f['action']==c['action']:continue
        out['changed_actions']+=1
        if f['action']:
            out['additional_gain']+=max(d,0);out['incurred_loss']+=max(-d,0)
        else:
            out['avoided_loss']+=max(-d,0);out['missed_gain']+=max(d,0)
    out['difference']=out['additional_gain']+out['avoided_loss']-out['missed_gain']-out['incurred_loss']
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--full',type=Path,required=True);ap.add_argument('--adapter',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--task-protocol',type=Path,required=True);args=ap.parse_args()
    taskout=args.output/args.task;results=json.loads((taskout/'results.json').read_text());full=json.loads(args.full.read_text())
    data,pre,entry=nn.binding(args);ft={t['seed']:t['results']['precision_joint'] for t in full['trials']}
    trials=[];checks=[]
    for trial in results['trials']:
        seed=trial['seed'];events=nn.helper.engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,nn.helper.CFG)
        reference=nn.helper.explicit_service(events,pre,[],nn.helper.CFG)
        for rule,saved in trial['results'].items():
            rs=saved['rows'];reserves={}
            for r in rs:
                e=events[r['k']];common=nn.helper.CFG['probe_drop']+(2 if e['refresh'] else 0)
                reserves[r['k']]=guard.current_decision_reserve(e['x'],e['candidate'],e['reference'],common,nn.helper.CFG)[0]
                fork=nn.helper.explicit_fork(events,r,nn.helper.CFG);assert abs(fork['actual']-r['local_net'])<1e-8
            for reserve in ['fixed130','decision']:
                rr={r['k']:130. for r in rs} if reserve=='fixed130' else reserves
                for budget in [0.,110.,130.,260.]:
                    guarded=guard.replay(rs,len(events),budget,'gross_loss',rr)
                    actual=nn.helper.explicit_service(events,pre,guarded['rows'],nn.helper.CFG)
                    prefix=guard.reconstruct_prefix_increment(events,guarded['rows'],nn.helper.CFG)
                    acts=[r for r in guarded['rows'] if r['action']];loss=sum(max(-r['local_net'],0) for r in acts)
                    delta=actual['net']-reference['net'];error=max(abs(delta-sum(r['local_net'] for r in acts)),abs(delta-prefix['final']))
                    assert error<1e-8 and loss<=budget+1e-8 and prefix['minimum']>=-budget-1e-8
                    fullguard=guard.replay(ft[seed]['rows'],len(events),budget,'gross_loss',rr)
                    decomposition=decompose(fullguard['rows'],guarded['rows'])
                    assert abs(decomposition['difference']-(sum(r['local_net'] for r in fullguard['rows'] if r['action'])-delta))<1e-8
                    trials.append(dict(seed=seed,rule=rule,reserve=reserve,budget=budget,net=actual['net'],increment=delta,
                        admissions=len(acts),harmful=sum(r['local_net']<0 for r in acts),beneficial=sum(r['local_net']>0 for r in acts),
                        negative_loss=loss,refused=sum(r['proposed_action'] and not r['action'] for r in guarded['rows']),
                        minimum_prefix_increment=prefix['minimum'],full_minus_control=decomposition,
                        severity=severity(guarded['rows'])))
                    checks.append(error)
    summary={}
    for rule in ['pdf','qmf']:
        raw=[t['results'][rule] for t in results['trials']];rows=[r for t in raw for r in t['rows']]
        summary[rule]=dict(unguarded_mean_net=float(np.mean([t['net'] for t in raw])),
            unguarded_mean_increment=float(np.mean([t['net']-full['trials'][i]['results']['reference']['net'] for i,t in enumerate(raw)])),
            unguarded_harmful=sum(t['harmful'] for t in raw),unguarded_beneficial=sum(t['beneficial'] for t in raw),
            severity=severity(rows),operating_points=[])
        for reserve in ['fixed130','decision']:
            for budget in [0.,110.,130.,260.]:
                tt=[t for t in trials if t['rule']==rule and t['reserve']==reserve and t['budget']==budget]
                fields=['admissions','harmful','beneficial','negative_loss','refused']
                summary[rule]['operating_points'].append(dict(reserve=reserve,budget=budget,mean_net=float(np.mean([t['net'] for t in tt])),
                    mean_increment=float(np.mean([t['increment'] for t in tt])),**{k:sum(t[k] for t in tt) for k in fields},
                    full_minus_control={k:sum(t['full_minus_control'][k] for t in tt) for k in tt[0]['full_minus_control']},
                    admitted_severity=dict(covered=sum(t['severity']['admitted']['covered'] for t in tt),
                        total=sum(t['severity']['admitted']['total'] for t in tt),
                        total_scaled_excess=sum(t['severity']['admitted']['total_scaled_excess'] for t in tt),
                        max_scaled_excess=max(t['severity']['admitted']['max_scaled_excess'] for t in tt))))
    dump(taskout/'budget_and_severity.json',dict(task=args.task,summary=summary,trials=trials,
        verification=dict(passed=True,guarded_trajectories=len(trials),max_accounting_error=max(checks),source_results_unchanged=True)))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
