"""One-at-a-time diagnostic sensitivity with frozen selected controls.

No test-driven selection: fixed selected lambda/margin and prefix q. Changes are
quality prior power, risk multiplier, posterior aversion, and origin-age beta.
Gate floor and all candidate/source/service work remain unchanged. All outcomes
including ties and negative utility are retained. Original results are untouched.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import independent_bayes_fusion as engine
from audit_canary_execution import load_study

ROOT=Path(__file__).resolve().parent


def paired(values):
    x=np.asarray(values,dtype=float);mean=float(x.mean());n=len(x)
    half=2.776445105*float(x.std(ddof=1))/np.sqrt(n) if n==5 else None
    return dict(mean=mean,ci95_t4=[mean-half,mean+half] if half is not None else None,
                positive=int(np.sum(x>0)),negative=int(np.sum(x<0)),ties=int(np.sum(x==0)),values=x.tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'independent_data');parser.add_argument('--output',type=Path,default=ROOT/'new_bayes_sensitivity');parser.add_argument('--datasets',nargs='+',default=['occupancy357','occupancy864','mhealth319','har240']);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    locations={'occupancy357':ROOT/'new_bayes_fusion','occupancy864':ROOT/'new_bayes_fusion','mhealth319':ROOT/'new_bayes_extension','har240':ROOT/'new_bayes_extension'}
    changes=[('baseline',{}),('quality_power_0p5',{'quality_power':.5}),('quality_power_2',{'quality_power':2.}),('risk_lambda_0p5',{'lam_multiplier':.5}),('risk_lambda_2',{'lam_multiplier':2.}),('posterior_eta_0',{'eta':0.}),('posterior_eta_8',{'eta':8.}),('history_beta_0p90',{'beta':.9}),('history_beta_1',{'beta':1.})]
    allresults={name:json.loads((args.output/f'{name}_sensitivity.json').read_text()) for name in locations if (args.output/f'{name}_sensitivity.json').exists()};summary={name:r['summary'] for name,r in allresults.items()};max_baseline_error=0.
    for name in args.datasets:
        folder=locations[name]
        data,pre,saved=load_study(args.data,folder,name);cfg0=dict(saved['selected']['bayes_both']);qinit=float(saved['q_initial']['bayes_both']);trials=[]
        for trial in saved['trials']:
            seed=trial['seed'];events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,engine.BASE)
            streams={beta:engine.precompute(events,pre,dict(cfg0,beta=beta)) for beta in (.9,.97,1.)}
            results={}
            for label,change in changes:
                cfg=dict(cfg0);cfg.update({k:v for k,v in change.items() if k!='lam_multiplier'})
                if 'lam_multiplier' in change:cfg['lam']=cfg0['lam']*change['lam_multiplier']
                result=engine.run(streams[cfg['beta']],pre,cfg,'bayes_both',qinit)
                baseline_saved=trial['results']['bayes_both'];actiondiff=int(np.sum(np.asarray(result['actions'])!=np.asarray(baseline_saved['actions'])))
                if label=='baseline':
                    error=max(abs(result['net']-baseline_saved['net']),abs(result['harmful']-baseline_saved['harmful']),actiondiff)
                    max_baseline_error=max(max_baseline_error,error);assert error<1e-8,'baseline reproduction mismatch'
                results[label]=dict(config=cfg,net=result['net'],deployments=result['deployments'],harmful=result['harmful'],beneficial=result['beneficial'],accuracy=result['accuracy'],action_changes_vs_baseline=actiondiff,actions=result['actions'],q_final=result['q_final'],calibration_identity_error=result['calibration_identity_error'],solver_failures=result['solver_failures'],max_kkt=result['max_kkt'])
            trials.append(dict(seed=seed,results=results));print('SENSITIVITY_DIAGNOSTIC',name,seed,'retained',len(results),'variants',flush=True)
        table={}
        for label,_ in changes:
            rows=[t['results'][label] for t in trials];deltas=[t['results'][label]['net']-t['results']['baseline']['net'] for t in trials]
            table[label]=dict(mean_net=float(np.mean([r['net'] for r in rows])),mean_deployments=float(np.mean([r['deployments'] for r in rows])),mean_harmful=float(np.mean([r['harmful'] for r in rows])),mean_beneficial=float(np.mean([r['beneficial'] for r in rows])),mean_action_changes=float(np.mean([r['action_changes_vs_baseline'] for r in rows])),mean_accuracy=float(np.mean([r['accuracy'] for r in rows])),net_delta_vs_baseline=paired(deltas),solver_failures=sum(r['solver_failures'] for r in rows))
        out=dict(kind='diagnostic sensitivity, not confirmatory tuning or selected winner',dataset=name,engine_sha=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),source_result_sha=hashlib.sha256((folder/f'{name}_results.json').read_bytes()).hexdigest(),source_selected=cfg0,prefix_q_frozen=qinit,variance_floor_fixed=1e-4,grid=[dict(label=k,changes=v) for k,v in changes],trials=trials,summary=table)
        (args.output/f'{name}_sensitivity.json').write_text(json.dumps(out,indent=2));allresults[name]=out;summary[name]=table
    (args.output/'results.json').write_text(json.dumps(allresults,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2));(args.output/'reproduction_audit.json').write_text(json.dumps(dict(all_passed=True,max_baseline_error=max_baseline_error,original_source_results_unchanged=True,source_sha=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()),indent=2));print('SENSITIVITY_COMPLETE',dict(datasets=list(allresults),max_baseline_error=max_baseline_error),flush=True)

if __name__=='__main__':main()
