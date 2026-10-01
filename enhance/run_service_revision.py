"""Run frozen stationary and joint-shift protocols; all results are retained."""
from pathlib import Path
import copy,json,sys,platform
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from closed_loop import calibrate,write_records
from data_pipeline import read_json,write_json,generate,sha256
from run_experiments import write_csv
from revision_service import run_revision
ROOT=Path(__file__).resolve().parent;DEST=ROOT.parent

def run_task(task):
    scenario,seed=task
    settings=read_json(ROOT/'service_revision_config.json')
    experiment=copy.deepcopy(settings['experiment']);experiment.update(settings['scenarios'][scenario]);experiment['scenario']=scenario
    core=read_json(ROOT/'core/config.json');profiles=read_json(ROOT/'core/profiles.json')
    for profile,steps,delay in zip(profiles['training'],settings['profile_updates'],settings['profile_delays']):
        profile['updates']=steps;profile['deployment_delay']=delay
    data=generate(experiment,core['coordination']['nodes'],seed)
    # One prescribed resource shift, applied equally across methods.
    if scenario=='joint_shift':
        phase=data['phase'];data['capacities'][phase==1,:,1]*=.8
    data['provenance']=np.array('controlled_synthetic_declared_service_shift')
    d=DEST/'data/service_revision'/scenario;d.mkdir(parents=True,exist_ok=True)
    path=d/f'exogenous_seed_{seed}.npz';np.savez_compressed(path,**data)
    out=DEST/'results/service'/scenario;summaries=[]
    for backbone in experiment['backbones']:
        initial,reference,calibrated,prefix=calibrate(data,experiment,core,profiles,seed,backbone)
        d=out/'calibration'/f'{backbone}_seed_{seed}';d.mkdir(parents=True,exist_ok=True)
        write_json(d/'reference.json',reference);write_json(d/'profiles.json',calibrated);write_records(d/'prefix.csv.gz',prefix)
        torch.save(initial.checkpoint(),d/'initial_policy.pt')
        # Fixed profile chosen from calibration only, with >=99% prefix capacity feasibility.
        load=.65+.12*data['arrivals'][:experiment['calibration_slots']].sum(axis=(1,2))/10
        feasible=[]
        for p in calibrated['inference']:
            required=load[:,None,None]*np.array(p['resources'])[None,None,:]
            fraction=np.mean(np.all(required<=data['capacities'][:experiment['calibration_slots']]+1e-12,axis=(1,2)))
            if fraction>=.99:feasible.append(p)
        fixed=max(feasible,key=lambda p:p['beta']*sum(core['coordination']['quality_weights'][k]*v for k,v in p['quality_prior'].items()))['id']
        for method in experiment['main_methods']:
            records,windows,events,summary=run_revision(data,experiment,core,calibrated,initial,reference,method,seed,fixed)
            summary['input_sha256']=sha256(path)
            d=out/'runs'/f'{backbone}_{method}_seed_{seed}';d.mkdir(parents=True,exist_ok=True)
            write_records(d/'slots.csv.gz',records);write_json(d/'windows.json',windows);write_json(d/'deployments.json',events);write_json(d/'summary.json',summary)
            summaries.append(summary)
            print(f'{scenario} {backbone} seed={seed} {method}: return={summary["return"]:.2f} completion={summary["completion_fraction"]:.4f} updates={summary["gradient_updates"]}',flush=True)
    return summaries

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    settings=read_json(ROOT/'service_revision_config.json')
    out=DEST/'results/service';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'manifest.json',{'settings':settings,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'training_cost_unit':'normalized simulated resource cost, not wall-clock/GPU time','causal_return':'independent closed-loop executions','all_methods_share_initial_weights':True})
    tasks=[(s,n) for s in settings['scenarios'] for n in settings['experiment']['seeds']]
    if args.smoke:tasks=[('joint_shift',10)]
    summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_task,tasks):summaries.extend(result)
    write_json(out/'seed_results.json',summaries);write_csv(out/'seed_results.csv',summaries)
if __name__=='__main__':main()
