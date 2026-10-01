"""Re-execute the original synthetic protocol; retain negative results."""
from pathlib import Path
import argparse,copy,json,sys,platform,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from closed_loop import calibrate,run_closed,write_records
from data_pipeline import read_json,write_json,sha256
from run_experiments import write_csv
ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent

def run_seed(seed):
    experiment=read_json(ROOT/'experiment_config.json')
    core=read_json(ROOT/'core/config.json'); profiles=read_json(ROOT/'core/profiles.json')
    data=dict(np.load(DEST/'data/original_synthetic'/f'exogenous_seed_{seed}.npz',allow_pickle=False))
    out=DEST/'results/original'; summaries=[]
    for backbone in ['dqn','ppo']:
        agent,reference,calibrated,prefix=calibrate(data,experiment,core,profiles,seed,backbone)
        d=out/'calibration'/f'{backbone}_seed_{seed}'; d.mkdir(parents=True,exist_ok=True)
        write_json(d/'reference.json',reference); write_json(d/'profiles.json',calibrated)
        write_records(d/'prefix.csv.gz',prefix); torch.save(agent.checkpoint(),d/'initial_policy.pt')
        for method in ['no_rt','fusion','periodic']:
            records,windows,events,summary=run_closed(data,experiment,core,calibrated,agent,reference,method,seed)
            d=out/'runs'/f'{backbone}_{method}_seed_{seed}'; d.mkdir(parents=True,exist_ok=True)
            write_records(d/'slots.csv.gz',records); write_json(d/'windows.json',windows)
            write_json(d/'deployments.json',events); write_json(d/'summary.json',summary)
            summaries.append(summary)
            print(f'original {backbone} seed={seed} {method}: return={summary["return"]:.3f} completion={summary["completion_fraction"]:.4f} deployments={summary["deployed_jobs"]}',flush=True)
    return summaries

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    out=DEST/'results/original';out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(ROOT/'experiment_config.json')
    manifest={'kind':'reexecuted_controlled_synthetic','original_protocol':cfg,'seeds':cfg['seeds'],'methods':['no_rt','fusion','periodic'],'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'note':'Environment differs from upstream Linux/Torch2.3.1. Values must be reported as new execution, not assumed bitwise-identical.'}
    write_json(out/'manifest.json',manifest)
    summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_seed,cfg['seeds']):summaries.extend(result)
    write_json(out/'seed_results.json',summaries);write_csv(out/'seed_results.csv',summaries)
if __name__=='__main__':main()
