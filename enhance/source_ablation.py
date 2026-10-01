"""Three source-removal controls; keep the original 700-run study unchanged."""
from pathlib import Path
import copy,json,sys,time
import quality_benchmark as q
ROOT=Path(__file__).resolve().parent.parent

def main():
    started=time.perf_counter();out=ROOT/'results/source_ablation';out.mkdir(parents=True,exist_ok=True)
    core=json.loads((ROOT/'code/core/config.json').read_text())
    specs=[{'name':name+'_only','kind':'original','config':{'enabled_sources':[s]}} for s,name in enumerate(['workload','operating','service'])]
    q.write_json(out/'design.json',{'base_design':q.DESIGN,'source_variants':specs,'declared_after_main_quality_results':True,'comparison':'All 31 methods rescored on the same common origins; base 700 runs are not overwritten.'})
    results=[]
    for seed in q.DESIGN['seeds']:
        clean=q.load_feedback(ROOT/'results/original/runs'/f'dqn_no_rt_seed_{seed}'/'slots.csv.gz')
        reference=json.loads((ROOT/'results/original/calibration'/f'dqn_seed_{seed}'/'reference.json').read_text())
        for scenario in q.DESIGN['scenarios']:
            obs,interventions=q.corrupt_feedback(clean,scenario,seed,reference)
            targets=json.loads((ROOT/'results/quality/targets'/f'{scenario}_seed_{seed}.json').read_text())['clean_targets_for_offline_scoring_only']
            targets={int(k):v for k,v in targets.items()}
            methods={}
            for p in (ROOT/'results/quality/windows').glob(f'{scenario}_*_seed_{seed}.json'):
                win=json.loads(p.read_text());methods[win[0]['method']]=win
            for spec in specs:
                win=q.run_method(obs,core['fusion'],reference,spec,seed)
                methods[spec['name']]=win
                q.write_json(out/'windows'/f'{scenario}_{spec["name"]}_seed_{seed}.json',win)
                q.write_csv(out/'windows'/f'{scenario}_{spec["name"]}_seed_{seed}.csv',win)
            sets=[{w['window_index'] for w in win if w['window_index']>=2 and w['forecast_valid'] and targets.get(w['window_index']+1) is not None} for win in methods.values()]
            common=set.intersection(*sets)
            for name,win in methods.items():results.append(q.summarize(win,targets,common,scenario,seed))
            print(f'source-controls seed={seed} {scenario} common={len(common)} methods={len(methods)}',flush=True)
    q.write_json(out/'seed_results.json',results);q.write_csv(out/'seed_results.csv',results)
    summary=q.aggregate(results);q.write_json(out/'summary.json',summary);q.write_csv(out/'summary.csv',summary)
    q.write_json(out/'manifest.json',{'source_runs':75,'rescored_method_seed_scenario_groups':len(results),'source_windows':4500,'elapsed_seconds':time.perf_counter()-started,'code_sha256':q.sha256(Path(__file__)),'main_quality_manifest_sha256':q.sha256(ROOT/'results/quality/run_manifest.json')})
if __name__=='__main__':main()
