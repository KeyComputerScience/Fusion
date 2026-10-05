"""Retained diagnostic delay replays of four byte-frozen multisensor controllers."""
from __future__ import annotations
import datetime,hashlib,json,math,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent.parent/'outputs'/'bayes_closed_loop_repro'
sys.path.insert(0,str(PACKAGE))
from cached_runner import install_cached_loaders
engine,adapter=install_cached_loaders()
from audit_canary_execution import load_study,explicit_service,explicit_fork

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def paired(values):
    a=np.asarray(values,dtype=float);mean=float(a.mean());half=2.776445105*float(a.std(ddof=1))/math.sqrt(len(a))
    return dict(mean=mean,ci95_t4=[mean-half,mean+half],positive=int(np.sum(a>1e-9)),negative=int(np.sum(a< -1e-9)),ties=int(np.sum(abs(a)<=1e-9)),values=a.tolist())

def execute_audit(events,stream,pre,cfg,mode,result):
    physical=explicit_service(events,pre,result['rows'],cfg)
    discrepancies={k:abs(float(physical[k])-float(result[k])) for k in ('gross','fees','net','drops','accuracy','deployments')}
    service_error=max(discrepancies.values());fork_error=0.;target_error=0.;ncovered=0;max_simplex=0.;max_cap=0.;ncalibration=0
    lookup={int(d['k']):d for d in stream['decisions']}
    for row in result['rows']:
        physical_fork=explicit_fork(events,row,cfg)
        fork_error=max(fork_error,abs(physical_fork['gross']-row['truegross']),abs(physical_fork['actual']-row['local_net']))
        target_error=max(target_error,physical_fork['identity_error'],physical_fork['covered_actual_lower_error'],abs(physical_fork['standardized']-row['standardized_score']))
        ncovered+=int(physical_fork['covered']);w=np.asarray(row['weights']);max_simplex=max(max_simplex,abs(float(w.sum())-1.),max(0.,-float(w.min())));cap=max(cfg['cap'],1/len(w));max_cap=max(max_cap,max(0.,float(w.max())-cap))
    for q in result['q_updates']:
        assert q['maturity']<=q['update_index'],'future calibration feedback consumed'
        assert q['issued_index']<q['maturity'],'calibration target matured before issue'
        ncalibration+=1
    moment_identity=max(abs(float(d['identity_error'])) for d in stream['decisions']) if stream['decisions'] else 0.
    passed=(service_error<1e-8 and fork_error<1e-8 and target_error<1e-8 and abs(result['closed_loop_identity_error'])<1e-8 and abs(result['calibration_identity_error'])<1e-8 and result['solver_failures']==0 and result['max_kkt']<=1e-6 and max_simplex<1e-7 and max_cap<1e-7 and moment_identity<1e-8)
    return dict(all_passed=bool(passed),mode=mode,explicit_service=physical,service_errors=discrepancies,max_service_error=service_error,max_fork_error=fork_error,max_target_identity_error=target_error,closed_loop_identity_error=result['closed_loop_identity_error'],calibration_identity_error=result['calibration_identity_error'],posterior_moment_identity_error=moment_identity,solver_failures=result['solver_failures'],max_kkt=result['max_kkt'],max_simplex_error=max_simplex,max_cap_error=max_cap,causal_q_updates=ncalibration,forks=len(result['rows']),covered_forks=ncovered)

def summarize(trials,modes):
    out={}
    for mode in modes:
        rows=[t['results'][mode] for t in trials]
        out[mode]=dict(mean_net=float(np.mean([r['net'] for r in rows])),mean_deployments=float(np.mean([r['deployments'] for r in rows])),mean_harmful=float(np.mean([r['harmful'] for r in rows])),mean_beneficial=float(np.mean([r['beneficial'] for r in rows])),mean_accuracy=float(np.mean([r['accuracy'] for r in rows])),mean_coverage=float(np.mean([r['coverage'] for r in rows])),mean_informative_coverage=float(np.mean([r['informative_coverage'] for r in rows if r['informative_coverage'] is not None])),solver_failures=sum(r['solver_failures'] for r in rows),max_kkt=max(r['max_kkt'] for r in rows))
    pairs={}
    for arm in ('bayes_gate','bayes_both'):
        for control in ('joint','bayes_gate','frequentist_gate'):
            if arm==control or arm not in modes or control not in modes:continue
            differences=[dict(seed=t['seed'],net=t['results'][arm]['net']-t['results'][control]['net'],deployments=t['results'][arm]['deployments']-t['results'][control]['deployments'],harmful=t['results'][arm]['harmful']-t['results'][control]['harmful'],coverage=t['results'][arm]['coverage']-t['results'][control]['coverage'],informative_coverage=None if t['results'][arm]['informative_coverage'] is None or t['results'][control]['informative_coverage'] is None else t['results'][arm]['informative_coverage']-t['results'][control]['informative_coverage'],action_disagreements=int(np.sum(np.asarray(t['results'][arm]['actions'])!=np.asarray(t['results'][control]['actions'])))) for t in trials]
            pairs[f'{arm}_minus_{control}']=dict(net=paired([v['net'] for v in differences]),deployments=paired([v['deployments'] for v in differences]),harmful=paired([v['harmful'] for v in differences]),coverage=paired([v['coverage'] for v in differences]),mean_informative_coverage_difference=float(np.mean([v['informative_coverage'] for v in differences if v['informative_coverage'] is not None])),mean_action_disagreements=float(np.mean([v['action_disagreements'] for v in differences])),rows=differences)
    return dict(controllers=out,paired=pairs)

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());freeze=json.loads((ROOT/'pre_replay_freeze.json').read_text());assert sha(ROOT/'protocol.json')==freeze['protocol_sha256'],'protocol changed after freeze'
    assert sha(PACKAGE/'independent_bayes_fusion.py')==freeze['base_engine_sha256'];assert sha(PACKAGE/'independent_bayes_extension.py')==freeze['adapter_sha256']
    modes=protocol['controllers'];all_results={};audits=[];bench=None
    for name in protocol['datasets']:
        entry=protocol['source_studies'][name];location=PACKAGE/entry['folder'];assert sha(location/f'{name}_results.json')==entry['result_sha256']
        t0=time.perf_counter();data,pre,saved=load_study(PACKAGE/'independent_data',location,name);load_seconds=time.perf_counter()-t0
        assert data['hashes']==entry['data_hashes'];assert pre['split']==entry['split'];assert saved['selected']==json.loads((location/f'{name}_results.json').read_text())['selected']
        trials=[]
        for seed in protocol['delay_seeds']:
            start=time.perf_counter();events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,engine.BASE);stream=engine.precompute(events,pre,engine.BASE);world_seconds=time.perf_counter()-start;results={};checks={}
            for mode in modes:
                conf=entry['controllers'][mode];assert conf['config']==saved['selected'][mode];assert conf['q_initial']==saved['q_initial'][mode]
                results[mode]=engine.run(stream,pre,dict(conf['config']),mode,float(conf['q_initial']))
                checks[mode]=execute_audit(events,stream,pre,conf['config'],mode,results[mode]);audits.append(dict(dataset=name,seed=seed,**checks[mode]));assert checks[mode]['all_passed'],f'execution audit failed {name} {seed} {mode}'
            elapsed=time.perf_counter()-start
            if bench is None:
                bench=dict(dataset=name,seed=seed,load_seconds=load_seconds,world_precompute_seconds=world_seconds,all_arm_run_audit_seconds=elapsed-world_seconds,total_seed_seconds=elapsed,conservative_20_trial_estimate_seconds=20*elapsed+4*load_seconds,arm_set_retained=modes)
                (ROOT/'first_seed_benchmark.json').write_text(json.dumps(bench,indent=2));print('BENCHMARK',bench,flush=True)
                assert bench['conservative_20_trial_estimate_seconds']<=1200,'Runtime exceeded protocol benchmark threshold; do not silently reduce arms after reading results'
            trials.append(dict(seed=seed,windows=stream['windows'],episodes=len(stream['decisions']),baseline=dict(gross=stream['basegross'],fees=stream['commonfees'],drops=stream['commondrops'],net=stream['basegross']-stream['commonfees']),results=results,execution_audits=checks,elapsed_seconds=elapsed))
            print('DELAY_EXTENSION',name,seed,{m:(round(r['net'],6),r['deployments'],r['harmful'],r['coverage']) for m,r in results.items()},flush=True)
            (ROOT/f'{name}_results_partial.json').write_text(json.dumps(dict(dataset=name,trials=trials),indent=2))
        out=dict(kind=protocol['kind'],dataset=name,protocol_sha256=freeze['protocol_sha256'],source_result_sha256=entry['result_sha256'],engine_sha256=freeze['base_engine_sha256'],adapter_sha256=freeze['adapter_sha256'],data_hashes=data['hashes'],split=pre['split'],selected={m:entry['controllers'][m]['config'] for m in modes},q_initial={m:entry['controllers'][m]['q_initial'] for m in modes},trials=trials,summary=summarize(trials,modes));all_results[name]=out;(ROOT/f'{name}_results.json').write_text(json.dumps(out,indent=2));(ROOT/'summary.json').write_text(json.dumps({k:v['summary'] for k,v in all_results.items()},indent=2))
    original=json.loads((ROOT/'original_files_before.json').read_text());after={str(p.relative_to(PACKAGE)):sha(p) for p in PACKAGE.rglob('*') if p.is_file()};changed=[name for name,h in original.items() if after.get(name)!=h];new_files=sorted(set(after)-set(original))
    execution=dict(kind='independent request-level physical reconstruction on supplementary fixed-controller delay trials',all_passed=all(a['all_passed'] for a in audits) and not changed and not new_files,protocol_sha256=freeze['protocol_sha256'],protocol_created_utc=freeze['created_utc'],runner_sha256=sha(__file__),engine_sha256=sha(PACKAGE/'independent_bayes_fusion.py'),adapter_sha256=sha(PACKAGE/'independent_bayes_extension.py'),numpy_version=np.__version__,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_package_files_unchanged=not changed and not new_files,changed_original_files=changed,new_package_files=new_files,bench=bench,checks=audits,max_service_error=max(a['max_service_error'] for a in audits),max_fork_error=max(a['max_fork_error'] for a in audits),max_target_identity_error=max(a['max_target_identity_error'] for a in audits),max_closed_loop_identity_error=max(abs(a['closed_loop_identity_error']) for a in audits),max_calibration_identity_error=max(abs(a['calibration_identity_error']) for a in audits),max_posterior_moment_identity_error=max(abs(a['posterior_moment_identity_error']) for a in audits),max_kkt=max(a['max_kkt'] for a in audits),solver_failures=sum(a['solver_failures'] for a in audits),service_comparisons=len(audits),forks=sum(a['forks'] for a in audits))
    (ROOT/'results.json').write_text(json.dumps(all_results,indent=2));(ROOT/'execution_audit.json').write_text(json.dumps(execution,indent=2));print('COMPLETE',dict(all_passed=execution['all_passed'],service_comparisons=len(audits),max_kkt=execution['max_kkt'],package_unchanged=execution['original_package_files_unchanged']),flush=True);assert execution['all_passed']

if __name__=='__main__':main()
