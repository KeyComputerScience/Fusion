"""Read-only budget validation of frozen complete-lease fusion proposals.

Reconstructs the exogenous model world, actual request/fee returns, permanent
negative-loss ledger, and pending maturity liabilities. Matrices stay in the
frozen source files; compact guarded traces retain their input-file hashes.
No guarded outcome selects or changes a controller.
"""
from __future__ import annotations
import argparse,datetime,gzip,hashlib,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'work/fusion_focus_20261003/controls'))
import run_strong_controls as support
ARMS=('precision_joint','precision_diagonal','scalar_mass','no_posterior','complete_only')
BUDGETS=(0.,110.,130.,260.)
RESERVES=('fixed130','decision')
CORE_SHA='811f98531ad3e135ceafbc64152b41ffbf380878cb45a1f7106eb54364157ff0'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def relative(path):
    try:return Path(path).resolve().relative_to(PROJECT).as_posix()
    except ValueError:return str(Path(path).resolve())

def adapt_study(engine,cfg,old_study,name,spec):
    if name not in ('har780','rss348','pamap231'):return old_study(name,spec)
    path=PROJECT/'work/fusion_focus_20261003/fresh/new_data_adapter.py'
    loading=importlib.util.spec_from_file_location('budget_cache_adapter',path)
    adapter=importlib.util.module_from_spec(loading);loading.loader.exec_module(adapter)
    folder=(PROJECT/'work/fusion_focus_20261003/fresh/data' if name=='har780' else ROOT/'physical/data')
    data,derivation=adapter.load_dataset(folder,name)
    return data,adapter.make_prefix(data,derivation,cfg,engine)

def partial_lease_return(events,origin,cfg):
    """Independent request/fee prefix minimum for one immutable paid lease."""
    candidate=events[origin]['candidate'];reference=events[origin]['reference']
    increment=-2.;minimum=increment
    for k in range(origin,origin+cfg['horizon']):
        e=events[k]
        assert np.array_equal(reference,e['reference'])
        pc=np.argmax(e['x']@candidate,axis=1);pr=np.argmax(e['x']@reference,axis=1)
        common=(cfg['probe_drop'] if e['probe'] else 0)+(2 if e['refresh'] else 0)
        extra=cfg['deploy_drop'] if k==origin else 0
        for request in range(len(pc)):
            increment+=int(request>=common+extra and pc[request]==e['y'][request])-int(request>=common and pr[request]==e['y'][request])
            minimum=min(minimum,increment)
    increment-=1.;minimum=min(minimum,increment)
    return increment,minimum

def summarize(trajectories):
    output={}
    for name in sorted({r['task'] for r in trajectories}):
        output[name]={}
        for arm in ARMS:
            output[name][arm]={}
            for reserve in RESERVES:
                output[name][arm][reserve]={}
                for budget in BUDGETS:
                    rows=[r for r in trajectories if r['task']==name and r['arm']==arm and r['reserve_mode']==reserve and r['budget']==budget]
                    assert len(rows)==5
                    output[name][arm][reserve][str(int(budget))]=dict(
                        mean_net=float(np.mean([r['net'] for r in rows])),
                        mean_increment=float(np.mean([r['increment'] for r in rows])),
                        harmful=sum(r['harmful'] for r in rows),beneficial=sum(r['beneficial'] for r in rows),
                        admissions=sum(r['admissions'] for r in rows),refused=sum(r['refused'] for r in rows),
                        maximum_completed_negative_loss=max(r['negative_loss'] for r in rows),
                        minimum_service_prefix_increment=min(r['minimum_prefix_increment'] for r in rows),
                        seed_values=[r['increment'] for r in rows])
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'verification.json').exists(),'Use a new output directory for a rerun'
    engine,recovery,guard,source,cfgbase,old_study,service,fork=support.setup(support.DEFAULT_PACKAGE,support.DEFAULT_GUARD)
    scientific_paths=[ROOT/'temporal_fusion.py',support.DEFAULT_GUARD/'run_loss_budget_guard.py',
                      support.DEFAULT_PACKAGE/'base/audit_canary_execution.py',
                      support.DEFAULT_PACKAGE/'base/independent_bayes_fusion.py',
                      PROJECT/'work/fusion_focus_20261003/fresh/new_data_adapter.py']
    before={relative(p):sha(p) for p in scientific_paths}
    assert before[relative(ROOT/'temporal_fusion.py')]==CORE_SHA
    inputs=[];task_bindings={}
    for folder in args.inputs:
        folder=folder.resolve();protocol=json.loads((folder/'protocol.json').read_text())
        assert protocol['code_sha256']==CORE_SHA
        assert tuple(protocol['arms'])==ARMS
        for name in protocol['tasks']:
            assert name not in task_bindings
            result_path=folder/(name+'_results.json');selection_path=folder/(name+'_selection.json')
            assert result_path.exists() and selection_path.exists(),f'Frozen results not ready: {name}'
            task_bindings[name]=(folder,protocol,result_path,selection_path)
            inputs.append(dict(task=name,protocol=relative(folder/'protocol.json'),protocol_sha=sha(folder/'protocol.json'),
                               results=relative(result_path),results_sha=sha(result_path),
                               selection=relative(selection_path),selection_sha=sha(selection_path)))
    declaration=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     script_sha256=sha(__file__),core_sha256=CORE_SHA,
                     tasks=list(task_bindings),arms=list(ARMS),budgets=list(BUDGETS),reserves=list(RESERVES),
                     mode='permanent_gross_negative_lease_loss_no_profit_refill',
                     scope='guard validation of existing frozen proposals; no controller selection',
                     inputs=inputs,scientific_source_hashes=before)
    dump(out/'declaration.json',declaration)
    report=dict(service_trajectories=0,input_service_checks=0,reference_checks=0,fork_checks=0,
                future_truth_checks=0,causal_checks_with_unresolved_admissions=0,
                max_service_error=0.,max_fork_error=0.,max_reference_error=0.,
                min_ledger_capacity=0.,max_loss_violation=0.,max_prefix_violation=0.,
                max_partial_lease_reserve_violation=0.,partial_lease_reserve_checks=0,maximum_pending_leases=0,
                maximum_test_end_pending_leases=0,tasks={})
    trajectories=[];binding=[];line_hash=hashlib.sha256();line_count=0
    with (out/'guarded_traces.jsonl.gz').open('wb') as raw_file:
      with gzip.GzipFile(fileobj=raw_file,mode='wb',mtime=0) as compressed:
        for name,(folder,protocol,result_path,selection_path) in task_bindings.items():
            results=json.loads(result_path.read_text());selection=json.loads(selection_path.read_text())
            cfgbase=dict(protocol['cfg']);spec=source['studies'].get(name,{})
            data,pre=adapt_study(engine,cfgbase,old_study,name,spec)
            assert data['hashes']==selection['data_hashes']
            assert pre['split']==selection['split']
            assert results['selected']==selection['selected']
            assert len(results['trials'])==5
            start_count=report['service_trajectories']
            for trial in results['trials']:
                seed=trial['seed'];events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfgbase)
                reference=service(events,pre,[],cfgbase)
                referr=abs(reference['net']-trial['results']['reference']['net'])
                report['reference_checks']+=1;report['max_reference_error']=max(report['max_reference_error'],referr)
                assert referr<1e-8
                for arm in ARMS:
                    saved=trial['results'][arm];selected=selection['selected'][arm]
                    cfg=dict(cfgbase,prior_sd=selected['prior_sd'],q_floor=selected['q_floor'],threshold=selected['margin'])
                    rows=[{k:r[k] for k in ('k','action','maturity','gain','posterior_or_block_sd','q_issued',
                                            'gate_score','standardized_score','truegross','local_net','lower_covered')}
                          for r in saved['rows']]
                    original_service=service(events,pre,rows,cfg)
                    service_error=max(abs(float(original_service[k])-float(saved[k])) for k in ('net','gross','fees','drops','accuracy','deployments'))
                    report['input_service_checks']+=1;report['max_service_error']=max(report['max_service_error'],service_error)
                    assert service_error<1e-8
                    decision={};reserve_logs={};actuals=[]
                    for row in rows:
                        k=int(row['k']);e=events[k]
                        assert row['maturity']>=k+cfg['horizon'],'Settlement precedes service closure'
                        common=(cfg['probe_drop'] if e['probe'] else 0)+(2 if e['refresh'] else 0)
                        assert common>=1,'Fixed130 requires at least one common origin preparation drop'
                        reserve,detail=guard.current_decision_reserve(e['x'],e['candidate'],e['reference'],common,cfg)
                        assert reserve<=130.
                        decision[k]=reserve;reserve_logs[k]=detail
                        actual=fork(events,row,cfg)
                        terminal,partial_minimum=partial_lease_return(events,k,cfg)
                        discrepancy=max(abs(actual['actual']-row['local_net']),abs(actual['gross']-row['truegross']),actual['identity_error'])
                        discrepancy=max(discrepancy,abs(terminal-actual['actual']))
                        report['fork_checks']+=1;report['max_fork_error']=max(report['max_fork_error'],discrepancy)
                        assert discrepancy<1e-8
                        assert actual['actual']>=-reserve-1e-8
                        for bound in (reserve,130.):
                            violation=max(0.,-bound-partial_minimum)
                            report['partial_lease_reserve_checks']+=1
                            report['max_partial_lease_reserve_violation']=max(report['max_partial_lease_reserve_violation'],violation)
                            assert violation<1e-8
                        actuals.append(actual['actual'])
                    values=np.asarray(actuals)
                    for reserve_mode in RESERVES:
                      reserves={int(r['k']):130. for r in rows} if reserve_mode=='fixed130' else decision
                      for budget in BUDGETS:
                        guarded=guard.replay(rows,len(events),budget,'gross_loss',reserves)
                        independent=service(events,pre,guarded['rows'],cfg)
                        prefix=guard.reconstruct_prefix_increment(events,guarded['rows'],cfg)
                        chosen=np.array([r['action'] for r in guarded['rows']],bool)
                        increment=independent['net']-reference['net']
                        loss=float(np.maximum(-values[chosen],0).sum())
                        discrepancy=max(abs(increment-float(values[chosen].sum())),abs(increment-prefix['final']),
                                        abs(increment-guarded['final_settled_increment']),abs(loss-guarded['final_spent_loss']))
                        report['service_trajectories']+=1;report['max_service_error']=max(report['max_service_error'],discrepancy)
                        report['maximum_test_end_pending_leases']=max(report['maximum_test_end_pending_leases'],len(guarded['test_end_state']['pending_origins']))
                        for ledger in guarded['ledger']:
                            report['min_ledger_capacity']=min(report['min_ledger_capacity'],ledger['capacity_after'])
                            report['maximum_pending_leases']=max(report['maximum_pending_leases'],len(ledger['pending_origins']))
                            assert ledger['spent_loss']+ledger['reserved']<=budget+1e-8
                            assert all(c['maturity']<=ledger['k'] for c in ledger['arrived'])
                        report['max_loss_violation']=max(report['max_loss_violation'],max(0.,loss-budget),max(0.,prefix['maximum_prefix_negative_loss']-budget))
                        report['max_prefix_violation']=max(report['max_prefix_violation'],max(0.,-budget-prefix['minimum']))
                        assert discrepancy<1e-8 and loss<=budget+1e-8 and prefix['minimum']>=-budget-1e-8
                        assert prefix['maximum_prefix_negative_loss']<=budget+1e-8
                        if budget==0:assert not chosen.any()
                        for origin,increment_of_lease in prefix['independent_lease_increments'].items():
                            assert increment_of_lease>=-reserves[int(origin)]-1e-8
                        for cutoff in sorted({0,len(events)//2,max(0,len(events)-1)}):
                            poison=guard.replay(rows,len(events),budget,'gross_loss',reserves,perturb_after=cutoff)
                            assert [r['action'] for r in guarded['rows'] if r['k']<=cutoff]==[r['action'] for r in poison['rows'] if r['k']<=cutoff]
                            report['future_truth_checks']+=1
                            if any(r['action'] and r['k']<=cutoff<r['maturity'] for r in guarded['rows']):
                                report['causal_checks_with_unresolved_admissions']+=1
                        compact=dict(task=name,seed=seed,arm=arm,budget=budget,reserve_mode=reserve_mode,
                                     net=independent['net'],reference_net=reference['net'],increment=increment,
                                     unguarded_increment=saved['net']-reference['net'],negative_loss=loss,
                                     admissions=int(chosen.sum()),harmful=int(np.sum(values[chosen]<0)),beneficial=int(np.sum(values[chosen]>0)),
                                     refused=sum(r['proposed_action'] and not r['action'] for r in guarded['rows']),
                                     actions=chosen.tolist(),origins=[r['k'] for r in rows],admitted_returns=values[chosen].tolist(),
                                     minimum_prefix_increment=prefix['minimum'],maximum_prefix_negative_loss=prefix['maximum_prefix_negative_loss'],
                                     accounting_error=discrepancy,test_end_state=guarded['test_end_state'])
                        trajectories.append(compact)
                        full=dict(**compact,ledger=guarded['ledger'],prefix_window_trace=prefix['trace'],
                                  independent_lease_increments=prefix['independent_lease_increments'],independent_service=independent,
                                  decision_reserves=reserve_logs)
                        encoded=canonical(full)+b'\n';compressed.write(encoded);line_hash.update(encoded);line_count+=1
                        for row in guarded['rows']:
                            if row['proposed_action'] and not row['action']:
                                binding.append(dict(task=name,seed=seed,arm=arm,budget=budget,reserve_mode=reserve_mode,
                                                    origin=row['k'],actual_complete_increment=row['local_net'],reserve=reserves[row['k']]))
                print('BUDGET',name,seed,'complete',flush=True)
            report['tasks'][name]=dict(guarded_trajectories=report['service_trajectories']-start_count,seeds=5)
    after={relative(p):sha(p) for p in scientific_paths}
    report['scientific_sources_unchanged']=before==after
    report['input_results_unchanged']=all(sha(PROJECT/i['results'])==i['results_sha'] and sha(PROJECT/i['selection'])==i['selection_sha'] for i in inputs)
    report['compact_jsonl_lines']=line_count;report['uncompressed_jsonl_sha256']=line_hash.hexdigest()
    report['compressed_jsonl_sha256']=sha(out/'guarded_traces.jsonl.gz')
    report['passed']=bool(report['max_service_error']<1e-8 and report['max_fork_error']<1e-8
                          and report['max_reference_error']<1e-8 and report['min_ledger_capacity']>=-1e-8
                          and report['max_loss_violation']==0 and report['max_prefix_violation']==0
                          and report['max_partial_lease_reserve_violation']==0
                          and report['scientific_sources_unchanged'] and report['input_results_unchanged'])
    dump(out/'results.json',trajectories);dump(out/'summary.json',summarize(trajectories))
    dump(out/'binding_actions.json',binding);dump(out/'verification.json',report)
    assert report['passed'],report
    print('COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
