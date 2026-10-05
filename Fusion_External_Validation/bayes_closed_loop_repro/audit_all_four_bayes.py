"""Independent per-service-window replay of all four frozen policy experiments."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import independent_bayes_fusion as engine
import independent_bayes_extension as adapter
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'independent_data'


def actual_service(events,pre,cfg,actions):
    action_by={4*i:action for i,action in enumerate(actions)};lease=None;expires=-1;gross=0.;fees=0.;drops=0;correct=0;n=0
    for k,e in enumerate(events):
        if k>=expires:lease=None
        deploy=e['reference'] if lease is None else lease
        if action_by.get(k,False):lease=e['candidate'].copy();expires=k+4;deploy=lease
        pred=(e['x']@deploy).argmax(1);cd=(1 if e['probe'] else 0)+(2 if e['refresh'] else 0)+(2 if action_by.get(k,False) else 0);sel=np.arange(len(pred))>=cd
        gross+=int(np.sum(pred[sel]==e['y'][sel]));fees+=(pre['m']+1)*20*.001 if e['probe'] else 0.;fees+=2. if e['refresh'] else 0.;fees+=3. if action_by.get(k,False) else 0.;drops+=min(len(pred),cd);correct+=int(np.sum(pred==e['y']));n+=len(pred)
    return dict(net=gross-fees,gross=gross,fees=fees,drops=drops,accuracy=correct/n)


def actual_fork(events,k):
    current=events[k];ref=current['reference'];candidate=current['candidate'];net=-3.;truegross=0.;extra=0.
    for j in range(k,k+4):
        e=events[j];pr=(e['x']@ref).argmax(1);pc=(e['x']@candidate).argmax(1);cd=(1 if e['probe'] else 0)+(2 if e['refresh'] else 0);rkeep=np.arange(len(pr))>=cd;ckeep=np.arange(len(pc))>=cd+(2 if j==k else 0)
        net+=int(np.sum(pc[ckeep]==e['y'][ckeep])-np.sum(pr[rkeep]==e['y'][rkeep]));truegross+=int(np.sum(pc[rkeep]==e['y'][rkeep])-np.sum(pr[rkeep]==e['y'][rkeep]))
        if j==k:extra=int(np.sum(pc[cd:cd+2]==e['y'][cd:cd+2]))
    return dict(net=net,truegross=truegross,opportunity_slack=2-extra,candidate_extra=extra)


def main():
    old=json.loads((ROOT/'new_bayes_fusion/results.json').read_text());new=json.loads((ROOT/'new_bayes_extension/results.json').read_text());oldprot=json.loads((ROOT/'new_bayes_fusion/protocol.json').read_text());newprot=json.loads((ROOT/'new_bayes_extension/protocol.json').read_text());results={};service_count=0;fork_count=0;unique_count=0;maxservice=0.;maxidentity=0.;maxkkt=0.
    for name,study in {**old,**new}.items():
        if name in old:
            data=engine.load_dataset(DATA,name,oldprot['datasets'][name]);spec=oldprot['datasets'][name]
        else:
            spec=newprot['datasets'][name];data=adapter.load_mhealth(DATA/name,spec) if name=='mhealth319' else adapter.load_har(DATA/name,spec)
        pre=engine.prefix(data,spec,engine.BASE);checks=[];pairs={}
        for trial in study['trials']:
            events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,trial['seed'],engine.BASE);forks={k:actual_fork(events,k) for k in range(0,len(events)-3,4)};unique_count+=len(forks)
            for mode,out in trial['results'].items():
                observed=actual_service(events,pre,engine.BASE,out['actions']);err=max(abs(observed[k]-out[k]) for k in observed);maxservice=max(maxservice,err);service_count+=1
                assert err<1e-8
                for row in out['rows']:
                    f=forks[row['k']];assert abs(row['local_net']-f['net'])<1e-8;assert abs(row['truegross']-f['truegross'])<1e-8;assert -1e-8<=f['opportunity_slack']<=2+1e-8
                    if 'opportunity_slack' in row:assert abs(row['opportunity_slack']-f['opportunity_slack'])<1e-8
                    assert all(w>0 for w in row['weights']);assert abs(sum(row['weights'])-1)<1e-8;assert max(row['weights'])<=max(.8,1/len(row['weights']))+1e-8;fork_count+=1
                for q in out['q_updates']:assert q['issued_index']<=q['maturity']<=q['update_index']
                identity=sum(q['violation'] for q in out['q_updates'])-(.1*len(out['q_updates'])+(out['q_final']-out['q_initial']-sum(q['regulator'] for q in out['q_updates']))/.05);maxidentity=max(maxidentity,abs(identity));assert abs(identity)<1e-8;assert out['solver_failures']==0;maxkkt=max(maxkkt,out['max_kkt'])
                if mode=='posterior_unused':assert out['actions']==trial['results']['joint']['actions'] and abs(out['net']-trial['results']['joint']['net'])<1e-8
                checks.append(dict(seed=trial['seed'],mode=mode,service_error=err,calibration_identity_error=identity,leases=len(out['rows']),q_callbacks=len(out['q_updates']),causal_maturities_passed=True))
        for alt in study['trials'][0]['results']:
            delta=[t['results']['bayes_both']['net']-t['results'][alt]['net'] for t in study['trials']];mean=statistics.mean(delta);se=statistics.stdev(delta)/math.sqrt(5);pairs[alt]=dict(delta=delta,mean=mean,t4_interval=[mean-2.7764451051977987*se,mean+2.7764451051977987*se])
        results[name]=dict(split=study['split'],checks=checks,paired_vs_full=pairs)
    output=dict(passed=True,base_code_sha=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),adapter_sha=hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),policy_trajectories=service_count,policy_fork_checks=fork_count,unique_scenario_forks=unique_count,max_independent_service_error=maxservice,max_calibration_identity_error=maxidentity,max_kkt=maxkkt,results=results)
    (ROOT/'new_bayes_extension/all_four_integrity_audit.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:v for k,v in output.items() if k!='results'},indent=2))

if __name__=='__main__':main()
