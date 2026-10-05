"""Independently reconstruct physical canary service and conservative gate targets.

Does not call precompute/run or use saved local_net for reconstruction. Frozen
prefix models and exogenous worlds are regenerated; served predictions, service
loss, fees, restoration and potential accuracy are counted request by request.
NumPy and the frozen engine are the only numerical dependencies.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import independent_bayes_fusion as engine

ROOT=Path(__file__).resolve().parent


def explicit_service(events,pre,rows,cfg):
    by_origin={int(r['k']):bool(r['action']) for r in rows}
    gross=0.;fees=0.;drops=0;potential=0;n=0;adm=0;restores=0
    current=None;until=-1;service=[]
    for k,e in enumerate(events):
        if current is not None and k==until:
            current=None;fees+=1.;restores+=1
        admission=by_origin.get(k,False)
        if admission:
            assert current is None,'overlapping admitted lease'
            assert k+cfg['horizon']<=len(events),'unclosed terminal lease'
            current=e['candidate'].copy();until=k+cfg['horizon'];fees+=2.;adm+=1
        model=e['reference'] if current is None else current
        prediction=np.argmax(e['x']@model,axis=1)
        common=(cfg['probe_drop'] if e['probe'] else 0)+(2 if e['refresh'] else 0)
        extra=cfg['deploy_drop'] if admission else 0
        keep=np.arange(len(prediction))>=common+extra
        correct=prediction==e['y'];window_gross=float(np.sum(correct[keep]))
        gross+=window_gross;potential+=int(correct.sum());n+=len(prediction)
        drops+=min(len(prediction),common+extra)
        if e['probe']:fees+=(pre['m']+1)*cfg['probe_steps']*cfg['step_fee']
        if e['refresh']:fees+=2.
        service.append(dict(window=k,gross=window_gross,common_drop=common,admission_drop=extra,admission=admission))
    if current is not None:
        assert until==len(events),'terminal lease incomplete'
        fees+=1.;restores+=1;current=None
    assert restores==adm,'restoration count differs from admission count'
    return dict(gross=gross,fees=fees,net=gross-fees,drops=drops,
                accuracy=potential/n,potential_correct=potential,n=n,
                deployments=adm,restorations=restores,windows=len(events))


def explicit_fork(events,row,cfg):
    origin=int(row['k']);candidate=events[origin]['candidate'];gross=0.;extra=0.;accdiff=0
    for k in range(origin,origin+cfg['horizon']):
        e=events[k]
        assert np.array_equal(e['reference'],events[origin]['reference']),'reference refreshed inside fixed fork lease'
        pc=np.argmax(e['x']@candidate,axis=1)
        pr=np.argmax(e['x']@e['reference'],axis=1)
        common=(cfg['probe_drop'] if e['probe'] else 0)+(2 if e['refresh'] else 0)
        keep=np.arange(len(pr))>=common
        gross+=float(np.sum(pc[keep]==e['y'][keep])-np.sum(pr[keep]==e['y'][keep]))
        accdiff+=int(np.sum(pc==e['y'])-np.sum(pr==e['y']))
        if k==origin:
            extra=float(np.sum(pc[common:common+cfg['deploy_drop']]==e['y'][common:common+cfg['deploy_drop']]))
    actual=gross-extra-3.;pessimistic=gross-5.;slack=2.-extra
    sd=float(row['posterior_or_block_sd']);q=float(row['q_issued'])
    standardized=(float(row['gain'])-gross)/sd
    lower=float(row['gain'])-5.-q*sd
    identity=actual-(lower-sd*(standardized-q)+slack)
    return dict(gross=gross,actual=actual,pessimistic=pessimistic,slack=slack,
                extra=extra,accdiff=accdiff,standardized=standardized,lower=lower,
                identity_error=abs(identity),covered=standardized<=q,
                covered_actual_lower_error=max(0.,lower-actual) if standardized<=q else 0.)


def load_study(data_root,result_root,name):
    protocol=json.loads((result_root/'protocol.json').read_text());spec=protocol['datasets'][name]
    if name in ('occupancy357','occupancy864'):
        data=engine.load_dataset(data_root,name,spec)
    else:
        import independent_bayes_extension as extension
        data=extension.load_mhealth(data_root/name,spec) if name=='mhealth319' else extension.load_har(data_root/name,spec)
    result=json.loads((result_root/f'{name}_results.json').read_text())
    pre=engine.prefix(data,spec,engine.BASE)
    assert data['hashes']==result['data_hashes'],'raw data hashes changed'
    assert pre['split']==result['split'],'frozen chronological split changed'
    return data,pre,result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'independent_data');parser.add_argument('--results',type=Path,default=ROOT/'new_bayes_fusion');parser.add_argument('--datasets',nargs='+',default=['occupancy357','occupancy864']);parser.add_argument('--output',type=Path,default=ROOT/'new_bayes_fusion'/'canary_execution_audit.json');args=parser.parse_args()
    report=dict(kind='independent request-level service reconstruction; no precomputed local net used',engine_sha=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=[],failures=[],datasets={},all_passed=False)
    max_service_error=0.;max_fork_error=0.;max_target_error=0.;n_forks=0;n_covered=0
    for name in args.datasets:
        data,pre,result=load_study(args.data,args.results,name)
        saved_sha=result.get('code_sha',result.get('base_code_sha'))
        assert report['engine_sha']==saved_sha,'frozen engine SHA differs'
        modes=list(result['trials'][0]['results']);report['datasets'][name]=dict(seeds=len(result['trials']),arms=len(modes),checks=0)
        for trial in result['trials']:
            seed=trial['seed'];events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,engine.BASE)
            # Reconstruction never reads trial.baseline or precompute.
            for mode,saved in trial['results'].items():
                explicit=explicit_service(events,pre,saved['rows'],engine.BASE)
                errors={key:abs(float(explicit[key])-float(saved[key])) for key in ('gross','fees','net','drops','accuracy','deployments')}
                service_error=max(errors.values());max_service_error=max(max_service_error,service_error)
                if service_error>1e-8:report['failures'].append(dict(dataset=name,seed=seed,mode=mode,kind='service',errors=errors))
                forkerr=0.;targeterr=0.;covered=0
                for row in saved['rows']:
                    actual=explicit_fork(events,row,engine.BASE);n_forks+=1
                    discrepancy=max(abs(actual['gross']-row['truegross']),abs(actual['actual']-row['local_net']))
                    forkerr=max(forkerr,discrepancy);max_fork_error=max(max_fork_error,discrepancy)
                    err=max(actual['identity_error'],abs(actual['standardized']-row['standardized_score']),actual['covered_actual_lower_error'])
                    targeterr=max(targeterr,err);max_target_error=max(max_target_error,err)
                    if actual['covered']:covered+=1;n_covered+=1
                    if not -1e-8<=actual['slack']<=2.+1e-8 or discrepancy>1e-8 or err>1e-8:
                        report['failures'].append(dict(dataset=name,seed=seed,mode=mode,kind='fork',origin=row['k'],actual=actual,local_net_saved=row['local_net'],truegross_saved=row['truegross']))
                report['checks'].append(dict(dataset=name,seed=seed,mode=mode,service_max_error=service_error,fork_max_error=forkerr,conservative_target_max_error=targeterr,forks=len(saved['rows']),covered_forks=covered,explicit=explicit))
                report['datasets'][name]['checks']+=1
            print('EXECUTION_AUDIT',name,seed,'arms',len(trial['results']),'failures',len(report['failures']),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    report.update(all_passed=not report['failures'],service_comparisons=len(report['checks']),total_forks=n_forks,covered_forks=n_covered,max_service_error=max_service_error,max_fork_error=max_fork_error,max_target_error=max_target_error)
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('checks','failures')}),flush=True)
    if report['failures']:raise SystemExit(1)

if __name__=='__main__':main()
