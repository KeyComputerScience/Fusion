"""Published PDF/QMF fixed-backbone rules with the same complete-lease feedback gate."""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));import run_recovery_fusion as recovery
sys.path.insert(0,str(ROOT.parent/'external_fusion_validation_20261002'));import run_external_pdf as pdf
ARMS=('pdf_point','pdf_calibrated','qmf_point','qmf_calibrated')

def qmf(p):
    if np.any(p<=0):raise ValueError('raw-logit recovery needs positive probabilities')
    z=np.log(p);z-=z.mean(2,keepdims=True);mx=z.max(2);energy=mx+np.log(np.exp(z-mx[:,:,None]).sum(2));v=.1*energy;return pdf.softmax(np.einsum('sn,snc->nc',v,z),1),v.T

def stream_for(events,pre,base,heads,rule,cfg):
    pending=[];arrived=[];rows=[];pp=dict(pre,m=1)
    for d in base['decisions']:
        k=d['k'];e=events[k];arrived.extend([r for r in pending if r['maturity']<=k]);pending=[r for r in pending if r['maturity']>k];arrived=arrived[-cfg['lease_archive']:];ids=d['ids']
        if rule=='pdf':pf,weights,tcp=pdf.pdf_fuse(e['p'][ids],heads,ids)
        else:pf,weights=qmf(e['p'][ids]);tcp=None
        current=dict(d,ids=np.array([0]),context=e['context'],candidate=e['candidate'],reference=e['reference']);eligible=[r for r in arrived if r['physical_mask'][ids].all()];mm=recovery.lease_moments(eligible,current,pp,cfg)
        pc=(e['x']@e['candidate']).argmax(1);pr=(e['x']@e['reference']).argmax(1);b=np.eye(pre['classes'])[pc]-np.eye(pre['classes'])[pr];h=float(np.mean(np.einsum('ic,ic->i',pf,b)))
        row=dict(d,external=dict(h=h,mu=float(mm['mu'][0]),B=float(mm['B'][0,0]),U=float(mm['U'][0,0]),mass=mm['mass'],blocks=mm['blocks'],eligible=mm['eligible'],source_influences=weights.tolist(),tcp=None if tcp is None else tcp.tolist()))
        future=[]
        for j in range(k,k+cfg['horizon']):
            f=events[j];cd=(cfg['probe_drop'] if f['probe'] else 0)+(2 if f['refresh'] else 0);future.append(dict(x=f['x'],y=f['y'],keep=np.arange(len(f['x']))>=cd))
        pending.append(dict(k=k,maturity=d['maturity'],context=e['context'],x=e['x'],p=pf[None].copy(),mask=np.ones(1,bool),physical_mask=e['mask'],future=future,N=d['N']));rows.append(row)
    return dict(base,decisions=rows)

def forecast(d,pre,cfg,arm):
    a=d['external'];gain=d['N']*(a['h']-a['mu']);sd=d['N']*math.sqrt(a['B']+cfg['gate_floor']);return np.ones(1),dict(kkt=0.,primal=0.,converged=True,iterations=0),gain,sd,(gain-d['truegross'])/sd

def run(engine,stream,pre,cfg,arm,qinit,selection=False):
    old=engine.decision_forecast;engine.decision_forecast=lambda d,p,c,m:forecast(d,p,c,arm)
    try:r=engine.run(stream,pre,cfg,'joint' if arm.endswith('point') else 'bayes_gate',qinit,selection)
    finally:engine.decision_forecast=old
    for row,d in zip(r['rows'],stream['decisions']):row.update(d['external']);assert all(z['maturity']<=row['k'] for z in row['eligible'])
    r['external_arm']=arm;r['optimizer_applicability']=False;return r

def qi(streams,pre,cfg,arm):
    scores=[]
    for s in streams:
        for d in s['decisions']:
            if d['maturity']<=s['windows']//2:scores.append(forecast(d,pre,cfg,arm)[4])
    return max(0.,float(np.quantile(scores,.9,method='higher'))) if scores else 0.,scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=('calibrate','test'));p.add_argument('--package',type=Path,default=ROOT.parent.parent/'outputs'/'Fusion_External_Validation_Repro'/'bayes_closed_loop_repro');p.add_argument('--output',type=Path,default=ROOT/'external');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.package));from cached_runner import install_cached_loaders
    engine,adapter=install_cached_loaders();from audit_canary_execution import load_study,explicit_service,explicit_fork;from new_data_adapter import load_dataset,make_prefix
    protocol=json.loads((ROOT/'protocol.json').read_text());ep=json.loads((out/'protocol.json').read_text());cfgbase=dict(engine.BASE,**protocol['extension']);code_sha=recovery.sha(__file__);helper_sha=recovery.sha(ROOT/'run_recovery_fusion.py');psha=recovery.sha(out/'protocol.json')
    def study(name,entry):
        if entry.get('new_task'):
            data,spec=load_dataset(ROOT/'data',name);return data,make_prefix(data,spec,cfgbase,engine)
        data,pre,saved=load_study(a.package/'independent_data',a.package/entry['folder'],name);return data,pre
    if a.phase=='calibrate':
        assert not (out/'prefix_freeze.json').exists();recovery.dump(out/'pre_calibration_freeze.json',dict(utc=recovery.now(),runner_sha=code_sha,recovery_helper_sha=helper_sha,protocol_sha=psha,source_rule_helper_sha=recovery.sha(Path(pdf.__file__)),data_used_for_external_test_before_fitting=False,primary_recovery_test_results_already_exist=True));hashes={}
        for name,entry in protocol['studies'].items():
            data,pre=study(name,entry);ev=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,cfgbase) for s in entry.get('calibration_seeds',protocol['calibration_seeds'])];base=[engine.precompute(e,pre,cfgbase) for e in ev];grid=[];heads_by={};fits={}
            for multiplier in (0.,.25,1.):
                heads,fit=pdf.fit_tcp(pre,multiplier,ep);heads_by[str(multiplier)]=heads.tolist();fits[str(multiplier)]=fit
                pdfstreams=[stream_for(e,pre,b,heads,'pdf',cfgbase) for e,b in zip(ev,base)];qmfstreams=[stream_for(e,pre,b,None,'qmf',cfgbase) for e,b in zip(ev,base)]
                for margin in (0.,4.,12.):
                    cfg=dict(cfgbase,threshold=margin)
                    for arm in ARMS:
                        ss=pdfstreams if arm.startswith('pdf') else qmfstreams;init,scores=qi(ss,pre,cfg,arm);q0=init if arm.endswith('calibrated') else 0.;rs=[run(engine,s,pre,cfg,arm,q0,True) for s in ss];grid.append(dict(arm=arm,l2_multiplier=multiplier,inactive_qmf_slot=multiplier if arm.startswith('qmf') else None,margin=margin,q_initial=q0,q_fit_scores=scores if arm.endswith('calibrated') else [],mean_selection_net=float(np.mean([r['selection_net'] for r in rs])),mean_selection_actions=float(np.mean([r['deployments'] for r in rs]))))
                    print('CAL_EXTERNAL',name,multiplier,margin,flush=True)
            selected={arm:max((r for r in grid if r['arm']==arm),key=lambda r:(r['mean_selection_net'],-r['margin'],-r['l2_multiplier'])) for arm in ARMS};path=out/(name+'_prefix_selection.json');recovery.dump(path,dict(dataset=name,data_hashes=data['hashes'],split=pre['split'],heads=heads_by,head_fits=fits,grid=grid,selected=selected));hashes[name]=recovery.sha(path);print('SELECTED_EXTERNAL',name,{arm:(x['l2_multiplier'],x['margin'],x['q_initial']) for arm,x in selected.items()},flush=True)
        recovery.dump(out/'prefix_freeze.json',dict(utc=recovery.now(),runner_sha=code_sha,recovery_helper_sha=helper_sha,protocol_sha=psha,selection_sha=hashes));return
    freeze=json.loads((out/'prefix_freeze.json').read_text());assert freeze['runner_sha']==code_sha and freeze['recovery_helper_sha']==helper_sha and freeze['protocol_sha']==psha;all_results={};checks=[];primary=json.loads((ROOT/'results.json').read_text())
    for name,entry in protocol['studies'].items():
        selp=out/(name+'_prefix_selection.json');assert recovery.sha(selp)==freeze['selection_sha'][name];sel=json.loads(selp.read_text());data,pre=study(name,entry);assert data['hashes']==sel['data_hashes'] and pre['split']==sel['split'];trials=[]
        for seed in entry.get('test_seeds',protocol['test_seeds']):
            events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfgbase);base=engine.precompute(events,pre,cfgbase);streams={};rr={};aa={}
            for arm in ARMS:
                chosen=sel['selected'][arm];key=(arm[:3],str(chosen['l2_multiplier']));rule='pdf' if arm.startswith('pdf') else 'qmf';heads=np.asarray(sel['heads'][str(chosen['l2_multiplier'])]) if rule=='pdf' else None
                if key not in streams:streams[key]=stream_for(events,pre,base,heads,rule,cfgbase)
                cfg=dict(cfgbase,threshold=chosen['margin']);r=run(engine,streams[key],pre,cfg,arm,chosen['q_initial']);check=recovery.execute_check(events,pre,cfg,r,explicit_service,explicit_fork);assert check['passed'];rr[arm]=r;aa[arm]=check;checks.append(dict(dataset=name,seed=seed,arm=arm,**check))
            prim=next(t for t in primary[name]['trials'] if t['seed']==seed);rr['full_transfer']=prim['results']['full_transfer'];rr['reference']=prim['results']['reference'];trials.append(dict(seed=seed,results=rr,checks=aa));print('TEST_EXTERNAL',name,seed,{k:(round(r['net'],2),r['deployments'],r['harmful'],r['beneficial']) for k,r in rr.items()},flush=True)
        controllers={}
        for arm in ARMS+('full_transfer','reference'):
            rs=[t['results'][arm] for t in trials];rows=[z for r in rs for z in r['rows']];informative=[z for z in rows if z['disagreement']>0];admitted=[z for z in rows if z['action']];controllers[arm]=dict(mean_net=float(np.mean([r['net'] for r in rs])),harmful=sum(r['harmful'] for r in rs),beneficial=sum(r['beneficial'] for r in rs),admissions=sum(r['deployments'] for r in rs),coverage=dict(all=[sum(z['lower_covered'] for z in rows),len(rows)],informative=[sum(z['lower_covered'] for z in informative),len(informative)],admitted=[sum(z['lower_covered'] for z in admitted),len(admitted)]))
        paired={}
        for arm in ARMS:
            v=np.array([t['results']['full_transfer']['net']-t['results'][arm]['net'] for t in trials]);hw=2.776445105*v.std(ddof=1)/math.sqrt(5);paired[arm]=dict(values=v.tolist(),mean=float(v.mean()),conditional_t4_ci=[float(v.mean()-hw),float(v.mean()+hw)])
        result=dict(dataset=name,selected=sel['selected'],trials=trials,summary=dict(controllers=controllers,paired_full_minus=paired));all_results[name]=result;recovery.dump(out/(name+'_results.json'),result)
    recovery.dump(out/'results.json',all_results);recovery.dump(out/'summary.json',{n:r['summary'] for n,r in all_results.items()});recovery.dump(out/'execution_checks.json',dict(passed=all(r['passed'] for r in checks),trajectories=len(checks),forks=sum(r['forks'] for r in checks),max_service_error=max(max(r['service_errors'].values()) for r in checks),max_fork_error=max(r['fork_error'] for r in checks),checks=checks));print('COMPLETE_EXTERNAL',flush=True)

if __name__=='__main__':main()
