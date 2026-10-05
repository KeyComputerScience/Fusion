"""Post-primary, fixed-form PDF probability-interface deployment adaptation."""
from __future__ import annotations
import argparse,datetime,hashlib,json,math,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parent
ARMS=('pdf_point','pdf_calibrated')
L2S=(0.,.25,1.)
MARGINS=(0.,4.,12.)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False))

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def softmax(x,axis=-1):
    z=x-x.max(axis=axis,keepdims=True);p=np.exp(z);return p/p.sum(axis=axis,keepdims=True)

def fit_tcp(pre,l2_multiplier,protocol):
    heads=[];records=[];steps=protocol['tcp_head']['steps'];lr=protocol['tcp_head']['learning_rate']
    penalty=protocol['tcp_head']['base_l2']*l2_multiplier
    for s in range(pre['m']):
        p=np.concatenate([r['p'][s] for r in pre['prior']]);y=np.concatenate([r['y'] for r in pre['prior']]);x=np.c_[p,np.ones(len(p))]
        target=p[np.arange(len(y)),y];average=np.clip(target.mean(),1e-5,1-1e-5)
        theta=np.zeros(x.shape[1]);theta[-1]=np.log(average/(1-average))
        initial=float(np.mean((sigmoid(x@theta)-target)**2))
        for _ in range(steps):
            pred=sigmoid(x@theta);g=x.T@(2*(pred-target)*pred*(1-pred))/len(x)
            g[:-1]+=penalty*theta[:-1];theta-=lr*g
        pred=sigmoid(x@theta);heads.append(theta)
        records.append(dict(source=s,prefix_audit_rows=len(y),steps=steps,l2_multiplier=l2_multiplier,l2=penalty,initial_mse=initial,final_mse=float(np.mean((pred-target)**2)),target_mean=float(target.mean()),forecast_mean=float(pred.mean())))
    return np.stack(heads),records

def pdf_fuse(p,heads,ids):
    # p has active source x request x class dimensions. All class probabilities
    # are original issued values. Confidence heads never inspect current labels.
    m,n,c=p.shape
    if m==1:
        confidence=sigmoid(np.c_[p[0],np.ones(n)]@heads[ids[0]])[:,None]
        return p[0].copy(),np.ones((n,1)),confidence
    cp=np.clip(p,1e-12,1.)
    features=np.concatenate([p,np.ones((m,n,1))],axis=2)
    tcp=np.clip(sigmoid(np.einsum('snc,sc->sn',features,heads[ids])),1e-6,1-1e-6)
    logs=np.log(tcp);total=logs.sum(0,keepdims=True)
    co=tcp+(total-logs)/total
    du=np.mean(np.abs(p-1/c),axis=2);peers=du.sum(0,keepdims=True)-du
    rc=np.ones_like(du);nonzero=peers>1e-12
    rc[nonzero]=np.minimum(1.,((m-1)*du[nonzero])/peers[nonzero])
    weights=softmax((co*rc).T,axis=1)
    fused=softmax(np.einsum('ns,snc->nc',weights,np.log(cp)),axis=1)
    return fused,weights,tcp.T

def rule_audit():
    # Two-source formula is checked independently against official code algebra.
    rng=np.random.default_rng(224517);p=rng.dirichlet(np.ones(6),size=(2,24));heads=rng.normal(size=(2,7));f,w,tcp=pdf_fuse(p,heads,np.arange(2))
    cb0=tcp[:,0]+np.log(tcp[:,1])/(np.log(tcp[:,0]*tcp[:,1]));cb1=tcp[:,1]+np.log(tcp[:,0])/(np.log(tcp[:,0]*tcp[:,1]));du=np.mean(np.abs(p-1/6),axis=2).T
    cb=np.c_[cb0*np.minimum(1,du[:,0]/du[:,1]),cb1*np.minimum(1,du[:,1]/du[:,0])];expected=softmax(cb,1)
    logits=np.log(p)+rng.normal(size=(2,24,1));oracle=softmax(np.einsum('ns,snc->nc',expected,logits),1)
    uniform=np.ones((4,9,6))/6;uf,uw,_=pdf_fuse(uniform,np.zeros((4,7)),np.arange(4))
    sf,sw,_=pdf_fuse(p[:1],heads,np.array([0]))
    errors=dict(official_two_source_weight=float(np.max(abs(w-expected))),logit_shift_invariance=float(np.max(abs(f-oracle))),uniform_output=float(np.max(abs(uf-1/6))),uniform_weights=float(np.max(abs(uw-1/4))),singleton_output=float(np.max(abs(sf-p[0]))),singleton_weight=float(np.max(abs(sw-1))))
    assert max(errors.values())<1e-10
    return dict(passed=True,errors=errors,weight_rule='published softmax; no cap; no optimization/KKT assertion')

def augment_stream(events,pre,stream,heads,cfg):
    # Preserve each external fused forecast at origin, including its origin mask.
    prior=[]
    for r in pre['prior']:
        pf,_,_=pdf_fuse(r['p'],heads,np.arange(pre['m']));prior.append((cfg['prior_mass']/len(pre['prior']),dict(r,pdf_p=pf)))
    pending=[];archive=[];by_k={d['k']:d for d in stream['decisions']};external=[]
    for k,e in enumerate(events):
        for due,r in pending:
            if due<=k:archive.append(r)
        archive=archive[-cfg['archive']:];pending=[x for x in pending if x[0]>k]
        ids=np.flatnonzero(e['mask']);pf,weights,tcp=pdf_fuse(e['p'][ids],heads,ids)
        if k in by_k:
            d=by_k[k];chunks=prior.copy()
            for r in archive:
                if not r['mask'][ids].all():continue
                a=cfg['beta']**(k-r['origin'])*math.exp(-float(np.sum((e['context']-r['context'])**2))/cfg['kernel_bandwidth']**2)
                chunks.append((max(a,1e-12),r))
            means=[];seconds=[];alphas=[]
            for a,r in chunks:
                active=(r['x']@e['reference']).argmax(1);candidate=(r['x']@e['candidate']).argmax(1)
                b=np.eye(pre['classes'])[candidate]-np.eye(pre['classes'])[active]
                z=np.einsum('ic,ic->i',r['pdf_p']-np.eye(pre['classes'])[r['y']],b)
                means.append(float(z.mean()));seconds.append(float(np.mean(z*z)));alphas.append(a)
            alpha=np.array(alphas);mass=float(alpha.sum());a=alpha/mass;means=np.array(means);mu=float(a@means);Q=float(a@seconds)
            B=max(0.,float(a@(means*means)-mu*mu));Sigma=max(0.,Q-mu*mu);U=B/(mass+1)
            candidate=(e['x']@e['candidate']).argmax(1);reference=(e['x']@e['reference']).argmax(1)
            b=np.eye(pre['classes'])[candidate]-np.eye(pre['classes'])[reference];h=float(np.mean(np.einsum('ic,ic->i',pf,b)))
            gain=float(d['N']*(h-mu));sd=d['N']*math.sqrt(U+1e-4)
            ext=dict(gain=gain,sd=sd,standardized=(gain-d['truegross'])/sd,mu=mu,predictive_variance=Sigma,between_block_variance=B,posterior_variance=U,mass=mass,blocks=len(chunks),active_sources=ids.tolist(),pdf_weights=weights.tolist(),pdf_weight_mean=weights.mean(0).tolist(),tcp_mean=tcp.mean(0).tolist(),fused_probability_sha256=hashlib.sha256(pf.tobytes()).hexdigest())
            external.append(dict(d,external_pdf=ext))
        ar=e['audit']
        if ar.any():pending.append((k+e['delay'],dict(x=e['x'][ar].copy(),y=e['y'][ar].copy(),p=e['p'][:,ar].copy(),pdf_p=pf[ar].copy(),context=e['context'].copy(),mask=e['mask'].copy(),origin=k)))
    return dict(stream,decisions=external)

def external_forecast(d,pre,cfg,mode):
    a=d['external_pdf'];return np.ones(1),dict(kkt=0.,primal=0.,converged=True,iterations=0),a['gain'],a['sd'],a['standardized']

def external_run(engine,stream,pre,cfg,arm,qinit,selection=False):
    original=engine.decision_forecast;engine.decision_forecast=external_forecast
    try:r=engine.run(stream,pre,cfg,'bayes_gate' if arm=='pdf_calibrated' else 'joint',qinit,selection)
    finally:engine.decision_forecast=original
    for row,d in zip(r['rows'],stream['decisions']):row.update(d['external_pdf'])
    r['external_arm']=arm;r['gate_weights_semantics']='one scalar immutable external fused forecast; [1]'
    r['source_weights_semantics']='per-request PDF softmax; no cap'
    r['optimizer_applicability']='none; closed-form source softmax; kkt=0 is unused engine-interface placeholder'
    r['coverage_semantics']='executed delayed calibrated gate' if arm=='pdf_calibrated' else 'diagnostic q tracker, unused by point action; no issued lower-gate protection'
    return r

def initial_q(engine,streams,pre,cfg):
    original=engine.decision_forecast;engine.decision_forecast=external_forecast
    try:return engine.initial_q(streams,pre,cfg,'bayes_gate')
    finally:engine.decision_forecast=original

def execute_audit(events,pre,cfg,result,explicit_service,explicit_fork):
    truth=explicit_service(events,pre,result['rows'],cfg);errors={k:abs(float(truth[k])-float(result[k])) for k in ('gross','fees','net','drops','accuracy','deployments')}
    forkerr=targeterr=weight_error=0.;max_weight=0.
    for row in result['rows']:
        f=explicit_fork(events,row,cfg);forkerr=max(forkerr,abs(f['gross']-row['truegross']),abs(f['actual']-row['local_net']))
        targeterr=max(targeterr,f['identity_error'],abs(f['standardized']-row['standardized_score']),f['covered_actual_lower_error'])
        w=np.asarray(row['pdf_weights']);weight_error=max(weight_error,float(np.max(abs(w.sum(1)-1))),max(0.,-float(w.min())));max_weight=max(max_weight,float(w.max()))
    for row in result['q_updates']:assert row['issued_index']<row['maturity']<=row['update_index']
    passed=max(errors.values())<1e-8 and forkerr<1e-8 and targeterr<1e-8 and abs(result['calibration_identity_error'])<1e-8 and abs(result['closed_loop_identity_error'])<1e-8 and weight_error<1e-10 and result['solver_failures']==0 and result['max_kkt']<1e-6
    applicability='none: closed-form external rule' if 'external_arm' in result else 'frozen internal convex source optimizer'
    return dict(all_passed=bool(passed),explicit_service=truth,service_errors=errors,max_service_error=max(errors.values()),max_fork_error=forkerr,max_target_identity_error=targeterr,max_pdf_weight_simplex_error=weight_error,max_pdf_source_weight=max_weight,cap_applicability='not applied: original PDF softmax' if 'external_arm' in result else 'frozen internal source cap',optimizer_kkt_applicability=applicability,solver_failures=result['solver_failures'],max_kkt=result['max_kkt'] if 'external_arm' not in result else None,calibration_identity_error=result['calibration_identity_error'],closed_loop_identity_error=result['closed_loop_identity_error'],forks=len(result['rows']))

def paired(values):
    x=np.array(values,dtype=float);mean=float(x.mean());half=2.776445105*float(x.std(ddof=1))/math.sqrt(len(x));return dict(mean=mean,ci95_conditional_t4=[mean-half,mean+half],values=x.tolist(),positive=int(np.sum(x>1e-9)),negative=int(np.sum(x< -1e-9)),ties=int(np.sum(abs(x)<=1e-9)))

def summary(trials):
    names=('pdf_point','pdf_calibrated','bayes_both','joint','frequentist_gate','frozen');out={}
    for name in names:
        rs=[t['results'][name] for t in trials];rows=[row for r in rs for row in r['rows']];inf=[r for r in rows if r['disagreement']>0];admitted=[r for r in rows if r['action']]
        out[name]=dict(mean_net=float(np.mean([r['net'] for r in rs])),mean_actions=float(np.mean([r['deployments'] for r in rs])),mean_harmful=float(np.mean([r['harmful'] for r in rs])),mean_beneficial=float(np.mean([r['beneficial'] for r in rs])),pooled_coverage=dict(overall=[sum(r['lower_covered'] for r in rows),len(rows)],informative=[sum(r['lower_covered'] for r in inf),len(inf)],admitted=[sum(r['lower_covered'] for r in admitted),len(admitted)]),coverage_applicability='diagnostic tracker only' if name=='pdf_point' else 'gate' if name in ('pdf_calibrated','bayes_both','frequentist_gate') else 'diagnostic internal tracker')
    pairs={}
    for baseline in ARMS+('joint','frequentist_gate','frozen'):
        rs=[t['results']['bayes_both']['net']-t['results'][baseline]['net'] for t in trials]
        pairs['full_minus_'+baseline]=dict(net=paired(rs),mean_action_disagreements=float(np.mean([np.sum(np.array(t['results']['bayes_both']['actions'])!=np.array(t['results'][baseline]['actions'])) for t in trials])),harmful_admissions=paired([t['results']['bayes_both']['harmful']-t['results'][baseline]['harmful'] for t in trials]))
    return dict(controllers=out,paired=pairs)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,default=ROOT.parent.parent/'outputs'/'bayes_closed_loop_repro');parser.add_argument('--output',type=Path,default=ROOT);parser.add_argument('--phase',choices=('calibrate','test'),required=True);args=parser.parse_args();package=args.package.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(package));from cached_runner import install_cached_loaders
    engine,adapter=install_cached_loaders();from audit_canary_execution import load_study,explicit_service,explicit_fork
    protocol=json.loads((out/'protocol.json').read_text());protocolsha=sha(out/'protocol.json');scriptsha=sha(__file__)
    assert sha(package/'independent_bayes_fusion.py')==protocol['frozen_engine_sha256'];assert sha(package/'independent_bayes_extension.py')==protocol['frozen_adapter_sha256']
    if args.phase=='calibrate':
        assert not (out/'prefix_freeze.json').exists(),'prefix already frozen; do not retune'
        originals={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()};dump(out/'original_files_before.json',originals)
        dump(out/'formula_audit.json',rule_audit());dump(out/'pre_calibration_freeze.json',dict(created_utc=now(),protocol_sha256=protocolsha,script_sha256=scriptsha,new_test_metrics_inspected=False))
        selection_hashes={}
        for name,entry in protocol['source_studies'].items():
            start=time.perf_counter();data,pre,saved=load_study(package/'independent_data',package/entry['folder'],name)
            events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,seed,engine.BASE) for seed in protocol['calibration_seeds']];base=[engine.precompute(e,pre,engine.BASE) for e in events];heads_by={};fits={};grid=[]
            for multiplier in L2S:
                heads,fit=fit_tcp(pre,multiplier,protocol);heads_by[str(multiplier)]=heads.tolist();fits[str(multiplier)]=fit
                streams=[augment_stream(e,pre,s,heads,engine.BASE) for e,s in zip(events,base)]
                for margin in MARGINS:
                    cfg=dict(engine.BASE,threshold=margin);qi,scores=initial_q(engine,streams,pre,cfg)
                    for arm in ARMS:
                        q0=qi if arm=='pdf_calibrated' else 0.;rr=[external_run(engine,s,pre,cfg,arm,q0,True) for s in streams]
                        grid.append(dict(arm=arm,l2_multiplier=multiplier,margin=margin,q_initial=q0,q_fit_scores=scores if arm=='pdf_calibrated' else [],mean_selection_net=float(np.mean([r['selection_net'] for r in rr])),mean_selection_actions=float(np.mean([r['deployments'] for r in rr]))))
                    print('CALIBRATION',name,multiplier,margin,flush=True)
            selected={arm:max([r for r in grid if r['arm']==arm],key=lambda r:(r['mean_selection_net'],-r['margin'],-r['l2_multiplier'])) for arm in ARMS}
            sel=dict(dataset=name,protocol_sha256=protocolsha,script_sha256=scriptsha,data_hashes=data['hashes'],split=pre['split'],heads=heads_by,head_fit=fits,grid=grid,selected=selected,elapsed_seconds=time.perf_counter()-start,created_utc=now());path=out/(name+'_prefix_selection.json');dump(path,sel);selection_hashes[name]=sha(path);print('PREFIX_SELECTED',name,{a:(r['l2_multiplier'],r['margin'],r['q_initial']) for a,r in selected.items()},sel['elapsed_seconds'],flush=True)
        dump(out/'prefix_freeze.json',dict(created_utc=now(),protocol_sha256=protocolsha,script_sha256=scriptsha,all_prefix_selection_sha256=selection_hashes,new_test_metrics_inspected=False));print('ALL_PREFIX_FROZEN',flush=True);return
    freeze=json.loads((out/'prefix_freeze.json').read_text());assert protocolsha==freeze['protocol_sha256'];assert scriptsha==freeze['script_sha256'],'runner changed after prefix freeze'
    allresults={};checks=[]
    for name,entry in protocol['source_studies'].items():
        selpath=out/(name+'_prefix_selection.json');assert sha(selpath)==freeze['all_prefix_selection_sha256'][name];sel=json.loads(selpath.read_text());data,pre,saved=load_study(package/'independent_data',package/entry['folder'],name);assert data['hashes']==sel['data_hashes'];assert pre['split']==sel['split'];trials=[];saved_trials={t['seed']:t for t in saved['trials']}
        for seed in protocol['test_seeds']:
            start=time.perf_counter();events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,engine.BASE);base=engine.precompute(events,pre,engine.BASE);streams={};results={};audit={}
            for arm in ARMS:
                chosen=sel['selected'][arm];key=str(chosen['l2_multiplier']);heads=np.asarray(sel['heads'][key]);cfg=dict(engine.BASE,threshold=chosen['margin'])
                if key not in streams:streams[key]=augment_stream(events,pre,base,heads,engine.BASE)
                r=external_run(engine,streams[key],pre,cfg,arm,chosen['q_initial']);results[arm]=r;audit[arm]=execute_audit(events,pre,cfg,r,explicit_service,explicit_fork);assert audit[arm]['all_passed'];checks.append(dict(dataset=name,seed=seed,arm=arm,**audit[arm]))
            for mode in ('bayes_both','joint','frequentist_gate','frozen'):
                cfg=saved['selected'].get(mode,engine.BASE);qi=saved['q_initial'].get(mode,0.);r=engine.run(base,pre,cfg,mode,qi);original=saved_trials[seed]['results'][mode];assert abs(r['net']-original['net'])<1e-8 and r['actions']==original['actions'],'frozen primary reproduction differs';results[mode]=r
                audit[mode]=execute_audit(events,pre,cfg,{**r,'rows':[dict(row,pdf_weights=[[1.]]) for row in r['rows']]},explicit_service,explicit_fork);assert audit[mode]['all_passed'];checks.append(dict(dataset=name,seed=seed,arm=mode,**audit[mode]))
            trials.append(dict(seed=seed,elapsed_seconds=time.perf_counter()-start,results=results,execution_audit=audit));print('TEST_EXTERNAL',name,seed,{m:(round(r['net'],6),r['deployments'],r['harmful']) for m,r in results.items()},'seconds',trials[-1]['elapsed_seconds'],flush=True);dump(out/(name+'_results_partial.json'),dict(dataset=name,trials=trials))
        result=dict(dataset=name,protocol_sha256=protocolsha,prefix_selection_sha256=sha(selpath),script_sha256=scriptsha,data_hashes=data['hashes'],split=pre['split'],selected=sel['selected'],trials=trials,summary=summary(trials));allresults[name]=result;dump(out/(name+'_results.json'),result);dump(out/'summary.json',{n:r['summary'] for n,r in allresults.items()})
    originals=json.loads((out/'original_files_before.json').read_text());after={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()};changed=[p for p,h in originals.items() if after.get(p)!=h];added=sorted(set(after)-set(originals));assert not changed and not added,'original package changed'
    report=dict(all_passed=all(x['all_passed'] for x in checks),created_utc=now(),protocol_sha256=protocolsha,script_sha256=scriptsha,original_package_unchanged=True,changed=[],added=[],trajectory_count=len(checks),fork_count=sum(x['forks'] for x in checks),max_service_error=max(x['max_service_error'] for x in checks),max_fork_error=max(x['max_fork_error'] for x in checks),max_target_identity_error=max(x['max_target_identity_error'] for x in checks),max_calibration_identity_error=max(abs(x['calibration_identity_error']) for x in checks),max_closed_loop_identity_error=max(abs(x['closed_loop_identity_error']) for x in checks),checks=checks)
    dump(out/'results.json',allresults);dump(out/'execution_audit.json',report);print('COMPLETE',report['trajectory_count'],report['fork_count'],report['all_passed'],flush=True)

if __name__=='__main__':main()
