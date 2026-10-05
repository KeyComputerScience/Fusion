"""Predeclared known-trace RSS test of fully factorized information."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,importlib.util,json,math,sys
sys.dont_write_bytecode=True
import numpy as np
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('factorized_callable',OUT/'factorized_control.py')
fc=importlib.util.module_from_spec(spec);spec.loader.exec_module(fc)
matrix=fc.matrix;core=fc.core;ARM=fc.ARM;FROZEN=matrix.FROZEN
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    data=json.dumps(v,indent=2,allow_nan=False)
    if str(p).endswith('.gz'):
        with gzip.open(p,'wt') as f:f.write(data)
    else:Path(p).write_text(data)
def load(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify():
    protocol=load(OUT/'protocol.json')
    for p,h in protocol['source_hashes'].items():assert sha(p)==h,p
    return protocol
def freeze():
    assert not (OUT/'protocol.json').exists()
    files=[Path(__file__),OUT/'factorized_control.py',matrix.CORE,matrix.OUT/'run_matrix_controls.py',matrix.OUT/'protocol.json',
           FROZEN/'rss_validation/rss348_selection.json',FROZEN/'rss_validation/rss348_results.json',
           FROZEN/'physical/data/rss348/cached_dataset.npz',FROZEN/'physical/data/rss348/cache_metadata.json',
           matrix.PROJECT/'outputs/Fusion_Loss_Budget_Extension_Repro/guard/run_loss_budget_guard.py']
    dump(OUT/'protocol.json',dict(utc=now(),source_hashes={str(p):sha(p) for p in files},task='rss348',
        scope='known-trace post hoc extension; formula and calibration/test protocol frozen before new factorized results',
        arm=ARM,formula=dict(working_covariance='D=diag(R), global m-source R estimated from identical complete joint blocks',
            coordinate_precision='lambda_s=(q_s/mean(q))/sigma_p^2 + sum_{j:s observed} a_j/R_ss',
            coordinate_mean='mu_s=[sum_{j:s observed}a_j*z_js/R_ss]/lambda_s',
            state='P=diag(1/lambda_s); restrict global mu,D,P to current available source IDs after conditioning',
            objective='same full paid-value objective but using factorized mu,D,P; same causal readiness rule'),
        shared='immutable physical forecasts, complete matured blocks, masks, contrast/error projections, weights, quality, archive, models, solver, entropy, fees, q callbacks and ledger',
        calibration=dict(sd=[.05,.1,.2],floor=[0.,.64,1.2815515655446004],seeds=[87001,87002,87003],evaluations=27,
            q_initial='ready completely matured first-half pilot higher 90th percentile clipped at floor',
            selection='maximum second-half complete net return; ties higher floor then smaller sd'),
        test=dict(seeds=[88001,88002,88003,88004,88005],no_retuning=True,reserves=['fixed130','decision'],budgets=[0,110,130,260]),
        interventions='every original full RSS state, original full prior_sd and q; replace global likelihood covariance by its diagonal and rebuild mu,P; no models/history changes',
        limits=['RSS test already examined; this is not new independent validation',
                'five delays share one held-out physical trace',
                'joint and factorized selected ridge scales and q trajectories may differ; local intervention fixes original ridge and q',
                'shared budget bound is independent of covariance calibration'],python=sys.version,numpy=np.__version__))
    print('FROZEN_FACTORIZED',sha(OUT/'protocol.json'),flush=True)
def calibration():
    verify();assert not (OUT/'selection.json').exists()
    engine,recovery,guard,source,cfg0,study,service,fork=matrix.binding()
    data,pre=study('rss348',source['studies']['rss348'])
    events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,cfg0) for s in (87001,87002,87003)]
    bases=[engine.precompute(e,pre,cfg0) for e in events];grid=[];trials=[]
    for sd in (.05,.1,.2):
        cfg=dict(cfg0,prior_sd=sd);streams=[core.augment(e,pre,b,cfg) for e,b in zip(events,bases)]
        for floor in (0.,.64,1.2815515655446004):
            cc=dict(cfg,q_floor=floor,threshold=0.);qi,scores=core.initial_q(streams,pre,cc,ARM)
            rr=[core.run(engine,s,pre,cc,ARM,qi,True) for s in streams]
            entry=dict(prior_sd=sd,q_floor=floor,q_initial=qi,margin=0.,fit_scores=scores,net=float(np.mean([r['selection_net'] for r in rr])),solver_failures=sum(r['solver_failures'] for r in rr))
            assert entry['solver_failures']==0
            grid.append(entry);trials.append(dict(configuration=entry,results=rr));print('CAL_FACTORIZED',sd,floor,entry['net'],flush=True)
    selected=max(grid,key=lambda g:(g['net'],g['q_floor'],-g['prior_sd']))
    dump(OUT/'selection.json',dict(data_hashes=data['hashes'],split=pre['split'],grid=grid,selected=selected))
    dump(OUT/'calibration_trials.json.gz',trials)
    dump(OUT/'selection_freeze.json',dict(utc=now(),selection_sha256=sha(OUT/'selection.json'),trials_sha256=sha(OUT/'calibration_trials.json.gz'),protocol_sha256=sha(OUT/'protocol.json')))
    print('SELECT_FACTORIZED',selected,flush=True)
def decomposition(full,control,full_actions=None,control_actions=None):
    categories={x:dict(count=0,absolute_return=0.) for x in ('retained_gain','missed_gain','avoided_loss','incurred_loss','zero_return')};cases=[]
    for x,y in zip(full,control):
        assert x['k']==y['k'] and x['local_net']==y['local_net']
        k=x['k'];a=bool(x['action']) if full_actions is None else bool(full_actions[k]);b=bool(y['action']) if control_actions is None else bool(control_actions[k])
        if a==b:continue
        D=x['local_net'];cat=('retained_gain' if D>0 else 'incurred_loss' if D<0 else 'zero_return') if a else ('missed_gain' if D>0 else 'avoided_loss' if D<0 else 'zero_return')
        categories[cat]['count']+=1;categories[cat]['absolute_return']+=abs(D)
        cases.append(dict(k=k,category=cat,complete_return=D,full_action=a,control_action=b,full_minus_control_return=(int(a)-int(b))*D))
    return dict(categories=categories,cases=cases,total_full_minus_control_return=sum(c['full_minus_control_return'] for c in cases))
def aggregate_decomposition(ds):
    categories={x:dict(count=sum(d['categories'][x]['count'] for d in ds),absolute_return=sum(d['categories'][x]['absolute_return'] for d in ds)) for x in ds[0]['categories']}
    total=sum(d['total_full_minus_control_return'] for d in ds)
    return dict(categories=categories,changed_actions=sum(len(d['cases']) for d in ds),total_full_minus_control_return=total,mean_full_minus_control_return=total/5,cases=[c for d in ds for c in d['cases']])
def test():
    verify();assert not (OUT/'results.json').exists()
    selection=load(OUT/'selection.json');assert sha(OUT/'selection.json')==load(OUT/'selection_freeze.json')['selection_sha256']
    engine,recovery,guard,source,cfg0,study,service,fork=matrix.binding();data,pre=study('rss348',source['studies']['rss348'])
    assert data['hashes']==selection['data_hashes']
    original=load(FROZEN/'rss_validation/rss348_results.json');trials=[];budget_trials=[];local_rows=[];local_ds=[];dynamic_ds=[]
    state_relation=dict(states=0,nonzero_mean_differences=0,nonzero_precision_differences=0,max_mean_difference=0.,max_precision_difference=0.)
    for seed in (88001,88002,88003,88004,88005):
        events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg0);base=engine.precompute(events,pre,cfg0)
        ss=selection['selected'];cfg=dict(cfg0,prior_sd=ss['prior_sd'],q_floor=ss['q_floor'],threshold=0.)
        stream=core.augment(events,pre,base,cfg);result=core.run(engine,stream,pre,cfg,ARM,ss['q_initial'])
        check=recovery.execute_check(events,pre,cfg,result,service,fork);assert check['passed'] and result['solver_failures']==0
        old=next(t for t in original['trials'] if t['seed']==seed);full=old['results']['precision_joint'];reference=old['results']['reference']
        d=decomposition(full['rows'],result['rows']);d['seed']=seed
        for c in d['cases']:c['seed']=seed
        dynamic_ds.append(d)
        results=dict(precision_joint=full,factorized_information=result,reference=reference)
        for arm,r in results.items():
            if arm=='reference':continue
            reserves={}
            for row in r['rows']:
                event=events[row['k']];common=cfg0['probe_drop']+(2 if event['refresh'] else 0)
                reserves[row['k']]=guard.current_decision_reserve(event['x'],event['candidate'],event['reference'],common,cfg0)[0]
            for rule in ('fixed130','decision'):
                rs=reserves if rule=='decision' else {r['k']:130. for r in r['rows']}
                for budget in (0,110,130,260):
                    g=guard.replay(r['rows'],len(events),budget,'gross_loss',rs)
                    actual=service(events,pre,g['rows'],cfg0);prefix=guard.reconstruct_prefix_increment(events,g['rows'],cfg0)
                    admitted=[r for r in g['rows'] if r['action']];delta=actual['net']-reference['net'];loss=sum(max(0.,-r['local_net']) for r in admitted)
                    error=max(abs(delta-sum(r['local_net'] for r in admitted)),abs(delta-prefix['final']),abs(loss-g['final_spent_loss']))
                    assert error<1e-8 and loss<=budget+1e-8 and prefix['minimum']>=-budget-1e-8
                    budget_trials.append(dict(seed=seed,arm=arm,reserve=rule,budget=budget,increment=delta,net=actual['net'],admissions=len(admitted),
                        harmful=sum(r['local_net']<0 for r in admitted),beneficial=sum(r['local_net']>0 for r in admitted),negative_loss=loss,
                        refusals=sum(r['proposed_action'] and not r['action'] for r in g['rows']),accounting_error=error,ledger=g['ledger']))
        old_ss=original['selected']['precision_joint'];old_cfg=dict(cfg0,prior_sd=old_ss['prior_sd'],q_floor=old_ss['q_floor'],threshold=0.)
        old_stream=core.augment(events,pre,base,old_cfg);replaced=[]
        for d,row in zip(old_stream['decisions'],full['rows']):
            mm=d['temporal'];assert np.max(np.abs(mm['mu']-np.array(row['joint_mu'])))<1e-8
            w,cert,F,S,T,extra=fc.factorized_forecast(d,pre,old_cfg,ARM,row['q_issued']);assert cert['converged']
            score=F-row['q_issued']*S-5.;r=dict(row,action=bool(score>0),weights=w.tolist(),gain=F,posterior_or_block_sd=S,gate_score=score,standardized_score=T,lower_covered=bool(T<=row['q_issued']),**extra)
            replaced.append(r);local_rows.append(dict(r,seed=seed))
            state_relation['states']+=1
            for field,key in [('factorized_mean_difference','mean'),('factorized_precision_difference','precision')]:
                difference=mm[field];state_relation['max_'+key+'_difference']=max(state_relation['max_'+key+'_difference'],difference)
                state_relation['nonzero_'+key+'_differences']+=difference>1e-10
        d=decomposition(full['rows'],replaced)
        for c in d['cases']:
            c['seed']=seed;k=c['k'];x=next(r for r in full['rows'] if r['k']==k);y=next(r for r in replaced if r['k']==k)
            c.update(full_score=x['gate_score'],factorized_score=y['gate_score'],full_mu=x['joint_mu'],factorized_mu=y['factorized_mu'],
                     full_P=x['joint_P'],factorized_P=y['factorized_P'],full_R=x['joint_R'],factorized_R=y['factorized_R'],q=x['q_issued'],
                     full_excess=max(0.,x['gain']-x['truegross']-x['q_issued']*x['posterior_or_block_sd']),
                     factorized_excess=max(0.,y['gain']-y['truegross']-y['q_issued']*y['posterior_or_block_sd']))
        local_ds.append(d);trials.append(dict(seed=seed,results=results,checks={ARM:check}))
        print('TEST_FACTORIZED',seed,result['net'],result['deployments'],result['harmful'],flush=True)
    summary=core.summarize(trials)
    for arm in ('precision_joint',ARM):
        rr=[t['results'][arm] for t in trials];summary[arm]['increments']=[r['net']-t['results']['reference']['net'] for r,t in zip(rr,trials)]
        summary[arm]['mean_increment']=float(np.mean(summary[arm]['increments']));summary[arm]['severity']=matrix.severity([r for result in rr for r in result['rows']])
    difference=np.array(summary['precision_joint']['increments'])-np.array(summary[ARM]['increments']);half=2.776445105*difference.std(ddof=1)/math.sqrt(5)
    paired=dict(values=difference.tolist(),mean=float(difference.mean()),conditional_t4_ci=[float(difference.mean()-half),float(difference.mean()+half)])
    budgets={};budget_decompositions={}
    for rule in ('fixed130','decision'):
        budgets[rule]={};budget_decompositions[rule]={}
        for value in (0,110,130,260):
            budgets[rule][str(value)]={};ds=[]
            for arm in ('precision_joint',ARM):
                rows=[t for t in budget_trials if t['arm']==arm and t['reserve']==rule and t['budget']==value]
                budgets[rule][str(value)][arm]=dict(mean_net=float(np.mean([r['net'] for r in rows])),mean_increment=float(np.mean([r['increment'] for r in rows])),
                    increments=[r['increment'] for r in rows],admissions=sum(r['admissions'] for r in rows),harmful=sum(r['harmful'] for r in rows),beneficial=sum(r['beneficial'] for r in rows),refusals=sum(r['refusals'] for r in rows),total_negative_loss=sum(r['negative_loss'] for r in rows))
            for trial in trials:
                seed=trial['seed'];f=next(t for t in budget_trials if t['seed']==seed and t['arm']=='precision_joint' and t['reserve']==rule and t['budget']==value);c=next(t for t in budget_trials if t['seed']==seed and t['arm']==ARM and t['reserve']==rule and t['budget']==value)
                d=decomposition(trial['results']['precision_joint']['rows'],trial['results'][ARM]['rows'],{r['k']:r['action'] for r in f['ledger']},{r['k']:r['action'] for r in c['ledger']})
                for row in d['cases']:row['seed']=seed
                ds.append(d)
            budget_decompositions[rule][str(value)]=aggregate_decomposition(ds)
    dump(OUT/'results.json',dict(protocol_sha256=sha(OUT/'protocol.json'),selected=ss,summary=summary,full_minus_factorized=paired,trials=trials,
        budget_summary=budgets,dynamic_decomposition=aggregate_decomposition(dynamic_ds),budget_decomposition=budget_decompositions,
        fixed_state=dict(state_relation=state_relation,decomposition=aggregate_decomposition(local_ds),severity=matrix.severity(local_rows))))
    dump(OUT/'budget_trials.json.gz',budget_trials);dump(OUT/'fixed_state_all_rows.json.gz',local_rows)
    print('SUMMARY_FACTORIZED',summary[ARM],paired,'STATE_RELATION',state_relation,flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('freeze','calibrate','test'),required=True);args=parser.parse_args()
    dict(freeze=freeze,calibrate=calibration,test=test)[args.phase]()
    if args.phase!='freeze':verify()
if __name__=='__main__':main()
