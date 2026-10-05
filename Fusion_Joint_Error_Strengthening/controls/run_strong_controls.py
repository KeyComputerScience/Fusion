"""Same-information empirical adaptive-risk controls and fixed130 guard.

All six source tasks are development data. Selection uses calibration only.
The deterministic mass control is algebraically identical to the posterior
formula. No exclusive Bayesian gain can be inferred from that comparison.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = ROOT.parents[2] / 'outputs' / 'Fusion_Recovery_Repro'
DEFAULT_GUARD = ROOT.parents[2] / 'outputs' / 'Fusion_Loss_Budget_Extension_Repro' / 'guard'
BUDGETS = (0, 130, 133, 260, 266, 532)
RESERVES = ('fixed130', 'decision', 'fixed133')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def empirical_uncertainty(mm, dimension):
    """Regularized weighted-mean covariance using Kish effective mass.

    Prior atoms have equal mass a0/(2m). Completed lease atoms use exactly
    the old kernel/age masses. Covariance = B/(n_eff-1) incorporates the
    weighted sample-covariance degrees-of-freedom correction. This is a
    non-Bayesian adaptive penalty; temporal independence is not asserted.
    """
    squared_mass = mm['prior_mass']**2/(2*dimension)
    squared_mass += sum(float(e['weight'])**2 for e in mm['eligible'])
    effective_mass = float(mm['mass'])**2/squared_mass
    assert effective_mass > 1
    return mm['B']/(effective_mass-1), effective_mass, squared_mass

def uncertainty(mm, dimension, arm):
    if arm == 'mass_equivalent':
        # A deterministic regularized risk penalty; no posterior is sampled.
        return mm['B']/(mm['mass']+1), None, None
    if arm == 'kish_adaptive': return empirical_uncertainty(mm, dimension)
    raise ValueError(arm)

def forecast(recovery, d, pre, cfg, arm):
    mm = d['lease']; u, _, _ = uncertainty(mm, len(d['ids']), arm)
    rate = d['h']-mm['mu']
    w, cert = recovery.convex_influence(pre['q'][d['ids']], rate, mm['S'], u, cfg, 'full_transfer')
    norm = math.sqrt(max(0., float(w@u@w))+cfg['norm_floor']**2)
    gain = float(d['N']*(w@rate-cfg['norm_kappa']*norm))
    sd = d['N']*math.sqrt(max(0., float(w@mm['B']@w))+cfg['gate_floor'])
    return w, cert, gain, sd, (gain-d['truegross'])/sd

def run(engine, recovery, stream, pre, cfg, arm, q0, selection=False):
    old = engine.decision_forecast
    engine.decision_forecast = lambda d,p,c,m: forecast(recovery,d,p,c,arm)
    try: result = engine.run(stream,pre,cfg,'bayes_gate',q0,selection)
    finally: engine.decision_forecast = old
    for row,d in zip(result['rows'],stream['decisions']):
        mm = d['lease']; u,neff,sumsq = uncertainty(mm,len(d['ids']),arm)
        row.update(control=arm, lease_mu=mm['mu'].tolist(), lease_B=mm['B'].tolist(),
                   adaptive_risk_matrix=u.tolist(), lease_mass=mm['mass'],
                   effective_mass=neff, squared_atom_mass=sumsq,
                   eligible_residuals=mm['eligible'], current_source_rate=d['h'].tolist(),
                   active_source_ids=d['ids'].tolist(), contrast_context=d['context'].tolist())
        assert all(e['maturity']<=row['k'] for e in mm['eligible'])
    result['control'] = arm
    return result

def q_initial(recovery, streams, pre, cfg, arm):
    scores = [forecast(recovery,d,pre,cfg,arm)[4] for s in streams
              for d in s['decisions'] if d['maturity']<=s['windows']//2]
    return max(0.,float(np.quantile(scores,.9,method='higher'))) if scores else 0., scores

def setup(package, guard_path):
    sys.path.insert(0,str(package/'base'))
    from cached_runner import install_cached_loaders
    engine,_ = install_cached_loaders()
    from audit_canary_execution import load_study,explicit_service,explicit_fork
    sys.path.insert(0,str(package/'extension'))
    import run_recovery_fusion as recovery
    from new_data_adapter import load_dataset,make_prefix
    sys.path.insert(0,str(guard_path))
    import run_loss_budget_guard as guard
    protocol = json.loads((package/'extension/protocol.json').read_text())
    cfg = dict(engine.BASE,**protocol['extension'])
    def study(name,spec):
        if spec.get('new_task'):
            data,derivation=load_dataset(package/'extension/data',name)
            return data,make_prefix(data,derivation,cfg,engine)
        data,pre,_=load_study(package/'base/independent_data',package/'base'/spec['folder'],name)
        return data,pre
    return engine,recovery,guard,protocol,cfg,study,explicit_service,explicit_fork

def control_summary(trials):
    out={}
    for arm in ('full_transfer','mass_equivalent','kish_adaptive','reference'):
        results=[t['results'][arm] for t in trials]; rows=[z for r in results for z in r['rows']]
        inf=[z for z in rows if z['disagreement']>0]; act=[z for z in rows if z['action']]
        out[arm]=dict(mean_net=float(np.mean([r['net'] for r in results])),
                      mean_admissions=float(np.mean([r['deployments'] for r in results])),
                      harmful=sum(r['harmful'] for r in results),beneficial=sum(r['beneficial'] for r in results),
                      coverage={k:[sum(z['lower_covered'] for z in rr),len(rr)]
                                for k,rr in (('issued',rows),('informative',inf),('admitted',act))})
    for arm in ('mass_equivalent','kish_adaptive'):
        dif=np.array([t['results']['full_transfer']['net']-t['results'][arm]['net'] for t in trials])
        half=2.776445105*dif.std(ddof=1)/np.sqrt(len(dif))
        out['full_minus_'+arm]=dict(values=dif.tolist(),mean=float(dif.mean()),
                                   conditional_t4_ci=[float(dif.mean()-half),float(dif.mean()+half)],
                                   changed_actions=sum(sum(a!=b for a,b in zip(t['results']['full_transfer']['actions'],t['results'][arm]['actions'])) for t in trials))
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,default=DEFAULT_PACKAGE)
    parser.add_argument('--guard',type=Path,default=DEFAULT_GUARD)
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--phase',required=True,choices=('calibrate','test','guard'))
    args=parser.parse_args(); package=args.package.resolve(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True); protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['runner_sha256']==sha(__file__)
    assert protocol['source_results_sha256']==sha(package/'extension/results.json')
    assert protocol['guard_sha256']==sha(args.guard/'run_loss_budget_guard.py')
    engine,recovery,guard,source,cfgbase,study,explicit_service,explicit_fork=setup(package,args.guard)
    frozen=json.loads((package/'extension/results.json').read_text())
    if args.phase=='calibrate':
        assert not (out/'prefix_freeze.json').exists() and not (out/'results.json').exists()
        dump(out/'pre_calibration_freeze.json',dict(utc=now(),protocol_sha256=sha(out/'protocol.json'),runner_sha256=sha(__file__)))
        hashes={}
        for name,spec in source['studies'].items():
            data,pre=study(name,spec)
            events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,seed,cfgbase)
                    for seed in spec.get('calibration_seeds',source['calibration_seeds'])]
            streams=[recovery.augment_stream(e,pre,engine.precompute(e,pre,cfgbase),cfgbase) for e in events]
            grid=[]
            for lam in (0.,.25,1.):
                for margin in (0.,4.,12.):
                    cfg=dict(cfgbase,lam=lam,threshold=margin)
                    qi,scores=q_initial(recovery,streams,pre,cfg,'kish_adaptive')
                    rr=[run(engine,recovery,s,pre,cfg,'kish_adaptive',qi,True) for s in streams]
                    grid.append(dict(lam=lam,margin=margin,q_initial=qi,q_fit_scores=scores,
                                     mean_selection_net=float(np.mean([r['selection_net'] for r in rr])),
                                     solver_failures=sum(r['solver_failures'] for r in rr)))
                    print('CAL_STRONG',name,lam,margin,flush=True)
            selected=max(grid,key=lambda x:(x['mean_selection_net'],-x['margin'],-x['lam']))
            path=out/(name+'_prefix_selection.json')
            dump(path,dict(dataset=name,data_hashes=data['hashes'],split=pre['split'],grid=grid,
                           kish_adaptive=selected,mass_equivalent=frozen[name]['selected']['full_transfer']))
            hashes[name]=sha(path);print('SELECT_STRONG',name,selected['lam'],selected['margin'],selected['q_initial'],flush=True)
        dump(out/'prefix_freeze.json',dict(utc=now(),protocol_sha256=sha(out/'protocol.json'),runner_sha256=sha(__file__),selection_sha256=hashes))
        return
    freeze=json.loads((out/'prefix_freeze.json').read_text())
    assert freeze['runner_sha256']==sha(__file__) and freeze['protocol_sha256']==sha(out/'protocol.json')
    if args.phase=='test':
        assert not (out/'results.json').exists(); results={}; checks=[]
        clone=dict(issued_decisions=0,max_weight_error=0.,max_gain_error=0.,max_scale_error=0.,max_q_error=0.,action_mismatches=0)
        for name,spec in source['studies'].items():
            path=out/(name+'_prefix_selection.json');assert sha(path)==freeze['selection_sha256'][name]
            selected=json.loads(path.read_text());data,pre=study(name,spec);assert data['hashes']==selected['data_hashes']
            trials=[]
            for seed in spec.get('test_seeds',source['test_seeds']):
                events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfgbase)
                stream=recovery.augment_stream(events,pre,engine.precompute(events,pre,cfgbase),cfgbase)
                old=next(t for t in frozen[name]['trials'] if t['seed']==seed)
                rr={'full_transfer':old['results']['full_transfer'],'reference':old['results']['reference']}
                for arm in ('mass_equivalent','kish_adaptive'):
                    s=selected[arm];cfg=dict(cfgbase,lam=s['lam'],threshold=s['margin'])
                    r=run(engine,recovery,stream,pre,cfg,arm,s['q_initial'])
                    check=recovery.execute_check(events,pre,cfg,r,explicit_service,explicit_fork)
                    assert check['passed']; checks.append(dict(dataset=name,seed=seed,arm=arm,**check));rr[arm]=r
                for x,y in zip(rr['mass_equivalent']['rows'],rr['full_transfer']['rows']):
                    clone['issued_decisions']+=1
                    for field,key in (('gain','max_gain_error'),('posterior_or_block_sd','max_scale_error'),('q_issued','max_q_error')):
                        clone[key]=max(clone[key],abs(x[field]-y[field]))
                    clone['max_weight_error']=max(clone['max_weight_error'],float(np.max(abs(np.array(x['weights'])-np.array(y['weights'])))))
                    clone['action_mismatches']+=x['action']!=y['action']
                trials.append(dict(seed=seed,windows=stream['windows'],results=rr))
                print('TEST_STRONG',name,seed,{a:(round(r['net'],2),r['deployments']) for a,r in rr.items()},flush=True)
            result=dict(dataset=name,split=pre['split'],selected=selected,trials=trials,summary=control_summary(trials))
            results[name]=result;dump(out/(name+'_results.json'),result)
        assert clone['action_mismatches']==0 and max(clone[k] for k in ('max_gain_error','max_scale_error','max_q_error','max_weight_error'))<1e-8
        dump(out/'results.json',results);dump(out/'summary.json',{n:r['summary'] for n,r in results.items()})
        dump(out/'execution_checks.json',dict(passed=all(c['passed'] for c in checks),trajectories=len(checks),
             forks=sum(c['forks'] for c in checks),max_service_error=max(max(c['service_errors'].values()) for c in checks),
             max_fork_error=max(c['fork_error'] for c in checks),mass_equivalence=clone,checks=checks))
        return
    # Reserve tests include all internal and published-rule controls; the
    # exact mass-equivalent control need not duplicate identical ledgers.
    strong=json.loads((out/'results.json').read_text())
    external=json.loads((package/'extension/external/results.json').read_text())
    trajectories=[];report=dict(service_checks=0,fork_checks=0,max_accounting_error=0.,max_fork_error=0.,
                                max_prefix_negative_loss_violation=0.,max_prefix_net_floor_violation=0.,causality_checks=0)
    for name,spec in source['studies'].items():
        _,pre=study(name,spec)
        for trial in frozen[name]['trials']:
            seed=trial['seed'];events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfgbase)
            reference=explicit_service(events,pre,[],cfgbase)
            arms={a:r for a,r in trial['results'].items() if a!='reference'}
            arms['kish_adaptive']=next(t for t in strong[name]['trials'] if t['seed']==seed)['results']['kish_adaptive']
            ext=next(t for t in external[name]['trials'] if t['seed']==seed)
            arms.update({a:r for a,r in ext['results'].items() if a.startswith(('pdf_','qmf_'))})
            for arm,r in arms.items():
                rows=r['rows'];adaptive={};values=[]
                for row in rows:
                    k=int(row['k']);event=events[k];common=(cfgbase['probe_drop'] if event['probe'] else 0)+(2 if event['refresh'] else 0)
                    assert event['probe'], 'Fixed130 requires a common origin preparation drop'
                    adaptive[k]=guard.current_decision_reserve(event['x'],event['candidate'],event['reference'],common,cfgbase)[0]
                    f=explicit_fork(events,row,cfgbase);err=abs(f['actual']-row['local_net']);report['fork_checks']+=1
                    report['max_fork_error']=max(report['max_fork_error'],err);assert err<1e-8 and f['actual']>=-130
                    values.append(f['actual'])
                values=np.array(values)
                for reserve in RESERVES:
                    reservations=adaptive if reserve=='decision' else {int(z['k']):float(reserve[5:]) for z in rows}
                    for budget in BUDGETS:
                        guarded=guard.replay(rows,trial['windows'],budget,'gross_loss',reservations)
                        audit=explicit_service(events,pre,guarded['rows'],cfgbase)
                        prefix=guard.reconstruct_prefix_increment(events,guarded['rows'],cfgbase)
                        actions=np.array([z['action'] for z in guarded['rows']],bool)
                        inc=float(audit['net']-reference['net']);loss=float(np.maximum(-values[actions],0).sum())
                        err=max(abs(inc-values[actions].sum()),abs(inc-prefix['final']),abs(inc-guarded['final_settled_increment']),abs(loss-guarded['final_spent_loss']))
                        violation=max(0.,prefix['maximum_prefix_negative_loss']-budget);netv=max(0.,-budget-prefix['minimum'])
                        report['service_checks']+=1;report['max_accounting_error']=max(report['max_accounting_error'],float(err))
                        report['max_prefix_negative_loss_violation']=max(report['max_prefix_negative_loss_violation'],violation)
                        report['max_prefix_net_floor_violation']=max(report['max_prefix_net_floor_violation'],netv)
                        assert err<1e-8 and violation==0 and netv==0
                        # Poison later truths; unchanged issued decisions before
                        # the cutoff verify that future outcomes are inaccessible.
                        cutoff=trial['windows']//2
                        poison=guard.replay(rows,trial['windows'],budget,'gross_loss',reservations,perturb_after=cutoff)
                        assert [z['action'] for z in guarded['rows'] if z['k']<=cutoff]==[z['action'] for z in poison['rows'] if z['k']<=cutoff]
                        report['causality_checks']+=1
                        trajectories.append(dict(dataset=name,seed=seed,arm=arm,reserve=reserve,budget=budget,
                              net=audit['net'],reference_net=reference['net'],increment=inc,negative_loss=loss,
                              admissions=int(actions.sum()),harmful=int(np.sum(values[actions]<0)),beneficial=int(np.sum(values[actions]>0)),
                              budget_refusals=sum(z['proposed_action'] and not z['action'] for z in guarded['rows']),
                              minimum_prefix_increment=prefix['minimum'],maximum_prefix_negative_loss=prefix['maximum_prefix_negative_loss'],
                              origins=[z['k'] for z in rows],actions=actions.tolist(),admitted_returns=values[actions].tolist(),
                              ledger=guarded['ledger'],accounting_error=err))
            print('GUARD_STRONG',name,seed,'complete',flush=True)
    summary={}
    for reserve in RESERVES:
        summary[reserve]={}
        for name in strong:
            summary[reserve][name]={}
            for arm in dict.fromkeys(t['arm'] for t in trajectories):
                summary[reserve][name][arm]={}
                for budget in BUDGETS:
                    tt=[t for t in trajectories if t['reserve']==reserve and t['dataset']==name and t['arm']==arm and t['budget']==budget]
                    summary[reserve][name][arm][str(budget)]=dict(mean_increment=float(np.mean([t['increment'] for t in tt])),
                         mean_net=float(np.mean([t['net'] for t in tt])),mean_admissions=float(np.mean([t['admissions'] for t in tt])),
                         increments=[t['increment'] for t in tt],harmful=sum(t['harmful'] for t in tt),beneficial=sum(t['beneficial'] for t in tt),
                         negative_loss=sum(t['negative_loss'] for t in tt),max_negative_loss=max(t['negative_loss'] for t in tt),
                         budget_refusals=sum(t['budget_refusals'] for t in tt),minimum_prefix_increment=min(t['minimum_prefix_increment'] for t in tt))
    dump(out/'guard_results.json',dict(trajectories=trajectories));dump(out/'guard_summary.json',summary)
    dump(out/'guard_verification.json',dict(passed=True,**report))
    with (out/'guard_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['reserve','dataset','arm','budget','mean_increment','mean_net','mean_admissions','harmful','beneficial','negative_loss','max_negative_loss','budget_refusals','minimum_prefix_increment'])
        writer.writeheader()
        for reserve,tasks in summary.items():
            for name,arms in tasks.items():
                for arm,budgets in arms.items():
                    for budget,metrics in budgets.items():writer.writerow(dict(reserve=reserve,dataset=name,arm=arm,budget=budget,**{k:v for k,v in metrics.items() if k!='increments'}))
    print('COMPLETE_STRONG_GUARD',report,flush=True)

if __name__=='__main__': main()
