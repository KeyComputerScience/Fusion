"""Frozen-core deterministic GLS and empirical joint-block sandwich controls.

Writes exclusively beneath this runner's directory. The imported frozen core
is extended only in memory; it is never edited. Formula and protocol are
declared before calibration or test execution.
"""
from __future__ import annotations
import argparse, datetime, gzip, hashlib, importlib.util, json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
FROZEN = PROJECT / 'work/fusion_temporal_20261003'
CORE = FROZEN / 'temporal_fusion.py'
spec = importlib.util.spec_from_file_location('frozen_temporal_matrix_reference', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
original_state, original_forecast = core.state, core.forecast
ARMS = ('gls_exact', 'block_sandwich')
RISK_FLOOR = 1e-6

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path, obj):
    text = json.dumps(obj, indent=2, allow_nan=False)
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt') as f: f.write(text)
    else: Path(path).write_text(text)
def load(path): return json.loads(Path(path).read_text())

def collect_blocks(arrived, current, pre, cfg):
    """Reconstruct complete lease residuals without imputing missing sources."""
    observations, complete = [], []
    for record in arrived[-cfg['lease_archive']:]:
        weight = cfg['beta'] ** (current['k'] - record['k'])
        weight *= math.exp(-float(np.sum((current['context']-record['context'])**2)) / cfg['kernel_bandwidth']**2)
        if weight < 1e-12: continue
        reference = (record['x'] @ current['reference']).argmax(1)
        candidate = (record['x'] @ current['candidate']).argmax(1)
        contrast = np.eye(pre['classes'])[candidate] - np.eye(pre['classes'])[reference]
        ids = np.flatnonzero(record['mask'])
        issued = np.einsum('sic,ic->s', record['p'][ids], contrast) / len(record['x'])
        rates, count, total = [], 0, 0
        for future in record['future']:
            p0 = (future['x'] @ current['reference']).argmax(1)
            p1 = (future['x'] @ current['candidate']).argmax(1)
            keep = future['keep']
            rates.append(float(np.sum((p1[keep] == future['y'][keep]).astype(float)
                                      - (p0[keep] == future['y'][keep]).astype(float))) / len(record['x']))
            count += int(np.sum(p1 != p0)); total += len(p1)
        weight *= count / max(1, total)
        if weight < 1e-12: continue
        residual = issued - float(np.sum(rates)) / cfg['horizon']
        observations.append(dict(weight=weight, ids=ids, residual=residual,
                                 origin=record['k'], maturity=record['maturity']))
        if len(ids) == pre['m']: complete.append((weight, residual))
    return observations, complete

def matrix_state(arrived, current, pre, cfg):
    mm = original_state(arrived, current, pre, cfg)
    observations, complete = collect_blocks(arrived, current, pre, cfg)
    R, _ = core.covariance(complete, pre['m'], cfg['prior_mass'], cfg['prior_variance'])
    quality = pre['q'] / np.mean(pre['q'])
    # Deterministic quality-weighted ridge: minimize
    # 0.5 theta' K theta + 0.5 sum a_j (z_j-H_j theta)' R_j^-1 (z_j-H_j theta).
    K = np.diag(quality / cfg['prior_sd']**2)
    A, rhs = K.copy(), np.zeros(pre['m'])
    inverses = []
    for block in observations:
        ix, a = block['ids'], block['weight']
        inverse = np.linalg.solve(R[np.ix_(ix, ix)], np.eye(len(ix)))
        A[np.ix_(ix, ix)] += a * inverse
        rhs[ix] += a * (inverse @ block['residual'])
        inverses.append(inverse)
    bread = np.linalg.solve(A, np.eye(pre['m']))
    bread = (bread + bread.T) / 2
    mu = np.linalg.solve(A, rhs)
    # One cluster per complete matured lease. Each outer product preserves
    # its jointly observed physical-source coordinates and is PSD.
    meat = np.zeros_like(A)
    for block, inverse in zip(observations, inverses):
        ix = block['ids']
        score = np.zeros(pre['m'])
        score[ix] = block['weight'] * (inverse @ (block['residual'] - mu[ix]))
        meat += np.outer(score, score)
    raw = bread @ meat @ bread.T
    raw = (raw + raw.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(raw)
    sandwich = (eigenvectors * np.maximum(eigenvalues, RISK_FLOOR)) @ eigenvectors.T
    ix = current['ids']
    gls_mu, gls_P = mu[ix], bread[np.ix_(ix, ix)]
    mu_error = float(np.max(np.abs(gls_mu - mm['mu'])))
    P_error = float(np.max(np.abs(gls_P - mm['P'])))
    assert mu_error < 1e-10 and P_error < 1e-10
    assert [b['origin'] for b in observations] == [b['origin'] for b in mm['eligible']]
    assert np.linalg.eigvalsh(sandwich).min() > 0
    assert all(b['maturity'] <= current['k'] for b in observations)
    mm.update(gls_mu=gls_mu, gls_P=gls_P,
              sandwich_P=sandwich[np.ix_(ix, ix)],
              sandwich_raw_eigenvalues=eigenvalues.tolist(),
              gls_mu_error=mu_error, gls_P_error=P_error,
              sandwich_clusters=len(observations), sandwich_meat_rank=int(np.linalg.matrix_rank(meat)),
              full_matrix_dimension=pre['m'])
    return mm

def matrix_forecast(d, pre, cfg, arm, q):
    if arm not in ARMS: return original_forecast(d, pre, cfg, arm, q)
    mm = d['temporal']
    changed = dict(mm)
    if arm == 'gls_exact': changed.update(mu=mm['gls_mu'], P=mm['gls_P'])
    else: changed.update(P=mm['sandwich_P'])
    decision = dict(d, temporal=changed)
    w, cert, F, S, T, extra = original_forecast(decision, pre, cfg, 'precision_joint', q)
    extra.update(matrix_control=arm, adaptive_risk_matrix=changed['P'].tolist(),
                 gls_mu_error=mm['gls_mu_error'], gls_P_error=mm['gls_P_error'],
                 sandwich_clusters=mm['sandwich_clusters'],
                 sandwich_raw_eigenvalues=mm['sandwich_raw_eigenvalues'],
                 sandwich_meat_rank=mm['sandwich_meat_rank'])
    return w, cert, F, S, T, extra

core.state, core.forecast = matrix_state, matrix_forecast

def binding():
    engine, recovery, guard, source, cfg, study, service, fork = core.support.setup(
        core.support.DEFAULT_PACKAGE, core.support.DEFAULT_GUARD)
    cfg.update(prior_sd=.1, tau=.01, q_floor=1.2815515655446004)
    sys.path.insert(0, str(PROJECT/'work/fusion_focus_20261003/fresh'))
    from new_data_adapter import load_dataset, make_prefix
    old_study = study
    def study(name, entry):
        if name == 'rss348':
            data, derivation = load_dataset(FROZEN/'physical/data', name)
            return data, make_prefix(data, derivation, cfg, engine)
        return old_study(name, entry)
    source['studies']['rss348'] = dict(calibration_seeds=[87001,87002,87003],
                                     test_seeds=[88001,88002,88003,88004,88005])
    return engine, recovery, guard, source, cfg, study, service, fork

def freeze():
    assert not (OUT/'protocol.json').exists()
    inputs = [CORE, PROJECT/'work/fusion_focus_20261003/controls/run_strong_controls.py',
              PROJECT/'work/fusion_focus_20261003/fresh/new_data_adapter.py',
              FROZEN/'physical/data/rss348/cache_metadata.json',
              FROZEN/'physical/data/rss348/cached_dataset.npz',
              FROZEN/'rss_validation/protocol.json', FROZEN/'rss_validation/rss348_selection.json',
              FROZEN/'rss_validation/rss348_results.json',
              FROZEN/'final_development/gas224_results.json']
    protocol = dict(utc=now(), runner_sha256=sha(__file__), frozen_input_sha256={str(p):sha(p) for p in inputs},
        arms=list(ARMS), tasks=dict(rss348='previously examined frozen validation trace; new comparator is post hoc',
                                  gas224='development only; fixed-state interventions only'),
        formula=dict(ridge='K=diag((q_s/mean(q))/prior_sd^2)',
                     bread='A=K+sum_j a_j H_j^T (H_j R H_j^T)^(-1) H_j; B=A^(-1)',
                     mean='theta_hat=A^(-1) sum_j a_j H_j^T R_j^(-1) z_j',
                     gls_exact_risk='B: deterministic inverse curvature exactly matches the frozen precision matrix',
                     cluster_score='psi_j=a_j H_j^T R_j^(-1)(z_j-H_j theta_hat)',
                     sandwich_raw='B [sum_j psi_j psi_j^T] B',
                     sandwich_stabilization='raise global raw sandwich eigenvalues below 1e-6 to 1e-6, then restrict active IDs',
                     no_small_sample_rescaling=True,
                     interpretation='empirical HC0 block sandwich of penalized estimating equations; ridge is deterministic, not a random prior'),
        shared='identical matured masks, complete lease blocks, context/support/age weights, R, quality and ridge; same solver, norm floor, entropy, gate and delayed callbacks',
        calibration=dict(prior_sd=[.05,.1,.2],q_floor=[0.,.64,1.2815515655446004],
            seeds=[87001,87002,87003],evaluations_per_arm=27,
            q_initial='ready first-half completely matured pilot scores; higher 90th percentile clipped at floor',
            selection='maximum second-half mean complete net return; ties higher floor then lower prior_sd'),
        test=dict(seeds=[88001,88002,88003,88004,88005],
            no_test_retuning=True,reserves=['fixed130','decision'],budgets=[0,110,130,260]),
        fixed_state='all original full-controller RSS and Gas issued states; identical current mu,R,q,quality,h and local return; replace only risk matrix; enumerate every changed action',
        severity='S*(T-q)_+=max(F-g-q*S,0), summed separately over all issued, informative, proposed and admitted origins',
        limits=['HC0 is a matrix comparator, not a distribution-free confidence sequence',
                'reweighted lease blocks can remain dependent; no asymptotic or finite-sample coverage claim',
                'ridge shrinkage bias is not represented by HC0 sampling covariance',
                'RSS test was already inspected for the original study; freezing this extension does not make it a new independent task',
                'five delays share one held-out physical trace; Gas is development'],
        runtime=dict(python=sys.version,numpy=np.__version__))
    dump(OUT/'protocol.json',protocol)
    print('FROZEN_PROTOCOL',sha(OUT/'protocol.json'),flush=True)

def verify_freeze():
    protocol=load(OUT/'protocol.json');assert protocol['runner_sha256']==sha(__file__)
    for path, expected in protocol['frozen_input_sha256'].items(): assert sha(path)==expected, path
    return protocol

def calibrate():
    verify_freeze();assert not (OUT/'rss348_selection.json').exists()
    engine,recovery,guard,source,cfg0,study,service,fork=binding()
    data,pre=study('rss348',source['studies']['rss348'])
    events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,cfg0) for s in source['studies']['rss348']['calibration_seeds']]
    bases=[engine.precompute(e,pre,cfg0) for e in events]
    grid=[];all_trials=[]
    for sd in (.05,.1,.2):
        cfg=dict(cfg0,prior_sd=sd)
        streams=[core.augment(e,pre,b,cfg) for e,b in zip(events,bases)]
        for arm in ARMS:
            for floor in (0.,.64,1.2815515655446004):
                cc=dict(cfg,threshold=0.,q_floor=floor);qi,scores=core.initial_q(streams,pre,cc,arm)
                rr=[core.run(engine,s,pre,cc,arm,qi,True) for s in streams]
                entry=dict(arm=arm,prior_sd=sd,margin=0.,q_floor=floor,q_initial=qi,fit_scores=scores,
                           net=float(np.mean([r['selection_net'] for r in rr])),failures=sum(r['solver_failures'] for r in rr))
                grid.append(entry);all_trials.append(dict(configuration=entry,results=rr))
                assert entry['failures']==0
                print('CAL_MATRIX',sd,arm,floor,round(entry['net'],6),flush=True)
    selected={a:max((g for g in grid if g['arm']==a),key=lambda g:(g['net'],g['q_floor'],-g['prior_sd'])) for a in ARMS}
    original=load(FROZEN/'rss_validation/rss348_selection.json')
    equivalence=[]
    for g in grid:
        if g['arm']!='gls_exact':continue
        match=next(x for x in original['grid'] if x['arm']=='precision_joint' and x['prior_sd']==g['prior_sd'] and x['q_floor']==g['q_floor'])
        equivalence.append(dict(prior_sd=g['prior_sd'],q_floor=g['q_floor'],
                                net_error=abs(g['net']-match['net']),q_initial_error=abs(g['q_initial']-match['q_initial'])))
    assert max(x['net_error'] for x in equivalence)<1e-8
    assert max(x['q_initial_error'] for x in equivalence)<1e-8
    assert all(selected['gls_exact'][key]==original['selected']['precision_joint'][key] for key in ('prior_sd','q_floor','margin'))
    dump(OUT/'rss348_calibration_trials.json.gz',all_trials)
    dump(OUT/'rss348_selection.json',dict(data_hashes=data['hashes'],split=pre['split'],grid=grid,selected=selected,
                                        exact_calibration_equivalence=equivalence,original_full_selected=original['selected']['precision_joint']))
    dump(OUT/'selection_freeze.json',dict(utc=now(),selection_sha256=sha(OUT/'rss348_selection.json'),
                                        trials_sha256=sha(OUT/'rss348_calibration_trials.json.gz'),protocol_sha256=sha(OUT/'protocol.json')))
    print('SELECT_MATRIX',{a:(v['prior_sd'],v['q_floor'],v['q_initial'],v['net']) for a,v in selected.items()},flush=True)

def severity(rows):
    groups=dict(issued=rows,informative=[r for r in rows if r['disagreement']>0],
                proposed=[r for r in rows if r['gate_score']>0],admitted=[r for r in rows if r['action']])
    return {k:dict(n=len(rr),violations=sum(r['standardized_score']>r['q_issued'] for r in rr),
                   coverage=[sum(r['lower_covered'] for r in rr),len(rr)],
                   total_scaled_excess=sum(max(0.,r['gain']-r['truegross']-r['q_issued']*r['posterior_or_block_sd']) for r in rr),
                   max_scaled_excess=max((max(0.,r['gain']-r['truegross']-r['q_issued']*r['posterior_or_block_sd']) for r in rr),default=0.))
            for k,rr in groups.items()}

def test():
    verify_freeze();assert not (OUT/'rss348_results.json').exists()
    selection=load(OUT/'rss348_selection.json');sf=load(OUT/'selection_freeze.json')
    assert sf['selection_sha256']==sha(OUT/'rss348_selection.json') and sf['protocol_sha256']==sha(OUT/'protocol.json')
    engine,recovery,guard,source,cfg0,study,service,fork=binding()
    data,pre=study('rss348',source['studies']['rss348']);assert data['hashes']==selection['data_hashes']
    original=load(FROZEN/'rss_validation/rss348_results.json');trials=[];guard_trials=[]
    equivalence=dict(decisions=0,action_mismatches=0,max_weight_error=0.,max_gain_error=0.,max_q_error=0.,max_scale_error=0.,max_mean_error=0.,max_risk_matrix_error=0.)
    for seed in source['studies']['rss348']['test_seeds']:
        events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg0)
        base=engine.precompute(events,pre,cfg0);streams={};results={};checks={}
        old=next(t for t in original['trials'] if t['seed']==seed)
        for arm in ARMS:
            ss=selection['selected'][arm];cfg=dict(cfg0,prior_sd=ss['prior_sd'],threshold=0.,q_floor=ss['q_floor'])
            if ss['prior_sd'] not in streams:streams[ss['prior_sd']]=core.augment(events,pre,base,cfg)
            result=core.run(engine,streams[ss['prior_sd']],pre,cfg,arm,ss['q_initial'])
            check=recovery.execute_check(events,pre,cfg,result,service,fork);assert check['passed'] and result['solver_failures']==0
            results[arm]=result;checks[arm]=check
        results['precision_joint']=old['results']['precision_joint'];results['reference']=old['results']['reference']
        for x,y in zip(results['gls_exact']['rows'],results['precision_joint']['rows']):
            equivalence['decisions']+=1;equivalence['action_mismatches']+=x['action']!=y['action']
            for field,key in [('gain','max_gain_error'),('q_issued','max_q_error'),('posterior_or_block_sd','max_scale_error')]:equivalence[key]=max(equivalence[key],abs(x[field]-y[field]))
            equivalence['max_weight_error']=max(equivalence['max_weight_error'],float(np.max(np.abs(np.array(x['weights'])-np.array(y['weights'])))))
            equivalence['max_mean_error']=max(equivalence['max_mean_error'],x['gls_mu_error'])
            equivalence['max_risk_matrix_error']=max(equivalence['max_risk_matrix_error'],x['gls_P_error'])
        for arm in ('precision_joint',)+ARMS:
            r=results[arm];reserves={}
            for row in r['rows']:
                event=events[row['k']];common=cfg0['probe_drop']+(2 if event['refresh'] else 0)
                reserves[row['k']]=guard.current_decision_reserve(event['x'],event['candidate'],event['reference'],common,cfg0)[0]
            for rule in ('fixed130','decision'):
                rs=reserves if rule=='decision' else {row['k']:130. for row in r['rows']}
                for budget in (0,110,130,260):
                    g=guard.replay(r['rows'],len(events),budget,'gross_loss',rs)
                    actual=service(events,pre,g['rows'],cfg0);prefix=guard.reconstruct_prefix_increment(events,g['rows'],cfg0)
                    actions=[row for row in g['rows'] if row['action']]
                    delta=actual['net']-results['reference']['net'];loss=sum(max(0.,-row['local_net']) for row in actions)
                    error=max(abs(delta-sum(row['local_net'] for row in actions)),abs(delta-prefix['final']),abs(loss-g['final_spent_loss']))
                    assert error<1e-8 and loss<=budget+1e-8 and prefix['minimum']>=-budget-1e-8
                    guard_trials.append(dict(seed=seed,arm=arm,reserve=rule,budget=budget,increment=delta,
                        admissions=len(actions),harmful=sum(row['local_net']<0 for row in actions),beneficial=sum(row['local_net']>0 for row in actions),
                        negative_loss=loss,budget_refusals=sum(row['proposed_action'] and not row['action'] for row in g['rows']),
                        accounting_error=error,minimum_prefix_increment=prefix['minimum'],ledger=g['ledger']))
        trials.append(dict(seed=seed,results=results,checks=checks))
        print('TEST_MATRIX',seed,{a:(round(r['net'],6),r['harmful'],r['beneficial']) for a,r in results.items()},flush=True)
    assert equivalence['action_mismatches']==0
    assert max(v for k,v in equivalence.items() if k.startswith('max_'))<1e-8
    summary=core.summarize(trials)
    for arm in ('precision_joint',)+ARMS:
        summary[arm]['severity']=severity([r for t in trials for r in t['results'][arm]['rows']])
        differences=np.array([t['results']['precision_joint']['net']-t['results'][arm]['net'] for t in trials])
        half=2.776445105*float(differences.std(ddof=1))/math.sqrt(5)
        summary[arm]['full_minus_control']=dict(values=differences.tolist(),mean=float(differences.mean()),conditional_t4_ci=[float(differences.mean()-half),float(differences.mean()+half)])
    dump(OUT/'rss348_results.json',dict(protocol_sha256=sha(OUT/'protocol.json'),selection=selection['selected'],trials=trials,summary=summary,exact_equivalence=equivalence))
    dump(OUT/'rss348_budget_results.json.gz',guard_trials)
    print('EXACT_EQUIVALENCE',equivalence,flush=True)

def interventions():
    verify_freeze();assert not (OUT/'fixed_state_interventions.json').exists()
    engine,recovery,guard,source,cfg0,study,service,fork=binding()
    output={}
    for name,folder in [('rss348','rss_validation'),('gas224','final_development')]:
        old=load(FROZEN/folder/(name+'_results.json'));data,pre=study(name,source['studies'][name]);all_rows=[];changes=[]
        ss=old['selected']['precision_joint'];cfg=dict(cfg0,prior_sd=ss['prior_sd'],threshold=0.,q_floor=ss['q_floor'])
        for trial in old['trials']:
            seed=trial['seed'];events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg0)
            stream=core.augment(events,pre,engine.precompute(events,pre,cfg0),cfg)
            for d,row in zip(stream['decisions'],trial['results']['precision_joint']['rows']):
                assert np.max(np.abs(np.array(row['joint_mu'])-d['temporal']['mu']))<1e-8
                for arm in ARMS:
                    w,cert,F,S,T,extra=matrix_forecast(d,pre,cfg,arm,row['q_issued'])
                    assert cert['converged'];score=F-row['q_issued']*S-5;action=score>0
                    rebuilt=dict(row,action=bool(action),weights=w.tolist(),gain=F,gate_score=score,standardized_score=T,lower_covered=bool(T<=row['q_issued']),posterior_or_block_sd=S)
                    all_rows.append(dict(rebuilt,seed=seed,arm=arm,**extra))
                    if arm=='gls_exact':assert action==row['action'] and abs(score-row['gate_score'])<1e-8
                    if action!=row['action']:
                        D=row['local_net'];full_action=bool(row['action'])
                        category=('retained_gain' if D>0 else 'incurred_loss' if D<0 else 'zero_return') if full_action else ('missed_gain' if D>0 else 'avoided_loss' if D<0 else 'zero_return')
                        changes.append(dict(seed=seed,arm=arm,k=row['k'],category=category,full_action=full_action,control_action=bool(action),complete_return=D,
                            full_minus_control_return=(int(full_action)-int(action))*D,
                            full_score=row['gate_score'],control_score=score,full_weights=row['weights'],control_weights=w.tolist(),
                            full_scaled_excess=max(0.,row['gain']-row['truegross']-row['q_issued']*S),
                            control_scaled_excess=max(0.,F-row['truegross']-row['q_issued']*S),
                            identical_mu=d['temporal']['mu'].tolist(),identical_R=d['temporal']['R'].tolist(),identical_q=row['q_issued'],
                            full_P=d['temporal']['P'].tolist(),control_P=extra['adaptive_risk_matrix']))
        summaries={}
        for arm in ARMS:
            cc=[c for c in changes if c['arm']==arm]
            summaries[arm]=dict(issued_states=sum(r['arm']==arm for r in all_rows),changed_actions=len(cc),
                full_minus_control_return=sum(c['full_minus_control_return'] for c in cc),
                categories={cat:dict(count=sum(c['category']==cat for c in cc),absolute_return=sum(abs(c['complete_return']) for c in cc if c['category']==cat)) for cat in ('retained_gain','missed_gain','avoided_loss','incurred_loss','zero_return')},
                severity=severity([r for r in all_rows if r['arm']==arm]))
        output[name]=dict(scope='fixed original full states; not dynamic policy gain',summary=summaries,changes=changes)
        dump(OUT/(name+'_fixed_state_all_rows.json.gz'),all_rows)
        print('FIXED_MATRIX',name,{a:(v['changed_actions'],v['full_minus_control_return']) for a,v in summaries.items()},flush=True)
    dump(OUT/'fixed_state_interventions.json',output)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('freeze','calibrate','test','interventions'),required=True);args=parser.parse_args()
    dict(freeze=freeze,calibrate=calibrate,test=test,interventions=interventions)[args.phase]()
    if args.phase!='freeze':verify_freeze()
if __name__=='__main__':main()
