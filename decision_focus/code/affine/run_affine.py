"""One Mahalanobis complete-value estimator; development amendment only.

All three physical traces were inspected before this amendment. Calibration
selects parameters; complete test outcomes, including losses, are retained.
The previously frozen executable is imported and extended only in memory.
"""
from pathlib import Path
import argparse, datetime, gzip, hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
FACTOR = PROJECT/'work/fusion_strengthening_20261003/matrix/factorized/factorized_control.py'
spec = importlib.util.spec_from_file_location('affine_parent', FACTOR)
parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
core = parent.core
ADAPTER = PROJECT/'work/fusion_strengthening_20261003/fresh/new_data_adapter.py'
spec = importlib.util.spec_from_file_location('affine_fresh_adapter', ADAPTER)
adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
ARMS = ('affine_joint', 'affine_diagonal', 'affine_factorized', 'affine_sandwich')
TASKS = ('rss348', 'arem366', 'gashome362')
CORE = PROJECT/'work/fusion_temporal_20261003/temporal_fusion.py'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v):
    s = json.dumps(v, indent=2, allow_nan=False)
    if str(p).endswith('.gz'):
        with gzip.open(p, 'wt') as f: f.write(s)
    else: Path(p).write_text(s)
def load(p):
    if str(p).endswith('.gz'):
        with gzip.open(p, 'rt') as f: return json.load(f)
    return json.loads(Path(p).read_text())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def affine_forecast(d, pre, cfg, arm, q):
    if arm not in ARMS: return parent.factorized_forecast(d, pre, cfg, arm, q)
    mm = d['temporal']; mu = mm['mu']; C = mm['R'] + mm['P']
    if arm == 'affine_diagonal': C = np.diag(np.diag(C))
    if arm == 'affine_factorized':
        mu = mm['factorized_mu']; C = mm['factorized_R'] + mm['factorized_P']
    if arm == 'affine_sandwich': C = mm['R'] + mm['sandwich_P']
    one = np.ones(len(mu)); invone = np.linalg.solve(C, one)
    variance = 1. / float(one @ invone); w = variance * invone
    rate = d['h'] - mu; t = float(w @ rate)
    # h-t1 is an action-error observation only after the complete target
    # matures; its source differences D h are available at issuance.
    D = np.concatenate((np.eye(len(mu)-1), -np.ones((len(mu)-1, 1))), axis=1)
    difference = D @ rate
    conditioned_mu = mu.copy()
    conditioned_cov = C.copy()
    if len(mu) > 1:
        middle = D @ C @ D.T
        conditioned_mu += C @ D.T @ np.linalg.solve(middle, difference)
        conditioned_cov -= C @ D.T @ np.linalg.solve(middle, D @ C)
    mean_identity = float(np.max(np.abs((d['h']-conditioned_mu)-t)))
    covariance_identity = float(np.max(np.abs(conditioned_cov-variance*np.outer(one, one))))
    residual = float(np.max(np.abs(C @ w-variance*one)))
    assert mean_identity < 1e-8 and covariance_identity < 1e-8
    ready = bool(mm['eligible'])
    raw_gain = float(d['N']*t); gain = raw_gain if ready else 0.
    scale = float(d['N']*np.sqrt(variance+cfg['norm_floor']**2))
    # Offline outcome is returned only for the delayed callback, never used
    # by the estimator, its variance, readiness or admission score.
    standardized = (gain-d['truegross'])/scale
    cert = dict(converged=True, kkt=residual)
    extra = dict(information_ready=ready, raw_optimized_gain=raw_gain,
                 affine_rate=t, affine_variance=variance, affine_mu=mu.tolist(),
                 affine_C=C.tolist(), disagreement_innovation=difference.tolist(),
                 conditioned_error_mean=conditioned_mu.tolist(),
                 conditioned_error_covariance=conditioned_cov.tolist(),
                 conditional_mean_identity_error=mean_identity,
                 conditional_covariance_identity_error=covariance_identity,
                 negative_influences=int(np.sum(w<0)),
                 rate_outside_physical_support=bool(abs(t)>1))
    return w, cert, gain, scale, standardized, extra

core.forecast = affine_forecast

def binding():
    return parent.matrix.binding()

def task_data(name, cfg, engine, source, study):
    if name == 'rss348': return study(name, source['studies'][name])
    data, derivation = adapter.load_dataset(PROJECT/'work/fusion_strengthening_20261003/fresh/data', name)
    return data, adapter.make_prefix(data, derivation, cfg, engine)

def seeds(name):
    return ([87001,87002,87003], list(range(88001,88006))) if name=='rss348' else ([91001,91002,91003], list(range(92001,92006)))

def freeze():
    assert not (ROOT/'protocol.json').exists()
    inputs = [CORE, FACTOR, parent.MATRIX_SOURCE, ADAPTER]
    dump(ROOT/'protocol.json', dict(utc=now(), runner_sha256=sha(__file__),
        input_sha256={str(p):sha(p) for p in inputs}, tasks=TASKS, arms=ARMS,
        status='development amendment on previously examined physical traces; not independent validation',
        estimator='t_hat=(1^T C^-1(h-mu))/(1^T C^-1 1); v=(1^T C^-1 1)^-1',
        covariance='C=R+P; diagonal holds full mu fixed; factorized rebuilds mu; sandwich replaces P only',
        gate='N*t_hat-5-q*N*sqrt(v+0.01^2)>0; no posterior kappa deduction, no entropy, no source cap',
        signed_influences=True, clipping=False,
        source_difference='D1=0 so D(h-t1)=Dh is label-free; exact conditioning under working Gaussian law only',
        calibration=dict(sd=[.05,.1,.2],q_floor=[0.,.64,1.2815515655446004],
            trials_per_arm=27, q_initial='matured first-half 90th percentile',
            selection='maximum second-half mean net; tie higher floor then lower sd', task_seeds={n:seeds(n) for n in TASKS}),
        primary_guard=dict(reserve=130.,budget=130.,loss='gross negative complete return; no replenishment'),
        diagnostic_budgets=[0.,110.,130.,260.],
        truth='all test outcomes and all tuning trials retained; no adaptive revision during held-out replay',
        shared='same source models, contexts, arrived feedback, full leases, masks, quality, kernel, archive and costs',
        runtime=dict(python=sys.version,numpy=np.__version__)))

def verify():
    p=load(ROOT/'protocol.json'); assert p['runner_sha256']==sha(__file__)
    for path, expected in p['input_sha256'].items(): assert sha(path)==expected
    return p

def calibrate():
    verify(); engine,recovery,guard,source,cfg0,study,service,fork=binding()
    for name in TASKS:
        assert not (ROOT/(name+'_selection.json')).exists()
        data,pre=task_data(name,cfg0,engine,source,study); cal,_=seeds(name)
        events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,cfg0) for s in cal]
        bases=[engine.precompute(e,pre,cfg0) for e in events];grid=[];trials=[]
        for sd in (.05,.1,.2):
            cfg=dict(cfg0,prior_sd=sd); streams=[core.augment(e,pre,b,cfg) for e,b in zip(events,bases)]
            for arm in ARMS:
                for floor in (0.,.64,1.2815515655446004):
                    cc=dict(cfg,q_floor=floor,threshold=0.); qi,scores=core.initial_q(streams,pre,cc,arm)
                    rr=[core.run(engine,s,pre,cc,arm,qi,True) for s in streams]
                    g=dict(arm=arm,prior_sd=sd,q_floor=floor,margin=0.,q_initial=qi,
                           fit_scores=scores,net=float(np.mean([r['selection_net'] for r in rr])))
                    grid.append(g);trials.append(dict(configuration=g,results=rr))
                print('CAL',name,sd,arm,flush=True)
        selected={a:max((g for g in grid if g['arm']==a),key=lambda g:(g['net'],g['q_floor'],-g['prior_sd'])) for a in ARMS}
        dump(ROOT/(name+'_calibration_trials.json.gz'),trials)
        dump(ROOT/(name+'_selection.json'),dict(selected=selected,grid=grid,split=pre['split'],data_hashes=data['hashes']))
        print('SELECT',name,{a:(s['prior_sd'],s['q_floor'],s['q_initial'],s['net']) for a,s in selected.items()},flush=True)
    dump(ROOT/'before_test.json',dict(utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),
         selections={n:sha(ROOT/(n+'_selection.json')) for n in TASKS}))

def test():
    protocol=verify();fr=load(ROOT/'before_test.json');assert fr['protocol_sha256']==sha(ROOT/'protocol.json')
    engine,recovery,guard,source,cfg0,study,service,fork=binding()
    for name in TASKS:
        assert not (ROOT/(name+'_results.json.gz')).exists()
        selection=load(ROOT/(name+'_selection.json'));assert sha(ROOT/(name+'_selection.json'))==fr['selections'][name]
        data,pre=task_data(name,cfg0,engine,source,study);assert data['hashes']==selection['data_hashes']
        _,testseeds=seeds(name);trials=[];guarded=[];maxerr=0.
        for seed in testseeds:
            events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg0)
            base=engine.precompute(events,pre,cfg0);streams={};results={};checks={};reference=service(events,pre,[],cfg0)
            for arm in ARMS:
                ss=selection['selected'][arm];cfg=dict(cfg0,prior_sd=ss['prior_sd'],q_floor=ss['q_floor'],threshold=0.)
                if ss['prior_sd'] not in streams:streams[ss['prior_sd']]=core.augment(events,pre,base,cfg)
                result=core.run(engine,streams[ss['prior_sd']],pre,cfg,arm,ss['q_initial'])
                check=recovery.execute_check(events,pre,cfg,result,service,fork);assert check['passed']
                results[arm]=result;checks[arm]=check
                for budget in protocol['diagnostic_budgets']:
                    gg=guard.replay(result['rows'],len(events),budget,'gross_loss',{r['k']:130. for r in result['rows']})
                    actual=service(events,pre,gg['rows'],cfg);prefix=guard.reconstruct_prefix_increment(events,gg['rows'],cfg)
                    act=[r for r in gg['rows'] if r['action']];inc=actual['net']-reference['net'];loss=sum(max(0.,-r['local_net']) for r in act)
                    err=max(abs(inc-sum(r['local_net'] for r in act)),abs(inc-prefix['final']),abs(inc-gg['final_settled_increment']))
                    maxerr=max(maxerr,err);assert err<1e-8 and loss<=budget+1e-8 and prefix['minimum']>=-budget-1e-8
                    assert all(l['spent_loss']+l['reserved']<=budget+1e-8 for l in gg['ledger'])
                    guarded.append(dict(task=name,seed=seed,arm=arm,budget=budget,reserve=130.,
                        increment=inc,net=actual['net'],negative_loss=loss,minimum_prefix=prefix['minimum'],
                        admissions=len(act),harmful=sum(r['local_net']<0 for r in act),
                        beneficial=sum(r['local_net']>0 for r in act),zero=sum(r['local_net']==0 for r in act),
                        refused=sum(r['proposed_action'] and not r['action'] for r in gg['rows']),
                        rows=gg['rows'],ledger=gg['ledger']))
            trials.append(dict(seed=seed,reference=reference,results=results,checks=checks))
            print('TEST',name,seed,{a:(round(r['net']-reference['net'],2),r['harmful'],r['beneficial']) for a,r in results.items()},flush=True)
        dump(ROOT/(name+'_results.json.gz'),dict(trials=trials,selected=selection['selected']))
        dump(ROOT/(name+'_guarded.json.gz'),guarded)
        dump(ROOT/(name+'_verification.json'),dict(passed=True,max_service_error=maxerr,unguarded=20,guarded=len(guarded),unchanged_core=sha(CORE)==protocol['input_sha256'][str(CORE)]))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','calibrate','test']);args=ap.parse_args()
    {'freeze':freeze,'calibrate':calibrate,'test':test}[args.phase]()
