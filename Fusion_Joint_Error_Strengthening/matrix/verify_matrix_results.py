"""Read-only audit of saved matrix controls; writes only its summary report."""
import datetime, gzip, hashlib, json, math, sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt') as f:return json.load(f)
    return json.loads(Path(p).read_text())
protocol=load(ROOT/'protocol.json')
assert sha(ROOT/'run_matrix_controls.py')==protocol['runner_sha256']
assert all(sha(path)==digest for path,digest in protocol['frozen_input_sha256'].items())
selection=load(ROOT/'rss348_selection.json');results=load(ROOT/'rss348_results.json')
fixed=load(ROOT/'fixed_state_interventions.json');budget=load(ROOT/'rss348_budget_results.json.gz')
calibration=load(ROOT/'rss348_calibration_trials.json.gz')
assert len(calibration)==18
assert all(len(t['results'])==3 for t in calibration)
assert len(budget)==120
matrix_min=math.inf;max_kkt=0.;max_accounting=0.;issued=0
for trial in results['trials']:
    for arm in ('gls_exact','block_sandwich'):
        r=trial['results'][arm]
        assert r['solver_failures']==0
        max_kkt=max(max_kkt,r['max_kkt'])
        assert trial['checks'][arm]['passed']
        for row in r['rows']:
            issued+=1
            P=np.array(row['adaptive_risk_matrix'])
            assert P.shape==(len(row['active_source_ids']),)*2
            matrix_min=min(matrix_min,float(np.linalg.eigvalsh((P+P.T)/2).min()))
            assert np.max(np.abs(P-P.T))<1e-10
            assert all(b['maturity']<=row['k'] for b in row['eligible'])
for t in budget:
    max_accounting=max(max_accounting,t['accounting_error'])
    assert t['negative_loss']<=t['budget']+1e-8
    assert t['minimum_prefix_increment']>=-t['budget']-1e-8
assert matrix_min>0
budget_summary={}
for arm in ('precision_joint','gls_exact','block_sandwich'):
    budget_summary[arm]={}
    for reserve in ('fixed130','decision'):
        budget_summary[arm][reserve]={}
        for value in (0,110,130,260):
            rows=[t for t in budget if t['arm']==arm and t['reserve']==reserve and t['budget']==value]
            budget_summary[arm][reserve][str(value)]=dict(
                increments=[t['increment'] for t in rows],
                mean_increment=float(np.mean([t['increment'] for t in rows])),
                admissions=sum(t['admissions'] for t in rows),harmful=sum(t['harmful'] for t in rows),
                beneficial=sum(t['beneficial'] for t in rows),budget_refusals=sum(t['budget_refusals'] for t in rows),
                total_negative_loss=sum(t['negative_loss'] for t in rows))
for arm in ('precision_joint','gls_exact','block_sandwich'):
    results['summary'][arm]['increments']=[t['results'][arm]['net']-t['results']['reference']['net'] for t in results['trials']]
    results['summary'][arm]['mean_increment']=float(np.mean(results['summary'][arm]['increments']))
def decomposition(reserve=None,value=None):
    categories={x:dict(count=0,absolute_return=0.) for x in ('retained_gain','missed_gain','avoided_loss','incurred_loss','zero_return')}
    cases=[];total=0.
    for trial in results['trials']:
        seed=trial['seed'];full=trial['results']['precision_joint']['rows'];control=trial['results']['block_sandwich']['rows']
        if reserve:
            f=next(x for x in budget if x['seed']==seed and x['arm']=='precision_joint' and x['reserve']==reserve and x['budget']==value)
            c=next(x for x in budget if x['seed']==seed and x['arm']=='block_sandwich' and x['reserve']==reserve and x['budget']==value)
            fa={r['k']:r['action'] for r in f['ledger']};ca={r['k']:r['action'] for r in c['ledger']}
        else:fa={r['k']:r['action'] for r in full};ca={r['k']:r['action'] for r in control}
        for x,y in zip(full,control):
            k=x['k'];assert k==y['k'] and x['local_net']==y['local_net']
            if fa[k]==ca[k]:continue
            D=x['local_net']
            category=('retained_gain' if D>0 else 'incurred_loss' if D<0 else 'zero_return') if fa[k] else ('missed_gain' if D>0 else 'avoided_loss' if D<0 else 'zero_return')
            categories[category]['count']+=1;categories[category]['absolute_return']+=abs(D)
            contribution=(int(fa[k])-int(ca[k]))*D;total+=contribution
            cases.append(dict(seed=seed,k=k,category=category,complete_return=D,full_action=fa[k],control_action=ca[k],full_minus_control_return=contribution))
    return dict(categories=categories,changed_actions=len(cases),total_full_minus_control_return=total,mean_full_minus_control_return=total/5,cases=cases)
dynamic_decomposition=dict(unguarded=decomposition())
for reserve in ('fixed130','decision'):
    dynamic_decomposition[reserve]={str(v):decomposition(reserve,v) for v in (0,110,130,260)}
manifest={str(p.relative_to(ROOT)):sha(p) for p in ROOT.iterdir() if p.is_file() and p.name not in ('matrix_summary.json','RESULTS.md')}
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
    protocol_sha256=sha(ROOT/'protocol.json'),source_integrity=True,exact_equivalence=results['exact_equivalence'],
    calibration_evaluations_per_arm=27,dynamic_policy_trajectories=10,test_matrix_states=issued,
    min_risk_eigenvalue=matrix_min,max_kkt=max_kkt,budget_trajectories=len(budget),
    max_guard_accounting_error=max_accounting,
    dynamic_summary={a:results['summary'][a] for a in ('precision_joint','gls_exact','block_sandwich')},
    budget_summary=budget_summary,
    fixed_state_summary={name:task['summary'] for name,task in fixed.items()},
    dynamic_action_decomposition=dynamic_decomposition,manifest=manifest)
(ROOT/'matrix_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({k:report[k] for k in ('passed','source_integrity','min_risk_eigenvalue','max_kkt','max_guard_accounting_error','exact_equivalence')},indent=2))
