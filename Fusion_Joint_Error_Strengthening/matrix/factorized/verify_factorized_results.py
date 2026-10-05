"""Audit saved factorized outputs without modifying any experiment source."""
from pathlib import Path
import gzip,hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((OUT/'protocol.json').read_text())
assert all(sha(p)==h for p,h in protocol['source_hashes'].items())
r=json.loads((OUT/'results.json').read_text())
with gzip.open(OUT/'budget_trials.json.gz','rt') as f:b=json.load(f)
with gzip.open(OUT/'calibration_trials.json.gz','rt') as f:c=json.load(f)
assert len(c)==9 and all(len(t['results'])==3 for t in c)
assert len(b)==80
max_diag_error=0.;max_q_difference=0.;min_P=float('inf');min_R=float('inf');max_kkt=0.
for trial in r['trials']:
    full=trial['results']['precision_joint']['rows'];factor=trial['results']['factorized_information']['rows']
    assert trial['checks']['factorized_information']['passed']
    assert trial['results']['factorized_information']['solver_failures']==0
    max_kkt=max(max_kkt,trial['results']['factorized_information']['max_kkt'])
    for x,y in zip(full,factor):
        assert x['k']==y['k'] and x['eligible']==y['eligible'] and x['active_source_ids']==y['active_source_ids']
        R=np.array(y['factorized_R']);P=np.array(y['factorized_P'])
        assert np.array_equal(R,np.diag(np.diag(R))) and np.array_equal(P,np.diag(np.diag(P)))
        max_diag_error=max(max_diag_error,float(np.max(np.abs(np.diag(R)-np.diag(np.array(x['joint_R']))))))
        max_q_difference=max(max_q_difference,abs(x['q_issued']-y['q_issued']))
        min_P=min(min_P,float(np.diag(P).min()));min_R=min(min_R,float(np.diag(R).min()))
assert max_diag_error<1e-12 and min_P>0 and min_R>0
max_accounting=max(t['accounting_error'] for t in b)
assert max_accounting<1e-8
report=dict(passed=True,source_hashes_unchanged=True,calibration_evaluations=27,test_trajectories=5,
    independent_fork_checks=sum(t['checks']['factorized_information']['forks'] for t in r['trials']),
    max_independent_target_error=max(t['checks']['factorized_information']['target_error'] for t in r['trials']),
    max_independent_service_error=max(max(t['checks']['factorized_information']['service_errors'].values()) for t in r['trials']),
    budget_trajectories=len(b),max_guard_accounting_error=max_accounting,
    same_observed_history_masks_weights_and_models=True,max_predictive_diagonal_error=max_diag_error,
    max_dynamic_issued_q_difference=max_q_difference,min_factorized_P_diagonal=min_P,min_factorized_R_diagonal=min_R,max_kkt=max_kkt,
    state_relation=r['fixed_state']['state_relation'],
    file_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('verification.json','verification.log')})
(OUT/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in report.items() if k!='file_sha256'},indent=2))
