"""Synthetic schema checks only; these are not physical gain evidence."""
from pathlib import Path
import copy,importlib.util,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
sys.path.insert(0,str(ROOT))
import immutable_conditional_calibration_v2 as cal
import final_risk_report as audit
gp=PROJECT/'work/dominance_next_20261003/budget_guard/run_loss_budget_guard.py'
sp=importlib.util.spec_from_file_location('final_report_shared_guard_check',gp)
guard=importlib.util.module_from_spec(sp);sys.modules[sp.name]=guard;sp.loader.exec_module(guard)
def rejects(fn):
    try:fn()
    except (ValueError,KeyError,AssertionError):return True
    raise AssertionError('Invalid actual execution report was accepted')
def main():
    stream=dict(windows=16,recording='synthetic_schema_only',decisions=[dict(k=k,maturity=k+6,N=128,cal_context=[.1,.2,1.]) for k in (0,4,8,12)])
    raw=[dict(k=d['k'],maturity=d['maturity'],gate_score=-2.,information_ready=True,disagreement=.2,truegross=7.,local_net=4.) for d in stream['decisions']]
    scored=cal.calibrated_run(stream,raw,cal.configurations()[4])
    actual=guard.replay(scored['rows'],16,130.,'gross_loss',{r['k']:130. for r in scored['rows']})
    refused=sum(r['proposed_action'] and not r['action'] for r in actual['rows'])
    pipelines=[]
    for i in range(9):
        pipelines.append(dict(arm=f'synthetic_pipeline_{i}',recording=stream['recording'],seed=1,rows=copy.deepcopy(actual['rows']),ledger=copy.deepcopy(actual['ledger']),increment=actual['final_settled_increment'],negative_loss=actual['final_spent_loss'],refused=refused))
    results=dict(guarded=pipelines);r=audit.report(results)
    assert r['pipeline_count']==9 and r['reported_replay_chains']==1
    assert all(c['risk_envelope']['policies']==9 for c in r['chain_reports'])
    checks={}
    def altered(fn):
        x=copy.deepcopy(results);fn(x);return rejects(lambda:audit.report(x))
    checks['missing_saved_loss_rejected']=altered(lambda x:x['guarded'][0].pop('negative_loss'))
    checks['inconsistent_saved_loss_rejected']=altered(lambda x:x['guarded'][0].update(negative_loss=1.))
    checks['inconsistent_refusal_count_rejected']=altered(lambda x:x['guarded'][0].update(refused=99))
    checks['missing_ledger_field_rejected']=altered(lambda x:x['guarded'][0]['ledger'][0].pop('spent_loss'))
    checks['missing_pipeline_rejected']=altered(lambda x:x['guarded'].pop())
    checks['duplicate_pipeline_rejected']=altered(lambda x:x['guarded'].append(copy.deepcopy(x['guarded'][0])))
    assert all(checks.values())
    out=dict(role='Synthetic final-report schema and accounting checks only; no physical gain validation',nine_policy_error_allocation=True,actual_shared_guard_schema=True,checks=checks)
    (ROOT/'final_report_schema_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
