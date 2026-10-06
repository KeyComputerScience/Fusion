"""Causality/schema audits; synthetic inputs are not performance evidence."""
from pathlib import Path
import importlib.util,json,sys,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
sys.path.insert(0,str(ROOT))
import immutable_conditional_calibration as m
gp=PROJECT/'work/dominance_next_20261003/budget_guard/run_loss_budget_guard.py'
s=importlib.util.spec_from_file_location('calibration_guard_audit',gp);g=importlib.util.module_from_spec(s);sys.modules[s.name]=g;s.loader.exec_module(g)
def reject(fn):
    try:fn()
    except (ValueError,KeyError,TypeError):return True
    raise AssertionError('Malformed/premature input accepted')
def main():
    stream=dict(windows=16,recording='synthetic_api_only',decisions=[dict(k=k,maturity=k+6,N=128,cal_context=[.1,.2,1.]) for k in (0,4,8,12)])
    raw=[dict(k=d['k'],maturity=d['maturity'],gate_score=-2.,information_ready=True,disagreement=.2,truegross=7.,local_net=4.) for d in stream['decisions']]
    c=m.configurations()[4];r=m.calibrated_run(stream,raw,c)
    assert [x['action'] for x in r['rows']]==[False,False,True,True]
    for cutoff in (0,4,8,12):
        q=m.calibrated_run(stream,raw,c,poison_after=cutoff)
        for a,b in zip(r['rows'],q['rows']):
            if a['k']<=cutoff:assert a['gate_score']==b['gate_score'] and a['action']==b['action'] and a['calibration_correction']==b['calibration_correction']
    pool=m.complete_initial_pool(stream,raw,boundary=8,provenance=dict(stage='known_development_calibration',canonical_seed=1,physical_partition='synthetic',completed_deadline=8))
    assert len(pool)==1
    sel=m.calibrated_run(stream,raw,c,pool,True,selection_boundary=8)
    assert not any(x['action'] for x in sel['rows'] if x['k']<8)
    full=m.calibrated_run(stream,raw,c,pool,True)
    assert full['selection_boundary']==0 and full['rows'][0]['calibration_prior_history_count']==1
    # Every fit-stage point persists; online callbacks have a separate cap.
    many=[dict(pool[0],origin=i) for i in range(60)]
    state=m.ConditionalScoreCalibration(c,many)
    assert len(state.prior)==60 and not state.history
    state.issue(m.ImmutableIssue(0,10,128,-2.,(.1,.2,1.),True));state.issue(m.ImmutableIssue(4,8,128,-2.,(.1,.2,1.),True))
    assert reject(lambda:state.callback(7,4,2.))
    state.callback(8,4,2.);assert 0 not in state.completed
    state.callback(10,0,2.);assert state.completed=={0,4}
    assert reject(lambda:state.callback(10,0,2.))
    assert len(state.prior)==60 and len(state.history)==2
    # Actual original guard schema, including a real reserve refusal in
    # this explicit synthetic replay, is used to check the report adapter.
    guarded=g.replay(r['rows'],16,130.,'gross_loss',{x['k']:130. for x in r['rows']})
    report=m.actual_admission_report(guarded['rows'],guarded['ledger'])
    assert report['max_spent_plus_pending']==130. and report['guard_refusals']==1
    assert reject(lambda:m.actual_admission_report(guarded['rows'],[dict(k=0)]))
    malformed={}
    malformed['bad_context']=reject(lambda:m.checked_context([2.,.1,1.]))
    malformed['nan_context']=reject(lambda:m.checked_context([float('nan'),.1,1.]))
    malformed['invalid_config']=reject(lambda:m.ConditionalScoreCalibration(dict(bandwidth=.4,probability=.9)))
    malformed['future_initial_pool']=reject(lambda:m.ConditionalScoreCalibration(c,[dict(pool[0],source_stage='test')]))
    bad=[dict(raw[0],maturity=6.5)]+raw[1:]
    malformed['fractional_maturity']=reject(lambda:m.calibrated_run(stream,bad,c))
    malformed['duplicate_origin']=reject(lambda:m.calibrated_run(stream,[raw[0],raw[0]]+raw[2:],c))
    out=dict(role='Synthetic API/maturity/support/schema verification only; no physical superiority evidence',scientific_source_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),shared_guard_sha256=hashlib.sha256(gp.read_bytes()).hexdigest(),future_truth_poisoning_cutoffs=[0,4,8,12],out_of_order_complete_callbacks=True,whole_dedicated_selection_session=True,explicit_legacy_boundary=True,persistent_full_score_fit_pool=60,actual_original_guard_report=report,malformed_input_rejections=malformed)
    (ROOT/'api_checks.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
