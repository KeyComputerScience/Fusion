"""Independent post-analysis of the complete frozen Mobile755 inventory."""
from pathlib import Path
import gzip,hashlib,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;COLLECTION=ROOT.parent/'physical_mobile755';DATA=COLLECTION/'mobile755'
sys.path.insert(0,str(ROOT))
from immutable_conditional_calibration_v2 import actual_admission_report
def load(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else p.open()) as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(a,b):assert abs(a-b)<1e-8,(a,b)
def main():
    freeze=load(COLLECTION/'mobile755_freeze.json');lock=load(COLLECTION/'mobile755_freeze_lock.json')
    assert sha(COLLECTION/'mobile755_freeze.json')==lock['sha256']
    for field in ('algorithm_source_hashes','declaration_hashes'):
        for p,want in freeze[field].items():assert sha(p)==want,('Frozen source changed',p)
    selection=load(DATA/'selection.json');selection_lock=load(DATA/'selection_lock.json')
    assert sha(DATA/'selection.json')==selection_lock['selection_sha256']
    assert selection_lock['freeze_sha256']==sha(COLLECTION/'mobile755_freeze.json') and selection_lock['test_exists'] is False
    results=load(DATA/'results.json.gz');risk=load(DATA/'risk_report.json')
    assert risk['input_sha256']==sha(DATA/'results.json.gz')
    for p,want in risk['analysis_source_sha256'].items():assert sha(p)==want,('Risk analysis source changed',p)
    arms=set(freeze['arms']);seeds=set(freeze['test_seeds']);recording='mobile755-test'
    expected={(arm,recording,seed) for arm in arms for seed in seeds}
    assert len(arms)==9 and len(seeds)==5
    for field in ('issued','guarded','raw'):
        values=results[field];assert len(values)==45
        assert {(v['arm'],v['recording'],v['seed']) for v in values}==expected,field
    assert risk['reported_replay_chains']==5 and risk['pipeline_count']==9
    assert len(risk['chain_reports'])==45
    for c in risk['chain_reports']:
        bound=c['risk_envelope'];assert bound['policies']==9 and bound['reported_replay_chains']==5
        near(bound['error_per_endpoint'],.05/(9*5*4))
        if c['empirical']['admissions']==0:
            assert c['empirical']['coverage']['admitted']['rate'] is None
            assert all(x is None for x in bound['average_conditional_risk_upper'].values())
    budget_rows=results['budget_rows'];budgets=set(freeze['budgets'])
    assert budgets=={0.,110.,130.,260.} and len(budget_rows)==180
    assert {(v['arm'],v['recording'],v['seed'],v['budget']) for v in budget_rows}=={(*x,b) for x in expected for b in budgets}
    budget_summary={}
    for B in sorted(budgets):
        grouped={}
        for arm in freeze['arms']:
            values=[v for v in budget_rows if v['budget']==B and v['arm']==arm]
            reports=[]
            for v in values:
                r=actual_admission_report(v['rows'],v['ledger']);reports.append(r)
                near(r['net_increment'],v['increment']);near(r['negative_loss'],v['negative_loss'])
                near(r['net_increment'],v['ledger'][-1]['settled_increment']);near(r['negative_loss'],v['ledger'][-1]['spent_loss'])
                assert r['guard_refusals']==v['refused'] and r['max_spent_plus_pending']<=B+1e-8
                assert v['minimum_prefix']>=-B-1e-8
            grouped[arm]=dict(mean_increment=sum(v['increment'] for v in values)/5,admissions=sum(v['admissions'] for v in values),harmful=sum(v['harmful'] for v in values),beneficial=sum(v['beneficial'] for v in values),negative_loss=sum(v['negative_loss'] for v in values),guard_refusals=sum(v['refused'] for v in values),max_chain_negative_loss=max(v['negative_loss'] for v in values),max_spent_plus_pending=max(v['max_spent_plus_pending'] for v in reports))
        budget_summary[str(B)]=grouped
    readiness={}
    for arm in freeze['arms']:
        values=[v for v in results['issued'] if v['arm']==arm]
        first=[v['rows'][0] for v in values]
        readiness[arm]=dict(state_fit_library_count=len(selection['prior_library']) if arm not in ('pdf','qmf','dbf') else None,score_fit_pool_count=len(selection['score_pools'][arm]),first_issue_ready=sum(v['information_ready'] for v in first),first_issue_empty_stratum=sum(v['calibration_empty_stratum_fallback'] for v in first),all_chain_initial_pool_counts=sorted({v['initial_pool_count'] for v in values}),first_ready_origins={str(v['seed']):next((r['k'] for r in v['rows'] if r['information_ready']),None) for v in values})
    out=dict(scope='Strict45-pipeline/5-delay-chain and180-budget-path post-analysis. One ordinal physical collection; no additional independent participant/site claim.',planned_inventory_verified=True,freeze_and_selected_sources_unchanged=True,analysis_hashes_verified=True,scientific_score_sha256=sha(ROOT/'immutable_conditional_calibration_v2.py'),freeze_sha256=sha(COLLECTION/'mobile755_freeze.json'),result_sha256=sha(DATA/'results.json.gz'),risk_report_sha256=sha(DATA/'risk_report.json'),analysis_source_sha256=sha(Path(__file__)),pipeline_count=9,chain_count=5,primary_risk_error_per_endpoint=.05/(9*5*4),primary=risk['aggregate'],budget=budget_summary,initial_evidence_and_readiness=readiness)
    dest=DATA/'risk_integrity_checks.json';assert not dest.exists(),'Do not overwrite retained post-analysis'
    dest.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps(dict(planned_inventory_verified=True,budget=budget_summary,initial_evidence_and_readiness=readiness),indent=2))
if __name__=='__main__':main()
