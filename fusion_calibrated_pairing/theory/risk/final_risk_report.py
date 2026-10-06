"""Complete post-decision risk accounting for the nine frozen pipelines.

This analysis never rewrites calibration, selection, or scientific outputs.
Per-chain time-uniform error allocation uses all reported replay chains;
those chains are not counted as independent physical environments.
"""
from pathlib import Path
import argparse,gzip,json,sys,math,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from immutable_conditional_calibration_v2 import actual_admission_report
from admission_risk_report import actual_prefix_risk
FINAL_POLICIES=9
def load(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:return json.load(f)
def near(a,b):
    if abs(a-b)>1e-8:raise AssertionError((a,b))
def report(results):
    guarded=results['guarded'];arms=sorted({x['arm'] for x in guarded})
    if len(arms)!=FINAL_POLICIES:raise ValueError(f'Expected all nine frozen pipelines; found {arms}')
    chain_ids={(x['recording'],x['seed']) for x in guarded}
    if len(guarded)!=len(chain_ids)*FINAL_POLICIES:raise ValueError('Missing pipeline/chain comparison')
    if len({(x['arm'],x['recording'],x['seed']) for x in guarded})!=len(guarded):raise ValueError('Duplicate pipeline/chain')
    chains=[];aggregate={}
    for x in guarded:
        if not x['ledger']:raise ValueError('No actual execution ledger supplied')
        final=x['ledger'][-1]
        if not {'spent_loss','settled_increment','reserved'}<=set(final):raise ValueError('Missing actual final ledger fields')
        near(final['reserved'],0.)
        r=actual_admission_report(x['rows'],x['ledger'])
        near(r['net_increment'],x['increment']);near(r['negative_loss'],x['negative_loss']);near(r['negative_loss'],final['spent_loss']);near(r['net_increment'],final['settled_increment'])
        if r['guard_refusals']!=x['refused']:raise ValueError('Saved guard-refusal count disagrees with actual rows and ledger')
        if r['max_spent_plus_pending']>130.+1e-8:raise AssertionError('Frozen chain budget exceeded')
        risk=actual_prefix_risk(x['rows'],policies=FINAL_POLICIES,reported_chains=len(chain_ids))
        chains.append(dict(arm=x['arm'],recording=x['recording'],seed=x['seed'],empirical=r,risk_envelope=risk))
    for arm in arms:
        subset=[x for x in guarded if x['arm']==arm];rows=[r for x in subset for r in x['rows']]
        # Aggregation is descriptive; the confidence curves remain attached
        # to their predeclared complete physical/replay history chains.
        count={key:[] for key in ('issued','ready','informative','admitted')}
        for r in rows:
            count['issued'].append(r)
            if r['information_ready']:count['ready'].append(r)
            if r['disagreement']>0:count['informative'].append(r)
            if r['action']:count['admitted'].append(r)
        acts=count['admitted'];excess=[max(0.,r['gate_score']-(r['truegross']-5)) for r in acts]
        seeds=sorted({x['seed'] for x in subset});increments=[sum(x['increment'] for x in subset if x['seed']==seed) for seed in seeds]
        empty=[r for r in rows if r['calibration_empty_stratum_fallback']]
        empty_positive=[r for r in empty if r['information_ready'] and r['raw_gate_score']>0]
        aggregate[arm]=dict(mean_increment=sum(increments)/len(increments),seed_increments=dict(zip(seeds,increments)),coverage={key:dict(covered=sum(r['lower_covered'] for r in vals),total=len(vals),rate=sum(r['lower_covered'] for r in vals)/len(vals) if vals else None) for key,vals in count.items()},admissions=len(acts),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),zero=sum(r['local_net']==0 for r in acts),negative_loss=sum(max(0.,-r['local_net']) for r in acts),max_chain_negative_loss=max(x['negative_loss'] for x in subset),guard_refusals=sum(x['refused'] for x in subset),max_spent_plus_pending=max(x['empirical']['max_spent_plus_pending'] for x in chains if x['arm']==arm),excess_max=max(excess,default=0.),excess_sum=sum(excess),excess_mean=sum(excess)/len(excess) if excess else None,empty_stratum_issued=len(empty),empty_stratum_positive_raw_refusals=len(empty_positive),empty_stratum_beneficial_opportunities=sum(r['local_net']>0 for r in empty_positive),empty_stratum_harmful_opportunities=sum(r['local_net']<0 for r in empty_positive),empty_stratum_missed_potential_gain=sum(max(0.,r['local_net']) for r in empty_positive),descriptive_only=True)
    return dict(complete_nine_pipelines=True,pipeline_count=FINAL_POLICIES,reported_replay_chains=len(chain_ids),physical_recordings=len({x[0] for x in chain_ids}),physical_environment_scope='Recorded sessions and repeated delays are reported separately; session/participant dependence is retained. No independent-site count inferred.',aggregate=aggregate,chain_reports=chains,scope='Actual empirical admission calibration and audited chain budgets. Time-uniform bounds concern mean conditional risk of complete observed prefixes only, not an arbitrary next environment.')
def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Do not overwrite a retained risk report')
    value=report(load(a.results));value['input_sha256']=hashlib.sha256(a.results.read_bytes()).hexdigest()
    value['analysis_source_sha256']={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),ROOT/'admission_risk_report.py',ROOT/'immutable_conditional_calibration_v2.py')}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2,allow_nan=False))
    print(json.dumps(value['aggregate'],indent=2))
if __name__=='__main__':main()
