"""Complete factual analysis; never changes selected controllers or scores."""
from pathlib import Path
import csv,gzip,hashlib,json,statistics,math
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:return json.load(f)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def interval(values):
    values=np.asarray(values,float);n=len(values);mean=float(values.mean())
    # Only the predeclared five delays and thirteen participant means enter
    # these descriptive intervals. Fixed Student quantiles avoid a runtime
    # dependency absent from the delivered NumPy-only execution environment.
    t975={4:2.7764451051977987,12:2.178812829663418}
    radius=float(t975[n-1]*values.std(ddof=1)/math.sqrt(n)) if n>1 else None
    return dict(n=n,mean=mean,interval=[mean-radius,mean+radius] if radius is not None else None,
        wins=int((values>1e-9).sum()),ties=int((abs(values)<=1e-9).sum()),losses=int((values<-1e-9).sum()),values=values.tolist())
def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run():
    primary=load(ROOT/'selfback/results.json.gz');summary=load(ROOT/'selfback/summary.json');selection=load(ROOT/'selfback/selection.json')
    guarded=primary['guarded'];issued=primary['issued'];budgets=primary['budget_rows'];arms=list(summary['summary']);checks=list(summary['target_checks'])
    dbfroot=ROOT/'selfback_dbf'
    extension=None
    if (dbfroot/'results.json.gz').exists():
        extension=load(dbfroot/'results.json.gz');exsummary=load(dbfroot/'summary.json');guarded+=extension['guarded'];issued+=extension['issued'];budgets+=extension['budget_rows'];arms+=list(exsummary['summary']);checks+=exsummary['target_checks']
    people=summary['metadata']['test_ids'];seeds=(152001,152002,152003,152004,152005)
    audit=dict(ratio_rows=0,proposal_checks=0,coverage_checks=0,budget_paths=len(budgets),ledger_states=0,maximum_service_error=0.,maximum_target_error=max(c['maximum_error'] for c in checks))
    by={(g['arm'],g['person'],g['seed']):g for g in guarded}
    ratioarms=('paired','marginal','factorized','zero_increment')
    for result in issued:
        if result['arm'] not in ratioarms:continue
        g=by[result['arm'],result['person'],result['seed']]
        for r,z in zip(result['rows'],g['rows']):
            assert r['k']==z['k'] and r['gate_score']==z['gate_score']
            assert r['action']==bool(r['information_ready'] and r['gate_score']>0)
            assert z['proposed_action']==r['action']
            assert r['lower_covered']==(r['gate_score']<=r['truegross']-5.)
            assert z['lower_covered']==r['lower_covered']
            assert r['q_issued']==z['q_issued']
            audit['ratio_rows']+=1;audit['proposal_checks']+=2;audit['coverage_checks']+=2
    for g in budgets:
        loss=sum(max(0.,-r['local_net']) for r in g['rows'] if r['action'])
        net=sum(r['local_net'] for r in g['rows'] if r['action'])
        assert abs(net-g['increment'])<1e-9 and abs(loss-g['negative_loss'])<1e-9
        assert loss<=g['budget']+1e-9 and g['minimum_prefix']>=-g['budget']-1e-9
        audit['maximum_service_error']=max(audit['maximum_service_error'],g['service_error'])
        for l in g['ledger']:
            assert l['spent_loss']+l['reserved']<=g['budget']+1e-9
            if l['action']:assert l['issued_reserve']==130. and l['capacity_before']>=130.-1e-9
            audit['ledger_states']+=1
    units=[];overall=[];diagnostics=[];changed=[];comparison={}
    for arm in arms:
        gs=[g for g in guarded if g['arm']==arm];rows=[r for g in gs for r in g['rows']];acts=[r for r in rows if r['action']]
        cov=sum(r['lower_covered'] for r in acts);excess=[max(0.,r['gate_score']-(r['truegross']-5)) for r in acts]
        seedvalues=[sum(g['increment'] for g in gs if g['seed']==seed) for seed in seeds]
        overall.append(dict(arm=arm,mean_paid_increment=np.mean(seedvalues),mean_service_net=np.mean([sum(g['net'] for g in gs if g['seed']==seed) for seed in seeds]),
            admissions=len(acts),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),zero=sum(r['local_net']==0 for r in acts),
            negative_loss=sum(max(0.,-r['local_net']) for r in acts),covered=cov,coverage_n=len(acts),coverage_percent=100*cov/len(acts) if acts else None,
            max_excess=max(excess,default=0.),mean_excess=float(np.mean(excess)) if acts else None,total_excess=sum(excess),budget_refused=sum(g['refused'] for g in gs),
            budget_binding_chains=sum(g['refused']>0 for g in gs),maximum_chain_loss=max(g['negative_loss'] for g in gs),minimum_chain_prefix=min(g['minimum_prefix'] for g in gs),
            issued_covered=sum(r['lower_covered'] for r in rows),issued_n=len(rows),ready_covered=sum(r['lower_covered'] for r in rows if r['information_ready']),ready_n=sum(bool(r['information_ready']) for r in rows)))
        for person in people:
            values=[by[arm,person,seed]['increment'] for seed in seeds]
            rows0=[r for seed in seeds for r in by[arm,person,seed]['rows']];a=[r for r in rows0 if r['action']]
            units.append(dict(arm=arm,person=person,mean_paid_increment=np.mean(values),seed_increments=json.dumps(values),admissions=len(a),beneficial=sum(r['local_net']>0 for r in a),harmful=sum(r['local_net']<0 for r in a),negative_loss=sum(max(0.,-r['local_net']) for r in a),covered=sum(r['lower_covered'] for r in a),coverage_n=len(a),budget_refused=sum(by[arm,person,seed]['refused'] for seed in seeds)))
        for B in (0.,110.,130.,260.):
            gg=[g for g in budgets if g['arm']==arm and g['budget']==B];rr=[r for g in gg for r in g['rows'] if r['action']]
            diagnostics.append(dict(arm=arm,budget=B,mean_paid_increment=np.mean([sum(g['increment'] for g in gg if g['seed']==seed) for seed in seeds]),admissions=len(rr),beneficial=sum(r['local_net']>0 for r in rr),harmful=sum(r['local_net']<0 for r in rr),negative_loss=sum(max(0.,-r['local_net']) for r in rr),refused=sum(g['refused'] for g in gg),binding_chains=sum(g['refused']>0 for g in gg),maximum_chain_loss=max(g['negative_loss'] for g in gg),minimum_chain_prefix=min(g['minimum_prefix'] for g in gg)))
        if arm=='paired':continue
        terms={k:dict(count=0,value=0.) for k in ('added_gain','avoided_loss','missed_gain','incurred_loss')}
        for person in people:
            for seed in seeds:
                j,c=by['paired',person,seed],by[arm,person,seed]
                assert len(j['rows'])==len(c['rows'])
                for a,b in zip(j['rows'],c['rows']):
                    assert a['k']==b['k'] and a['local_net']==b['local_net']
                    if a['action']==b['action']:continue
                    D=a['local_net'];changed.append(dict(control=arm,person=person,seed=seed,k=a['k'],paired_action=a['action'],control_action=b['action'],D=D,paired_score=a['gate_score'],control_score=b['gate_score']))
                    if D==0:continue
                    key=('added_gain' if D>0 else 'incurred_loss') if a['action'] else ('missed_gain' if D>0 else 'avoided_loss');terms[key]['count']+=1;terms[key]['value']+=abs(D)
        value=terms['added_gain']['value']+terms['avoided_loss']['value']-terms['missed_gain']['value']-terms['incurred_loss']['value']
        unitdiff=[np.mean([by['paired',person,s]['increment']-by[arm,person,s]['increment'] for s in seeds]) for person in people]
        seeddiff=[sum(by['paired',person,s]['increment']-by[arm,person,s]['increment'] for person in people) for s in seeds]
        assert abs(value-sum(seeddiff))<1e-8
        comparison[arm]=dict(terms=terms,mean_total_difference=value/5,participant_difference=interval(unitdiff),delay_difference=interval(seeddiff),changed_actions=sum(1 for x in changed if x['control']==arm))
    csvwrite(ROOT/'selfback_results.csv',overall);csvwrite(ROOT/'selfback_participant_results.csv',units);csvwrite(ROOT/'selfback_budget_results.csv',diagnostics)
    if changed:csvwrite(ROOT/'selfback_changed_actions.csv',changed)
    primarywinner=max(selection['selected'],key=lambda arm:selection['selected'][arm]['net'])
    sourceclosure=load(ROOT/'selfback_freeze.json')['algorithm_source_hashes']
    for path,want in sourceclosure.items():assert sha(path)==want,path
    report=dict(role='All frozen outcomes and later separately frozen DBF extension retained; analysis cannot change controllers',primary_calibration_winner=primarywinner,
        primary_calibration_winner_config=selection['selected'][primarywinner],overall=overall,comparisons=comparison,units=units,budgets=diagnostics,audit=audit,
        test_people=people,primary_policy_trajectories=len(primary['guarded']),extension_policy_trajectories=len(extension['guarded']) if extension else 0,
        original_source_closure_unchanged=len(sourceclosure),primary_selection_sha256=sha(ROOT/'selfback/selection.json'),primary_result_sha256=sha(ROOT/'selfback/results.json.gz'),
        metadata=summary['metadata'],inference_scope='Participant means average five imposed delays. Participant t12 and delay t4 intervals are descriptive; the collection is one physical environment, 13 people are not13sites, and repeated delays are not independent physical replicates.')
    (ROOT/'selfback_report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')
    print(json.dumps(dict(overall=overall,comparisons=comparison,audit=audit),indent=2))
if __name__=='__main__':run()
