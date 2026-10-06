"""Independent complete reporting of every frozen physical outcome.

Reads immutable outputs only. It cannot choose, refit or change a score.
Sessions remain nested within two participants; delays are descriptive
replays of the same observations, not independent physical environments.
"""
from pathlib import Path
import csv,gzip,hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'opportunity'
SEEDS=(162001,162002,162003,162004,162005)
def load(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:return json.load(f)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def interval(values):
    values=np.asarray(values,float);n=len(values);mean=float(values.mean())
    quantile={1:12.706204736432095,4:2.7764451051977987}.get(n-1)
    radius=float(quantile*values.std(ddof=1)/math.sqrt(n)) if quantile is not None else None
    return dict(n=n,mean=mean,values=values.tolist(),interval=[mean-radius,mean+radius] if radius is not None else None,
        wins=int((values>1e-9).sum()),ties=int((abs(values)<=1e-9).sum()),losses=int((values< -1e-9).sum()),
        scope='Descriptive fixed-trace delay or two-person summary; no independent-session or site-level inference')
def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def report(gs):
    rows=[row for g in gs for row in g['rows']];a=[row for row in rows if row['action']]
    excess=[max(0.,row['gate_score']-(row['truegross']-5.)) for row in a]
    empty=[row for row in rows if row['calibration_empty_stratum_fallback']]
    groups=dict(issued=rows,ready=[row for row in rows if row['information_ready']],
        informative=[row for row in rows if row['disagreement']>0],admitted=a)
    increments=[sum(g['increment'] for g in gs if g['seed']==seed) for seed in SEEDS]
    return dict(mean_paid_increment=float(np.mean(increments)),seed_increments=increments,
        mean_service_net=float(np.mean([sum(g['net'] for g in gs if g['seed']==seed) for seed in SEEDS])),
        admissions=len(a),beneficial=sum(row['local_net']>0 for row in a),harmful=sum(row['local_net']<0 for row in a),zero=sum(row['local_net']==0 for row in a),
        negative_loss=sum(max(0.,-row['local_net']) for row in a),max_excess=max(excess,default=0.),total_excess=sum(excess),mean_excess=float(np.mean(excess)) if a else None,
        coverage={key:dict(covered=sum(row['lower_covered'] for row in rr),total=len(rr),rate=sum(row['lower_covered'] for row in rr)/len(rr) if rr else None) for key,rr in groups.items()},
        refused=sum(g['refused'] for g in gs),binding_chains=sum(g['refused']>0 for g in gs),
        max_chain_loss=max(g['negative_loss'] for g in gs),minimum_chain_prefix=min(g['minimum_prefix'] for g in gs),
        empty_stratum_issues=len(empty),empty_stratum_missed_beneficial=sum(row['local_net']>0 for row in empty),
        empty_stratum_missed_beneficial_value=sum(max(0.,row['local_net']) for row in empty),
        empty_stratum_harmful_potential=sum(row['local_net']<0 for row in empty),
        empty_stratum_harmful_potential_loss=sum(max(0.,-row['local_net']) for row in empty))
def main():
    data=load(OUT/'results.json.gz');selection=load(OUT/'selection.json');checks=load(OUT/'test_checks.json')
    freeze=load(ROOT/'opportunity_freeze.json');lock=load(ROOT/'opportunity_freeze_lock.json')
    assert sha(ROOT/'opportunity_freeze.json')==lock['sha256']
    for kind in ('algorithm_source_hashes','declaration_hashes'):
        for path,want in freeze[kind].items():assert sha(path)==want,path
    assert sha(OUT/'selection.json')==load(OUT/'selection_lock.json')['selection_sha256']
    arms=freeze['arms'];gs=data['guarded'];budgets=data['budget_rows'];issued=data['issued'];people=[3,4]
    by={(g['arm'],g['recording'],g['seed']):g for g in gs}
    assert len(by)==len(gs)==9*12*5
    audit=dict(proposal_checks=0,coverage_checks=0,ledger_states=0,policy_trajectories=len(gs),budget_paths=len(budgets),
        maximum_service_error=0.,maximum_target_error=max(c['maximum_error'] for c in checks['target_checks']))
    for result in issued:
        guarded=by[result['arm'],result['recording'],result['seed']]
        for row,saved in zip(result['rows'],guarded['rows']):
            assert row['k']==saved['k'] and row['gate_score']==saved['gate_score']
            assert row['action']==bool(row['information_ready'] and row['gate_score']>0)
            assert row['action']==saved['proposed_action']
            assert row['lower_covered']==(row['gate_score']<=row['truegross']-5.)==saved['lower_covered']
            audit['proposal_checks']+=2;audit['coverage_checks']+=2
    for g in budgets:
        loss=sum(max(0.,-row['local_net']) for row in g['rows'] if row['action'])
        increment=sum(row['local_net'] for row in g['rows'] if row['action'])
        assert abs(increment-g['increment'])<1e-8 and abs(loss-g['negative_loss'])<1e-8
        assert loss<=g['budget']+1e-8 and g['minimum_prefix']>=-g['budget']-1e-8
        audit['maximum_service_error']=max(audit['maximum_service_error'],g['service_error'])
        for row in g['ledger']:
            assert row['spent_loss']+row['reserved']<=g['budget']+1e-8
            if row['action']:assert row['issued_reserve']==130. and row['capacity_before']>=130.-1e-8
            audit['ledger_states']+=1
    overall={arm:report([g for g in gs if g['arm']==arm]) for arm in arms}
    units={str(person):{arm:report([g for g in gs if g['arm']==arm and g['person']==person]) for arm in arms} for person in people}
    sessions=sorted({g['recording'] for g in gs});session_results={session:{arm:report([g for g in gs if g['arm']==arm and g['recording']==session]) for arm in arms} for session in sessions}
    comparisons={};changed=[]
    for arm in arms:
        if arm=='paired':continue
        terms={key:dict(count=0,value=0.) for key in ('added_gain','avoided_loss','missed_gain','incurred_loss')}
        for session in sessions:
            for seed in SEEDS:
                paired,control=by['paired',session,seed],by[arm,session,seed]
                assert len(paired['rows'])==len(control['rows'])
                for a,b in zip(paired['rows'],control['rows']):
                    assert a['k']==b['k'] and a['local_net']==b['local_net']
                    if a['action']==b['action']:continue
                    D=a['local_net'];changed.append(dict(control=arm,recording=session,person=paired['person'],seed=seed,k=a['k'],paired_action=a['action'],control_action=b['action'],D=D,paired_score=a['gate_score'],control_score=b['gate_score']))
                    if D==0:continue
                    key=('added_gain' if D>0 else 'incurred_loss') if a['action'] else ('missed_gain' if D>0 else 'avoided_loss')
                    terms[key]['count']+=1;terms[key]['value']+=abs(D)
        value=terms['added_gain']['value']+terms['avoided_loss']['value']-terms['missed_gain']['value']-terms['incurred_loss']['value']
        seeddiff=[sum(by['paired',session,seed]['increment']-by[arm,session,seed]['increment'] for session in sessions) for seed in SEEDS]
        persondiff=[units[str(person)]['paired']['mean_paid_increment']-units[str(person)][arm]['mean_paid_increment'] for person in people]
        assert abs(value-sum(seeddiff))<1e-8
        comparisons[arm]=dict(terms=terms,mean_total_difference=value/5.,delay_difference=interval(seeddiff),person_difference=interval(persondiff),changed_actions=sum(row['control']==arm for row in changed))
    budget_summary={str(B):{arm:report([g for g in budgets if g['budget']==B and g['arm']==arm]) for arm in arms} for B in (0.,110.,130.,260.)}
    summary=dict(overall=overall,units=units,sessions=session_results,comparisons=comparisons,budget_summary=budget_summary,audit=audit,
        strongest_calibration_control=selection['strongest_calibration_control'],
        strongest_calibration_control_co_winners=selection['strongest_calibration_control_co_winners'],
        selected=selection['selected'],metadata=selection['metadata'],availability=data['availability'],
        freeze_sha256=sha(ROOT/'opportunity_freeze.json'),selection_sha256=sha(OUT/'selection.json'),results_sha256=sha(OUT/'results.json.gz'),
        interpretation='All nine frozen pipelines/all60 test chains retained. Two held-out people at one collection; six sessions/person and five delays are not independent sites. No method or selected parameter changed using these physical outcomes.')
    csvwrite(ROOT/'opportunity_results.csv',[dict(arm=arm,**{key:value for key,value in entry.items() if key not in ('coverage','seed_increments')},
        admitted_covered=entry['coverage']['admitted']['covered'],admitted_total=entry['coverage']['admitted']['total']) for arm,entry in overall.items()])
    csvwrite(ROOT/'opportunity_changed_actions.csv',changed)
    (ROOT/'opportunity_report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(overall=overall,comparisons=comparisons,audit=audit),indent=2),flush=True)

if __name__=='__main__':main()
