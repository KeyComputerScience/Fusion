"""Complete report of all locked mobile755 physical-pipeline outcomes."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import run_mobile755 as r

def main():
    r.verify();selection=r.load('mobile755/selection.json');payload=r.load('mobile755/results.json.gz');checks=r.load('mobile755/test_checks.json')
    groups=payload['guarded'];summary={};indexed={}
    for arm in r.ARMS:
        gg=[g for g in groups if g['arm']==arm]
        report=r.cal.actual_admission_report([x for g in gg for x in g['rows']],[x for g in gg for x in g['ledger']])
        increments=[sum(g['increment'] for g in gg if g['seed']==seed) for seed in r.TEST_SEEDS]
        summary[arm]=dict(report,mean_increment=float(r.np.mean(increments)),seed_increments=increments)
        indexed[arm]={(g['recording'],g['seed'],x['k']):x for g in gg for x in g['rows']}
    decomposition={}
    for arm in r.ARMS:
        if arm=='paired':continue
        terms={name:dict(count=0,value=0.) for name in ('added_gain','avoided_loss','missed_gain','incurred_loss')};changes=[]
        assert set(indexed[arm])==set(indexed['paired'])
        for key,a in indexed['paired'].items():
            b=indexed[arm][key];assert a['local_net']==b['local_net']
            if a['action']==b['action']:continue
            D=a['local_net'];changes.append(dict(recording=key[0],seed=key[1],origin=key[2],D=D,paired=a['action'],control=b['action'],paired_score=a['gate_score'],control_score=b['gate_score']))
            if D==0:continue
            name=('added_gain' if D>0 else 'incurred_loss') if a['action'] else ('missed_gain' if D>0 else 'avoided_loss')
            terms[name]['count']+=1;terms[name]['value']+=abs(D)
        total=terms['added_gain']['value']+terms['avoided_loss']['value']-terms['missed_gain']['value']-terms['incurred_loss']['value']
        measured=5*(summary['paired']['mean_increment']-summary[arm]['mean_increment']);assert abs(total-measured)<1e-8
        decomposition[arm]=dict(terms=terms,changes=changes,mean_difference=total/5,accounting_error=abs(total-measured))
    budget=[]
    for arm in r.ARMS:
        for cap in r.BUDGETS:
            gg=[g for g in payload['budget_rows'] if g['arm']==arm and g['budget']==cap]
            budget.append(dict(arm=arm,budget=cap,mean_increment=sum(g['increment'] for g in gg)/5,
                beneficial=sum(g['beneficial'] for g in gg),harmful=sum(g['harmful'] for g in gg),
                guard_refusals=sum(g['refused'] for g in gg),negative_loss=sum(g['negative_loss'] for g in gg),
                max_spent_plus_pending=max(float(x['spent_loss'])+float(x['reserved']) for g in gg for x in g['ledger'])))
    out=dict(scope='One unused independently collected physical collection with ordinal record holdout; no participant/site independence or validated physical duration claim',summary=summary,
        selected=selection['selected'],strongest_calibration_control=selection['strongest_calibration_control'],strongest_control_co_winners=selection['strongest_calibration_control_co_winners'],
        decomposition=decomposition,budget=budget,checks=checks,availability=payload['availability'],metadata=selection['metadata'],
        policy_trajectories=len(groups),budget_paths=len(payload['budget_rows']),all_outcomes_retained=True)
    r.dump('mobile755/summary.json',out);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
