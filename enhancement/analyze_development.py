"""Read-only complete comparison of every saved CJ-R development outcome."""
from pathlib import Path
import csv,gzip,hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
TASKS=('rss348','arem366','gashome362','localization196')
def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(Path(path).read_text())
def dump(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def terms(new,old):
    totals=dict(kept_gain=0.,missed_gain=0.,incurred_loss=0.,avoided_loss=0.,changed=0)
    cases=[]
    for x,y in zip(new,old):
        assert x['k']==y['k'] and x['local_net']==y['local_net']
        if x['action']==y['action']:continue
        totals['changed']+=1;D=x['local_net']
        key=('kept_gain' if D>0 else 'incurred_loss' if D<0 else 'zero') if x['action'] else ('missed_gain' if D>0 else 'avoided_loss' if D<0 else 'zero')
        if key!='zero':totals[key]+=abs(D)
        cases.append(dict(k=x['k'],D=D,category=key,new_action=x['action'],old_action=y['action'],new_score=x['gate_score'],old_score=y['gate_score']))
    totals['increment_difference']=totals['kept_gain']-totals['missed_gain']-totals['incurred_loss']+totals['avoided_loss']
    return totals,cases

def main():
    reports={};csvrows=[];cases=[];failures=[]
    for task in TASKS:
        summary=load(ROOT/(task+'_development_summary.json'));raw=load(ROOT/(task+'_development_results.json.gz'))
        for arm,v in summary['arms'].items():csvrows.append(dict(task=task,arm=arm,**{k:v[k] for k in ('mean_total_increment','admissions','beneficial','harmful','zero','loss','optimistic_excess','refused')},covered=v['coverage'][0],coverage_denominator=v['coverage'][1]))
        primary=[x for x in raw['guarded'] if x['budget']==130.]
        comparisons={}
        for control in ('conditional_joint','legacy_factorized','legacy_sandwich','r_product','r_gaussian','r_copula'):
            total={};increments=[]
            for new in [x for x in primary if x['arm']=='r_joint']:
                old=next(x for x in primary if x['arm']==control and x['chain']==new['chain'] and x['seed']==new['seed'])
                t,cc=terms(new['rows'],old['rows'])
                assert abs(t['increment_difference']-(new['increment']-old['increment']))<1e-8
                for k,v in t.items():total[k]=total.get(k,0)+v
                increments.append(new['increment']-old['increment'])
                cases.extend(dict(task=task,chain=new['chain'],seed=new['seed'],control=control,**x) for x in cc)
            total['mean_chain_difference']=float(np.mean(increments))
            comparisons[control]=total
        ready=[]
        for trial in raw['trials']:
            old={x['k']:x for x in trial['results']['conditional_joint']['rows']}
            for row in trial['results']['r_joint']['rows']:
                info=row['reliability']
                if info['blocks']:ready.append(info)
            g=next(x for x in primary if x['arm']=='r_joint' and x['chain']==trial['chain'] and x['seed']==trial['seed'])
            for row in g['rows']:
                if row['action'] and row['local_net']<0:
                    failures.append(dict(task=task,chain=trial['chain'],seed=trial['seed'],k=row['k'],D=row['local_net'],
                        score=row['gate_score'],q=row['q_issued'],covered=row['lower_covered'],old_CJ_action=old[row['k']]['action'],old_CJ_score=old[row['k']]['gate_score'],
                        **{k:row['reliability'][k] for k in ('raw_mass','effective_mass','multiplier','blocks')}))
        reliability=dict(ready_states=len(ready),raw_mass_min=min((x['raw_mass'] for x in ready),default=None),
            raw_mass_max=max((x['raw_mass'] for x in ready),default=None),effective_mass_max=max((x['effective_mass'] for x in ready),default=None),
            multiplier_max=max((x['multiplier'] for x in ready),default=None),
            tiny_overlap_states=sum(x['raw_mass']<.01 for x in ready))
        reports[task]=dict(arms=summary['arms'],comparisons=comparisons,reliability=reliability,
            fixed_state=summary['fixed_state'],max_service_error=summary['max_service_error'],development_only=True)
    dump(ROOT/'development_analysis.json',dict(tasks=reports,harmful_CJR_admissions=failures,
        source_outputs={str(ROOT/(t+'_development_results.json.gz')):sha(ROOT/(t+'_development_results.json.gz')) for t in TASKS},
        conclusion='CJ-R is mathematically coherent but is not a demonstrated improvement; RSS primary falls from35.6 to2.0 and prior UCI196 coldstart remains inactive',
        invariants='original CJ and legacy RSS guarded seed outputs reproduce the unchanged archives exactly'))
    for name,rows in (('development_comparison.csv',csvrows),('changed_actions.csv',cases),('harmful_CJR_admissions.csv',failures)):
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({t:r['comparisons']['conditional_joint'] for t,r in reports.items()},indent=2))

if __name__=='__main__':main()
