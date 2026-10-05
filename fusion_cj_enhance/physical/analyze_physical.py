"""Outcome-complete analysis of source-frozen physical guard replays."""
from pathlib import Path
import gzip,hashlib,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parent

def load(p):
    return json.load(gzip.open(p,'rt')) if str(p).endswith('.gz') else json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def interval(x,df):
    x=np.asarray(x,dtype=float);t={4:2.776445105,9:2.262157163}.get(df)
    return [float(x.mean()-t*x.std(ddof=1)/np.sqrt(len(x))),float(x.mean()+t*x.std(ddof=1)/np.sqrt(len(x)))] if t and len(x)>1 else None

def coverage(rows):
    return [int(sum(r['lower_covered'] for r in rows)),len(rows)]

def main():
    import importlib.util
    spec=importlib.util.spec_from_file_location('analyzed_frozen_physical',ROOT/'run_physical.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    frozen=p.verify();meta=load(ROOT/'data/cache_metadata.json');primary=frozen['primary']
    chains=[]
    for path in sorted(ROOT.glob('S*_guarded.json.gz')):chains+=load(path)
    assert chains and (ROOT/'test_chain_summary.json').exists()
    external=ROOT/'external/guarded_all.json.gz'
    if external.exists():
        ext=load(external)
        if isinstance(ext,dict):ext=ext['records']
        chains += [dict(x,arm=x['rule']+'_MLP',negative_loss=x['loss']) for x in ext]
    result=dict(primary=primary,collection='HARTH779',metadata=meta,arms={},comparisons={},budget={},provenance={},accounting=dict(max_target_identity_error=0.,minimum_return_bound_slack=None,checked_budget_trajectories=len(chains)))
    target_error=0.;bound_slacks=[]
    for trajectory in chains:
        cert=0.
        for row in trajectory['rows']:
            S=row['posterior_or_block_sd'];q=row['q_issued'];T=row['standardized_score']
            L=row['gain']-q*S-5;delta=row['local_net']-row['truegross']+5
            assert -1e-8<=delta<=2+1e-8
            target_error=max(target_error,abs(row['local_net']-(L-S*(T-q)+delta)))
            if row['action']:cert+=L-S*max(T-q,0.)
        slack=trajectory['increment']-cert;assert slack>=-1e-8
        bound_slacks.append(slack)
    result['accounting'].update(max_target_identity_error=target_error,minimum_return_bound_slack=min(bound_slacks))
    arms=sorted({x['arm'] for x in chains});persons=sorted({x['person'] for x in chains});seeds=sorted({x['seed'] for x in chains})
    baseline=[x for x in chains if x['budget']==130.]
    for arm in arms:
        rr=[x for x in baseline if x['arm']==arm]
        byseed=[sum(x['increment'] for x in rr if x['seed']==s) for s in seeds]
        byperson={str(h):float(np.mean([sum(x['increment'] for x in rr if x['person']==h and x['seed']==s) for s in seeds])) for h in persons}
        admitted=[row for x in rr for row in x['rows'] if row['action']]
        covered=sum(row['lower_covered'] for row in admitted)
        excess=[row['posterior_or_block_sd']*max(row['standardized_score']-row['q_issued'],0.) for row in admitted]
        issued=[row for x in rr for row in x['rows']]
        informative=[row for row in issued if row['disagreement']>0]
        ready=[row for row in issued if row.get('information_ready',True)]
        person_actions={str(h):[row for x in rr if x['person']==h for row in x['rows'] if row['action']] for h in persons}
        result['arms'][arm]=dict(mean_total_increment=float(np.mean(byseed)),seed_total=byseed,participant_mean=byperson,
            participant_mean_interval=interval(list(byperson.values()),len(persons)-1),admissions=len(admitted),
            beneficial=sum(row['local_net']>0 for row in admitted),harmful=sum(row['local_net']<0 for row in admitted),zero=sum(row['local_net']==0 for row in admitted),
            total_negative_loss=sum(max(0.,-row['local_net']) for row in admitted),admitted_coverage=[covered,len(admitted)],
            admitted_excess=dict(total=float(sum(excess)),mean=float(np.mean(excess)) if excess else None,maximum=max(excess,default=None)),
            issued_coverage=coverage(issued),informative_coverage=coverage(informative),ready_coverage=coverage(ready),
            coverage_ready_definition='information_ready for internal methods; all issued origins for external helper',
            counterfactual_beneficial=sum(row['local_net']>0 for row in issued),
            counterfactual_positive_return=sum(max(row['local_net'],0.) for row in issued),
            participant_actions={h:dict(admissions=len(rows),beneficial=sum(r['local_net']>0 for r in rows),
                harmful=sum(r['local_net']<0 for r in rows),coverage=coverage(rows),
                excess=sum(r['posterior_or_block_sd']*max(r['standardized_score']-r['q_issued'],0.) for r in rows))
                for h,rows in person_actions.items()},budget_refused=sum(x['refused'] for x in rr))
    lookup={(x['recording'],x['seed'],x['arm']):x for x in baseline}
    changes=[]
    for arm in arms:
        if arm==primary:continue
        gains=misses=avoided=incurred=0.;useful_extra=useful_missed=0;paired=[];person_deltas={str(h):[] for h in persons}
        for key,P in lookup.items():
            if key[2]!=primary:continue
            C=lookup[key[:2]+(arm,)];pr={r['k']:r for r in P['rows']};cr={r['k']:r for r in C['rows']}
            assert set(pr)==set(cr)
            d=0.
            for k,a in pr.items():
                b=cr[k];assert abs(a['local_net']-b['local_net'])<1e-10
                difference=int(a['action'])-int(b['action']);D=a['local_net'];d+=difference*D
                if difference:
                    changes.append(dict(control=arm,recording=key[0],seed=key[1],origin=k,primary_action=a['action'],control_action=b['action'],complete_return=D))
                    if difference>0:
                        if D>0:gains+=D;useful_extra+=1
                        elif D<0:incurred-=D
                    else:
                        if D>0:misses+=D;useful_missed+=1
                        elif D<0:avoided-=D
            assert abs(d-(P['increment']-C['increment']))<1e-8
            paired.append(dict(person=P['person'],seed=P['seed'],delta=d));person_deltas[str(P['person'])].append(d)
        byseed=[sum(x['delta'] for x in paired if x['seed']==s) for s in seeds]
        # Sum recording chains per participant within a shared delay, then average schedules.
        pp={str(h):float(np.mean([sum(x['delta'] for x in paired if x['person']==h and x['seed']==s) for s in seeds])) for h in persons}
        result['comparisons'][arm]=dict(mean_total_difference=float(np.mean(byseed)),paired_delay_interval=interval(byseed,len(seeds)-1),
            participant_mean_difference=float(np.mean(list(pp.values()))),participant_difference_interval=interval(list(pp.values()),len(persons)-1),participant_differences=pp,
            positive_participants=sum(x>0 for x in pp.values()),tied_participants=sum(x==0 for x in pp.values()),negative_participants=sum(x<0 for x in pp.values()),
            additional_beneficial=useful_extra,missed_beneficial=useful_missed,retained_extra_return=gains,missed_return=misses,avoided_loss=avoided,incurred_extra_loss=incurred,
            exact_pooled_action_gain=gains-misses+avoided-incurred)
    for budget in frozen['execution']['diagnostic_budgets']:
        rr=[x for x in chains if x['arm']==primary and x['budget']==budget]
        result['budget'][str(budget)]=dict(total_negative_loss=sum(x['negative_loss'] for x in rr),refused=sum(x['refused'] for x in rr),
            binding_chains=sum(any(row['proposed_action'] and not row['action'] for row in x['rows']) for x in rr),
            admissions=sum(x['admissions'] for x in rr),min_prefix=min(x['minimum_prefix'] for x in rr),
            max_single_chain_negative_loss=max(x['negative_loss'] for x in rr),
            mean_total_increment=float(np.mean([sum(x['increment'] for x in rr if x['seed']==s) for s in seeds])),
            beneficial=sum(x['beneficial'] for x in rr),harmful=sum(x['harmful'] for x in rr))
    result['provenance']=dict(algorithm_sha256=p.sha(ROOT/'algorithm_freeze.json'),protocol_sha256=p.sha(ROOT/'protocol.json'),
        all_selections_before_test_sha256=p.sha(ROOT/'all_selections_before_test.json'),analysis_source_sha256=p.sha(__file__),
        guarantee_scope='per actual participant/history chain, all controllers same execution guarantee',inference_scope='ten held-out physical participants, one collection; five schedules reuse their observations')
    dump(ROOT/'analysis.json',result)
    with (ROOT/'changed_actions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(changes[0]) if changes else ['control','recording','seed','origin','primary_action','control_action','complete_return']);w.writeheader();w.writerows(changes)
    print(json.dumps({a:(v['mean_total_increment'],v['beneficial'],v['harmful'],v['admitted_coverage']) for a,v in result['arms'].items()},indent=2))
if __name__=='__main__':main()
