"""Read immutable four-task trial logs and summarize every replay comparison."""
import json, math, hashlib
from pathlib import Path
import numpy as np

import argparse
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--package',type=Path,default=HERE.parent/'bayes_closed_loop_repro')
parser.add_argument('--output',type=Path,default=HERE/'analysis_rerun')
args=parser.parse_args()
BASE=args.package.resolve();WORK=args.output.resolve();WORK.mkdir(parents=True,exist_ok=True)
TASKS = ['occupancy357','occupancy864','mhealth319','har240']
LABELS = {'occupancy357':'Occupancy 357','occupancy864':'Room 864','mhealth319':'MHEALTH','har240':'HAR'}
MODES = ['joint','bayes_gate','frequentist_gate','periodic','frozen']
MODE_LABELS = {'joint':'Joint predictive point gate','bayes_gate':'Posterior gate only','frequentist_gate':'Matched empirical-block gate','periodic':'Periodic admission','frozen':'Keep shared reference'}
FULL = 'bayes_both'
INPUTS = {n:BASE/('new_bayes_fusion' if n.startswith('occupancy') else 'new_bayes_extension')/(n+'_results.json') for n in TASKS}
T4 = 2.7764451051977987
EPS = 1e-8

def stats(vals):
    x=np.asarray(vals,float)
    half=T4*x.std(ddof=1)/math.sqrt(len(x))
    return {'by_seed':[float(v) for v in x],'sum':float(x.sum()),'mean':float(x.mean()),'min':float(x.min()),'max':float(x.max()),'positive_seeds':int((x>EPS).sum()),'tie_seeds':int((abs(x)<=EPS).sum()),'negative_seeds':int((x < -EPS).sum()),'denominator_seeds':len(x),'conditional_delay_t4_descriptive_interval':[float(x.mean()-half),float(x.mean()+half)]}

def counts(rows):
    acts=[r for r in rows if r['action']]
    return {'eligible':len(rows),'admissions':len(acts),'harmful':sum(r['local_net']<0 for r in acts),'beneficial':sum(r['local_net']>0 for r in acts),'neutral':sum(r['local_net']==0 for r in acts),'net_lease_sum':float(sum(r['local_net'] for r in acts)),'admitted_harmful_fraction':sum(r['local_net']<0 for r in acts)/len(acts) if acts else None,'admitted_beneficial_fraction':sum(r['local_net']>0 for r in acts)/len(acts) if acts else None}

def ratio(n,d):
    return {'numerator':n,'denominator':d,'ratio':n/d if d else None}

def compare(trials,control):
    diffs=[];harm_diffs=[];benefit_diffs=[];gross_diffs=[];fee_savings=[];rows_full=[];rows_control=[];detail=[]
    events={k:[] for k in ['avoided_harmful','missed_beneficial','new_harmful','new_beneficial','changed_neutral']}
    retained_benefit=retained_harm=0
    for t in trials:
        a=t['results'][FULL];b=t['results'][control]
        ar=a['rows'];br=b['rows']
        assert len(ar)==len(br)
        ca=counts(ar);cb=counts(br)
        # Rebuild counts instead of assuming the saved summary is correct.
        for s,c in [(a,ca),(b,cb)]:
            assert c['admissions']==s['deployments'] and c['harmful']==s['harmful'] and c['beneficial']==s['beneficial']
        contributions={k:0.0 for k in events};changed=0
        for ra,rb in zip(ar,br):
            assert ra['k']==rb['k'] and abs(ra['local_net']-rb['local_net'])<EPS
            d=ra['local_net'];fa=bool(ra['action']);fb=bool(rb['action'])
            retained_benefit+=fb and fa and d>0
            retained_harm+=fb and fa and d<0
            if fa==fb:continue
            changed+=1
            if fb:
                k='avoided_harmful' if d<0 else 'missed_beneficial' if d>0 else 'changed_neutral'
                delta=-d
            else:
                k='new_harmful' if d<0 else 'new_beneficial' if d>0 else 'changed_neutral'
                delta=d
            contributions[k]+=delta
            events[k].append({'seed':t['seed'],'window':ra['k'],'local_net':d,'net_difference':delta})
        delta=a['net']-b['net']
        assert abs(sum(contributions.values())-delta)<EPS
        gross_delta=a['gross']-b['gross'];fees_delta=b['fees']-a['fees']
        assert abs(gross_delta+fees_delta-delta)<EPS
        assert abs(fees_delta-3*(cb['admissions']-ca['admissions']))<EPS
        gross_diffs.append(gross_delta);fee_savings.append(fees_delta)
        diffs.append(delta);harm_diffs.append(cb['harmful']-ca['harmful']);benefit_diffs.append(ca['beneficial']-cb['beneficial'])
        rows_full.extend(ar);rows_control.extend(br)
        detail.append({'seed':t['seed'],'full_net':a['net'],'control_net':b['net'],'net_gain':delta,'served_correctness_change':gross_delta,'saved_fees':fees_delta,'full_counts':ca,'control_counts':cb,'changed_actions':changed,'paired_gain_decomposition':contributions})
    fc=counts(rows_full);bc=counts(rows_control)
    return {'control':control,'control_label':MODE_LABELS[control],'paired_net_gain':stats(diffs),'served_correctness_change':stats(gross_diffs),'saved_fees':stats(fee_savings),'harmful_count_reduction':stats(harm_diffs),'beneficial_count_difference':stats(benefit_diffs),'full_counts':fc,'control_counts':bc,'harmful_reduction_fraction':ratio(bc['harmful']-fc['harmful'],bc['harmful']),'baseline_harmful_avoided':ratio(len(events['avoided_harmful']),bc['harmful']),'beneficial_retention':ratio(retained_benefit,bc['beneficial']),'baseline_harmful_retained':ratio(retained_harm,bc['harmful']),'events':events,'gain_decomposition_total':{k:sum(v['net_difference'] for v in es) for k,es in events.items()},'by_seed':detail}

data={n:json.loads(f.read_text()) for n,f in INPUTS.items()}
old=json.loads((BASE/'analysis/analysis.json').read_text())
out={'source':'All locked test trials; no new method or configuration selection. Actions and signs recomputed from local_net in per-origin trial rows.','interpretation':'Each task has one fixed real observation trace and five imposed delay schedules. Seed counts are conditional replay consistency, not independent sites or participants. Pooling counts is descriptive bookkeeping; pooling utility does not estimate a cross-task population effect. A null ratio has a zero denominator.','definitions':{'harmful':'Admitted complete lease with actual local_net < 0, including interruption and admission/restoration fees.','beneficial':'Admitted complete lease with actual local_net > 0.','harmful_reduction_fraction':'(control harmful count − full harmful count) / control harmful count; can be negative.','beneficial_retention':'Number of control-admitted beneficial origin/seed leases also admitted by full / all control-admitted beneficial origin/seed leases. New full beneficial leases are counted separately.','precision':'Beneficial admissions / all admissions; not posterior coverage.','descriptive_interval':'Paired t4 interval over five imposed delay schedules on the same physical trace; not a generalization confidence interval.','event_identity':'Paired net gain equals avoided negative increments + new positive increments − missed positive increments + new negative increments; common work and fees cancel.'},'input_sha256':{n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in INPUTS.items()},'tasks':{},'aggregate_comparisons':{}}
all_comparisons={m:[] for m in MODES}
for n in TASKS:
    x=data[n];trials=x['trials'];assert [t['seed'] for t in trials]==list(range(72001,72006))
    rs=[t['results'][FULL] for t in trials]
    c=counts([r for v in rs for r in v['rows']])
    potential=[r for r in trials[0]['results']['periodic']['rows']]
    task={'label':LABELS[n],'seeds':[t['seed'] for t in trials],'eligible_leases_each_seed':[len(t['results'][FULL]['rows']) for t in trials],'full_net_by_seed':[r['net'] for r in rs],'full_net_mean':float(np.mean([r['net'] for r in rs])),'full_counts':c,'full_mean_counts':{k:c[k]/5 for k in ['admissions','harmful','beneficial','neutral']},'full_counts_each_seed':[counts(r['rows']) for r in rs],'selected':x['selected'],'full_coverage':old[n]['arms'][FULL]['pooled_coverage'],'full_informative_coverage':old[n]['arms'][FULL]['informative_coverage'],'full_admitted_coverage':old[n]['arms'][FULL]['admitted_coverage'],'comparisons':{}}
    for m in MODES:
        comp=compare(trials,m)
        assert abs(comp['paired_net_gain']['mean']-old[n]['full_comparisons'][m]['net']['mean'])<EPS
        task['comparisons'][m]=comp;all_comparisons[m].append(comp)
    out['tasks'][n]=task

for m,comps in all_comparisons.items():
    agg={'number_tasks':4,'number_task_seed_replays':20,'eligible_origin_seed_leases':sum(c['full_counts']['eligible'] for c in comps),'full_counts':{},'control_counts':{}}
    for dest in ['full_counts','control_counts']:
        for k in ['admissions','harmful','beneficial','neutral','net_lease_sum']:
            agg[dest][k]=sum(c[dest][k] for c in comps)
        c=agg[dest];c['admitted_harmful_fraction']=c['harmful']/c['admissions'] if c['admissions'] else None;c['admitted_beneficial_fraction']=c['beneficial']/c['admissions'] if c['admissions'] else None
    fc=agg['full_counts'];bc=agg['control_counts']
    agg['harmful_reduction_fraction']=ratio(bc['harmful']-fc['harmful'],bc['harmful'])
    agg['beneficial_retention']=ratio(sum(c['beneficial_retention']['numerator'] for c in comps),bc['beneficial'])
    agg['baseline_harmful_avoided']=ratio(sum(c['baseline_harmful_avoided']['numerator'] for c in comps),bc['harmful'])
    agg['event_counts']={k:sum(len(c['events'][k]) for c in comps) for k in comps[0]['events']}
    agg['paired_gain_decomposition_total']={k:sum(c['gain_decomposition_total'][k] for c in comps) for k in comps[0]['events']}
    agg['sum_paired_net_gain_bookkeeping_only']=sum(c['paired_net_gain']['sum'] for c in comps)
    agg['task_seed_signs']={k:sum(c['paired_net_gain'][k] for c in comps) for k in ['positive_seeds','tie_seeds','negative_seeds']}
    out['aggregate_comparisons'][m]=agg

(WORK/'performance_gain_analysis.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')

def frac(v):return 'NA (0 denominator)' if v['ratio'] is None else f"{v['numerator']}/{v['denominator']} ({100*v['ratio']:.2f}%)"
lines=['# Complete locked performance gain analysis','','All four tasks and all five test delay schedules (72001–72005) are retained. The same-trace schedules are not independent sites. Signs and deployment categories are recomputed from complete lease increments, and every paired return is exactly decomposed into changed actions. Saved analysis.json means agree to 1e−8. No method or parameter is selected by this analysis.','','## Task-specific paired gains','','| Task | Comparator | Mean net gain | Five gains, in seed order | Positive/tie/negative | Harmful reduction | Beneficial retention |','|---|---|---:|---|---|---|---|']
for n,t in out['tasks'].items():
    for m,c in t['comparisons'].items():
        p=c['paired_net_gain'];vals=', '.join(f'{v:g}' for v in p['by_seed'])
        lines.append(f"| {t['label']} | {MODE_LABELS[m]} | {p['mean']:.2f} | [{vals}] | {p['positive_seeds']}/{p['tie_seeds']}/{p['negative_seeds']} | {frac(c['harmful_reduction_fraction'])} | {frac(c['beneficial_retention'])} |")
lines+=['','Harmful reduction compares total harmful admissions. Beneficial retention is the paired overlap with comparator-admitted positive leases; it does not count a new positive lease as a retained one. Zero-denominator retention is undefined, not 100%.','','## Descriptive action totals across all 20 task-seed replays','','There are 740 eligible origin/seed lease records. Full admits 15: 11 harmful, 4 beneficial, 0 neutral. Thus its harmful fraction among admissions is 11/15=73.33%, and beneficial fraction 4/15=26.67%. These denominators must accompany safety wording.','','| Comparator | Admissions | Harmful | Beneficial | Full harmful reduction | Beneficial retention | Full new beneficial leases |','|---|---:|---:|---:|---|---|---:|']
for m,c in out['aggregate_comparisons'].items():
    b=c['control_counts'];lines.append(f"| {MODE_LABELS[m]} | {b['admissions']} | {b['harmful']} | {b['beneficial']} | {frac(c['harmful_reduction_fraction'])} | {frac(c['beneficial_retention'])} | {c['event_counts']['new_beneficial']} |")
lines+=['','## Net gain separates served correctness and saved fees','','For every seed, full−control net utility equals (full served gross−control served gross)+(control fees−full fees). Common fees cancel, and saved fees equal three times the reduction in admission count. Served gross already includes actual interruption losses; it is not potential classification accuracy.','','| Task | Full−joint mean net | Mean served correctness change | Mean saved fees |','|---|---:|---:|---:|']
for n,t in out['tasks'].items():
    c=t['comparisons']['joint'];lines.append(f"| {t['label']} | {c['paired_net_gain']['mean']:.2f} | {c['served_correctness_change']['mean']:.2f} | {c['saved_fees']['mean']:.2f} |")
lines+=['','## Complete changed-action gain decomposition','','Each entry is a sum over five schedules per task, with counts of replayed origin/seed events. The values below can be divided by five to recover each task mean. Benefits from losses avoided and gains newly admitted must be reported together with useful updates missed and losses newly admitted.','','| Task | Comparator | Avoided harmful: count / gain | Missed beneficial: count / change | New harmful: count / change | New beneficial: count / gain | Net sum |','|---|---|---|---|---|---|---:|']
for n,t in out['tasks'].items():
    for m,c in t['comparisons'].items():
        cells=[f"{len(c['events'][k])} / {c['gain_decomposition_total'][k]:g}" for k in ['avoided_harmful','missed_beneficial','new_harmful','new_beneficial']]
        lines.append(f"| {t['label']} | {MODE_LABELS[m]} | "+' | '.join(cells)+f" | {c['paired_net_gain']['sum']:g} |")
lines+=['','## Accurate advantage-first statements','','- Relative to the joint predictive point gate, full reduces harmful admissions from 43 to 11 across the complete replay set: 32/43=74.42%. Four of the comparator’s eight beneficial admissions are retained (50%); the lower loss count does not establish preserved recall of all beneficial updates.','- On Room 864, all five point-gate losses are avoided; gains are [16,38,37,19,38], mean29.6. This equals keeping the reference and matches the calibrated block gate, so it is uncertainty-admission benefit rather than Bayesian-exclusive gain.','- On MHEALTH, harmful point-gate admissions decrease from27 to11 (16/27=59.26%), and the sole point-gate beneficial lease is retained (1/1). Gains against the point gate are positive in all five schedules. Relative to posterior-gate-only, full avoids two additional −22 leases, harmful13→11 (2/13=15.38%), while retaining its one beneficial lease.','- On HAR, all11 joint-point harmful leases are avoided, while3/7 comparator beneficial leases are retained. Gains against the point gate are [7,14,10,−2,3], positive in4/5 schedules. Relative to the calibrated block gate, full retains both positive admissions (2/2) and adds one +18 lease: mean extra gain3.6, positive in1/5 and tied in4/5 schedules.','- Relative to periodic admission, full has positive net gain in every one of the20 task-seed comparisons. Avoiding most periodic leases contributes this result; periodic admission is weaker than the matched calibrated comparator and must not be the sole primary comparison.','- Full exactly matches the posterior-gate-only controller on both occupancy tasks and HAR. Its incremental weight contribution is confined to two MHEALTH scenario crossings, not evidence of independent weight benefit in every task.','','## Material boundaries that cannot be deleted','','- Full is6.6 mean units below the matched block gate on MHEALTH, with four negative and one positive paired schedule. It is23.2 below keeping the reference there.','- Full still makes11 harmful admissions, all on MHEALTH, versus one beneficial MHEALTH admission. A statement of zero harmful deployments applies only to the two occupancy tasks and HAR, and must identify those domains.','- Nominal one-sided coverage is not attained on MHEALTH: all70/105, informative5/24 and admitted0/12. Room864 informative coverage is0/5 despite aggregate150/155. Posterior moments or algebra audits do not remove these failures.','- The t4 intervals describe imposed-delay variability on a fixed trace. They do not provide independent-site or participant generalization evidence. Mechanism examples are two delayed replays of one MHEALTH origin and one HAR origin.','- Raw net-utility totals have different trace lengths; per-task differences remain the main reporting unit. The JSON also contains pooled bookkeeping totals, not an across-site estimated effect.','','All events, control denominators, per-seed values, conditional descriptive intervals, positive-gain retention and changed-action gain decompositions are in performance_gain_analysis.json.']
review='\n'.join(lines)+'\n'
for before,after in {
    'mean29.6':'mean 29.6','from27 to11':'from 27 to 11','harmful13→11':'harmful 13→11',
    'all11':'all 11','while3/7':'while 3/7','in4/5':'in 4/5','both positive admissions (2/2)':'both positive admissions (2/2)',
    'gain3.6':'gain 3.6','in1/5':'in 1/5','of the20':'of the 20','is6.6':'is 6.6','is23.2':'is 23.2',
    'makes11':'makes 11','all70/105':'all 70/105','informative5/24':'informative 5/24',
    'admitted0/12':'admitted 0/12','Room864':'Room 864','is0/5':'is 0/5','aggregate150/155':'aggregate 150/155',
}.items():review=review.replace(before,after)
(WORK/'performance_gain_review.md').write_text(review)
print(json.dumps({'aggregate':out['aggregate_comparisons'],'output_files':[str(WORK/'performance_gain_analysis.json'),str(WORK/'performance_gain_review.md')]},indent=2))
