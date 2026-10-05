import json,math,statistics,hashlib
from pathlib import Path
root=Path('work')
air=json.load((root/'real_air_invariant_results.json').open())
air0=json.load((root/'real_air_results.json').open())
gas=json.load((root/'real_action_results.json').open())
audit=json.load((root/'real_air_invariant_audit.json').open())
def mean(x):return statistics.mean(x)
def paired(x):
 n=len(x);m=mean(x);sd=statistics.stdev(x) if n>1 else 0;hw=1.96*sd/math.sqrt(n);pos=sum(z>1e-8 for z in x);neg=sum(z< -1e-8 for z in x);nz=pos+neg
 two=min(1,2*sum(math.comb(nz,k) for k in range(0,min(pos,neg)+1))/2**nz) if nz else 1
 return dict(mean=m,median=statistics.median(x),sd=sd,ci95_normal=[m-hw,m+hw],positive=pos,negative=neg,ties=n-pos-neg,two_sided_exact_sign_p=two,range=[min(x),max(x)])
def compact(d):
 ps=d['per_seed']; modes=list(ps[0]['results']);out={}
 for mode in modes:
  rs=[dict(s['results'][mode]) for s in ps];full=[dict(s['results']['decision_full']) for s in ps]
  for rr in rs+full:
   for old,new in [('potential_accuracy','accuracy'),('probe_fees','probe_fee'),('deploy_fees','deploy_fee'),('total_fees','total_fee')]:
    if old in rr and new not in rr:rr[new]=rr[old]
  diff=[a['net_return']-b['net_return'] for a,b in zip(full,rs)]
  n=sum(next(iter(rs))['monthly'][m]['n'] for m in rs[0].get('monthly',{})) or (10635 if 'Gas' in d['dataset'] else None)
  out[mode]=dict(net=mean([r['net_return'] for r in rs]),deployments=mean([r['deployments'] for r in rs]),potential_accuracy=mean([r['accuracy'] for r in rs]),paired_full_minus_comparator=paired(diff),per_seed_difference=diff)
  for key in ['probe_steps','probe_fee','deploy_fee','total_fee','dropped_service']:
   if all(key in r for r in rs):out[mode][key]=mean([r[key] for r in rs])
  if all('total_fee' in r for r in rs):
   out[mode]['gross_served_reward']=mean([r['net_return']+r['total_fee'] for r in rs]);out[mode]['reward_change_full_minus_comparator']=mean([a['net_return']+a['total_fee']-b['net_return']-b['total_fee'] for a,b in zip(full,rs)]);out[mode]['fee_saving_comparator_minus_full']=mean([b['total_fee']-a['total_fee'] for a,b in zip(full,rs)])
  if n and all('dropped_service' in r for r in rs):out[mode]['served_opportunities']=n-out[mode]['dropped_service'];out[mode]['test_opportunities']=n
  if 'actual_action_disagreements' in ps[0]:out[mode]['mean_actual_action_differences']=mean([s['actual_action_disagreements'][mode] for s in ps])
 return out
c=compact(air);monthly=[]
for m in air['split']['test']:
 vals={mode:dict(accuracy=mean([p['results'][mode]['monthly'][m]['accuracy'] for p in air['per_seed']]),balanced_accuracy=mean([p['results'][mode]['monthly'][m]['balanced_accuracy'] for p in air['per_seed']])) for mode in ['decision_full','context_joint','decision_diagonal','rf_c','periodic','frozen_deploy']}
 dd=[100*(p['results']['decision_full']['monthly'][m]['accuracy']-p['results']['context_joint']['monthly'][m]['accuracy']) for p in air['per_seed']]
 monthly.append(dict(month=m,n=air['per_seed'][0]['results']['decision_full']['monthly'][m]['n'],metrics=vals,paired_accuracy_pp_full_minus_joint=paired(dd)))
seedrows=[]
for p in air['per_seed']:
 a=p['results']['decision_full'];b=p['results']['context_joint'];seedrows.append(dict(seed=p['seed'],net_full=a['net_return'],net_joint=b['net_return'],net_gain=a['net_return']-b['net_return'],deploy_full=a['deployments'],deploy_joint=b['deployments'],actual_action_changes=p['actual_action_disagreements']['context_joint'],shadow=p['same_state_shadow_audit']['context_joint']))
scales=[v['scale_audit'] for p in air['per_seed'] for v in p['results']['decision_full']['audit']]
rawscores=[v['gain_score'] for p in air['per_seed'] for v in p['results']['decision_full']['audit']]
result=dict(study_status=air['study_status'],input_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['real_air_invariant_results.json','real_air_results.json','real_action_results.json','real_air_invariant_audit.json']},air_invariant=c,air_first_evaluation=compact(air0),gas=compact(gas),air_monthly=monthly,air_paired_seed_rows=seedrows,same_state_shadow=audit['same_state_shadow_summary'],sensitivity=air['sensitivity'],recovery=audit['recovery'],scale_diagnostics=dict(decision_count=len(scales),fallbacks=sum(v['fallback'] for v in scales),trace_min=min(v['tangent_trace'] for v in scales),trace_max=max(v['tangent_trace'] for v in scales)),interpretation=['Post-test algorithm change on already seen fixed Air trajectory; all Air invariant results exploratory.', '10 seeds alter simulated delay, not independent datasets; normal paired intervals conditional on fixed data, trained models, and design; no multiplicity correction.', 'Potential accuracy scores all retained complete-case records, including unserved records; gross_served_reward counts only served correct jobs and assigned synthetic prices.', 'Same-state shadow local horizon advantages are descriptive forks without future redeployment, not additive causal decomposition of entire policy return.', 'All five probe models trained and paid identically; policy controls deployment admission rather than retraining trigger.', 'Recovery means eventual three consecutive windows at least3percentage points above frozen prefix model; later deployments can occur before recovery and horizon extends to end, so not isolated recovery guarantee.', 'Current evidence establishes benefits of online archives and paired gain correction on Air, but does not establish necessity of context kernel or delay replay.', 'No new experiment or modified algorithm was run for this analysis.'])
(root/'positive_evidence_analysis.json').write_text(json.dumps(result,indent=2))
print('written',root/'positive_evidence_analysis.json')
print('Air core',json.dumps({k:{key:v for key,v in row.items() if key not in ('per_seed_difference',)} for k,row in c.items() if k in ['decision_full','context_joint','decision_diagonal','rf_c','prefix_only_archive','uncorrected_gain','periodic','frozen_deploy']},indent=1))
print('monthly accuracy difference pp',[(m['month'],round(m['paired_accuracy_pp_full_minus_joint']['mean'],3)) for m in monthly])
print('seedrows',[(v['seed'],v['net_gain'],v['actual_action_changes']) for v in seedrows])
print('scale',result['scale_diagnostics'])
