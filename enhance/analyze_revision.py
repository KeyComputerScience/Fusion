"""Generate manuscript tables directly from executed result files; no estimated cells."""
from pathlib import Path
import csv, itertools, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tables'; OUT.mkdir(exist_ok=True)
def load(p): return json.loads((ROOT/'results'/p).read_text())
NAMES={'original_reliability':'Original RF','revised_residual_fusion':'Risk RF','baseline_EF':'Equal','baseline_FF':'Prefix fixed','baseline_IMSE':'IMSE','baseline_EWA':'EWA','baseline_BOA':'BOA','baseline_DS':'Dempster','baseline_IMSE_capped':'IMSE + cap','baseline_BOA_capped':'BOA + cap','workload_only':'Workload only','operating_only':'Operating only','service_only':'Service only','no_rt':'No retraining','static':'Fixed profile','periodic':'Periodic','fusion':'Original RF','revised':'Risk RF','equal':'Equal','bpf':'IMSE','boa':'BOA','observed_loss':'Observed loss'}
def esc(s): return str(s).replace('_',r'\_').replace('%',r'\%')
def name(s): return NAMES.get(s,esc(s))
def pm(mean,sd,d=3,scale=1): return f'${mean*scale:.{d}f}\\pm{sd*scale:.{d}f}$'
def stat(rows,key,d=2,scale=1):
 a=np.array([r[key] for r in rows],float); return pm(a.mean(),a.std(ddof=1),d,scale)
def group(rows,fields):
 keys=sorted(set(tuple(r[k] for k in fields) for r in rows))
 return [(k,[r for r in rows if tuple(r[f] for f in fields)==k]) for k in keys]
def table(file,caption,label,headers,rows,width=None,long=False):
 cols=width or ('l'+'r'*(len(headers)-1))
 head=' & '.join(headers)+r' \\'+ '\n'+r'\midrule'+'\n'
 text=(r'\begin{longtable}{'+cols+'}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\\'+'\n' if long else r'\begin{table}[htbp]'+'\n'+r'\centering\small'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\begin{tabular}{'+cols+'}\n')
 text+=r'\toprule'+'\n'+head
 if long: text+=r'\endfirsthead'+'\n'+r'\toprule'+'\n'+head+r'\endhead'+'\n'
 for row in rows: text+=' & '.join(row)+r' \\'+ '\n'
 text+=r'\bottomrule'+'\n'+(r'\end{longtable}' if long else r'\end{tabular}'+'\n'+r'\end{table}')+'\n'
 (OUT/file).write_text(text)

def maxw(records,method,scenario):
 a=[r['maximum_linear_weight'] for r in records if r['method']==method and r['scenario']==scenario and r['maximum_linear_weight'] is not None]
 return f'{max(a):.3f}' if a else 'NA'
source=load('source_ablation/summary.json'); source_seed=load('source_ablation/seed_results.json')
quality=load('quality/summary.json')
methods=['workload_only','operating_only','service_only','baseline_EF','baseline_FF','baseline_IMSE','baseline_IMSE_capped','baseline_EWA','baseline_BOA','baseline_BOA_capped','baseline_DS','original_reliability','revised_residual_fusion']
rows=[]
for m in methods:
 r=next(x for x in source if x['scenario']=='clean' and x['method']==m)
 rows.append([name(m),pm(r['forecast_mae_common_mean'],r['forecast_mae_common_sd'],5),pm(r['raw_alarm_fraction_mean'],r['raw_alarm_fraction_sd'],2,100),pm(r['admission_fraction_mean'],r['admission_fraction_sd'],2,100),str(r['raw_events_detected_total'])+'/10',str(r['admitted_events_detected_total'])+'/10',maxw(source_seed,m,'clean')])
table('table4.tex','Clean observer replay on five paired seeds. Forecast MAE uses 58 identical origins per seed. Nominal raw alarms and hysteresis admissions are percentages of 57 non-event windows; event trials are two generator boundaries per seed. Max is the global observed linear weight, not the configured cap. DS has no linear weight.','tab:clean',['Method','Forecast MAE','Raw (\\%)','Admit (\\%)','Raw events','Admit events','Max'],rows)
rows=[]
for scenario,caption in [('clean','Clean'),('service_outage','Service outage'),('operating_noise','Operating noise'),('workload_conflict','Conflict'),('simultaneous_loss','All-source loss')]:
 for m in ['original_reliability','revised_residual_fusion','baseline_DS']:
  r=next(x for x in quality if x['scenario']==scenario and x['method']==m)
  rows.append([caption,name(m),pm(r['forecast_mae_common_mean'],r['forecast_mae_common_sd'],5),f"{100*r['raw_alarm_fraction_mean']:.2f}",f"{100*r['admission_fraction_mean']:.2f}",str(r['admitted_events_detected_total'])+'/10',f"{r['fallback_windows_mean']:.0f}"])
table('table5.tex','Quality interventions: five paired seeds, 28 methods in the complete CSV. These displayed methods share 58 forecast origins per seed except all-source loss (46). Raw and admitted nominal percentages are separate from admitted event detection. DS diagnostic support and raw drift scores have different semantics. Fallback counts are per seed.','tab:stress',['Scenario','Method','Forecast MAE','Raw (\\%)','Admit (\\%)','Events','Fallback'],rows,long=True)
original=load('original/seed_results.json'); service=load('service/seed_results.json')
budget=load('update_necessity/budget_sensitivity/analysis.json')['groups']
queue_groups=[]
for study,a in [('original',original),('service',service)]:
 for keys,rs in group(a, ['backbone','method'] if study=='original' else ['scenario','backbone','method']):
  v={'study':study,**dict(zip(['backbone','method'] if study=='original' else ['scenario','backbone','method'],keys)),'n':len(rs)}
  for field in ['return','completion_fraction','training_cost','deployed_jobs']:
   x=np.array([r[field] for r in rs]);v[field+'_mean']=float(x.mean());v[field+'_sd']=float(x.std(ddof=1))
  queue_groups.append(v)
(ROOT/'results/queue_summary.json').write_text(json.dumps(queue_groups,indent=2))
rows=[]
rows.append([r'\multicolumn{6}{l}{\textbf{A. Original queue protocol: actual re-execution}}'])
for b in ['dqn','ppo']:
 for m in ['no_rt','periodic','fusion']:
  rs=[r for r in original if r['backbone']==b and r['method']==m]
  rows.append([b.upper(),name(m),stat(rs,'return'),stat(rs,'completion_fraction',2,100),stat(rs,'deployed_jobs',1),stat(rs,'training_cost',3)])
rows.append([r'\multicolumn{6}{l}{\textbf{B. Joint demand/resource/action shift: actual queue execution}}'])
for b in ['dqn','ppo']:
 for m in ['static','no_rt','periodic','equal','bpf','boa','fusion','revised']:
  rs=[r for r in service if r['backbone']==b and r['method']==m and r['scenario']=='joint_shift']
  rows.append([b.upper(),name(m),stat(rs,'return'),stat(rs,'completion_fraction',2,100),stat(rs,'deployed_jobs',1),stat(rs,'training_cost',3)])
rows.append([r'\multicolumn{6}{l}{\textbf{C. Independent hidden-relation update diagnostic}}'])
rows.append(['Backbone','Trigger (steps/job)','Post acc. (\\%)','Tail acc. (\\%)','Grad. steps','Cost'])
for b in ['dqn','ppo']:
 for m in ['no_rt','observed_loss','periodic']:
  for dose in ([8] if m=='no_rt' else [8,32,128]):
   r=next(x for x in budget if x['scenario']=='relation_shift' and x['backbone']==b and x['method']==m and x['budget_updates']==dose)
   val=lambda k,d,s:pm(r[k]['mean'],r[k]['sd'],d,s)
   rows.append([b.upper(),name(m)+(f' ({dose})' if m!='no_rt' else ' (0)'),val('post_boundary_accuracy',2,100),val('last_512_accuracy',2,100),val('actual_gradient_updates',1,1),val('normalized_training_cost',3,1)])
table('table6.tex','Corrected Table 6. Panels A/B are actual queue execution: return, completion percentage, deployments and simulated profile cost. Panel C is a separate contextual-bandit diagnostic: post-boundary accuracy, final-512 accuracy, actual gradient steps and steps/1000 cost. All cells are mean $\\pm$ sample SD over five paired seeds. No retraining in A/B still adapts inference; fixed profile does not. C uses observed-loss triggers, not RF. Steps/job 32 and 128 are a declared post-hoc extension with matching 32/128-slot deployment delays; 8 is the initial design. Stationary and no-shift controls appear in Tables~\\ref{tab:stationary} and~\\ref{tab:update}.','tab:queue',['Backbone','Method','Return / Post','Complete / Tail','Deploy / Steps','Cost'],rows,long=True)
# Every ablation is shown, not selected by outcome.
rows=[]
for family in ['original','revised']:
 base='original_reliability' if family=='original' else 'revised_residual_fusion'
 labels=[r['method'] for r in quality if r['scenario']=='clean' and r['method'].startswith(family+'_') and r['method']!=base]
 for m in sorted(labels):
  label=m.replace(family+'_','').replace('without_','-- ').replace('with_raw_disagreement','raw disagreement')
  clean=next(x for x in quality if x['method']==m and x['scenario']=='clean')
  noise=next(x for x in quality if x['method']==m and x['scenario']=='operating_noise')
  outage=next(x for x in quality if x['method']==m and x['scenario']=='service_outage')
  conflict=next(x for x in quality if x['method']==m and x['scenario']=='workload_conflict')
  rows.append([family.capitalize(),esc(label),f"{clean['forecast_mae_common_mean']:.5f}",f"{100*noise['raw_alarm_fraction_mean']:.2f}",str(noise['admitted_events_detected_total'])+'/10',str(outage['admitted_events_detected_total'])+'/10',f"{100*conflict['admission_fraction_mean']:.2f}"])
table('table7.tex','All individual component ablations (28-method replay study). Clean MAE, nominal noise raw alarm percentage, admitted noise/outage event trials, and nominal conflict admission percentage. Removal of delayed qE does not remove the shared affine calibrator. Full five-scenario values and SDs are in the CSV.','tab:ablation',['Family','Variant','Clean MAE','Noise raw','Noise event','Outage event','Conflict admit'],rows,long=True)
rows=[]
for b in ['dqn','ppo']:
 for m in ['static','no_rt','periodic','equal','bpf','boa','fusion','revised']:
  rs=[r for r in service if r['backbone']==b and r['method']==m and r['scenario']=='stationary']
  rows.append([b.upper(),name(m),stat(rs,'return'),stat(rs,'completion_fraction',2,100),stat(rs,'deployed_jobs',1),stat(rs,'training_cost',3)])
table('stationary.tex','Stationary queue negative control. Same five seeds and initial models as its paired joint-shift study. No hidden action change or capacity reduction is introduced.','tab:stationary',['Backbone','Method','Return','Complete (\\%)','Deploy','Cost'],rows,long=True)
budget=load('update_necessity/budget_sensitivity/analysis.json')['groups']
rows=[]
for scenario,title in [('no_shift','No-shift negative control')]:
 rows.append([r'\multicolumn{7}{l}{\textbf{'+title+'}}'])
 for b in ['dqn','ppo']:
  for m in ['no_rt','observed_loss','periodic']:
   for dose in ([8] if m=='no_rt' else [8,32,128]):
    r=next(x for x in budget if x['scenario']==scenario and x['backbone']==b and x['method']==m and x['budget_updates']==dose)
    val=lambda k,d,s:pm(r[k]['mean'],r[k]['sd'],d,s)
    rows.append([b.upper(),name(m),'0' if m=='no_rt' else str(dose),val('post_boundary_accuracy',2,100),val('last_512_accuracy',2,100),val('actual_gradient_updates',1,1),val('normalized_training_cost',3,1)])
table('update_diagnostic.tex','Negative controls for Table 6C: independently executed contextual-bandit update diagnostic. Post accuracy uses slots 1025--3072; tail uses the final 512 slots. Budgets 8/32/128 are actual gradient steps per job with deployment delay 8/32/128 slots. All budgets are retained, including failures and costs. Budgets 32/128 are a declared post-hoc extension. No-retraining rows share one unchanged control per backbone/scenario. These are neither queue completion nor RF triggers.','tab:update',['Backbone','Trigger','Budget','Post acc. (\\%)','Tail acc. (\\%)','Grad. steps','Cost'],rows,long=True)
# Paired seed statistics: enumerated bootstrap + exact two-sided sign permutation.
def paired(a,b,metric,label):
 aa={r['seed']:r[metric] for r in a}; bb={r['seed']:r[metric] for r in b}; assert aa.keys()==bb.keys()
 ds=np.array([aa[s]-bb[s] for s in sorted(aa)])
 boots=np.array([ds[list(ix)].mean() for ix in itertools.product(range(len(ds)),repeat=len(ds))])
 perm=np.array([(ds*np.array(sign)).mean() for sign in itertools.product([-1,1],repeat=len(ds))])
 return {'comparison':label,'metric':metric,'n':len(ds),'mean_difference':float(ds.mean()),'bootstrap_ci95':np.quantile(boots,[.025,.975]).tolist(),'exact_sign_permutation_p':float(np.mean(np.abs(perm)>=abs(ds.mean())-1e-14)),'seed_differences':dict(zip(map(str,sorted(aa)),ds.tolist()))}
comparisons=[]
for a,label in [(original,'original'),(service,'joint_shift')]:
 for b in ['dqn','ppo']:
  x=[r for r in a if r['backbone']==b and r.get('scenario','joint_shift')=='joint_shift']
  ctrl=[r for r in x if r['method']=='no_rt']
  for m in ['fusion','periodic']+(['revised','static'] if label=='joint_shift' else []):
   cand=[r for r in x if r['method']==m]
   for k in ['return','completion_fraction']:
    comparisons.append(paired(cand,ctrl,k,f'{label}/{b}/{m} - no_rt'))
(ROOT/'results/queue_paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
if comparisons:
 with (ROOT/'results/queue_paired_comparisons.csv').open('w',newline='') as f:
  cols=['comparison','metric','n','mean_difference','ci95_low','ci95_high','exact_sign_permutation_p'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for r in comparisons:w.writerow({**{k:r[k] for k in cols if k in r},'ci95_low':r['bootstrap_ci95'][0],'ci95_high':r['bootstrap_ci95'][1]})
print('Generated tables from existing raw results; no experimental cells inferred.')
