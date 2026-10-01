"""Machine-readable summaries corresponding exactly to the Performance tables."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tables'
def read(p):return json.loads((ROOT/'results'/p).read_text())
def write(file,rows):
 cols=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/file).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
source=read('source_ablation/summary.json'); seeds=read('source_ablation/seed_results.json')
methods=['workload_only','operating_only','service_only','baseline_EF','baseline_FF','baseline_IMSE','baseline_IMSE_capped','baseline_EWA','baseline_BOA','baseline_BOA_capped','baseline_DS','original_reliability','revised_residual_fusion']
clean=[]
for r in source:
 if r['scenario']=='clean' and r['method'] in methods:
  r=dict(r);ws=[x['maximum_linear_weight'] for x in seeds if x['scenario']=='clean' and x['method']==r['method'] and x['maximum_linear_weight'] is not None]
  r['global_observed_maximum_linear_weight']=max(ws) if ws else None;clean.append(r)
write('table4_clean.csv',clean)
quality=read('quality/summary.json')
write('table5_interventions.csv',[r for r in quality if r['method'] in ['original_reliability','revised_residual_fusion','baseline_DS']])
queue=read('queue_summary.json')
write('table6_AB_queue.csv',[r for r in queue if r['study']=='original' or r.get('scenario')=='joint_shift'])
write('table7_ablations.csv',[r for r in quality if ('without_' in r['method'] or 'with_raw_disagreement' in r['method'])])
write('table8_modern.csv',read('official_scalar/summary.json'))
write('table9_stationary.csv',[r for r in queue if r.get('scenario')=='stationary'])
budget=read('update_necessity/budget_sensitivity/analysis.json')['groups']
def flatten(r):
 out={}
 for k,v in r.items():
  if isinstance(v,dict):out.update({k+'_'+kk:vv for kk,vv in v.items()})
  else:out[k]=v
 return out
for scenario,file in [('relation_shift','table6_C_updates.csv'),('no_shift','table10_negative_control.csv')]:
 write(file,[flatten(r) for r in budget if r['scenario']==scenario and (r['method']!='no_rt' or r['budget_updates']==8)])
print('Eight table CSV files saved, preserving unrounded values and units.')
