from pathlib import Path
import gzip,json,csv,hashlib
ROOT=Path(__file__).resolve().parent
out={}
rows=[]
for task in ('rss','gas'):
 data=json.load(open(ROOT/task/'grid_diagnostic.json'))
 trajectories=json.load(gzip.open(ROOT/task/'grid_diagnostic_trajectories.json.gz','rt'))
 for x in data['grid']:
  s=x['summary'];rows.append(dict(task=task,arm=x['arm'],**x['config'],mean_increment=s['mean_increment'],admissions=s['admissions'],beneficial=s['beneficial'],harmful=s['harmful'],negative_loss=s['negative_loss'],admitted_covered=s['admitted_coverage'][0],admitted_count=s['admitted_coverage'][1],admitted_max_excess=s['admitted_max_excess'],admitted_total_excess=s['admitted_total_excess']))
 chosen={t['arm']:t for t in trajectories if t['config']=={'bandwidth':1.,'zeta':1.}}
 by={(g['arm'],g['seed'],g['recording']):g for t in chosen.values() for g in t['guarded']}
 comparisons={}
 for control in ('marginal','factorized','zero_increment'):
  terms={key:dict(count=0,value=0.) for key in ('added_gain','avoided_loss','missed_gain','incurred_loss')};changes=[]
  for J in chosen['paired']['guarded']:
   C=by[control,J['seed'],J['recording']]
   for j,c in zip(J['rows'],C['rows']):
    assert j['k']==c['k'] and j['local_net']==c['local_net']
    if j['action']==c['action']:continue
    D=j['local_net'];changes.append(dict(recording=J['recording'],seed=J['seed'],k=j['k'],actual_net=D,paired_action=j['action'],control_action=c['action'],paired_score=j['gate_score'],control_score=c['gate_score']))
    if D:
     key=('added_gain' if D>0 else 'incurred_loss') if j['action'] else ('missed_gain' if D>0 else 'avoided_loss');terms[key]['count']+=1;terms[key]['value']+=abs(D)
  total=terms['added_gain']['value']+terms['avoided_loss']['value']-terms['missed_gain']['value']-terms['incurred_loss']['value']
  expected=sum(g['increment'] for g in chosen['paired']['guarded'])-sum(g['increment'] for g in chosen[control]['guarded'])
  assert abs(total-expected)<1e-9
  comparisons[control]=dict(terms=terms,changes=changes,pooled_difference=total,mean_difference=total/5)
 out[task]=dict(operating_point={'bandwidth':1.,'zeta':1.},status='Post-primary known-data diagnostic, not selected/frozen primary policy',comparisons=comparisons)
with (ROOT/'complete_grid.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'grid_action_decomposition.json').write_text(json.dumps(out,indent=2))
print(out)
