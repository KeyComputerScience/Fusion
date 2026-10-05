"""Reconstruct raw posterior/block quantities at retained supplementary crossings."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent.parent/'outputs'/'bayes_closed_loop_repro'
sys.path.insert(0,str(PACKAGE))
from cached_runner import install_cached_loaders
engine,adapter=install_cached_loaders()
from audit_canary_execution import load_study,explicit_fork

def main():
    data=json.loads((ROOT/'results.json').read_text());rows=[];maxratio=0.;minslack=2.;maxslack=0.;coverage_failures=0;forkcount=0
    for name in ('har240','mhealth319'):
        location=PACKAGE/('new_bayes_extension' if name in ('har240','mhealth319') else 'new_bayes_fusion')
        raw,pre,saved=load_study(PACKAGE/'independent_data',location,name)
        comparison='frequentist_gate' if name=='har240' else 'bayes_gate'
        for trial in data[name]['trials']:
            events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,trial['seed'],engine.BASE);stream=engine.precompute(events,pre,engine.BASE);lookup={d['k']:d for d in stream['decisions']}
            full=trial['results']['bayes_both'];control=trial['results'][comparison]
            for a,b in zip(full['rows'],control['rows']):
                if a['action']==b['action']:continue
                d=lookup[a['k']];w=np.asarray(a['weights']);wc=np.asarray(b['weights']);vu=float(w@d['U']@w);vb=float(w@d['B']@w);ratio_error=float(np.max(np.abs(d['U']-d['B']/(d['mass']+1))));maxratio=max(maxratio,ratio_error)
                fork=explicit_fork(events,a,engine.BASE)
                same_q_full_score=a['gain']-b['q_issued']*a['posterior_or_block_sd']-5.-data[name]['selected']['bayes_both']['threshold']
                same_q_control_score=b['gain']-a['q_issued']*b['posterior_or_block_sd']-5.-data[name]['selected'][comparison]['threshold']
                rows.append(dict(dataset=name,seed=trial['seed'],origin=a['k'],control=comparison,full_action=a['action'],control_action=b['action'],mass=d['mass'],blocks=d['blocks'],predictive_tangent_scale=d['scale'],raw_directional_posterior_variance=vu,raw_directional_block_variance=vb,posterior_concentration_identity_error=ratio_error,full_weights=w.tolist(),control_weights=wc.tolist(),weight_l2=float(np.linalg.norm(w-wc)),full_gain=a['gain'],control_gain=b['gain'],full_sd=a['posterior_or_block_sd'],control_sd=b['posterior_or_block_sd'],full_issued_q=a['q_issued'],control_issued_q=b['q_issued'],full_penalty=a['penalty'],control_penalty=b['penalty'],full_gate_score=a['gate_score'],control_gate_score=b['gate_score'],full_score_with_control_q=same_q_full_score,control_score_with_full_q=same_q_control_score,action_difference_value=fork['actual'] if a['action'] else -fork['actual'],true_gross=fork['gross'],local_net=fork['actual'],opportunity_slack=fork['slack'],covered=fork['covered'],maturity=a['maturity']))
    # Independent extra bound audit for all original extension forks; no new
    # outcome influences controller states or the previously frozen runner.
    for name,task in data.items():
        folder=PACKAGE/('new_bayes_fusion' if name.startswith('occupancy') else 'new_bayes_extension');raw,pre,saved=load_study(PACKAGE/'independent_data',folder,name)
        for trial in task['trials']:
            events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,trial['seed'],engine.BASE)
            for mode,result in trial['results'].items():
                for row in result['rows']:
                    fork=explicit_fork(events,row,task['selected'][mode]);forkcount+=1;minslack=min(minslack,fork['slack']);maxslack=max(maxslack,fork['slack']);coverage_failures+=int(not fork['covered']);assert -1e-8<=fork['slack']<=2.+1e-8
    out=dict(kind='Retained action crossings; raw posterior reconstruction and q-swapped diagnostics, no controller changes',runner_sha256=hashlib.sha256((ROOT/'run_extension.py').read_bytes()).hexdigest(),rows=rows,max_posterior_concentration_identity_error=maxratio,total_reconstructed_forks=forkcount,min_opportunity_slack=minslack,max_opportunity_slack=maxslack,uncovered_forks=coverage_failures,all_slack_bounds_passed=True)
    (ROOT/'crossing_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
