"""Recompute manuscript endpoints from immutable trial logs, not summary fields."""
from pathlib import Path
import argparse, json, math
import numpy as np

NAMES=('occupancy357','occupancy864','mhealth319','har240')
MODES=('mean_only','diagonal','joint','bayes_weights','bayes_gate','bayes_both','frequentist_gate','diagonal_posterior_gate','posterior_unused','periodic','frozen')

def interval(x):
    x=np.asarray(x,float);half=2.7764451051977987*x.std(ddof=1)/math.sqrt(len(x))
    return dict(values=x.tolist(),mean=float(x.mean()),conditional_delay_t4_interval=[float(x.mean()-half),float(x.mean()+half)])

def analyze(data):
    out={}
    for name in NAMES:
        x=data[name];trials=x['trials'];rout={}
        for mode in MODES:
            if mode not in trials[0]['results']:continue
            rs=[t['results'][mode] for t in trials];rows=[r for s in rs for r in s['rows']];inf=[r for r in rows if r['disagreement']>0];acts=[r for r in rows if r['action']]
            weighted=[];identity=[];margin=x['selected'].get(mode,{}).get('threshold',0)
            for t,s in zip(trials,rs):
                if mode in ('bayes_both','bayes_gate','frequentist_gate','diagonal_posterior_gate'):
                    lower=sum(r['gate_score']+margin for r in s['rows'] if r['action'])
                    excess=sum(r['posterior_or_block_sd']*max(0,r['standardized_score']-r['q_issued']) for r in s['rows'] if r['action'])
                    weighted.append(lower-excess)
                identity.append(s['net']-t['baseline']['net']-sum(r['local_net'] for r in s['rows'] if r['action']))
            rout[mode]=dict(net_mean=float(np.mean([s['net'] for s in rs])),gain_vs_reference=interval([s['net']-t['baseline']['net'] for t,s in zip(trials,rs)]),admissions=float(np.mean([s['deployments'] for s in rs])),harmful=float(np.mean([s['harmful'] for s in rs])),beneficial=float(np.mean([s['beneficial'] for s in rs])),potential_accuracy_percent=100*float(np.mean([s['accuracy'] for s in rs])),pooled_coverage=dict(covered=sum(r['lower_covered'] for r in rows),total=len(rows),rate=sum(r['lower_covered'] for r in rows)/len(rows)),informative_coverage=dict(covered=sum(r['lower_covered'] for r in inf),total=len(inf),rate=sum(r['lower_covered'] for r in inf)/len(inf) if inf else None),admitted_coverage=dict(covered=sum(r['lower_covered'] for r in acts),total=len(acts),rate=sum(r['lower_covered'] for r in acts)/len(acts) if acts else None),mean_weighted_certificate=float(np.mean(weighted)) if weighted else None,max_closed_loop_error=max(map(abs,identity)),max_calibration_identity_error=max(abs(s['calibration_identity_error']) for s in rs))
        comparisons={}
        full=[t['results']['bayes_both'] for t in trials]
        for mode in rout:
            base=[t['results'][mode] for t in trials];differences=[];crossings=[]
            for t,a,b in zip(trials,full,base):
                count=0
                for ra,rb in zip(a['rows'],b['rows']):
                    assert ra['k']==rb['k']
                    if ra['action']!=rb['action']:
                        count+=1;crossings.append(dict(seed=t['seed'],k=ra['k'],full_action=ra['action'],control_action=rb['action'],actual_lease_net=ra['local_net'],full_score=ra['gate_score'],control_score=rb['gate_score'],full_weights=ra['weights'],control_weights=rb['weights'],full_q=ra['q_issued'],control_q=rb['q_issued'],full_sd=ra['posterior_or_block_sd'],control_sd=rb['posterior_or_block_sd']))
                differences.append(count)
            comparisons[mode]=dict(net=interval([a['net']-b['net'] for a,b in zip(full,base)]),action_differences=differences,mean_action_differences=float(np.mean(differences)),crossings=crossings)
        out[name]=dict(split=x['split'],selected={k:dict(lam=v['lam'],margin=v['threshold']) for k,v in x['selected'].items()},participants=x.get('participant_overlap'),arms=rout,full_comparisons=comparisons)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--room-results',type=Path,default=Path('work/new_bayes_fusion/results.json'));p.add_argument('--wearable-results',type=Path,default=Path('work/new_bayes_extension/results.json'));p.add_argument('--out',type=Path,default=Path('work/new_bayes_analysis'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data={**json.loads(a.room_results.read_text()),**json.loads(a.wearable_results.read_text())};out=analyze(data);(a.out/'analysis.json').write_text(json.dumps(out,indent=2))
    for n,v in out.items():
        f=v['arms']['bayes_both'];print(n,'fullnet',f['net_mean'],'coverage',f['pooled_coverage'],'informative',f['informative_coverage'],'admitted',f['admitted_coverage'],'weightedLB',f['mean_weighted_certificate'])
        for c in ('joint','bayes_gate','frequentist_gate','frozen','diagonal_posterior_gate'):
            if c in v['full_comparisons']:print(c,v['full_comparisons'][c]['net'], 'changed',v['full_comparisons'][c]['action_differences'])
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        labels=['Occ.357','Room864','MHEALTH','HAR'];pos=np.arange(4);fig,axs=plt.subplots(1,3,figsize=(12,3.3),layout='constrained')
        for offset,c,label in [(-.2,'joint','Full − joint'),(.2,'frequentist_gate','Full − matched B-gate')]:
            y=[out[n]['full_comparisons'][c]['net']['mean'] for n in NAMES];axs[0].bar(pos+offset,y,width=.38,label=label)
        axs[0].axhline(0,color='black',lw=.7);axs[0].set_ylabel('Mean net utility difference');axs[0].legend(fontsize=7)
        for offset,c,label in [(-.2,'joint','Joint plug-in'),(.2,'bayes_both','Full posterior')]:axs[1].bar(pos+offset,[out[n]['arms'][c]['harmful'] for n in NAMES],width=.38,label=label)
        axs[1].set_ylabel('Mean harmful deployments');axs[1].legend(fontsize=7)
        for offset,c,label in [(-.2,'pooled_coverage','All issued'),(.2,'informative_coverage','Nonzero disagreement')]:axs[2].bar(pos+offset,[100*out[n]['arms']['bayes_both'][c]['rate'] for n in NAMES],width=.38,label=label)
        axs[2].axhline(90,ls='--',color='black',lw=.7);axs[2].set_ylabel('One-sided coverage (%)');axs[2].legend(fontsize=7)
        for ax in axs:ax.set_xticks(pos,labels,rotation=20)
        fig.savefig(a.out/'posterior_execution.png',dpi=220);fig.savefig(a.out/'posterior_execution.pdf');plt.close(fig)
    except ImportError:pass

if __name__=='__main__':main()
