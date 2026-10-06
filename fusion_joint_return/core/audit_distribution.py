"""Post-issuance distribution audit against actual FULL-lease outcomes.

No new policy selection. Rebuild exact selected laws, verify issued score
trajectories, then compute CDF/PIT and nominal tenth-quantile coverage.
"""
from pathlib import Path
import argparse,importlib.util,sys
import numpy as np

def cdf(law,x):
    if hasattr(law,'shift'):
        if x>=1:return 1.
        if x< -1:return 0.
        return law.partial(x-law.shift)[0]
    if hasattr(law,'values'):return float(law.weights[law.values<=x].sum())
    return law.partial(x)[0]

def quantile(law,p):
    lo,hi=-1.,1.
    for _ in range(55):
        mid=(lo+hi)/2
        if cdf(law,mid)<p:lo=mid
        else:hi=mid
    return (lo+hi)/2

def main(directory,task):
    directory=Path(directory).resolve();sys.path.insert(0,str(directory));spec=importlib.util.spec_from_file_location('selected_return_audit',directory/'run_return.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);runner.verify()
    selected=runner.load(task+'/selection.json')['selected'];results=runner.load(task+'/results.json.gz')
    engine,guard,cfg0,service,fork,pre,data,cache,records=runner.worlds(task,'test');out=[];maximum=0.
    saved={(x['arm'],x['recording'],x['seed']):x for x in results['guarded']}
    for rec,seed,e,b in records:
        streams=runner.prepared_streams([(rec,seed,e,b)],pre,cfg0)
        for arm,entry in selected.items():
            cfg=dict(cfg0,**entry['config']);runner.decorate(streams,pre,cfg,arm)
            for d,r in zip(streams[0]['decisions'],saved[arm,rec['recording'],seed]['rows']):
                law=d['tail_laws'][arm];score=d['N']*law.lower_tail(r['alpha_issued'])-5
                maximum=max(maximum,abs(score-r['gate_score']))
                u=d['truegross']/d['N'];q=quantile(law,.1)
                out.append(dict(arm=arm,recording=rec['recording'],seed=seed,k=d['k'],ready=r['information_ready'],admitted=r['action'],actual_full_U=u,
                    fitted_mean=law.mean,cdf_at_full_target=cdf(law,u),nominal_tenth_quantile=q,tenth_quantile_covered=u>=q,
                    tenth_quantile_excess=d['N']*max(0,q-u),mean_error=d['N']*(law.mean-u)))
    assert maximum<1e-7
    summaries={}
    for arm in runner.ARMS:
        summaries[arm]={}
        for name,predicate in [('all',lambda r:True),('ready',lambda r:r['ready']),('admitted',lambda r:r['admitted'])]:
            rows=[r for r in out if r['arm']==arm and predicate(r)];n=len(rows)
            summaries[arm][name]=dict(n=n,nominal_tenth_quantile_coverage=[sum(r['tenth_quantile_covered'] for r in rows),n],
                mean_full_target_cdf=float(np.mean([r['cdf_at_full_target'] for r in rows])) if n else None,
                mean_gross_forecast_error=float(np.mean([r['mean_error'] for r in rows])) if n else None,
                max_tenth_quantile_excess=max((r['tenth_quantile_excess'] for r in rows),default=0.),sum_tenth_quantile_excess=sum(r['tenth_quantile_excess'] for r in rows))
    runner.dump(task+'/distribution_audit.json',dict(scope='KNOWN DEVELOPMENT data; posterior-predictive CDF diagnostic of full-target outcomes; no selection or independent coverage guarantee',summary=summaries,max_issued_score_reconstruction_error=maximum,rows=out))
    print(summaries)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--task',choices=['rss','gas'],default='rss');a=p.parse_args();main(a.directory,a.task)
