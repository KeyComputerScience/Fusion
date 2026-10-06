"""Complete post-primary, known-data operating grid; never selects a policy."""
import run_ratio as r
from pathlib import Path
import argparse
np=r.np;api=r.api;ROOT=r.ROOT

def run(task):
    r.verify();engine,guard,cfg,service,fork,pre,cache,records,streams,native=r.prepare(task,'test')
    seedlist=r.base.TEST if task=='rss' else r.rc.physical.TEST
    grid=[];trajectories=[]
    for arm in r.ARMS:
        for conf in r.configurations():
            gg=[]
            for (rec,seed,e,_),s in zip(records,streams):
                issued=r.issue(s,arm,conf);g=dict(recording=rec['recording'],seed=seed,arm=arm,**api.db.guarded(e,pre,cfg,issued,guard,service,130.));gg.append(g)
            sm=r.summarize(gg,seedlist)[arm];grid.append(dict(arm=arm,config=conf,summary=sm));trajectories.append(dict(arm=arm,config=conf,guarded=gg));print(task,arm,conf,sm,flush=True)
    r.dump(task+'/grid_diagnostic.json',dict(status='Post-primary full operating grid on known development data; no result replaces locked selections; all configurations retained',grid=grid,max_service_error=max(g['service_error'] for t in trajectories for g in t['guarded'])))
    r.dump(task+'/grid_diagnostic_trajectories.json.gz',trajectories)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);p.add_argument('--task',choices=['rss','gas'],default='rss');a=p.parse_args()
    if a.phase=='freeze':r.dump('grid_diagnostic_protocol.json',dict(utc=r.now(),status='Declared after primary known-data runs; full grid diagnostic only; no controller selection',runner_sha256=api.sha(Path(__file__)),primary_runner_sha256=api.sha(ROOT/'run_ratio.py'),arms=r.ARMS,grid=r.configurations(),data='Same previously inspected RSS/gas test traces',selection='No result enters primary selection or changes algorithm'))
    else:run(a.task)
