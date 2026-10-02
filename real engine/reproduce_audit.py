"""Reproduce one preserved Air replay and audit deployment boundary fixes.

This audit does not turn the post-test exploratory study into confirmation.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types
import numpy as np
import real_air_invariant_fusion as air
from block_bayesian_fusion import block_bayesian_moments

ROOT=Path(__file__).resolve().parent

def bayesian_check(events, full, models, prior, cfg):
    errors=[]
    # Current-model re-projection of actual archived audit records at 3 origins.
    for k in (0,80,len(events)-1):
        deployed=models[4].copy()
        for j in range(k):
            if full['actions'][j]:deployed=events[j]['candidate'].copy()
        candidate=events[k]['candidate']
        records=[]
        for e in events[:k]:
            ar=e['audit'];arrival=e['origin']+max(1,e['delay'])
            if arrival<=k and ar.any():
                records.append(dict(x=e['x'][ar],p=e['source'][:,ar,:],y=e['y'][ar],context=e['context'],price=e['price'][ar],mask=e['mask'],origin=e['origin'],arrival=arrival))
        records.sort(key=lambda r:(r['arrival'],r['origin']))
        records=records[-cfg['archive']:]
        mask=np.flatnonzero(events[k]['mask']);c=events[k]['context']
        chunks=[(cfg['prior_mass'],prior)]
        for r in records:
            if np.all(r['mask'][mask]):
                a=cfg['beta']**(k-r['origin'])*math.exp(-float(np.sum((c-r['context'])**2))/cfg['context_bandwidth']**2)
                chunks.append((a,r))
        masses=[];means=[];seconds=[]
        for a,r in chunks:
            e=r['p'][mask]-np.eye(6)[r['y']][None,:,:]
            active=(r['x']@deployed).argmax(1);cand=(r['x']@candidate).argmax(1)
            b=r['price'][:,None]*(np.eye(6)[cand]-np.eye(6)[active])
            z=np.einsum('sic,ic->si',e,b)
            masses.append(a);means.append(z.mean(1));seconds.append(z@z.T/len(r['y']))
        bb=block_bayesian_moments(masses,means,seconds)
        s,_,mu=air.paired_moments(records,prior,c,k,mask,deployed,candidate,cfg,'decision_full')
        errors.append(dict(origin=k,blocks=len(chunks),mean_max_abs_difference=float(np.max(np.abs(mu-bb['predictive_mean']))),covariance_max_abs_difference=float(np.max(np.abs(s-bb['predictive_covariance']))),total_concentration=float(bb['total_concentration']),posterior_mean_covariance_trace=float(np.trace(bb['posterior_mean_covariance'])),law_total_covariance_max_abs_difference=float(np.max(np.abs(bb['predictive_covariance']-bb['expected_conditional_covariance']-bb['posterior_mean_covariance'])))))
    return errors

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data'/'AirQualityUCI.csv');p.add_argument('--output',type=Path,default=ROOT/'reproduction_audit.json');args=p.parse_args()
    started=time.perf_counter()
    with gzip.open(ROOT/'results'/'real_air_invariant_results.json.gz','rt') as f:stored=json.load(f)
    cfg=stored['config'];rows,meta=air.load_air(args.data);models,prior,q,train_rows=air.prefix_fit(rows,cfg)
    seed=99001;events=air.world(rows,models,train_rows,stored['split']['test'],seed,cfg)
    reference=next(r for r in stored['per_seed'] if r['seed']==seed)['results']
    checks={};runs={}
    for mode in ('decision_full','context_joint','decision_diagonal','quality_only','rf_c'):
        r=air.run(events,models,prior,q,cfg,mode);runs[mode]=r;ref=reference[mode]
        checks[mode]=dict(net_return=r['net_return'],reference_net_return=ref['net_return'],net_abs_difference=abs(r['net_return']-ref['net_return']),actions_identical=r['actions']==ref['actions'],deployments=r['deployments'],solver_failures=r['solver_failures'],max_kkt=r['max_kkt'])
        assert checks[mode]['net_abs_difference']<1e-9 and checks[mode]['actions_identical']
    corners=[]
    for n in (0,1,3):
        cc=copy.deepcopy(events[:12])
        for e in cc:e['mask']=np.arange(4)<n
        for mode in ('decision_full','context_joint','decision_diagonal','quality_only','rf_c','periodic'):
            r=air.run(cc,models,prior,q,cfg,mode)
            if n==0:
                assert r['deployments']==0
                assert all(not a['gain_available'] and not a['eligible'] and a['weight']==[] for a in r['audit'])
            if n==1:assert all(a['weight']==[1.] for a in r['audit'])
            assert r['solver_failures']==0
            corners.append(dict(active_sources=n,mode=mode,deployments=r['deployments'],solver_failures=r['solver_failures'],finite_return=bool(np.isfinite(r['net_return']))))
    bb=bayesian_check(events,runs['decision_full'],models,prior,cfg)
    assert max(x['covariance_max_abs_difference'] for x in bb)<1e-10
    original=types.ModuleType('original_air_invariant')
    original.__file__=str(ROOT/'real_air_invariant_fusion_original.py')
    with gzip.open(ROOT/'results'/'original_real_air_invariant_fusion.py.gz','rt') as f:original_source=f.read()
    exec(compile(original_source,original.__file__,'exec'),original.__dict__)
    stress_cfg=dict(cfg,mask_stress=True)
    stress_events=air.world(rows,models,train_rows,stored['split']['test'],seed,stress_cfg)
    stress_checks={}
    for mode in ('decision_full','context_joint','decision_diagonal','quality_only','rf_c'):
        r=air.run(stress_events,models,prior,q,stress_cfg,mode)
        old=original.run(stress_events,models,prior,q,stress_cfg,mode)
        change=max(float(np.linalg.norm(np.array(a['weight'])-np.array(b['weight']))) for a,b in zip(r['audit'],old['audit']))
        stress_checks[mode]=dict(net_abs_difference=abs(r['net_return']-old['net_return']),actions_identical=r['actions']==old['actions'],weight_max_l2_difference=change)
        assert stress_checks[mode]['net_abs_difference']<1e-9 and stress_checks[mode]['actions_identical'] and change<1e-12
    audit=dict(scope='One preserved exploratory Air seed, five main comparators; full 3-source outage stress compatibility; synthetic mask API corners; analytic Bayesian representation of real archive records.',study_status=stored['study_status'],python_version=sys.version,numpy_version=np.__version__,seed=seed,data_sha256=meta['csv_sha256'],main_replay=checks,three_source_stress_compatibility=stress_checks,original_invariant_source_sha256=hashlib.sha256(original_source.encode()).hexdigest(),mask_corners=corners,bayesian_archive_equivalence=bb,elapsed_seconds=time.perf_counter()-started,not_established=['No new independent sensor dataset/trajectory was tested.','Bayesian block uncertainty has no separately estimated performance gain.','0/1-source checks validate behavior, not missing-source performance benefit.','This does not rerun all ten scenarios or establish field deployment costs.'])
    args.output.write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
