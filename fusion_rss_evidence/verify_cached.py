"""Portable NumPy-only replay of lossless frozen RSS bridge inputs.

No experimental engines, cached physical data or absolute paths are needed.
This rechecks posteriors/proposals and the fixed-reserve schedule; physical
request accounting remains established by the immutable provenance archive.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math
import numpy as np
from bridge_math import bridge_state,TAUS,FAMILIES
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(Path(path).read_text())
def replay(rows):
    pending={};spent=0.;total=0.;actions=[]
    for row in sorted(rows,key=lambda x:x['k']):
        for origin in sorted([k for k,v in pending.items() if v['maturity']<=row['k']],key=lambda k:(pending[k]['maturity'],k)):
            old=pending.pop(origin);spent+=max(0.,-old['D']);total+=old['D']
        action=bool(row['proposal'] and spent+130*len(pending)+130<=130+1e-12)
        if action:pending[row['k']]=dict(maturity=row['maturity'],D=row['D'])
        assert spent+130*len(pending)<=130+1e-8
        actions.append((row['k'],action))
    for old in pending.values():spent+=max(0.,-old['D']);total+=old['D']
    assert spent<=130+1e-8
    return total,spent,actions
def verify():
    manifest=load(ROOT/'cached_manifest.json')
    for name,key in (('cached_states.json.gz','cache_sha256'),('bridge_math.py','math_sha256'),('verify_cached.py','verifier_sha256'),
                     ('predictions.csv','predictions_sha256'),('summary.json','summary_sha256')):
        assert sha(ROOT/name)==manifest[key],name
    states=load(ROOT/'cached_states.json.gz');assert len(states)==manifest['state_count']
    with (ROOT/'predictions.csv').open(newline='') as f:
        saved={(int(x['seed']),int(x['k']),x['family'],float(x['tau'])):x for x in csv.DictReader(f)}
    rows={};maxerr=0.;maxquad=0.
    for s in states:
        inputs=[np.asarray(s[key]) for key in ('h','alpha','locations','covariances','mu','C','P')]
        post,audit=bridge_state(*inputs);maxquad=max(maxquad,audit['quadrature_error'])
        for (family,tau),p in post.items():
            F=s['N']*p['mean'] if s['ready'] else 0.;S=s['N']*math.sqrt(p['variance']+s['norm_floor']**2)
            score=F-s['q']*S-5;record=saved[(s['seed'],s['k'],family,tau)]
            for field,value in (('F',F),('S',S),('score',score),('mean',p['mean']),('variance',p['variance'])):
                maxerr=max(maxerr,abs(value-float(record[field])))
            proposal=bool(score>0);assert proposal==(record['proposal']=='True')
            rows.setdefault((s['seed'],family,tau),[]).append(dict(k=s['k'],maturity=s['maturity'],proposal=proposal,D=s['complete_fork_D']))
    assert maxerr<1e-8,maxerr
    summary=load(ROOT/'summary.json')['summaries'];maxguard=0.
    for item in summary:
        seeds=sorted({s['seed'] for s in states})
        for seed,want in zip(seeds,item['guarded_increments']):
            total,loss,actions=replay(rows[(seed,item['family'],item['tau'])]);maxguard=max(maxguard,abs(total-want))
    assert maxguard<1e-8
    return dict(passed=True,states=len(states),posterior_scores=len(saved),max_prediction_error=maxerr,
                max_guard_increment_error=maxguard,max_quadrature_error=maxquad,
                provenance_scope='cached frozen-state numeric/schedule replay; original archive carries full physical service checks')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();print(json.dumps(verify(),indent=2))
