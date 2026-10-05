"""Known-RSS fixed-state intervention: preserve conditional mean/variance.

The maximum-entropy bounded law exp(a*u+b*u²) discards all remaining shape.
Original issued alpha/readiness are copied, not recalibrated or reselected.
"""
import datetime,math
import numpy as np
import run_replay as api

def maxent(mean,variance,bins=64):
    target,qw=api.c.quadrature(bins);X=np.stack((target,target**2),axis=1)
    wanted=np.array([mean,variance+mean**2]);theta=np.array([mean/max(variance,1e-8),-.5/max(variance,1e-8)])
    def evaluate(t):
        log=X@t+np.log(qw);normalizer=api.c.logsumexp(log);p=np.exp(log-normalizer)
        moments=p@X;center=X-moments;H=np.einsum('i,ij,ik->jk',p,center,center)
        return normalizer-t@wanted,moments-wanted,H,p
    for it in range(80):
        value,gradient,H,p=evaluate(theta)
        if np.max(abs(gradient))<1e-11:break
        step=np.linalg.solve(H+1e-14*np.eye(2),gradient);rate=1.
        for _ in range(40):
            proposed=theta-rate*step
            if evaluate(proposed)[0]<=value-1e-4*rate*float(gradient@step)+1e-14:break
            rate*=.5
        theta=proposed
    value,gradient,H,p=evaluate(theta);error=float(np.max(abs(gradient)))
    assert error<1e-8,(mean,variance,error)
    law=api.DiscreteLaw(target,p);return law,error

def main():
    root=api.ROOT/'postmoment_intervention';root.mkdir(exist_ok=True)
    assert not (root/'results.json').exists()
    api.dump(root/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='known RSS fixed-state diagnostic declared after CJRT development outcomes; no replacement selection',
        source_sha256=api.sha(__file__),frozen_runner_sha256=api.sha(api.ROOT/'run_replay.py'),
        contrasts='full conditional paired law versus maximum-entropy bounded law preserving exact conditional target mean and variance',
        common='joint selected bandwidth, original issued alpha, current h, mask, readiness, full history and same guard',
        log='retain every score/action and four-term netdifference; conditional maxentropy moments numericalerror and gridrefinement'))
    originals=api.load(api.ROOT/'rss_development/results.json.gz')
    originals={(r['seed'],r['arm']):r for r in originals['issued']}
    selected=api.load(api.ROOT/'rss_development/selection.json')['selected']['joint']
    engine,recovery,guard,cfg0,service,fork,data,pre,records,cache=api.db.prepared('rss348')
    cfg=dict(api.p.base_cfg(cfg0),slice_bandwidth=selected['bandwidth']);allrows=[];guarded=[];maxerr=0.;maxref=0.
    for seed in range(88001,88006):
        events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg)
        stream=api.decorate(api.core.augment(events,pre,engine.precompute(events,pre,cfg),cfg),pre,cfg)
        new=[]
        for d,old in zip(stream['decisions'],originals[(seed,'joint')]['rows']):
            p=d['tail_laws']['joint'];matrix,error=maxent(p.mean,p.variance,64);refined,referror=maxent(p.mean,p.variance,128)
            level=old['alpha_issued'];score=api.paid_score(refined,d['N'],level)['score']
            coarse=api.paid_score(matrix,d['N'],level)['score'];maxerr=max(maxerr,error,referror);maxref=max(maxref,abs(score-coarse))
            lawfull=api.paid_score(p,d['N'],level);assert abs(lawfull['score']-old['gate_score'])<1e-8
            row=dict(old,gate_score=score,gain=score+5,standardized_score=score-(d['truegross']-5),
                lower_covered=bool(score<=d['truegross']-5),action=bool(old['information_ready'] and score>0))
            new.append(row);allrows.append(dict(seed=seed,k=d['k'],D=d['localnet'],ready=old['information_ready'],alpha=level,
                full_score=old['gate_score'],moment_score=score,mean=p.mean,variance=p.variance,moment_error=referror,score_grid_error=abs(score-coarse)))
        guarded.append(dict(seed=seed,**api.db.guarded(events,pre,cfg,dict(rows=new),guard,service,130.)))
    originalg=api.load(api.ROOT/'rss_development/results.json.gz')['guarded']
    originalg={r['seed']:r for r in originalg if r['arm']=='joint' and r['budget']==130.}
    changed=[];terms=dict(added_gain=0.,avoided_loss=0.,missed_gain=0.,incurred_loss=0.)
    for g in guarded:
        for old,row in zip(originalg[g['seed']]['rows'],g['rows']):
            if old['action']==row['action']:continue
            D=row['local_net'];category=('added_gain' if D>0 else 'incurred_loss') if old['action'] else ('missed_gain' if D>0 else 'avoided_loss')
            terms[category]+=abs(D);changed.append(dict(seed=g['seed'],k=row['k'],D=D,full_action=old['action'],moment_action=row['action'],category=category))
    values=[g['increment'] for g in guarded]
    api.dump(root/'results.json',dict(mean_increment=float(np.mean(values)),increments=values,
        beneficial=sum(g['beneficial'] for g in guarded),harmful=sum(g['harmful'] for g in guarded),admissions=sum(g['admissions'] for g in guarded),
        coverage=[sum(g['coverage'][0] for g in guarded),sum(g['coverage'][1] for g in guarded)],
        max_moment_error=maxerr,max_score_grid_error=maxref,changed=changed,four_terms=terms,rows=allrows,guarded=guarded))
    print('POSTMOMENT',values,terms,changed,'error',maxerr,'refinement',maxref,flush=True)

if __name__=='__main__':main()
