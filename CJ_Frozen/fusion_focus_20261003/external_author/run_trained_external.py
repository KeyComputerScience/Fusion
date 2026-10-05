"""Published PDF/QMF training objectives on physical-source linear encoders.

Architecture adaptation is explicit: no claim to reproduce image/text networks.
The paid candidate/reference and exogenous feedback are the frozen common world.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parents[2]/'outputs'/'Fusion_Recovery_Repro'
# The actual project directory is configurable, avoiding an assumed checkout.
PROJECT=Path('/Users/key/Documents/Codex/2026-10-02/jih')
PACKAGE=PROJECT/'outputs'/'Fusion_Recovery_Repro'
sys.path[:0]=[str(PACKAGE/'extension'),str(PACKAGE/'base'),str(PACKAGE/'external_fusion_validation_20261002')]
import run_recovery_fusion as rec
import run_recovery_external as ext
from cached_runner import install_cached_loaders
from audit_canary_execution import load_study, explicit_service, explicit_fork
from new_data_adapter import load_dataset, make_prefix
engine,_=install_cached_loaders()
PROTOCOL=json.loads((PACKAGE/'extension'/'protocol.json').read_text())
CFG=dict(engine.BASE,**PROTOCOL['extension'])

def sm(z):
    a=z-z.max(-1,keepdims=True);a=np.exp(a);return a/a.sum(-1,keepdims=True)
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))
def ce(z,y):
    p=sm(z);return -np.log(np.maximum(p[np.arange(len(y)),y],1e-300))
def lse(z):
    a=z.max(-1);return a+np.log(np.exp(z-a[...,None]).sum(-1))

def pdf_weights(p,confidence,calibrate):
    m,n,c=p.shape
    if m==1:return np.ones((m,n))
    confidence=np.clip(confidence,1e-6,1-1e-6)
    logs=np.log(confidence);total=logs.sum(0,keepdims=True)
    cb=confidence+(total-logs)/(total+1e-8)
    if calibrate:
        du=np.abs(p-1/c).mean(-1);den=du.sum(0,keepdims=True)-du
        relative=np.ones_like(du);ok=den>1e-12
        relative[ok]=np.minimum(1,(m-1)*du[ok]/den[ok]);cb*=relative
    return sm(cb.T).T

def gradients(xs,ws,heads,y,rule,strength,history=None):
    """Published stopped-gradient TCP and history ranking semantics."""
    z=np.stack([x@w for x,w in zip(xs,ws)]);p=sm(z);n=len(y);m=len(ws)
    if rule=='pdfj':
        conf=np.stack([sigmoid(x@a) for x,a in zip(xs,heads)])
        v=pdf_weights(p,conf,False)
    else:v=.1*lse(z)
    fused=np.einsum('sn,snc->nc',v,z)
    gf=sm(fused);gf[np.arange(n),y]-=1;gf/=n
    loss=float(ce(fused,y).mean());gws=[];gh=[]
    for s,(x,w) in enumerate(zip(xs,ws)):
        gp=p[s].copy();gp[np.arange(n),y]-=1;gp/=n
        gz=gp+v[s,:,None]*gf;loss+=float(ce(z[s],y).mean())
        if rule=='qmfj':
            gz+=.1*p[s]*np.sum(gf*z[s],axis=1)[:,None]
            hh=history[s];span=hh.max()-hh.min()
            normalized=(hh-hh.min())/span if span>1e-12 else np.zeros(n)
            target=np.sign(normalized-np.roll(normalized,-1))
            margin=np.abs(normalized-np.roll(normalized,-1))
            difference=target*(v[s]-np.roll(v[s],-1))-margin
            loss+=strength*float(np.maximum(difference,0).mean())
            gconf=strength*target*(difference>0)/n
            gconf-=np.roll(gconf,1)
            gz+=.1*p[s]*gconf[:,None]
        else:
            target=p[s,np.arange(n),y].copy() # published detached TCP target
            ghead=strength*np.sign(conf[s]-target)*conf[s]*(1-conf[s])/n
            gh.append(x.T@ghead);loss+=strength*float(np.abs(conf[s]-target).mean())
        gws.append(x.T@gz+CFG['l2']*w)
    return gws,gh,loss,np.stack([ce(zz,y) for zz in z])

def train(xs,y,rule,strength,steps,ws=None,heads=None):
    c=CFG_CURRENT['classes']
    if ws is None:ws=[np.zeros((x.shape[1],c)) for x in xs]
    else:ws=[w.copy() for w in ws]
    if heads is None:heads=[np.zeros(x.shape[1]) for x in xs]
    else:heads=[a.copy() for a in heads]
    history=np.zeros((len(xs),len(y)));losses=[];clips=0
    for _ in range(steps):
        gw,gh,loss,individual=gradients(xs,ws,heads,y,rule,strength,history)
        for s,g in enumerate(gw):
            norm=np.linalg.norm(g)
            if norm>10:g=g*10/norm;clips+=1
            ws[s]-=CFG['lr']*g
        if rule=='pdfj':
            for s,g in enumerate(gh):heads[s]-=CFG['lr']*g
        history+=individual;losses.append(loss)
    if not all(np.isfinite(w).all() for w in ws+heads):raise ValueError('nonfinite encoder')
    return ws,heads,dict(steps=steps,first_loss=losses[0],last_loss=losses[-1],gradient_clips=clips,
        ranking_history='cumulative CE within this training buffer; reset at changed buffer',
        tcp_target='detached true-class probability; published author-code L1')

def external_world(events,pre,rule,strength,initial):
    ws,heads,_=initial;ws=[w.copy() for w in ws];heads=[w.copy() for w in heads]
    pending=[];tx=pre['train_x'].copy();ty=pre['train_y'].copy();out=[];checks=[]
    for k,e in enumerate(events):
        for due,x,y in pending:
            if due<=k:tx=np.concatenate((tx,x))[-CFG['train_buffer']:];ty=np.concatenate((ty,y))[-CFG['train_buffer']:]
        pending=[x for x in pending if x[0]>k]
        if e['probe']:
            ws,heads,tr=train([tx[:,ids] for ids in pre['features'][:-1]],ty,rule,strength,CFG['probe_steps'],ws,heads);checks.append(dict(k=k,**tr))
        z=np.stack([e['x'][:,ids]@w for ids,w in zip(pre['features'][:-1],ws)]);p=sm(z);ids=np.flatnonzero(e['mask'])
        if rule=='pdfj':
            confidence=np.stack([sigmoid(e['x'][:,pre['features'][s]]@heads[s]) for s in range(pre['m'])]);v=pdf_weights(p[ids],confidence[ids],True)
        else:v=.1*lse(z[ids])
        pf=sm(np.einsum('sn,snc->nc',v,z[ids]))
        out.append(dict(e,p=p,external_probability=pf,external_weights=v.T))
        if (~e['audit']).any():pending.append((k+e['delay'],e['x'][~e['audit']].copy(),e['y'][~e['audit']].copy()))
    return out,checks

def scalar_stream(events,pre):
    base=engine.precompute(events,pre,CFG);pending=[];arrived=[];rows=[];pp=dict(pre,m=1)
    for d in base['decisions']:
        k=d['k'];e=events[k];arrived.extend(r for r in pending if r['maturity']<=k);pending=[r for r in pending if r['maturity']>k];arrived=arrived[-CFG['lease_archive']:]
        ids=d['ids'];pf=e['external_probability'];current=dict(d,ids=np.array([0]),context=e['context'],candidate=e['candidate'],reference=e['reference'])
        eligible=[r for r in arrived if r['physical_mask'][ids].all()];mm=rec.lease_moments(eligible,current,pp,CFG)
        pc=(e['x']@e['candidate']).argmax(1);pr=(e['x']@e['reference']).argmax(1);b=np.eye(pre['classes'])[pc]-np.eye(pre['classes'])[pr]
        h=float(np.mean(np.einsum('ic,ic->i',pf,b)))
        rows.append(dict(d,external=dict(h=h,mu=float(mm['mu'][0]),B=float(mm['B'][0,0]),U=float(mm['U'][0,0]),mass=mm['mass'],blocks=mm['blocks'],eligible=mm['eligible'],source_influences=e['external_weights'].tolist(),tcp=None)))
        future=[]
        for j in range(k,k+CFG['horizon']):
            f=events[j];cd=(CFG['probe_drop'] if f['probe'] else 0)+(2 if f['refresh'] else 0)
            future.append(dict(x=f['x'],y=f['y'],keep=np.arange(len(f['x']))>=cd))
        pending.append(dict(k=k,maturity=d['maturity'],context=e['context'],x=e['x'],p=pf[None].copy(),mask=np.ones(1,bool),physical_mask=e['mask'],future=future,N=d['N']))
    return dict(base,decisions=rows)

def study(name):
    entry=PROTOCOL['studies'][name]
    if entry.get('new_task'):
        data,spec=load_dataset(PACKAGE/'extension'/'data',name);pre=make_prefix(data,spec,CFG,engine)
    else:data,pre,_=load_study(PACKAGE/'base'/'independent_data',PACKAGE/'base'/entry['folder'],name)
    return data,pre,entry

def run(args):
    global CFG_CURRENT
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(utc=rec.now(),task_list=args.datasets,old_test_outcomes_known=True,rules=['pdfj','qmfj'],
        strengths=[0.,.25,1.],margins=[0.,4.,12.],trials=27,
        architecture='linear physical-source encoders; sigmoid-linear TCP heads on features',
        objectives='per-source CE + fused CE; PDF detached confidence plus L1 TCP; QMF differentiable energy plus author cumulative-loss ranking',
        source_commits={'pdf':'864867426cde076fb3d0529d4df6b440d6963451','qmf':'fe6c4c6ef7cb23f0a89594ee413d485f1854268b'},
        gradient_clip=10.,steps=500,online_steps=20,lr=.08,ranking_buffer_reset=True,
        statement='training-objective/physical-architecture adaptations; original architectures/benchmark ranking not claimed')
    if args.phase=='calibrate':
        if (out/'freeze.json').exists():raise ValueError('already frozen')
        rec.dump(out/'protocol.json',protocol);rec.dump(out/'pre_calibration_freeze.json',dict(utc=rec.now(),runner_sha=rec.sha(__file__),protocol_sha=rec.sha(out/'protocol.json')))
        hashes={}
        for name in args.datasets:
            data,pre,entry=study(name);CFG_CURRENT=pre;ev=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,CFG) for s in entry.get('calibration_seeds',PROTOCOL['calibration_seeds'])]
            grid=[];models={};training={}
            for rule in ('pdfj','qmfj'):
                for strength in (0.,.25,1.):
                    initial=train([pre['train_x'][:,i] for i in pre['features'][:-1]],pre['train_y'],rule,strength,CFG['initial_steps'])
                    models[f'{rule}:{strength}']=dict(ws=[w.tolist() for w in initial[0]],heads=[w.tolist() for w in initial[1]]);training[f'{rule}:{strength}']=initial[2]
                    streams=[]
                    for e in ev:
                        world,_=external_world(e,pre,rule,strength,initial);streams.append(scalar_stream(world,pre))
                    q0,scores=ext.qi(streams,pre,CFG,rule+'_calibrated')
                    for margin in (0.,4.,12.):
                        cfg=dict(CFG,threshold=margin);rs=[ext.run(engine,s,pre,cfg,rule+'_calibrated',q0,True) for s in streams]
                        grid.append(dict(rule=rule,strength=strength,margin=margin,q0=q0,mean_selection_net=float(np.mean([r['selection_net'] for r in rs])),q_fit_scores=scores))
                    print('CAL',name,rule,strength,flush=True)
            selected={r:max((g for g in grid if g['rule']==r),key=lambda g:(g['mean_selection_net'],-g['margin'],-g['strength'])) for r in ('pdfj','qmfj')}
            path=out/(name+'_selection.json');rec.dump(path,dict(dataset=name,selected=selected,grid=grid,models=models,training=training,data_hashes=data['hashes'],split=pre['split']));hashes[name]=rec.sha(path)
            print('SELECT',name,selected,flush=True)
        rec.dump(out/'freeze.json',dict(utc=rec.now(),runner_sha=rec.sha(__file__),protocol_sha=rec.sha(out/'protocol.json'),selection_sha=hashes));return
    freeze=json.loads((out/'freeze.json').read_text());assert freeze['runner_sha']==rec.sha(__file__)
    results={};checks=[]
    for name in args.datasets:
        data,pre,entry=study(name);CFG_CURRENT=pre;path=out/(name+'_selection.json');assert rec.sha(path)==freeze['selection_sha'][name];selected=json.loads(path.read_text());trials=[]
        for seed in entry.get('test_seeds',PROTOCOL['test_seeds']):
            events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,CFG);rr={};tr={}
            for rule in ('pdfj','qmfj'):
                s=selected['selected'][rule];model=selected['models'][f"{rule}:{s['strength']}"];initial=([np.array(w) for w in model['ws']],[np.array(w) for w in model['heads']],None)
                world,work=external_world(events,pre,rule,s['strength'],initial);stream=scalar_stream(world,pre);cfg=dict(CFG,threshold=s['margin']);r=ext.run(engine,stream,pre,cfg,rule+'_calibrated',s['q0'])
                check=rec.execute_check(events,pre,cfg,r,explicit_service,explicit_fork);assert check['passed'];checks.append(dict(dataset=name,seed=seed,rule=rule,**check));rr[rule]=r;tr[rule]=work
            trials.append(dict(seed=seed,results=rr,online_training=tr));print('TEST',name,seed,{k:round(v['net'],2) for k,v in rr.items()},flush=True)
        results[name]=dict(selected=selected['selected'],trials=trials);rec.dump(out/(name+'_results.json'),results[name])
    rec.dump(out/'results.json',results);rec.dump(out/'checks.json',dict(passed=True,trajectories=len(checks),forks=sum(c['forks'] for c in checks),max_service_error=max(max(c['service_errors'].values()) for c in checks),checks=checks))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=('calibrate','test'));ap.add_argument('--datasets',nargs='+',default=list(PROTOCOL['studies']));ap.add_argument('--output',type=Path,required=True);run(ap.parse_args())
