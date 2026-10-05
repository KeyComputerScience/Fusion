"""Active PDF/QMF training objectives with nonlinear tabular source encoders.

Architecture adaptation, not reproduction of the original image/text models.
All known and new task outcomes are retained. No outcome-driven core revision.
"""
from __future__ import annotations
import argparse, copy, datetime, hashlib, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
PROJECT=Path('/Users/key/Documents/Codex/2026-10-02/jih')
AUTHOR=PROJECT/'work/fusion_focus_20261003/external_author'
sys.path[:0]=[str(AUTHOR),str(PROJECT/'work/fusion_focus_20261003/fresh')]
import run_trained_external as helper

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False,
    default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def forward(xs,models):
    hs=[np.tanh(x@a['W']) for x,a in zip(xs,models)]
    hb=[np.c_[h,np.ones(len(h))] for h in hs]
    z=np.stack([h@a['V'] for h,a in zip(hb,models)])
    conf=np.stack([helper.sigmoid(h@a['tcp']) for h,a in zip(hb,models)])
    return hs,hb,z,helper.sm(z),conf

def loss_grad(xs,y,models,rule,strength,history,l2=.003):
    hs,hb,z,p,conf=forward(xs,models);m=len(models);n=len(y)
    v=helper.pdf_weights(p,conf,False) if rule=='pdf' else .1*helper.lse(z)
    fused=np.einsum('sn,snc->nc',v,z);gf=helper.sm(fused)
    gf[np.arange(n),y]-=1;gf/=n
    loss=float(helper.ce(fused,y).mean());grad=[]
    for s,(x,a) in enumerate(zip(xs,models)):
        gp=p[s].copy();gp[np.arange(n),y]-=1;gp/=n
        dz=gp+v[s,:,None]*gf;loss+=float(helper.ce(z[s],y).mean())
        if rule=='qmf':
            dz+=.1*p[s]*np.sum(gf*z[s],axis=1)[:,None]
            hh=history[s];span=hh.max()-hh.min()
            normalized=(hh-hh.min())/span if span>1e-12 else np.zeros(n)
            target=np.sign(normalized-np.roll(normalized,-1))
            margin=np.abs(normalized-np.roll(normalized,-1))
            hinge=target*(v[s]-np.roll(v[s],-1))-margin
            loss+=strength*float(np.maximum(hinge,0).mean())
            gc=strength*target*(hinge>0)/n;gc-=np.roll(gc,1)
            dz+=.1*p[s]*gc[:,None]
            dt=np.zeros_like(a['tcp'])
        else:
            # Author semantics: confidence and feature/target paths detached.
            target=p[s,np.arange(n),y].copy()
            dc=strength*np.sign(conf[s]-target)*conf[s]*(1-conf[s])/n
            dt=hb[s].T@dc
            loss+=strength*float(np.abs(conf[s]-target).mean())
        dw=x.T@((dz@a['V'][:-1].T)*(1-hs[s]**2))+l2*a['W']
        dv=hb[s].T@dz+l2*a['V']
        # The detached PDF confidence path does not update the encoder.
        grad.append(dict(W=dw,V=dv,tcp=dt))
        loss+=.5*l2*float(np.sum(a['W']**2)+np.sum(a['V']**2))
    return loss,grad,np.stack([helper.ce(t,y) for t in z])

def train(xs,y,rule,lr,steps,classes,models=None):
    if models is None:
        rng=np.random.default_rng(20261003)
        models=[dict(W=rng.normal(0,1/np.sqrt(x.shape[1]),(x.shape[1],32)),
                     V=rng.normal(0,.05,(33,classes)),tcp=np.zeros(33)) for x in xs]
    else:models=copy.deepcopy(models)
    history=np.zeros((len(xs),len(y)));first=None;clips=0
    strength=1. if rule=='pdf' else .1
    for _ in range(steps):
        loss,grad,individual=loss_grad(xs,y,models,rule,strength,history)
        if first is None:first=loss
        for a,g in zip(models,grad):
            norm=np.sqrt(sum(float(np.sum(t*t)) for t in g.values()))
            factor=min(1.,10/max(norm,1e-30));clips+=norm>10
            for key in a:a[key]-=lr*factor*g[key]
        history+=individual
    if not all(np.isfinite(t).all() for a in models for t in a.values()):raise ValueError('nonfinite')
    return models,dict(first_loss=first,last_loss=loss,steps=steps,clips=clips,
        objective_strength=strength,ranking_history='cumulative within fitting buffer; reset when buffer changes')

def world(events,pre,rule,lr,initial):
    models=copy.deepcopy(initial);pending=[];tx=pre['train_x'].copy();ty=pre['train_y'].copy();out=[]
    for k,e in enumerate(events):
        for due,x,y in pending:
            if due<=k:
                tx=np.concatenate((tx,x))[-helper.CFG['train_buffer']:]
                ty=np.concatenate((ty,y))[-helper.CFG['train_buffer']:]
        pending=[t for t in pending if t[0]>k]
        if e['probe']:models,_=train([tx[:,ix] for ix in pre['features'][:-1]],ty,rule,lr,
                                    helper.CFG['probe_steps'],pre['classes'],models)
        _,_,z,p,conf=forward([e['x'][:,ix] for ix in pre['features'][:-1]],models)
        ids=np.flatnonzero(e['mask'])
        v=helper.pdf_weights(p[ids],conf[ids],True) if rule=='pdf' else .1*helper.lse(z[ids])
        pf=helper.sm(np.einsum('sn,snc->nc',v,z[ids]))
        out.append(dict(e,p=p,external_probability=pf,external_weights=v.T))
        if (~e['audit']).any():pending.append((k+e['delay'],e['x'][~e['audit']].copy(),e['y'][~e['audit']].copy()))
    return out

def gradient_checks(out):
    rng=np.random.default_rng(5003);xs=[np.c_[rng.normal(size=(6,3)),np.ones(6)] for _ in range(2)]
    y=np.array([0,1,0,1,1,0]);models,_=train(xs,y,'qmf',.02,1,2)
    history=rng.uniform(0,2,(2,6));_,g,_=loss_grad(xs,y,models,'qmf',.1,history)
    errors=[]
    for s in range(2):
        for key in ('W','V'):
            for ix in [(0,0),(1,1)]:
                old=models[s][key][ix];eps=1e-6
                models[s][key][ix]=old+eps;lp=loss_grad(xs,y,models,'qmf',.1,history)[0]
                models[s][key][ix]=old-eps;lm=loss_grad(xs,y,models,'qmf',.1,history)[0]
                models[s][key][ix]=old;errors.append(abs((lp-lm)/(2*eps)-g[s][key][ix]))
    # PDF pseudo-gradient has intentionally stopped paths. Check confidence
    # head exactly while encoder gradients use fixed co-belief/TCP targets.
    _,g,_=loss_grad(xs,y,models,'pdf',1.,history)
    for s in range(2):
        old=models[s]['tcp'][0];eps=1e-6
        _,hb,_,p,_=forward(xs,models);target=p[s,np.arange(6),y].copy()
        def head_loss():return float(np.abs(helper.sigmoid(hb[s]@models[s]['tcp'])-target).mean())
        models[s]['tcp'][0]=old+eps;lp=head_loss();models[s]['tcp'][0]=old-eps;lm=head_loss()
        models[s]['tcp'][0]=old;errors.append(abs((lp-lm)/(2*eps)-g[s]['tcp'][0]))
    assert max(errors)<1e-6,errors
    dump(out,dict(passed=True,max_error=max(errors),checked_derivatives=len(errors),
        pdf_semantics='full finite differences invalid for intentionally detached target/weights'))

def binding(args):
    spec=importlib.util.spec_from_file_location('new_adapter_for_neural',args.adapter)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    protocol=json.loads(args.task_protocol.read_text());entry=protocol['studies'][args.task]
    helper.CFG=dict(helper.engine.BASE,**protocol['extension'])
    data,derivation=adapter.load_dataset(args.data,args.task)
    pre=adapter.make_prefix(data,derivation,helper.CFG,helper.engine)
    return data,pre,entry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['declare','check','calibrate','test'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--task');ap.add_argument('--adapter',type=Path)
    ap.add_argument('--data',type=Path);ap.add_argument('--task-protocol',type=Path);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.phase=='declare':
        assert not (out/'declaration.json').exists()
        dump(out/'declaration.json',dict(utc=now(),runner_sha256=sha(__file__),tasks=['rss348','arem366','gashome362'],
            rss_status='known physical outcomes, benchmark extension',new_tasks='not yet evaluated with these pipelines',
            rules=['PDF nonlinear full-objective','QMF nonlinear full-objective'],hidden=32,activation='tanh',
            lr_grid=[.02,.08,.2],margin_grid=[0,4,12],initial_steps=500,online_steps=20,
            author_losses={'pdf':1.,'qmf':.1},grid_trials=27,all_results_retained=True,
            primary_reserve='Fixed130',primary_budget=130,nonbinding_budget=260,
            architecture_scope='nonlinear tabular adaptation; not original BERT/ResNet networks',
            prefix_extra_training_outside_scored_interval=True,ranking_history_buffer_reset=True,
            source_commits={'pdf':'864867426cde076fb3d0529d4df6b440d6963451','qmf':'fe6c4c6ef7cb23f0a89594ee413d485f1854268b'}));return
    declaration=json.loads((out/'declaration.json').read_text());assert declaration['runner_sha256']==sha(__file__)
    if args.phase=='check':gradient_checks(out/'gradient_checks.json');return
    assert args.task in declaration['tasks'];taskout=out/args.task;taskout.mkdir(exist_ok=True)
    data,pre,entry=binding(args)
    if args.phase=='calibrate':
        assert not (taskout/'freeze.json').exists()
        events=[helper.engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,helper.CFG) for s in entry['calibration_seeds']]
        grid=[];saved={};work=[]
        for rule in ['pdf','qmf']:
            for lr in [.02,.08,.2]:
                initial,training=train([pre['train_x'][:,ix] for ix in pre['features'][:-1]],pre['train_y'],rule,lr,500,pre['classes'])
                saved[f'{rule}:{lr}']=[{k:v.tolist() for k,v in a.items()} for a in initial];work.append(dict(rule=rule,lr=lr,**training))
                streams=[helper.scalar_stream(world(e,pre,rule,lr,initial),pre) for e in events]
                q0,scores=helper.ext.qi(streams,pre,helper.CFG,rule+'_calibrated')
                for margin in [0.,4.,12.]:
                    cfg=dict(helper.CFG,threshold=margin)
                    results=[helper.ext.run(helper.engine,s,pre,cfg,rule+'_calibrated',q0,True) for s in streams]
                    grid.append(dict(rule=rule,lr=lr,margin=margin,q0=q0,selection_net=float(np.mean([r['selection_net'] for r in results])),fit_scores=scores))
                print('CAL',args.task,rule,lr,flush=True)
        selected={rule:max([t for t in grid if t['rule']==rule],key=lambda t:(t['selection_net'],-t['margin'],-t['lr'])) for rule in ['pdf','qmf']}
        dump(taskout/'selection.json',dict(selected=selected,grid=grid,models=saved,training=work,split=pre['split'],data_hashes=data['hashes']))
        dump(taskout/'freeze.json',dict(utc=now(),source_sha256=sha(__file__),selection_sha256=sha(taskout/'selection.json'),adapter_sha256=sha(args.adapter),protocol_sha256=sha(args.task_protocol)));return
    freeze=json.loads((taskout/'freeze.json').read_text())
    assert freeze['source_sha256']==sha(__file__) and freeze['selection_sha256']==sha(taskout/'selection.json')
    assert freeze['adapter_sha256']==sha(args.adapter) and freeze['protocol_sha256']==sha(args.task_protocol)
    assert not (taskout/'results.json').exists()
    selection=json.loads((taskout/'selection.json').read_text());assert selection['data_hashes']==data['hashes']
    trials=[];checks=[]
    for seed in entry['test_seeds']:
        events=helper.engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,helper.CFG);rs={}
        for rule in ['pdf','qmf']:
            s=selection['selected'][rule];models=[{k:np.array(v) for k,v in a.items()} for a in selection['models'][f"{rule}:{s['lr']}"]]
            stream=helper.scalar_stream(world(events,pre,rule,s['lr'],models),pre);cfg=dict(helper.CFG,threshold=s['margin'])
            result=helper.ext.run(helper.engine,stream,pre,cfg,rule+'_calibrated',s['q0'])
            check=helper.rec.execute_check(events,pre,cfg,result,helper.explicit_service,helper.explicit_fork)
            assert check['passed'],check;checks.append(dict(seed=seed,rule=rule,**check));rs[rule]=result
        trials.append(dict(seed=seed,results=rs));print('TEST',args.task,seed,{k:v['net'] for k,v in rs.items()},flush=True)
    dump(taskout/'results.json',dict(task=args.task,selected=selection['selected'],trials=trials))
    dump(taskout/'checks.json',dict(passed=True,checks=checks))

if __name__=='__main__':main()
