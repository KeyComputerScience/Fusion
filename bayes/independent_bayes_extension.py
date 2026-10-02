"""Prospective adapters for frozen Bayesian convex controller on two new tasks.

The base executable is imported unchanged and its SHA is checked. This adapter
adds real physical source views and one prespecified diagonal posterior gate.
Every recorded subject sequence is a constructed order, not a global calendar.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import independent_bayes_fusion as engine

ROOT=Path(__file__).resolve().parent
BASE_SHA='15e3fc99d3f0935d2e66853b9b1949ef7825869c2f0d56a8ba49513838376e57'
MODES=engine.MODES+('diagonal_posterior_gate',)
TUNE=engine.MODES[:7]+('diagonal_posterior_gate',)
ORIGINAL_FORECAST=engine.decision_forecast
DIAGONAL_GATE=False


def forecast_adapter(d,pre,cfg,mode):
    if not DIAGONAL_GATE:return ORIGINAL_FORECAST(d,pre,cfg,mode)
    # Base gate execution/calibration is unchanged; only this predeclared control
    # changes the posterior matrix entering directional standard deviation.
    w,cert=engine.convex_fuse(pre['q'][d['ids']],d['S'],d['Us'],cfg,'joint')
    gain=float(d['N']*w@(d['h']-d['mu']));sd=d['N']*np.sqrt(max(0.,float(w@np.diag(np.diag(d['U']))@w))+1e-4)
    return w,cert,gain,float(sd),(gain-d['truegross'])/float(sd)

engine.decision_forecast=forecast_adapter


def initialize_q(streams,pre,cfg,mode):
    global DIAGONAL_GATE
    DIAGONAL_GATE=mode=='diagonal_posterior_gate'
    try:return engine.initial_q(streams,pre,cfg,'bayes_gate' if DIAGONAL_GATE else mode)
    finally:DIAGONAL_GATE=False


def run(stream,pre,cfg,mode,qinit,selection=False):
    global DIAGONAL_GATE
    DIAGONAL_GATE=mode=='diagonal_posterior_gate'
    try:
        result=engine.run(stream,pre,cfg,'bayes_gate' if DIAGONAL_GATE else mode,qinit,selection)
    finally:DIAGONAL_GATE=False
    # This diagnostic changes no score, action, optimizer, or feedback callback.
    lookup={d['k']:d for d in stream['decisions']}
    for r in result['rows']:
        d=lookup[r['k']];r['pessimistic_net_target']=d['truegross']-5.;r['opportunity_slack']=d['localnet']-(d['truegross']-5.);r['candidate_extra_correct']=2.-r['opportunity_slack']
        assert -1e-8<=r['opportunity_slack']<=2.+1e-8
    return result


def load_mhealth(root,spec):
    xx=[];yy=[];stamp=[];subjects=[];hashes={};raw_rows=0;retained={}
    files=sorted(root.rglob('mHealth_subject*.log'))
    mapping={int(p.stem.split('subject')[-1]):p for p in files}
    for subject in spec['participants']:
        p=mapping[subject];hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest();a=np.loadtxt(p);raw_rows+=len(a);rows=np.arange(0,len(a),50);labels=a[rows,-1].astype(int);keep=(labels>=1)&(labels<=12);selected=rows[keep];v=a[selected][:,spec['feature_raw_indices']]
        assert np.isfinite(v).all();xx.append(v);yy.append(labels[keep]-1);subjects.extend([subject]*len(v));stamp.extend([f'MHEALTH:subject{subject:02d}:raw{r:09d}' for r in selected]);retained[str(subject)]=dict(raw=len(a),downsampled=len(rows),labeled=len(v));del a
    return dict(raw=np.concatenate(xx),y=np.concatenate(yy),timestamp=stamp,subject=np.array(subjects),hashes=hashes,n=sum(len(a) for a in xx),raw_rows=raw_rows,participants=retained)


def load_har(root,spec):
    base=next(p.parent for p in root.rglob('activity_labels.txt'));xx=[];yy=[];sub=[];stamp=[];orig=[];hashes={}
    for fold in spec['folds']:
        folder=base/fold;columns=[]
        for group in spec['signal_groups']:
            for name in group:
                p=folder/'Inertial Signals'/f'{name}_{fold}.txt';hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest();a=np.loadtxt(p);assert a.shape[1]==128;columns.extend([a.mean(1),a.std(1),a.min(1),a.max(1)])
        xp=np.column_stack(columns);labelpath=folder/f'y_{fold}.txt';subjectpath=folder/f'subject_{fold}.txt';y=np.loadtxt(labelpath,dtype=int)-1;s=np.loadtxt(subjectpath,dtype=int)
        for p in (labelpath,subjectpath):hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
        assert len(xp)==len(y)==len(s);xx.append(xp);yy.append(y);sub.append(s);stamp.extend([f'HAR:subject{sid:02d}:{fold}:segment{i:06d}' for i,sid in enumerate(s)]);orig.extend(range(len(s)))
    x=np.concatenate(xx);y=np.concatenate(yy);subject=np.concatenate(sub);order=np.lexsort((np.array(orig),subject));timestamps=np.array(stamp)[order].tolist()
    return dict(raw=x[order],y=y[order],subject=subject[order],timestamp=timestamps,hashes=hashes,n=len(y),raw_rows=len(y),participants={str(s):int(np.sum(subject==s)) for s in np.unique(subject)})


def participant_overlap(data):
    n=data['n'];a=int(n*.3);b=int(n*.6);splits={'fit':set(data['subject'][:a].tolist()),'calibration':set(data['subject'][a:b].tolist()),'test':set(data['subject'][b:].tolist())}
    return dict(participants={k:sorted(v) for k,v in splits.items()},fit_test_overlap=sorted(splits['fit']&splits['test']),fit_calibration_overlap=sorted(splits['fit']&splits['calibration']),calibration_test_overlap=sorted(splits['calibration']&splits['test']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'independent_data');parser.add_argument('--output',type=Path,default=ROOT/'new_bayes_extension');parser.add_argument('--task',choices=('mhealth319','har240'));args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base_path=Path(engine.__file__);assert hashlib.sha256(base_path.read_bytes()).hexdigest()==BASE_SHA,'Base executable changed';pp=args.output/'protocol.json';protocol=json.loads(pp.read_text());sha=hashlib.sha256(pp.read_bytes()).hexdigest();adapter_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();freeze=dict(base_sha=BASE_SHA,adapter_sha=adapter_sha,protocol_sha=sha,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_metrics_read=False);freeze_path=args.output/'pretest_freeze.json'
    if freeze_path.exists():
        oldfreeze=json.loads(freeze_path.read_text());assert all(oldfreeze[k]==freeze[k] for k in ('base_sha','adapter_sha','protocol_sha'))
    else:freeze_path.write_text(json.dumps(freeze,indent=2))
    print('PRETEST_FREEZE',freeze,flush=True)
    audit=engine.algebra_audit();(args.output/'algebra_audit.json').write_text(json.dumps(audit,indent=2));all_results={}
    for name,spec in protocol['datasets'].items():
        if args.task and name!=args.task:continue
        data=load_mhealth(args.data/name,spec) if name=='mhealth319' else load_har(args.data/name,spec);pre=engine.prefix(data,spec,engine.BASE);overlap=participant_overlap(data);streams=[engine.precompute(engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,seed,engine.BASE),pre,engine.BASE) for seed in protocol['equal_selection_grid']['calibration_seeds']];cal=[];selected={};qfit={}
        for lam in (0.,.25,1.):
            for margin in (0.,4.,12.):
                cfg=dict(engine.BASE,lam=lam,threshold=margin)
                for mode in TUNE:
                    qi,scores=initialize_q(streams,pre,cfg,mode);rr=[run(s,pre,cfg,mode,qi,True) for s in streams];cal.append(dict(mode=mode,lam=lam,margin=margin,q_initial=qi,q_fit_scores=scores,mean_selection_net=float(np.mean([r['selection_net'] for r in rr])),mean_selection_actions=float(np.mean([r['deployments'] for r in rr]))))
                print('CAL',name,lam,margin,flush=True)
        for mode in TUNE:
            choice=max([r for r in cal if r['mode']==mode],key=lambda r:(r['mean_selection_net'],-r['margin'],-r['lam']));selected[mode]=dict(engine.BASE,lam=choice['lam'],threshold=choice['margin']);qfit[mode]=choice['q_initial']
        selected['posterior_unused']=dict(selected['joint']);qfit['posterior_unused']=qfit['joint'];sp=args.output/f'{name}_prefix_selection.json';sp.write_text(json.dumps(dict(protocol_sha=sha,adapter_sha=adapter_sha,base_sha=BASE_SHA,data_hashes=data['hashes'],split=pre['split'],participant_overlap=overlap,grid=cal,selected=selected,q_initial=qfit),indent=2));selection_sha=hashlib.sha256(sp.read_bytes()).hexdigest();print('SELECTION_FROZEN',name,selection_sha,{m:(c['lam'],c['threshold'],qfit[m]) for m,c in selected.items()},flush=True)
        trials=[]
        for seed in protocol['equal_selection_grid']['test_seeds']:
            stream=engine.precompute(engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,engine.BASE),pre,engine.BASE);rr={mode:run(stream,pre,selected.get(mode,engine.BASE),mode,qfit.get(mode,0)) for mode in MODES};trials.append(dict(seed=seed,windows=stream['windows'],episodes=len(stream['decisions']),baseline=dict(gross=stream['basegross'],fees=stream['commonfees'],drops=stream['commondrops'],net=stream['basegross']-stream['commonfees']),results=rr));print('TEST',name,seed,{m:(round(r['net'],2),r['deployments'],r['harmful']) for m,r in rr.items()},flush=True);(args.output/f'{name}_results_partial.json').write_text(json.dumps(dict(trials=trials),indent=2))
        summary={mode:dict(mean_net=float(np.mean([t['results'][mode]['net'] for t in trials])),mean_deployments=float(np.mean([t['results'][mode]['deployments'] for t in trials])),mean_harmful=float(np.mean([t['results'][mode]['harmful'] for t in trials])),mean_beneficial=float(np.mean([t['results'][mode]['beneficial'] for t in trials])),mean_accuracy=float(np.mean([t['results'][mode]['accuracy'] for t in trials])),mean_coverage=float(np.mean([t['results'][mode]['coverage'] for t in trials])),mean_informative_coverage=float(np.mean([t['results'][mode]['informative_coverage'] for t in trials if t['results'][mode]['informative_coverage'] is not None])),solver_failures=sum(t['results'][mode]['solver_failures'] for t in trials),max_kkt=max(t['results'][mode]['max_kkt'] for t in trials)) for mode in MODES}
        result=dict(dataset=name,protocol_sha=sha,base_code_sha=BASE_SHA,adapter_sha=adapter_sha,selection_sha=selection_sha,data_hashes=data['hashes'],raw_rows=data['raw_rows'],retained_rows=data['n'],participants=data['participants'],participant_overlap=overlap,split=pre['split'],quality=pre['q'].tolist(),selected=selected,q_initial=qfit,calibration=cal,trials=trials,summary=summary);(args.output/f'{name}_results.json').write_text(json.dumps(result,indent=2));all_results[name]=result;print('DATASET_COMPLETE',name,summary,flush=True)
    for name in protocol['datasets']:
        existing=args.output/f'{name}_results.json'
        if name not in all_results and existing.exists():all_results[name]=json.loads(existing.read_text())
    (args.output/'results.json').write_text(json.dumps(all_results,indent=2));(args.output/'summary.json').write_text(json.dumps({k:v['summary'] for k,v in all_results.items()},indent=2));print('COMPLETE',flush=True)

if __name__=='__main__':main()
