"""Separately frozen, later DBF-2025 rule extension on all selfBACK people.

The primary nine-pipeline freeze remains unchanged. This extension is designed
after acquisition and primary calibration; its own 18 trials are locked before
DBF calibration/test, and it is never a new independent physical validation.
"""
from pathlib import Path
import argparse, datetime, importlib.util, sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
p=module('selfback_primary_for_dbf',ROOT/'run_selfback_v2.py')
api=p.api;np=p.np
OP_PATH=p.PROJECT/'work/fusion_coupling_focus_20261005/dbf/dbf_operator.py'
op=module('selfback_published_dbf_operator',OP_PATH)
MULTIPLIERS=(.25,1.,4.)
MARGINS=(0.,2.,4.,8.,12.,16.)
OUT=ROOT/'selfback_dbf'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def configurations():return [dict(kappa_multiplier=k,margin=m) for k in MULTIPLIERS for m in MARGINS]
def freeze():
    p.verify();OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    sources={Path(path) for path in p.load('selfback_freeze_repair2.json')['algorithm_source_hashes']}|{Path(__file__),OP_PATH}
    for obj in list(sys.modules.values()):
        file=getattr(obj,'__file__',None)
        if file and str(file).startswith(str(p.PROJECT)) and str(file).endswith('.py'):sources.add(Path(file))
    protocol=dict(utc=now(),status='Later published-rule benchmark extension, frozen after raw acquisition and primary calibration but before any DBF calibration/test output',
        primary_freeze_sha256=api.sha(ROOT/'selfback_freeze_repair2.json'),primary_selection_sha256=api.sha(ROOT/'selfback/selection.json'),
        source_hashes={str(path):api.sha(path) for path in sorted(sources)},data_hashes={str(path):api.sha(path) for path in (ROOT/'selfback_data/cached_dataset.npz',ROOT/'selfback_data/cache_metadata.json')},
        published_rule='DBF2025 conflict discount: get_doc_belief_fusion lambda1 epsilon1e-6 includes self comparison',
        paper='https://proceedings.mlr.press/v258/bezirganyan25a.html',author_code='https://github.com/bezirganyan/DBF_uncertainty/blob/main/model.py',
        adapter='Common unchanged issued linear source forecasts become evidence e_s=kappa*(prefix_quality_s/mean_all_prefix_quality)*p_s; original evidential backbone/loss not reproduced',
        scope='Published fusion rule and explicit quality-based evidence adaptation under same calibrated paid service; not complete original-paper architecture reproduction',
        configurations=configurations(),trials=18,calibration_seeds=p.CAL,test_seeds=p.TEST,participants=p.load('selfback_data/cache_metadata.json')['test_ids'],
        initial_quantile='same all-issued mature firsthalf scalar external quantile as primary PDF/QMF',selection='maximum guarded secondhalf paid return summed across all8calparticipants/3delays, ties smallermargin then smallerkappa',
        service='same candidate/reference, feedback, masks, history, Fixed130 reserve/B130 perparticipant and all B0/110/130/260 diagnostics',
        outcomes='all13people×5delays retained; actual admissions, coverage/excess, harms/losses, guardbinding, requestservice/fork checks; no new testbased method changes')
    api.dump(OUT/'protocol.json',protocol);api.dump(OUT/'operator_checks.json',op.audit())
    api.dump(OUT/'pre_calibration_freeze.json',dict(utc=now(),protocol_sha256=api.sha(OUT/'protocol.json'),selection_exists=False,test_exists=False))
    print('FROZEN_DBFSelFBACK',api.sha(OUT/'protocol.json'),len(sources),flush=True)
def verify():
    p.verify();protocol=api.load(OUT/'protocol.json');assert api.sha(OUT/'protocol.json')==api.load(OUT/'pre_calibration_freeze.json')['protocol_sha256']
    for key in ('source_hashes','data_hashes'):
        for path,want in protocol[key].items():assert api.sha(path)==want,path
    return protocol
def stream(events,pre,multiplier):
    worlds=[];quality=pre['q']/np.mean(pre['q']);kappa=pre['classes']*multiplier
    for e in events:
        ids=np.flatnonzero(e['mask']);prob,state=op.fuse_evidence(op.evidence(e['p'][ids],quality[ids],kappa),'dbf')
        worlds.append(dict(e,external_probability=prob,external_weights=state['discount'].T))
    s=p.nn.helper.scalar_stream(worlds,pre)
    for d in s['decisions']:d['external']['information_ready']=bool(d['external']['eligible'])
    return s
def calibrate():
    verify();assert not (OUT/'selection.json').exists();engine,guard,cfg0,service,fork,pre,meta,cache,records=p.worlds('calibration');trials=[]
    for multiplier in MULTIPLIERS:
        ss=[stream(e,pre,multiplier) for _,_,e,_ in records]
        q0,values=p.nn.helper.ext.qi(ss,pre,cfg0,'dbf_calibrated')
        for margin in MARGINS:
            cfg=dict(cfg0,threshold=margin);rr=[p.nn.helper.ext.run(engine,s,pre,cfg,'dbf_calibrated',q0,True) for s in ss]
            gg=p.guarded(records,rr,pre,cfg,guard,service,'dbf')
            entry=dict(arm='dbf',config=dict(kappa_multiplier=multiplier,margin=margin),q0=q0,fit_count=len(values),net=sum(g['increment'] for g in gg)/3)
            trials.append(dict(configuration=entry,guarded=gg));print('CAL_DBFSelFBACK',entry,flush=True)
    selected=max((x['configuration'] for x in trials),key=lambda x:(x['net'],-x['config']['margin'],-x['config']['kappa_multiplier']))
    api.dump(OUT/'calibration_trials.json.gz',trials);api.dump(OUT/'selection.json',dict(selected=selected))
    api.dump(OUT/'selection_lock.json',dict(utc=now(),protocol_sha256=api.sha(OUT/'protocol.json'),selection_sha256=api.sha(OUT/'selection.json'),test_exists=False,cache=cache.report()))
    print('SELECT_DBFSelFBACK',selected,flush=True)
def test():
    verify();assert not (OUT/'results.json.gz').exists();assert api.sha(OUT/'selection.json')==api.load(OUT/'selection_lock.json')['selection_sha256']
    selected=api.load(OUT/'selection.json')['selected'];engine,guard,cfg0,service,fork,pre,meta,cache,records=p.worlds('test');issued=[];gs=[];bs=[];checks=[]
    conf=selected['config'];cfg=dict(cfg0,threshold=conf['margin'])
    for rec,seed,e,_ in records:
        s=stream(e,pre,conf['kappa_multiplier']);r=p.nn.helper.ext.run(engine,s,pre,cfg,'dbf_calibrated',selected['q0']);identity=dict(arm='dbf',recording=rec['recording'],person=rec['person'],batch=rec['batch'],seed=seed)
        g=api.db.guarded(e,pre,cfg,r,guard,service,130.);issued.append(dict(**identity,**r));gs.append(dict(**identity,**g));checks.append(dict(**identity,**p.base.target_checks(e,r['rows'],cfg,fork)))
        for B in p.BUDGETS:bs.append(dict(**identity,budget=B,**(g if B==130 else api.db.guarded(e,pre,cfg,r,guard,service,B))))
        print('TEST_DBFSelFBACK',rec['recording'],seed,flush=True)
    summary=p.v.summary(gs,p.TEST,('dbf',));units={str(person):p.v.summary([g for g in gs if g['person']==person],p.TEST,('dbf',)) for person in meta['test_ids']}
    api.dump(OUT/'results.json.gz',dict(issued=issued,guarded=gs,budget_rows=bs,selected=selected));api.dump(OUT/'summary.json',dict(summary=summary,units=units,selected=selected,budget_summary={str(B):p.v.summary([g for g in bs if g['budget']==B],p.TEST,('dbf',)) for B in p.BUDGETS},metadata=meta,policy_trajectories=len(gs),budget_paths=len(bs),forks=sum(c['forks'] for c in checks),target_checks=checks,max_service_error=max(g['service_error'] for g in bs),cache=cache.report()))
    print('SUMMARY_DBFSelFBACK',summary,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','verify','calibrate','test']);args=parser.parse_args();globals()[args.phase]()
