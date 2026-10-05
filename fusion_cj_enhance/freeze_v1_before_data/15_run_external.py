"""Frozen active-objective PDF/QMF comparison for the new HARTH collection.

Immutable author-loss tabular adapters; independent calibration selection uses
the identical paid budget and chronology. No held-out value changes a formula.
"""
from pathlib import Path
import argparse, datetime, gzip, hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
import numpy as np
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
OUT = ROOT/'external'
NEURAL = PROJECT/'work/fusion_strengthening_20261003/external/neural_external.py'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
p=module('cjr_external_harth_physical',ROOT/'run_physical.py')
nn=module('cjr_external_harth_neural',NEURAL)
RULES=('pdf','qmf');RATES=(.02,.08,.2);MARGINS=(0.,4.,12.)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,value):
    body=json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
    if str(path).endswith('.gz'):
        with gzip.open(path,'wt') as f:f.write(body)
    else:Path(path).write_text(body+'\n')
def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(Path(path).read_text())
def freeze():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    assert not (ROOT/'data/cache_metadata.json').exists() and not (ROOT/'raw/harth779.zip').exists()
    sources=[Path(__file__),ROOT/'run_physical.py',ROOT/'adapter.py',ROOT/'protocol.json',
        NEURAL,Path(nn.helper.__file__),Path(nn.helper.ext.__file__),
        PROJECT/'outputs/Fusion_Recovery_Repro/external_fusion_validation_20261002/run_external_pdf.py']
    dump(OUT/'protocol.json',dict(utc=now(),source_sha256=sha(__file__),
        input_sha256={str(f):sha(f) for f in sources},task='harth779',rules=RULES,
        split='numeric subject ranks1–8fit,9–12calibration,13–22test; no cross-person or gap leases',
        source_groups=p.ad.GROUPS,classes=12,architecture='two32-tanh physical-source encoders,33x12 class heads and33-coefficient confidence heads',
        learning_rates=RATES,paid_margins=MARGINS,configurations_per_rule=9,
        calibration_seeds=p.CAL_SEEDS,test_seeds=p.TEST_SEEDS,initial_steps=500,online_steps=20,
        active_objective_strength=dict(pdf=1.,qmf=.1),
        author_commits=dict(pdf='864867426cde076fb3d0529d4df6b440d6963451',qmf='fe6c4c6ef7cb23f0a89594ee413d485f1854268b'),
        selection='sum actual Fixed130/B130 guarded second-half calibration increment over chains; average3 shared delays; ties smaller margin then smaller learning rate',
        calibration='immutable external helper pools all first-half complete matured origins, including unready; higher90th percentile; online callbacks likewise use all matured issued origins, as in original helper. This differs from internal ready-only denominator and is disclosed.',
        execution='same current candidate/reference, data, masks, delays, model-work interruptions, lease fees and restoration; same130 reserve and130 permanent loss budget per chain',
        budgets=[0.,110.,130.,260.],
        disclosure='active published objectives with tabular encoders; original image/text architectures and benchmarks not reproduced; additional encoder work not charged as identical hardware cost',
        reporting='all selections/trials/actions/ties/losses retained; all internal/external selections must lock before test worlds',
        runtime=dict(python=sys.version,numpy=np.__version__)))
    nn.gradient_checks(OUT/'gradient_checks.json')
def verify():
    data=load(OUT/'protocol.json');assert data['source_sha256']==sha(__file__)
    for path,want in data['input_sha256'].items():assert sha(path)==want,path
    return data
def prepared():
    values=p.prepared();engine,_,_,cfg,_,_,_,_=values
    assert engine is nn.helper.engine
    nn.helper.CFG=cfg;return values
def guarded(ev,pre,cfg,result,guard,service,B):
    replay=guard.replay(result['rows'],len(ev),B,'gross_loss',{r['k']:130. for r in result['rows']})
    actual=service(ev,pre,replay['rows'],cfg);reference=service(ev,pre,[],cfg)
    prefix=guard.reconstruct_prefix_increment(ev,replay['rows'],cfg)
    acts=[r for r in replay['rows'] if r['action']];inc=actual['net']-reference['net']
    loss=sum(max(0.,-r['local_net']) for r in acts)
    err=max(abs(inc-sum(r['local_net'] for r in acts)),abs(inc-prefix['final']),abs(inc-replay['final_settled_increment']))
    assert err<1e-8 and loss<=B+1e-8 and prefix['minimum']>=-B-1e-8
    assert all(r['spent_loss']+r['reserved']<=B+1e-8 for r in replay['ledger'])
    return dict(increment=inc,net=actual['net'],loss=loss,minimum_prefix=prefix['minimum'],service_error=err,
        admissions=len(acts),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),
        zero=sum(r['local_net']==0 for r in acts),coverage=[sum(r['lower_covered'] for r in acts),len(acts)],
        optimistic_excess=sum(r['posterior_or_block_sd']*max(0.,r['standardized_score']-r['q_issued']) for r in acts),
        refused=sum(r['proposed_action'] and not r['action'] for r in replay['rows']),rows=replay['rows'],ledger=replay['ledger'])
def calibrate():
    verify();p.verify();assert not (OUT/'selection.json').exists()
    engine,_,guard,cfg0,service,_,data,pre=prepared()
    recordings=[r for r in pre['recording_streams'] if r['partition']=='calibration']
    worlds=[(r['recording'],r['person'],seed,engine.world(r['x'],r['y'],r['timestamp'],pre,seed,cfg0))
        for r in recordings for seed in p.CAL_SEEDS]
    grid=[];models={};trials=[];training=[]
    for rule in RULES:
        for rate in RATES:
            initial,work=nn.train([pre['train_x'][:,ids] for ids in pre['features'][:-1]],pre['train_y'],rule,rate,500,pre['classes'])
            models[f'{rule}:{rate}']=[{key:val.tolist() for key,val in m.items()} for m in initial]
            training.append(dict(rule=rule,lr=rate,**work))
            streams=[nn.helper.scalar_stream(nn.world(ev,pre,rule,rate,initial),pre) for _,_,_,ev in worlds]
            qi,scores=nn.helper.ext.qi(streams,pre,cfg0,rule+'_calibrated')
            for margin in MARGINS:
                cfg=dict(cfg0,threshold=margin)
                results=[nn.helper.ext.run(engine,s,pre,cfg,rule+'_calibrated',qi,True) for s in streams]
                guard_results=[guarded(ev,pre,cfg,result,guard,service,130.) for (_,_,_,ev),result in zip(worlds,results)]
                entry=dict(rule=rule,lr=rate,margin=margin,q0=qi,fit_scores=scores,
                    guarded_net=sum(r['increment'] for r in guard_results)/len(p.CAL_SEEDS))
                grid.append(entry);trials.append(dict(configuration=entry,
                    chains=[dict(recording=rec,person=person,seed=seed,result=result,guarded=g)
                        for (rec,person,seed,_),result,g in zip(worlds,results,guard_results)]))
            print('CAL_EXTERNAL',rule,rate,flush=True)
    selected={rule:max((g for g in grid if g['rule']==rule),key=lambda g:(g['guarded_net'],-g['margin'],-g['lr'])) for rule in RULES}
    dump(OUT/'calibration_trials.json.gz',trials)
    dump(OUT/'selection.json',dict(selected=selected,grid=grid,models=models,training=training,split=pre['split'],data_hashes=data['hashes']))
    dump(OUT/'selection_freeze.json',dict(utc=now(),selection_sha256=sha(OUT/'selection.json'),
        protocol_sha256=sha(OUT/'protocol.json'),cache_sha256=data['metadata']['cache_sha256'],
        algorithm_freeze_sha256=sha(ROOT/'algorithm_freeze.json'),cache_metadata_sha256=sha(ROOT/'data/cache_metadata.json')))
    print('SELECT_EXTERNAL',{r:(g['guarded_net'],g['lr'],g['margin'],g['q0']) for r,g in selected.items()},flush=True)
def test():
    protocol=verify();p.verify();own=load(OUT/'selection_freeze.json');selection=load(OUT/'selection.json')
    assert own['selection_sha256']==sha(OUT/'selection.json') and own['algorithm_freeze_sha256']==sha(ROOT/'algorithm_freeze.json')
    locked=load(ROOT/'all_selections_before_test.json')
    for path,want in locked['inputs'].items():assert sha(path)==want,path
    assert locked['inputs'][str((OUT/'selection.json').resolve())]==own['selection_sha256']
    engine,recovery,guard,cfg0,service,fork,data,pre=prepared()
    assert data['metadata']['cache_sha256']==own['cache_sha256']
    assert sha(ROOT/'data/cache_metadata.json')==own['cache_metadata_sha256']
    records=[r for r in pre['recording_streams'] if r['partition']=='test'];all_guarded=[];issued=[]
    for record in records:
        path=OUT/(record['recording'].replace('/','_')+'_results.json.gz');assert not path.exists()
        trials=[];record_guarded=[]
        for seed in p.TEST_SEEDS:
            ev=engine.world(record['x'],record['y'],record['timestamp'],pre,seed,cfg0)
            results={};checks={};gg=[]
            for rule in RULES:
                sel=selection['selected'][rule]
                initial=[{key:np.array(val) for key,val in m.items()} for m in selection['models'][f"{rule}:{sel['lr']}"]]
                stream=nn.helper.scalar_stream(nn.world(ev,pre,rule,sel['lr'],initial),pre)
                cfg=dict(cfg0,threshold=sel['margin'])
                result=nn.helper.ext.run(engine,stream,pre,cfg,rule+'_calibrated',sel['q0'])
                check=recovery.execute_check(ev,pre,cfg,result,service,fork);assert check['passed']
                results[rule]=result;checks[rule]=check
                issued.extend(dict(person=record['person'],recording=record['recording'],seed=seed,rule=rule,**r) for r in result['rows'])
                for B in protocol['budgets']:
                    g=dict(person=record['person'],recording=record['recording'],seed=seed,rule=rule,budget=B,
                        **guarded(ev,pre,cfg,result,guard,service,B));gg.append(g);all_guarded.append(g)
            record_guarded.extend(gg)
            trials.append(dict(recording=record['recording'],person=record['person'],seed=seed,results=results,checks=checks))
            print('TEST_EXTERNAL',record['recording'],seed,{r:sum(g['increment'] for g in gg if g['rule']==r and g['seed']==seed and g['budget']==130.) for r in RULES},flush=True)
        dump(path,dict(trials=trials,guarded=record_guarded))
    dump(OUT/'guarded_all.json.gz',all_guarded);dump(OUT/'issued_all.json.gz',issued)
    summary={}
    for rule in RULES:
        rs=[r for r in all_guarded if r['rule']==rule and r['budget']==130.]
        per_seed={s:sum(r['increment'] for r in rs if r['seed']==s) for s in p.TEST_SEEDS}
        summary[rule]=dict(mean_total_increment=float(np.mean(list(per_seed.values()))),seed_increments=per_seed,
            admissions=sum(r['admissions'] for r in rs),beneficial=sum(r['beneficial'] for r in rs),harmful=sum(r['harmful'] for r in rs),
            loss=sum(r['loss'] for r in rs),coverage=[sum(r['coverage'][0] for r in rs),sum(r['coverage'][1] for r in rs)],
            optimistic_excess=sum(r['optimistic_excess'] for r in rs),refused=sum(r['refused'] for r in rs),
            maximum_service_error=max(r['service_error'] for r in rs),
            per_person_mean={person:sum(r['increment'] for r in rs if r['person']==person)/len(p.TEST_SEEDS) for person in data['metadata']['test_participants']})
    dump(OUT/'summary.json',summary);print('EXTERNAL_SUMMARY',summary,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','calibrate','test']);args=ap.parse_args()
    {'freeze':freeze,'calibrate':calibrate,'test':test}[args.phase]()
