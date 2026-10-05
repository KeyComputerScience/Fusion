"""Frozen CJ-T development: guarded calibration selection, all outcomes kept."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
import numpy as np
import transport as api
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2];c=api.c
TASKS=('rss348','arem366','gashome362','localization196')
FLOORS=c.FLOORS;BANDWIDTHS=c.BANDWIDTHS;SDS=(.05,.1,.2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):c.dump(p,x)
def load(p):return c.load(p)
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
ad=module('cjr_known_localization_adapter',api.FROZEN/'physical/adapter.py')

def freeze():
    assert not (ROOT/'protocol.json').exists();c.verify()
    sources=[Path(__file__),ROOT/'transport.py',api.FROZEN/'run_conditional.py',api.FROZEN/'copula_control.py',
        c.PARENT,c.parent.FACTOR,c.parent.parent.MATRIX_SOURCE,c.parent.CORE,c.parent.ADAPTER,api.FROZEN/'physical/adapter.py']
    engine,recovery,guard,*_=api.binding()
    sources += [Path(m.__file__) for m in (engine,recovery,guard)]
    data=[PROJECT/'work/fusion_temporal_20261003/physical/data/rss348/cache_metadata.json',
        PROJECT/'work/fusion_temporal_20261003/physical/data/rss348/cached_dataset.npz',
        api.FROZEN/'physical/data/cache_metadata.json',api.FROZEN/'physical/data/cached_dataset.npz']
    data += [PROJECT/f'work/fusion_strengthening_20261003/fresh/data/{task}/{name}'
        for task in ('arem366','gashome362') for name in ('cache_metadata.json','cached_dataset.npz')]
    dump(ROOT/'protocol.json',dict(utc=now(),source_hashes={str(p):sha(p) for p in sources},data_hashes={str(p):sha(p) for p in data},
        tasks=TASKS,arms=api.ARMS,method='CJ-T paired forecast-complete-target transport',
        innovation='fit paired historical current-action H,U joint Gaussian mixture; raw observed pairs and weights unchanged; missing H only Schur completed; condition U on current H directly',
        constraints='same raw relevance weights and1e-12 cutoff, paired blocks, source models, costs, q callbacks and loss ledger',
        readiness='nonempty mature contrast-supported H,U pair history; U is observed in every historical block; no prior-only authorization',
        grids=dict(conditional=dict(bandwidth=BANDWIDTHS,q_floor=FLOORS,configurations=9),
                   legacy=dict(prior_sd=SDS,q_floor=FLOORS,configurations=9)),
        selection='first-half ready fully-matured higher90th percentile fits q; maximize ACTUAL second-half Fixed130/B130 guarded calibration increment; ties larger floor then larger bandwidth (legacy smaller prior_sd)',
        calibration_seeds=dict(rss348=[87001,87002,87003],arem366=[91001,91002,91003],gashome362=[91001,91002,91003],localization196=[112001,112002,112003]),
        development_seeds=dict(rss348=list(range(88001,88006)),arem366=list(range(92001,92006)),gashome362=list(range(92001,92006)),localization196=list(range(113001,113006))),
        evidence_scope='all four physical datasets and their earlier test partitions already inspected, development only; no new physical holdout read',
        execution=dict(reserve=130.,primary_budget=130.,budget_diagnostics=[0.,110.,130.,260.],localization_scope='per independently reset recording chain'),
        mandatory='all configurations/trials/actions/negative outcomes/coverage retained; no further variant search based on development wins',
        prior='Gaussian H=U1+E, mean0,varU1/3,independentE covariance prior_variance*qualityD; matches neutral bounded-target second moment but is not uniformU',
        kernel='H: b² prior_variance qualityD; U: b² norm_floor²; no extra tuned parameter',
        product='f_U(u) product_s f(H_s=h_s|U=u); preserves exact primitive source-target bivariate and target marginal of raw working law, then boundedU condition',
        gaussian='single full Gaussian with exact mean/covariance of same smoothed raw H,U law',
        factorized='fully factorized coordinate law preserves exact univariate marginals; target posterior equals common target marginal',
        diagonal_completion='same raw observations/weights, diagonal shared R before missingH Schur completion; target marginal preserved',
        limitations=['working joint H,U density; U restricted[-1,1], no fullHbox or physical sampling guarantee',
            'partial observations do not identify missing joint laws; Gaussian Schur completion is declared',
            'raw relevance, prior and kernel remain assumptions; calibration/return can fail under drift',
            'delay schedules reuse physical traces; allknownpartitions development only',
            'pathwise loss guarantee shared; profitability and superiority are not guaranteed'],
        frozen_legacy_RSS=dict(old_CJ=35.6,factorized=44.8,sandwich=44.4),runtime=dict(python=sys.version,numpy=np.__version__)))
def verify():
    p=load(ROOT/'protocol.json')
    for field in ('source_hashes','data_hashes'):
        for f,w in p[field].items():assert sha(f)==w,f
    return p

def prepared(task):
    engine,recovery,guard,source,cfg,study,service,fork=api.binding()
    cfg.update(prior_sd=.1,threshold=0.)
    if task=='localization196':
        cfg.update(lease_archive=48)
        data=ad.load_data(api.FROZEN/'physical/data');pre=ad.make_prefix(data,cfg,engine)
    else:data,pre=c.parent.task_data(task,cfg,engine,source,study)
    return engine,recovery,guard,cfg,service,fork,data,pre

def events_for(task,pre,engine,cfg,calibration):
    seeds=load(ROOT/'protocol.json')['calibration_seeds' if calibration else 'development_seeds'][task]
    if task=='localization196':
        rec=[x for x in pre['recording_streams'] if x['partition']==('calibration' if calibration else 'test')]
        return [(x['recording'],s,engine.world(x['x'],x['y'],x['timestamp'],pre,s,cfg)) for x in rec for s in seeds]
    prefix='cal' if calibration else 'test'
    return [(task,s,engine.world(pre[prefix+'_x'],pre[prefix+'_y'],pre[prefix+'_timestamp'],pre,s,cfg)) for s in seeds]

def configs(arm,cfg0):
    if arm.startswith('legacy_'):
        return [dict(api.configure(cfg0,arm),slice_bandwidth=1.,prior_sd=sd,q_floor=floor) for sd in SDS for floor in FLOORS]
    return [dict(api.configure(cfg0,arm),slice_bandwidth=b,q_floor=floor) for b in BANDWIDTHS for floor in FLOORS]
def stream_key(cfg):return (cfg['slice_bandwidth'],cfg['prior_sd'])
def stream_for(streams,index,ev,bases,pre,cfg):
    key=(index,)+stream_key(cfg)
    if key not in streams:streams[key]=api.core.augment(ev,pre,bases[index],cfg)
    return streams[key]

def guarded(guard,service,ev,pre,cfg,rows,B=130.):
    gg=guard.replay(rows,len(ev),B,'gross_loss',{x['k']:130. for x in rows})
    actual=service(ev,pre,gg['rows'],cfg);reference=service(ev,pre,[],cfg)
    prefix=guard.reconstruct_prefix_increment(ev,gg['rows'],cfg)
    acts=[x for x in gg['rows'] if x['action']];inc=actual['net']-reference['net']
    err=max(abs(inc-sum(x['local_net'] for x in acts)),abs(inc-prefix['final']),abs(inc-gg['final_settled_increment']))
    loss=sum(max(0.,-x['local_net']) for x in acts)
    assert err<1e-8 and loss<=B+1e-8 and prefix['minimum']>=-B-1e-8
    assert all(x['spent_loss']+x['reserved']<=B+1e-8 for x in gg['ledger'])
    return dict(increment=inc,net=actual['net'],loss=loss,minimum_prefix=prefix['minimum'],service_error=err,
        admissions=len(acts),beneficial=sum(x['local_net']>0 for x in acts),harmful=sum(x['local_net']<0 for x in acts),
        zero=sum(x['local_net']==0 for x in acts),coverage=[sum(x['lower_covered'] for x in acts),len(acts)],
        optimistic_excess=sum(x['posterior_or_block_sd']*max(x['standardized_score']-x['q_issued'],0.) for x in acts),
        refused=sum(x['proposed_action'] and not x['action'] for x in gg['rows']),rows=gg['rows'],ledger=gg['ledger'])

def calibrate(tasks):
    verify()
    for task in tasks:
        assert not (ROOT/(task+'_selection.json')).exists()
        engine,recovery,guard,cfg0,service,fork,data,pre=prepared(task)
        worlds=events_for(task,pre,engine,cfg0,True);bases=[engine.precompute(ev,pre,cfg0) for _,_,ev in worlds]
        streams={};trials=[];grid=[]
        for arm in api.ARMS:
            for cfg in configs(arm,cfg0):
                ss=[stream_for(streams,i,ev,bases,pre,cfg) for i,(_,_,ev) in enumerate(worlds)]
                qi,scores=api.core.initial_q(ss,pre,cfg,arm);rr=[api.core.run(engine,s,pre,cfg,arm,qi,True) for s in ss]
                assert all(r['solver_failures']==0 for r in rr)
                gr=[guarded(guard,service,ev,pre,cfg,r['rows']) for (_,_,ev),r in zip(worlds,rr)]
                entry=dict(arm=arm,bandwidth=cfg['slice_bandwidth'],prior_sd=cfg['prior_sd'],q_floor=cfg['q_floor'],q_initial=qi,
                    guarded_net=float(np.mean([g['increment'] for g in gr])),unguarded_net=float(np.mean([r['selection_net'] for r in rr])),
                    fit_scores=scores,guarded_beneficial=sum(g['beneficial'] for g in gr),guarded_harmful=sum(g['harmful'] for g in gr))
                grid.append(entry);trials.append(dict(configuration=entry,worlds=[dict(chain=n,seed=s) for n,s,_ in worlds],results=rr,guarded=gr))
            print('CAL',task,arm,flush=True)
        selected={a:max((x for x in grid if x['arm']==a),key=lambda x:(x['guarded_net'],x['q_floor'],
            -x['prior_sd'] if a.startswith('legacy_') else x['bandwidth'])) for a in api.ARMS}
        dump(ROOT/(task+'_calibration_trials.json.gz'),trials)
        dump(ROOT/(task+'_selection.json'),dict(selected=selected,grid=grid,data_hashes=data.get('hashes',{}),scope='development calibration only'))
        print('SELECT',task,{a:(x['guarded_net'],x['bandwidth'],x['q_floor'],x['q_initial']) for a,x in selected.items()},flush=True)
    for task in tasks:
        dump(ROOT/(task+'_selection_lock.json'),dict(utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),selection_sha256=sha(ROOT/(task+'_selection.json')),trials_sha256=sha(ROOT/(task+'_calibration_trials.json.gz'))))

def summarize(records):
    result={}
    for arm in api.ARMS:
        rr=[x for x in records if x['arm']==arm and x['budget']==130.]
        byseed={s:sum(x['increment'] for x in rr if x['seed']==s) for s in sorted({x['seed'] for x in rr})}
        result[arm]=dict(mean_total_increment=float(np.mean(list(byseed.values()))),seed_increments=byseed,
            admissions=sum(x['admissions'] for x in rr),beneficial=sum(x['beneficial'] for x in rr),harmful=sum(x['harmful'] for x in rr),
            zero=sum(x['zero'] for x in rr),loss=sum(x['loss'] for x in rr),coverage=[sum(x['coverage'][0] for x in rr),sum(x['coverage'][1] for x in rr)],
            optimistic_excess=sum(x['optimistic_excess'] for x in rr),refused=sum(x['refused'] for x in rr))
    return result

def test(tasks):
    verify()
    for task in tasks:
        locked=load(ROOT/(task+'_selection_lock.json'))
        assert locked['protocol_sha256']==sha(ROOT/'protocol.json')
        assert sha(ROOT/(task+'_selection.json'))==locked['selection_sha256']
        assert sha(ROOT/(task+'_calibration_trials.json.gz'))==locked['trials_sha256']
        assert not (ROOT/(task+'_development_results.json.gz')).exists()
        engine,recovery,guard,cfg0,service,fork,data,pre=prepared(task)
        worlds=events_for(task,pre,engine,cfg0,False);selected=load(ROOT/(task+'_selection.json'))['selected']
        trials=[];guarded_records=[];fixed=[];poisons=0;maxquad=0.
        for chain,seed,ev in worlds:
            base=engine.precompute(ev,pre,cfg0);streams={};results={};checks={}
            for arm in api.ARMS:
                x=selected[arm];cfg=dict(api.configure(cfg0,arm),slice_bandwidth=x['bandwidth'],prior_sd=x['prior_sd'],q_floor=x['q_floor'])
                sk=stream_key(cfg)
                if sk not in streams:streams[sk]=api.core.augment(ev,pre,base,cfg)
                stream=streams[sk];r=api.core.run(engine,stream,pre,cfg,arm,x['q_initial'])
                assert r['solver_failures']==0
                checks[arm]=recovery.execute_check(ev,pre,cfg,r,service,fork);assert checks[arm]['passed'];results[arm]=r
                for d,row in zip(stream['decisions'],r['rows']):
                    a=api.forecast(d,pre,cfg,arm,row['q_issued']);b=api.forecast(dict(d,truegross=1e50,localnet=-1e50),pre,cfg,arm,row['q_issued'])
                    assert a[2:4]==b[2:4];poisons+=1
                    assert all(x['maturity']<=d['k'] for x in d['temporal']['eligible'])
                    maxquad=max(maxquad,row.get('quadrature_error',0.))
                for B in (0.,110.,130.,260.):
                    gg=guarded(guard,service,ev,pre,cfg,r['rows'],B)
                    guarded_records.append(dict(task=task,chain=chain,seed=seed,arm=arm,budget=B,**gg))
            x=selected['t_joint'];cfg=dict(api.configure(cfg0,'t_joint'),slice_bandwidth=x['bandwidth'],prior_sd=x['prior_sd'],q_floor=x['q_floor'])
            stream=streams[stream_key(cfg)]
            for d,row in zip(stream['decisions'],results['t_joint']['rows']):
                predictions={}
                for arm in api.NEW_ARMS:
                    _,_,F,S,_,extra=api.forecast(d,pre,cfg,arm,row['q_issued'])
                    predictions[arm]=dict(F=F,S=S,score=F-row['q_issued']*S-5,action=bool(F-row['q_issued']*S-5>0))
                fixed.append(dict(chain=chain,seed=seed,k=d['k'],D=d['localnet'],q=row['q_issued'],
                    transport_inputs=dict(h=d['h'].tolist(),alpha=d['temporal']['transport_law']['alpha'].tolist(),locations=d['temporal']['transport_law']['locations'].tolist(),covariances=d['temporal']['transport_law']['covariances'].tolist(),mean=d['temporal']['transport_law']['mean'].tolist(),C=d['temporal']['transport_law']['C'].tolist(),N=d['N'],ready=d['temporal']['transport_ready'],norm_floor=cfg['norm_floor'],maturity=row['maturity']),predictions=predictions))
            trials.append(dict(chain=chain,seed=seed,reference=service(ev,pre,[],cfg0),results=results,checks=checks))
            print('DEV',task,chain,seed,{a:round(results[a]['net']-trials[-1]['reference']['net'],2) for a in api.ARMS},flush=True)
        summary=summarize(guarded_records)
        if task=='localization196':
            persons={person:summarize([x for x in guarded_records if x['chain'].startswith(person)]) for person in 'DE'}
        else:persons={}
        comparison={}
        for arm in api.NEW_ARMS[1:]:
            changed=[x for x in fixed if x['predictions']['t_joint']['action']!=x['predictions'][arm]['action']]
            comparison[arm]=dict(changed=len(changed),joint_only=sum(x['predictions']['t_joint']['action'] for x in changed),
                utility_difference=sum((int(x['predictions']['t_joint']['action'])-int(x['predictions'][arm]['action']))*x['D'] for x in changed))
        report=dict(task=task,selected=selected,arms=summary,persons=persons,fixed_state=comparison,
            max_service_error=max(x['service_error'] for x in guarded_records),current_truth_poison_checks=poisons,max_quadrature_refinement=maxquad,
            development_only=True,scope='delay seeds reuse the same physical recordings; UCI totals sum10 separately reset chains per seed')
        dump(ROOT/(task+'_development_results.json.gz'),dict(trials=trials,guarded=guarded_records,fixed=fixed))
        dump(ROOT/(task+'_development_summary.json'),report)
        print('SUMMARY',task,{a:(x['mean_total_increment'],x['beneficial'],x['harmful']) for a,x in summary.items()},flush=True)
    verify()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','calibrate','test']);ap.add_argument('--tasks',nargs='+',default=TASKS);args=ap.parse_args()
    {'freeze':freeze,'calibrate':lambda:calibrate(tuple(args.tasks)),'test':lambda:test(tuple(args.tasks))}[args.phase]()
