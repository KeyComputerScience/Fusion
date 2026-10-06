"""Prospectively frozen selfBACK521 participant validation, all outcomes retained."""
from pathlib import Path
import argparse,datetime,importlib.util,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT/'work/fusion_joint_return_20261005/ratio'))
import prospective_v2 as v
api=v.api;np=v.np;base=v.r.base;rc=v.r.rc
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
adapter=module('selfback_frozen_adapter',ROOT/'selfback_adapter_v2.py')
NEURAL=PROJECT/'work/fusion_strengthening_20261003/external/neural_external.py'
nn=module('selfback_active_objectives',NEURAL)
NATIVE=('legacy_factorized','legacy_sandwich');RATIO=v.ARMS;LAWS=('kde_aligned',)
EXTERNAL=('pdf','qmf');ARMS=RATIO+NATIVE+LAWS+EXTERNAL
CAL=(151001,151002,151003);TEST=(152001,152002,152003,152004,152005)
BUDGETS=(0.,110.,130.,260.);RATES=(.02,.08,.2);MARGINS=(0.,2.,4.,8.,12.,16.);NATIVE_FLOORS=(0.,.32,.64,.96,1.2815515655446004,1.64);KDE_PERP=(.125,.25,.5,1.,1.5,2.)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,x):api.dump(ROOT/name,x)
def load(name):return api.load(ROOT/name)
def law_configs():return [dict(b_parallel=bp,b_perp=bt,cap=1.) for bp in rc.BANDS for bt in KDE_PERP]
def native_configs():return [dict(prior_sd=sd,q_floor=f,slice_bandwidth=1.) for sd in base.PRIORS for f in NATIVE_FLOORS]
def freeze():
    assert not (ROOT/'selfback_freeze_repair2.json').exists() and not (ROOT/'raw/selfback521.zip').exists()
    v.r.verify();base.verify();rc.physical.verify();nn.gradient_checks(ROOT/'selfback_gradient_checks.json')
    extra={Path(__file__),ROOT/'selfback_adapter_v2.py',ROOT/'selfback_protocol_proposal.json',NEURAL}
    upstream=api.load(PROJECT/'work/fusion_revision_20261005/new_physical/upstream_source_closure_audit.json')['union_source_hashes']
    for path,want in upstream.items():assert api.sha(path)==want,path
    sources=set(v.scientific_sources())|set(api.p.scientific_sources())|{Path(path) for path in upstream}|extra
    for obj in list(sys.modules.values()):
        file=getattr(obj,'__file__',None)
        if file and str(file).startswith(str(PROJECT)) and str(file).endswith('.py'):sources.add(Path(file))
    p=dict(version=2,utc=now(),dataset='selfback521',raw_access_authorized=True,
        status='complete source and protocol freeze before any selfBACK raw transfer or signal/class observation',
        algorithm_source_hashes={str(x):api.sha(x) for x in sorted(sources)},metadata_proposal_sha256=api.sha(ROOT/'selfback_protocol_proposal.json'),
        raw_archive='https://archive.ics.uci.edu/static/public/521/selfback.zip',raw_path=str(ROOT/'raw/selfback521.zip'),
        prior_inventory_sha256=api.sha(ROOT/'inventory.json'),arms=ARMS,participant_ranks=dict(fit=(1,12),calibration=(13,20),test=(21,33)),
        preprocessing='original wrist/thigh calendar-second bins,>=50samples/panel,12moments/panel,9nativeactivitylabels,all33people fixed numeric rank,participant/gap causal lag resets,fit-only scaling and quality',
        grids=dict(ratio=v.configurations(),native=native_configs(),kde=law_configs(),external=[dict(lr=r,margin=m) for r in RATES for m in MARGINS]),
        calibration_seeds=CAL,test_seeds=TEST,model_work='unchanged shared500prefix initialsoftmax steps and20/4window online steps;linearcandidate/reference common;nonlinearexternal32tanh objectives separately trained',
        shared_native_anchor='strongest factorized controller selected from9nativeconfigs on actualguarded secondhalfcalreturn, same3caldelays; allnativeq callbacks/allissued riskfeedback unchanged and maturityproper',
        ratio_selection='9additionalconfigs perarm; maximize actualguarded secondhalfpaidutility /3caldelays; ties largestzeta then largestbandwidth uniformlyallarms',
        native_selection='18distinctconfigs perarm, guardedpaidsecondhalfreturn; ties largerqfloor then smallerpriorsd',
        kde_selection='18distinctconfigs, samejointpairedhistory andsameanchorselectionprior; ownready-only lower-targetalphacalibration; ties larger bperp then bparallel',
        external_selection='18distinctactiveobjectiveconfigs perrule, rate xpaidmargin; actualguarded secondhalfpaidutility; ties smallermargin then smallerrate; alltrials retained',
        budgets=BUDGETS,primary_budget=130.,reserve=130.,service=dict(window=32,horizon=4,install=2,restore=1,extra_interruptions=2,conservative_cost=5),
        tuning_scope='ratio shared9trialnativefactorfirststage (prior .05,.1,.2 xqfloor0,.64,1.28155) plus9owntrials equals18completepipelineconfigevaluations; eachstandalonecontrol18distinctownconfigs; all3caldelays andsamepaidselection objective. Thecommonanchor9subsetisselectedindependentlyfromthe18confignativecomparatorwinner.',
        calibration='ratio anchor q is a predictable input but doesnot certify ratio coverage; actual admittedcoverage/excess recorded; KDE firsthalf mature readyalpha fits<=.1 violation andready-only callbacks; externals ownall-issuedq',
        external_scope='PDF2024/QMF2023 unchanged activepublished objectives with nonlinear tabularsensor adapters, not original image/text architecture orauthor datasets; all source CE/fusedCE/confidence/ranking objectivesactive',
        safeguards='all13withheldparticipants×5delaysretained; nofuturelabelsatissuance; separateparticipantrestarthistory/model/budget; nooutcome-selectedreplacement; no test-basedmethodrevision; allties/lossescalibrationfailuresretained',
        outputs='actualparticipantandseednetpairedvscompletepipelines,fouractionterms,issued/ready/informative/admittedcoverage,excessmax/sum,negativeDandloss,B0/110/130/260 paths,requestserviceandforkaudit',
        runtime=dict(python=sys.version,numpy=np.__version__))
    dump('selfback_freeze.json',p);dump('selfback_pre_acquisition.json',dict(utc=now(),freeze_sha256=api.sha(ROOT/'selfback_freeze_repair2.json'),raw_exists=False,data_exists=False))
    print('FROZEN_SELFBACK',api.sha(ROOT/'selfback_freeze_repair2.json'),len(sources),flush=True)
def verify():
    p=load('selfback_execution_repair3.json');assert api.sha(ROOT/'selfback_execution_repair3.json')==load('selfback_execution_repair3_lock.json')['freeze_sha256']
    for path,want in p['algorithm_source_hashes'].items():assert api.sha(path)==want,path
    return p
def process():verify();print(adapter.process(ROOT/'raw/selfback521.zip',ROOT/'selfback_data',ROOT/'selfback_freeze_repair2.json'),flush=True)
def prepare():
    verify();engine,recovery,guard,source,cfg,study,service,fork=api.p.r.binding();cfg=api.p.base_cfg(cfg)
    cache=api.db.ExactGradientCache(engine);engine.gradient=cache
    pre,meta=adapter.prepared(ROOT/'selfback_data',engine,cfg);api.db.helper.CFG=cfg;nn.helper.CFG=cfg
    for rec in pre['recording_streams']:rec['timestamp']=[f"{rec['person']}:{stamp}" for stamp in rec['timestamp_ns']]
    return engine,guard,cfg,service,fork,pre,meta,cache
def worlds(part):
    engine,guard,cfg,service,fork,pre,meta,cache=prepare();records=[]
    for rec in pre['recording_streams']:
        if rec['partition']!=part:continue
        for seed in CAL if part=='calibration' else TEST:
            e=engine.world(rec['x'],rec['y'],rec['timestamp'],pre,seed,cfg);records.append((rec,seed,e,engine.precompute(e,pre,cfg)))
    return engine,guard,cfg,service,fork,pre,meta,cache,records
def make_streams(records,pre,cfg):
    out=[]
    for rec,seed,e,b in records:
        with base.bindings(api.c.parent.cumulant_state):out.append(api.core.augment(e,pre,b,cfg))
    return out
def guarded(records,results,pre,cfg,guard,service,arm,B=130.):
    return [dict(arm=arm,recording=rec['recording'],person=rec['person'],batch=rec['batch'],seed=seed,**api.db.guarded(e,pre,cfg,r,guard,service,B)) for (rec,seed,e,_),r in zip(records,results)]
def fit_alpha(streams,arm):
    eligible=[d for s in streams for d in s['decisions'] if d['maturity']<=s['windows']//2 and d['temporal']['eligible']]
    lo,hi=1e-4,1.
    if not eligible:return lo,0
    for _ in range(35):
        mid=(lo+hi)/2
        if np.mean([d['truegross']<d['N']*d['tail_laws'][arm].lower_tail(mid) for d in eligible])<=.1:lo=mid
        else:hi=mid
    return lo,len(eligible)
def decorate_kde(streams,pre,cfg):
    for s in streams:
        for d in s['decisions']:d['tail_laws']={'kde_aligned':rc.make_law(d,pre,cfg,'kde_aligned',api)}
def calibrate():
    verify();out=ROOT/'selfback';out.mkdir(exist_ok=True);assert not (out/'selection.json').exists()
    engine,guard,cfg0,service,fork,pre,meta,cache,records=worlds('calibration');grid=[];trials=[];selected={};streams_by_prior={}
    for conf in native_configs():
        cfg=dict(cfg0,**conf);sd=conf['prior_sd']
        if sd not in streams_by_prior:streams_by_prior[sd]=make_streams(records,pre,cfg)
        streams=streams_by_prior[sd]
        for arm in NATIVE:
            with base.bindings():q0,values=api.core.initial_q(streams,pre,cfg,arm)
            with base.bindings():rr=[api.core.run(engine,s,pre,cfg,arm,q0,True) for s in streams]
            gg=guarded(records,rr,pre,cfg,guard,service,arm)
            entry=dict(arm=arm,config=conf,q0=q0,fit_count=len(values),net=sum(g['increment'] for g in gg)/3)
            grid.append(entry);trials.append(dict(configuration=entry,guarded=gg));print('CAL_NATIVE',arm,conf,entry['net'],flush=True)
    for arm in NATIVE:selected[arm]=max((x for x in grid if x['arm']==arm),key=lambda x:(x['net'],x['config']['q_floor'],-x['config']['prior_sd']))
    anchor=max((x for x in grid if x['arm']=='legacy_factorized' and x['config']['q_floor'] in base.FLOORS),key=lambda x:(x['net'],x['config']['q_floor'],-x['config']['prior_sd']));cfg=dict(cfg0,**anchor['config']);streams=streams_by_prior[anchor['config']['prior_sd']]
    with base.bindings():natives=[api.core.run(engine,s,pre,cfg,'legacy_factorized',anchor['q0'],True) for s in streams]
    for s,n in zip(streams,natives):v.decorate_ratios(v.attach_native(s,n),pre,cfg)
    ratio_grid=[]
    for arm in RATIO:
        for conf in v.configurations():
            rr=[v.issue(s,arm,conf,True) for s in streams];gg=guarded(records,rr,pre,cfg,guard,service,arm)
            entry=dict(arm=arm,config=conf,net=sum(g['increment'] for g in gg)/3);grid.append(entry);ratio_grid.append(entry);trials.append(dict(configuration=entry,guarded=gg));print('CAL_RATIO',arm,conf,entry['net'],flush=True)
    selected.update(v.select(ratio_grid))
    for conf in law_configs():
        cc=dict(cfg,**conf);decorate_kde(streams,pre,cc);alpha,count=fit_alpha(streams,'kde_aligned')
        rr=[api.run(s,'kde_aligned',1.,alpha,True) for s in streams];gg=guarded(records,rr,pre,cc,guard,service,'kde_aligned')
        entry=dict(arm='kde_aligned',config=conf,alpha_initial=alpha,fit_count=count,net=sum(g['increment'] for g in gg)/3);grid.append(entry);trials.append(dict(configuration=entry,guarded=gg));print('CAL_KDE',conf,entry['net'],flush=True)
    selected['kde_aligned']=max((x for x in grid if x['arm']=='kde_aligned'),key=lambda x:(x['net'],x['config']['b_perp'],x['config']['b_parallel']))
    models={};training=[]
    for rule in EXTERNAL:
        for rate in RATES:
            initial,work=nn.train([pre['train_x'][:,ids] for ids in pre['features'][:-1]],pre['train_y'],rule,rate,500,pre['classes'])
            models[f'{rule}:{rate}']=[{k:v0.tolist() for k,v0 in m.items()} for m in initial];training.append(dict(rule=rule,lr=rate,**work))
            ss=[nn.helper.scalar_stream(nn.world(e,pre,rule,rate,initial),pre) for _,_,e,_ in records]
            q0,values=nn.helper.ext.qi(ss,pre,cfg0,rule+'_calibrated')
            for margin in MARGINS:
                cc=dict(cfg0,threshold=margin);rr=[nn.helper.ext.run(engine,s,pre,cc,rule+'_calibrated',q0,True) for s in ss];gg=guarded(records,rr,pre,cc,guard,service,rule)
                entry=dict(arm=rule,config=dict(lr=rate,margin=margin),q0=q0,fit_count=len(values),net=sum(g['increment'] for g in gg)/3);grid.append(entry);trials.append(dict(configuration=entry,guarded=gg));print('CAL_EXTERNAL',rule,rate,margin,entry['net'],flush=True)
        selected[rule]=max((x for x in grid if x['arm']==rule),key=lambda x:(x['net'],-x['config']['margin'],-x['config']['lr']))
    dump('selfback/calibration_trials.json.gz',trials);dump('selfback/selection.json',dict(selected=selected,anchor=anchor,grid=grid,models=models,training=training,split=pre['split'],quality=pre['q'],metadata=meta))
    dump('selfback/selection_lock.json',dict(utc=now(),freeze_sha256=api.sha(ROOT/'selfback_freeze_repair2.json'),selection_sha256=api.sha(out/'selection.json'),test_exists=False,cache=cache.report()))
    print('SELECT_SELFBACK',{k:(e['config'],e['net']) for k,e in selected.items()},flush=True)
def target_checks(events,rows,cfg,fork,arm):
    # Ratio score already contains its lower-tail penalty. The legacy fork
    # expects an unpenalized gain/q pair: use a canonical audit-only q=0
    # view while preserving native q_issued in every saved issued row.
    canonical=[dict(row,q_issued=0.) if arm in RATIO else row for row in rows]
    result=base.target_checks(events,canonical,cfg,fork)
    return dict(result,ratio_canonical_views=len(rows) if arm in RATIO else 0,issued_rows_changed=False)
def test():
    verify();out=ROOT/'selfback';assert not (out/'results.json.gz').exists();assert api.sha(out/'selection.json')==load('selfback/selection_lock.json')['selection_sha256']
    selection=load('selfback/selection.json');selected=selection['selected'];engine,guard,cfg0,service,fork,pre,meta,cache,records=worlds('test')
    issued=[];gs=[];bs=[];checks=[];anchor=selection['anchor']
    for rec,seed,e,b in records:
        cfg=dict(cfg0,**anchor['config']);stream=make_streams([(rec,seed,e,b)],pre,cfg)[0]
        with base.bindings():native=api.core.run(engine,stream,pre,cfg,'legacy_factorized',anchor['q0'])
        v.decorate_ratios(v.attach_native(stream,native),pre,cfg)
        for arm in ARMS:
            entry=selected[arm]
            if arm in RATIO:r=v.issue(stream,arm,entry['config']);cc=cfg
            elif arm in NATIVE:
                cc=dict(cfg0,**entry['config']);ss=stream if cc['prior_sd']==cfg['prior_sd'] else make_streams([(rec,seed,e,b)],pre,cc)[0]
                with base.bindings():r=api.core.run(engine,ss,pre,cc,arm,entry['q0'])
            elif arm=='kde_aligned':
                cc=dict(cfg,**entry['config']);decorate_kde([stream],pre,cc);r=api.run(stream,arm,1.,entry['alpha_initial'])
            else:
                conf=entry['config'];initial=[{k:np.asarray(value) for k,value in m.items()} for m in selection['models'][f"{arm}:{conf['lr']}"]]
                cc=dict(cfg0,threshold=conf['margin']);ss=nn.helper.scalar_stream(nn.world(e,pre,arm,conf['lr'],initial),pre);r=nn.helper.ext.run(engine,ss,pre,cc,arm+'_calibrated',entry['q0'])
            identity=dict(arm=arm,recording=rec['recording'],person=rec['person'],batch=rec['batch'],seed=seed);g=api.db.guarded(e,pre,cc,r,guard,service,130.)
            issued.append(dict(**identity,**r));gs.append(dict(**identity,**g));checks.append(dict(**identity,**target_checks(e,r['rows'],cc,fork,arm)))
            for B in BUDGETS:bs.append(dict(**identity,budget=B,**(g if B==130. else api.db.guarded(e,pre,cc,r,guard,service,B))))
        print('TEST_SELFBACK',rec['recording'],seed,flush=True)
    summaries=v.summary(gs,TEST,ARMS);unit={str(p):v.summary([g for g in gs if g['person']==p],TEST,ARMS) for p in meta['test_ids']}
    # Exact action decomposition uses the common potential full-lease returns.
    by={(g['arm'],g['recording'],g['seed']):g for g in gs};decomposition={}
    for arm in ARMS[1:]:
        terms={k:dict(count=0,value=0.) for k in ('added_gain','avoided_loss','missed_gain','incurred_loss')};changes=[]
        for j in [g for g in gs if g['arm']=='paired']:
            c=by[arm,j['recording'],j['seed']]
            for a0,b0 in zip(j['rows'],c['rows']):
                assert a0['k']==b0['k'] and a0['local_net']==b0['local_net']
                if a0['action']==b0['action']:continue
                D=a0['local_net'];changes.append(dict(person=j['person'],seed=j['seed'],k=a0['k'],paired=a0['action'],control=b0['action'],D=D,paired_score=a0['gate_score'],control_score=b0['gate_score']))
                if D==0:continue
                key=('added_gain' if D>0 else 'incurred_loss') if a0['action'] else ('missed_gain' if D>0 else 'avoided_loss');terms[key]['count']+=1;terms[key]['value']+=abs(D)
        value=terms['added_gain']['value']+terms['avoided_loss']['value']-terms['missed_gain']['value']-terms['incurred_loss']['value']
        assert abs(value-5*(summaries['paired']['mean_increment']-summaries[arm]['mean_increment']))<1e-7
        decomposition[arm]=dict(terms=terms,changes=changes,mean_difference=value/5)
    dump('selfback/results.json.gz',dict(issued=issued,guarded=gs,budget_rows=bs,selected=selected))
    dump('selfback/summary.json',dict(summary=summaries,units=unit,selected=selected,decomposition=decomposition,budget_summary={str(B):v.summary([g for g in bs if g['budget']==B],TEST,ARMS) for B in BUDGETS},metadata=meta,policy_trajectories=len(gs),budget_paths=len(bs),forks=sum(x['forks'] for x in checks),target_checks=checks,max_service_error=max(g['service_error'] for g in bs),cache=cache.report()))
    print('SUMMARY_SELFBACK',summaries,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','verify','process','calibrate','test']);a=p.parse_args();globals()[a.phase]()
