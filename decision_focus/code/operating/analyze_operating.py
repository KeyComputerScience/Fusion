"""All-point descriptive operating diagnostics of frozen causal forecasts.

No source, model, covariance, q state, callback or selected parameter changes.
The only intervention is an additional constant admission margin, followed by
the unchanged Fixed130/B130 delayed permanent-loss ledger.
"""
from pathlib import Path
import argparse,datetime,gzip,hashlib,importlib.util,json,math,sys
sys.dont_write_bytecode=True
import numpy as np
OUT=Path(__file__).resolve().parent;PROJECT=OUT.parents[2]
OLD=PROJECT/'work/fusion_strengthening_20261003';TEMP=PROJECT/'work/fusion_temporal_20261003'
TASKS=('rss348','arem366','gashome362')
ARMS=('precision_joint','precision_diagonal','scalar_mass','no_posterior','complete_only',
      'gls_exact','block_sandwich','factorized_information','pdf_mlp','qmf_mlp')
FIXED=(0.,1.,2.,4.,8.,12.,16.,24.,32.,48.,64.,96.,128.)
RISK_CAPS=(0.,1.,2.,4.,8.,16.,32.,64.,128.,256.,512.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt') as f:return json.load(f)
    return json.loads(Path(p).read_text())
def dump(p,v):
    text=json.dumps(v,indent=2,allow_nan=False)
    if str(p).endswith('.gz'):
        with gzip.open(p,'wt') as f:f.write(text)
    else:Path(p).write_text(text)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def paths():
    return [TEMP/'rss_validation/rss348_results.json',OLD/'matrix/rss348_results.json',
        OLD/'matrix/factorized/results.json',OLD/'fresh/evaluation/arem366_results.json',
        OLD/'fresh/evaluation/gashome362_results.json']+[OLD/'external/results'/t/'results.json' for t in TASKS]
def freeze():
    assert not (OUT/'protocol.json').exists()
    files=paths()+[Path(__file__),TEMP/'temporal_fusion.py',
        OLD/'matrix/run_matrix_controls.py',OLD/'matrix/factorized/factorized_control.py',
        OLD/'fresh/run_fresh.py',OLD/'fresh/new_data_adapter.py',
        PROJECT/'outputs/Fusion_Loss_Budget_Extension_Repro/guard/run_loss_budget_guard.py']
    files += [TEMP/'physical/data/rss348/cached_dataset.npz']+[OLD/'fresh/data'/t/'cached_dataset.npz' for t in TASKS[1:]]
    declaration=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tasks=list(TASKS),arms=list(ARMS),source_sha256={str(p):sha(p) for p in files},
        intervention='proposal_k(delta)=1{original saved gate_score_k>delta}; delta is constant within every task/method/five-seed point',
        retained='immutable F/S/q, issued standardized score, all matured rejected callbacks, forecast masks/history, model versions and originally selected calibration parameters',
        primary=dict(extra_margin=0.,reserve='fixed130',budget=130.),
        fixed_extra_margins=list(FIXED),critical_grid='common union within each task of all positive saved gate scores across all arms/seeds, rounded upward to 1e-8 utility units; add all fixed margins',
        score_grid_scope='retrospective score-derived breakpoints using no outcome labels; no point is promoted to a new deployed or confirmatory policy',
        fixed_grid_scope='declared constants, all reported without selecting a favorable test margin',
        risk_caps=list(RISK_CAPS),risk_cost='pooled admitted S*(T-q)_+, using original issued q and scores; actual negative loss reported separately',
        matching=['same pooled positive admission count','same full five-seed positive admission count vector'],
        deduplication='identical five-seed admitted action pattern; representative smallest extra margin is chosen by score only; all raw points retained',
        comparisons='all matched full-versus-control operating pairs and primary-full comparisons, exact additional/missed gain/avoided/incurred loss decomposition',
        frontiers='descriptive nondominated return versus score-excess and return versus actual negative loss; preserve all points and risk-cap feasible IDs',
        verification='independent served-request/fee reconstruction at every point; all complete forks; delayed ledger and prefix checks; guard future-outcome perturbation invariance',
        limits=['known-trace diagnostic on all three previously inspected task results',
                'count matching under the permanent-loss ledger conditions on outcomes and is descriptive, not randomized causal isolation',
                'critical score grid uses future score states; selecting its test optimum would not be a prospective margin-selection rule',
                'actual score-excess risk matching is retrospective; it cannot be imposed by an admission-time oracle',
                'five delays share each physical trace; operating points are not independent statistical replications',
                'neural controls are tabular service adaptations, not original state-of-the-art architecture reproductions'],
        runtime=dict(python=sys.version,numpy=np.__version__))
    dump(OUT/'protocol.json',declaration);print('FROZEN_OPERATING',sha(OUT/'protocol.json'),flush=True)
def verify():
    protocol=load(OUT/'protocol.json')
    for p,h in protocol['source_sha256'].items():assert sha(p)==h,p
    return protocol
def saved_rows():
    source={t:{} for t in TASKS};refs={t:{} for t in TASKS}
    original=load(TEMP/'rss_validation/rss348_results.json')
    for trial in original['trials']:
        seed=trial['seed'];refs['rss348'][seed]=trial['results']['reference']['net']
        source['rss348'][seed]={a:trial['results'][a]['rows'] for a in ARMS[:5]}
    for p,arms in [(OLD/'matrix/rss348_results.json',('gls_exact','block_sandwich')),
                   (OLD/'matrix/factorized/results.json',('factorized_information',))]:
        for trial in load(p)['trials']:
            for a in arms:source['rss348'][trial['seed']][a]=trial['results'][a]['rows']
    for task in TASKS[1:]:
        for trial in load(OLD/'fresh/evaluation'/(task+'_results.json'))['trials']:
            source[task][trial['seed']]={a:trial['results'][a]['rows'] for a in ARMS[:8]}
            refs[task][trial['seed']]=trial['results']['reference']['net']
    for task in TASKS:
        for trial in load(OLD/'external/results'/task/'results.json')['trials']:
            for raw,name in [('pdf','pdf_mlp'),('qmf','qmf_mlp')]:source[task][trial['seed']][name]=trial['results'][raw]['rows']
        for seed,methods in source[task].items():
            assert set(methods)==set(ARMS)
            full=methods['precision_joint']
            for arm,rows in methods.items():
                assert len(rows)==len(full)
                for x,y in zip(full,rows):
                    assert x['k']==y['k'] and x['local_net']==y['local_net'] and x['truegross']==y['truegross']
                    assert bool(y['action'])==bool(y['gate_score']>0)
    return source,refs
def environment(task):
    if task=='rss348':
        m=module('operating_matrix_env',OLD/'matrix/run_matrix_controls.py')
        engine,recovery,guard,source,cfg,study,service,fork=m.binding();_,pre=study(task,source['studies'][task])
    else:
        m=module('operating_fresh_env',OLD/'fresh/run_fresh.py')
        engine,recovery,guard,cfg,service,fork=m.binding()
        data,derivation=m.adapter.load_dataset(OLD/'fresh/data',task);pre=m.adapter.make_prefix(data,derivation,cfg,engine)
    return engine,guard,cfg,pre,service,fork
def excess(row):return max(0.,row['posterior_or_block_sd']*(row['standardized_score']-row['q_issued']))
def score_grid(methods):
    scores={math.ceil(row['gate_score']/1e-8)*1e-8 for methods1 in methods.values() for rows in methods1.values() for row in rows if row['gate_score']>0}
    return sorted(scores.union(FIXED))
def decompose(source,p,c):
    categories={x:dict(count=0,units=0.) for x in ('additional_gain','avoided_loss','missed_gain','incurred_loss','zero_return')}
    cases=[];shared=0;sharedunits=0.
    for seed,pa,ca in zip(p['seeds'],p['actions'],c['actions']):
        full=source[seed]['precision_joint'];other=source[seed][c['arm']]
        for x,y,a,b in zip(full,other,pa,ca):
            D=x['local_net']
            if a and b and D>0:shared+=1;sharedunits+=D
            if a==b:continue
            cat=('additional_gain' if D>0 else 'incurred_loss' if D<0 else 'zero_return') if a else ('missed_gain' if D>0 else 'avoided_loss' if D<0 else 'zero_return')
            categories[cat]['count']+=1;categories[cat]['units']+=abs(D)
            cases.append(dict(seed=seed,origin=x['k'],actual_return=D,full_action=bool(a),control_action=bool(b),category=cat,
                full_issued_score=x['gate_score'],control_issued_score=y['gate_score'],full_q=x['q_issued'],control_q=y['q_issued'],
                full_scaled_excess=excess(x),control_scaled_excess=excess(y),full_minus_control=(int(a)-int(b))*D))
    pooled=sum(x['full_minus_control'] for x in cases)
    assert abs(pooled/5-(p['mean_increment']-c['mean_increment']))<1e-8
    return dict(full_point=p['id'],control_point=c['id'],same_pooled_count=p['admissions']==c['admissions'],
        same_count_vector=p['admission_vector']==c['admission_vector'],categories=categories,changed_actions=len(cases),
        shared_beneficial_count=shared,shared_beneficial_units=sharedunits,total_full_minus_control=pooled,mean_full_minus_control=pooled/5,
        full_excess=p['admitted_excess'],control_excess=c['admitted_excess'],cases=cases)
def dominates(a,b,risk):
    return a['mean_increment']>=b['mean_increment']-1e-10 and a[risk]<=b[risk]+1e-10 and (a['mean_increment']>b['mean_increment']+1e-10 or a[risk]<b[risk]-1e-10)
def run():
    verify();assert not (OUT/'summary.json').exists()
    sources,references=saved_rows();task_summaries={};all_points=[];all_trajectories=[];all_matches=[]
    verification=dict(trajectories=0,independent_forks=0,max_service_error=0.,max_fork_error=0.,max_prefix_accounting_error=0.,max_loss_violation=0.,future_truth_checks=0,source_integrity=True)
    for task in TASKS:
        methods=sources[task];seeds=sorted(methods);grid=score_grid(methods)
        dump(OUT/(task+'_score_grid.json'),dict(task=task,extra_margins=grid,fixed_margins=list(FIXED),derived_without_labels=True))
        engine,guard,cfg,pre,service,fork=environment(task);worlds={}
        for seed in seeds:
            e=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg)
            reference=service(e,pre,[],cfg);assert abs(reference['net']-references[task][seed])<1e-8;worlds[seed]=(e,reference)
            for row in methods[seed]['precision_joint']:
                f=fork(e,row,cfg);error=max(abs(f['actual']-row['local_net']),abs(f['gross']-row['truegross']))
                verification['independent_forks']+=1;verification['max_fork_error']=max(verification['max_fork_error'],error);assert error<1e-8
        points=[]
        for arm in ARMS:
            for index,delta in enumerate(grid):
                trials=[];actions=[]
                for seed in seeds:
                    old=methods[seed][arm];e,reference=worlds[seed]
                    rows=[dict(r,action=bool(r['gate_score']>delta)) for r in old]
                    g=guard.replay(rows,len(e),130.,'gross_loss',{r['k']:130. for r in rows})
                    actual=service(e,pre,g['rows'],cfg);prefix=guard.reconstruct_prefix_increment(e,g['rows'],cfg)
                    selected=[r for r in g['rows'] if r['action']];increment=sum(r['local_net'] for r in selected);loss=sum(max(0.,-r['local_net']) for r in selected)
                    error=abs(actual['net']-reference['net']-increment);prefix_error=abs(prefix['final']-increment)
                    violation=max(0.,loss-130.,-130.-prefix['minimum'])
                    verification['trajectories']+=1;verification['max_service_error']=max(verification['max_service_error'],error)
                    verification['max_prefix_accounting_error']=max(verification['max_prefix_accounting_error'],prefix_error)
                    verification['max_loss_violation']=max(verification['max_loss_violation'],violation)
                    assert error<1e-8 and prefix_error<1e-8 and violation<1e-8
                    cutoff=len(e)//2;poison=guard.replay(rows,len(e),130.,'gross_loss',{r['k']:130. for r in rows},perturb_after=cutoff)
                    assert [r['action'] for r in g['rows'] if r['k']<=cutoff]==[r['action'] for r in poison['rows'] if r['k']<=cutoff]
                    verification['future_truth_checks']+=1
                    bits=[bool(r['action']) for r in g['rows']];actions.append(bits)
                    trial=dict(task=task,arm=arm,grid_index=index,seed=seed,extra_margin=delta,increment=increment,admissions=len(selected),
                        harmful=sum(r['local_net']<0 for r in selected),beneficial=sum(r['local_net']>0 for r in selected),zero=sum(r['local_net']==0 for r in selected),
                        negative_loss=loss,admitted_excess=sum(excess(r) for r in selected),admitted_max_excess=max((excess(r) for r in selected),default=0.),
                        covered=sum(r['lower_covered'] for r in selected),penalty_cost=sum(r['penalty'] for r in selected),
                        refusals=sum(r['proposed_action'] and not r['action'] for r in g['rows']),proposal_count=sum(r['proposed_action'] for r in g['rows']),
                        minimum_prefix_increment=prefix['minimum'],actions=bits,
                        admitted=[dict(origin=r['k'],complete_return=r['local_net'],issued_score=r['gate_score'],issued_q=r['q_issued'],scaled_excess=excess(r)) for r in selected])
                    trials.append(trial);all_trajectories.append(trial)
                point=dict(id=f'{task}:{arm}:{index}',task=task,arm=arm,extra_margin=delta,fixed_margin=delta in FIXED,seeds=seeds,
                    mean_increment=float(np.mean([t['increment'] for t in trials])),increments=[t['increment'] for t in trials],
                    admissions=sum(t['admissions'] for t in trials),admission_vector=[t['admissions'] for t in trials],
                    harmful=sum(t['harmful'] for t in trials),beneficial=sum(t['beneficial'] for t in trials),zero=sum(t['zero'] for t in trials),
                    negative_loss=sum(t['negative_loss'] for t in trials),admitted_excess=sum(t['admitted_excess'] for t in trials),
                    admitted_max_excess=max(t['admitted_max_excess'] for t in trials),coverage=[sum(t['covered'] for t in trials),sum(t['admissions'] for t in trials)],
                    penalty_cost=sum(t['penalty_cost'] for t in trials),refusals=sum(t['refusals'] for t in trials),actions=actions)
                points.append(point);all_points.append(point)
            print('OPERATING',task,arm,'points',len(grid),flush=True)
        unique={};aliases={}
        for point in points:
            key=(point['arm'],json.dumps(point['actions']))
            if key not in unique:unique[key]=point;aliases[point['id']]=[]
            else:aliases[unique[key]['id']].append(point['id'])
        candidates=list(unique.values());fulls=[p for p in candidates if p['arm']=='precision_joint'];primary=next(p for p in fulls if p['extra_margin']==0.)
        matches=[]
        for full in fulls:
            if full['admissions']==0:continue
            for other in candidates:
                if other['arm']=='precision_joint' or other['admissions']!=full['admissions']:continue
                matches.append(decompose(methods,full,other))
        primary_comparisons=[decompose(methods,primary,p) for p in candidates if p['arm']!='precision_joint' and p['admissions']==primary['admissions'] and primary['admissions']>0]
        all_matches.extend(matches)
        risk_frontier=[p['id'] for p in candidates if not any(dominates(q,p,'admitted_excess') for q in candidates)]
        loss_frontier=[p['id'] for p in candidates if not any(dominates(q,p,'negative_loss') for q in candidates)]
        cap_points={str(cap):[p['id'] for p in candidates if p['admitted_excess']<=cap+1e-10] for cap in RISK_CAPS}
        primary_rows={arm:next(p for p in points if p['arm']==arm and p['extra_margin']==0.) for arm in ARMS}
        diagnostic=dict(task=task,score_grid_size=len(grid),raw_points=len(points),unique_action_patterns=len(candidates),aliases=aliases,
            primary=primary_rows,unique_points=candidates,primary_full_matched_count=primary_comparisons,
            all_matched_count_pairs=len(matches),all_strict_count_vector_pairs=sum(c['same_count_vector'] for c in matches),
            score_excess_frontier_ids=risk_frontier,realized_loss_frontier_ids=loss_frontier,risk_cap_feasible_ids=cap_points,
            original_full_on_score_excess_frontier=primary['id'] in risk_frontier,original_full_on_realized_loss_frontier=primary['id'] in loss_frontier)
        task_summaries[task]=diagnostic
        dump(OUT/(task+'_diagnostic.json'),diagnostic)
        dump(OUT/(task+'_matched_comparisons.json.gz'),matches)
        print('TASK_COMPLETE',task,'raw',len(points),'unique',len(candidates),'matched',len(matches),'primary_count_matches',len(primary_comparisons),flush=True)
    verify();dump(OUT/'all_points.json.gz',all_points);dump(OUT/'all_trajectories.json.gz',all_trajectories)
    dump(OUT/'summary.json',dict(protocol_sha256=sha(OUT/'protocol.json'),tasks={t:{k:v for k,v in d.items() if k not in ('unique_points','aliases','risk_cap_feasible_ids')} for t,d in task_summaries.items()},verification=verification))
    print('OPERATING_PASS',verification,flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('freeze','run'),required=True);args=parser.parse_args()
    dict(freeze=freeze,run=run)[args.phase]()
if __name__=='__main__':main()
