"""Known-RSS frozen-state dependence/shape bridges; no policy tuning.

Density mixing occurs BEFORE target-line normalization. All tau levels use
the exact historical state, bandwidth and causally issued q of frozen CJ.
These are intervention replays, not replacement closed-loop controllers.
"""
from pathlib import Path
import argparse,csv,datetime,gzip,hashlib,importlib.util,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
FROZEN=PROJECT/'work/fusion_conditional_20261004'
SP=importlib.util.spec_from_file_location('bridge_frozen_conditional',FROZEN/'run_development.py')
r=importlib.util.module_from_spec(SP);SP.loader.exec_module(r);c=r.c
TAUS=(0.,.25,.5,.75,1.)
FAMILIES=('product_to_joint','product_to_joint_diagP','gaussian_to_joint')
SEEDS=tuple(range(88001,88006))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return c.load(p)
def dump(p,v):return c.dump(p,v)
def close(a,b,tol=1e-8):assert abs(float(a)-float(b))<tol,(a,b)
from bridge_math import bridge_state

def freeze():
    r.verify();assert not (ROOT/'protocol.json').exists()
    paths=[ROOT/'bridge_math.py',ROOT/'verify_cached.py',FROZEN/'run_conditional.py',FROZEN/'run_development.py',FROZEN/'tie_v2_protocol.json',FROZEN/'rss348_tie_v2_results.json.gz',
           c.PARENT,c.parent.FACTOR,c.parent.parent.MATRIX_SOURCE,c.parent.CORE,c.parent.ADAPTER,
           PROJECT/'work/fusion_temporal_20261003/physical/data/rss348/cache_metadata.json']
    dump(ROOT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),
        input_sha256={str(p):sha(p) for p in paths},tau=TAUS,families=FAMILIES,seeds=SEEDS,
        formula='f_tau=(1-tau)*baseline+tau*joint; p_tau(u|h) proportional to 1[-1,1](u)*f_tau(h-u1); mix before line normalization',
        product_bridge='preserves every exact coordinate mixture marginal but changes joint dependence/covariance',
        diagonalP_bridge='same coordinate marginals and correction P diagonal as product; isolates retained innovation pairing from P offdiagonals',
        gaussian_bridge='Gaussian baseline has exact joint mean/covariance; all mixtures preserve those moments, isolating distributional shape',
        shared='exact frozen v2 CJ state, source h, mature archive, b=selected b, issued q sequence, model versions, quality, paid targets and fees',
        evidence_scope='known RSS development fixed-state diagnostic, no tuning, no replacement policy, no monotonic gain claim',
        feedback='CJ-issued causal q is held fixed; each bridge is not a separately calibrated closed-loop controller',
        execution='shared Fixed130/B130 replay and independent served-request/prefix reconstruction',
        reporting='all15 family/tau levels, all predictions, utilities, actions, endpoint checks and numerical certificates retained'))
def verify():
    p=load(ROOT/'protocol.json');assert p['script_sha256']==sha(__file__)
    for path,want in p['input_sha256'].items():assert sha(path)==want,path
    r.verify();return p
def row_stats(rows):
    acts=[x for x in rows if x['action']]
    return dict(admissions=len(acts),beneficial=sum(x['local_net']>0 for x in acts),harmful=sum(x['local_net']<0 for x in acts),
        zero=sum(x['local_net']==0 for x in acts),return_sum=sum(x['local_net'] for x in acts),
        negative_loss=sum(max(0.,-x['local_net']) for x in acts),admitted_coverage=[sum(x['lower_covered'] for x in acts),len(acts)])
def differences(rows,baseline):
    terms=dict(kept_gain=0.,missed_gain=0.,incurred_loss=0.,avoided_loss=0.,retained_beneficial=0,baseline_beneficial=0,changed=0);cases=[]
    assert len(rows)==len(baseline)
    for x,y in zip(rows,baseline):
        assert x['k']==y['k'];close(x['local_net'],y['local_net']);u=x['local_net']
        terms['baseline_beneficial']+=bool(y['action'] and u>0);terms['retained_beneficial']+=bool(x['action'] and y['action'] and u>0)
        if x['action']==y['action']:continue
        terms['changed']+=1
        key=('kept_gain' if u>0 else 'incurred_loss' if u<0 else 'zero_return') if x['action'] else ('missed_gain' if u>0 else 'avoided_loss' if u<0 else 'zero_return')
        if key!='zero_return':terms[key]+=abs(u)
        cases.append(dict(k=x['k'],utility=u,category=key,action=x['action'],baseline_action=y['action'],score=x['gate_score'],baseline_score=y['gate_score']))
    terms['difference']=terms['kept_gain']-terms['missed_gain']-terms['incurred_loss']+terms['avoided_loss']
    close(terms['difference'],row_stats(rows)['return_sum']-row_stats(baseline)['return_sum'])
    return terms,cases

def run():
    verify();assert not (ROOT/'results.json.gz').exists()
    frozen=load(FROZEN/'rss348_tie_v2_results.json.gz');selection=load(FROZEN/'tie_v2_protocol.json')['selected']['rss348']['conditional_joint']
    engine,recovery,guard,source,cfg0,study,service,fork=c.parent.binding()
    data,pre=c.parent.task_data('rss348',cfg0,engine,source,study)
    cfg=dict(cfg0,slice_bandwidth=selection['bandwidth'],q_floor=selection['q_floor'],threshold=0.)
    rows_csv=[];audits=[];trials=[];cached=[];maxservice=0.;maxendpoint=0.;poisons=0
    for seed in SEEDS:
        events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg0)
        base=engine.precompute(events,pre,cfg0);stream=c.core.augment(events,pre,base,cfg);reference=service(events,pre,[],cfg)
        old=next(t for t in frozen['trials'] if t['seed']==seed);issued=old['results']['conditional_joint']['rows']
        close(reference['net'],old['reference']['net']);predictions={(f,t):[] for f in FAMILIES for t in TAUS}
        for d,row in zip(stream['decisions'],issued):
            mm=d['temporal'];a,l,V,mu,C,jitter=c.law(mm,pre,d['ids'],cfg)
            post,audit=bridge_state(d['h'],a,l,V,mu,C,mm['P'])
            cached.append(dict(seed=seed,k=d['k'],fork_id=f'rss348:{seed}:{d["k"]}:{row["maturity"]}',
                h=d['h'].tolist(),alpha=a.tolist(),locations=l.tolist(),covariances=V.tolist(),mu=mu.tolist(),C=C.tolist(),P=mm['P'].tolist(),
                N=int(d['N']),q=float(row['q_issued']),ready=bool(mm['eligible']),norm_floor=cfg['norm_floor'],
                complete_fork_D=float(row['local_net']),complete_fork_g=float(row['truegross']),maturity=int(row['maturity']),
                frozen_CJ_F=row['gain'],frozen_CJ_S=row['posterior_or_block_sd'],frozen_CJ_proposal=row['action'],
                original_row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()))
            cleanh=d['h'].copy();poison=dict(d,truegross=1e50,localnet=-1e50)
            assert np.array_equal(cleanh,poison['h']);poisons+=1
            for name,endpoint in (('joint','conditional_joint'),('product','conditional_marginal'),
                                  ('joint_diagP','conditional_joint_diagP'),('gaussian','conditional_gaussian')):
                expected=mm['conditional'][endpoint];observed=audit['endpoints'][name]
                maxendpoint=max(maxendpoint,abs(expected['mean']-observed['mean']),abs(expected['variance']-observed['variance']))
            assert maxendpoint<1e-8
            audits.append(dict(seed=seed,k=d['k'],h=d['h'].tolist(),active_source_ids=d['ids'].tolist(),q=row['q_issued'],
                eligible=mm['eligible'],information_ready=bool(mm['eligible']),**audit))
            for key,p in post.items():
                family,tau=key;F=d['N']*p['mean'] if mm['eligible'] else 0.
                S=d['N']*math.sqrt(p['variance']+cfg['norm_floor']**2);score=F-row['q_issued']*S-5;T=(F-row['truegross'])/S
                rr=dict(row,gain=F,posterior_or_block_sd=S,gate_score=score,standardized_score=T,lower_covered=bool(T<=row['q_issued']),
                    action=bool(score>0),bridge_family=family,bridge_tau=tau,bridge_conditional_mean=p['mean'],bridge_conditional_variance=p['variance'],
                    effective_joint_posterior_mass=p['effective_joint_posterior_mass'],log_line_evidence=p['log_line_evidence'],
                    evidence_ratio_log=p['evidence_ratio_log'])
                predictions[key].append(rr)
                if family=='product_to_joint' and tau==1:
                    close(F,row['gain']);close(S,row['posterior_or_block_sd']);assert rr['action']==row['action']
                rows_csv.append(dict(seed=seed,family=family,tau=tau,k=row['k'],q=row['q_issued'],F=F,S=S,score=score,
                    proposal=rr['action'],utility=row['local_net'],truegross=row['truegross'],mean=p['mean'],variance=p['variance'],
                    effective_joint_mass=p['effective_joint_posterior_mass'],log_evidence=p['log_line_evidence'],log_evidence_ratio=p['evidence_ratio_log']))
        results={}
        for (family,tau),rows in predictions.items():
            unguarded=service(events,pre,rows,cfg);inc=unguarded['net']-reference['net'];close(inc,row_stats(rows)['return_sum'])
            gg=guard.replay(rows,len(events),130.,'gross_loss',{x['k']:130. for x in rows})
            actual=service(events,pre,gg['rows'],cfg);prefix=guard.reconstruct_prefix_increment(events,gg['rows'],cfg)
            gm=row_stats(gg['rows']);ginc=actual['net']-reference['net']
            err=max(abs(ginc-gm['return_sum']),abs(ginc-prefix['final']),abs(ginc-gg['final_settled_increment']))
            maxservice=max(maxservice,err);assert err<1e-8 and gm['negative_loss']<=130.+1e-8 and prefix['minimum']>=-130.-1e-8
            assert all(x['spent_loss']+x['reserved']<=130.+1e-8 for x in gg['ledger'])
            results[f'{family}:{tau}']=dict(family=family,tau=tau,unguarded_increment=inc,unguarded_metrics=row_stats(rows),rows=rows,
                guarded_increment=ginc,guarded_metrics=gm,guarded_rows=gg['rows'],ledger=gg['ledger'],minimum_prefix=prefix['minimum'],
                refused=sum(x['proposed_action'] and not x['action'] for x in gg['rows']))
        trials.append(dict(seed=seed,reference=reference,results=results))
        print('BRIDGE',seed,{f:[results[f'{f}:{t}']['guarded_increment'] for t in TAUS] for f in FAMILIES},flush=True)
    summaries=[];changed=[]
    for family in FAMILIES:
        for tau in TAUS:
            key=f'{family}:{tau}';chosen=[t['results'][key] for t in trials]
            rows=[x for t in chosen for x in t['rows']];grows=[x for t in chosen for x in t['guarded_rows']]
            sums={};gsums={}
            for trial,result in zip(trials,chosen):
                baseline=trial['results'][f'{family}:0.0'];term,cases=differences(result['rows'],baseline['rows']);gterm,gcases=differences(result['guarded_rows'],baseline['guarded_rows'])
                for name,value in term.items():sums[name]=sums.get(name,0)+value
                for name,value in gterm.items():gsums[name]=gsums.get(name,0)+value
                changed += [dict(seed=trial['seed'],family=family,tau=tau,mode=mode,**x) for mode,cc in (('fixed_proposal',cases),('guarded130',gcases)) for x in cc]
            summaries.append(dict(family=family,tau=tau,unguarded_mean_increment=float(np.mean([x['unguarded_increment'] for x in chosen])),
                guarded_mean_increment=float(np.mean([x['guarded_increment'] for x in chosen])),unguarded_increments=[x['unguarded_increment'] for x in chosen],
                guarded_increments=[x['guarded_increment'] for x in chosen],unguarded=row_stats(rows),guarded=row_stats(grows),
                fixed_four_terms_vs_tau0=sums,guarded_four_terms_vs_tau0=gsums,refused=sum(x['refused'] for x in chosen)))
    verification=dict(passed=True,max_endpoint_moment_error=maxendpoint,max_service_error=maxservice,
        max_quadrature_error=max(a['quadrature_error'] for a in audits),max_quadrature_nodes=max(a['quadrature_nodes'] for a in audits),
        max_coordinate_marginal_variance_error=max(a['diagP_coordinate_variance_identity_error'] for a in audits),
        max_gaussian_bridge_mean_covariance_error=max(a['gaussian_bridge_same_mean_covariance_error'] for a in audits),
        max_mixture_moment_identity_error=max(a['mixture_moment_identity_error'] for a in audits),
        posterior_inputs='h and frozen error-law parameters only; current truths used only after prediction for offline callback/accounting',
        future_truth_poison_input_checks=poisons,all_maturity_checks=all(z['maturity']<=a['k'] for a in audits for z in a['eligible']))
    dump(ROOT/'results.json.gz',dict(trials=trials,summaries=summaries,changed_actions=changed,verification=verification))
    dump(ROOT/'state_audit.json.gz',audits);dump(ROOT/'summary.json',dict(summaries=summaries,verification=verification))
    fields=list(rows_csv[0])
    with (ROOT/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows_csv)
    with (ROOT/'summary.csv').open('w',newline='') as f:
        fields=['family','tau','unguarded_mean_increment','guarded_mean_increment','admissions','beneficial','harmful','negative_loss','covered','refused']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for x in summaries:writer.writerow(dict(family=x['family'],tau=x['tau'],unguarded_mean_increment=x['unguarded_mean_increment'],
            guarded_mean_increment=x['guarded_mean_increment'],**{k:x['guarded'][k] for k in ('admissions','beneficial','harmful','negative_loss')},
            covered=x['guarded']['admitted_coverage'][0],refused=x['refused']))
    dump(ROOT/'cached_states.json.gz',cached)
    dump(ROOT/'cached_manifest.json',dict(cache_sha256=sha(ROOT/'cached_states.json.gz'),
        math_sha256=sha(ROOT/'bridge_math.py'),verifier_sha256=sha(ROOT/'verify_cached.py'),
        original_frozen_RSS_sha256=sha(FROZEN/'rss348_tie_v2_results.json.gz'),
        predictions_sha256=sha(ROOT/'predictions.csv'),summary_sha256=sha(ROOT/'summary.json'),
        state_count=len(cached),source_hashes=load(ROOT/'protocol.json')['input_sha256'],
        scope='lossless frozen-state numerical/guard replay; physical service provenance remains in unchanged dependency archive',
        verification=verification))
    print('SUMMARY',[(x['family'],x['tau'],x['guarded_mean_increment']) for x in summaries],flush=True)
    print('VERIFIED',verification,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','run']);args=ap.parse_args()
    freeze() if args.phase=='freeze' else run()
