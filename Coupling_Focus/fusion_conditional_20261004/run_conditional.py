"""Declared development prototype: bounded scalar complete-target line posterior.

All imported sources are immutable. Current labels never enter the posterior.
The strongest marginal control sees exactly the same current h and each
source marginal; it also conditions on current disagreement. There is no
claimed source-weight optimization after exact affine conditioning.
"""
from pathlib import Path
import argparse, datetime, gzip, hashlib, importlib.util, json, math, sys
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
PARENT=PROJECT/'work/fusion_increment_20261003/cumulant/run_cumulant.py'
spec=importlib.util.spec_from_file_location('conditional_cumulant_parent',PARENT)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
core=parent.core
ARMS=('conditional_joint','conditional_marginal','conditional_factorized',
      'conditional_gaussian','conditional_sandwich')
TASKS=('rss348','arem366','gashome362')
BANDWIDTHS=(.25,.5,1.)
FLOORS=(0.,.64,1.2815515655446004)
LOG2PI=math.log(2*math.pi)
QUAD_X,QUAD_W=np.polynomial.legendre.leggauss(8)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,obj):
    s=json.dumps(obj,indent=2,allow_nan=False)
    if str(p).endswith('.gz'):
        with gzip.open(p,'wt') as f:f.write(s)
    else:Path(p).write_text(s)
def load(p):return parent.load(p)
def logsumexp(a,axis=None):
    high=np.max(a,axis=axis,keepdims=True)
    out=high+np.log(np.sum(np.exp(a-high),axis=axis,keepdims=True))
    return float(out.ravel()[0]) if axis is None else np.squeeze(out,axis=axis)
def cdf(x):return .5*math.erfc(-float(x)/math.sqrt(2))
def phi(x):return math.exp(-.5*float(x)**2)/math.sqrt(2*math.pi)

def truncated_components(alpha,means,variances,log_evidence):
    """Analytic Gaussian line components, restricted to the target [-1,1]."""
    logs=[];first=[];second=[]
    for a,mu,var,ev in zip(alpha,means,variances,log_evidence):
        sd=math.sqrt(float(var));lo=(-1-mu)/sd;hi=(1-mu)/sd
        # Use complementary tails on one side to avoid cancellation.
        Z=(cdf(hi)-cdf(lo)) if lo<0 else (cdf(-lo)-cdf(-hi))
        if Z<=1e-280:continue
        shift=(phi(lo)-phi(hi))/Z
        mean=float(mu+sd*shift)
        variance=float(var*(1+(lo*phi(lo)-hi*phi(hi))/Z-shift**2))
        assert -1.00000001<=mean<=1.00000001
        variance=max(0.,variance)
        logs.append(math.log(float(a))+float(ev)+math.log(Z))
        first.append(mean);second.append(variance+mean**2)
    assert logs,'All bounded posterior component masses underflowed'
    logs=np.array(logs);prob=np.exp(logs-logsumexp(logs))
    mean=float(prob@first);variance=max(0.,float(prob@second)-mean**2)
    return dict(mean=mean,variance=variance,component_count=len(logs),
                posterior_effective_components=float(1/(prob@prob)),
                posterior_component_weights=prob.tolist(),quadrature_error=0.)

def joint_posterior(h,alpha,locations,covariances):
    """p(T|h) proportional to sum alpha N(h-T*1; location, covariance)."""
    means=[];variances=[];evidence=[];one=np.ones(len(h))
    for loc,S in zip(locations,covariances):
        precision=np.linalg.solve(S,np.eye(len(h)))
        v=1/float(one@precision@one);r=h-loc
        mu=v*float(one@precision@r)
        _,ld=np.linalg.slogdet(S)
        ev=-.5*(len(h)*LOG2PI+ld+float(r@precision@r)-mu**2/v)+.5*math.log(2*math.pi*v)
        means.append(mu);variances.append(v);evidence.append(ev)
    return truncated_components(alpha,np.array(means),np.array(variances),np.array(evidence))

def quadrature(bins):
    edges=np.linspace(-1.,1.,bins+1);mid=.5*(edges[:-1]+edges[1:]);half=1/bins
    return (mid[:,None]+half*QUAD_X).ravel(),np.tile(half*QUAD_W,bins)

def product_integral(h,alpha,locations,variances,bins):
    target,qw=quadrature(bins);logdensity=np.zeros(len(target))
    for s in range(len(h)):
        r=h[s]-target[:,None]-locations[None,:,s]
        log=np.log(alpha)[None,:]-.5*(LOG2PI+np.log(variances[:,s])[None,:]+r*r/variances[:,s][None,:])
        logdensity+=logsumexp(log,axis=1)
    lp=logdensity+np.log(qw);prob=np.exp(lp-logsumexp(lp))
    mu=float(prob@target);variance=max(0.,float(prob@(target-mu)**2))
    return mu,variance

def product_posterior(h,alpha,locations,covariances):
    """Strong product control: each original coordinate mixture is unchanged.

    One-dimensional line integration avoids a 49**m Cartesian atom expansion.
    Coarse/refined moments are retained, and failed refinement is not hidden.
    """
    variances=np.diagonal(covariances,axis1=1,axis2=2)
    minsd=math.sqrt(float(1/np.sum(1/variances.min(0))))
    bins=max(32,int(math.ceil(1/minsd)))
    prev=product_integral(h,alpha,locations,variances,bins)
    err=1.;steps=0
    for steps in range(4):
        bins*=2;out=product_integral(h,alpha,locations,variances,bins)
        err=max(abs(out[0]-prev[0]),abs(out[1]-prev[1]))
        if err<1e-9:break
        prev=out
    return dict(mean=out[0],variance=out[1],component_count=len(alpha),
                posterior_effective_components=None,quadrature_error=err,quadrature_nodes=8*bins,
                quadrature_refinements=steps+1)

def law(mm,pre,ids,cfg,factorized=False):
    prefix='factorized_' if factorized else ''
    atoms=mm[prefix+'atoms'] if factorized else mm['paired_atoms']
    mu=mm[prefix+'mu'];P=mm[prefix+'P']
    # Kernel bandwidth is source-quality-scaled and shared by every arm.
    jitter=cfg['slice_bandwidth']**2*cfg['prior_variance']*np.diag(np.mean(pre['q'])/pre['q'][ids])
    locations=mu+atoms['v'];covariances=atoms['V']+P+jitter
    return atoms['alpha'],locations,covariances,mu,atoms['C']+P+jitter,jitter

def conditional_state(arrived,current,pre,cfg):
    mm=parent.cumulant_state(arrived,current,pre,cfg)
    h=current['h'];ids=current['ids']
    a,l,V,mu,C,jitter=law(mm,pre,ids,cfg)
    fa,fl,fV,_,_,_=law(mm,pre,ids,cfg,True)
    post={
        'conditional_joint':joint_posterior(h,a,l,V),
        'conditional_marginal':product_posterior(h,a,l,V),
        'conditional_factorized':product_posterior(h,fa,fl,fV),
        'conditional_gaussian':joint_posterior(h,np.ones(1),mu[None],C[None]),
        'conditional_sandwich':joint_posterior(h,np.ones(1),mu[None],
            (mm['paired_atoms']['C']+mm['sandwich_P']+jitter)[None])}
    # Deterministic complete-target GLS line = untruncated Gaussian mode.
    one=np.ones(len(h));cinv=np.linalg.solve(C,np.eye(len(h)))
    gls=float(one@cinv@(h-mu)/(one@cinv@one))
    mm.update(conditional=post,current_source_rate=h.tolist(),
              current_disagreement=(h-h.mean()).tolist(),line_gls_untruncated_mean=gls,
              conditioning_inputs=dict(h=h.tolist(),source_ids=ids.tolist(),target_bounds=[-1.,1.]),
              current_label_access=False)
    assert max(p['quadrature_error'] for p in post.values())<1e-7
    return mm

def conditional_forecast(d,pre,cfg,arm,q):
    mm=d['temporal'];p=mm['conditional'][arm]
    ready=bool(mm['eligible']);F=float(d['N']*p['mean']) if ready else 0.
    S=float(d['N']*math.sqrt(p['variance']+cfg['norm_floor']**2))
    # Quality weights are only a compatibility field in the legacy logger.
    # The scalar posterior and admission do not depend on them.
    weights=core.capped(pre['q'][d['ids']]/sum(pre['q'][d['ids']]),max(cfg['cap'],1/len(d['ids'])))
    return weights,dict(converged=True,kkt=0.,primal=0.,iterations=0),F,S,(F-d['truegross'])/S,dict(
        information_ready=ready,raw_optimized_gain=float(d['N']*p['mean']),posterior_norm=0.,entropy=0.,
        conditional_mean=p['mean'],conditional_variance=p['variance'],current_source_rate=mm['current_source_rate'],
        current_disagreement=mm['current_disagreement'],source_weights_redundant=True,
        slice_bandwidth=cfg['slice_bandwidth'],quadrature_error=p['quadrature_error'],
        posterior_effective_components=p['posterior_effective_components'],line_gls_untruncated_mean=mm['line_gls_untruncated_mean'])

core.state=conditional_state;core.forecast=conditional_forecast

def protocol():
    assert not (ROOT/'protocol.json').exists()
    files=[PARENT,parent.FACTOR,parent.parent.MATRIX_SOURCE,parent.CORE,parent.ADAPTER,
           PROJECT/'outputs/Fusion_Recovery_Repro/base/independent_bayes_fusion.py']
    dump(ROOT/'protocol.json',dict(utc=now(),runner_sha256=sha(__file__),input_sha256={str(p):sha(p) for p in files},
        tasks=TASKS,status='development only: every physical trace already examined; calibration partition diagnostics',
        arms=ARMS,formula='p(T|h) proportional to f_Z(h-T*1) 1[-1<=T<=1]; F=N E[T|h]; S=N sqrt(Var[T|h]+nu²)',
        target_prior='uniform[-1,1], a declared working prior, not an empirically proven sampling law',
        joint='alpha mixture locations mu+v_j; component covariance V_j+P+b² sigma0² diag(mean(q)/q)',
        marginal='product of same coordinate mixture marginals; same h, mean correction, P diagonal marginals and jitter',
        factorized='global diagonal R before correction and masked atom completion, then product marginals',
        gaussian='single Gaussian with exactly same joint-law mean/covariance; deterministic line GLS and posterior moment integral are identical',
        sandwich='same Gaussian covariance with block HC0 sandwich replacing correction P',
        source_weight_invariance="exact Z=h-T1 implies w'h-w'Z=T for every simplex w; displayed weights are legacy fields only",
        context_audit='engine.context returns source-group means of raw standardized current covariates only; no probability disagreement',
        shared='identical current h, masks, matured reprojected archive, context/support/age weights, quality, source models, work costs, delayed q and guard',
        calibration=dict(bandwidth=BANDWIDTHS,q_floor=FLOORS,configs_per_arm=9,
            initial_q='higher90th percentile firsthalf fullymatured ready calibration scores',
            select='max secondhalf mean complete return; ties larger qfloor then smaller bandwidth',
            seeds={n:parent.seeds(n)[0] for n in TASKS}),
        diagnostic='no new physical holdout read; evaluation is calibration secondhalf, selected on that same development segment',
        causal='current truegross used only after prediction for retained offline score/callback; history blocks require complete maturity',
        limitations=['working error density estimated and weighted under drift','target uniform prior and Gaussian completion/smoothing are assumptions',
            'partial observed source errors do not identify missing joint laws','same-state marginal retains mu and marginal P learned jointly, and is the stronger pairing-only control',
            'Gaussian control can equal or beat joint mixture; no universal superiority','five simulation delays are not independent physical sites'],
        runtime=dict(python=sys.version,numpy=np.__version__)))

def verify():
    p=load(ROOT/'protocol.json');assert p['runner_sha256']==sha(__file__)
    for path,want in p['input_sha256'].items():assert sha(path)==want
    return p

def calibrate(tasks=TASKS):
    verify();engine,recovery,guard,source,cfg0,study,service,fork=parent.binding()
    for name in tasks:
        assert not (ROOT/(name+'_selection.json')).exists()
        data,pre=parent.task_data(name,cfg0,engine,source,study)
        # Only the known calibration partition is made into physical events.
        events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,cfg0) for s in parent.seeds(name)[0]]
        bases=[engine.precompute(e,pre,cfg0) for e in events];trials=[];grid=[];audit=[]
        for b in BANDWIDTHS:
            cfg=dict(cfg0,slice_bandwidth=b)
            streams=[core.augment(e,pre,base,cfg) for e,base in zip(events,bases)]
            audit.extend(dict(seed=seed,bandwidth=b,k=d['k'],eligible=d['temporal']['eligible'],
                h=d['h'].tolist(),conditional=d['temporal']['conditional'],
                current_label_access=False) for seed,s in zip(parent.seeds(name)[0],streams) for d in s['decisions'])
            for arm in ARMS:
                for floor in FLOORS:
                    cc=dict(cfg,q_floor=floor,threshold=0.)
                    qi,scores=core.initial_q(streams,pre,cc,arm)
                    rr=[core.run(engine,s,pre,cc,arm,qi,True) for s in streams]
                    assert all(r['solver_failures']==0 for r in rr)
                    g=dict(arm=arm,bandwidth=b,q_floor=floor,q_initial=qi,fit_scores=scores,
                           net=float(np.mean([r['selection_net'] for r in rr])))
                    grid.append(g);trials.append(dict(configuration=g,seeds=parent.seeds(name)[0],results=rr))
                print('CAL',name,b,arm,flush=True)
        selected={a:max((g for g in grid if g['arm']==a),key=lambda g:(g['net'],g['q_floor'],-g['bandwidth'])) for a in ARMS}
        dump(ROOT/(name+'_calibration_trials.json.gz'),trials)
        dump(ROOT/(name+'_state_audit.json.gz'),audit)
        dump(ROOT/(name+'_selection.json'),dict(selected=selected,grid=grid,data_hashes=data['hashes'],split=pre['split']))
        print('SELECT',name,{a:(v['bandwidth'],v['q_floor'],v['q_initial'],v['net']) for a,v in selected.items()},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','calibrate']);ap.add_argument('--tasks',nargs='+',default=TASKS)
    args=ap.parse_args()
    protocol() if args.phase=='freeze' else calibrate(tuple(args.tasks))
