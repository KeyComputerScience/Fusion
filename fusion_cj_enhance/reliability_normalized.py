"""CJ-R: reliability-normalized complete-error conditioning.

One working importance-likelihood change, applied to whole mature masked
blocks before covariance, correction, completion and predictive density.
All original sources remain immutable; extension hooks are in memory only.
"""
from pathlib import Path
import importlib.util,math,sys
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
FROZEN=PROJECT/'work/fusion_conditional_20261004'
sp=importlib.util.spec_from_file_location('cjr_frozen_copula',FROZEN/'copula_control.py')
cp=importlib.util.module_from_spec(sp);sp.loader.exec_module(cp)
c=cp.c;core=c.core;matrix=c.parent.parent.matrix
old_collect=matrix.collect_blocks;old_base_state=matrix.original_state
NEW_ARMS=('r_joint','r_product','r_factorized','r_gaussian','r_sandwich','r_joint_diagP','r_copula')
MAP=dict(zip(NEW_ARMS,('conditional_joint','conditional_marginal','conditional_factorized',
    'conditional_gaussian','conditional_sandwich','conditional_joint_diagP','conditional_copula')))
ARMS=('conditional_joint',)+NEW_ARMS+('legacy_factorized','legacy_sandwich')

def normalize_weights(weights):
    """ESS-weighted relevance; no claim of independent observations."""
    w=np.asarray(weights,float)
    if not len(w):return w.copy(),dict(raw_mass=0.,raw_square_mass=0.,effective_mass=0.,multiplier=0.,blocks=0)
    assert np.all(w>0) and np.all(np.isfinite(w))
    total=float(w.sum());square=float(w@w);ess=total*total/square;multiplier=total/square
    out=w*multiplier
    assert 1.-1e-10<=ess<=len(w)+1e-10
    return out,dict(raw_mass=total,raw_square_mass=square,effective_mass=ess,multiplier=multiplier,blocks=len(w))

def collect(arrived,current,pre,cfg):
    obs,complete=old_collect(arrived,current,pre,cfg)
    if not cfg.get('reliability_normalized',False):return obs,complete
    # A wholly unobserved block has no error information and cannot add
    # effective mass merely through Gaussian completion.
    obs=[x for x in obs if len(x['ids'])>0]
    weights,_=normalize_weights([x['weight'] for x in obs])
    changed=[dict(x,raw_weight=x['weight'],weight=float(w)) for x,w in zip(obs,weights)]
    complete=[(x['weight'],x['residual']) for x in changed if len(x['ids'])==pre['m']]
    return changed,complete

def base_state(arrived,current,pre,cfg):
    if not cfg.get('reliability_normalized',False):return old_base_state(arrived,current,pre,cfg)
    obs,complete=collect(arrived,current,pre,cfg)
    R,mass=core.covariance(complete,pre['m'],cfg['prior_mass'],cfg['prior_variance'])
    quality=pre['q']/np.mean(pre['q']);K=np.diag(quality/cfg['prior_sd']**2)
    A=K.copy();rhs=np.zeros(pre['m']);Ac=K.copy();rhsc=rhs.copy()
    for x in obs:
        ids=x['ids'];iv=np.linalg.solve(R[np.ix_(ids,ids)],np.eye(len(ids)))
        A[np.ix_(ids,ids)]+=x['weight']*iv;rhs[ids]+=x['weight']*(iv@x['residual'])
        if len(ids)==pre['m']:Ac[np.ix_(ids,ids)]+=x['weight']*iv;rhsc[ids]+=x['weight']*(iv@x['residual'])
    P=np.linalg.solve(A,np.eye(pre['m']));P=(P+P.T)/2;mu=np.linalg.solve(A,rhs)
    Pc=np.linalg.solve(Ac,np.eye(pre['m']));muc=np.linalg.solve(Ac,rhsc)
    ids=current['ids'];Pa=P[np.ix_(ids,ids)];Ra=R[np.ix_(ids,ids)]
    scalar=float(np.sum(Pa*Ra)/np.sum(Ra*Ra))
    _,info=normalize_weights([x['raw_weight'] for x in obs])
    info.update(block_masks=[x['ids'].tolist() for x in obs],complete_blocks=len(complete),
        original_cutoff=1e-12,readiness='nonempty eligible observed mature contrast blocks; zero-coordinate blocks excluded',
        interpretation='effective mass is weight concentration, not independent sample count')
    return dict(mu=mu[ids],R=Ra,P=Pa,complete_mu=muc[ids],complete_P=Pc[np.ix_(ids,ids)],mass=mass,
        all_mass=cfg['prior_mass']+sum(x['weight'] for x in obs),
        non_scalar=float(np.linalg.norm(Pa-scalar*Ra)/max(np.linalg.norm(Pa),1e-15)),
        eligible=[dict(origin=x['origin'],maturity=x['maturity'],weight=x['weight'],raw_weight=x['raw_weight'],
                       observed_components=len(x['ids'])) for x in obs],
        complete_blocks=len(complete),partial_blocks=sum(len(x['ids'])<pre['m'] for x in obs),reliability=info)

matrix.collect_blocks=collect;matrix.original_state=base_state

def state(arrived,current,pre,cfg):
    mm=cp.state(arrived,current,pre,cfg)
    a,l,V,_,_,_=c.law(mm,pre,current['ids'],cfg)
    mm['conditional']['conditional_joint_diagP']=c.joint_posterior(current['h'],a,l,V-mm['P']+np.diag(np.diag(mm['P'])))
    return mm

def forecast(d,pre,cfg,arm,q):
    if arm=='legacy_factorized':return c.parent.parent.factorized_forecast(d,pre,cfg,'factorized_information',q)
    if arm=='legacy_sandwich':return matrix.matrix_forecast(d,pre,cfg,'block_sandwich',q)
    answer=c.conditional_forecast(d,pre,cfg,MAP.get(arm,arm),q)
    if cfg.get('reliability_normalized',False):answer[-1]['reliability']=d['temporal']['reliability']
    return answer

core.state=state;core.forecast=forecast

def configure(cfg,arm):
    return dict(cfg,reliability_normalized=arm in NEW_ARMS)

def binding():return c.parent.binding()
