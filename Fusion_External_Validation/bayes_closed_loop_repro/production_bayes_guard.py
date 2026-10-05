"""Live-safe admission wrapper around the unchanged frozen convex solver.

Only arrived archive moments and issued current forecasts are required. Future
fork outcomes are not read here. Empty forecast masks abstain BEFORE moments;
one source has exactly unit influence. This wrapper is not additional measured
performance and does not modify the frozen experimental executable.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import independent_bayes_fusion as engine

FROZEN_ENGINE_SHA='15e3fc99d3f0935d2e66853b9b1949ef7825869c2f0d56a8ba49513838376e57'


def _matrix(value,n,name):
    value=np.asarray(value,dtype=float)
    if value.shape!=(n,n) or not np.isfinite(value).all():
        raise ValueError(name+' must be a finite active-source square matrix')
    if not np.allclose(value,value.T,rtol=0,atol=1e-9):
        raise ValueError(name+' must be symmetric')
    if np.linalg.eigvalsh(value).min()<-1e-9:
        raise ValueError(name+' must be positive semidefinite')
    return value


def guarded_decision(record,quality,cfg,mode,q_issued,*,feasible=True,
                     conservative_cost=5.,variance_floor=1e-4):
    """Return a decision from live information, with explicit zero/one-source paths.

    record requires ids plus N,h,mu,S,Us,U,B on a nonempty mask; matrices have
    already been sliced/reestimated from complete joint rows for that mask.
    It deliberately ignores truegross/localnet/standardized_score if an offline
    dictionary happens to contain them. q must come from matured callbacks.
    """
    if hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()!=FROZEN_ENGINE_SHA:
        raise RuntimeError('Frozen solver changed')
    ids=np.asarray(record['ids'])
    if ids.ndim!=1 or (ids.size and not np.issubdtype(ids.dtype,np.integer)):
        raise ValueError('ids must be a one-dimensional integer active-source list')
    quality=np.asarray(quality,dtype=float)
    if quality.ndim!=1:raise ValueError('quality must be one-dimensional')
    if ids.size and (np.any(ids<0) or np.any(ids>=len(quality)) or len(set(ids.tolist()))!=len(ids)):
        raise ValueError('ids must be unique in-range source indices')
    if not ids.size:
        return dict(action=False,abstained=True,reason='no_available_forecast_source',
                    weights=[],gain=None,lower=None,score=None,sd=None,q_issued=float(q_issued))
    if not np.isfinite(q_issued) or q_issued<0:raise ValueError('q_issued must be finite and nonnegative')
    if not np.isfinite(conservative_cost) or conservative_cost<0:raise ValueError('cost must be finite and nonnegative')
    if not np.isfinite(variance_floor) or variance_floor<=0:raise ValueError('variance floor must be positive')
    n=len(ids);N=float(record['N'])
    if not np.isfinite(N) or N<0:raise ValueError('N must be finite and nonnegative')
    if N==0:
        return dict(action=False,abstained=True,reason='empty_request_window',
                    weights=[],gain=None,lower=None,score=None,sd=None,q_issued=float(q_issued))
    h=np.asarray(record['h'],dtype=float);mu=np.asarray(record['mu'],dtype=float)
    if h.shape!=(n,) or mu.shape!=(n,) or not np.isfinite(h).all() or not np.isfinite(mu).all():
        raise ValueError('h and mu must be finite active-source vectors')
    if not np.isfinite(quality[ids]).all() or np.any(quality[ids]<=0):
        raise ValueError('active-source quality must be finite and strictly positive')
    matrices={name:_matrix(record[name],n,name) for name in ('S','Us','U','B')}
    solver_mode='joint' if mode=='diagonal_posterior_gate' else mode
    if n==1:
        weights=np.ones(1);cert=dict(kkt=0.,primal=0.,converged=True,iterations=0)
    else:
        weights,cert=engine.convex_fuse(quality[ids],matrices['S'],matrices['Us'],cfg,solver_mode)
    variance=matrices['B'] if mode=='frequentist_gate' else matrices['U']
    if mode=='diagonal_posterior_gate':variance=np.diag(np.diag(variance))
    gain=float(N*weights@(h-mu));sd=N*float(np.sqrt(max(0.,float(weights@variance@weights))+variance_floor))
    gated=mode in ('bayes_gate','bayes_both','frequentist_gate','diagonal_posterior_gate')
    lower=gain-conservative_cost-(float(q_issued)*sd if gated else 0.)
    score=lower-float(cfg['threshold']);action=bool(score>0)
    if mode=='periodic':action=True
    if mode=='frozen':action=False
    action=bool(action and feasible)
    return dict(action=action,abstained=False,reason='eligible_score' if feasible else 'infeasible_service_state',
                weights=weights.tolist(),gain=gain,lower=lower,score=score,sd=sd,
                q_issued=float(q_issued),certificate=cert)
