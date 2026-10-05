"""Fully factorized information, including correction mean and precision.

Same joint data and covariance estimator; replace the GLOBAL working R by its
diagonal before conditioning, not only after conditioning in the objective.
"""
from pathlib import Path
import importlib.util, sys
sys.dont_write_bytecode=True
import numpy as np
MATRIX_SOURCE=Path(__file__).resolve().parents[1]/'run_matrix_controls.py'
spec=importlib.util.spec_from_file_location('factorized_parent_matrix',MATRIX_SOURCE)
matrix=importlib.util.module_from_spec(spec);spec.loader.exec_module(matrix)
core=matrix.core
ARM='factorized_information'

def factorized_state(arrived,current,pre,cfg):
    # Full state stays available for within-state mechanism comparisons.
    mm=matrix.matrix_state(arrived,current,pre,cfg)
    observations,complete=matrix.collect_blocks(arrived,current,pre,cfg)
    R,_=core.covariance(complete,pre['m'],cfg['prior_mass'],cfg['prior_variance'])
    diagonal=np.diag(R).copy()
    precision=(pre['q']/np.mean(pre['q']))/cfg['prior_sd']**2
    rhs=np.zeros(pre['m'])
    for block in observations:
        ids=block['ids'];weight=block['weight']
        precision[ids]+=weight/diagonal[ids]
        rhs[ids]+=weight*block['residual']/diagonal[ids]
    mu=rhs/precision;P=1/precision
    ids=current['ids']
    mm.update(factorized_mu=mu[ids],factorized_P=np.diag(P[ids]),
              factorized_R=np.diag(diagonal[ids]),
              factorized_global_precision=precision.tolist(),
              factorized_mean_difference=float(np.max(np.abs(mu[ids]-mm['mu']))),
              factorized_precision_difference=float(np.max(np.abs(np.diag(P[ids])-mm['P']))))
    assert np.min(precision)>0 and np.min(diagonal)>0
    return mm

def factorized_forecast(d,pre,cfg,arm,q):
    if arm!=ARM:return matrix.matrix_forecast(d,pre,cfg,arm,q)
    mm=d['temporal'];changed=dict(mm,mu=mm['factorized_mu'],R=mm['factorized_R'],P=mm['factorized_P'])
    w,cert,F,S,T,extra=matrix.original_forecast(dict(d,temporal=changed),pre,cfg,'precision_joint',q)
    extra.update(information_control=ARM,factorized_mu=mm['factorized_mu'].tolist(),
                 factorized_R=mm['factorized_R'].tolist(),factorized_P=mm['factorized_P'].tolist(),
                 factorized_global_precision=mm['factorized_global_precision'],
                 factorized_mean_difference=mm['factorized_mean_difference'],
                 factorized_precision_difference=mm['factorized_precision_difference'])
    return w,cert,F,S,T,extra

core.state=factorized_state
core.forecast=factorized_forecast
