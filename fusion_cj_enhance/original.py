"""Final source-bound original CJ; no numerical formula or new component."""
from pathlib import Path
import importlib.util,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[2]
FROZEN=PROJECT/'work/fusion_conditional_20261004'
sp=importlib.util.spec_from_file_location('final_original_cj_copula',FROZEN/'copula_control.py')
cp=importlib.util.module_from_spec(sp);sp.loader.exec_module(cp)
c=cp.c;core=c.core;matrix=c.parent.parent.matrix
MATCHED_ARMS=c.ARMS+('conditional_joint_diagP',cp.ARM)
ARMS=MATCHED_ARMS+('legacy_factorized','legacy_sandwich')
NEW_ARMS=();CAPS=()
def binding():return c.parent.binding()
def configure(cfg,arm):return dict(cfg)
def forecast(d,pre,cfg,arm,q):
    if arm=='legacy_factorized':return c.parent.parent.factorized_forecast(d,pre,cfg,'factorized_information',q)
    if arm=='legacy_sandwich':return matrix.matrix_forecast(d,pre,cfg,'block_sandwich',q)
    return c.conditional_forecast(d,pre,cfg,arm,q)
def initial(streams,pre,cfg,arm):return core.initial_q(streams,pre,cfg,arm)
def run(engine,stream,pre,cfg,arm,initial,selection=False):return core.run(engine,stream,pre,cfg,arm,initial,selection)
def stream_key(cfg):return cfg['slice_bandwidth'],cfg['prior_sd']
def state(arrived,current,pre,cfg):
    mm=cp.state(arrived,current,pre,cfg)
    a,l,V,_,_,_=c.law(mm,pre,current['ids'],cfg)
    V=V-mm['P']+np.diag(np.diag(mm['P']))
    mm['conditional']['conditional_joint_diagP']=c.joint_posterior(current['h'],a,l,V)
    return mm
core.state=state;core.forecast=forecast
