"""V2: one previsible-stratified prequential calibration of a paid score.

Only immutable raw scores and contexts at issuance are stored. Complete
counterfactual targets are received after maturity, including rejected
leases. The same nine configurations serve every fusion information arm.
The empirical local quantile is not itself a conditional-risk certificate.
All complete feedback is retained. The quantile query uses only records
matching the current previsible binary disagreement stratum. With no such
record, correction is two and the score is the lower physical support.
This certifies only the support fallback, not empirical nonempty strata.
"""
from dataclasses import dataclass
import math
import numpy as np

HISTORY_LIMIT=48
AGE_DISCOUNT=.97
CALIBRATION_PRIOR_MASS=2.
BANDWIDTHS=(.25,.5,1.)
PROBABILITIES=(.8,.9,.95)
def configurations():
    return [dict(bandwidth=b,probability=p) for b in BANDWIDTHS for p in PROBABILITIES]

def weighted_quantile(values,weights,probability):
    x=np.asarray(values,dtype=float);w=np.asarray(weights,dtype=float)
    if x.ndim!=1 or w.shape!=x.shape or len(x)==0:raise ValueError('Nonempty one-dimensional weighted sample required')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)) or np.any(w<0) or w.sum()<=0:raise ValueError('Invalid weighted calibration values')
    if not 0<probability<1:raise ValueError('Quantile probability must lie in(0,1)')
    order=np.argsort(x,kind='stable');c=np.cumsum(w[order]);idx=min(len(x)-1,int(np.searchsorted(c,probability*c[-1],side='left')))
    return float(x[order[idx]])

def checked_context(context):
    x=np.asarray(context,dtype=float)
    if x.shape!=(3,) or not np.all(np.isfinite(x)):raise ValueError('Context must be finite[anchor,disagreement_fraction,active_fraction]')
    if not -1.-1e-10<=x[0]<=1.+1e-10 or np.any(x[1:]< -1e-10) or np.any(x[1:]>1.+1e-10):raise ValueError('Context outside declared support')
    return tuple(map(float,x))

@dataclass(frozen=True)
class ImmutableIssue:
    origin:int
    maturity:int
    N:float
    raw_score:float
    context:tuple
    eligible:bool
    def __post_init__(self):
        if not isinstance(self.origin,(int,np.integer)) or not isinstance(self.maturity,(int,np.integer)):
            raise ValueError('Origin and maturity must be integer window timestamps')
        if self.maturity<=self.origin:raise ValueError('Complete feedback must mature after issuance')
        if not 0<self.N<=128. or not math.isfinite(self.raw_score):raise ValueError('Invalid complete lease support/score')
        if not -self.N-5.-1e-7<=self.raw_score<=self.N-5.+1e-7:raise ValueError('Raw conditional score outside physical support')
        checked_context(self.context)

class ConditionalScoreCalibration:
    def __init__(self,config,initial_pool=()):
        if config not in configurations():raise ValueError('Configuration outside frozen nine-trial grid')
        self.config=dict(config);self.prior=[];self.history=[]
        for item in initial_pool:
            if not math.isfinite(item['residual']):raise ValueError('Nonfinite calibration residual')
            if item.get('source_stage') in ('selection','test','online_test'):
                raise ValueError('Initial residual pool cannot contain selection/test outcomes')
            context=checked_context(item['context'])
            self.prior.append(dict(item,context=context))
        self.issued={};self.completed=set()

    def correction(self,context):
        z=np.asarray(checked_context(context));stratum=int(z[1]>0.)
        prior=[r for r in self.prior if int(r['context'][1]>0.)==stratum]
        online=[(age,r) for age,r in enumerate(reversed(self.history)) if int(r['context'][1]>0.)==stratum]
        online=list(reversed(online));npast=len(prior);nonline=len(online);records=prior+[r for _,r in online];n=len(records)
        if n==0:return dict(correction=2.,history_count=0,prior_history_count=0,online_history_count=0,effective_count=0.,weight_mass=0.,certificate=False,calibration_stratum=stratum,empty_stratum_fallback=True,retained_prior_history_count=len(self.prior),retained_online_history_count=len(self.history))
        zs=np.asarray([h['context'] for h in records]);distance=np.sum((zs-z[None,:])**2,axis=1)
        logs=-.5*distance/self.config['bandwidth']**2
        if npast:logs[:npast]+=math.log(CALIBRATION_PRIOR_MASS/npast)
        # Preserve actual callback-arrival age, including callbacks retained
        # in the other stratum. Query filtering cannot rejuvenate history.
        if nonline:logs[npast:]+=np.asarray([age for age,_ in online])*math.log(AGE_DISCOUNT)
        # Rescaling avoids underflow and preserves the weighted quantile.
        weights=np.exp(logs-logs.max());values=[h['residual'] for h in records]
        q=weighted_quantile(values,weights,self.config['probability'])
        scaled_mass=float(weights.sum());effective=scaled_mass*scaled_mass/float(weights@weights);mass=scaled_mass*math.exp(float(logs.max()))
        return dict(correction=q,history_count=n,prior_history_count=npast,online_history_count=nonline,effective_count=effective,weight_mass=mass,certificate=False,calibration_stratum=stratum,empty_stratum_fallback=False,retained_prior_history_count=len(self.prior),retained_online_history_count=len(self.history))

    def issue(self,record):
        if record.origin in self.issued:raise ValueError('Duplicate immutable issue')
        self.issued[record.origin]=record
        local=self.correction(record.context)
        score=min(record.N-5.,max(-record.N-5.,record.raw_score-record.N*local['correction']))
        if local['empty_stratum_fallback']:score=-record.N-5.
        return dict(score=float(score),**local)

    def callback(self,now,origin,complete_target):
        if origin not in self.issued:raise ValueError('Unknown immutable score origin')
        r=self.issued[origin]
        if r.maturity>now:raise ValueError('Immature complete feedback')
        if origin in self.completed:raise ValueError('Duplicate complete callback')
        X=float(complete_target)
        if not math.isfinite(X) or not -r.N-5.-1e-10<=X<=r.N-5.+1e-10:raise ValueError('Complete target outside recorded lease support')
        self.completed.add(origin)
        if r.eligible:
            e=(r.raw_score-X)/r.N
            self.history.append(dict(context=r.context,residual=float(e),origin=origin,maturity=r.maturity,raw_score=r.raw_score,N=r.N,partition='online_mature_counterfactual'))
            self.history=self.history[-HISTORY_LIMIT:]
            return dict(origin=origin,maturity=r.maturity,raw_score=r.raw_score,complete_target=X,normalized_residual=float(e),context=list(r.context),online_history_count=len(self.history),prior_history_count=len(self.prior))
        return None

def complete_initial_pool(stream,raw_rows,boundary=None,provenance=None):
    """All fully matured eligible raw forecasts from an earlier fit block.

    A calibration-selection first-half fit uses boundary=windows//2.
    Separate residual-fitting sessions use boundary=None, which includes
    only feedback matured by that session's end. A explicitly completed
    post-session flush must supply its declared deadline as boundary. The
    caller must retain dataset/session IDs and freeze their provenance.
    """
    by={r['k']:r for r in raw_rows};out=[];provenance=dict(provenance or {});deadline=stream['windows'] if boundary is None else boundary
    if len(by)!=len(raw_rows):raise ValueError('Duplicate raw score origins')
    if provenance.get('stage','score_fit') not in ('score_fit','known_development_calibration'):
        raise ValueError('Initial pool must come from the declared completed score-fitting partition')
    for d in stream['decisions']:
        r=by[d['k']]
        if not r.get('information_ready',False):continue
        if d['maturity']>deadline:continue
        out.append(dict(context=checked_context(d['cal_context']),residual=(r['gate_score']-(r['truegross']-5.))/d['N'],origin=d['k'],maturity=d['maturity'],raw_score=r['gate_score'],N=d['N'],partition='completed_calibration_fit',source_stage=provenance.get('stage','score_fit'),stream_id=stream.get('recording'),provenance=dict(provenance)))
    return out

def calibrated_run(stream,raw_rows,config,initial_pool=(),selection=False,poison_after=None,selection_boundary=None):
    """Offline scheduler of the live issue/callback API, with no extra gate.

    `action` here is a proposal; caller must apply the common paid ledger.
    Every counterfactual eligible complete callback calibrates the score,
    independently of proposal and actual admission. Reporting attaches
    complete outcomes only after the causal score replay has finished.
    `selection=True` labels a separately declared whole selection session;
    it does NOT split that session in half. An explicitly requested legacy
    within-session split uses selection_boundary, including its causal
    initialization semantics, and must not replace a dedicated score-fit
    stage with outcomes from selection/test participants.
    """
    if len(raw_rows)!=len(stream['decisions']):raise ValueError('Unaligned raw score stream')
    raw={r['k']:r for r in raw_rows};decisions={d['k']:d for d in stream['decisions']}
    if len(raw)!=len(raw_rows) or len(decisions)!=len(stream['decisions']):raise ValueError('Duplicate decision/score origins')
    if set(raw)!=set(decisions):raise ValueError('Unaligned raw score origins')
    boundary=int(selection_boundary) if selection_boundary is not None else 0
    if not 0<=boundary<=stream['windows']:raise ValueError('Invalid explicit selection boundary')
    state=ConditionalScoreCalibration(config,initial_pool if boundary==0 else ())
    queue=[];issued=[];callbacks=[];fit_installed=boundary==0
    def mature(now):
        nonlocal queue
        due=sorted((x for x in queue if x[0]<=now),key=lambda x:(x[0],x[1]));queue=[x for x in queue if x[0]>now]
        for maturity,origin in due:
            X=raw[origin]['truegross']-5.
            if poison_after is not None and maturity>poison_after:X=-decisions[origin]['N']-5.
            # Fitting data already installed at boundary are not duplicated.
            if boundary>0 and maturity<=boundary:continue
            result=state.callback(now,origin,X)
            if result is not None:callbacks.append(result)
    for k in range(stream['windows']+1):
        if boundary>0 and not fit_installed and k>=boundary:
            # Keep issued tokens for first-half origins that have not yet
            # matured; their future complete callbacks remain causal.
            old=state;state=ConditionalScoreCalibration(config,initial_pool)
            state.issued=dict(old.issued);state.completed=set(old.completed);fit_installed=True
        mature(k)
        if k not in decisions:continue
        d=decisions[k];r=raw[k]
        if not float(d['maturity']).is_integer() or not float(r['maturity']).is_integer() or int(r['maturity'])!=int(d['maturity']):raise ValueError('Unaligned or fractional maturity')
        rec=ImmutableIssue(k,int(d['maturity']),float(d['N']),float(r['gate_score']),checked_context(d['cal_context']),bool(r.get('information_ready',False)))
        value=state.issue(rec)
        new={key:v for key,v in r.items() if key not in ('truegross','local_net','lower_covered','standardized_score')}
        new.update(action=bool(rec.eligible and k>=boundary and value['score']>0),raw_gate_score=rec.raw_score,gate_score=value['score'],gain=value['score']+5.,N=rec.N,cal_context=list(rec.context),calibration_correction=value['correction'],calibration_history_count=value['history_count'],calibration_prior_history_count=value['prior_history_count'],calibration_online_history_count=value['online_history_count'],calibration_effective_count=value['effective_count'],calibration_weight_mass=value['weight_mass'],calibration_stratum=value['calibration_stratum'],calibration_empty_stratum_fallback=value['empty_stratum_fallback'],calibration_retained_prior_history_count=value['retained_prior_history_count'],calibration_retained_online_history_count=value['retained_online_history_count'],risk_certificate=False)
        issued.append(new);queue.append((rec.maturity,k))
    pending_at_end=len(queue)
    for t in sorted({x[0] for x in queue}):mature(t)
    if queue:raise AssertionError('Unprocessed callbacks')
    # Offline truth is used only for reporting after all causal scores and
    # actions have been fixed. It is absent from the live calibration API.
    for r in issued:
        source=raw[r['k']];X=source['truegross']-5.
        r.update(truegross=source['truegross'],local_net=source['local_net'],lower_covered=bool(X>=r['gate_score']),standardized_score=r['gate_score']-X)
    return dict(rows=issued,calibration_callbacks=callbacks,pending_at_test_end=pending_at_end,calibration_config=dict(config),initial_pool_count=len(initial_pool),initial_pool_prior_mass=CALIBRATION_PRIOR_MASS,selection_role=bool(selection),selection_boundary=boundary,calibration_role='V2 previsible disagreement-stratified weighted empirical correction; empty stratum q=2 gives physical lower support; nonempty strata have no automatic conditional certificate')

def actual_admission_report(guarded_rows,ledger=None):
    for r in guarded_rows:
        if not {'k','action','proposed_action','local_net','truegross','gate_score','lower_covered','calibration_effective_count'}<=set(r):
            raise ValueError('Expected actual guard.replay row schema, including proposed_action')
    if ledger is not None:
        for x in ledger:
            if not {'k','spent_loss','reserved','proposal','action'}<=set(x):raise ValueError('Missing actual shared-guard ledger fields')
            if float(x['spent_loss'])<0 or float(x['reserved'])<0:raise ValueError('Negative ledger loss/reserve')
    groups=dict(issued=list(guarded_rows),ready=[r for r in guarded_rows if r.get('information_ready',False)],informative=[r for r in guarded_rows if r.get('disagreement',0)>0],admitted=[r for r in guarded_rows if r['action']])
    a=groups['admitted'];excess=[max(0.,r['gate_score']-(r['truegross']-5.)) for r in a]
    refused=sum(r['proposed_action'] and not r['action'] for r in guarded_rows)
    if ledger is not None and refused!=sum(x['proposal'] and not x['action'] for x in ledger):raise ValueError('Guard row/ledger refusal disagreement')
    empty=[r for r in guarded_rows if r['calibration_empty_stratum_fallback']]
    empty_positive=[r for r in empty if r.get('information_ready',False) and r['raw_gate_score']>0]
    return dict(coverage={key:dict(covered=sum(r['lower_covered'] for r in rows),total=len(rows),rate=sum(r['lower_covered'] for r in rows)/len(rows) if rows else None) for key,rows in groups.items()},admissions=len(a),beneficial=sum(r['local_net']>0 for r in a),harmful=sum(r['local_net']<0 for r in a),zero=sum(r['local_net']==0 for r in a),negative_loss=sum(max(0.,-r['local_net']) for r in a),net_increment=sum(r['local_net'] for r in a),excess_max=max(excess,default=0.),excess_sum=sum(excess),excess_mean=sum(excess)/len(excess) if excess else None,guard_refusals=refused,max_spent_plus_pending=max((float(x['spent_loss'])+float(x['reserved']) for x in ledger),default=None) if ledger is not None else None,ledger_supplied=ledger is not None,calibration_effective_count_min=min((r['calibration_effective_count'] for r in a),default=None),calibration_effective_count_max=max((r['calibration_effective_count'] for r in a),default=None),empty_stratum_issued=len(empty),empty_stratum_positive_raw_refusals=len(empty_positive),empty_stratum_beneficial_opportunities=sum(r['local_net']>0 for r in empty_positive),empty_stratum_harmful_opportunities=sum(r['local_net']<0 for r in empty_positive),empty_stratum_missed_potential_gain=sum(max(0.,r['local_net']) for r in empty_positive))
