"""Magnitude-aware immutable-score calibration on previously known traces.

The full-return fusion score is an input, and the calibrated score is S-c.
Only matured immutable residuals enter c; neither reprojected archive errors
nor future complete returns enter admission. All configurations are retained.
Finite-binomial reference limits are displayed but are NOT certified risk
bounds for these dependent, adaptively fitted physical traces.
"""
from pathlib import Path
import datetime,gzip,hashlib,importlib.util,json,math,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
SOURCE=PROJECT/'work/fusion_revision_20261005/calibration/selective_offset.py'
spec=importlib.util.spec_from_file_location('known_selected_offset',SOURCE)
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
PROBS=(.8,.9,.95)
WINDOWS=(16,32,48)
ARMS=old.ARMS
def dump(p,o):
    with (gzip.open(p,'wt') if str(p).endswith('.gz') else Path(p).open('w')) as f:json.dump(o,f,indent=2,allow_nan=False)
def quantile(values,p):
    if not values:return 0.
    vals=sorted(float(x) for x in values)
    return max(0.,vals[min(len(vals)-1,math.ceil(p*len(vals))-1)])
class PrequentialResidualQuantile:
    def __init__(self,initial=(),probability=.9,window=48):
        if not 0<probability<1 or window<1:raise ValueError('Invalid residual quantile specification')
        self.probability=float(probability);self.window=int(window)
        self.values=list(map(float,initial))[-self.window:]
    def offset(self):return quantile(self.values,self.probability)
    def callback(self,raw_issued_score,complete_target):
        if not math.isfinite(raw_issued_score) or not math.isfinite(complete_target):raise ValueError('Nonfinite callback')
        self.values.append(float(raw_issued_score)-float(complete_target));self.values=self.values[-self.window:]
    def issue(self,raw_score):return float(raw_score)-self.offset()
def binomial_upper(v,n,error=.05):
    if n==0 or v>=n:return 1.
    if not 0<=v<=n or not 0<error<1:raise ValueError('Invalid binomial reference inputs')
    def cdf(p):return sum(math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(v+1))
    lo=0.;hi=1.
    for _ in range(90):
        mid=(lo+hi)/2
        if cdf(mid)>error:lo=mid
        else:hi=mid
    return hi
def first_half_residuals(row_lists,windows):
    return [r['gate_score']-(r['truegross']-5) for rows in row_lists for r in rows
            if r['maturity']<=windows//2 and r.get('information_ready',False)]
def replay(rows,windows,initial,prob,window,select=False,population='ready',budget=130.,poison_after=None):
    state=PrequentialResidualQuantile(initial,prob,window)
    scheduler=[];out=[];callbacks=[];ledger=[]
    g=old.guard.DelayedLossBudgetGuard(float(budget),'gross_loss')
    by={r['k']:r for r in rows};boundary=windows//2 if select else 0
    def mature(now):
        nonlocal scheduler
        due=sorted((x for x in scheduler if x['maturity']<=now),key=lambda x:(x['maturity'],x['k']))
        scheduler=[x for x in scheduler if x['maturity']>now]
        actual=[]
        for x in due:
            if x['learn']:
                before=state.offset();state.callback(x['S'],x['X'])
                callbacks.append(dict(k=x['k'],maturity=x['maturity'],raw_issued_score=x['S'],complete_target=x['X'],before=before,after=state.offset(),action=x['action']))
            if x['action']:actual.append((x['k'],x['maturity'],x['D']))
        return actual
    for k in range(windows+1):
        due=mature(k);r=by.get(k);offset=state.offset();score=state.issue(r['gate_score']) if r else None
        proposal=bool(r and k>=boundary and r.get('information_ready',False) and score>0)
        action,log=g.step(k,proposal,due,maximum_loss=130.);ledger.append(log)
        if r:
            X=float(r['truegross']-5);D=float(r['local_net']);S=float(r['gate_score'])
            new=dict(r,action=action,proposed_action=proposal,raw_gate_score=S,offset_issued=offset,gate_score=score,
                     lower_covered=X>=score,standardized_score=score-X)
            out.append(new)
            # First-half calibration residuals are already supplied once as
            # initial fitting information. During selection learn only new
            # origins at/after that split; earlier proposals remain disabled.
            learn=bool(k>=boundary and r.get('information_ready',False) and (population=='ready' or action))
            if action or learn:
                if poison_after is not None and r['maturity']>poison_after:X=S+256.;D=-130.
                scheduler.append(dict(k=k,maturity=r['maturity'],S=S,X=X,D=D,action=action,learn=learn))
    end_pending=[x for x in scheduler if x['action']]
    for t in sorted({x['maturity'] for x in scheduler}):
        _,log=g.step(t,False,mature(t),maximum_loss=130.);ledger.append(log)
    assert not g.pending and not scheduler
    inc=sum(r['local_net'] for r in out if r['action'])
    if poison_after is None:assert abs(inc-g.settled_increment)<1e-8
    assert all(x['spent_loss']+x['reserved']<=budget+1e-8 for x in ledger)
    return dict(rows=out,callbacks=callbacks,ledger=ledger,increment=inc,initial_offset=quantile(initial[-window:],prob),final_offset=state.offset(),at_end_pending=len(end_pending))
def summarize(results,seeds):
    acts=[r for v in results for r in v['rows'] if r['action']]
    vals=[sum(v['increment'] for v in results if v['seed']==seed) for seed in seeds]
    excess=[max(0.,r['standardized_score']) for r in acts];v=sum(not r['lower_covered'] for r in acts)
    return dict(mean_increment=sum(vals)/len(vals),seed_increments=vals,admissions=len(acts),beneficial=sum(r['local_net']>0 for r in acts),harmful=sum(r['local_net']<0 for r in acts),negative_loss=sum(max(0.,-r['local_net']) for r in acts),admitted_coverage=[len(acts)-v,len(acts)],max_excess=max(excess,default=0.),sum_excess=sum(excess),binomial_reference_upper=binomial_upper(v,len(acts),.05),reference_scope='numerical IID binomial feasibility only; repeated physical records and adaptive scores do not satisfy its assumptions')
def main():
    oldroot=old.OLD
    assert not (ROOT/'protocol.json').exists()
    dump(ROOT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),role='known-data development; original outcomes already known',source_sha256=old.sha(__file__),score='unchanged previously selected raw conditional-return score minus magnitude-aware immutable prequential residual quantile',grid=dict(probability=PROBS,window=WINDOWS),feedback_populations=['all ready counterfactual matured scores','actually admitted matured scores'],first_half='all matured ready origins from original calibration first half; immutable issued raw scores',selection='guarded second-half mean paid increment, ties larger probability and longer window',guard='unchanged Fixed130 shared guard',quantile_scope='empirical operating calibration; no unconditional selected90% claim',binomial_reference='candidate actual-admission counts only; numerical IID reference not a valid bound for dependent replay data'))
    selection={};grid=[];results={};summaries={}
    for task,cw,tw,seeds in [('rss',165,168,list(range(88001,88006))),('gas',47,47,list(range(143001,143006)))]:
        if task=='rss':
            selected={**old.load(oldroot/'geometry_joint/selection.json')['selected'],**{k:v for k,v in old.load(oldroot/'strong/selection.json')['selected'].items() if k.startswith('legacy')}}
            trials=old.load(oldroot/'geometry_joint/calibration_trials.json.gz')+old.load(oldroot/'strong/calibration_trials.json.gz')
            test=old.load(oldroot/'geometry_joint/results.json.gz')['issued']+[x for x in old.load(oldroot/'strong/results.json.gz')['issued'] if x['arm'].startswith('legacy')]
        else:
            selected=old.load(oldroot/'physical/selection.json')['selected'];trials=old.load(oldroot/'physical/calibration_trials.json.gz');test=old.load(oldroot/'physical/results.json.gz')['issued']
        results[task]={};summaries[task]={};selection[task]={}
        for arm in ARMS:
            trial=old.selected_trial(trials,selected,arm,task);calrows=[x['rows'] for x in trial['issued']];initial=first_half_residuals(calrows,cw)
            for population in ('ready','admitted'):
                candidates=[]
                for prob in PROBS:
                    for window in WINDOWS:
                        rr=[replay(x,cw,initial,prob,window,True,population) for x in calrows]
                        acts=[r for v in rr for r in v['rows'] if r['action']];fail=sum(not r['lower_covered'] for r in acts)
                        candidates.append(dict(arm=arm,population=population,probability=prob,window=window,initial_fit_count=len(initial),initial_offset=quantile(initial[-window:],prob),calibration_mean_increment=sum(x['increment'] for x in rr)/3,calibration_admissions=len(acts),calibration_violations=fail,selection_adjusted_iid_reference_upper=binomial_upper(fail,len(acts),.05/9),certified=False))
                best=max(candidates,key=lambda x:(x['calibration_mean_increment'],x['probability'],x['window']));name=arm+'_'+population;selection[task][name]=best;grid+=candidates;rs=[]
                for inp in test:
                    if inp['arm']!=arm:continue
                    value=replay(inp['rows'],tw,initial,best['probability'],best['window'],False,population)
                    value.update(arm=arm,population=population,seed=inp['seed'],recording=inp.get('recording'))
                    for cutoff in (0,12,tw//2):
                        bad=replay(inp['rows'],tw,initial,best['probability'],best['window'],False,population,poison_after=cutoff)
                        assert [r['action'] for r in value['rows'] if r['k']<=cutoff]==[r['action'] for r in bad['rows'] if r['k']<=cutoff]
                    rs.append(value)
                results[task][name]=rs;summaries[task][name]=summarize(rs,seeds)
    dump(ROOT/'selection.json',dict(selected=selection,grid=grid));dump(ROOT/'results.json.gz',results);dump(ROOT/'summary.json',summaries)
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
