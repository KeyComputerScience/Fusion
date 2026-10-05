"""New conditional-tail rule, frozen Daphnet protocol, all results retained."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,importlib.util,json,math,sys
sys.dont_write_bytecode=True
import numpy as np
from tail_fusion import joint_line,paid_score,DiscreteLaw
import physical_adapter as adapter
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

db=module('joint_tail_frozen_environment',PROJECT/'work/fusion_coupling_focus_20261005/dbf/run_dbf.py')
p=db.p;c=p.c;core=c.core
ARMS=('joint','marginal','factorized','gaussian','sandwich')
SEEDS={'calibration':[132001,132002,132003],'test':[133001,133002,133003,133004,133005]}
BANDS=(.25,.5,1.);CAPS=(.1,.5,1.)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:return json.load(f)
def dump(path,obj):
    with (gzip.open(path,'wt') if str(path).endswith('.gz') else Path(path).open('w')) as f:
        json.dump(obj,f,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())

def freeze():
    assert not (ROOT/'protocol.json').exists() and not (ROOT/'raw/daphnet245.zip').exists()
    p.verify();files=set(p.scientific_sources())|{ROOT/'tail_fusion.py',ROOT/'run_replay.py',ROOT/'physical_adapter.py',ROOT/'verify_tail.py'}
    for obj in list(sys.modules.values()):
        name=getattr(obj,'__file__',None)
        if name and str(name).startswith(str(PROJECT)) and str(name).endswith('.py'):files.add(Path(name))
    dump(ROOT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='frozen before Daphnet acquisition; RSS historical evidence remains separate',source_hashes={str(x):sha(x) for x in sorted(files)},
        method='one conditional complete-return lower-tail functional; original joint error construction; no source weight optimizer',
        dataset='UCI245 Daphnet; official archive not read before freeze',fit=adapter.FIT,calibration=adapter.CAL,test=adapter.TEST,
        stride=64,sources=adapter.GROUPS,service=dict(window=32,horizon=4,action_fees=3,max_extra_drops=2,cost_lower_target=5,reserve=130,budget=130),
        arms=ARMS,grid=dict(bandwidth=BANDS,tail_cap=CAPS,configs=9),seeds=SEEDS,
        initial_alpha='largest alpha in[1e-4,cap] with empirical firsthalf fullymatured ready lower-score violation <=10percent; binary search60; no firsthalf score if empty ->1e-4',
        online='ready issued lower-score violations only; at verified complete maturity alpha=clip(alpha+.02*(.1-violation),1e-4,cap); selection phase updates secondhalf only',
        select='guarded secondhalf actual net, pooled /3 calibration delays; ties smaller cap then larger bandwidth',
        radius='zero in measured rule; no conditional W1 certificate inferred from Gaussian posterior',
        scope='one fresh physical collection, three heldout participants; five delay replays perparticipant are dependent',
        output='all configurations/actions/coverage/excess/ledger/results, including ties/losses, immutable original evidence',
        runtime=dict(python=sys.version,numpy=np.__version__)))
    dump(ROOT/'pre_acquisition_freeze.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),raw_exists=False))
    print('FROZEN',sha(ROOT/'protocol.json'),flush=True)

def verify():
    pr=load(ROOT/'protocol.json');assert sha(ROOT/'protocol.json')==load(ROOT/'pre_acquisition_freeze.json')['protocol_sha256']
    for path,want in pr['source_hashes'].items():assert sha(path)==want,path
    return pr

def prepared():
    verify();engine,recovery,guard,source,cfg,study,service,fork=p.r.binding();cfg=p.base_cfg(cfg)
    cache=db.ExactGradientCache(engine);engine.gradient=cache
    pre,meta=adapter.prepared(ROOT/'data',engine,cfg)
    return engine,guard,cfg,service,pre,meta,cache

def law(d,pre,cfg,arm):
    mm=d['temporal'];factor=arm=='factorized'
    a,l,V,mu,C,jitter=c.law(mm,pre,d['ids'],cfg,factor)
    if arm=='gaussian':return joint_line(d['h'],[1.],mu[None],C[None])
    if arm=='sandwich':return joint_line(d['h'],[1.],mu[None],(mm['paired_atoms']['C']+mm['sandwich_P']+jitter)[None])
    if arm=='joint':return joint_line(d['h'],a,l,V)
    variances=np.diagonal(V,axis1=1,axis2=2);bins=max(32,int(math.ceil(1/math.sqrt(float(1/np.sum(1/variances.min(0)))))))
    def evaluate(bins):
        u,qw=c.quadrature(bins);logs=np.zeros(len(u))
        for s in range(len(d['h'])):
            r=d['h'][s]-u[:,None]-l[None,:,s]
            logs+=c.logsumexp(np.log(a)[None,:]-.5*(c.LOG2PI+np.log(variances[:,s])[None,:]+r*r/variances[:,s][None,:]),axis=1)
        logs+=np.log(qw);prob=np.exp(logs-c.logsumexp(logs));return DiscreteLaw(u,prob)
    old=evaluate(bins)
    for _ in range(5):
        bins*=2;new=evaluate(bins)
        error=max(abs(old.lower_tail(alpha)-new.lower_tail(alpha)) for alpha in (.01,.1,.5,1.))
        if error<1e-6:break
        old=new
    # Quadrature is a disclosed numerical approximation, not a coverage radius.
    new.quadrature_error=error;return new

def decorate(stream,pre,cfg):
    for d in stream['decisions']:
        d['tail_laws']={arm:law(d,pre,cfg,arm) for arm in ARMS}
    return stream

def fit(streams,cap):
    eligible=[d for s in streams for d in s['decisions'] if d['maturity']<=s['windows']//2 and d['temporal']['eligible']]
    out={}
    for arm in ARMS:
        if not eligible:out[arm]=1e-4;continue
        left,right=1e-4,cap
        for _ in range(35):
            mid=(left+right)/2
            violation=np.mean([d['truegross']<d['N']*d['tail_laws'][arm].lower_tail(mid) for d in eligible])
            if violation<=.1:left=mid
            else:right=mid
        out[arm]=left
    return out,len(eligible)

def run(stream,arm,cap,alpha0,selection=False):
    alpha=alpha0;pending=[];updates=[];rows=[];boundary=stream['windows']//2 if selection else 0
    def mature(now):
        nonlocal alpha,pending
        for due,k,issued,v in sorted(pending,key=lambda x:(x[0],x[1])):
            if due<=now:
                before=alpha;raw=alpha+.02*(.1-v);alpha=float(np.clip(raw,1e-4,cap))
                updates.append(dict(maturity=due,k=k,alpha_issued=issued,violation=v,before=before,after=alpha,regulator=alpha-raw))
        pending=[x for x in pending if x[0]>now]
    for d in stream['decisions']:
        k=d['k'];mature(k);ready=bool(d['temporal']['eligible'])
        forecast=paid_score(d['tail_laws'][arm],d['N'],alpha)
        score=forecast['score'];proposal=bool(ready and score>0 and (not selection or k>=boundary))
        # Truth is read only after the issuance functional; future poison check.
        poison=dict(d,truegross=1e50,localnet=-1e50)
        assert paid_score(poison['tail_laws'][arm],poison['N'],alpha)==forecast
        covered=bool(d['truegross']-5>=score)
        rows.append(dict(k=k,action=proposal,maturity=d['maturity'],local_net=d['localnet'],truegross=d['truegross'],
            gate_score=score,gain=score+5,posterior_or_block_sd=1.,q_issued=0.,standardized_score=score-(d['truegross']-5),
            lower_covered=covered,information_ready=ready,alpha_issued=alpha,disagreement=d['disagreement'],
            conditional_paid_mean=forecast['mean'],conditional_paid_variance=forecast['variance'],active_source_ids=d['ids'].tolist()))
        if ready and (not selection or k>=boundary):pending.append((d['maturity'],k,alpha,int(not covered)))
    mature(stream['windows'])
    v=sum(x['violation'] for x in updates);reg=sum(x['regulator'] for x in updates)
    identity=v-(.1*len(updates)-(alpha-alpha0-reg)/.02)
    assert abs(identity)<1e-8
    return dict(rows=rows,alpha_updates=updates,alpha_initial=alpha0,alpha_final=alpha,identity_error=identity)

def worlds(part):
    engine,guard,cfg,service,pre,meta,cache=prepared();records=[r for r in pre['recording_streams'] if r['partition']==part]
    data=[]
    for record in records:
        for seed in SEEDS[part]:
            events=engine.world(record['x'],record['y'],record['timestamp'],pre,seed,cfg)
            base=engine.precompute(events,pre,cfg);data.append((record,seed,events,base))
    return engine,guard,cfg,service,pre,meta,cache,data

def calibrate():
    verify();assert not (ROOT/'selection.json').exists()
    engine,guard,cfg0,service,pre,meta,cache,data=worlds('calibration');grid=[];trials=[]
    for bandwidth in BANDS:
        cfg=dict(cfg0,slice_bandwidth=bandwidth)
        streams=[decorate(core.augment(e,pre,base,cfg),pre,cfg) for _,_,e,base in data]
        for cap in CAPS:
            initials,count=fit(streams,cap)
            for arm in ARMS:
                rr=[run(s,arm,cap,initials[arm],True) for s in streams]
                gg=[db.guarded(e,pre,cfg,r,guard,service,130.) for (_,_,e,_),r in zip(data,rr)]
                row=dict(arm=arm,bandwidth=bandwidth,cap=cap,alpha_initial=initials[arm],fit_count=count,
                    net=sum(g['increment'] for g in gg)/3,beneficial=sum(g['beneficial'] for g in gg),harmful=sum(g['harmful'] for g in gg))
                grid.append(row);trials.append(dict(configuration=row,results=rr,guarded=gg))
        print('CAL',bandwidth,flush=True)
    selected={arm:max((x for x in grid if x['arm']==arm),key=lambda x:(x['net'],-x['cap'],x['bandwidth'])) for arm in ARMS}
    dump(ROOT/'calibration_trials.json.gz',trials);dump(ROOT/'selection.json',dict(selected=selected,grid=grid,metadata=meta,split=pre['split']))
    dump(ROOT/'selection_lock.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),selection_sha256=sha(ROOT/'selection.json')))
    print('SELECT',selected,flush=True)

def test():
    verify();lock=load(ROOT/'selection_lock.json');assert lock['selection_sha256']==sha(ROOT/'selection.json')
    assert not (ROOT/'physical_results.json.gz').exists()
    selected=load(ROOT/'selection.json')['selected'];engine,guard,cfg0,service,pre,meta,cache,data=worlds('test')
    records=[];issued=[]
    for record,seed,events,base in data:
        streams={}
        for arm,s in selected.items():
            cfg=dict(cfg0,slice_bandwidth=s['bandwidth'])
            if s['bandwidth'] not in streams:streams[s['bandwidth']]=decorate(core.augment(events,pre,base,cfg),pre,cfg)
            result=run(streams[s['bandwidth']],arm,s['cap'],s['alpha_initial'])
            issued.append(dict(recording=record['recording'],person=record['person'],seed=seed,arm=arm,**result))
            for B in (0.,110.,130.,260.):
                records.append(dict(recording=record['recording'],person=record['person'],seed=seed,arm=arm,budget=B,
                    **db.guarded(events,pre,cfg,result,guard,service,B)))
        print('TEST',record['recording'],seed,flush=True)
    summary={}
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm and r['budget']==130.]
        values=[sum(r['increment'] for r in rows if r['seed']==seed) for seed in SEEDS['test']]
        summary[arm]=dict(seed_increments=values,mean_increment=float(np.mean(values)),admissions=sum(r['admissions'] for r in rows),
            beneficial=sum(r['beneficial'] for r in rows),harmful=sum(r['harmful'] for r in rows),zero=sum(r['zero'] for r in rows),
            negative_loss=sum(r['negative_loss'] for r in rows),coverage=[sum(r['coverage'][0] for r in rows),sum(r['coverage'][1] for r in rows)],
            excess=sum(r['optimistic_excess'] for r in rows),refused=sum(r['refused'] for r in rows))
    dump(ROOT/'physical_results.json.gz',dict(issued=issued,guarded=records))
    dump(ROOT/'physical_summary.json',dict(summary=summary,metadata=meta,split=pre['split'],selected=selected,
        controller_trajectories=len(issued),budget_trajectories=len(records),max_service_error=max(r['service_error'] for r in records),
        max_calibration_identity=max(abs(r['identity_error']) for r in issued),cache=cache.report()))
    print('SUMMARY',summary,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','verify','ingest','calibrate','test']);a=ap.parse_args()
    if a.phase=='ingest':verify();print(adapter.process(ROOT/'raw/daphnet245.zip',ROOT/'data'))
    else:{'freeze':freeze,'verify':lambda:print(verify()['status']),'calibrate':calibrate,'test':test}[a.phase]()
