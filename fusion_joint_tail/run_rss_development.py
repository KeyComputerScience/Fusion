"""Known RSS development replay of unchanged CJRT; not a new physical test."""
import datetime
import run_replay as api
ROOT=api.ROOT/'rss_development';ROOT.mkdir(exist_ok=True)
ARM=api.ARMS

def prepared():return api.db.prepared('rss348')

def calibrate():
    assert not (ROOT/'selection.json').exists()
    sources=[api.ROOT/x for x in ('tail_fusion.py','run_replay.py')]+[api.Path(__file__)]
    api.dump(ROOT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='known RSS physical records; unchanged frozen CJRT numerical functions; development only',
        source_hashes={str(x):api.sha(x) for x in sources},grid=dict(bandwidth=api.BANDS,cap=api.CAPS),
        calibration_seeds=[87001,87002,87003],test_seeds=[88001,88002,88003,88004,88005],
        selection='same nine trials perarm; firsthalf matured ready alpha fit; secondhalf guarded actualreturn; tie smaller cap then largerbandwidth'))
    engine,recovery,guard,cfg0,service,fork,data,pre,records,cache=prepared()
    cfg0=api.p.base_cfg(cfg0);seeds=[87001,87002,87003]
    events=[engine.world(pre['cal_x'],pre['cal_y'],pre['cal_timestamp'],pre,s,cfg0) for s in seeds]
    bases=[engine.precompute(e,pre,cfg0) for e in events];grid=[];trials=[]
    for b in api.BANDS:
        cfg=dict(cfg0,slice_bandwidth=b)
        streams=[api.decorate(api.core.augment(e,pre,base,cfg),pre,cfg) for e,base in zip(events,bases)]
        for cap in api.CAPS:
            initials,count=api.fit(streams,cap)
            for arm in ARM:
                results=[api.run(s,arm,cap,initials[arm],True) for s in streams]
                guarded=[api.db.guarded(e,pre,cfg,r,guard,service,130.) for e,r in zip(events,results)]
                row=dict(arm=arm,bandwidth=b,cap=cap,alpha_initial=initials[arm],fit_count=count,
                    net=sum(g['increment'] for g in guarded)/3,beneficial=sum(g['beneficial'] for g in guarded),harmful=sum(g['harmful'] for g in guarded))
                grid.append(row);trials.append(dict(configuration=row,results=results,guarded=guarded))
        print('CAL',b,flush=True)
    selected={arm:max((x for x in grid if x['arm']==arm),key=lambda x:(x['net'],-x['cap'],x['bandwidth'])) for arm in ARM}
    api.dump(ROOT/'calibration_trials.json.gz',trials);api.dump(ROOT/'selection.json',dict(selected=selected,grid=grid))
    api.dump(ROOT/'selection_lock.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=api.sha(ROOT/'selection.json'),protocol_sha256=api.sha(ROOT/'protocol.json')))
    print('SELECT',selected,flush=True)

def test():
    protocol=api.load(ROOT/'protocol.json')
    for path,want in protocol['source_hashes'].items():assert api.sha(path)==want
    assert api.load(ROOT/'selection_lock.json')['selection_sha256']==api.sha(ROOT/'selection.json')
    assert not (ROOT/'results.json.gz').exists()
    engine,recovery,guard,cfg0,service,fork,data,pre,records,cache=prepared();cfg0=api.p.base_cfg(cfg0)
    selected=api.load(ROOT/'selection.json')['selected'];records=[];issued=[]
    for seed in protocol['test_seeds']:
        events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg0)
        base=engine.precompute(events,pre,cfg0);streams={}
        for arm,s in selected.items():
            cfg=dict(cfg0,slice_bandwidth=s['bandwidth'])
            if s['bandwidth'] not in streams:streams[s['bandwidth']]=api.decorate(api.core.augment(events,pre,base,cfg),pre,cfg)
            result=api.run(streams[s['bandwidth']],arm,s['cap'],s['alpha_initial'])
            issued.append(dict(seed=seed,arm=arm,**result))
            for B in (0.,110.,130.,260.):records.append(dict(seed=seed,arm=arm,budget=B,**api.db.guarded(events,pre,cfg,result,guard,service,B)))
        print('DEV',seed,flush=True)
    summary={}
    for arm in ARM:
        rows=[r for r in records if r['arm']==arm and r['budget']==130.]
        summary[arm]=dict(mean_increment=api.np.mean([r['increment'] for r in rows]),seed_increments=[r['increment'] for r in rows],
            admissions=sum(r['admissions'] for r in rows),beneficial=sum(r['beneficial'] for r in rows),harmful=sum(r['harmful'] for r in rows),
            zero=sum(r['zero'] for r in rows),negative_loss=sum(r['negative_loss'] for r in rows),coverage=[sum(r['coverage'][0] for r in rows),sum(r['coverage'][1] for r in rows)],
            excess=sum(r['optimistic_excess'] for r in rows),refused=sum(r['refused'] for r in rows))
    api.dump(ROOT/'results.json.gz',dict(issued=issued,guarded=records));api.dump(ROOT/'summary.json',dict(summary=summary,
        selected=selected,max_service_error=max(r['service_error'] for r in records),
        max_identity_error=max(abs(r['identity_error']) for r in issued),cache=cache.report()))
    print('SUMMARY',summary,flush=True)

if __name__=='__main__':
    import sys
    {'calibrate':calibrate,'test':test}[sys.argv[1]]()
