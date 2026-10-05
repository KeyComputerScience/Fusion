"""Known RSS same-issuance information interventions; no retuning."""
import datetime
import run_geometry as kernel
a=kernel.api
ROOT=kernel.ROOT/'fixed_information';ROOT.mkdir(exist_ok=True)

def main():
    assert not (ROOT/'results.json').exists()
    a.dump(ROOT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=a.sha(__file__),
        scope='known-data information intervention after development outcomes; all contrasts retained',
        common='full selected bandwidth and original issued alpha/readiness; same error-law information inputs and new chronological common130 guard',
        arms=a.ARMS,selection='none'))
    kernel.verify();saved=kernel.load('results.json.gz');issued={(r['seed'],r['arm']):r for r in saved['issued']}
    selected=kernel.load('selection.json')['selected']['joint']
    engine,recovery,guard,cfg0,service,fork,data,pre,records,cache=a.db.prepared('rss348')
    cfg=dict(a.p.base_cfg(cfg0),slice_bandwidth=selected['b_perp'],b_parallel=selected['b_parallel'],b_perp=selected['b_perp']);guarded=[];predictions=[]
    for seed in range(88001,88006):
        events=engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,seed,cfg)
        stream=a.decorate(a.core.augment(events,pre,engine.precompute(events,pre,cfg),cfg),pre,cfg)
        for arm in a.ARMS:
            rows=[]
            for d,old in zip(stream['decisions'],issued[(seed,'joint')]['rows']):
                score=a.paid_score(d['tail_laws'][arm],d['N'],old['alpha_issued'])['score']
                row=dict(old,gate_score=score,gain=score+5,standardized_score=score-(d['truegross']-5),
                    lower_covered=bool(score<=d['truegross']-5),action=bool(old['information_ready'] and score>0))
                rows.append(row);predictions.append(dict(seed=seed,arm=arm,k=d['k'],D=d['localnet'],score=score,proposal=row['action'],
                    N=d['N'],alpha=old['alpha_issued'],mean=d['tail_laws'][arm].mean,variance=d['tail_laws'][arm].variance))
            guarded.append(dict(seed=seed,arm=arm,**a.db.guarded(events,pre,cfg,dict(rows=rows),guard,service,130.)))
    full={r['seed']:r for r in guarded if r['arm']=='joint'};summary={};changed={}
    for arm in a.ARMS:
        rows=[r for r in guarded if r['arm']==arm];values=[r['increment'] for r in rows]
        summary[arm]=dict(mean_increment=float(a.np.mean(values)),values=values,
            admissions=sum(r['admissions'] for r in rows),beneficial=sum(r['beneficial'] for r in rows),harmful=sum(r['harmful'] for r in rows),
            coverage=[sum(r['coverage'][0] for r in rows),sum(r['coverage'][1] for r in rows)])
        changed[arm]=[dict(seed=r['seed'],k=x['k'],D=x['local_net'],full_action=f['action'],control_action=x['action'],full_score=f['gate_score'],control_score=x['gate_score'])
            for r in rows for x,f in zip(r['rows'],full[r['seed']]['rows']) if x['action']!=f['action']]
    a.dump(ROOT/'results.json',dict(summary=summary,changed=changed,predictions=predictions,guarded=guarded))
    print('FIXED',summary,changed,flush=True)

if __name__=='__main__':main()
