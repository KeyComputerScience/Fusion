"""Independent complete-fork and budget checks from regenerated physical worlds."""
import numpy as np
import run_replay as a

def audit():
    a.verify();saved=a.load(a.ROOT/'physical_results.json.gz')
    engine,guard,cfg,service,pre,meta,cache=a.prepared()
    recordmap={x['recording']:x for x in pre['recording_streams']}
    issued={}
    for x in saved['issued']:issued.setdefault((x['recording'],x['seed']),[]).append(x)
    _,_,_,_,_,_,_,fork=a.p.r.binding();maxerror=0.;forks=0
    for (name,seed),rules in issued.items():
        r=recordmap[name];events=engine.world(r['x'],r['y'],r['timestamp'],pre,seed,cfg)
        for rule in rules:
            for row in rule['rows']:
                actual=fork(events,row,cfg);forks+=1
                maxerror=max(maxerror,abs(actual['gross']-row['truegross']),abs(actual['actual']-row['local_net']),
                    abs(actual['lower']-row['gate_score']),actual['identity_error'])
                assert actual['covered']==row['lower_covered']
                assert -130<=actual['actual']<=130 and 0<=actual['slack']<=2
    assert maxerror<1e-8
    violations=[]
    for r in saved['guarded']:
        for v in r['ledger']:
            if v['spent_loss']+v['reserved']>r['budget']+1e-8:violations.append(v)
    assert not violations
    rss=a.load(a.ROOT/'rss_development/results.json.gz')
    rssrules={}
    for x in rss['issued']:rssrules.setdefault(x['seed'],[]).append(x)
    rssengine,recovery,rssguard,rsscfg,rssservice,rssfork,data,rsspre,records,rsscache=a.db.prepared('rss348')
    rsscfg=a.p.base_cfg(rsscfg);rss_error=0.;rss_count=0
    for seed,rules in rssrules.items():
        events=rssengine.world(rsspre['test_x'],rsspre['test_y'],rsspre['test_timestamp'],rsspre,seed,rsscfg)
        for rule in rules:
            for row in rule['rows']:
                f=rssfork(events,row,rsscfg);rss_count+=1
                rss_error=max(rss_error,abs(f['gross']-row['truegross']),abs(f['actual']-row['local_net']),abs(f['lower']-row['gate_score']),f['identity_error'])
                assert f['covered']==row['lower_covered']
    assert rss_error<1e-8
    result=dict(passed=True,method_specific_forks=forks,max_complete_target_error=maxerror,
        policy_trajectories=len(saved['issued']),budget_trajectories=len(saved['guarded']),budget_violations=0,
        rss_method_specific_forks=rss_count,rss_complete_target_error=rss_error,
        scope='regenerated physical worlds, all complete counterfactuals, all rules and budget levels; no additional efficacy observations')
    a.dump(a.ROOT/'independent_target_audit.json',result);print(result)

if __name__=='__main__':audit()
