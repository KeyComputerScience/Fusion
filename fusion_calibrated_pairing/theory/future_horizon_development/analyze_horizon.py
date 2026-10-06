"""Complete retained analysis of V0 and optional V1; never selects a policy."""
from pathlib import Path
import argparse,json,gzip,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(p):
    data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def indexed(results):
    by={}
    for g in results['guarded']:
        arm=by.setdefault(g['arm'],{})
        for row in g['rows']:
            key=(g['recording'],g['seed'],row['k'])
            if key in arm:raise AssertionError('Duplicate physical origin/delay')
            arm[key]=row
    return by

def difference(joint,control):
    if set(joint)!=set(control):raise AssertionError('Unaligned complete forks')
    values=dict(added_benefit=0.,avoided_loss=0.,missed_benefit=0.,incurred_loss=0.)
    counts={k:0 for k in values};rows=[]
    for key,j in joint.items():
        c=control[key];D=float(j['local_net'])
        if D!=float(c['local_net']):raise AssertionError('Different common potential return')
        delta=int(j['action'])-int(c['action'])
        if not delta:continue
        category=('added_benefit' if D>0 else 'incurred_loss') if delta>0 else ('missed_benefit' if D>0 else 'avoided_loss')
        values[category]+=abs(D);counts[category]+=1
        N=float(j['N'])
        rows.append(dict(recording=key[0],seed=key[1],origin=key[2],complete_return=D,
            category=category,joint_action=j['action'],control_action=c['action'],
            joint_proposal=j['proposed_action'],control_proposal=c['proposed_action'],
            raw_joint=j['raw_gate_score'],raw_control=c['raw_gate_score'],
            issued_joint=j['gate_score'],issued_control=c['gate_score'],
            correction_joint=j['calibration_correction'],correction_control=c['calibration_correction'],
            joint_raw_using_control_correction=j['raw_gate_score']-N*c['calibration_correction'],
            hard_disagreement=j['disagreement'],soft_support=j.get('soft_support')))
    total=values['added_benefit']+values['avoided_loss']-values['missed_benefit']-values['incurred_loss']
    direct=sum((int(joint[k]['action'])-int(control[k]['action']))*joint[k]['local_net'] for k in joint)
    if abs(total-direct)>1e-10:raise AssertionError('Four-term identity failed')
    return dict(totals=values,counts=counts,pooled_difference=total,mean_difference=total/5.,changed_actions=len(rows),rows=rows)

def analyze(folder):
    results=read(folder/'development_results.json.gz');summary=read(folder/'development_summary.json');selection=read(folder/'selection.json')
    by=indexed(results);comparisons={c:difference(by['paired'],b) for c,b in by.items() if c!='paired'}
    for c,r in comparisons.items():
        service=sum(g['increment'] for g in results['guarded'] if g['arm']=='paired')-sum(g['increment'] for g in results['guarded'] if g['arm']==c)
        r['service_identity_error']=abs(service-r['pooled_difference'])
    budgets={}
    for a in by:
        budgets[a]={}
        for B in (0.,110.,130.,260.):
            gs=[g for g in results['budget_rows'] if g['arm']==a and g['budget']==B]
            rows=[r for g in gs for r in g['rows']]
            budgets[a][str(B)]=dict(net=sum(g['increment'] for g in gs)/5.,
                admissions=sum(r['action'] for r in rows),negative_loss=sum(max(0.,-r['local_net']) for r in rows if r['action']),
                refusals=sum(r['proposed_action'] and not r['action'] for r in rows),
                maximum_chain_ledger=max(x['spent_loss']+x['reserved'] for g in gs for x in g['ledger']))
    promotions={}
    for a,rows in by.items():
        raw_promoted=[r for r in rows.values() if r['proposed_action'] and r['raw_gate_score']<=0.]
        actual=[r for r in raw_promoted if r['action']]
        promotions[a]=dict(proposed=len(raw_promoted),admitted=len(actual),
                           beneficial=sum(r['local_net']>0 for r in actual),harmful=sum(r['local_net']<0 for r in actual),
                           return_sum=sum(r['local_net'] for r in actual))
    units={}
    for a in by:
        person={}
        for g in results['guarded']:
            if g['arm']==a:person.setdefault(str(g['person']),[]).append(g['increment'])
        units[a]={p:float(np.mean(x)) for p,x in person.items()}
    report=dict(status='Already-used selfBACK DEVELOPMENT; full retained candidate, not fresh confirmation',
                source_protocol_sha256=sha(folder/'protocol.json'),result_sha256=sha(folder/'development_results.json.gz'),
                selected=selection['selected'],strongest_cal_control=selection['strongest_control'],
                summary=summary['summary'],comparisons=comparisons,budgets=budgets,
                calibration_promotions=promotions,person_means=units,
                max_action_identity_error=max(r['service_identity_error'] for r in comparisons.values()),
                max_service_error=summary['max_service_error'],
                max_fork_error=max(x['maximum_error'] for x in summary['target_checks']),
                fork_checks=sum(x['forks'] for x in summary['target_checks']),
                policy_trajectories=summary['policy_trajectories'],budget_paths=summary['budget_paths'])
    if folder==ROOT:
        diagnosis=read(ROOT/'descriptor_diagnosis.json');quiet=[d for d in diagnosis if d['old_hard_disagreement']==0.]
        report['descriptor_diagnosis']=dict(origins=len(diagnosis),hard_quiet=len(quiet),
             hard_quiet_soft_nonzero=sum(d['soft_disagreement']>0 for d in quiet),
             quiet_potential_beneficial=sum(d['potential_return']>0 for d in quiet),
             quiet_soft_nonzero_beneficial=sum(d['potential_return']>0 and d['soft_disagreement']>0 for d in quiet),
             observed_beneficial=sum(d['potential_return']>0 for d in diagnosis),
             paired_admitted_quiet=sum(r['action'] and r['disagreement']==0 for r in by['paired'].values()),
             paired_admitted_quiet_beneficial=sum(r['action'] and r['disagreement']==0 and r['local_net']>0 for r in by['paired'].values()))
    if folder!=ROOT:
        old=indexed(read(ROOT/'development_results.json.gz'))
        report['versus_v0_by_arm']={a:difference(by[a],old[a]) for a in by}
    (folder/'complete_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('strongest_cal_control','max_action_identity_error','max_service_error','max_fork_error','policy_trajectories','budget_paths')},indent=2))
    print('ALL_MEANS',{a:s['mean_increment'] for a,s in report['summary'].items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='?',default=str(ROOT));args=p.parse_args();analyze(Path(args.folder).resolve())
