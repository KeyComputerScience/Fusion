"""Read-only qualification audit of already executed development versions.

No controller is run, fitted, selected, or modified. A comparison is emitted
only after physical/person/seed/origin potential-target identity is verified.
Qualification rules are diagnostic requirements, not a preregistration or a
new independent physical experiment. Frozen Mobile/OPPORTUNITY are untouched.
"""
from pathlib import Path
import gzip, hashlib, json, math

W = Path(__file__).resolve().parent.parent
RISK = Path(__file__).resolve().parent
FOLDERS = {'P-V2': W/'design_v2', 'H-v0': W/'future_horizon_development',
           'H-v1': W/'future_horizon_development/v1_positive'}
TOL = 1e-8

def read(p):
    b = p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix == '.gz' else b)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def index(payload):
    arms = {}
    for ch in payload['guarded']:
        arm = arms.setdefault(ch['arm'], {})
        for r in ch['rows']:
            key = (ch['recording'], ch['person'], ch['seed'], r['k'])
            if key in arm: raise AssertionError(('duplicate', ch['arm'], key))
            arm[key] = r
        actual = sum(r['local_net'] for r in ch['rows'] if r['action'])
        loss = sum(max(0., -r['local_net']) for r in ch['rows'] if r['action'])
        assert abs(actual-ch['increment']) < TOL
        assert abs(loss-ch['negative_loss']) < TOL
        assert ch['service_error'] < TOL
        assert all(x['spent_loss']+x['reserved'] <= 130.+TOL for x in ch['ledger'])
    return arms

def metric(rows):
    adm = [r for r in rows.values() if r['action']]
    excess = [max(0., r['gate_score']-(r['truegross']-5.)) for r in adm]
    covered = sum(r['truegross']-5. >= r['gate_score']-1e-10 for r in adm)
    for r in rows.values():
        assert -TOL <= r['local_net']-(r['truegross']-5.) <= 2.+TOL
        assert -r['N']-5.-TOL <= r['gate_score'] <= r['N']-5.+TOL
    seeds = sorted({key[2] for key in rows})
    return dict(origins=len(rows), chains=len({key[:3] for key in rows}),
        seeds=seeds, pooled_net=sum(r['local_net'] for r in adm),
        mean_net_over_shared_delay_seeds=sum(r['local_net'] for r in adm)/len(seeds),
        admissions=len(adm), beneficial=sum(r['local_net']>0 for r in adm),
        harmful=sum(r['local_net']<0 for r in adm),
        negative_loss=sum(max(0., -r['local_net']) for r in adm),
        admitted_coverage=dict(covered=covered,total=len(adm),rate=covered/len(adm) if adm else None),
        admitted_excess_sum=sum(excess), admitted_excess_max=max(excess,default=0.),
        admitted_excess_mean=sum(excess)/len(adm) if adm else None,
        refused=sum(bool(r['proposed_action']) and not r['action'] for r in rows.values()))

def common(j,c):
    if set(j)!=set(c): return dict(valid=False, reason='Different physical/person/seed/origin sets')
    checks = {'local_net':0.,'truegross':0.,'N':0.,'maturity':0.,'anchor':0.}
    for key in j:
        for field in checks: checks[field]=max(checks[field],abs(float(j[key][field])-float(c[key][field])))
    # External rules have different issued fused anchors by design. A complete
    # policy contrast requires common potential targets, not common forecasts.
    valid=all(checks[k]<TOL for k in ['local_net','truegross','N','maturity'])
    return dict(valid=valid,origins=len(j),maximum_field_differences=checks,
                scope='Whole-version complete-policy comparison, not isolated descriptor attribution')

def delta(m,n):
    out={k:m[k]-n[k] for k in ['pooled_net','mean_net_over_shared_delay_seeds','admissions','beneficial','harmful','negative_loss','admitted_excess_sum','admitted_excess_max','refused']}
    for k in ['admitted_excess_mean']:
        out[k]=m[k]-n[k] if m[k] is not None and n[k] is not None else None
    out['admitted_coverage_rate']=m['admitted_coverage']['rate']-n['admitted_coverage']['rate'] if m['admitted_coverage']['rate'] is not None and n['admitted_coverage']['rate'] is not None else None
    return out

def criteria(m,n):
    checks={
      'net_not_lower':m['pooled_net']>=n['pooled_net']-TOL,
      'beneficial_not_fewer':m['beneficial']>=n['beneficial'],
      'harmful_not_more':m['harmful']<=n['harmful'],
      'negative_loss_not_more':m['negative_loss']<=n['negative_loss']+TOL,
      'excess_sum_not_more':m['admitted_excess_sum']<=n['admitted_excess_sum']+TOL,
      'excess_max_not_more':m['admitted_excess_max']<=n['admitted_excess_max']+TOL,
      'actual_coverage_not_lower':None,
      'excess_mean_not_more':None,
    }
    if m['admitted_coverage']['rate'] is not None and n['admitted_coverage']['rate'] is not None:
        checks['actual_coverage_not_lower']=m['admitted_coverage']['rate']+TOL>=n['admitted_coverage']['rate']
        checks['excess_mean_not_more']=m['admitted_excess_mean']<=n['admitted_excess_mean']+TOL
    return dict(checks=checks,failed=[k for k,v in checks.items() if v is False],
                unestimated=[k for k,v in checks.items() if v is None],
                all_evaluable_non_regression=all(v is not False for v in checks.values()),
                all_metrics_demonstrated=all(v is True for v in checks.values()),
                strictly_more_net_and_benefits=m['pooled_net']>n['pooled_net']+TOL and m['beneficial']>n['beneficial'])

def compare(j,c):
    aligned=common(j,c)
    if not aligned['valid']: return dict(common_potential_targets=aligned,paired_comparison=None)
    jm,cm=metric(j),metric(c)
    parts={k:dict(count=0,value=0.) for k in ['added_benefit','avoided_loss','missed_benefit','incurred_loss']}
    changed=[]
    for key,r in j.items():
        s=c[key];a=int(r['action'])-int(s['action']);D=r['local_net']
        if not a:continue
        if D==0:category='zero'
        else:
            category=('added_benefit' if D>0 else 'incurred_loss') if a>0 else ('missed_benefit' if D>0 else 'avoided_loss')
            parts[category]['count']+=1;parts[category]['value']+=abs(D)
        changed.append(dict(recording=key[0],person=key[1],seed=key[2],origin=key[3],category=category,local_net=D,gain=a*D))
    identity=parts['added_benefit']['value']+parts['avoided_loss']['value']-parts['missed_benefit']['value']-parts['incurred_loss']['value']
    assert abs(identity-(jm['pooled_net']-cm['pooled_net']))<TOL
    persons={}
    for p in sorted({key[1] for key in j}):
        jj={k:v for k,v in j.items() if k[1]==p};cc={k:v for k,v in c.items() if k[1]==p}
        mm,nn=metric(jj),metric(cc)
        persons[str(p)]=dict(candidate=mm,reference=nn,delta=delta(mm,nn),qualification=criteria(mm,nn))
    seeds={}
    for s in sorted({key[2] for key in j}):
        mm=metric({k:v for k,v in j.items() if k[2]==s});nn=metric({k:v for k,v in c.items() if k[2]==s})
        seeds[str(s)]=dict(candidate=mm,reference=nn,delta=delta(mm,nn),qualification=criteria(mm,nn))
    return dict(common_potential_targets=aligned,candidate=jm,reference=cm,delta=delta(jm,cm),
                qualification=criteria(jm,cm),action_value_parts=parts,changed_actions=changed,
                persons=persons,seeds=seeds,
                people_with_net_regression=[p for p,v in persons.items() if v['delta']['pooled_net']<-TOL],
                people_with_any_evaluable_regression=[p for p,v in persons.items() if v['qualification']['failed']],
                seeds_with_net_regression=[s for s,v in seeds.items() if v['delta']['pooled_net']<-TOL])

def main():
    inputs={};payloads={};indices={};reference_worlds={}
    for version,folder in FOLDERS.items():
        p=folder/'development_results.json.gz';s=folder/'development_summary.json';lock=folder/'selection.json'
        inputs[version]=dict(result_path=str(p),result_sha256=sha(p),summary_sha256=sha(s),selection_sha256=sha(lock),protocol_sha256=sha(folder/'protocol.json'))
        payloads[version]=read(p);indices[version]=index(payloads[version])
        # Exact complete chain/origin inventory, no silent unit omission.
        assert len({k[:3] for k in indices[version]['paired']})==65
        assert len(indices[version]['paired'])==710
        assert len({k[1] for k in indices[version]['paired']})==13
        assert len({k[2] for k in indices[version]['paired']})==5
        assert all(set(rows)==set(indices[version]['paired']) for rows in indices[version].values())
        refs={}
        for ch in payloads[version]['guarded']:
            key=(ch['recording'],ch['person'],ch['seed'])
            value=ch['net']-ch['increment']
            if key in refs: assert abs(refs[key]-value)<TOL
            refs[key]=value
        reference_worlds[version]=refs
    common_reference_error={v:max(abs(ref-reference_worlds['P-V2'][key]) for key,ref in refs.items()) for v,refs in reference_worlds.items()}
    assert all(error<TOL for error in common_reference_error.values())
    families={}
    for version,idx in indices.items():
        sel=read(FOLDERS[version]/'selection.json')
        strongest=sel.get('strongest_control')
        families[version]=dict(arms=list(idx),summary={a:metric(rows) for a,rows in idx.items()},
            strongest_calibration_control=strongest,
            paired_vs_all_controls={a:compare(idx['paired'],r) for a,r in idx.items() if a!='paired'},
            shared_target_rows=len(idx['paired']),physical_people=13,delay_arrangements=5,
            qualifies_for_all_control_all_person_all_metric_advantage=None)
        comparisons=families[version]['paired_vs_all_controls'].values()
        families[version]['qualifies_for_all_control_all_person_all_metric_advantage']=all(
            c['qualification']['all_metrics_demonstrated'] and c['qualification']['strictly_more_net_and_benefits']
            and all(p['qualification']['all_evaluable_non_regression'] for p in c['persons'].values())
            for c in comparisons)
    version_comparisons={}
    for new,old in [('H-v0','P-V2'),('H-v1','P-V2'),('H-v1','H-v0')]:
        version_comparisons[new+' vs '+old]={a:compare(indices[new][a],indices[old][a]) for a in set(indices[new])&set(indices[old])}
    # P-V2 has no stored external pipeline outputs. Borrowing them is explicitly
    # cross-version and occurs only after exact common-target checks.
    families['P-V2']['external_outputs_absent']=True
    families['P-V2']['paired_vs_Hv0_external_on_verified_common_targets']={a:compare(indices['P-V2']['paired'],indices['H-v0'][a]) for a in ['pdf','qmf','dbf']}
    report=dict(status='READ-ONLY used-selfBACK DEVELOPMENT qualification audit; no algorithm changes, no new run, no frozen-test reuse',
        checker_sha256=sha(Path(__file__)),inputs=inputs,families=families,version_comparisons=version_comparisons,
        common_reference_service_maximum_error=common_reference_error,
        qualification_rule='For each named whole pipeline and physical person: net/beneficial count/actual coverage not lower, harmful count/negative loss/admitted excess sum-max-mean not higher; strict superiority additionally needs more net and beneficial actions. All named comparisons must hold simultaneously. Undefined zero-admission coverage is not 100% and cannot establish risk superiority.',
        not_a_preregistration='These criteria audit already known development outcomes; they are not confirmation or a prospective selection protocol.',
        conclusion='No executed candidate qualifies. H-v0 improves aggregate net over P-V2 and CAL-selected QMF but loses beneficial actions, raises harmful count, lowers actual coverage, regresses on some people, and loses net to unconditional. H-v1 further regresses aggregate net/loss relative to H-v0. Conditional-moment controls have identical paired actions in H-v0 and H-v1. Frozen Mobile755/OPPORTUNITY remain the original frozen algorithm.',
        preservation_check={v:sha(Path(info['result_path']))==info['result_sha256'] for v,info in inputs.items()})
    (RISK/'non_regression_candidate_check.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Read-only candidate non-regression qualification','',report['status'],'',
           'The inventory is complete: each version retains 13 physical people × 5 shared delays, 65 chains and 710 complete origins per arm. Potential complete return, gross target, capacity and maturity are checked origin by origin before any paired comparison. Anchor differences are recorded separately: external fused forecasts differ by design, whereas the paired cross-version common anchor matches. P-V2 stores six arms; H-v0/H-v1 store nine. P-V2 external comparisons below are explicitly cross-version against unchanged H-v0 adapters after common-target verification, not fabricated P-V2 trajectories.','',
           'Mean net sums people within each delay arrangement and averages five arrangements; action, loss and excess counts pool all 65 chains. These delays do not add physical people.','',
           '| Version | Mean net | Good | Harm | Loss | Actual admission coverage | Sum excess | Max excess |',
           '|---|---:|---:|---:|---:|---|---:|---:|']
    for v in FOLDERS:
        m=families[v]['summary']['paired'];cov=m['admitted_coverage']
        lines.append(f"| {v} | {m['mean_net_over_shared_delay_seeds']:.1f} | {m['beneficial']} | {m['harmful']} | {m['negative_loss']:.1f} | {cov['covered']}/{cov['total']} | {m['admitted_excess_sum']:.2f} | {m['admitted_excess_max']:.2f} |")
    lines+=['','## H-v0/H-v1 against every stored complete competitor','',
            '| Version/control | Δmean net | Δgood | Δharm | Δloss | Δcoverage (percentage points) | Failed aggregate non-regression criteria |',
            '|---|---:|---:|---:|---:|---:|---|']
    for v in ['H-v0','H-v1']:
        for a,c in families[v]['paired_vs_all_controls'].items():
            d=c['delta'];cv=d['admitted_coverage_rate']
            lines.append(f"| {v}/{a} | {d['mean_net_over_shared_delay_seeds']:.1f} | {d['beneficial']} | {d['harmful']} | {d['negative_loss']:.1f} | {100*cv:.2f}"+f" | {', '.join(c['qualification']['failed']) or 'none; strict extra-value still required'} |")
    lines+=['','## Every physical person: paired candidate versus P-V2','',
            '| Person | P-V2 net | H-v0 net | H-v1 net | H-v0 Δgood / Δharm / Δloss | H-v1 Δgood / Δharm / Δloss |',
            '|---|---:|---:|---:|---|---|']
    p0=version_comparisons['H-v0 vs P-V2']['paired']['persons'];p1=version_comparisons['H-v1 vs P-V2']['paired']['persons']
    for p in p0:
        a,b=p0[p],p1[p];d,e=a['delta'],b['delta']
        lines.append(f"| {p} | {a['reference']['mean_net_over_shared_delay_seeds']:.1f} | {a['candidate']['mean_net_over_shared_delay_seeds']:.1f} | {b['candidate']['mean_net_over_shared_delay_seeds']:.1f} | {d['beneficial']} / {d['harmful']} / {d['negative_loss']:.1f} | {e['beneficial']} / {e['harmful']} / {e['negative_loss']:.1f} |")
    lines+=['','Full per-person coverage numerators/denominators, excess sum/max/mean, per-delay changes, four-term action decomposition, all controls and hashes are in the companion JSON. Empty admitted populations have `rate: null`.','',
            '## Qualification conclusion','',report['conclusion'],'',report['qualification_rule'],'',report['not_a_preregistration'],'',
            'No candidate was run or changed. All input result hashes match before/after analysis. Existing frozen Mobile755/OPPORTUNITY outputs cannot be relabelled as evidence for a newly tuned development version.']
    (RISK/'non_regression_candidate_check.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'conclusion':report['conclusion'],'result_hashes_preserved':report['preservation_check'],
        'cross_version_alignment':{k:v['paired']['common_potential_targets'] for k,v in version_comparisons.items()},
        'person_regressions':{k:v['paired']['people_with_net_regression'] for k,v in version_comparisons.items()}},indent=2))

if __name__=='__main__':main()
