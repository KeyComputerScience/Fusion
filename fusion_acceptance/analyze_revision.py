"""Manuscript analysis from immutable completed outcomes; no policy selection."""
from pathlib import Path
import gzip, hashlib, importlib.util, json, math, statistics

ROOT = Path(__file__).resolve().parent

def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else path.open()) as f:
        return json.load(f)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sp = importlib.util.spec_from_file_location('reporting_monitor', ROOT/'risk_monitor.py')
monitor = importlib.util.module_from_spec(sp)
sp.loader.exec_module(monitor)

def summarize_rows(rows):
    admitted = [r for r in rows if r['action']]
    return dict(issued=len(rows), beneficial=sum(r['local_net'] > 0 for r in admitted),
                harmful=sum(r['local_net'] < 0 for r in admitted), admitted=len(admitted),
                negative_loss=sum(max(0., -r['local_net']) for r in admitted),
                opportunities=dict(positive=sum(r['local_net'] > 0 for r in rows),
                                   negative=sum(r['local_net'] < 0 for r in rows),
                                   zero=sum(r['local_net'] == 0 for r in rows)),
                paid_gain=sum(r['local_net'] for r in admitted),
                lower_certificate=sum(r['gate_score']-max(0., r['gate_score']-(r['truegross']-5.)) for r in admitted),
                score_sum=sum(r['gate_score'] for r in admitted),
                coverage=[sum(r['lower_covered'] for r in admitted), len(admitted)])

def check_result(result):
    checks = []
    for g in result['budget_rows']:
        rows = g['rows']; acts = [r for r in rows if r['action']]
        assert abs(sum(r['local_net'] for r in acts)-g['increment']) < 1e-9
        assert abs(sum(max(0., -r['local_net']) for r in acts)-g['negative_loss']) < 1e-9
        assert g['negative_loss'] <= g['budget']+1e-9
        assert g['minimum_prefix'] >= -g['budget']-1e-9
        for state in g['ledger']:
            assert state['spent_loss']+state['reserved'] <= g['budget']+1e-9
            if state['action']:
                assert state['issued_reserve'] == 130.
                assert state['capacity_before'] >= 130.-1e-9
        checks.append(dict(arm=g['arm'], seed=g['seed'], recording=g.get('recording'),
                           budget=g['budget'], negative_loss=g['negative_loss'],
                           minimum_prefix=g['minimum_prefix'], service_error=g['service_error']))
    return checks

out = dict(role='Outcome analysis only; algorithms, selections and frozen sources are unchanged',
           monitor_checks=monitor.checks(), inputs={})
for name in ('geometry_joint', 'physical'):
    source = ROOT/name/'results.json.gz'; result = read(source)
    summary = read(ROOT/name/'summary.json')
    out['inputs'][name] = {str(p.relative_to(ROOT)): sha(p) for p in
                          (source, ROOT/name/'summary.json', ROOT/name/'protocol.json', ROOT/name/'selection_lock.json')}
    arm_groups = {}
    for g in result['guarded']:
        arm_groups.setdefault(g['arm'], []).append(g)
    analysis = dict(checks=check_result(result), arms={}, risk_paths=[], batches={})
    # Each physical history chain is monitored separately. The union allocation
    # covers all arms/chains/endpoints; repeating the same trace is not new data.
    delta = .05/(3*len(result['guarded']))
    for arm, gs in arm_groups.items():
        rr = [r for g in gs for r in g['rows']]
        analysis['arms'][arm] = summarize_rows(rr)
        for g in gs:
            certificate = summarize_rows(g['rows'])['lower_certificate']
            assert certificate <= g['increment']+1e-8
            rows = g['rows']
            now = max((r['maturity'] for r in rows), default=0)
            entry = monitor.matured_prefix(rows, now, delta)
            entry.update(arm=arm, seed=g['seed'], recording=g.get('recording'),
                         lower_certificate=certificate, complete_increment=g['increment'])
            # The normalized excess bound is used only for bounded tail laws.
            if arm in ('legacy_factorized', 'legacy_sandwich', 'dbf', 'eavg'):
                entry['normalized_excess']['theorem_scope'] = 'empirical report only; uncapped native scores lack the CJRT bounded-score premise'
            analysis['risk_paths'].append(entry)
    for batch in sorted({g.get('batch', 'RSS') for g in result['guarded']}):
        gs = [g for g in result['guarded'] if g['arm']=='joint' and g.get('batch', 'RSS')==batch]
        rr = [r for g in gs for r in g['rows']]
        seeds = sorted({g['seed'] for g in gs})
        values = [sum(g['increment'] for g in gs if g['seed']==s) for s in seeds]
        analysis['batches'][batch] = dict(**summarize_rows(rr),
            mean_increment=statistics.mean(values), seed_increments=values,
            experiments_with_benefit=sorted({g.get('recording', 'RSS') for g in gs if g['beneficial']>0}))
    joint = analysis['arms']['joint']
    joint['mean_lower_certificate'] = joint['lower_certificate']/5
    joint['mean_paid_increment'] = joint['paid_gain']/5
    joint['maximum_chain_negative_loss'] = max(g['negative_loss'] for g in arm_groups['joint'])
    joint['minimum_chain_prefix'] = min(g['minimum_prefix'] for g in arm_groups['joint'])
    jr = [r for g in arm_groups['joint'] for r in g['rows'] if r['information_ready']]
    joint['smallest_ready_absolute_score'] = min(abs(r['gate_score']) for r in jr)
    joint['score_shift_actions'] = {str(shift):sum((r['gate_score']>0) != (r['gate_score']+shift>0) for r in jr)
                                     for shift in (-.25,.25)}
    analysis['monitored_nonempty_paths'] = sum(e['violation']['count']>0 for e in analysis['risk_paths'])
    analysis['smallest_nonempty_violation_upper'] = min(e['violation']['upper'] for e in analysis['risk_paths'] if e['violation']['count']>0)
    analysis['independent_service_error'] = summary['max_service_error']
    analysis['target_error'] = max(x['maximum_error'] for x in summary['target_checks'])
    out[name] = analysis

fixed = read(ROOT/'geometry_joint/fixed_information/results.json')
out['fixed_information'] = dict(summary=fixed['summary'], changed=fixed['changed'])
(ROOT/'manuscript_analysis.json').write_text(json.dumps(out, indent=2, allow_nan=False))
print(json.dumps({n:dict(joint=out[n]['arms']['joint'], batches=out[n]['batches'],
                        target_error=out[n]['target_error'],
                        budget_paths=len(out[n]['checks']),
                        smallest_nonempty_violation_upper=out[n]['smallest_nonempty_violation_upper'])
                  for n in ('geometry_joint','physical')},indent=2))
