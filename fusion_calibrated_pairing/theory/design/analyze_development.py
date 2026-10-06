"""Lossless action accounting for the known-selfBACK development run.

This analysis never selects, changes, or replays a controller.  All controls,
ties, missed benefits, and additional losses remain in the output.
"""
from pathlib import Path
import gzip
import json
import math

ROOT = Path(__file__).resolve().parent


def load(name):
    path = ROOT / name
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as file:
            return json.load(file)
    return json.loads(path.read_text())


def row_key(chain, row):
    return (chain['recording'], chain['person'], chain['seed'], row['k'])


def main():
    payload = load('development_results.json.gz')
    summary = load('development_summary.json')
    chains = payload['guarded']
    arms = list(summary['summary'])
    indexed = {arm: {} for arm in arms}
    for chain in chains:
        for row in chain['rows']:
            key = row_key(chain, row)
            assert key not in indexed[chain['arm']]
            indexed[chain['arm']][key] = row
    base_keys = set(indexed['paired'])
    assert all(set(indexed[arm]) == base_keys for arm in arms)
    comparisons = {}
    changed_actions = []
    for arm in arms:
        if arm == 'paired':
            continue
        entries = []
        parts = dict(added_benefits=0., added_losses=0., missed_benefits=0.,
                     avoided_losses=0., added_zero=0, omitted_zero=0)
        counts = dict(added_benefits=0, added_losses=0, missed_benefits=0,
                      avoided_losses=0)
        for key in sorted(base_keys):
            joint = indexed['paired'][key]
            control = indexed[arm][key]
            assert abs(joint['local_net'] - control['local_net']) < 1e-10
            assert abs(joint['truegross'] - control['truegross']) < 1e-10
            if joint['action'] == control['action']:
                continue
            value = joint['local_net']
            sign = 1 if joint['action'] else -1
            if value > 0:
                category = 'added_benefits' if sign == 1 else 'missed_benefits'
            elif value < 0:
                category = 'added_losses' if sign == 1 else 'avoided_losses'
            else:
                category = 'added_zero' if sign == 1 else 'omitted_zero'
            if value == 0:
                parts[category] += 1
            else:
                parts[category] += sign * value
                counts[category] += 1
            item = dict(control=arm, recording=key[0], person=key[1], seed=key[2],
                        origin=key[3], category=category, local_net=value,
                        paired_admit=joint['action'], control_admit=control['action'],
                        paired_score=joint['gate_score'], control_score=control['gate_score'],
                        paired_raw_score=joint['raw_gate_score'],
                        control_raw_score=control['raw_gate_score'],
                        paired_calibration=joint['calibration_correction'],
                        control_calibration=control['calibration_correction'],
                        gain=sign * value)
            entries.append(item)
            changed_actions.append(item)
        action_gain = sum(item['gain'] for item in entries)
        measured = sum(c['increment'] for c in chains if c['arm'] == 'paired') - sum(
            c['increment'] for c in chains if c['arm'] == arm)
        assert abs(action_gain - measured) < 1e-8
        comparisons[arm] = dict(changed_actions=len(entries), counts=counts,
                                pooled_parts=parts, pooled_net=action_gain,
                                mean_net_over_five_delays=action_gain / 5,
                                accounting_error=abs(action_gain - measured))
    units = []
    for person in sorted(set(chain['person'] for chain in chains)):
        for arm in arms:
            selected = [c for c in chains if c['person'] == person and c['arm'] == arm]
            rows = [r for c in selected for r in c['rows']]
            admitted = [r for r in rows if r['action']]
            units.append(dict(person=person, arm=arm, chains=len(selected),
                              mean_net=sum(c['increment'] for c in selected)/len(selected),
                              beneficial=sum(r['local_net'] > 0 for r in admitted),
                              harmful=sum(r['local_net'] < 0 for r in admitted),
                              admissions=len(admitted),
                              admitted_covered=sum(r['lower_covered'] for r in admitted),
                              admitted_excess_max=max([0.] + [max(0., r['gate_score']-(r['truegross']-5)) for r in admitted])))
    budgets = []
    for arm in arms:
        for cap in (0., 110., 130., 260.):
            selected = [c for c in payload['budget_rows'] if c['arm'] == arm and c['budget'] == cap]
            budgets.append(dict(arm=arm, budget=cap,
                                mean_net=sum(c['increment'] for c in selected)/5,
                                beneficial=sum(c['beneficial'] for c in selected),
                                harmful=sum(c['harmful'] for c in selected),
                                negative_loss=sum(c['negative_loss'] for c in selected),
                                guard_refusals=sum(c['refused'] for c in selected),
                                max_per_chain_loss=max(c['negative_loss'] for c in selected),
                                max_service_error=max(c['service_error'] for c in selected)))
    out = dict(status='Known-data development only; not independent confirmation',
               controls=summary['summary'], selections=payload['selected'],
               complete_action_decomposition=comparisons, changed_actions=changed_actions,
               physical_unit_results=units, execution_budget_sensitivity=budgets,
               max_service_error=summary['max_service_error'])
    (ROOT/'development_analysis.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(comparisons, indent=2))
    print('Maximum service error:', summary['max_service_error'])


if __name__ == '__main__':
    main()
