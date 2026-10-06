"""Read-only audit of all action differences and their issued-score sources.

Common-offset scores are explanatory recalculations, not extra evaluated
policies or new actual returns. No method or selected setting is changed.
"""
from pathlib import Path
import gzip, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parent

def load(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as f:
        return json.load(f)

def clipped(raw, N, q):
    return min(N - 5., max(-N - 5., raw - N * q))

def main():
    data = load(ROOT / 'opportunity/results.json.gz')
    selection = load(ROOT / 'opportunity/selection.json')
    guarded = {(g['arm'], g['recording'], g['seed']): g for g in data['guarded']}
    raw = {(g['arm'], g['recording'], g['seed']): g for g in data['raw']}
    arms = list(data['selected']); controls = [a for a in arms if a != 'paired']
    assert len(guarded) == 540
    reports = {}; records = []; maximum_score_error = 0.; moment_mean_error = 0.; moment_variance_error = 0.
    for arm in controls:
        terms = {key: dict(count=0, value=0.) for key in ('added_gain', 'avoided_loss', 'missed_gain', 'incurred_loss')}
        shared_inputs = 0; extra_positive = 0; robust_geometry = []; budget_path = 0; changed_zero = 0
        for (kind, recording, seed), paired in guarded.items():
            if kind != 'paired': continue
            control = guarded[arm, recording, seed]
            rp = {r['k']: r for r in raw['paired', recording, seed]['rows']}
            rc = {r['k']: r for r in raw[arm, recording, seed]['rows']}
            assert len(paired['rows']) == len(control['rows'])
            for p, c in zip(paired['rows'], control['rows']):
                assert p['k'] == c['k'] and p['local_net'] == c['local_net']
                a, b = rp[p['k']], rc[c['k']]
                N = p['N']; assert N == c['N']
                for issued in (p, c):
                    err = abs(clipped(issued['raw_gate_score'], issued['N'], issued['calibration_correction']) - issued['gate_score'])
                    maximum_score_error = max(maximum_score_error, err); assert err < 1e-8
                same = all(p[key] == c[key] for key in ('N', 'anchor', 'cal_context', 'source_descriptor', 'active_source_ids'))
                if arm in ('pair_factorized', 'full_gaussian', 'joint_diagonal_kernel', 'unconditional', 'conditional_moment'):
                    assert same, ('Information arms received different current inputs', arm, recording, seed, p['k'])
                    shared_inputs += 1
                if arm == 'conditional_moment':
                    moment_mean_error = max(moment_mean_error, abs(a['conditional_paid_mean'] - b['conditional_paid_mean']))
                    moment_variance_error = max(moment_variance_error, abs(a['conditional_paid_variance'] - b['conditional_paid_variance']))
                if p['action'] == c['action']: continue
                D = p['local_net']
                key = ('added_gain' if D > 0 else 'incurred_loss') if p['action'] else ('missed_gain' if D > 0 else 'avoided_loss')
                if D:
                    terms[key]['count'] += 1; terms[key]['value'] += abs(D)
                else:
                    changed_zero += 1
                qp = p['calibration_correction']; qc = c['calibration_correction']
                p_at_p = clipped(a['raw_gate_score'], N, qp); c_at_p = clipped(b['raw_gate_score'], N, qp)
                p_at_c = clipped(a['raw_gate_score'], N, qc); c_at_c = clipped(b['raw_gate_score'], N, qc)
                robust = bool(same and p['action'] and not c['action'] and D > 0 and not c['proposed_action']
                    and p_at_p > 0 and c_at_p <= 0 and p_at_c > 0 and c_at_c <= 0)
                reason = 'control_budget_refusal' if c['proposed_action'] and not c['action'] else (
                    'paired_budget_refusal' if p['proposed_action'] and not p['action'] else 'issued_proposal_difference')
                budget_path += reason != 'issued_proposal_difference'
                extra_positive += bool(p['action'] and not c['action'] and D > 0)
                entry = dict(control=arm, recording=recording, person=paired['person'], seed=seed, k=p['k'], D=D,
                    paired_action=p['action'], control_action=c['action'], paired_proposal=p['proposed_action'], control_proposal=c['proposed_action'],
                    same_current_inputs=same, change_reason=reason, paired_raw_score=a['raw_gate_score'], control_raw_score=b['raw_gate_score'],
                    paired_score=p['gate_score'], control_score=c['gate_score'], paired_correction=qp, control_correction=qc,
                    paired_at_paired_correction=p_at_p, control_at_paired_correction=c_at_p,
                    paired_at_control_correction=p_at_c, control_at_control_correction=c_at_c,
                    extra_positive_crossing_at_both_common_offsets=robust,
                    paired_conditional_paid_mean=a.get('conditional_paid_mean'), control_conditional_paid_mean=b.get('conditional_paid_mean'),
                    paired_conditional_paid_variance=a.get('conditional_paid_variance'), control_conditional_paid_variance=b.get('conditional_paid_variance'),
                    paired_actual_lower_covered=p['lower_covered'], control_actual_lower_covered=c['lower_covered'])
                records.append(entry)
                if robust: robust_geometry.append(entry)
        expected = sum(guarded['paired', recording, seed]['increment'] - g['increment']
            for (kind, recording, seed), g in guarded.items() if kind == arm)
        actual = terms['added_gain']['value'] + terms['avoided_loss']['value'] - terms['missed_gain']['value'] - terms['incurred_loss']['value']
        assert abs(actual - expected) < 1e-8
        reports[arm] = dict(all_four_action_terms=terms, mean_total_paid_difference=actual / 5.,
            matched_current_input_episodes=shared_inputs, additional_actual_beneficial_actions=extra_positive,
            changed_actions_due_to_different_budget_history=budget_path, changed_zero_return_actions=changed_zero,
            extra_positive_crossings_robust_to_both_issued_offsets=len(robust_geometry),
            robust_geometry_realized_return=sum(row['D'] for row in robust_geometry), robust_geometry_records=robust_geometry)
    out = dict(results_sha256=hashlib.sha256((ROOT / 'opportunity/results.json.gz').read_bytes()).hexdigest(),
        strongest_calibration_control=selection['strongest_calibration_control'], comparisons=reports,
        every_changed_action=records, maximum_clipped_score_reconstruction_error=maximum_score_error,
        maximum_conditional_moment_mean_difference=moment_mean_error,
        maximum_conditional_moment_variance_difference=moment_variance_error,
        scope='All observed changed actions retained. Common-offset computations are explanatory diagnostics, not new executed policies. Conditional-moment and full-joint KDE controls retain joint information; this audit does not exclude all possible full-joint methods.')
    (ROOT / 'opportunity_crossing_audit.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('every_changed_action', 'comparisons')}, indent=2))
    for arm, value in reports.items():
        print(arm, value['mean_total_paid_difference'], value['all_four_action_terms'],
              'robust beneficial geometry', value['extra_positive_crossings_robust_to_both_issued_offsets'])

if __name__ == '__main__':
    main()
