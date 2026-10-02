#!/usr/bin/env python3
"""Recompute reported statistics from saved real-sensor replay result files.

This command analyzes saved traces; it does not train or run a new experiment.
Only the Python standard library and NumPy are required. Default paths are
relative to this script, so the command can be called from any working directory.
The original saved summaries use normal 1.96 intervals. This analysis uses
Student t9 for ten-seed comparisons and t4 for five-seed sensitivities, while
retaining the former intervals explicitly as descriptive secondary output.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE = Path(__file__).resolve().parent
T975 = {1: 12.7062047364, 2: 4.3026527299, 3: 3.1824463053,
        4: 2.7764451052, 5: 2.5705818356, 6: 2.4469118511,
        7: 2.3646242510, 8: 2.3060041352, 9: 2.2621571627,
        10: 2.2281388519}
TIE_EPS = 1e-8


def load_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def paired(values: list[float], unique_replays: int | None = None) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        raise ValueError('Paired comparison has no observations.')
    n_unique = n if unique_replays is None else unique_replays
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
    pos = int(np.sum(v > TIE_EPS))
    neg = int(np.sum(v < -TIE_EPS))
    result: dict[str, Any] = dict(mean=m, median=float(np.median(v)), sd=sd,
        range=[float(np.min(v)), float(np.max(v))], positive=pos,
        negative=neg, ties=n-pos-neg, reported_rows=n,
        unique_replays=n_unique)
    if n_unique == 1:
        result.update(ci95_student_t=None, ci95_normal=None,
            two_sided_exact_sign_p=None, two_sided_exact_signflip_p=None,
            interval_interpretation='One deterministic replay repeated; no independent-seed interval.')
        return result
    if n_unique != n:
        raise ValueError('Partial duplicates require explicitly grouped independent units.')
    if n < 2 or n-1 not in T975:
        raise ValueError(f'No built-in Student critical value for {n} paired rows.')
    se = sd / math.sqrt(n)
    hw = T975[n-1] * se
    result['ci95_student_t'] = [m-hw, m+hw]
    result[f'ci95_student_t{n-1}'] = [m-hw, m+hw]
    result['ci95_normal'] = [m-1.96*se, m+1.96*se]
    nonzero = pos+neg
    result['two_sided_exact_sign_p'] = (min(1.0, 2*sum(math.comb(nonzero, k)
        for k in range(min(pos, neg)+1))/2**nonzero) if nonzero else 1.0)
    if n <= 20:
        observed = abs(float(np.sum(v)))
        extreme = sum(abs(sum(s*x for s, x in zip(signs, v))) >= observed-TIE_EPS
            for signs in itertools.product((-1, 1), repeat=n))
        result['two_sided_exact_signflip_p'] = extreme/2**n
    else:
        result['two_sided_exact_signflip_p'] = None
    result['interval_interpretation'] = ('Conditional assigned-delay/scenario interval; '
        'no dataset, model-fit, method-selection or multiplicity uncertainty.')
    result['signflip_assumption'] = ('Reference test assumes symmetric independent sign flips '
        'of paired differences; not an exploration-adjusted confirmation.')
    return result


def canonical(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for old, new in [('potential_accuracy', 'accuracy'), ('probe_fees', 'probe_fee'),
                     ('deploy_fees', 'deploy_fee'), ('total_fees', 'total_fee')]:
        if new not in out and old in out:
            out[new] = out[old]
    return out


def study_summary(data: dict[str, Any]) -> dict[str, Any]:
    seeds = data['per_seed']
    methods = list(seeds[0]['results'])
    full = [canonical(s['results']['decision_full']) for s in seeds]
    out: dict[str, Any] = {}
    for method in methods:
        rs = [canonical(s['results'][method]) for s in seeds]
        delta = [a['net_return']-b['net_return'] for a, b in zip(full, rs)]
        action_diff = [int(np.sum(np.asarray(a['actions'], dtype=bool) !=
                         np.asarray(b['actions'], dtype=bool))) for a, b in zip(full, rs)]
        row = dict(net=mean([r['net_return'] for r in rs]),
            deployments=mean([r['deployments'] for r in rs]),
            potential_accuracy=mean([r['accuracy'] for r in rs]),
            paired_full_minus_comparator=paired(delta), per_seed_difference=delta,
            mean_actual_action_differences=mean(action_diff),
            actual_action_differences_total=sum(action_diff))
        for field in ['probe_steps', 'probe_fee', 'deploy_fee', 'total_fee', 'dropped_service']:
            if all(field in r for r in rs):
                row[field] = mean([r[field] for r in rs])
        if all('total_fee' in r for r in rs):
            row['gross_served_reward'] = mean([r['net_return']+r['total_fee'] for r in rs])
            row['reward_change_full_minus_comparator'] = mean([a['net_return']+a['total_fee']-
                b['net_return']-b['total_fee'] for a, b in zip(full, rs)])
            row['fee_saving_comparator_minus_full'] = mean([b['total_fee']-a['total_fee']
                for a, b in zip(full, rs)])
            row['decomposition_error'] = (row['reward_change_full_minus_comparator']+
                row['fee_saving_comparator_minus_full']-row['paired_full_minus_comparator']['mean'])
        if 'monthly' in rs[0]:
            n = sum(v['n'] for v in rs[0]['monthly'].values())
        else:
            n = rs[0].get('n')
        if n is not None:
            row['test_opportunities'] = n
            if 'dropped_service' in row:
                row['served_opportunities'] = n-row['dropped_service']
        out[method] = row
    return out


def analyze_months(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ps = data['per_seed']
    methods = ['decision_full', 'context_joint', 'decision_diagonal', 'rf_c',
               'periodic', 'frozen_deploy']
    out = []
    for month in data['split']['test']:
        metrics = {}
        for method in methods:
            rows = [p['results'][method]['monthly'][month] for p in ps]
            metrics[method] = dict(accuracy=mean([r['accuracy'] for r in rows]),
                balanced_accuracy=mean([r['balanced_accuracy'] for r in rows]),
                classes_present=sorted({r['classes_present'] for r in rows}))
        delta = [100*(p['results']['decision_full']['monthly'][month]['accuracy']-
                     p['results']['context_joint']['monthly'][month]['accuracy']) for p in ps]
        out.append(dict(month=month, n=ps[0]['results']['decision_full']['monthly'][month]['n'],
            metrics=metrics, paired_accuracy_pp_full_minus_joint=paired(delta)))
    counts = dict(positive=sum(r['paired_accuracy_pp_full_minus_joint']['mean']>TIE_EPS for r in out),
        negative=sum(r['paired_accuracy_pp_full_minus_joint']['mean']<-TIE_EPS for r in out),
        tie=sum(abs(r['paired_accuracy_pp_full_minus_joint']['mean'])<=TIE_EPS for r in out))
    return out, counts


def action_audit(data: dict[str, Any]) -> dict[str, Any]:
    ps = data['per_seed']
    records = [v for p in ps for v in p['results']['decision_full']['audit']]
    eligible = sum(bool(v['eligible']) for v in records)
    out = dict(total_decisions=len(records), eligible_decisions=eligible,
               eligible_rate=eligible/len(records), comparators={})
    for method in ['context_joint', 'decision_diagonal', 'quality_only', 'rf_c']:
        crossing = [v for v in records if v['action'] != v['shadow'][method]['action']]
        signed = [v['local_advantage']*(1 if v['action'] else -1) for v in crossing]
        if any(not v['eligible'] for v in crossing):
            raise ValueError('An ineligible same-state action crossed the admission gate.')
        out['comparators'][method] = dict(crossings=len(crossing),
            beneficial=sum(x>TIE_EPS for x in signed), harmful=sum(x<-TIE_EPS for x in signed),
            ties=sum(abs(x)<=TIE_EPS for x in signed), local_sum=sum(signed),
            eligible_crossing_rate=len(crossing)/eligible,
            all_window_crossing_rate=len(crossing)/len(records),
            mean_weight_l2=mean([v['shadow'][method]['weight_l2'] for v in records]))
    return out


def sensitivity_summary(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seeds = data['per_seed'][:5]
    base = {p['seed']: p['results']['decision_full'] for p in seeds}
    default_diff = [p['results']['decision_full']['net_return']-
                   p['results']['context_joint']['net_return'] for p in seeds]
    default = dict(mean_return=mean([r['net_return'] for r in base.values()]),
        mean_actions=mean([r['deployments'] for r in base.values()]), mean_action_changes=0,
        paired_full_minus_joint=paired(default_diff), seeds=list(base))
    settings = []
    for setting in data['sensitivity']:
        rows = setting['rows']
        fixed = setting['setting'] in ('delay_1', 'delay_8')
        if fixed:
            for field in ['full_return', 'joint_return', 'deployments']:
                if len({r[field] for r in rows}) != 1:
                    raise ValueError(f"Fixed-delay rows unexpectedly differ: {setting['setting']} {field}")
        delta = [r['full_return']-r['joint_return'] for r in rows]
        base_delta = [r['full_return']-base[r['seed']]['net_return'] for r in rows]
        row = dict(setting=setting['setting'], change=setting['change'], rows=rows,
            unique_replay_count=1 if fixed else len(rows),
            mean_return=mean([r['full_return'] for r in rows]),
            mean_actions=mean([r['deployments'] for r in rows]),
            mean_action_changes=mean([r['actions_changed_vs_base'] for r in rows]),
            paired_vs_joint=paired(delta, 1 if fixed else None),
            paired_setting_full_minus_default_full=paired(base_delta, 1 if fixed else None))
        # A changed fixed-delay world versus five random-delay worlds is not
        # five independent fixed-delay replications; retain mean differences
        # without using this secondary contrast as a confirmatory interval.
        row['paired_setting_full_minus_default_full']['interpretation'] = (
            'Descriptive sensitivity against the five original delay scenarios; '
            'world/candidate changes can accompany the intervention.')
        settings.append(row)
    return default, settings


def recovery_summary(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(definition=data['definition'], studies={})
    for study, block in data['studies'].items():
        out['studies'][study] = {}
        for method, scenarios in block['per_seed'].items():
            events = [e for s in scenarios for e in s['events']]
            sustained = [any(all(g>=0.03 for g in e['accuracy_gains'][i:i+3])
                            for i in range(max(0, len(e['accuracy_gains'])-2))) for e in events]
            positive = sum(e['local_net_gain']>0 for e in events)
            harmful = sum(e['local_net_gain']<0 for e in events)
            good = sum(s and e['local_net_gain']>0 for s, e in zip(sustained, events))
            n = len(events)
            row = dict(admissions=n, positive_local_gain=positive, harmful_local_gain=harmful,
                zero_local_gain=n-positive-harmful, sustained=sum(sustained), valuable_sustained=good,
                valuable_sustained_fraction=good/n if n else None,
                minimum_parameter_change=min(e['parameter_change'] for e in events) if events else None,
                truncated_horizons=sum(e.get('truncated_horizon', False) for e in events),
                max_return_error=max(s['absolute_return_error'] for s in scenarios),
                max_local_error=max(s['max_local_error'] for s in scenarios))
            cached = block['summary'][method]
            for field in ['admissions', 'positive_local_gain', 'harmful_local_gain', 'sustained', 'valuable_sustained']:
                if row[field] != cached[field]:
                    raise ValueError(f'Recovery audit mismatch: {study}/{method}/{field}')
            out['studies'][study][method] = row
    return out


def validate_reference(result: dict[str, Any], path: Path) -> dict[str, Any]:
    if not path.exists():
        return dict(passed=None, reason='Reference file absent.', path=str(path))
    ref, _ = load_json(path)
    differences: list[float] = []
    def close(actual: Any, expected: Any, name: str) -> None:
        a, b = float(actual), float(expected)
        differences.append(abs(a-b))
        if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f'Reference mismatch: {name}: {a} != {b}')
    for family in ['air_invariant', 'air_first_evaluation', 'gas']:
        for method, expected in ref[family].items():
            actual = result[family][method]
            for field in ['net', 'deployments', 'potential_accuracy', 'gross_served_reward',
                          'reward_change_full_minus_comparator', 'fee_saving_comparator_minus_full']:
                if field in expected:
                    close(actual[field], expected[field], f'{family}/{method}/{field}')
            for field in ['mean', 'positive', 'negative', 'ties']:
                close(actual['paired_full_minus_comparator'][field],
                      expected['paired_full_minus_comparator'][field], f'{family}/{method}/{field}')
            for a, b in zip(actual['paired_full_minus_comparator']['ci95_student_t9'],
                            expected['paired_full_minus_comparator']['ci95_student_t9']):
                close(a, b, f'{family}/{method}/t9')
    for a, b in zip(result['air_monthly'], ref['air_monthly']):
        close(a['n'], b['n'], f"monthly/{a['month']}/n")
        for method in b['metrics']:
            for field in ['accuracy', 'balanced_accuracy']:
                close(a['metrics'][method][field], b['metrics'][method][field], f'monthly/{method}/{field}')
    for field in ['eligible_decisions', 'total_decisions']:
        close(result['same_state_action_audit'][field], ref['same_state_eligibility'][field], field)
    for method, b in ref['same_state_shadow'].items():
        a = result['same_state_action_audit']['comparators'][method]
        for field in ['crossings', 'beneficial', 'harmful', 'local_sum']:
            close(a[field], b[field], f'shadow/{method}/{field}')
    for a, b in zip(result['sensitivity'], ref['sensitivity']):
        for field in ['mean_return', 'mean_actions', 'mean_action_changes']:
            close(a[field], b[field], f"sensitivity/{a['setting']}/{field}")
        close(a['paired_vs_joint']['mean'], b['paired_vs_joint']['mean'], f"sensitivity/{a['setting']}/mean")
        if a['unique_replay_count'] > 1:
            for x, y in zip(a['paired_vs_joint']['ci95_student_t4'], b['ci95_student_t4']):
                close(x, y, f"sensitivity/{a['setting']}/t4")
    return dict(passed=True, checked_metrics=len(differences),
                max_absolute_difference=max(differences, default=0.0), path=str(path),
                fixed_delay_note='Repeated deterministic rows deliberately receive no independent CI.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--air-invariant', type=Path, default=PACKAGE/'results/real_air_invariant_results.json.gz')
    parser.add_argument('--air-original', type=Path, default=PACKAGE/'results/real_air_results.json.gz')
    parser.add_argument('--gas', type=Path, default=PACKAGE/'results/real_action_results.json.gz')
    parser.add_argument('--recovery', type=Path, default=PACKAGE/'results/real_recovery_causal_audit.json')
    parser.add_argument('--output', type=Path, default=PACKAGE/'analysis_reproduced.json')
    parser.add_argument('--reference', type=Path, default=PACKAGE/'analysis.json')
    parser.add_argument('--skip-reference-check', action='store_true', help='For analyzing a new result family.')
    args = parser.parse_args()
    paths = dict(air_invariant=args.air_invariant, air_first_evaluation=args.air_original,
                 gas=args.gas, causal_recovery=args.recovery)
    loaded = {name: load_json(path) for name, path in paths.items()}
    air = loaded['air_invariant'][0]
    months, month_counts = analyze_months(air)
    default, sensitivities = sensitivity_summary(air)
    full_records = [v for p in air['per_seed'] for v in p['results']['decision_full']['audit']]
    scales = [v['scale_audit'] for v in full_records]
    output = dict(study_status=air['study_status'], analysis_version=1,
        input_files={name:dict(path=str(paths[name]), uncompressed_sha256=sha)
                     for name, (_, sha) in loaded.items()},
        air_invariant=study_summary(air),
        air_first_evaluation=study_summary(loaded['air_first_evaluation'][0]),
        gas=study_summary(loaded['gas'][0]),
        air_monthly=months, accuracy_month_counts_vs_joint=month_counts,
        same_state_action_audit=action_audit(air),
        sensitivity_default_first5=default, sensitivity=sensitivities,
        causal_recovery=recovery_summary(loaded['causal_recovery'][0]),
        scale_diagnostics=dict(decision_count=len(scales),
            fallbacks=sum(v['fallback'] for v in scales),
            trace_min=min(v['tangent_trace'] for v in scales),
            trace_max=max(v['tangent_trace'] for v in scales)),
        interpretive_limits=[
            'All Air invariant results are post-test exploratory on one already inspected chronological dataset.',
            'Seeds vary assigned feedback delay; they are not independent datasets or field deployments.',
            'Potential accuracy includes unserved retained records; gross served reward excludes dropped service positions.',
            'All five probe models are trained and paid equally; the controlled decision is candidate deployment admission.',
            'Same-state local forks suppress later admissions; sums do not decompose whole-policy causal return.',
            'Fixed-delay seeds repeat one identical deterministic replay and receive no independent confidence interval.',
            'No month-specific net-return decomposition exists in the saved traces; month outputs are accuracy only.',
            'Costs, delays and queue occupancy are experimental settings on measured source covariates and targets.',
            'The legacy eventual-recovery field is confounded by later deployments; causal recovery uses separate fixed-state forks.',
            'Normal and Student intervals are conditional summaries, with no method-selection or multiplicity correction.'])
    output['air_paired_seed_rows'] = []
    for p in air['per_seed']:
        a, b = p['results']['decision_full'], p['results']['context_joint']
        output['air_paired_seed_rows'].append(dict(seed=p['seed'], net_full=a['net_return'],
            net_joint=b['net_return'], net_gain=a['net_return']-b['net_return'],
            deploy_full=a['deployments'], deploy_joint=b['deployments'],
            actual_action_changes=int(np.sum(np.asarray(a['actions']) != np.asarray(b['actions'])))))
    output['reference_check'] = (dict(passed=None, reason='Disabled by explicit CLI option.')
        if args.skip_reference_check else validate_reference(output, args.reference))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    core = output['air_invariant']['context_joint']
    print(json.dumps(dict(output=str(args.output), reference_check=output['reference_check'],
        full_minus_joint=core['paired_full_minus_comparator']['mean'],
        t9_interval=core['paired_full_minus_comparator']['ci95_student_t9'],
        served_gross_increment=core['reward_change_full_minus_comparator'],
        fee_saving=core['fee_saving_comparator_minus_full'],
        actual_action_disagreements=core['actual_action_differences_total'],
        same_state_eligible=output['same_state_action_audit']['eligible_decisions']), indent=2))


if __name__ == '__main__':
    main()
