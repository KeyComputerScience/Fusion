"""Read immutable primary, supplementary and external logs with a metadata-only shim."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
import coverage_analysis as audit

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('occupancy357', 'occupancy864', 'mhealth319', 'har240')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def compare_existing(a, b):
    # The base engine used directly by external runners does not append the
    # wearable adapter's three outcome-only diagnostic fields.
    ignored = {'pessimistic_net_target', 'opportunity_slack', 'candidate_extra_correct'}
    if isinstance(a, dict):
        return max([compare_existing(v, b[k]) for k, v in a.items() if k not in ignored] + [0.])
    if isinstance(a, list):
        assert len(a) == len(b)
        return max([compare_existing(x, y) for x, y in zip(a, b)] + [0.])
    if isinstance(a, bool) or a is None or isinstance(a, str):
        assert a == b
        return 0.
    return abs(float(a) - float(b))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, default=ROOT / 'bayes_closed_loop_repro')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); package = args.package.resolve(); out = args.output.resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit('Analysis output must be outside the delivered bundle')
    out.mkdir(parents=True, exist_ok=True)
    primary_paths = {n: package / ('new_bayes_fusion' if n.startswith('occupancy') else 'new_bayes_extension') / (n + '_results.json') for n in TASKS}
    supplement_path = package.parent / 'performance_gain_evidence/results.json'
    inputs = [*primary_paths.values(), supplement_path, Path(audit.__file__)]
    before = {str(path): sha(path) for path in inputs}
    primary = {n: read(path) for n, path in primary_paths.items()}
    supplement = read(supplement_path)
    report = dict(kind='Portable read-only coverage/action/accounting analysis',
        cohorts={}, external={}, input_sha256={}, original_arm_seed_log_audits=0,
        interpretation='Original cohorts remain separate; ten-seed combination is descriptive on the same traces. External point-arm q coverage is diagnostic when q is unused by its gate.')
    for name in TASKS:
        assert [t['seed'] for t in primary[name]['trials']] == list(range(72001, 72006))
        assert [t['seed'] for t in supplement[name]['trials']] == list(range(72006, 72011))
        assert primary[name]['data_hashes'] == supplement[name]['data_hashes']
        assert primary[name]['split'] == supplement[name]['split']
    for label, studies in (('primary', primary), ('supplement', supplement),
        ('combined_ten_seed_descriptive', {n: audit.combine_fixed_trace(primary[n], supplement[n]) for n in TASKS})):
        report['cohorts'][label] = {n: audit.analyze_study(s) for n, s in studies.items()}
    report['original_arm_seed_log_audits'] = sum(sum(a['seed_count'] for a in s['arms'].values())
        for label in ('primary', 'supplement') for s in report['cohorts'][label].values())
    for method in ('pdf', 'qmf'):
        folder = package.parent / 'external' / method
        if not (folder / 'results.json').is_file():
            continue
        new_inputs = [folder / name for name in ('protocol.json', 'prefix_freeze.json', 'execution_audit.json', 'results.json')]
        before.update({str(path): sha(path) for path in new_inputs})
        inputs.extend(new_inputs)
        protocol = read(folder / 'protocol.json'); freeze = read(folder / 'prefix_freeze.json')
        execution = read(folder / 'execution_audit.json'); studies = read(folder / 'results.json')
        assert sha(folder / 'protocol.json') == freeze['protocol_sha256'] == execution['protocol_sha256']
        script = folder / ('run_external_pdf.py' if method == 'pdf' else 'run_qmf_energy.py')
        runner_key = 'script_sha256' if method == 'pdf' else 'runner_sha256'
        selection_key = 'all_prefix_selection_sha256' if method == 'pdf' else 'selection_sha256'
        assert sha(script) == freeze[runner_key] == execution[runner_key]
        if method == 'qmf':
            helper = folder.parent / 'run_external_pdf.py'
            before[str(helper)] = sha(helper); inputs.append(helper)
            assert sha(helper) == protocol['pdf_helper_sha256'] == freeze['helper_sha256'] == execution['helper_sha256']
        assert execution['all_passed'] and execution['original_package_unchanged']
        assert sha(package / 'independent_bayes_fusion.py') == protocol['frozen_engine_sha256']
        assert sha(package / 'independent_bayes_extension.py') == protocol['frozen_adapter_sha256']
        inputs.append(script); before[str(script)] = sha(script)
        augmented = copy.deepcopy(studies); max_old_difference = max_penalty_error = max_source_error = 0.
        for name, study in studies.items():
            assert study['data_hashes'] == primary[name]['data_hashes']
            assert study['split'] == primary[name]['split']
            assert sha(primary_paths[name]) == protocol['source_studies'][name]['result_sha256']
            selection = folder / (name + '_prefix_selection.json'); inputs.append(selection); before[str(selection)] = sha(selection)
            assert sha(selection) == freeze[selection_key][name] == study['prefix_selection_sha256']
            assert read(selection)['selected'] == study['selected']
            original_trials = {t['seed']: t for t in primary[name]['trials']}
            for trial, normalized in zip(study['trials'], augmented[name]['trials']):
                old = original_trials[trial['seed']]; reference = trial['results']['frozen']
                baseline = {k: reference[k] for k in ('gross', 'fees', 'drops', 'net')}
                assert compare_existing(old['baseline'], baseline) < 1e-8
                windows = int(old['windows']); assert windows == math.ceil(study['split']['test_rows'] / 32)
                for physical in trial['execution_audit'].values():
                    assert physical['all_passed'] and int(physical['explicit_service']['windows']) == windows
                for arm in ('bayes_both', 'joint', 'frequentist_gate', 'frozen'):
                    max_old_difference = max(max_old_difference, compare_existing(old['results'][arm], trial['results'][arm]))
                for arm, result in trial['results'].items():
                    if arm in ('bayes_both', 'joint', 'frequentist_gate', 'frozen'):
                        continue
                    for row in result['rows']:
                        expected = row['q_issued'] * row['posterior_or_block_sd'] if 'calibrated' in arm else 0.
                        max_penalty_error = max(max_penalty_error, abs(row['penalty'] - expected))
                        values = np.asarray(row['pdf_weights' if method == 'pdf' else 'qmf_confidences'], float)
                        assert values.ndim == 2 and values.shape[1] == len(row['active_sources'])
                        assert np.isfinite(values).all() and values.min() >= 0.
                        if method == 'pdf':
                            max_source_error = max(max_source_error, float(np.max(abs(values.sum(1) - 1.))))
                        if 'point' in arm:
                            assert result['q_initial'] == 0. and row['penalty'] == 0.
                normalized['baseline'] = baseline; normalized['windows'] = windows
        assert max_old_difference < 1e-8 and max_penalty_error < 1e-8 and max_source_error < 1e-8
        external_arms = tuple(a for a in next(iter(studies.values()))['trials'][0]['results']
                              if a not in ('bayes_both', 'joint', 'frequentist_gate', 'frozen'))
        comparators = ('reference', *external_arms, 'frequentist_gate', 'joint')
        tasks = {n: audit.analyze_study(s, comparators=comparators) for n, s in augmented.items()}
        report['external'][method] = dict(tasks=tasks,
            schema_shim_added_fields=['trial.baseline from same-trial frozen arm', 'trial.windows from original trial and explicit-service checks'],
            max_original_field_difference=max_old_difference, max_external_gate_penalty_error=max_penalty_error,
            max_physical_source_simplex_error=max_source_error if method == 'pdf' else None,
            physical_source_weight_contract='PDF simplex without internal cap' if method == 'pdf' else 'QMF finite nonnegative energy coefficients, not simplex weights',
            arm_seed_log_audits=sum(sum(a['seed_count'] for a in s['arms'].values()) for s in tasks.values()),
            source_scope='Declared probability/energy interface adaptation; not full jointly trained published benchmark reproduction')
    report['input_sha256'] = before
    # No input writes occur; this final check detects concurrent input alteration.
    report['input_hashes_unchanged'] = all(sha(path) == value for path, value in report['input_sha256'].items())
    assert report['input_hashes_unchanged']
    report['passed'] = True
    report['total_arm_trial_audits'] = report['original_arm_seed_log_audits']
    (out / 'coverage_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    audit.write_notes(report, out / 'coverage_notes.md')
    lines = ['# Retained external comparisons', '',
        'Each external adaptation remains a separate later diagnostic on the four original physical traces. Five delay schedules are not independent sites. Point-arm q-tracker coverage is diagnostic; calibrated-arm penalty uses issued q times issued scale. Complete action changes, fees and denominators are preserved.', '',
        '| Adaptation | Task | Comparator | Mean full-minus-control net | Five gains | Conditional t4 interval |',
        '|---|---|---|---:|---|---|']
    for method, external in report['external'].items():
        for name, task in external['tasks'].items():
            for arm in (method + '_point', method + '_calibrated'):
                comp = task['full_comparisons'][arm]['paired_net_gain']
                lines.append(f"| {method} | {name} | {arm} | {comp['mean']:.4f} | {comp['by_seed']} | {comp['conditional_delay_t95_descriptive_interval']} |")
    lines += ['', '| Adaptation | Task | Arm | Admissions / harmful / beneficial | Issued q coverage | Informative coverage | Admitted coverage |',
              '|---|---|---|---|---|---|---|']
    for method, external in report['external'].items():
        for name, task in external['tasks'].items():
            for arm in (method + '_point', method + '_calibrated', 'bayes_both'):
                state = task['arms'][arm]; counts = state['actions']; coverage = state['coverage']
                cells = ' | '.join(audit.fmt_ratio(coverage[key]) for key in ('issued', 'informative_current_disagreement', 'admitted'))
                lines.append(f"| {method} | {name} | {arm} | {counts['admissions']} / {counts['harmful']} / {counts['beneficial']} | {cells} |")
    (out / 'external_comparisons.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(passed=True, original_arm_seed_log_audits=report['original_arm_seed_log_audits'],
        external_arm_seed_log_audits={n: s['arm_seed_log_audits'] for n, s in report['external'].items()},
        input_hashes_unchanged=report['input_hashes_unchanged']), indent=2))


if __name__ == '__main__':
    main()
