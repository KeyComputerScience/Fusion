"""Offline unified entry point. Original algorithms, protocols and results are immutable."""
from __future__ import annotations
import argparse, datetime, hashlib, json, math, os, platform, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
BASE = ROOT / 'bayes_closed_loop_repro'
SUPP = ROOT / 'performance_gain_evidence'
TASKS = ('occupancy357', 'occupancy864', 'mhealth319', 'har240')
COMMANDS = ('verify', 'audit-primary', 'audit-supp', 'analyze', 'rerun-primary',
            'rerun-supp', 'sensitivity', 'crossings', 'benchmark', 'external',
            'external-pdf', 'external-qmf', 'analyze-external')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execute(argv, out, label):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, *map(str, argv)]
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('RUN', label, flush=True)
    with (out / (label + '.log')).open('w') as log:
        proc = subprocess.Popen(command, cwd=out, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            print(line, end='', flush=True)
        status = proc.wait()
    journal = out / 'commands.jsonl'
    with journal.open('a') as f:
        f.write(json.dumps(dict(command=command, cwd=str(out), start_utc=start,
                               exit_code=status, label=label)) + '\n')
    if status:
        raise SystemExit(status)


def cached(script, *args):
    return [BASE / 'cached_runner.py', script, *args]


def copy_protocol(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'protocol.json', destination / 'protocol.json')


def equal_values(a, b, where, errors, tolerance=1e-8):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            errors.append(where + ': dictionary keys differ')
            return
        for k in a:
            equal_values(a[k], b[k], where + '/' + str(k), errors, tolerance)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            errors.append(where + ': lengths differ')
            return
        for i, (x, y) in enumerate(zip(a, b)):
            equal_values(x, y, where + '/' + str(i), errors, tolerance)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b: errors.append(where + ': action/bool differs')
        elif not (math.isfinite(float(a)) and math.isfinite(float(b))):
            errors.append(where + ': nonfinite numeric result')
        elif abs(float(a) - float(b)) > tolerance:
            errors.append(where + ': numeric value differs')
    elif a != b:
        errors.append(where + ': value differs')


def compare_runs(out, supplied, generated, filename):
    errors = []; comparisons = []
    for name in TASKS:
        original = json.loads(supplied(name).read_text())
        replay = json.loads(generated(name).read_text())
        for field in ('selected', 'q_initial'):
            if field in original or field in replay:
                equal_values(original.get(field), replay.get(field), name + '/' + field, errors)
        # Freeze timestamps, serialization hashes and elapsed seconds are not
        # model predictions. Compare all execution fields and primary grids.
        if 'calibration' in original:
            equal_values(original['calibration'], replay['calibration'], name + '/calibration', errors)
        equal_values([t['seed'] for t in original['trials']], [t['seed'] for t in replay['trials']], name + '/seeds', errors)
        for first, second in zip(original['trials'], replay['trials']):
            equal_values(first['results'], second['results'], name + '/' + str(first['seed']), errors)
        comparisons.append(dict(dataset=name, trials=len(original['trials']),
                                arms=len(original['trials'][0]['results'])))
    result = dict(passed=not errors, tolerance=1e-8, comparisons=comparisons,
                  errors=errors[:100], differences=len(errors),
                  ignored_fields='timestamps, timing, output-location and serialization hash metadata only')
    (out / filename).write_text(json.dumps(result, indent=2) + '\n')
    if errors:
        raise AssertionError('Replay differs; see ' + filename)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=COMMANDS)
    p.add_argument('--output', type=Path, help='New output folder outside the delivered bundle.')
    args = p.parse_args()
    if args.command == 'verify':
        from manifest_tool import verify
        report = verify(); print(json.dumps(report, indent=2))
        if not report['passed']: raise SystemExit(1)
        return
    out = (args.output or ROOT.parent / 'Fusion_Reproduction_Runs' / args.command).resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit('--output must be outside this delivered bundle.')
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Choose a new empty --output directory; results are never overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    (out / 'run_environment.json').write_text(json.dumps(dict(
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        command=args.command, bundle=str(ROOT), engine_sha256=sha(BASE / 'independent_bayes_fusion.py'),
        adapter_sha256=sha(BASE / 'independent_bayes_extension.py'),
        note='Runtime verification or replay timestamp; original declaration times are preserved separately.'), indent=2) + '\n')
    if args.command == 'audit-primary':
        scratch = out / 'working_base'
        shutil.copytree(BASE, scratch, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        execute([scratch / 'cached_runner.py', 'audit_all_four_bayes.py'], out, 'primary_reconstruction')
        execute([scratch / 'cached_runner.py', 'audit_canary_execution.py', '--results',
            scratch / 'new_bayes_fusion', '--datasets', 'occupancy357', 'occupancy864',
            '--output', out / 'primary_occupancy_verification.json'], out, 'occupancy_reconstruction')
        execute([scratch / 'cached_runner.py', 'audit_canary_execution.py', '--results',
            scratch / 'new_bayes_extension', '--datasets', 'mhealth319', 'har240',
            '--output', out / 'primary_wearable_verification.json'], out, 'wearable_reconstruction')
        execute([scratch / 'audit_production_bayes_guard.py'], out, 'input_guard_verification')
        shutil.copy2(scratch / 'new_bayes_extension/all_four_integrity_audit.json', out / 'primary_integrity_verification.json')
        shutil.copy2(scratch / 'production_guard_audit.json', out / 'production_guard_verification.json')
    elif args.command == 'audit-supp':
        execute([ROOT / 'verify_supplementary_execution.py', '--output', out / 'supplementary_reconstruction.json'], out, 'supplementary_reconstruction')
    elif args.command == 'analyze':
        execute([BASE / 'analyze_posterior_trials.py', '--room-results', BASE / 'new_bayes_fusion/results.json',
            '--wearable-results', BASE / 'new_bayes_extension/results.json', '--out', out / 'primary_analysis'], out, 'primary_endpoints')
        execute([SUPP / 'primary_gain_analysis.py', '--package', BASE, '--output', out / 'gain_analysis'], out, 'primary_gain_decomposition')
    elif args.command == 'rerun-primary':
        room = out / 'primary_room'; wearable = out / 'primary_wearable'
        copy_protocol(BASE / 'new_bayes_fusion', room); copy_protocol(BASE / 'new_bayes_extension', wearable)
        execute(cached('independent_bayes_fusion.py', '--data', BASE / 'independent_data', '--output', room), out, 'primary_occupancy_full_grid')
        for task in ('mhealth319', 'har240'):
            execute(cached('independent_bayes_extension.py', '--data', BASE / 'independent_data', '--output', wearable, '--task', task), out, task + '_full_grid')
        compare_runs(out, lambda n: BASE / ('new_bayes_fusion' if n.startswith('occupancy') else 'new_bayes_extension') / (n + '_results.json'),
                     lambda n: (room if n.startswith('occupancy') else wearable) / (n + '_results.json'), 'primary_replay_comparison.json')
    elif args.command == 'rerun-supp':
        execute([SUPP / 'portable_launcher.py', '--package', BASE, '--output', out / 'supplementary_replay'], out, 'supplementary_fixed_controllers')
        original = json.loads((SUPP / 'results.json').read_text())
        # The original launcher supplies complete task files in the output.
        source = out / 'supplied_task_json'; source.mkdir()
        for name, task in original.items():
            (source / (name + '_results.json')).write_text(json.dumps(task))
        compare_runs(out, lambda n: source / (n + '_results.json'),
                     lambda n: out / 'supplementary_replay' / (n + '_results.json'), 'supplementary_replay_comparison.json')
    elif args.command == 'sensitivity':
        execute(cached('diagnostic_bayes_sensitivity.py', '--output', out / 'sensitivity'), out, 'one_at_a_time_sensitivity')
    elif args.command == 'crossings':
        # Retain source bytes; inject explicit paths before invoking original main.
        launcher = out / 'crossing_launcher.py'
        launcher.write_text('import sys,importlib.util\nfrom pathlib import Path\n'
            + 'sys.dont_write_bytecode=True\npackage=Path(' + repr(str(BASE)) + ')\n'
            + 'sys.path.insert(0,str(package))\nfrom cached_runner import install_cached_loaders\ninstall_cached_loaders()\n'
            + 'spec=importlib.util.spec_from_file_location("retained_crossing",' + repr(str(ROOT / 'supplementary_auxiliary/audit_crossings.py')) + ')\n'
            + 'module=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n'
            + 'module.PACKAGE=package\nmodule.ROOT=Path(' + repr(str(out / 'crossing_inputs')) + ')\nmodule.main()\n')
        staging = out / 'crossing_inputs'; staging.mkdir()
        for name in ('results.json', 'run_extension.py'):
            shutil.copy2(SUPP / name, staging / name)
        execute([launcher], out, 'retained_action_crossings')
    elif args.command == 'benchmark':
        execute([ROOT / 'external/benchmark_live_interface.py', '--package', BASE,
                 '--output', out / 'timing'], out, 'live_interface_timing')
    elif args.command in ('external', 'external-pdf', 'external-qmf'):
        methods = ('pdf', 'qmf') if args.command == 'external' else (args.command.split('-')[1],)
        for method in methods:
            if not (ROOT / 'external' / method / 'portable_launcher.py').is_file():
                raise SystemExit('Requested external method is not bundled: ' + method)
        for method in methods:
            source = ROOT / 'external' / method
            target = out / method
            execute([source / 'portable_launcher.py', '--package', BASE, '--output', target], out, method + '_full_calibration_and_test')
            compare_runs(out, lambda n: source / (n + '_results.json'),
                         lambda n: target / (n + '_results.json'), method + '_all_execution_fields_comparison.json')
    elif args.command == 'analyze-external':
        target = ROOT / 'analysis' / 'portable_analysis.py'
        if not target.is_file():
            raise SystemExit('Portable external analysis is not bundled yet.')
        execute([target, '--package', BASE, '--output', out / 'analysis'], out, 'all_retained_endpoint_analysis')
    print('COMPLETE', args.command, str(out), flush=True)


if __name__ == '__main__':
    main()
